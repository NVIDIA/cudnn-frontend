# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for the DSA indexer top-K CuTe DSL kernel."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from cutlass.jax import jax_to_cutlass_dtype

from ..._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    TupleDict,
    call_cutedsl,
    require_dtype,
)

_INT32_MAX = (1 << 31) - 1


def _launch(
    stream,
    input_values,
    seq_lens,
    output_indices,
    output_values,
    extra_buffer,
    *,
    cutlass_dtype: Any,
    dtype_bits: int,
    num_cols: int,
    top_k: int,
    next_n: int,
    num_copy_bits: int,
    large_occupancy: bool,
):
    # Load the configuration-specific kernel only when tracing the operation.
    from .indexer_top_k_decode_varlen import (
        IndexerTopKKernelVarlenDecode,
        _bucket_num_cols,
    )

    kernel = IndexerTopKKernelVarlenDecode(
        cutlass_dtype,
        _bucket_num_cols(num_cols),
        top_k,
        next_n,
        num_copy_bits=num_copy_bits,
        return_val=True,
        large_occupancy=large_occupancy,
    )
    require_supported_top_k_output(
        top_k=top_k,
        num_threads_per_cta=kernel.num_threads_per_cta,
        num_copy_bits=num_copy_bits,
        dtype_bits=dtype_bits,
    )

    kernel(
        input_values,
        None,  # Only used by the unsupported multi-CTA merge path.
        extra_buffer,
        None,  # Only used by the unsupported persistent scheduler.
        seq_lens,
        output_indices,
        output_values,
        stream,
        enable_persistent_dynamic_scheduling=False,
        min_blocks_per_mp=1,
    )


def _launch_local_to_global_fixed(
    stream,
    local_indices,
    global_indices,
    *,
    seqlen_k: int,
):
    from cutlass import Int32

    from .local_to_global_dsl import LocalToGlobalTopK

    kernel = LocalToGlobalTopK(is_varlen=False)
    kernel(
        local_indices,
        global_indices,
        Int32(seqlen_k),
        None,
        None,
        stream,
    )


def _launch_local_to_global_varlen(
    stream,
    local_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    global_indices,
    *,
    seqlen_k: int,
):
    from cutlass import Int32

    from .local_to_global_dsl import LocalToGlobalTopK

    kernel = LocalToGlobalTopK(is_varlen=True)
    kernel(
        local_indices,
        global_indices,
        Int32(seqlen_k),
        cu_seqlens_q,
        cu_seqlens_k,
        stream,
    )


def _launch_compactify(
    stream,
    indices,
    compact_indices,
    topk_length,
    *,
    rows: int,
    cols: int,
):
    from cutlass import Int32

    from .compactify import CompactifyKernel

    kernel = CompactifyKernel(cols=cols)
    kernel(indices, compact_indices, topk_length, Int32(rows), stream)


def require_supported_top_k_output(
    *,
    top_k: int,
    num_threads_per_cta: int,
    num_copy_bits: int,
    dtype_bits: int,
) -> None:
    """Validate the output vector width selected by the native kernel."""

    vector_size = min(
        top_k,
        (top_k + num_threads_per_cta - 1) // num_threads_per_cta,
        num_copy_bits // dtype_bits,
        2,
    )
    if top_k % vector_size:
        raise ValueError(f"top_k ({top_k}) must be divisible by the selected output vector " f"width ({vector_size}); adjust top_k or num_copy_bits")


def _indexer_top_k_impl(
    input_values: Any,
    seq_lens: Any,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
    _validate_only: bool = False,
) -> TupleDict:
    """Select the largest values from each row with variable valid lengths.

    This is the JAX counterpart of the Torch ``indexer_top_k_wrapper`` and is
    intended for use inside :func:`jax.jit`. ``input_values`` has shape
    ``(num_rows, num_cols)`` and ``seq_lens`` has shape ``(batch_size,)``, with
    ``num_rows == batch_size * next_n``. The effective valid length for row
    ``r`` is ``seq_lens[r // next_n] - next_n + (r % next_n) + 1``.
    Runtime lengths are trusted kernel inputs. Every ``seq_lens[b]`` must
    satisfy ``top_k + next_n - 1 <= seq_lens[b] <= num_cols`` so every
    staggered row has at least ``top_k`` valid entries and no row reads beyond
    ``input_values``. These values are not copied to the host for validation
    during tracing.

    Shapes and configuration arguments must be concrete while tracing. The
    API supports the kernel's single-launch path and always returns both
    indices and values.
    """

    if not return_val:
        raise NotImplementedError("The JAX indexer_top_k_wrapper requires return_val=True")

    if not hasattr(input_values, "shape") or not hasattr(input_values, "dtype"):
        raise TypeError("input_values must be a JAX array with shape and dtype metadata")
    if not hasattr(seq_lens, "shape") or not hasattr(seq_lens, "dtype"):
        raise TypeError("seq_lens must be a JAX array with shape and dtype metadata")
    if len(input_values.shape) != 2:
        raise ValueError(f"input_values must have rank 2, got shape {input_values.shape}")
    if len(seq_lens.shape) != 1:
        raise ValueError(f"seq_lens must have rank 1, got shape {seq_lens.shape}")

    num_rows, num_cols = input_values.shape
    (batch_size,) = seq_lens.shape

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"input_values dimensions must be positive, got {(num_rows, num_cols)}")
    if batch_size <= 0:
        raise ValueError(f"seq_lens must not be empty, got shape {seq_lens.shape}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if num_rows != batch_size * next_n:
        raise ValueError(f"num_rows ({num_rows}) must equal seq_lens.size * next_n " f"({batch_size} * {next_n} = {batch_size * next_n})")
    if top_k <= 0 or top_k > min(2048, num_cols):
        raise ValueError(f"top_k must be in (0, min(2048, num_cols={num_cols})], got {top_k}")
    if num_copy_bits <= 0 or num_copy_bits % 8:
        raise ValueError(f"num_copy_bits must be a positive whole-byte width, got {num_copy_bits}")
    copy_bytes = num_copy_bits // 8
    if copy_bytes & (copy_bytes - 1):
        raise ValueError("num_copy_bits must describe a power-of-two byte alignment, " f"got {num_copy_bits} bits ({copy_bytes} bytes)")

    input_dtype = require_dtype(
        "input_values.dtype",
        input_values,
        (jnp.float16, jnp.bfloat16, jnp.float32),
    )
    require_dtype("seq_lens.dtype", seq_lens, (jnp.int32,))

    dtype_bits = input_dtype.itemsize * 8
    if num_copy_bits % dtype_bits != 0:
        raise ValueError(f"num_copy_bits ({num_copy_bits}) must be divisible by the " f"input dtype width ({dtype_bits})")

    workspace_buffers = 2 if input_dtype == jnp.dtype(jnp.float32) else 1
    workspace_elements = num_rows * workspace_buffers * num_cols
    if workspace_elements > _INT32_MAX:
        raise NotImplementedError(
            "The JAX indexer_top_k_wrapper does not support the Torch "
            "row-chunking fallback used when the int32 workspace contains "
            f"more than {_INT32_MAX} elements (requested {workspace_elements})"
        )
    if _validate_only:
        return None

    output_shape = (num_rows, top_k)
    output_indices, output_values = call_cutedsl(
        _launch,
        (input_values, seq_lens),
        outputs=(
            BufferSpec("indices", output_shape, jnp.int32),
            BufferSpec("values", output_shape, input_dtype),
        ),
        workspaces=(
            BufferSpec(
                "extra_buffer",
                (num_rows, workspace_buffers, num_cols),
                jnp.int32,
            ),
        ),
        static_args={
            "cutlass_dtype": jax_to_cutlass_dtype(input_dtype),
            "dtype_bits": int(dtype_bits),
            "num_cols": int(num_cols),
            "top_k": int(top_k),
            "next_n": int(next_n),
            "num_copy_bits": int(num_copy_bits),
            "large_occupancy": bool(num_rows > 148),
        },
        use_static_tensors=True,
    )
    return TupleDict(indices=output_indices, values=output_values)


class IndexerTopK(ApiBaseJax):
    """Sample-signature-bound JAX callable for the DSA indexer top-K kernel."""

    def __init__(
        self,
        sample_input_values: Any,
        sample_seq_lens: Any,
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
        num_copy_bits: int = 256,
    ) -> None:
        super().__init__()
        self.input_desc = self.make_tensor_desc(sample_input_values, name="sample_input_values")
        self.seq_lens_desc = self.make_tensor_desc(sample_seq_lens, name="sample_seq_lens")
        self.top_k = top_k
        self.next_n = next_n
        self.return_val = return_val
        self.num_copy_bits = num_copy_bits

    def _check_support(self) -> bool:
        _indexer_top_k_impl(
            self.input_desc,
            self.seq_lens_desc,
            self.top_k,
            self.next_n,
            self.return_val,
            self.num_copy_bits,
            _validate_only=True,
        )
        return True

    def __call__(self, input_values: Any, seq_lens: Any) -> TupleDict:
        return super().__call__(input_values, seq_lens)

    def _call_impl(self, input_values: Any, seq_lens: Any) -> TupleDict:
        self.check_tensor_signature(input_values, self.input_desc, name="input_values")
        self.check_tensor_signature(seq_lens, self.seq_lens_desc, name="seq_lens")
        return _indexer_top_k_impl(
            input_values,
            seq_lens,
            self.top_k,
            self.next_n,
            self.return_val,
            self.num_copy_bits,
        )


def indexer_top_k_wrapper(
    input_values: Any,
    seq_lens: Any,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
) -> TupleDict:
    """Select the largest values from each row with variable valid lengths."""

    return IndexerTopK(
        input_values,
        seq_lens,
        top_k,
        next_n=next_n,
        return_val=return_val,
        num_copy_bits=num_copy_bits,
    )(input_values, seq_lens)


def local_to_global_wrapper(
    local_indices: Any,
    seqlen_k: int,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_k: Any | None = None,
) -> TupleDict:
    """Convert local top-K indices to the global flattened KV index space.

    Fixed-shape input has shape ``(B, S_q, topk)`` and uses
    ``batch_index * seqlen_k`` as its offset. Packed input has shape
    ``(total_q, topk)`` and requires matching ``cu_seqlens_q`` and
    ``cu_seqlens_k`` int32 arrays. Runtime cumulative lengths are consumed by
    the kernel and are not copied to the host while tracing.
    """

    if not hasattr(local_indices, "shape") or not hasattr(local_indices, "dtype"):
        raise TypeError("local_indices must have shape and dtype metadata")
    if seqlen_k <= 0:
        raise ValueError(f"seqlen_k must be positive, got {seqlen_k}")

    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("Packed local-to-global conversion requires both cu_seqlens_q " "and cu_seqlens_k")
        if len(local_indices.shape) != 2:
            raise ValueError("Packed local_indices must have rank 2, got shape " f"{local_indices.shape}")
        for name, value in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
        ):
            if not hasattr(value, "shape") or not hasattr(value, "dtype"):
                raise TypeError(f"{name} must have shape and dtype metadata")
            if len(value.shape) != 1:
                raise ValueError(f"{name} must have rank 1, got shape {value.shape}")
            require_dtype(f"{name}.dtype", value, (jnp.int32,))
        if tuple(cu_seqlens_q.shape) != tuple(cu_seqlens_k.shape):
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape, got " f"{cu_seqlens_q.shape} and {cu_seqlens_k.shape}")
        inputs = (local_indices, cu_seqlens_q, cu_seqlens_k)
    else:
        if len(local_indices.shape) != 3:
            raise ValueError("Fixed-shape local_indices must have rank 3, got shape " f"{local_indices.shape}")
        inputs = (local_indices,)

    require_dtype(
        "local_indices.dtype",
        local_indices,
        (jnp.int32, jnp.int64),
    )
    launcher = _launch_local_to_global_varlen if is_varlen else _launch_local_to_global_fixed
    (global_indices,) = call_cutedsl(
        launcher,
        inputs,
        outputs=(
            BufferSpec(
                "indices",
                tuple(local_indices.shape),
                jnp.int32,
            ),
        ),
        static_args={"seqlen_k": int(seqlen_k)},
        use_static_tensors=True,
    )
    return TupleDict(indices=global_indices)


def compactify_wrapper(indices: Any) -> TupleDict:
    """Pack nonnegative indices to the front of each row.

    ``indices`` may have shape ``(rows, topk)`` or ``(B, S_q, topk)``. As in
    the Torch API, a rank-3 input is flattened batch-major and the returned
    index array has shape ``(B * S_q, topk)``. ``topk_length`` contains the
    number of nonnegative entries in every returned row.
    """

    if not hasattr(indices, "shape") or not hasattr(indices, "dtype"):
        raise TypeError("indices must have shape and dtype metadata")
    if len(indices.shape) not in (2, 3):
        raise ValueError(f"indices must have rank 2 or 3, got shape {indices.shape}")
    require_dtype("indices.dtype", indices, (jnp.int32,))

    cols = indices.shape[-1]
    rows = 1
    for extent in indices.shape[:-1]:
        rows *= extent
    if rows <= 0 or cols <= 0:
        raise ValueError(f"indices dimensions must be positive, got {indices.shape}")

    flat_indices = jnp.reshape(indices, (rows, cols))
    compact_indices, topk_length = call_cutedsl(
        _launch_compactify,
        (flat_indices,),
        outputs=(
            BufferSpec("indices", (rows, cols), jnp.int32),
            BufferSpec("topk_length", (rows,), jnp.int32),
        ),
        static_args={"rows": int(rows), "cols": int(cols)},
        use_static_tensors=True,
    )
    return TupleDict(
        indices=compact_indices,
        topk_length=topk_length,
    )


__all__ = [
    "IndexerTopK",
    "compactify_wrapper",
    "indexer_top_k_wrapper",
    "local_to_global_wrapper",
]
