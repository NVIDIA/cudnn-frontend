# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for the DSA indexer top-K CuTe DSL kernel."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

from .cutedsl import BufferSpec, call_cutedsl
from .utils import require_concrete_dims, require_static_bool, require_static_int

_INT32_MAX = (1 << 31) - 1


class IndexerTopKResult(NamedTuple):
    """Functional JAX outputs for indexer top-K."""

    indices: Any
    values: Any


@lru_cache(maxsize=None)
def _make_launcher(
    cutlass_dtype: Any,
    dtype_bits: int,
    num_cols: int,
    top_k: int,
    next_n: int,
    num_copy_bits: int,
    large_occupancy: bool,
):
    # Keep the optional kernel import off the cudnn.jax import path.
    from ..deepseek_sparse_attention.indexer_top_k.indexer_top_k_decode_varlen import (
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

    def launch(stream, input_values, seq_lens, output_indices, output_values, extra_buffer):
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

    return launch


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
        raise ValueError(
            f"top_k ({top_k}) must be divisible by the selected output vector "
            f"width ({vector_size}); adjust top_k or num_copy_bits"
        )


def indexer_top_k_wrapper(
    input_values: Any,
    seq_lens: Any,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
) -> IndexerTopKResult:
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
    proof of concept supports the kernel's single-launch path and always
    returns both indices and values.
    """

    try:
        import jax.numpy as jnp
        from cutlass.jax import jax_to_cutlass_dtype
    except ImportError as exc:
        raise ImportError(
            "indexer_top_k_wrapper requires JAX and the CuTe DSL JAX "
            "integration; install the 'jax' optional dependencies"
        ) from exc

    return_val = require_static_bool(return_val, name="return_val")
    if not return_val:
        raise NotImplementedError(
            "The JAX indexer_top_k_wrapper proof of concept currently requires " "return_val=True"
        )

    top_k = require_static_int(top_k, name="top_k")
    next_n = require_static_int(next_n, name="next_n")
    num_copy_bits = require_static_int(num_copy_bits, name="num_copy_bits")

    if not hasattr(input_values, "shape") or not hasattr(input_values, "dtype"):
        raise TypeError("input_values must be a JAX array with shape and dtype metadata")
    if not hasattr(seq_lens, "shape") or not hasattr(seq_lens, "dtype"):
        raise TypeError("seq_lens must be a JAX array with shape and dtype metadata")
    if len(input_values.shape) != 2:
        raise ValueError(f"input_values must have rank 2, got shape {input_values.shape}")
    if len(seq_lens.shape) != 1:
        raise ValueError(f"seq_lens must have rank 1, got shape {seq_lens.shape}")

    num_rows, num_cols = require_concrete_dims(
        input_values.shape,
        "num_rows",
        "num_cols",
    )
    (batch_size,) = require_concrete_dims(seq_lens.shape, "batch_size")

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"input_values dimensions must be positive, got {(num_rows, num_cols)}")
    if batch_size <= 0:
        raise ValueError(f"seq_lens must not be empty, got shape {seq_lens.shape}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if num_rows != batch_size * next_n:
        raise ValueError(
            f"num_rows ({num_rows}) must equal seq_lens.size * next_n "
            f"({batch_size} * {next_n} = {batch_size * next_n})"
        )
    if top_k <= 0 or top_k > min(2048, num_cols):
        raise ValueError(f"top_k must be in (0, min(2048, num_cols={num_cols})], got {top_k}")
    if num_copy_bits <= 0 or num_copy_bits % 8:
        raise ValueError(f"num_copy_bits must be a positive whole-byte width, got {num_copy_bits}")
    copy_bytes = num_copy_bits // 8
    if copy_bytes & (copy_bytes - 1):
        raise ValueError(
            "num_copy_bits must describe a power-of-two byte alignment, "
            f"got {num_copy_bits} bits ({copy_bytes} bytes)"
        )

    input_dtype = jnp.dtype(input_values.dtype)
    supported_dtypes = {
        jnp.dtype(jnp.float16),
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float32),
    }
    if input_dtype not in supported_dtypes:
        supported = "float16, bfloat16, or float32"
        raise TypeError(f"input_values must have dtype {supported}, got {input_dtype}")
    if jnp.dtype(seq_lens.dtype) != jnp.dtype(jnp.int32):
        raise TypeError(f"seq_lens must have dtype int32, got {seq_lens.dtype}")

    dtype_bits = input_dtype.itemsize * 8
    if num_copy_bits % dtype_bits != 0:
        raise ValueError(
            f"num_copy_bits ({num_copy_bits}) must be divisible by the " f"input dtype width ({dtype_bits})"
        )

    workspace_buffers = 2 if input_dtype == jnp.dtype(jnp.float32) else 1
    workspace_elements = num_rows * workspace_buffers * num_cols
    if workspace_elements > _INT32_MAX:
        raise NotImplementedError(
            "The JAX indexer_top_k_wrapper does not yet implement the Torch "
            "row-chunking fallback required when the int32 workspace contains "
            f"more than {_INT32_MAX} elements (requested {workspace_elements})"
        )

    output_shape = (num_rows, top_k)
    output_indices, output_values = call_cutedsl(
        _make_launcher(
            jax_to_cutlass_dtype(input_dtype),
            dtype_bits,
            num_cols,
            top_k,
            next_n,
            num_copy_bits,
            num_rows > 148,
        ),
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
        use_static_tensors=True,
    )
    return IndexerTopKResult(indices=output_indices, values=output_values)


__all__ = ["IndexerTopKResult", "indexer_top_k_wrapper"]
