# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX APIs for DeepSeek indexer top-K and index utilities."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._jax import JaxApiBase, TupleDict
from ..utils.compiler import compile_options_for_target
from .op import INT32_MAX, IndexerTopKOp, SUPPORTED_COMPUTE_CAPABILITIES, SUPPORTED_INPUT_DTYPES


class IndexerTopK(JaxApiBase):
    """JAX callable specialized from top-K input metadata."""

    def __init__(
        self,
        sample_input_values: Any,
        sample_seq_lens: Any,
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
        num_copy_bits: int = 256,
        *,
        target_compute_capability: int | None = None,
    ) -> None:
        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "IndexerTopK",
        )
        self.input_desc = self._to_tensor_desc(sample_input_values, "sample_input_values")
        self.seq_lens_desc = self._to_tensor_desc(sample_seq_lens, "sample_seq_lens")
        self.top_k = int(top_k)
        self.next_n = int(next_n)
        self.return_val = bool(return_val)
        self.num_copy_bits = int(num_copy_bits)

        if self.input_desc.ndim != 2:
            raise ValueError(f"input_values must have rank 2, got {self.input_desc.shape}")
        if self.seq_lens_desc.ndim != 1:
            raise ValueError(f"seq_lens must have rank 1, got {self.seq_lens_desc.shape}")
        if self.input_desc.cudnn_dtype not in SUPPORTED_INPUT_DTYPES:
            raise ValueError(f"input_values must have dtype float16, bfloat16, or float32, got {self.input_desc.dtype}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        output_shape = (self.input_desc.shape[0], self.top_k)
        self.indices_desc = self.input_desc.compact_like(
            cudnn_dtype=data_type.INT32,
            shape=output_shape,
            name="indices",
        )
        self.values_desc = (
            self.input_desc.compact_like(
                cudnn_dtype=self.input_desc.cudnn_dtype,
                shape=output_shape,
                name="values",
            )
            if self.return_val
            else None
        )
        buffer_count = 2 if self.input_desc.cudnn_dtype == data_type.FLOAT else 1
        self.workspace_desc = self.input_desc.compact_like(
            cudnn_dtype=data_type.INT32,
            shape=(self.input_desc.shape[0], buffer_count, self.input_desc.shape[1]),
            name="extra_buffer",
        )
        self._op = IndexerTopKOp(
            input_values=self.input_desc,
            seq_lens=self.seq_lens_desc,
            output_indices=self.indices_desc,
            output_values=self.values_desc,
            workspace=self.workspace_desc,
            top_k=self.top_k,
            next_n=self.next_n,
            return_val=self.return_val,
            num_copy_bits=self.num_copy_bits,
            target_compute_capability=self.target_compute_capability,
        )

    def check_support(self) -> bool:
        self._op.check_support()
        workspace_elements = 1
        for extent in self.workspace_desc.shape:
            workspace_elements *= extent
        if workspace_elements > INT32_MAX:
            raise NotImplementedError(
                "JAX IndexerTopK does not support the Torch row-chunking fallback when "
                f"the workspace exceeds {INT32_MAX} elements (got {workspace_elements})"
            )
        return True

    def __call__(self, input_values: Any, seq_lens: Any) -> TupleDict:
        self.check_support()
        self._check_tensor_signature(input_values, self.input_desc)
        self._check_tensor_signature(seq_lens, self.seq_lens_desc)

        output_descs = (self.indices_desc,) if self.values_desc is None else (self.indices_desc, self.values_desc)
        results = self._call_kernel(
            (input_values, seq_lens),
            output_descs=output_descs,
            workspace_descs=(self.workspace_desc,),
            input_spec=(self._to_tensor_spec(self.input_desc), self._to_tensor_spec(self.seq_lens_desc)),
            compile_options=compile_options_for_target(self.target_compute_capability),
        )
        indices = results[0]
        values = results[1] if self.return_val else None
        return TupleDict(indices=indices, values=values)

    def _launch(
        self,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        workspaces: tuple[Any, ...],
        stream: Any,
    ) -> None:
        import cutlass

        from .indexer_top_k_decode_varlen import IndexerTopKKernelVarlenDecode

        dtype_by_cudnn = {
            data_type.HALF: cutlass.Float16,
            data_type.BFLOAT16: cutlass.BFloat16,
            data_type.FLOAT: cutlass.Float32,
        }
        input_values, seq_lens = inputs
        output_indices = outputs[0]
        output_values = outputs[1] if self.return_val else None
        (extra_buffer,) = workspaces
        resolved = (
            self._op.max_num_cols,
            self._op.large_occupancy,
        )
        if any(value is None for value in resolved):
            raise RuntimeError("IndexerTopK launch configuration was not resolved by check_support()")
        max_num_cols, large_occupancy = resolved
        kernel = IndexerTopKKernelVarlenDecode(
            dtype_by_cudnn[self.input_desc.cudnn_dtype],
            max_num_cols,
            self.top_k,
            self.next_n,
            num_copy_bits=self.num_copy_bits,
            return_val=self.return_val,
            large_occupancy=large_occupancy,
        )
        kernel(
            input_values,
            None,
            extra_buffer,
            None,
            seq_lens,
            output_indices,
            output_values,
            stream,
            enable_persistent_dynamic_scheduling=False,
            min_blocks_per_mp=1,
        )


class _LocalToGlobal(JaxApiBase):
    def __init__(
        self,
        sample_local_indices: Any,
        seqlen_k: int,
        *,
        sample_cu_seqlens_q: Any | None = None,
        sample_cu_seqlens_k: Any | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        if (sample_cu_seqlens_q is None) != (sample_cu_seqlens_k is None):
            raise ValueError("Packed local-to-global requires both cumulative sequence tensors")
        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "local_to_global",
        )
        self.local_desc = self._to_tensor_desc(sample_local_indices, "local_indices")
        self.cu_q_desc = None if sample_cu_seqlens_q is None else self._to_tensor_desc(sample_cu_seqlens_q, "cu_seqlens_q")
        self.cu_k_desc = None if sample_cu_seqlens_k is None else self._to_tensor_desc(sample_cu_seqlens_k, "cu_seqlens_k")
        self.is_varlen = self.cu_q_desc is not None
        self.seqlen_k = int(seqlen_k)
        self.output_desc = self.local_desc.compact_like(
            cudnn_dtype=data_type.INT32,
            shape=self.local_desc.shape,
            name="indices",
        )

    def check_support(self) -> bool:
        expected_rank = 2 if self.is_varlen else 3
        if self.local_desc.ndim != expected_rank:
            raise ValueError(f"local_indices must have rank {expected_rank}, got {self.local_desc.shape}")
        if self.local_desc.cudnn_dtype not in (data_type.INT32, data_type.INT64):
            raise ValueError(f"local_indices must have dtype int32 or int64, got {self.local_desc.dtype}")
        if any(extent <= 0 for extent in self.local_desc.shape):
            raise ValueError(f"local_indices dimensions must be positive, got {self.local_desc.shape}")
        if not self.local_desc.is_compact(tuple(reversed(range(expected_rank)))):
            raise ValueError("local_indices must be row-major contiguous")
        if self.seqlen_k <= 0:
            raise ValueError(f"seqlen_k must be positive, got {self.seqlen_k}")
        if self.is_varlen:
            if self.cu_q_desc is None or self.cu_k_desc is None:
                raise RuntimeError("Packed cumulative sequence descriptors were not configured")
            for desc, name in ((self.cu_q_desc, "cu_seqlens_q"), (self.cu_k_desc, "cu_seqlens_k")):
                if desc.ndim != 1 or desc.cudnn_dtype != data_type.INT32 or not desc.is_compact((0,)):
                    raise ValueError(f"{name} must be a contiguous rank-1 int32 tensor")
            if self.cu_q_desc.shape != self.cu_k_desc.shape:
                raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
            if self.cu_q_desc.shape[0] < 2:
                raise ValueError("Packed cumulative sequence tensors must contain at least two entries")
        return True

    def __call__(
        self,
        local_indices: Any,
        *,
        cu_seqlens_q: Any | None = None,
        cu_seqlens_k: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        self._check_tensor_signature(local_indices, self.local_desc)
        inputs = [local_indices]
        if self.is_varlen:
            if cu_seqlens_q is None or cu_seqlens_k is None:
                raise ValueError("Packed local-to-global requires both cumulative sequence tensors")
            if self.cu_q_desc is None or self.cu_k_desc is None:
                raise RuntimeError("Packed cumulative sequence descriptors were not configured")
            self._check_tensor_signature(cu_seqlens_q, self.cu_q_desc)
            self._check_tensor_signature(cu_seqlens_k, self.cu_k_desc)
            inputs.extend((cu_seqlens_q, cu_seqlens_k))
        elif cu_seqlens_q is not None or cu_seqlens_k is not None:
            raise ValueError("Fixed local-to-global does not accept cumulative sequence tensors")

        (result,) = self._call_kernel(
            tuple(inputs),
            output_descs=(self.output_desc,),
            compile_options=compile_options_for_target(self.target_compute_capability, "--opt-level 3"),
        )
        return TupleDict(indices=result)

    def _launch(self, inputs, outputs, workspaces, stream) -> None:
        from cutlass import Int32

        from .local_to_global_dsl import LocalToGlobalTopK

        local_indices, *optional = inputs
        if self.is_varlen:
            cu_q, cu_k = optional
        else:
            cu_q = cu_k = None
        (output,) = outputs
        if workspaces:
            raise RuntimeError("local_to_global does not use workspaces")
        LocalToGlobalTopK(is_varlen=self.is_varlen)(
            local_indices,
            output,
            Int32(self.seqlen_k),
            cu_q,
            cu_k,
            stream,
        )


class _Compactify(JaxApiBase):
    def __init__(self, sample_indices: Any, *, target_compute_capability: int | None = None) -> None:
        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "compactify",
        )
        self.input_desc = self._to_tensor_desc(sample_indices, "indices")
        if self.input_desc.ndim != 2:
            raise ValueError(f"Compactify expects flattened rank-2 input, got {self.input_desc.shape}")
        rows, cols = self.input_desc.shape
        self.rows = rows
        self.cols = cols
        self.output_desc = self.input_desc.compact_like(
            cudnn_dtype=data_type.INT32,
            shape=(rows, cols),
            name="indices",
        )
        self.length_desc = self.input_desc.compact_like(
            cudnn_dtype=data_type.INT32,
            shape=(rows,),
            name="topk_length",
        )

    def check_support(self) -> bool:
        if self.input_desc.cudnn_dtype != data_type.INT32:
            raise ValueError(f"indices must have dtype int32, got {self.input_desc.dtype}")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"indices dimensions must be positive, got {self.input_desc.shape}")
        if not self.input_desc.is_compact((1, 0)):
            raise ValueError("indices must be row-major contiguous")
        return True

    def __call__(self, indices: Any) -> TupleDict:
        self.check_support()
        self._check_tensor_signature(indices, self.input_desc)
        compact_indices, topk_length = self._call_kernel(
            (indices,),
            output_descs=(self.output_desc, self.length_desc),
            compile_options=compile_options_for_target(self.target_compute_capability, "--opt-level 3"),
        )
        return TupleDict(indices=compact_indices, topk_length=topk_length)

    def _launch(self, inputs, outputs, workspaces, stream) -> None:
        from cutlass import Int32

        from .compactify import CompactifyKernel

        (indices,) = inputs
        compact_indices, topk_length = outputs
        if workspaces:
            raise RuntimeError("compactify does not use workspaces")
        CompactifyKernel(cols=self.cols)(indices, compact_indices, topk_length, Int32(self.rows), stream)


@partial(
    jax.jit,
    static_argnames=("top_k", "next_n", "return_val", "num_copy_bits", "target_compute_capability"),
)
def indexer_top_k_wrapper(
    input_values: Any,
    seq_lens: Any,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
    *,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Select per-row top-K indices and optional values from JAX.

    Runtime lengths are consumed on device and cannot be inspected while
    tracing. Every ``seq_lens[b]`` must satisfy
    ``top_k + next_n - 1 <= seq_lens[b] <= input_values.shape[1]``.
    """

    return IndexerTopK(
        jax.ShapeDtypeStruct(input_values.shape, input_values.dtype),
        jax.ShapeDtypeStruct(seq_lens.shape, seq_lens.dtype),
        top_k,
        next_n=next_n,
        return_val=return_val,
        num_copy_bits=num_copy_bits,
        target_compute_capability=target_compute_capability,
    )(input_values, seq_lens)


@partial(jax.jit, static_argnames=("seqlen_k", "target_compute_capability"))
def local_to_global_wrapper(
    local_indices: Any,
    seqlen_k: int,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_k: Any | None = None,
    *,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Convert local top-K indices to the global flattened KV index space."""

    sample_cu_q = None if cu_seqlens_q is None else jax.ShapeDtypeStruct(cu_seqlens_q.shape, cu_seqlens_q.dtype)
    sample_cu_k = None if cu_seqlens_k is None else jax.ShapeDtypeStruct(cu_seqlens_k.shape, cu_seqlens_k.dtype)
    return _LocalToGlobal(
        jax.ShapeDtypeStruct(local_indices.shape, local_indices.dtype),
        seqlen_k,
        sample_cu_seqlens_q=sample_cu_q,
        sample_cu_seqlens_k=sample_cu_k,
        target_compute_capability=target_compute_capability,
    )(
        local_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )


@partial(jax.jit, static_argnames=("target_compute_capability",))
def compactify_wrapper(indices: Any, *, target_compute_capability: int | None = None) -> TupleDict:
    """Pack nonnegative indices to the front of every flattened row."""

    if indices.ndim not in (2, 3):
        raise ValueError(f"indices must have rank 2 or 3, got shape {indices.shape}")
    rows = 1
    for extent in indices.shape[:-1]:
        rows *= extent
    flat_indices = jnp.reshape(indices, (rows, indices.shape[-1]))
    return _Compactify(
        jax.ShapeDtypeStruct(flat_indices.shape, flat_indices.dtype),
        target_compute_capability=target_compute_capability,
    )(flat_indices)


__all__ = [
    "IndexerTopK",
    "compactify_wrapper",
    "indexer_top_k_wrapper",
    "local_to_global_wrapper",
]
