# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for SM100 NSA top-K reduction."""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._jax.compiler import compile_options_for_target
from ..._jax import JaxApiBase, JaxTensorDesc, TupleDict
from ..._jax.layout import to_public_axes
from ..jax_utils import (
    FIXED_LAYOUTS,
    describe_fixed_data,
    fixed_data_mode,
    make_fixed_output,
    normalize_attention_layout,
    normalize_supported_dtype,
    require_fixed_qkv,
)

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)
_PACKED_LAYOUT = "THD"
_KERNEL_DATA_AXES = "BHSD"
_KERNEL_DATA_STRIDE_ORDER = (3, 1, 2, 0)


class TopKReduction(JaxApiBase):
    """JAX callable specialized from fixed BHSD/BSHD or packed THD metadata.

    Packed inputs use logical ``(T, H, D)`` Q/K arrays and require both
    cumulative sequence-length arrays plus static ``max_s_q`` and ``max_s_k``.
    The cumulative arrays remain explicit custom-call operands. Their contents,
    including the final packed-token offsets, are therefore validated by the
    caller rather than read while JAX traces the operation.
    """

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_lse: Any,
        sample_cum_seqlen_q: Any | None = None,
        sample_cum_seqlen_k: Any | None = None,
        max_s_q: int | None = None,
        max_s_k: int | None = None,
        acc_dtype: Any = None,
        k_value: int = 16,
        selection_block_size: int = 64,
        compress_stride: int = 32,
        is_causal: bool = True,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        scale_softmax: float | None = None,
        layout: str | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        q_rank = len(tuple(sample_q.shape))
        k_rank = len(tuple(sample_k.shape))
        if q_rank != k_rank:
            raise ValueError(
                f"Q and K must use the same rank, got {q_rank} and {k_rank}"
            )
        self.input_layout = normalize_attention_layout(layout, q_rank)

        if self.input_layout in FIXED_LAYOUTS:
            self.data_mode = fixed_data_mode(
                self.input_layout, kernel_axes=_KERNEL_DATA_AXES
            )
            self._init_fixed_shape(
                sample_q,
                sample_k,
                sample_lse,
                sample_cum_seqlen_q,
                sample_cum_seqlen_k,
            )
        else:
            self.data_mode = None
            self._init_packed_shape(
                sample_q,
                sample_k,
                sample_lse,
                sample_cum_seqlen_q,
                sample_cum_seqlen_k,
                max_s_q,
                max_s_k,
            )

        normalize_supported_dtype(acc_dtype, jnp.float32, "acc_dtype", (jnp.float32,))
        self.k_value = int(k_value)
        self.selection_block_size = int(selection_block_size)
        self.compress_stride = int(compress_stride)
        self.is_causal = bool(is_causal)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self._check_configuration()
        self.scale_softmax = (
            1.0 / math.sqrt(self.head_dim)
            if scale_softmax is None
            else float(scale_softmax)
        )
        self.target_compute_capability = target_compute_capability
        self.compute_capability: int | None = None

        if self.input_layout in FIXED_LAYOUTS:
            canonical_output_shape = (
                self.batch,
                self.num_kv_heads,
                self.seqlen_q,
                self.k_value,
            )
            output_shape = to_public_axes(canonical_output_shape, self.data_mode)
            self.scores_desc = make_fixed_output(
                output_shape,
                jnp.float32,
                "topk_scores_tensor",
                layout=self.input_layout,
                kernel_axes=_KERNEL_DATA_AXES,
                kernel_stride_order=_KERNEL_DATA_STRIDE_ORDER,
                init_value=float("-inf"),
            )
            self.indices_desc = make_fixed_output(
                output_shape,
                jnp.int32,
                "topk_indices_tensor",
                layout=self.input_layout,
                kernel_axes=_KERNEL_DATA_AXES,
                kernel_stride_order=_KERNEL_DATA_STRIDE_ORDER,
                init_value=-1,
            )
        else:
            output_shape = (self.total_q_tokens, self.num_kv_heads, self.k_value)
            self.scores_desc = JaxTensorDesc.from_shape(
                output_shape,
                jnp.float32,
                name="topk_scores_tensor",
                init_value=float("-inf"),
            )
            self.indices_desc = JaxTensorDesc.from_shape(
                output_shape,
                jnp.int32,
                name="topk_indices_tensor",
                init_value=-1,
            )

    def _init_fixed_shape(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_lse: Any,
        sample_cum_seqlen_q: Any | None,
        sample_cum_seqlen_k: Any | None,
    ) -> None:
        if sample_cum_seqlen_q is not None or sample_cum_seqlen_k is not None:
            raise ValueError(
                "cum_seqlen_q and cum_seqlen_k must both be omitted for BHSD layout"
            )

        # The kernel consumes BHSD modes with the H-inside-S storage used by
        # the Torch wrapper's transposed BSHD tensors.
        self.q_desc = describe_fixed_data(
            sample_q,
            "sample_q",
            layout=self.input_layout,
            kernel_axes=_KERNEL_DATA_AXES,
            kernel_stride_order=_KERNEL_DATA_STRIDE_ORDER,
        )
        self.k_desc = describe_fixed_data(
            sample_k,
            "sample_k",
            layout=self.input_layout,
            kernel_axes=_KERNEL_DATA_AXES,
            kernel_stride_order=_KERNEL_DATA_STRIDE_ORDER,
        )
        (
            self.batch,
            self.num_query_heads,
            self.num_kv_heads,
            self.seqlen_q,
            self.seqlen_k,
            self.head_dim,
        ) = require_fixed_qkv(
            self.q_desc,
            self.k_desc,
            operation_name="TopKReduction",
            kernel_axes=_KERNEL_DATA_AXES,
        )
        self.total_q_tokens = self.batch * self.seqlen_q
        self.lse_desc = self._to_tensor_desc(sample_lse, "sample_lse")
        expected_lse_shape = (self.batch, self.num_query_heads, self.seqlen_q)
        if self.lse_desc.shape != expected_lse_shape:
            raise ValueError(
                f"sample_lse must have shape {expected_lse_shape}, got {self.lse_desc.shape}"
            )
        if self.lse_desc.cudnn_dtype != data_type.FLOAT:
            raise ValueError(
                f"sample_lse must have dtype float32, got {self.lse_desc.dtype}"
            )

        self.max_s_q = self.seqlen_q
        self.max_s_k = self.seqlen_k
        self.cum_seqlen_q_desc = None
        self.cum_seqlen_k_desc = None
        self.q_kernel_desc = self.q_desc
        self.k_kernel_desc = self.k_desc
        self.lse_kernel_desc = self.lse_desc

    def _init_packed_shape(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_lse: Any,
        sample_cum_seqlen_q: Any | None,
        sample_cum_seqlen_k: Any | None,
        max_s_q: int | None,
        max_s_k: int | None,
    ) -> None:
        if sample_cum_seqlen_q is None or sample_cum_seqlen_k is None:
            raise ValueError(
                "cum_seqlen_q and cum_seqlen_k are both required for THD layout"
            )
        if max_s_q is None or max_s_k is None:
            raise ValueError("max_s_q and max_s_k are both required for THD layout")

        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
        self.lse_desc = self._to_tensor_desc(sample_lse, "sample_lse")
        self.cum_seqlen_q_desc = self._to_tensor_desc(
            sample_cum_seqlen_q, "sample_cum_seqlen_q"
        )
        self.cum_seqlen_k_desc = self._to_tensor_desc(
            sample_cum_seqlen_k, "sample_cum_seqlen_k"
        )

        self.total_q_tokens, self.num_query_heads, self.head_dim = self.q_desc.shape
        self.total_k_tokens, self.num_kv_heads, k_head_dim = self.k_desc.shape
        dimensions = (
            self.total_q_tokens,
            self.total_k_tokens,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError(
                f"TopKReduction dimensions must be positive, got {dimensions}"
            )
        if self.q_desc.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
            raise ValueError(
                f"TopKReduction requires float16 or bfloat16 inputs, got {self.q_desc.dtype}"
            )
        if self.k_desc.cudnn_dtype != self.q_desc.cudnn_dtype:
            raise ValueError("sample_k must have the same dtype as sample_q")
        if k_head_dim != self.head_dim:
            raise ValueError(
                f"Q and K head dimensions must match, got {self.head_dim} and {k_head_dim}"
            )
        if self.head_dim not in (32, 64, 128):
            raise ValueError(
                f"head dimension must be one of 32, 64, 128, got {self.head_dim}"
            )
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"H_q ({self.num_query_heads}) must be divisible by H_kv ({self.num_kv_heads})"
            )

        expected_lse_shapes = (
            (self.total_q_tokens, self.num_query_heads),
            (self.total_q_tokens, self.num_query_heads, 1),
        )
        if self.lse_desc.shape not in expected_lse_shapes:
            raise ValueError(
                "sample_lse must have shape "
                f"{expected_lse_shapes[0]} or {expected_lse_shapes[1]}, got {self.lse_desc.shape}"
            )
        if self.lse_desc.cudnn_dtype != data_type.FLOAT:
            raise ValueError(
                f"sample_lse must have dtype float32, got {self.lse_desc.dtype}"
            )

        if (
            self.cum_seqlen_q_desc.ndim != 1
            or self.cum_seqlen_k_desc.ndim != 1
            or self.cum_seqlen_q_desc.shape != self.cum_seqlen_k_desc.shape
            or self.cum_seqlen_q_desc.shape[0] < 2
        ):
            raise ValueError(
                "cum_seqlen_q and cum_seqlen_k must have the same shape (B + 1,) with B > 0"
            )
        if (
            self.cum_seqlen_q_desc.cudnn_dtype != data_type.INT32
            or self.cum_seqlen_k_desc.cudnn_dtype != data_type.INT32
        ):
            raise ValueError("cum_seqlen_q and cum_seqlen_k must have dtype int32")

        self.batch = self.cum_seqlen_q_desc.shape[0] - 1
        self.seqlen_q = self.total_q_tokens
        self.seqlen_k = self.total_k_tokens
        self.max_s_q = int(max_s_q)
        self.max_s_k = int(max_s_k)
        if (
            self.max_s_q <= 0
            or self.max_s_k <= 0
            or self.max_s_q > self.total_q_tokens
            or self.max_s_k > self.total_k_tokens
        ):
            raise ValueError(
                "max_s_q and max_s_k must be positive and no larger than "
                "their packed token counts"
            )

        # These virtual rank-4 views preserve the packed THD byte order while
        # exposing the B,H,T,D shapes read by the CuTe kernel. LSE is likewise
        # presented as B,H,T, matching the Torch wrapper's transpose.
        self.q_kernel_desc = JaxTensorDesc.from_shape(
            (1, self.num_query_heads, self.total_q_tokens, self.head_dim),
            self.q_desc.dtype,
            name="q_tensor",
            public_stride_order=_KERNEL_DATA_STRIDE_ORDER,
        )
        self.k_kernel_desc = JaxTensorDesc.from_shape(
            (1, self.num_kv_heads, self.total_k_tokens, self.head_dim),
            self.k_desc.dtype,
            name="k_tensor",
            public_stride_order=_KERNEL_DATA_STRIDE_ORDER,
        )
        self.lse_kernel_desc = JaxTensorDesc.from_shape(
            (1, self.num_query_heads, self.total_q_tokens),
            jnp.float32,
            name="lse_tensor",
        )

    def _check_configuration(self) -> None:
        if self.mma_tiler_mn != (128, 128):
            raise ValueError(
                f"mma_tiler_mn must be (128, 128), got {self.mma_tiler_mn}"
            )
        if self.selection_block_size <= 0 or self.compress_stride <= 0:
            raise ValueError(
                "selection_block_size and compress_stride must be positive"
            )
        if self.selection_block_size % self.compress_stride:
            raise ValueError(
                "selection_block_size must be divisible by compress_stride"
            )
        reduction_width = self.selection_block_size // self.compress_stride
        if (
            reduction_width > self.mma_tiler_mn[1]
            or self.mma_tiler_mn[1] % reduction_width
        ):
            raise ValueError(
                "selection_block_size / compress_stride must divide the MMA N tile"
            )
        candidate_count = self.mma_tiler_mn[1] // reduction_width
        if self.k_value <= 0 or self.k_value % 4 or self.k_value > candidate_count:
            raise ValueError(
                "k_value must be a positive multiple of 4 no greater than the "
                f"per-tile candidate count ({candidate_count}), got {self.k_value}"
            )

    def check_support(self) -> bool:
        self.compute_capability = self._resolve_compute_capability(
            self.target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "TopKReduction",
        )
        return True

    def __call__(
        self,
        q_tensor: Any,
        k_tensor: Any,
        lse_tensor: Any,
        cum_seqlen_q_tensor: Any | None = None,
        cum_seqlen_k_tensor: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        if self.input_layout in FIXED_LAYOUTS:
            if cum_seqlen_q_tensor is not None or cum_seqlen_k_tensor is not None:
                raise ValueError(
                    "cum_seqlen_q and cum_seqlen_k must be omitted for fixed layout"
                )
            inputs = (q_tensor, k_tensor, lse_tensor)
            input_descs = (
                self.q_kernel_desc,
                self.k_kernel_desc,
                self.lse_kernel_desc,
            )
            launch = self._launch_kernel
        else:
            if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
                raise ValueError(
                    "cum_seqlen_q and cum_seqlen_k are both required for THD layout"
                )
            for value, desc in (
                (q_tensor, self.q_desc),
                (k_tensor, self.k_desc),
                (lse_tensor, self.lse_desc),
            ):
                self._check_tensor_signature(value, desc)
            q_storage = jnp.transpose(
                jnp.reshape(
                    q_tensor,
                    (1, self.total_q_tokens, self.num_query_heads, self.head_dim),
                ),
                (0, 2, 1, 3),
            )
            k_storage = jnp.transpose(
                jnp.reshape(
                    k_tensor,
                    (1, self.total_k_tokens, self.num_kv_heads, self.head_dim),
                ),
                (0, 2, 1, 3),
            )
            lse_storage = jnp.transpose(
                jnp.reshape(lse_tensor, (self.total_q_tokens, self.num_query_heads)),
                (1, 0),
            )[None, ...]
            inputs = (
                q_storage,
                k_storage,
                lse_storage,
                cum_seqlen_q_tensor,
                cum_seqlen_k_tensor,
            )
            input_descs = (
                self.q_kernel_desc,
                self.k_kernel_desc,
                self.lse_kernel_desc,
                self.cum_seqlen_q_desc,
                self.cum_seqlen_k_desc,
            )
            launch = self._launch_packed_kernel

        scores, indices = self._call_kernel(
            inputs,
            launch=launch,
            output_descs=(self.scores_desc, self.indices_desc),
            input_descs=input_descs,
            compile_options=compile_options_for_target(self.compute_capability),
        )
        return TupleDict(topk_scores_tensor=scores, topk_indices_tensor=indices)

    def _launch_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        lse: Any,
        topk_scores: Any,
        topk_indices: Any,
    ) -> None:
        self._invoke_kernel(
            stream,
            q,
            k,
            lse,
            topk_scores,
            topk_indices,
            None,
            None,
        )

    def _launch_packed_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        lse: Any,
        cum_seqlen_q: Any,
        cum_seqlen_k: Any,
        topk_scores: Any,
        topk_indices: Any,
    ) -> None:
        self._invoke_kernel(
            stream,
            q,
            k,
            lse,
            topk_scores,
            topk_indices,
            cum_seqlen_q,
            cum_seqlen_k,
        )

    def _invoke_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        lse: Any,
        topk_scores: Any,
        topk_indices: Any,
        cum_seqlen_q: Any | None,
        cum_seqlen_k: Any | None,
    ) -> None:
        from cutlass import Float32, Int32
        from cutlass.jax import jax_to_cutlass_dtype

        from .nsa_top_k_reduction_fwd import FineGrainedReductionQK

        kernel = FineGrainedReductionQK(
            element_dtype=jax_to_cutlass_dtype(self.q_desc.dtype),
            acc_dtype=Float32,
            k_value=self.k_value,
            selection_block_size=self.selection_block_size,
            compress_block_sliding_stride=self.compress_stride,
            mma_tiler=(128, 128, self.head_dim),
            is_causal=self.is_causal,
        )
        problem_size = tuple(
            Int32(value)
            for value in (
                self.batch,
                self.max_s_q,
                self.max_s_k,
                self.num_query_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        )
        kernel(
            problem_size,
            q,
            k,
            lse,
            topk_scores,
            topk_indices,
            Float32(self.scale_softmax * math.log2(math.e)),
            cum_seqlen_q,
            cum_seqlen_k,
            stream,
        )


@partial(
    jax.jit,
    static_argnames=(
        "max_s_q",
        "max_s_k",
        "acc_dtype",
        "k_value",
        "selection_block_size",
        "compress_stride",
        "is_causal",
        "mma_tiler_mn",
        "scale_softmax",
        "layout",
        "target_compute_capability",
    ),
)
def topk_reduction_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    lse_tensor: Any,
    cum_seqlen_q_tensor: Any | None = None,
    cum_seqlen_k_tensor: Any | None = None,
    max_s_q: int | None = None,
    max_s_k: int | None = None,
    acc_dtype: Any = None,
    k_value: int = 16,
    selection_block_size: int = 64,
    compress_stride: int = 32,
    is_causal: bool = True,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    scale_softmax: float | None = None,
    layout: str | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Select blocks for fixed BHSD/BSHD or packed THD inputs.

    Fixed outputs follow ``layout``; fixed LSE remains ``(B, H, S)``.
    """

    return TopKReduction(
        q_tensor,
        k_tensor,
        lse_tensor,
        cum_seqlen_q_tensor,
        cum_seqlen_k_tensor,
        max_s_q=max_s_q,
        max_s_k=max_s_k,
        acc_dtype=acc_dtype,
        k_value=k_value,
        selection_block_size=selection_block_size,
        compress_stride=compress_stride,
        is_causal=is_causal,
        mma_tiler_mn=mma_tiler_mn,
        scale_softmax=scale_softmax,
        layout=layout,
        target_compute_capability=target_compute_capability,
    )(
        q_tensor,
        k_tensor,
        lse_tensor,
        cum_seqlen_q_tensor,
        cum_seqlen_k_tensor,
    )


__all__ = [
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "TopKReduction",
    "topk_reduction_wrapper",
]
