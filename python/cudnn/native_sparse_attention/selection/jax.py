# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for packed selection attention."""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._cute_compiler import compile_options_for_target
from ..._jax import JaxApiBase, TupleDict
from ..jax_utils import normalize_supported_dtype

SUPPORTED_COMPUTE_CAPABILITIES = (90, 100, 103, 107)


class SelectionAttention(JaxApiBase):
    """JAX callable specialized from packed THD selection metadata."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_block_indices: Any,
        sample_block_counts: Any,
        sample_cum_seqlen: Any,
        *,
        max_s_q: int,
        max_s_k: int,
        block_size: int = 64,
        scale_softmax: float | None = None,
        o_dtype: Any = None,
        acc_dtype: Any = None,
        target_compute_capability: int | None = None,
    ) -> None:
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
        self.v_desc = self._to_tensor_desc(sample_v, "sample_v")
        self.block_indices_desc = self._to_tensor_desc(
            sample_block_indices, "sample_block_indices"
        )
        self.block_counts_desc = self._to_tensor_desc(
            sample_block_counts, "sample_block_counts"
        )
        self.cum_seqlen_desc = self._to_tensor_desc(
            sample_cum_seqlen, "sample_cum_seqlen"
        )

        if self.q_desc.ndim != 3 or self.k_desc.ndim != 3 or self.v_desc.ndim != 3:
            raise ValueError(
                "SelectionAttention requires packed rank-3 (T, H, D) Q, K, and V"
            )
        if self.q_desc.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
            raise ValueError(
                f"Q must have dtype float16 or bfloat16, got {self.q_desc.dtype}"
            )
        if (
            self.k_desc.cudnn_dtype != self.q_desc.cudnn_dtype
            or self.v_desc.cudnn_dtype != self.q_desc.cudnn_dtype
        ):
            raise ValueError("Q, K, and V must have the same dtype")

        self.total_tokens, self.num_query_heads, self.head_dim = self.q_desc.shape
        k_tokens, self.num_kv_heads, k_head_dim = self.k_desc.shape
        v_tokens, num_value_heads, self.value_dim = self.v_desc.shape
        dimensions = (
            self.total_tokens,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.value_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError(
                f"SelectionAttention dimensions must be positive, got {dimensions}"
            )
        if (k_tokens, v_tokens) != (self.total_tokens, self.total_tokens):
            raise ValueError("Q, K, and V must have the same packed token count")
        if k_head_dim != self.head_dim:
            raise ValueError("Q and K head dimensions must match")
        if num_value_heads != self.num_kv_heads:
            raise ValueError("K and V head counts must match")
        if self.head_dim % 16 or self.value_dim % 16:
            raise ValueError("Q/K and V head dimensions must be multiples of 16")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("H_q must be divisible by H_kv")
        self.gqa_group_size = self.num_query_heads // self.num_kv_heads
        if self.gqa_group_size not in (1, 2, 4, 8, 16):
            raise ValueError(
                f"H_q / H_kv must be one of {{1, 2, 4, 8, 16}}, got {self.gqa_group_size}"
            )

        if self.block_indices_desc.ndim != 3 or self.block_indices_desc.shape[:2] != (
            self.total_tokens,
            self.num_kv_heads,
        ):
            raise ValueError("block_indices must have shape (T, H_kv, K)")
        if self.block_indices_desc.shape[2] <= 0:
            raise ValueError("block_indices K dimension must be positive")
        if self.block_counts_desc.shape != (self.total_tokens, self.num_kv_heads):
            raise ValueError("block_counts must have shape (T, H_kv)")
        if (
            self.block_indices_desc.cudnn_dtype != data_type.INT32
            or self.block_counts_desc.cudnn_dtype != data_type.INT32
        ):
            raise ValueError("block_indices and block_counts must have dtype int32")
        if self.cum_seqlen_desc.ndim != 1 or self.cum_seqlen_desc.shape[0] < 2:
            raise ValueError("cum_seqlen must have shape (B + 1,) with B > 0")
        if self.cum_seqlen_desc.cudnn_dtype not in (
            data_type.INT32,
            data_type.INT64,
        ):
            raise ValueError(
                "cumulative sequence lengths must have dtype int32 or int64"
            )

        self.max_s_q = int(max_s_q)
        self.max_s_k = int(max_s_k)
        if self.max_s_q <= 0 or self.max_s_k <= 0:
            raise ValueError("max_s_q and max_s_k must be positive")
        if self.max_s_q != self.max_s_k:
            raise ValueError(
                "SelectionAttention requires max_s_q and max_s_k to be identical"
            )
        self.block_size = int(block_size)
        if self.block_size not in (16, 32, 64):
            raise ValueError("block_size must be 16, 32, or 64")

        self.output_dtype = normalize_supported_dtype(
            o_dtype,
            sample_q.dtype,
            "o_dtype",
            (sample_q.dtype,),
        )
        normalize_supported_dtype(acc_dtype, jnp.float32, "acc_dtype", (jnp.float32,))
        self.scale_softmax = (
            1.0 / math.sqrt(self.head_dim)
            if scale_softmax is None
            else float(scale_softmax)
        )
        self.target_compute_capability = target_compute_capability
        self.compute_capability: int | None = None

        self.q_kernel_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct((1, *self.q_desc.shape), self.q_desc.dtype),
            "q_tensor",
        )
        self.k_kernel_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct((1, *self.k_desc.shape), self.k_desc.dtype),
            "k_tensor",
        )
        self.v_kernel_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct((1, *self.v_desc.shape), self.v_desc.dtype),
            "v_tensor",
        )
        self.o_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct(
                (1, self.total_tokens, self.num_query_heads, self.value_dim),
                self.output_dtype,
            ),
            "o_tensor",
            init_value=0,
        )
        self.l_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct(
                (1, self.total_tokens, self.num_query_heads), jnp.float32
            ),
            "l_tensor",
            init_value=0.0,
        )
        self.m_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct(
                (1, self.total_tokens, self.num_query_heads), jnp.float32
            ),
            "m_tensor",
            init_value=float("-inf"),
        )

    def check_support(self) -> bool:
        self.compute_capability = self._resolve_compute_capability(
            self.target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "SelectionAttention",
        )
        return True

    def __call__(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
        block_indices_tensor: Any,
        block_counts_tensor: Any,
        cum_seqlen_tensor: Any,
    ) -> TupleDict:
        self.check_support()
        for value, desc in (
            (q_tensor, self.q_desc),
            (k_tensor, self.k_desc),
            (v_tensor, self.v_desc),
            (block_indices_tensor, self.block_indices_desc),
            (block_counts_tensor, self.block_counts_desc),
            (cum_seqlen_tensor, self.cum_seqlen_desc),
        ):
            self._check_tensor_signature(value, desc)

        q_storage = jnp.reshape(q_tensor, self.q_kernel_desc.shape)
        k_storage = jnp.reshape(k_tensor, self.k_kernel_desc.shape)
        v_storage = jnp.reshape(v_tensor, self.v_kernel_desc.shape)
        output, lse_sum, row_max = self._call_kernel(
            (
                q_storage,
                k_storage,
                v_storage,
                block_indices_tensor,
                block_counts_tensor,
                cum_seqlen_tensor,
            ),
            launch=self._launch_kernel,
            output_descs=(self.o_desc, self.l_desc, self.m_desc),
            input_spec=(
                self._to_tensor_spec(self.q_kernel_desc),
                self._to_tensor_spec(self.k_kernel_desc),
                self._to_tensor_spec(self.v_kernel_desc),
                self._to_tensor_spec(self.block_indices_desc),
                self._to_tensor_spec(self.block_counts_desc),
                self._to_tensor_spec(self.cum_seqlen_desc),
            ),
            compile_options=compile_options_for_target(self.compute_capability),
        )
        return TupleDict(
            o_tensor=jnp.reshape(
                output, (self.total_tokens, self.num_query_heads, self.value_dim)
            ),
            l_tensor=jnp.reshape(lse_sum, (self.total_tokens, self.num_query_heads, 1)),
            m_tensor=jnp.reshape(row_max, (self.total_tokens, self.num_query_heads, 1)),
        )

    def _launch_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        v: Any,
        block_indices: Any,
        block_counts: Any,
        cum_seqlen: Any,
        output: Any,
        lse_sum: Any,
        row_max: Any,
    ) -> None:
        from cutlass import Float32
        from cutlass.jax import jax_to_cutlass_dtype

        from .NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd

        kernel = HopperSelectAttentionFwd(
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            GQA_group_size=self.gqa_group_size,
            block_size=self.block_size,
            dtype=jax_to_cutlass_dtype(self.q_desc.dtype),
            acc_dtype=Float32,
        )
        kernel(
            q,
            k,
            v,
            output,
            lse_sum,
            row_max,
            block_indices,
            block_counts,
            self.max_s_q,
            cum_seqlen,
            Float32(self.scale_softmax),
            stream,
        )


@partial(
    jax.jit,
    static_argnames=(
        "max_s_q",
        "max_s_k",
        "block_size",
        "scale_softmax",
        "o_dtype",
        "acc_dtype",
        "target_compute_capability",
    ),
)
def selection_attention_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    block_indices_tensor: Any,
    block_counts_tensor: Any,
    cum_seqlen_tensor: Any,
    *,
    max_s_q: int,
    max_s_k: int,
    block_size: int = 64,
    scale_softmax: float | None = None,
    o_dtype: Any = None,
    acc_dtype: Any = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute packed THD selection attention."""

    values = (
        q_tensor,
        k_tensor,
        v_tensor,
        block_indices_tensor,
        block_counts_tensor,
        cum_seqlen_tensor,
    )
    samples = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values)
    return SelectionAttention(
        *samples,
        max_s_q=max_s_q,
        max_s_k=max_s_k,
        block_size=block_size,
        scale_softmax=scale_softmax,
        o_dtype=o_dtype,
        acc_dtype=acc_dtype,
        target_compute_capability=target_compute_capability,
    )(*values)


__all__ = [
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SelectionAttention",
    "selection_attention_wrapper",
]
