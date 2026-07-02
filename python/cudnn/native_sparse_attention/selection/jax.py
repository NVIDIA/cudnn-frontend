# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for SM90 NSA selection attention."""

from __future__ import annotations

import math
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


def require_array(
    name: str,
    value: Any,
    *,
    rank: int,
    dtype: Any | tuple[Any, ...],
) -> tuple[tuple[int, ...], Any]:
    """Require array shape/dtype metadata and return its shape and dtype."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must have shape and dtype metadata")
    shape = tuple(value.shape)
    if len(shape) != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {shape}")
    valid_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    resolved_dtype = require_dtype(f"{name}.dtype", value, valid_dtypes)
    return shape, resolved_dtype


def _launch(
    stream,
    q,
    k,
    v,
    block_indices,
    block_counts,
    cum_seqlen_q,
    cum_seqlen_k,
    output,
    lse_sum,
    row_max,
    *,
    element_dtype: Any,
    head_dim: int,
    value_dim: int,
    gqa_group_size: int,
    block_size: int,
    max_s_q: int,
    scale_softmax: float,
):
    from cutlass import Float32

    from .NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd

    kernel = HopperSelectAttentionFwd(
        head_dim=head_dim,
        value_dim=value_dim,
        GQA_group_size=gqa_group_size,
        block_size=block_size,
        dtype=element_dtype,
        acc_dtype=Float32,
    )

    # Selection attention is currently self-attention. The second runtime
    # offsets operand preserves API parity and is required to match Q's
    # offsets, but the native kernel has a single seq_offsets argument.
    del cum_seqlen_k
    kernel(
        q,
        k,
        v,
        output,
        lse_sum,
        row_max,
        block_indices,
        block_counts,
        max_s_q,
        cum_seqlen_q,
        Float32(scale_softmax),
        stream,
    )


def _selection_attention_impl(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    block_indices_tensor: Any,
    block_counts_tensor: Any,
    cum_seqlen_q_tensor: Any,
    cum_seqlen_k_tensor: Any,
    *,
    max_s_q: int,
    max_s_k: int,
    block_size: int = 64,
    scale_softmax: float | None = None,
    o_dtype: Any = None,
    acc_dtype: Any = None,
    _validate_only: bool = False,
) -> TupleDict:
    """Compute packed THD selection attention with the SM90 CuTe kernel.

    Q, K, and V use compact row-major ``(T, H, D)`` storage. Block indices
    have shape ``(T, H_kv, K)`` and block counts have shape ``(T, H_kv)``.
    Both cumulative-length arrays are runtime ``int32`` or ``int64`` operands of shape
    ``(B + 1,)``. ``max_s_q`` and ``max_s_k`` are required static integers.

    The current kernel implements self-attention: Q and KV must have the same
    packed token count, the two cumulative-length arrays must contain identical
    runtime values, and both static maxima must be equal. Runtime offsets must
    start at zero, be nondecreasing, end at T, and describe lengths no greater
    than the static maximum. Block counts and indices are trusted to be valid
    for their corresponding sequences while tracing with :func:`jax.jit`.

    O has shape ``(T, H_q, D_v)`` and the input dtype. L and M have shape
    ``(T, H_q, 1)`` and dtype ``float32``.
    """

    q_shape, input_dtype = require_array(
        "q_tensor",
        q_tensor,
        rank=3,
        dtype=(jnp.float16, jnp.bfloat16),
    )
    k_shape, _ = require_array("k_tensor", k_tensor, rank=3, dtype=input_dtype)
    v_shape, _ = require_array("v_tensor", v_tensor, rank=3, dtype=input_dtype)

    total_tokens, num_query_heads, head_dim = q_shape
    k_total_tokens, num_kv_heads, k_head_dim = k_shape
    v_total_tokens, num_value_heads, value_dim = v_shape
    dimensions = {
        "T": total_tokens,
        "H_q": num_query_heads,
        "H_kv": num_kv_heads,
        "D_qk": head_dim,
        "D_v": value_dim,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Selection-attention dimensions must be positive, got " + ", ".join(nonpositive))
    if (k_total_tokens, v_total_tokens) != (total_tokens, total_tokens):
        raise ValueError("Q, K, and V must have the same packed token count for self-attention, " f"got {total_tokens}, {k_total_tokens}, and {v_total_tokens}")
    if k_head_dim != head_dim:
        raise ValueError(f"Q and K head dimensions must match, got {head_dim} and {k_head_dim}")
    if num_value_heads != num_kv_heads:
        raise ValueError(f"K and V head counts must match, got {num_kv_heads} and {num_value_heads}")
    if head_dim % 16 or value_dim % 16:
        raise ValueError(f"D_qk and D_v must be multiples of 16, got {head_dim} and {value_dim}")
    if num_query_heads % num_kv_heads:
        raise ValueError(f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})")
    gqa_group_size = num_query_heads // num_kv_heads
    if gqa_group_size not in (1, 2, 4, 8):
        raise ValueError(f"H_q / H_kv must be one of {{1, 2, 4, 8}}, got {gqa_group_size}")

    block_indices_shape, _ = require_array(
        "block_indices_tensor",
        block_indices_tensor,
        rank=3,
        dtype=jnp.int32,
    )
    if block_indices_shape[:2] != (total_tokens, num_kv_heads) or block_indices_shape[2] <= 0:
        raise ValueError("block_indices_tensor must have shape " f"(T, H_kv, K) with K > 0, got {block_indices_shape}")
    block_counts_shape, _ = require_array(
        "block_counts_tensor",
        block_counts_tensor,
        rank=2,
        dtype=jnp.int32,
    )
    if block_counts_shape != (total_tokens, num_kv_heads):
        raise ValueError(f"block_counts_tensor must have shape {(total_tokens, num_kv_heads)}, " f"got {block_counts_shape}")

    cum_q_shape, offsets_dtype = require_array(
        "cum_seqlen_q_tensor",
        cum_seqlen_q_tensor,
        rank=1,
        dtype=(jnp.int32, jnp.int64),
    )
    cum_k_shape, _ = require_array(
        "cum_seqlen_k_tensor",
        cum_seqlen_k_tensor,
        rank=1,
        dtype=offsets_dtype,
    )
    if cum_q_shape != cum_k_shape or cum_q_shape[0] < 2:
        raise ValueError("cum_seqlen_q_tensor and cum_seqlen_k_tensor must have the same " f"shape (B + 1,) with B > 0, got {cum_q_shape} and {cum_k_shape}")

    if max_s_q <= 0 or max_s_k <= 0:
        raise ValueError(f"max_s_q and max_s_k must be positive, got {max_s_q} and {max_s_k}")
    if max_s_q != max_s_k:
        raise ValueError(f"max_s_q and max_s_k must be identical, got {max_s_q} and {max_s_k}")
    if block_size not in (16, 32, 64):
        raise ValueError(f"block_size must be 16, 32, or 64, got {block_size}")

    output_dtype = require_dtype(
        "o_dtype",
        o_dtype,
        (input_dtype,),
        default=input_dtype,
    )
    require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    resolved_scale = 1.0 / math.sqrt(head_dim) if scale_softmax is None else float(scale_softmax)
    if _validate_only:
        return None

    # The native kernel consumes the same singleton-batch views produced by
    # Torch's unsqueeze(0). These reshapes are storage-preserving XLA bitcasts.
    q_storage = jnp.reshape(q_tensor, (1, total_tokens, num_query_heads, head_dim))
    k_storage = jnp.reshape(k_tensor, (1, total_tokens, num_kv_heads, head_dim))
    v_storage = jnp.reshape(v_tensor, (1, total_tokens, num_kv_heads, value_dim))

    o_storage, l_storage, m_storage = call_cutedsl(
        _launch,
        (
            q_storage,
            k_storage,
            v_storage,
            block_indices_tensor,
            block_counts_tensor,
            cum_seqlen_q_tensor,
            cum_seqlen_k_tensor,
        ),
        outputs=(
            BufferSpec(
                "o_tensor",
                (1, total_tokens, num_query_heads, value_dim),
                output_dtype,
                fill_value=0,
            ),
            BufferSpec(
                "l_tensor",
                (1, total_tokens, num_query_heads),
                jnp.float32,
                fill_value=0.0,
            ),
            BufferSpec(
                "m_tensor",
                (1, total_tokens, num_query_heads),
                jnp.float32,
                fill_value=float("-inf"),
            ),
        ),
        static_args={
            "element_dtype": jax_to_cutlass_dtype(input_dtype),
            "head_dim": head_dim,
            "value_dim": value_dim,
            "gqa_group_size": gqa_group_size,
            "block_size": block_size,
            "max_s_q": max_s_q,
            "scale_softmax": resolved_scale,
        },
        use_static_tensors=True,
    )
    return TupleDict(
        o_tensor=jnp.reshape(o_storage, (total_tokens, num_query_heads, value_dim)),
        l_tensor=jnp.reshape(l_storage, (total_tokens, num_query_heads, 1)),
        m_tensor=jnp.reshape(m_storage, (total_tokens, num_query_heads, 1)),
    )


class SelectionAttention(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM90 selection attention."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_block_indices: Any,
        sample_block_counts: Any,
        sample_cum_seqlen_q: Any,
        sample_cum_seqlen_k: Any,
        *,
        max_s_q: int,
        max_s_k: int,
        block_size: int = 64,
        scale_softmax: float | None = None,
        o_dtype: Any = None,
        acc_dtype: Any = None,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self.make_tensor_desc(sample_v, name="sample_v")
        self.block_indices_desc = self.make_tensor_desc(sample_block_indices, name="sample_block_indices")
        self.block_counts_desc = self.make_tensor_desc(sample_block_counts, name="sample_block_counts")
        self.cum_q_desc = self.make_tensor_desc(sample_cum_seqlen_q, name="sample_cum_seqlen_q")
        self.cum_k_desc = self.make_tensor_desc(sample_cum_seqlen_k, name="sample_cum_seqlen_k")
        self.max_s_q = max_s_q
        self.max_s_k = max_s_k
        self.block_size = block_size
        self.scale_softmax = scale_softmax
        self.o_dtype = self.as_optional_dtype(o_dtype)
        self.acc_dtype = self.as_optional_dtype(acc_dtype)

    def _check_support(self) -> bool:
        _selection_attention_impl(
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.block_indices_desc,
            self.block_counts_desc,
            self.cum_q_desc,
            self.cum_k_desc,
            max_s_q=self.max_s_q,
            max_s_k=self.max_s_k,
            block_size=self.block_size,
            scale_softmax=self.scale_softmax,
            o_dtype=self.o_dtype,
            acc_dtype=self.acc_dtype,
            _validate_only=True,
        )
        return True

    def __call__(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
        block_indices_tensor: Any,
        block_counts_tensor: Any,
        cum_seqlen_q_tensor: Any,
        cum_seqlen_k_tensor: Any,
    ) -> TupleDict:
        return super().__call__(
            q_tensor,
            k_tensor,
            v_tensor,
            block_indices_tensor,
            block_counts_tensor,
            cum_seqlen_q_tensor,
            cum_seqlen_k_tensor,
        )

    def _call_impl(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
        block_indices_tensor: Any,
        block_counts_tensor: Any,
        cum_seqlen_q_tensor: Any,
        cum_seqlen_k_tensor: Any,
    ) -> TupleDict:
        for value, expected, name in (
            (q_tensor, self.q_desc, "Q"),
            (k_tensor, self.k_desc, "K"),
            (v_tensor, self.v_desc, "V"),
            (block_indices_tensor, self.block_indices_desc, "block_indices"),
            (block_counts_tensor, self.block_counts_desc, "block_counts"),
            (cum_seqlen_q_tensor, self.cum_q_desc, "cum_seqlen_q"),
            (cum_seqlen_k_tensor, self.cum_k_desc, "cum_seqlen_k"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        return _selection_attention_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            block_indices_tensor,
            block_counts_tensor,
            cum_seqlen_q_tensor,
            cum_seqlen_k_tensor,
            max_s_q=self.max_s_q,
            max_s_k=self.max_s_k,
            block_size=self.block_size,
            scale_softmax=self.scale_softmax,
            o_dtype=self.o_dtype,
            acc_dtype=self.acc_dtype,
        )


def selection_attention_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    block_indices_tensor: Any,
    block_counts_tensor: Any,
    cum_seqlen_q_tensor: Any,
    cum_seqlen_k_tensor: Any,
    *,
    max_s_q: int,
    max_s_k: int,
    block_size: int = 64,
    scale_softmax: float | None = None,
    o_dtype: Any = None,
    acc_dtype: Any = None,
) -> TupleDict:
    """Compute packed THD selection attention with the SM90 CuTe kernel."""

    return SelectionAttention(
        q_tensor,
        k_tensor,
        v_tensor,
        block_indices_tensor,
        block_counts_tensor,
        cum_seqlen_q_tensor,
        cum_seqlen_k_tensor,
        max_s_q=max_s_q,
        max_s_k=max_s_k,
        block_size=block_size,
        scale_softmax=scale_softmax,
        o_dtype=o_dtype,
        acc_dtype=acc_dtype,
    )(
        q_tensor,
        k_tensor,
        v_tensor,
        block_indices_tensor,
        block_counts_tensor,
        cum_seqlen_q_tensor,
        cum_seqlen_k_tensor,
    )


__all__ = [
    "SelectionAttention",
    "selection_attention_wrapper",
]
