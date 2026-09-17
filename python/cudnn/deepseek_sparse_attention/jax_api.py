# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX custom-call entry points for the DSv4 SM100 indexer and sparse attention."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute

from cudnn.jax import call, neg_inf_init, zeros_init
from .score_recompute.indexer_score_unified_sm100 import IndexerScoreUnifiedSm100
from .sparse_attention_forward.dsa_fwd_sm100_head64 import SparseAttentionForwardSm100Head64
from .sparse_attention_backward.dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

_indexer_kernels: dict[tuple, Any] = {}
_sparse_fwd_kernels: dict[tuple, Any] = {}
_sparse_bwd_kernels: dict[tuple, Any] = {}


@cute.jit
def _indexer_adapter(stream, q, k, weights, scores, denom, *, kernel, sm_scale, seqlen_q, seqlen_k):
    kernel(
        q, k, weights, scores, denom,
        cutlass.Float32(sm_scale), cutlass.Int32(seqlen_q), cutlass.Int32(seqlen_k),
        None, None, None, stream,
    )


def indexer_forward_jax_sm100(q: Any, k: Any, weights: Any, *, ratio: int = 4, sm_scale: float = 1.0) -> Any:
    """Compute dense ratio-causal DSv4 indexer scores using CuTeDSL."""
    if q.ndim != 4 or k.ndim != 4 or weights.ndim != 3:
        raise ValueError("q/k/weights must use BSHD/BSHD/BSH layouts")
    batch, seqlen_q, heads, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (1, head_dim):
        raise ValueError(f"k must have shape ({batch}, S_k, 1, {head_dim}), got {k.shape}")
    seqlen_k = k.shape[1]
    if weights.shape != (batch, seqlen_q, heads):
        raise ValueError(f"weights must have shape {(batch, seqlen_q, heads)}, got {weights.shape}")
    if q.dtype != jnp.bfloat16 or k.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("q, k, and weights must be bfloat16")
    if head_dim != 128 or heads != 64 or ratio != 4:
        raise ValueError("DSv4 JAX indexer requires H=64, D=128, ratio=4")
    if seqlen_k % 4:
        raise ValueError("indexer K length must be a multiple of four for FP32 TMA stores")

    key = (head_dim, heads, ratio)
    kernel = _indexer_kernels.get(key)
    if kernel is None:
        kernel = IndexerScoreUnifiedSm100(
            head_dim=head_dim, qhead_per_kvhead=heads, m_block_size=128,
            n_block_size=128, k_block_size=64, kv_stage=4, ratio=ratio,
            is_varlen=False, compute_lse=False, is_compressed_logits=False,
        )
        _indexer_kernels[key] = kernel

    scores, _ = call(
        _indexer_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((batch, seqlen_q, seqlen_k), jnp.float32),
            jax.ShapeDtypeStruct((batch, seqlen_q), jnp.float32),
        ),
        initialized_outputs={0: neg_inf_init, 1: zeros_init},
        kernel=kernel, sm_scale=float(sm_scale), seqlen_q=seqlen_q, seqlen_k=seqlen_k,
    )(q, k, weights)
    return scores


@cute.jit
def _sparse_fwd_adapter(
    stream, q, kv, indices, topk_length, attn_sink, out, max_logits, lse, lse_indexer, *, kernel, softmax_scale
):
    kernel(
        q, kv, indices, out, max_logits, lse, lse_indexer, attn_sink, topk_length,
        cutlass.Float32(softmax_scale), stream,
    )


def sparse_attention_forward_jax_sm100(
    q: Any,
    kv: Any,
    topk_indices: Any,
    topk_length: Any,
    attn_sink: Any,
    *,
    softmax_scale: float | None = None,
    indexer_topk: int = 512,
) -> tuple[Any, Any, Any, Any]:
    """Run DSv4 sparse attention forward on flat global index lists."""
    if q.ndim != 3 or q.shape[1:] != (64, 512) or q.dtype != jnp.bfloat16:
        raise ValueError("q must have shape (Tq, 64, 512) and dtype bfloat16")
    if kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype:
        raise ValueError("kv must have shape (Tkv, 512) with q's dtype")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != q.shape[0] or topk_indices.dtype != jnp.int32:
        raise ValueError("topk_indices must be (Tq, K) int32")
    if topk_indices.shape[1] % 64:
        raise ValueError("physical sparse-attention K must be a multiple of 64")
    if topk_length.shape != (q.shape[0],) or topk_length.dtype != jnp.int32:
        raise ValueError("topk_length must be (Tq,) int32")
    if attn_sink.shape != (64,) or attn_sink.dtype != jnp.float32:
        raise ValueError("attn_sink must be (64,) float32")
    if indexer_topk not in (512, 1024, 2048) or indexer_topk > topk_indices.shape[1]:
        raise ValueError("indexer_topk must be a supported prefix no wider than K")

    kernel_key = (512, int(indexer_topk))
    kernel = _sparse_fwd_kernels.get(kernel_key)
    if kernel is None:
        kernel = SparseAttentionForwardSm100Head64(head_dim=512, indexer_topk=indexer_topk)
        _sparse_fwd_kernels[kernel_key] = kernel
    scale = 1.0 / math.sqrt(512) if softmax_scale is None else float(softmax_scale)
    tq = q.shape[0]
    return call(
        _sparse_fwd_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((tq, 64, 512), q.dtype),
            jax.ShapeDtypeStruct((tq, 64), jnp.float32),
            jax.ShapeDtypeStruct((tq, 64), jnp.float32),
            jax.ShapeDtypeStruct((tq, 64), jnp.float32),
        ),
        kernel=kernel, softmax_scale=scale,
    )(q, kv, topk_indices, topk_length, attn_sink)


@cute.jit
def _sparse_bwd_adapter(
    stream,
    q,
    kv,
    out,
    dout,
    lse,
    attn_sink,
    indices,
    topk_length,
    dq,
    dkv,
    dsink,
    workspace_lse_odo,
    workspace_dkv,
    *,
    kernel,
    softmax_scale,
    total_q,
    total_kv,
):
    problem_shape = (
        cutlass.Int32(total_q),
        cutlass.Int32(total_kv),
        cutlass.Int32(512),
        (cutlass.Int32(64), cutlass.Int32(1)),
    )
    kernel(
        problem_shape, q, kv, out, dout, lse, attn_sink, indices, topk_length,
        dq, dkv, dsink, workspace_lse_odo, workspace_dkv,
        cutlass.Float32(softmax_scale), stream,
    )


def sparse_attention_backward_jax_sm100(
    q: Any,
    kv: Any,
    out: Any,
    dout: Any,
    lse: Any,
    attn_sink: Any,
    topk_indices: Any,
    topk_length: Any,
    *,
    softmax_scale: float | None = None,
) -> tuple[Any, Any, Any]:
    """Explicit JAX custom call for the generic H64/D512 sparse backward."""
    if q.shape[1:] != (64, 512) or q.dtype != jnp.bfloat16:
        raise ValueError("q must be (Tq, 64, 512) bfloat16")
    if kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype:
        raise ValueError("kv must be (Tkv, 512) bfloat16")
    if out.shape != q.shape or dout.shape != q.shape or out.dtype != q.dtype or dout.dtype != q.dtype:
        raise ValueError("out and dout must match q")
    if lse.shape != q.shape[:2] or lse.dtype != jnp.float32:
        raise ValueError("lse must be (Tq, 64) float32")
    if attn_sink.shape != (64,) or attn_sink.dtype != jnp.float32:
        raise ValueError("attn_sink must be (64,) float32")
    if topk_indices.shape[0] != q.shape[0] or topk_indices.dtype != jnp.int32:
        raise ValueError("topk_indices must be (Tq, K) int32")
    if topk_length.shape != (q.shape[0],) or topk_length.dtype != jnp.int32:
        raise ValueError("topk_length must be (Tq,) int32")

    total_q, total_kv, max_topk = q.shape[0], kv.shape[0], topk_indices.shape[1]
    key = (max_topk,)
    kernel = _sparse_bwd_kernels.get(key)
    if kernel is None:
        kernel = FlashAttentionDSABackwardSm100(
            element_dtype=cutlass.BFloat16, head_dim=512, head_dim_v=512,
            block_tile=64, max_topk=max_topk,
        )
        _sparse_bwd_kernels[key] = kernel
    workspace_lse_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
        total_q, 512, 64, 1, cutlass.Float32
    )
    workspace_dkv_shape = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(
        total_kv, 512, 1, cutlass.Float32
    )
    scale = 1.0 / math.sqrt(512) if softmax_scale is None else float(softmax_scale)
    results = call(
        _sparse_bwd_adapter,
        output_shape_dtype=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(kv.shape, kv.dtype),
            jax.ShapeDtypeStruct((64,), jnp.float32),
            jax.ShapeDtypeStruct(workspace_lse_shape, jnp.uint8),
            jax.ShapeDtypeStruct(workspace_dkv_shape, jnp.uint8),
        ),
        initialized_outputs={0: zeros_init, 1: zeros_init, 2: zeros_init, 3: zeros_init, 4: zeros_init},
        kernel=kernel, softmax_scale=scale, total_q=total_q, total_kv=total_kv,
    )(q, kv, out, dout, lse, attn_sink, topk_indices, topk_length)
    return results[:3]


__all__ = [
    "indexer_forward_jax_sm100",
    "sparse_attention_forward_jax_sm100",
    "sparse_attention_backward_jax_sm100",
]
