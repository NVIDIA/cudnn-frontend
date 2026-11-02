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
def _indexer_adapter(stream, q, k, weights, *buffers, kernel, is_varlen, sm_scale, seqlen_q, seqlen_k):
    # The bridge accepts arrays only: dense calls omit the two prefix operands.
    if cutlass.const_expr(is_varlen):
        cu_q, cu_k, scores, denom = buffers
    else:
        scores, denom = buffers
        cu_q, cu_k = None, None
    kernel(q, k, weights, scores, denom, cutlass.Float32(sm_scale), cutlass.Int32(seqlen_q), cutlass.Int32(seqlen_k), cu_q, cu_k, None, stream)


def indexer_forward_jax_sm100(
    q: Any,
    k: Any,
    weights: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
    cu_seqlens_q: Any = None,
    cu_seqlens_k: Any = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Any:
    """Dense ratio-causal scores for BSHD or contiguous packed THD inputs.

    THD takes Q [Tq,64,128], K [Tk,1,128], weights [Tq,64], matching INT32
    prefixes and static maximum lengths. Prefixes start at zero, are monotonic,
    and end at valid rows within the tensor capacities; callers own these values.
    Each segment's timeline starts at zero. max_seqlen_k must cover the partial
    final query block as well as the actual key lengths. Returns [Tq,ceil4(maxK)]
    FP32 scores indexed by segment-local key position, with invalid entries -inf.
    """
    if q.ndim not in (3, 4):
        raise ValueError("q must use BSHD or THD layout")
    is_varlen = q.ndim == 3
    heads, head_dim = q.shape[-2:]
    if head_dim != 128 or heads != 64 or ratio != 4:
        raise ValueError("DSv4 JAX indexer requires H=64, D=128, ratio=4")
    if is_varlen:
        if k.ndim != 3 or k.shape[1:] != (1, head_dim) or weights.shape != q.shape[:2]:
            raise ValueError("packed k/weights must have shapes [Tk,1,D]/[Tq,H]")
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("THD requires both sequence prefixes")
        if (
            cu_seqlens_q.ndim != 1
            or cu_seqlens_q.size < 2
            or cu_seqlens_k.shape != cu_seqlens_q.shape
            or cu_seqlens_q.dtype != jnp.int32
            or cu_seqlens_k.dtype != jnp.int32
        ):
            raise ValueError("sequence prefixes must be matching 1-D INT32 arrays")
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (max_seqlen_q, max_seqlen_k)):
            raise ValueError("THD requires positive static maximum sequence lengths")
        if max_seqlen_k < (max_seqlen_q + ratio - 1) // ratio:
            raise ValueError("max_seqlen_k must cover ceil(max_seqlen_q / ratio)")
        if q.shape[0] == 0 or k.shape[0] == 0:
            raise ValueError("THD requires nonempty Q/K storage capacities")
        seqlen_q, seqlen_k = max_seqlen_q, (max_seqlen_k + 3) // 4 * 4
        score_shape, denom_shape = (q.shape[0], seqlen_k), (q.shape[0],)
        operands = (q, k, weights, cu_seqlens_q, cu_seqlens_k)
    else:
        if any(x is not None for x in (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)):
            raise ValueError("sequence prefixes and maximum lengths are only accepted for THD")
        batch, seqlen_q = q.shape[:2]
        if k.ndim != 4 or k.shape[0] != batch or k.shape[2:] != (1, head_dim):
            raise ValueError(f"k must have shape ({batch}, S_k, 1, {head_dim}), got {k.shape}")
        if weights.shape != (batch, seqlen_q, heads):
            raise ValueError(f"weights must have shape {(batch, seqlen_q, heads)}, got {weights.shape}")
        seqlen_k = k.shape[1]
        if seqlen_k % 4:
            raise ValueError("indexer K length must be a multiple of four for FP32 TMA stores")
        score_shape, denom_shape = (batch, seqlen_q, seqlen_k), (batch, seqlen_q)
        operands = (q, k, weights)
    if q.dtype != jnp.bfloat16 or k.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("q, k, and weights must be bfloat16")

    key = (head_dim, heads, ratio, is_varlen)
    kernel = _indexer_kernels.get(key)
    if kernel is None:
        kernel = IndexerScoreUnifiedSm100(
            head_dim=head_dim,
            qhead_per_kvhead=heads,
            m_block_size=128,
            n_block_size=128,
            k_block_size=64,
            kv_stage=4,
            ratio=ratio,
            is_varlen=is_varlen,
            compute_lse=False,
            is_compressed_logits=False,
        )
        _indexer_kernels[key] = kernel
    scores, _ = call(
        _indexer_adapter,
        output_shape_dtype=(jax.ShapeDtypeStruct(score_shape, jnp.float32), jax.ShapeDtypeStruct(denom_shape, jnp.float32)),
        initialized_outputs={0: neg_inf_init, 1: zeros_init},
        kernel=kernel,
        is_varlen=is_varlen,
        sm_scale=float(sm_scale),
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )(*operands)
    return scores


@cute.jit
def _sparse_fwd_adapter(stream, q, kv, indices, topk_length, attn_sink, out, max_logits, lse, lse_indexer, *, kernel, softmax_scale):
    kernel(
        q,
        kv,
        indices,
        out,
        max_logits,
        lse,
        lse_indexer,
        attn_sink,
        topk_length,
        cutlass.Float32(softmax_scale),
        stream,
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
    # indexer_topk=0 disables the indexer-prefix LSE (e.g. HCA, whose index list is
    # a fixed local window plus causal compressed blocks); lse_indexer is then unwritten.
    if indexer_topk not in (0, 512, 1024, 2048) or indexer_topk > topk_indices.shape[1]:
        raise ValueError("indexer_topk must be 0 or a supported prefix no wider than K")

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
        kernel=kernel,
        softmax_scale=scale,
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
        problem_shape,
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
        cutlass.Float32(softmax_scale),
        stream,
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
            element_dtype=cutlass.BFloat16,
            head_dim=512,
            head_dim_v=512,
            block_tile=64,
            max_topk=max_topk,
        )
        _sparse_bwd_kernels[key] = kernel
    workspace_lse_shape = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(total_q, 512, 64, 1, cutlass.Float32)
    workspace_dkv_shape = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(total_kv, 512, 1, cutlass.Float32)
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
        kernel=kernel,
        softmax_scale=scale,
        total_q=total_q,
        total_kv=total_kv,
    )(q, kv, out, dout, lse, attn_sink, topk_indices, topk_length)
    return results[:3]


__all__ = [
    "indexer_forward_jax_sm100",
    "sparse_attention_forward_jax_sm100",
    "sparse_attention_backward_jax_sm100",
]
