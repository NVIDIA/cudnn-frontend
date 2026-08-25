# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Score Recompute Interface (SM100 Cute-DSL wrapper).

Orchestrates compilation, caching, and kernel dispatch for:
  - sparse_indexer_score_recompute: sparse index scores + softmax (predict)
  - sparse_attn_score_recompute: sparse attention scores + L1-norm (target)
  - dense_indexer_score_recompute: dense index scores + LSE denom
  - dense_attn_score_recompute: dense attention scores + L1-norm denom
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100
from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100
from .dense_score_recompute_sm100_mxfp8 import BwdDenseAttnScoreSm100Mxfp8
from .indexer_score_unified_sm100 import IndexerScoreUnifiedSm100
from .indexer_score_unified_sm100_mxfp8 import IndexerScoreUnifiedSm100Mxfp8
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    ceil_div as _ceil_div,
    device_major as _get_device_capability,
    maybe_contiguous,
    resolve_stream as _resolve_stream,
    torch_stream_context as _torch_stream_context,
    validate_q_causal_offsets,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _normalize_dense_precision(
    precision: str,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
) -> str:
    precision = precision.lower()
    if precision not in ("bf16", "mxfp8"):
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")
    if precision == "bf16" and (q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None):
        raise ValueError("MXFP8 scales and scale padded cu_seqlens require precision='mxfp8'")
    return precision


def _validate_dense_mxfp8_contract(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    per_head: torch.Tensor,
    per_head_name: str,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    cu_seqlens_q_scale_padded: Optional[torch.Tensor],
    cu_seqlens_k_scale_padded: Optional[torch.Tensor],
    qhead_per_kv_head: int,
    head_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    bs: int,
    n_heads_q: int,
    n_heads_kv: int,
    sf_vec_size: int,
    is_varlen: bool,
) -> None:
    if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
        raise TypeError("precision='mxfp8' requires q and k to be torch.float8_e4m3fn")
    if per_head_name == "weights" and per_head.dtype != torch.bfloat16:
        raise TypeError("precision='mxfp8' requires weights to be torch.bfloat16")
    if per_head_name == "lse" and per_head.dtype != torch.float32:
        raise TypeError("precision='mxfp8' requires lse to be torch.float32")
    if q_scale is None or k_scale is None:
        raise ValueError("precision='mxfp8' requires q_scale and k_scale")
    if q_scale.dtype != torch.float8_e8m0fnu or k_scale.dtype != torch.float8_e8m0fnu:
        raise TypeError("precision='mxfp8' requires q_scale and k_scale to be " "torch.float8_e8m0fnu")
    for scale, reference, name in ((q_scale, q, "q_scale"), (k_scale, k, "k_scale")):
        if scale.device != reference.device:
            raise ValueError(f"{name} must be on the same device as its data tensor")
    if sf_vec_size != 32:
        raise ValueError("precision='mxfp8' currently requires sf_vec_size=32")
    if head_dim != 128 and not (per_head_name == "lse" and head_dim == 512):
        raise ValueError("precision='mxfp8' currently supports head_dim=128, " "or attention head_dim=512")
    if n_heads_kv != 1:
        raise ValueError("precision='mxfp8' dense score currently requires n_heads_kv=1")
    if n_heads_q != n_heads_kv * qhead_per_kv_head:
        raise ValueError("n_heads_q must equal n_heads_kv * qhead_per_kv_head for " "precision='mxfp8'")

    sf_groups = _ceil_div(head_dim, sf_vec_size)
    sf_groups_padded = _ceil_div(sf_groups, 4) * 4
    if is_varlen:
        if cu_seqlens_q_scale_padded is None or cu_seqlens_k_scale_padded is None:
            raise ValueError("THD MXFP8 requires Q/K scale padded cu_seqlens")
        for prefix, name in (
            (cu_seqlens_q_scale_padded, "cu_seqlens_q_scale_padded"),
            (cu_seqlens_k_scale_padded, "cu_seqlens_k_scale_padded"),
        ):
            if prefix.dtype != torch.int32 or prefix.ndim != 1 or prefix.shape[0] != bs + 1 or not prefix.is_contiguous() or prefix.device != q.device:
                raise ValueError(f"{name} must be contiguous int32 shape ({bs + 1},) on {q.device}")
        for scale, name in ((q_scale, "q_scale"), (k_scale, "k_scale")):
            if scale.ndim != 3 or scale.shape[0] != n_heads_kv or scale.shape[1] % 128 != 0 or scale.shape[2] != sf_groups_padded:
                raise ValueError(f"THD {name} must have shape ({n_heads_kv}, multiple_of_128, " f"{sf_groups_padded}), got {tuple(scale.shape)}")
    else:
        if cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
            raise ValueError("BSHD MXFP8 requires scale padded cu_seqlens to be None")
        q_mn = seqlen_q * qhead_per_kv_head
        q_shape = (
            bs * n_heads_kv,
            _ceil_div(q_mn, 128) * 128,
            sf_groups_padded,
        )
        k_shape = (
            bs * n_heads_kv,
            _ceil_div(seqlen_k, 128) * 128,
            sf_groups_padded,
        )
        if tuple(q_scale.shape) != q_shape:
            raise ValueError(f"q_scale packed shape must be {q_shape}, got {tuple(q_scale.shape)}")
        if tuple(k_scale.shape) != k_shape:
            raise ValueError(f"k_scale packed shape must be {k_shape}, got {tuple(k_scale.shape)}")


def _sparse_indexer_score_recompute(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 128,
    sm_scale: float = 1.0,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Compute sparse index scores + softmax (predict) for backward pass.

    Args:
        q_indexer: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k_indexer: (bs, seqlen_k, head_dim) [BF16] — no head dim (MQA)
        weights: (bs, seqlen_q, n_heads_q) [BF16]
        topk_indices: (bs, seqlen_q, topk) [INT32]
        qhead_per_kv_head: auto inferred if None
        out: optional output tensor (bs, seqlen_q, topk) [FP32]
        topk_length: (bs, seqlen_q) [INT32] — per-q valid topk count (compact layout), optional
        m_block_size: defaults to qhead_per_kv_head (must equal qhead_per_kv_head)

    Returns:
        predict: (bs, seqlen_q, topk) [FP32] — softmax of sparse index scores
    """
    current_stream = _resolve_stream(current_stream)
    q_indexer, k_indexer, weights = [maybe_contiguous(t, current_stream) for t in (q_indexer, k_indexer, weights)]
    with _torch_stream_context(current_stream):
        topk_indices = topk_indices.to(torch.int32).contiguous()

    if qhead_per_kv_head is None:
        qhead_per_kv_head = q_indexer.shape[2]

    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    bs, seqlen_q, n_heads_q, head_dim = q_indexer.shape
    _, seqlen_k, _ = k_indexer.shape
    topk = topk_indices.shape[2]

    device = q_indexer.device
    have_topk_length = topk_length is not None

    if topk_length is None:
        with _torch_stream_context(current_stream):
            topk_length = torch.empty((1, 1), dtype=torch.int32, device=device)
    else:
        with _torch_stream_context(current_stream):
            topk_length = topk_length.to(torch.int32).contiguous()

    if out is None:
        with _torch_stream_context(current_stream):
            out = torch.empty((bs, seqlen_q, topk), dtype=torch.float32, device=device)

    # Compute kv_stage and topk_in_smem from SMEM budget (SM100: 228 KB)
    SM100_SMEM_BYTES = 228 * 1024
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    sK_per_stage = n_block_size * head_dim_padded * 2  # BF16
    sQ_size = m_block_size * head_dim_padded * 2  # BF16
    sTopkIdx_bytes = topk * 2 * 4  # double-buffer, INT32
    sPerHead_bytes = m_block_size * 2 * 2  # double-buffer, BF16
    smem_fixed = sPerHead_bytes + 2048  # barriers, sScoreAll, alignment

    topk_in_smem = True
    smem_overhead = sTopkIdx_bytes + smem_fixed
    kv_stage = min(4, max(1, (SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
    total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
    if total_smem_est > SM100_SMEM_BYTES:
        topk_in_smem = False
        smem_overhead = smem_fixed
        kv_stage = min(4, max(1, (SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
        total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
        assert total_smem_est <= SM100_SMEM_BYTES, (
            f"SMEM overflow ({total_smem_est} > {SM100_SMEM_BYTES}) even without sTopkIdx: "
            f"topk={topk}, head_dim={head_dim}(padded={head_dim_padded}), "
            f"m_block={m_block_size}, n_block={n_block_size}, kv_stage={kv_stage}."
        )

    compute_capability = _get_device_capability()

    if compute_capability >= 10:
        # sm_scale is a runtime arg (passed as cutlass.Float32), so it is
        # excluded from compile_key.
        # topk_indices_global bakes in (controls compile-time global→local
        # decode of topk ids).
        compile_key = (
            "indexer",
            q_indexer.dtype,
            head_dim,
            qhead_per_kv_head,
            topk,
            m_block_size,
            n_block_size,
            kv_stage,
            have_topk_length,
            topk_in_smem,
            topk_indices_global,
        )

        if compile_key not in _sparse_indexer_score_recompute.compile_cache:
            q_cute = to_cute_tensor(q_indexer)
            k_cute = to_cute_tensor(k_indexer)
            w_cute = to_cute_tensor(weights)
            topk_cute = to_cute_tensor(topk_indices)
            out_cute = to_cute_tensor(out)
            topk_length_cute = to_cute_tensor(topk_length)

            kernel_obj = SparseScoreRecomputeSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                topk=topk,
                kv_stage=kv_stage,
                score_type="indexer",
                have_topk_length=have_topk_length,
                topk_in_smem=topk_in_smem,
                topk_indices_global=topk_indices_global,
            )

            scale_arg = cutlass.Float32(sm_scale)

            _sparse_indexer_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                topk_cute,
                out_cute,
                topk_length_cute,
                scale_arg,
                current_stream,
                options=compile_options(),
            )

        scale_arg = cutlass.Float32(sm_scale)
        with torch.cuda.nvtx.range("sparse_indexer_score_recompute"):
            _sparse_indexer_score_recompute.compile_cache[compile_key](
                q_indexer,
                k_indexer,
                weights,
                topk_indices,
                out,
                topk_length,
                scale_arg,
                current_stream,
            )
        return out

    raise NotImplementedError(f"Sparse indexer backward score requires SM100+ (got compute capability {compute_capability}).")


_sparse_indexer_score_recompute.compile_cache = {}


def sparse_indexer_score_recompute(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Public entry point for sparse indexer backward predict computation.

    Computes sparse scores and applies softmax:
      S[b,q,i] = sm_scale * sum_h [ReLU(Q_h · K_{topk[b,q,i]}^T) · W_{b,q,h}]
      predict = softmax(S, dim=-1)  (invalid positions masked as -inf)

    sm_scale is applied to the fp32 head-reduced score inside the kernel
    (preserves precision vs pre-multiplying onto bf16 weights on the host).

    Args:
        q_indexer: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k_indexer: (bs, seqlen_k, head_dim) [BF16]
        weights: (bs, seqlen_q, n_heads_q) [BF16]
        topk_indices: (bs, seqlen_q, topk) [INT32]
        qhead_per_kv_head: auto inferred if None
        out: pre-allocated output (bs, seqlen_q, topk) [FP32], optional
        topk_length: (bs, seqlen_q) [INT32] — compact layout, optional
        sm_scale: scalar applied to fp32 score post head-reduce; default 1.0
        topk_indices_global: when True (default, matches public fwd output),
            ``topk_indices`` are global ids ``b * seqlen_k + local`` and the
            kernel decodes back to local. When False, ids are already local.

    Returns:
        predict: (bs, seqlen_q, topk) [FP32]
    """
    return _sparse_indexer_score_recompute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        qhead_per_kv_head,
        out,
        topk_length,
        sm_scale=sm_scale,
        topk_indices_global=topk_indices_global,
        current_stream=current_stream,
    )


# =============================================================================
# Sparse attention score: exp(QK*scale - LSE) -> head reduce -> L1-norm
# =============================================================================


def _sparse_attn_score_recompute(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 64,
    k_block_size: Optional[int] = None,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Compute sparse attention target (L1-normalized head-summed softmax) for backward.

    Args:
        q_attn: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k_attn: (bs, seqlen_k, head_dim) [BF16]
        lse: (bs, seqlen_q, n_heads_q) [FP32] — logsumexp from forward softmax
        topk_indices: (bs, seqlen_q, topk) [INT32]
        softmax_scale: float — attention softmax scale
        qhead_per_kv_head: auto inferred if None
        out: optional output tensor (bs, seqlen_q, topk) [FP32]
        topk_length: (bs, seqlen_q) [INT32] — per-q valid topk count (compact layout), optional
        m_block_size: defaults to qhead_per_kv_head (must equal qhead_per_kv_head)

    Returns:
        target: (bs, seqlen_q, topk) [FP32] — L1-normalized attention scores
    """

    current_stream = _resolve_stream(current_stream)
    q_attn, k_attn = [maybe_contiguous(t, current_stream) for t in (q_attn, k_attn)]
    with _torch_stream_context(current_stream):
        topk_indices = topk_indices.to(torch.int32).contiguous()

    if qhead_per_kv_head is None:
        qhead_per_kv_head = q_attn.shape[2]

    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    bs, seqlen_q, n_heads_q, head_dim = q_attn.shape
    _, seqlen_k, _ = k_attn.shape
    topk = topk_indices.shape[2]

    device = q_attn.device
    have_topk_length = topk_length is not None

    if topk_length is None:
        with _torch_stream_context(current_stream):
            topk_length = torch.empty((1, 1), dtype=torch.int32, device=device)
    else:
        with _torch_stream_context(current_stream):
            topk_length = topk_length.to(torch.int32).contiguous()

    lse = maybe_contiguous(lse, current_stream)

    if out is None:
        with _torch_stream_context(current_stream):
            out = torch.empty((bs, seqlen_q, topk), dtype=torch.float32, device=device)

    # Compute kv_stage and topk_in_smem from SMEM budget (SM100: 228 KB)
    SM100_SMEM_BYTES = 228 * 1024
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    k_block_size_eff = k_block_size if k_block_size is not None else head_dim_padded
    sK_per_stage = n_block_size * k_block_size_eff * 2  # BF16
    sQ_size = m_block_size * head_dim_padded * 2  # BF16 (Q total unchanged)
    sTopkIdx_bytes = topk * 2 * 4  # double-buffer, INT32
    sPerHead_bytes = m_block_size * 2 * 4  # double-buffer, FP32
    smem_fixed = sPerHead_bytes + 2048  # barriers, sScoreAll, alignment

    topk_in_smem = True
    smem_overhead = sTopkIdx_bytes + smem_fixed
    kv_stage = min(4, max(1, (SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
    total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
    if total_smem_est > SM100_SMEM_BYTES:
        topk_in_smem = False
        smem_overhead = smem_fixed
        kv_stage = min(4, max(1, (SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))
        total_smem_est = sQ_size + sK_per_stage * kv_stage + smem_overhead
        assert total_smem_est <= SM100_SMEM_BYTES, (
            f"SMEM overflow ({total_smem_est} > {SM100_SMEM_BYTES}) even without sTopkIdx: "
            f"topk={topk}, head_dim={head_dim}(padded={head_dim_padded}), "
            f"m_block={m_block_size}, n_block={n_block_size}, kv_stage={kv_stage}."
        )

    compute_capability = _get_device_capability()

    if compute_capability >= 10:
        compile_key = (
            "attention",
            q_attn.dtype,
            head_dim,
            qhead_per_kv_head,
            topk,
            m_block_size,
            n_block_size,
            k_block_size,
            kv_stage,
            have_topk_length,
            topk_in_smem,
            topk_indices_global,
        )

        if compile_key not in _sparse_attn_score_recompute.compile_cache:
            q_cute = to_cute_tensor(q_attn)
            k_cute = to_cute_tensor(k_attn)
            lse_cute = to_cute_tensor(lse)
            topk_cute = to_cute_tensor(topk_indices)
            out_cute = to_cute_tensor(out)
            topk_length_cute = to_cute_tensor(topk_length)

            kernel_obj = SparseScoreRecomputeSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                topk=topk,
                kv_stage=kv_stage,
                score_type="attention",
                have_topk_length=have_topk_length,
                topk_in_smem=topk_in_smem,
                topk_indices_global=topk_indices_global,
            )

            _sparse_attn_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                lse_cute,
                topk_cute,
                out_cute,
                topk_length_cute,
                cutlass.Float32(softmax_scale),
                current_stream,
                options=compile_options(),
            )

        with torch.cuda.nvtx.range("sparse_attn_score_recompute"):
            _sparse_attn_score_recompute.compile_cache[compile_key](
                q_attn,
                k_attn,
                lse,
                topk_indices,
                out,
                topk_length,
                cutlass.Float32(softmax_scale),
                current_stream,
            )
        return out

    raise NotImplementedError(f"Sparse attention backward score requires SM100+ (got compute capability {compute_capability}).")


_sparse_attn_score_recompute.compile_cache = {}


def _dispatch_sparse_attn_tile_params(
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    compact: bool = False,
):
    """Select (m_block_size, n_block_size, k_block_size) for sparse attention backward.

    Returns (m_block_size, n_block_size, k_block_size) where k_block_size=None
    means no head_dim splitting.

    Compact (have_topk_length) and non-compact have different optimal configs:
    compact benefits from finer-grained early termination (smaller n, less
    k-split overhead), non-compact benefits from n=128 + k-split for better
    pipeline utilization.

    Rules tuned on B200 via tests/sweep_tile_params.py (--compact / default).
    """
    m = qhead_per_kv_head

    if compact:
        # --- Compact tuned configs ---
        # hd=512
        if head_dim == 512 and qhead_per_kv_head == 64:
            return m, 64, None  # kvs=2, ~789 TFLOPS
        if head_dim == 512 and qhead_per_kv_head == 128:
            return m, 128, 128  # kvs=2, ~844/1004 TFLOPS
        # hd=576
        if head_dim == 576 and qhead_per_kv_head == 64:
            return m, 64, None  # kvs=2, ~827 TFLOPS
        if head_dim == 576 and qhead_per_kv_head == 128:
            return m, 64, 192  # kvs=2~3, ~677/1054 TFLOPS
        return m, 64, None

    # --- Non-compact tuned configs ---
    # hd=512
    if head_dim == 512 and qhead_per_kv_head == 64:
        return m, 128, 256  # kvs=2, ~800 TFLOPS
    if head_dim == 512 and qhead_per_kv_head == 128 and topk <= 512:
        return m, 128, 128  # kvs=2, ~1000 TFLOPS
    if head_dim == 512 and qhead_per_kv_head == 128:
        return m, 128, 256  # kvs=1, ~910 TFLOPS
    # hd=576
    if head_dim == 576 and qhead_per_kv_head == 64:
        return m, 128, 192  # kvs=3, ~770 TFLOPS
    if head_dim == 576 and qhead_per_kv_head == 128:
        return m, 128, 192  # kvs=1, ~830 TFLOPS

    return m, 64, None


def sparse_attn_score_recompute(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """
    Public entry point for sparse attention backward target computation.

    Recovers per-head softmax from LSE, sums across heads, L1-normalizes:
      P[b,q,h,i] = exp(Q_h · K_{topk[b,q,i]}^T · scale - LSE[b,q,h])
      S[b,q,i] = sum_h P[b,q,h,i]
      target = S / sum(S)  (L1-norm over topk dim)

    Args:
        q_attn: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k_attn: (bs, seqlen_k, head_dim) [BF16]
        lse: (bs, seqlen_q, n_heads_q) [FP32]
        topk_indices: (bs, seqlen_q, topk) [INT32]
        softmax_scale: float
        qhead_per_kv_head: auto inferred if None
        out: pre-allocated output (bs, seqlen_q, topk) [FP32], optional
        topk_length: (bs, seqlen_q) [INT32] — compact layout, optional

    Returns:
        target: (bs, seqlen_q, topk) [FP32]
    """
    if qhead_per_kv_head is None:
        qhead_per_kv_head = q_attn.shape[2]
    head_dim = q_attn.shape[3]
    topk = topk_indices.shape[2]

    compact = topk_length is not None
    m_block_size, n_block_size, k_block_size = _dispatch_sparse_attn_tile_params(
        head_dim,
        qhead_per_kv_head,
        topk,
        compact=compact,
    )

    # Optimization: when compact and non-compact tile configs match,
    # use non-compact kernel path (have_topk_length=False) for better
    # codegen — static loop bounds avoid dynamic branch overhead that
    # causes 18-28% regression in compact attention.
    # Invalid positions (topk_idx=-1) are skipped in K load and masked
    # in epilogue, preserving correctness.
    if compact:
        nc_params = _dispatch_sparse_attn_tile_params(
            head_dim,
            qhead_per_kv_head,
            topk,
            compact=False,
        )
        if (m_block_size, n_block_size, k_block_size) == nc_params:
            topk_length = None  # use non-compact kernel path

    return _sparse_attn_score_recompute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        softmax_scale,
        qhead_per_kv_head,
        out,
        topk_length,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
        topk_indices_global=topk_indices_global,
        current_stream=current_stream,
    )


# =============================================================================
# Dense backward: full KV via TMA, no topk
# =============================================================================

_SM100_SMEM_BYTES = 228 * 1024


def _select_dense_k_block_size(head_dim_padded, m_block_size, n_block_size, per_head_elem_bytes):
    """Auto-select k_block_size so that sQ + sK(1 stage) + overhead fits in SMEM.

    When head_dim_padded is too large (e.g. m_block=128, hd=512 -> sQ=128KB,
    sK=128KB > 228KB), we reduce k_block_size to shrink sK per stage.
    sQ total = m_block * head_dim_padded * 2 (fixed, since Q has num_k_chunks stages).

    k_block_size must be a multiple of 64 and divide head_dim_padded evenly.
    """
    sQ_size = m_block_size * head_dim_padded * 2
    sPerHead_bytes = m_block_size * 2 * per_head_elem_bytes
    smem_overhead = sPerHead_bytes + 2048

    avail_for_sK = _SM100_SMEM_BYTES - sQ_size - smem_overhead
    max_kbs = max(64, avail_for_sK // (n_block_size * 2))
    max_kbs = (max_kbs // 64) * 64

    kbs = head_dim_padded
    while kbs > max_kbs:
        # Try common divisors of head_dim_padded in descending order
        found = False
        for candidate in range(kbs - 64, 63, -64):
            if candidate > 0 and head_dim_padded % candidate == 0:
                kbs = candidate
                found = True
                break
        if not found:
            kbs = 64
            break

    assert head_dim_padded % kbs == 0, f"Cannot find valid k_block_size: head_dim_padded={head_dim_padded}, " f"m_block={m_block_size}, n_block={n_block_size}"
    return kbs


def _compute_dense_kv_stage(head_dim_padded, m_block_size, n_block_size, k_block_size_eff, per_head_elem_bytes):
    sK_per_stage = n_block_size * k_block_size_eff * 2
    sQ_size = m_block_size * head_dim_padded * 2
    sPerHead_bytes = m_block_size * 2 * per_head_elem_bytes
    smem_overhead = sPerHead_bytes + 2048
    return min(4, max(1, (_SM100_SMEM_BYTES - sQ_size - smem_overhead) // sK_per_stage))


# ---- Dispatch helpers (mirror _dispatch_sparse_attn_tile_params) -------------


def _dense_m_block_with_smem_check(qhead_per_kv_head, head_dim, per_head_elem_bytes):
    """Return m=qhpkv*2 if SMEM fits, else fall back to m=qhpkv."""
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    n = 128
    min_kbs = 64
    smem_overhead = 4096

    m2 = qhead_per_kv_head * 2
    sQ_m2 = m2 * head_dim_padded * 2
    sK_min = n * min_kbs * 2
    sPerHead_m2 = m2 * 2 * per_head_elem_bytes
    if sQ_m2 + sK_min + sPerHead_m2 + smem_overhead <= _SM100_SMEM_BYTES:
        return m2

    return qhead_per_kv_head


def _dispatch_dense_attn_tile_params(
    head_dim,
    qhead_per_kv_head,
    *,
    precision: str = "bf16",
    seqlen_k_hint: Optional[int] = None,
    k_block_size: Optional[int] = None,
):
    """Select (m_block_size, n_block_size, k_block_size) for dense attention backward.

    n_block_size is always 128 (required for Ld32x32bOp TMEM path).
    m_block_size = qhpkv * 2 (2 q_tokens per tile) when SMEM allows,
    falling back to qhpkv when it doesn't (e.g. nh_q=128, hd=512).

    Returns (m_block_size, n_block_size, k_block_size) where k_block_size=None
    means no head_dim splitting.

    Rules tuned on B200 via dense score sweeps.
    """
    precision = precision.lower()
    if precision == "mxfp8":
        if head_dim == 512:
            default_k = 256 if seqlen_k_hint is not None and seqlen_k_hint >= 4096 else 128
        else:
            default_k = 64
        return 128, 128, default_k if k_block_size is None else k_block_size
    if precision != "bf16":
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")

    m = _dense_m_block_with_smem_check(qhead_per_kv_head, head_dim, per_head_elem_bytes=4)
    n = 128

    if head_dim == 512:
        if qhead_per_kv_head == 64:
            return m, n, 128 if k_block_size is None else k_block_size
        if qhead_per_kv_head == 128:
            return m, n, 128 if k_block_size is None else k_block_size
        return m, n, 64 if k_block_size is None else k_block_size
    if head_dim == 576:
        return m, n, 64 if k_block_size is None else k_block_size

    # Fallback: auto-select from SMEM budget
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    k = _select_dense_k_block_size(head_dim_padded, m, n, per_head_elem_bytes=4)
    if k == head_dim_padded:
        k = None
    if k_block_size is not None:
        k = k_block_size
    return m, n, k


def _dispatch_dense_indexer_tile_params(
    head_dim,
    qhead_per_kv_head,
    *,
    precision: str = "bf16",
    k_block_size: Optional[int] = None,
):
    """Select (m_block_size, n_block_size, k_block_size) for dense indexer backward.

    m_block_size = qhpkv * 2 (2 q_tokens per tile) when SMEM allows,
    falling back to qhpkv when it doesn't.

    Returns (m_block_size, n_block_size, k_block_size) where k_block_size=None
    means no head_dim splitting.

    Rules tuned on B200 via dense score sweeps.
    """
    precision = precision.lower()
    if precision == "mxfp8":
        return 128, 128, 64 if k_block_size is None else k_block_size
    if precision != "bf16":
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")

    m = _dense_m_block_with_smem_check(qhead_per_kv_head, head_dim, per_head_elem_bytes=2)
    n = 128

    if head_dim == 128:
        return m, n, 64 if k_block_size is None else k_block_size

    # Fallback: auto-select from SMEM budget
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    k = _select_dense_k_block_size(head_dim_padded, m, n, per_head_elem_bytes=2)
    if k == head_dim_padded:
        k = None
    if k_block_size is not None:
        k = k_block_size
    return m, n, k


# ---- Internal dense functions (explicit tile params) ------------------------


def _dense_indexer_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 128,
    k_block_size: Optional[int] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal: dense indexer backward with explicit tile params.

    BSHD layout (default):
        q: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k: (bs, seqlen_k, n_heads_kv, head_dim) [BF16]
        weights: (bs, seqlen_q, n_heads_q) [BF16]
        out: optional (bs, seqlen_q, seqlen_k) [FP32]
        denom_out: optional (bs, seqlen_q) [FP32] — LogSumExp denom

    THD layout (when cu_seqlens_q/k supplied):
        q: (total_q, n_heads_q, head_dim) [BF16]
        k: (total_k, n_heads_kv, head_dim) [BF16]
        weights: (total_q, n_heads_q) [BF16]
        out: optional (total_q, max_seqlen_k) [FP32]
        denom_out: optional (total_q,) [FP32]

    Both dense-indexer implementations fill causal-invalid score entries
    with -inf, matching raw-logit mask semantics. Denom/LSE semantics are
    unchanged.

    Returns:
        (out, denom_out)
    """
    current_stream = _resolve_stream(current_stream)
    q, k, weights = [maybe_contiguous(t, current_stream) for t in (q, k, weights)]
    q_scale = maybe_contiguous(q_scale, current_stream)
    k_scale = maybe_contiguous(k_scale, current_stream)
    precision = _normalize_dense_precision(
        precision,
        q_scale,
        k_scale,
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    assert is_varlen_q == is_varlen_k, "THD input requires both cu_seqlens_q and cu_seqlens_k"
    is_varlen = is_varlen_q

    if is_varlen:
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1 and t.is_cuda, f"{name} must be 1D CUDA tensor"
            assert t.stride(0) == 1, f"{name} must be contiguous"
        assert q.ndim == 3 and k.ndim == 3 and weights.ndim == 2, "THD dense indexer expects q=(total_q,H,D), k=(total_k,H_kv,D), w=(total_q,H)"
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD dense indexer requires max_seqlen_q and max_seqlen_k")
        total_q, n_heads_q, head_dim = q.shape
        total_k, n_heads_kv, head_dim_k = k.shape
        bs = cu_seqlens_q.shape[0] - 1
        seqlen_q = int(max_seqlen_q)
        seqlen_k = int(max_seqlen_k)
        assert head_dim == head_dim_k
        assert weights.shape == (total_q, n_heads_q)
        assert cu_seqlens_q.shape == (bs + 1,) and cu_seqlens_k.shape == (bs + 1,)
    else:
        bs, seqlen_q, n_heads_q, head_dim = q.shape
        _, seqlen_k, n_heads_kv, _ = k.shape
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    if precision == "mxfp8":
        if qhead_per_kv_head not in (32, 64):
            raise ValueError("precision='mxfp8' dense indexer requires qhead_per_kv_head=32 or 64")
        if m_block_size is not None and m_block_size != 128:
            raise ValueError("precision='mxfp8' dense indexer requires m_block_size=128")
        if n_block_size != 128:
            raise ValueError("precision='mxfp8' dense indexer requires n_block_size=128")
        m_block_size = 128
        # k_block_size is locked to 64 (the D128 two-chunk plan): kv_stage is
        # hardcoded to 24 below, which keeps K-SMEM within the SM100 budget only
        # at k_block=64 (k_block>64 overflows SMEM; k_block<64 under-pipelines).
        k_block_size = 64 if k_block_size is None else k_block_size
        if k_block_size != 64:
            raise ValueError("precision='mxfp8' dense indexer requires k_block_size=64; " f"got {k_block_size}")
    elif m_block_size is None:
        m_block_size = qhead_per_kv_head

    if precision == "mxfp8":
        _validate_dense_mxfp8_contract(
            q=q,
            k=k,
            per_head=weights,
            per_head_name="weights",
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            qhead_per_kv_head=qhead_per_kv_head,
            head_dim=head_dim,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            bs=bs,
            n_heads_q=n_heads_q,
            n_heads_kv=n_heads_kv,
            sf_vec_size=sf_vec_size,
            is_varlen=is_varlen,
        )

    device = q.device

    if out is None:
        out_shape = (q.shape[0], seqlen_k) if is_varlen else (bs, seqlen_q, seqlen_k)
        with _torch_stream_context(current_stream):
            out = torch.empty(out_shape, dtype=torch.float32, device=device)
    if denom_out is None:
        denom_shape = (q.shape[0],) if is_varlen else (bs, seqlen_q)
        with _torch_stream_context(current_stream):
            denom_out = torch.empty(denom_shape, dtype=torch.float32, device=device)
    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    if k_block_size is None:
        k_block_size = _select_dense_k_block_size(head_dim_padded, m_block_size, n_block_size, per_head_elem_bytes=2)
    kv_stage = _compute_dense_kv_stage(head_dim_padded, m_block_size, n_block_size, k_block_size, per_head_elem_bytes=2)
    if precision == "mxfp8":
        # K-SMEM = n_block * k_block_size(=64) * kv_stage (fp8); 24 fits the
        # SM100 SMEM budget. k_block is locked to 64 above, so this is constant.
        kv_stage = 24

    compute_capability = _get_device_capability()
    if compute_capability < 10:
        raise NotImplementedError(f"Dense indexer backward requires SM100+ (got {compute_capability}).")

    if precision == "mxfp8":
        assert q_scale is not None and k_scale is not None
        compile_key = (
            "dense_indexer",
            "mxfp8",
            q.dtype,
            k.dtype,
            weights.dtype,
            q_scale.dtype,
            k_scale.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            head_dim_padded // k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
            sf_vec_size,
        )
    else:
        base_compile_key = (
            q.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
        )
        compile_key = ("dense_indexer",) + base_compile_key
    if compile_key not in _dense_indexer_score_recompute.compile_cache:
        q_cute = to_cute_tensor(q)
        k_cute = to_cute_tensor(k)
        w_cute = to_cute_tensor(weights)
        out_cute = to_cute_tensor(out)
        denom_cute = to_cute_tensor(denom_out)
        cu_q_cute = to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        cu_q_scale_cute = to_cute_tensor(cu_seqlens_q_scale_padded, leading_dim=0) if is_varlen and precision == "mxfp8" else None
        cu_k_scale_cute = to_cute_tensor(cu_seqlens_k_scale_padded, leading_dim=0) if is_varlen and precision == "mxfp8" else None
        q_offsets_cute = to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

        if precision == "mxfp8":
            assert q_scale is not None and k_scale is not None
            q_scale_cute = to_cute_tensor(q_scale)
            k_scale_cute = to_cute_tensor(k_scale)
            kernel_obj = IndexerScoreUnifiedSm100Mxfp8(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                ratio=ratio,
                is_varlen=is_varlen,
                sf_vec_size=sf_vec_size,
                compute_lse=True,
                is_compressed_logits=False,
            )
        else:
            q_scale_cute = None
            k_scale_cute = None
            kernel_obj = IndexerScoreUnifiedSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                ratio=ratio,
                is_varlen=is_varlen,
                compute_lse=True,
                is_compressed_logits=False,
            )
        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q)
        max_k_arg = cutlass.Int32(seqlen_k)

        if precision == "mxfp8":
            _dense_indexer_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                q_scale_cute,
                k_scale_cute,
                out_cute,
                denom_cute,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                cu_q_scale_cute,
                cu_k_scale_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )
        else:
            _dense_indexer_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                out_cute,
                denom_cute,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )

    # Dense indexer score is a raw-logit buffer, so masked/skipped positions
    # must remain outside the softmax/logsumexp domain.
    with _torch_stream_context(current_stream):
        out.fill_(float("-inf"))
    scale_arg = cutlass.Float32(sm_scale)
    max_q_arg = cutlass.Int32(seqlen_q)
    max_k_arg = cutlass.Int32(seqlen_k)
    with torch.cuda.nvtx.range("dense_indexer_score_recompute"):
        if precision == "mxfp8":
            _dense_indexer_score_recompute.compile_cache[compile_key](
                q,
                k,
                weights,
                q_scale,
                k_scale,
                out,
                denom_out,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                cu_seqlens_q_scale_padded if is_varlen else None,
                cu_seqlens_k_scale_padded if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
        else:
            _dense_indexer_score_recompute.compile_cache[compile_key](
                q,
                k,
                weights,
                out,
                denom_out,
                scale_arg,
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
    return out, denom_out


_dense_indexer_score_recompute.compile_cache = {}


def _dense_attn_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: int = 128,
    k_block_size: Optional[int] = None,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Internal: dense attention backward with explicit tile params.

    BSHD layout (default):
        q: (bs, seqlen_q, n_heads_q, head_dim) [BF16]
        k: (bs, seqlen_k, n_heads_kv, head_dim) [BF16]
        lse: (bs, seqlen_q, n_heads_q) [FP32]
        out: optional (bs, seqlen_q, seqlen_k) [FP32]
        denom_out: optional (bs, seqlen_q) [FP32] — L1-norm denom

    THD layout (when cu_seqlens_q/k supplied):
        q: (total_q, n_heads_q, head_dim) [BF16]
        k: (total_k, n_heads_kv, head_dim) [BF16]
        lse: (total_q, n_heads_q) [FP32]
        out: optional (total_q, max_seqlen_k) [FP32]
        denom_out: optional (total_q,) [FP32]

    Returns:
        (out, denom_out)
    """
    current_stream = _resolve_stream(current_stream)
    q, k = [maybe_contiguous(t, current_stream) for t in (q, k)]
    q_scale = maybe_contiguous(q_scale, current_stream)
    k_scale = maybe_contiguous(k_scale, current_stream)
    precision = _normalize_dense_precision(
        precision,
        q_scale,
        k_scale,
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    assert is_varlen_q == is_varlen_k, "THD input requires both cu_seqlens_q and cu_seqlens_k"
    is_varlen = is_varlen_q

    if is_varlen:
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1 and t.is_cuda, f"{name} must be 1D CUDA tensor"
            assert t.stride(0) == 1, f"{name} must be contiguous"
        assert q.ndim == 3 and k.ndim == 3 and lse.ndim == 2, "THD dense attn expects q=(total_q,H,D), k=(total_k,H_kv,D), lse=(total_q,H)"
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD dense attn requires max_seqlen_q and max_seqlen_k")
        total_q, n_heads_q, head_dim = q.shape
        total_k, n_heads_kv, head_dim_k = k.shape
        bs = cu_seqlens_q.shape[0] - 1
        seqlen_q = int(max_seqlen_q)
        seqlen_k = int(max_seqlen_k)
        assert head_dim == head_dim_k
        assert lse.shape == (total_q, n_heads_q)
        assert cu_seqlens_q.shape == (bs + 1,) and cu_seqlens_k.shape == (bs + 1,)
    else:
        bs, seqlen_q, n_heads_q, head_dim = q.shape
        _, seqlen_k, n_heads_kv, _ = k.shape
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    device = q.device

    lse = maybe_contiguous(lse, current_stream)

    if precision == "mxfp8":
        _validate_dense_mxfp8_contract(
            q=q,
            k=k,
            per_head=lse,
            per_head_name="lse",
            q_scale=q_scale,
            k_scale=k_scale,
            cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
            cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
            qhead_per_kv_head=qhead_per_kv_head,
            head_dim=head_dim,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            bs=bs,
            n_heads_q=n_heads_q,
            n_heads_kv=n_heads_kv,
            sf_vec_size=sf_vec_size,
            is_varlen=is_varlen,
        )

    if out is None:
        out_shape = (q.shape[0], seqlen_k) if is_varlen else (bs, seqlen_q, seqlen_k)
        with _torch_stream_context(current_stream):
            out = torch.empty(out_shape, dtype=torch.float32, device=device)
    if denom_out is None:
        denom_shape = (q.shape[0],) if is_varlen else (bs, seqlen_q)
        with _torch_stream_context(current_stream):
            denom_out = torch.empty(denom_shape, dtype=torch.float32, device=device)
    if precision == "mxfp8" and ((head_dim == 512 and qhead_per_kv_head >= 128) or (head_dim == 128 and qhead_per_kv_head == 64)):
        with _torch_stream_context(current_stream):
            out.zero_()
            denom_out.zero_()

    head_dim_padded = int(math.ceil(head_dim / 16) * 16)
    if k_block_size is None:
        k_block_size = _select_dense_k_block_size(head_dim_padded, m_block_size, n_block_size, per_head_elem_bytes=4)
    kv_stage = _compute_dense_kv_stage(head_dim_padded, m_block_size, n_block_size, k_block_size, per_head_elem_bytes=4)
    if precision == "mxfp8":
        if head_dim == 512:
            if k_block_size == 64:
                kv_stage = 7
            elif k_block_size == 256:
                kv_stage = 3
            elif qhead_per_kv_head >= 128:
                kv_stage = 9
            else:
                kv_stage = 7
        else:
            kv_stage = 4

    compute_capability = _get_device_capability()
    if compute_capability < 10:
        raise NotImplementedError(f"Dense attention backward requires SM100+ (got {compute_capability}).")

    if precision == "mxfp8":
        assert q_scale is not None and k_scale is not None
        compile_key = (
            "dense_attention",
            "mxfp8",
            q.dtype,
            k.dtype,
            lse.dtype,
            q_scale.dtype,
            k_scale.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            head_dim_padded // k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
            sf_vec_size,
        )
    else:
        compile_key = (
            "dense_attention",
            q.dtype,
            head_dim,
            qhead_per_kv_head,
            m_block_size,
            n_block_size,
            k_block_size,
            kv_stage,
            ratio,
            is_varlen,
            q_causal_offsets is not None,
        )

    if compile_key not in _dense_attn_score_recompute.compile_cache:
        q_cute = to_cute_tensor(q)
        k_cute = to_cute_tensor(k)
        lse_cute = to_cute_tensor(lse)
        out_cute = to_cute_tensor(out)
        denom_cute = to_cute_tensor(denom_out)
        cu_q_cute = to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        cu_q_scale_cute = to_cute_tensor(cu_seqlens_q_scale_padded, leading_dim=0) if is_varlen and precision == "mxfp8" else None
        cu_k_scale_cute = to_cute_tensor(cu_seqlens_k_scale_padded, leading_dim=0) if is_varlen and precision == "mxfp8" else None
        q_offsets_cute = to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

        if precision == "mxfp8":
            assert q_scale is not None and k_scale is not None
            q_scale_cute = to_cute_tensor(q_scale)
            k_scale_cute = to_cute_tensor(k_scale)
            kernel_obj = BwdDenseAttnScoreSm100Mxfp8(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                ratio=ratio,
                is_varlen=is_varlen,
                sf_vec_size=sf_vec_size,
            )
        else:
            q_scale_cute = None
            k_scale_cute = None
            kernel_obj = DenseScoreRecomputeSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=k_block_size,
                kv_stage=kv_stage,
                score_type="attention",
                ratio=ratio,
                is_varlen=is_varlen,
            )

        max_q_arg = cutlass.Int32(seqlen_q)
        max_k_arg = cutlass.Int32(seqlen_k)

        if precision == "mxfp8":
            _dense_attn_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                lse_cute,
                q_scale_cute,
                k_scale_cute,
                out_cute,
                denom_cute,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                cu_q_scale_cute,
                cu_k_scale_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )
        else:
            _dense_attn_score_recompute.compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                lse_cute,
                out_cute,
                denom_cute,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_q_cute,
                cu_k_cute,
                q_offsets_cute,
                current_stream,
                options=compile_options(),
            )

    max_q_arg = cutlass.Int32(seqlen_q)
    max_k_arg = cutlass.Int32(seqlen_k)
    # Dense score kernels skip K blocks that are wholly outside the ratio-causal
    # region, so skipped masked/padding columns must be pre-filled. MXFP8
    # dense-attention uses mOut as an atomic-add accumulation buffer for valid
    # scores in some paths; those kernels patch invalid/skipped scores to -inf.
    with _torch_stream_context(current_stream):
        if precision == "mxfp8":
            out.zero_()
        else:
            out.fill_(float("-inf"))
    with torch.cuda.nvtx.range("dense_attn_score_recompute"):
        if precision == "mxfp8":
            _dense_attn_score_recompute.compile_cache[compile_key](
                q,
                k,
                lse,
                q_scale,
                k_scale,
                out,
                denom_out,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                cu_seqlens_q_scale_padded if is_varlen else None,
                cu_seqlens_k_scale_padded if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
        else:
            _dense_attn_score_recompute.compile_cache[compile_key](
                q,
                k,
                lse,
                out,
                denom_out,
                cutlass.Float32(softmax_scale),
                max_q_arg,
                max_k_arg,
                cu_seqlens_q if is_varlen else None,
                cu_seqlens_k if is_varlen else None,
                q_causal_offsets,
                current_stream,
            )
    return out, denom_out


_dense_attn_score_recompute.compile_cache = {}


# ---- Public dense entry points (auto-dispatch tile params) ------------------


def dense_indexer_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    k_block_size: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense indexer backward score: sm_scale * ReLU(QK)*W head reduce over full KV.

    Tile params auto-selected via _dispatch_dense_indexer_tile_params. sm_scale
    is applied to the fp32 head-reduced score inside the kernel (preserves
    precision vs pre-multiplying onto bf16 weights on the host).
    The shared unified indexer kernel runs with ``compute_lse=True``;
    causal-invalid entries in ``out`` remain -inf and ``denom_out`` stores
    LogSumExp.

    Args:
        q: BSHD ``(bs, seqlen_q, n_heads_q, head_dim)`` or THD
           ``(total_q, n_heads_q, head_dim)`` [BF16]
        k: BSHD ``(bs, seqlen_k, n_heads_kv, head_dim)`` or THD
           ``(total_k, n_heads_kv, head_dim)`` [BF16]
        weights: BSH ``(bs, seqlen_q, n_heads_q)`` or TH ``(total_q, n_heads_q)`` [BF16]
        out: optional. BSHD ``(bs, seqlen_q, seqlen_k)``;
             THD ``(total_q, max_seqlen_k)`` [FP32]
        denom_out: optional. BSHD ``(bs, seqlen_q)`` ; THD ``(total_q,)`` [FP32]
        sm_scale: scalar applied to fp32 score post head-reduce; default 1.0
        ratio, cu_seqlens_*, max_seqlen_*: see causal mask + THD docs.

    Returns:
        (out, denom_out)
    """
    if qhead_per_kv_head is None:
        qhead_per_kv_head = (q.shape[-2] // k.shape[-2]) if q.ndim == 3 else (q.shape[2] // k.shape[2])
    head_dim = q.shape[-1]

    precision = _normalize_dense_precision(precision, q_scale, k_scale)
    m, n, kbs = _dispatch_dense_indexer_tile_params(
        head_dim,
        qhead_per_kv_head,
        precision=precision,
        k_block_size=k_block_size,
    )
    return _dense_indexer_score_recompute(
        q,
        k,
        weights,
        qhead_per_kv_head,
        out,
        denom_out,
        m_block_size=m,
        n_block_size=n,
        k_block_size=kbs,
        sm_scale=sm_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        current_stream=current_stream,
    )


def dense_attn_score_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    k_block_size: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense attention backward score: exp(QK*scale - LSE) head reduce over full KV.

    Tile params auto-selected via _dispatch_dense_attn_tile_params.

    Args:
        q: BSHD ``(bs, seqlen_q, n_heads_q, head_dim)`` or THD
           ``(total_q, n_heads_q, head_dim)`` [BF16]
        k: BSHD ``(bs, seqlen_k, n_heads_kv, head_dim)`` or THD
           ``(total_k, n_heads_kv, head_dim)`` [BF16]
        lse: BSH ``(bs, seqlen_q, n_heads_q)`` or TH ``(total_q, n_heads_q)`` [FP32]
        softmax_scale: float
        out: optional. BSHD ``(bs, seqlen_q, seqlen_k)``;
             THD ``(total_q, max_seqlen_k)`` [FP32]
        denom_out: optional. BSHD ``(bs, seqlen_q)`` ; THD ``(total_q,)`` [FP32]
        ratio, cu_seqlens_*, max_seqlen_*: see causal mask + THD docs.

    Returns:
        (out, denom_out)
    """
    if qhead_per_kv_head is None:
        qhead_per_kv_head = (q.shape[-2] // k.shape[-2]) if q.ndim == 3 else (q.shape[2] // k.shape[2])
    head_dim = q.shape[-1]

    precision = _normalize_dense_precision(precision, q_scale, k_scale)
    seqlen_k_hint = int(max_seqlen_k) if max_seqlen_k is not None else (k.shape[0] if q.ndim == 3 else k.shape[1])
    m, n, kbs = _dispatch_dense_attn_tile_params(
        head_dim,
        qhead_per_kv_head,
        precision=precision,
        seqlen_k_hint=seqlen_k_hint,
        k_block_size=k_block_size,
    )

    return _dense_attn_score_recompute(
        q,
        k,
        lse,
        softmax_scale,
        qhead_per_kv_head,
        out,
        denom_out,
        m_block_size=m,
        n_block_size=n,
        k_block_size=kbs,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        current_stream=current_stream,
    )
