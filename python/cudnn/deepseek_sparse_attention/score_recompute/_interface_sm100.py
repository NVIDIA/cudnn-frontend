"""
Score Recompute Interface (SM100 Cute-DSL wrapper).

Orchestrates compilation, caching, and kernel dispatch for:
  - sparse_indexer_score_recompute: sparse index scores + softmax (predict)
  - sparse_attn_score_recompute: sparse attention scores + L1-norm (target)
  - dense_indexer_score_recompute: dense index scores + LSE denom
  - dense_attn_score_recompute: dense attention scores + L1-norm denom
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100
from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100
from .config import (
    dispatch_sparse_attn_tile_params as _shared_dispatch_sparse_attn_tile_params,
    resolve_dense_score_kernel_config,
    resolve_dense_score_smem_config,
    resolve_sparse_score_smem_config,
)
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
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

    kv_stage, topk_in_smem = resolve_sparse_score_smem_config(
        score_type="indexer",
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=None,
        topk=topk,
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

    kv_stage, topk_in_smem = resolve_sparse_score_smem_config(
        score_type="attention",
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
        topk=topk,
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
    return _shared_dispatch_sparse_attn_tile_params(
        head_dim,
        qhead_per_kv_head,
        topk,
        compact=compact,
    )


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


def _dispatch_dense_attn_tile_params(head_dim, qhead_per_kv_head):
    config = resolve_dense_score_kernel_config(
        score_type="attention",
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    return config.m_block_size, config.n_block_size, config.k_block_size


def _dispatch_dense_indexer_tile_params(head_dim, qhead_per_kv_head):
    config = resolve_dense_score_kernel_config(
        score_type="indexer",
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    return config.m_block_size, config.n_block_size, config.k_block_size


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
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
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

    Returns:
        (out, denom_out)
    """
    current_stream = _resolve_stream(current_stream)
    q, k, weights = [maybe_contiguous(t, current_stream) for t in (q, k, weights)]

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
        _, seqlen_k, _, _ = k.shape
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q
    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    device = q.device

    if out is None:
        out_shape = (q.shape[0], seqlen_k) if is_varlen else (bs, seqlen_q, seqlen_k)
        with _torch_stream_context(current_stream):
            out = torch.empty(out_shape, dtype=torch.float32, device=device)
    if denom_out is None:
        denom_shape = (q.shape[0],) if is_varlen else (bs, seqlen_q)
        with _torch_stream_context(current_stream):
            denom_out = torch.empty(denom_shape, dtype=torch.float32, device=device)

    k_block_size, kv_stage = resolve_dense_score_smem_config(
        score_type="indexer",
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
    )

    compute_capability = _get_device_capability()
    if compute_capability < 10:
        raise NotImplementedError(f"Dense indexer backward requires SM100+ (got {compute_capability}).")

    compile_key = (
        "dense_indexer",
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

    if compile_key not in _dense_indexer_score_recompute.compile_cache:
        q_cute = to_cute_tensor(q)
        k_cute = to_cute_tensor(k)
        w_cute = to_cute_tensor(weights)
        out_cute = to_cute_tensor(out)
        denom_cute = to_cute_tensor(denom_out)
        cu_q_cute = to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        q_offsets_cute = to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

        kernel_obj = DenseScoreRecomputeSm100(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv_head,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            k_block_size=k_block_size,
            kv_stage=kv_stage,
            score_type="indexer",
            ratio=ratio,
            is_varlen=is_varlen,
        )

        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q)
        max_k_arg = cutlass.Int32(seqlen_k)

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

    scale_arg = cutlass.Float32(sm_scale)
    max_q_arg = cutlass.Int32(seqlen_q)
    max_k_arg = cutlass.Int32(seqlen_k)
    # Dense indexer score is a raw-logit buffer, so masked/skipped positions
    # must remain outside the softmax/logsumexp domain.
    with _torch_stream_context(current_stream):
        out.fill_(float("-inf"))
    with torch.cuda.nvtx.range("dense_indexer_score_recompute"):
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
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
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
        _, seqlen_k, _, _ = k.shape
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q
    if m_block_size is None:
        m_block_size = qhead_per_kv_head

    device = q.device

    lse = maybe_contiguous(lse, current_stream)

    if out is None:
        out_shape = (q.shape[0], seqlen_k) if is_varlen else (bs, seqlen_q, seqlen_k)
        with _torch_stream_context(current_stream):
            out = torch.empty(out_shape, dtype=torch.float32, device=device)
    if denom_out is None:
        denom_shape = (q.shape[0],) if is_varlen else (bs, seqlen_q)
        with _torch_stream_context(current_stream):
            denom_out = torch.empty(denom_shape, dtype=torch.float32, device=device)

    k_block_size, kv_stage = resolve_dense_score_smem_config(
        score_type="attention",
        head_dim=head_dim,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        k_block_size=k_block_size,
    )

    compute_capability = _get_device_capability()
    if compute_capability < 10:
        raise NotImplementedError(f"Dense attention backward requires SM100+ (got {compute_capability}).")

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
        q_offsets_cute = to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

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
    # Dense score kernels skip K blocks that are wholly outside the
    # ratio-causal region, so skipped masked/padding columns must be prefilled.
    with _torch_stream_context(current_stream):
        out.fill_(float("-inf"))
    with torch.cuda.nvtx.range("dense_attn_score_recompute"):
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
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dense indexer backward score: sm_scale * ReLU(QK)*W head reduce over full KV.

    Tile params auto-selected via _dispatch_dense_indexer_tile_params. sm_scale
    is applied to the fp32 head-reduced score inside the kernel (preserves
    precision vs pre-multiplying onto bf16 weights on the host).

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
        qhead_per_kv_head = q.shape[-2] if q.ndim == 3 else q.shape[2]
    head_dim = q.shape[-1]

    m, n, kbs = _dispatch_dense_indexer_tile_params(head_dim, qhead_per_kv_head)
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
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
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
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
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
        qhead_per_kv_head = q.shape[-2] if q.ndim == 3 else q.shape[2]
    head_dim = q.shape[-1]

    m, n, kbs = _dispatch_dense_attn_tile_params(head_dim, qhead_per_kv_head)
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
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        current_stream=current_stream,
    )
