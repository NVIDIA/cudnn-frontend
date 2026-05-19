"""
Score Recompute Interface (SM90 Cute-DSL wrapper).

Mirrors the SM100 interface layout, exposing four public entry points that
each compile / cache / launch one kernel variant and emit a dedicated NVTX
range:

  - sparse_indexer_score_recompute: sparse index scores + softmax (predict)
  - sparse_attn_score_recompute:    sparse attention scores + L1-norm (target)
  - dense_indexer_score_recompute:  dense index scores + LSE denom (3-WG)
  - dense_attn_score_recompute:     dense attention scores + L1-norm denom (3-WG)

The underlying kernel objects are ``SparseScoreRecomputeSm90`` / ``DenseScoreRecomputeSm90``.
The dense kernel is now the 3-WG pingpong implementation by default and no
longer exposes the legacy 1-WG/2-WG dense variants.

The legacy ``indexer_scores_bwd`` compatibility shim was removed; callers
should use the explicit sparse/dense score entry points below.
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor
from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_major as _get_device_capability,
    maybe_contiguous,
    resolve_stream as _resolve_stream,
)
from .sparse_score_recompute_sm90 import SparseScoreRecomputeSm90
from .dense_score_recompute_sm90 import DenseScoreRecomputeSm90
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options

# Dense SM90 score is now 3-WG-only.  The previous 1-WG/2-WG dense
# implementation was removed after benchmarking showed 3-WG is uniformly faster
# on dense attention (+11-20%) and non-regressing on dense indexer (+1-5%).
_DENSE_NUM_THREADS = 384


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


# =============================================================================
# Tile-param helpers shared across sparse/dense kernels
# =============================================================================

_MAX_TILE_M = 64
_TILE_N = 64
_KV_STAGE = 2
_SWAP_AB = True


def _compute_tile_m(qhead_per_kvhead: int) -> tuple[int, int]:
    """Cap tile_m at MAX_TILE_M; iterate over head tiles when qhpkv > MAX_TILE_M."""
    assert qhead_per_kvhead > 1, "SM90 kernel requires MQA/GQA (qhead_per_kvhead > 1)"
    tile_m = min(qhead_per_kvhead, _MAX_TILE_M)
    assert qhead_per_kvhead % tile_m == 0, f"qhead_per_kvhead ({qhead_per_kvhead}) must be divisible by tile_m ({tile_m})"
    num_head_tiles = qhead_per_kvhead // tile_m
    return tile_m, num_head_tiles


def _validate_and_prepare_common(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights_or_lse: torch.Tensor,
    is_index_scores: bool,
):
    """Common validation + strided-contiguous pass for q, kv, weights_or_lse.

    Returns (q, kv, weights_or_lse) after maybe_contiguous.
    """
    assert q.dtype in [torch.float16, torch.bfloat16], f"q dtype must be half precision, got {q.dtype}"
    assert q.dtype == kv.dtype, f"q/kv dtype mismatch: q={q.dtype}, kv={kv.dtype}"
    if is_index_scores:
        assert weights_or_lse.dtype == q.dtype, f"weights must match q dtype (half precision): got {weights_or_lse.dtype} vs {q.dtype}"
    else:
        assert weights_or_lse.dtype == torch.float32, f"lse must be float32, got {weights_or_lse.dtype}"
    assert all(t.is_cuda for t in (q, kv, weights_or_lse))
    return [maybe_contiguous(t) for t in (q, kv, weights_or_lse)]


# =============================================================================
# Sparse path
# =============================================================================


def _sparse_score_recompute(
    q: torch.Tensor,  # (bs, seqlen_q, n_heads_q, head_dim)
    kv: torch.Tensor,  # (bs, seqlen_k, n_heads_kv, head_dim) — 4D, n_heads_kv=1
    weights_or_lse: torch.Tensor,  # (bs, n_heads_q, seqlen_q) — already transposed to (B,H,S)
    topk_indices: torch.Tensor,  # (bs, seqlen_q, topk) int32
    is_index_scores: bool,
    softmax_scale: float,
    topk_length: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    output_log_probs: bool,
    nvtx_range_name: str,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """Compile + launch sparse score kernel. Internal helper used by the four
    public entry points; they own the ``(k, weights_or_lse)`` layout conversion
    so the kernel sees the legacy SM90 layout (kv 4D + weights (B,H,S))."""
    compute_capability = _get_device_capability()
    assert compute_capability == 9, f"SM90 kernel on compute capability {compute_capability}"

    q, kv, weights_or_lse = _validate_and_prepare_common(q, kv, weights_or_lse, is_index_scores)

    batch_size, seqlen_q, num_head, head_dim = q.shape
    _, seqlen_k, num_head_kv, _ = kv.shape
    assert num_head > num_head_kv and num_head % num_head_kv == 0, f"MQA required: num_head={num_head}, num_head_kv={num_head_kv}"
    qhead_per_kvhead = num_head // num_head_kv

    tile_m, num_head_tiles = _compute_tile_m(qhead_per_kvhead)

    topk_indices = topk_indices.to(torch.int32).contiguous()
    assert topk_indices.is_cuda
    topk_max = topk_indices.shape[-1]

    if topk_length is not None:
        topk_length = topk_length.to(torch.int32).contiguous()
        assert topk_length.shape == (batch_size, seqlen_q)
    has_topk_length = topk_length is not None

    if out is None:
        out = torch.zeros((batch_size, seqlen_q, topk_max), dtype=torch.float32, device=q.device)
    else:
        out = out if out.is_contiguous() else out.contiguous()

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = _resolve_stream(current_stream)

    # Sparse path: num_threads fixed at 256 (1 producer WG + 1 consumer WG).
    num_threads = 256

    compile_key = (
        dtype,
        head_dim,
        qhead_per_kvhead,
        tile_m,
        _TILE_N,
        _KV_STAGE,
        num_threads,
        _SWAP_AB,
        topk_max,
        is_index_scores,
        softmax_scale,
        has_topk_length,
        num_head_tiles,
        True,
        output_log_probs,
        bool(topk_indices_global),
    )

    cache = _sparse_score_recompute.compile_cache
    if compile_key not in cache:
        q_cute = to_cute_tensor(q)
        kv_cute = to_cute_tensor(kv)
        topk_idxs_cute = to_cute_tensor(topk_indices)
        topk_length_cute = to_cute_tensor(topk_length) if topk_length is not None else None
        out_cute = to_cute_tensor(out)
        weights_cute = to_cute_tensor(weights_or_lse)

        kernel_obj = SparseScoreRecomputeSm90(
            dtype,
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=tile_m,
            tile_n=_TILE_N,
            num_threads=num_threads,
            swap_AB=_SWAP_AB,
            topk_max=topk_max,
            is_index_scores=is_index_scores,
            softmax_scale=softmax_scale,
            has_topk_length=has_topk_length,
            num_head_tiles=num_head_tiles,
            is_sparse=True,
            output_log_probs=output_log_probs,
            topk_indices_global=topk_indices_global,
        )

        cache[compile_key] = cute.compile(
            kernel_obj,
            q_cute,
            kv_cute,
            topk_idxs_cute,
            current_stream,
            out_cute,
            weights_cute,
            topk_length_cute,
            None,  # mL1NormDenom (dense-only)
            options=compile_options(),
        )

    with torch.cuda.nvtx.range(nvtx_range_name):
        cache[compile_key](
            q,
            kv,
            topk_indices,
            current_stream,
            out,
            weights_or_lse,
            topk_length,
            None,  # mL1NormDenom (dense-only)
        )
    return out


_sparse_score_recompute.compile_cache = {}


def sparse_indexer_score_recompute(
    q_indexer: torch.Tensor,  # (bs, seqlen_q, n_heads_q, head_dim) half
    k_indexer: torch.Tensor,  # (bs, seqlen_k, head_dim) half — MQA, 3D
    weights: torch.Tensor,  # (bs, seqlen_q, n_heads_q) half
    topk_indices: torch.Tensor,  # (bs, seqlen_q, topk) int32
    out: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    output_log_probs: bool = False,
    sm_scale: float = 1.0,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """Sparse indexer backward predict:
        S[b,q,i] = sm_scale * sum_h [ReLU(Q_h · K_{topk[b,q,i]}^T) · W_{b,q,h}]
        predict = softmax(S, dim=-1)   (log-softmax if ``output_log_probs``)

    sm_scale is applied to the fp32 head-reduced score inside the kernel
    (preserves precision vs pre-multiplying onto bf16 weights on the host).
    Defaults to 1.0 (identity).
    """
    kv = k_indexer.unsqueeze(2)
    w_bhs = weights.transpose(1, 2).contiguous()
    return _sparse_score_recompute(
        q_indexer,
        kv,
        w_bhs,
        topk_indices,
        is_index_scores=True,
        softmax_scale=sm_scale,
        topk_length=topk_length,
        out=out,
        output_log_probs=output_log_probs,
        nvtx_range_name="sparse_indexer_score_recompute",
        topk_indices_global=topk_indices_global,
        current_stream=current_stream,
    )


def sparse_attn_score_recompute(
    q_attn: torch.Tensor,  # (bs, seqlen_q, n_heads_q, head_dim) half
    k_attn: torch.Tensor,  # (bs, seqlen_k, head_dim) half — MQA, 3D
    lse: torch.Tensor,  # (bs, seqlen_q, n_heads_q) float32
    topk_indices: torch.Tensor,  # (bs, seqlen_q, topk) int32
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor:
    """Sparse attention backward target:
    P[b,q,h,i] = exp(Q_h · K_{topk[b,q,i]}^T · scale − LSE[b,q,h])
    S[b,q,i]   = sum_h P[b,q,h,i]
    target     = S / sum(S)  (L1-norm over topk)
    """
    kv = k_attn.unsqueeze(2)
    lse_bhs = lse.transpose(1, 2).contiguous()
    return _sparse_score_recompute(
        q_attn,
        kv,
        lse_bhs,
        topk_indices,
        is_index_scores=False,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        out=out,
        output_log_probs=False,
        nvtx_range_name="sparse_attn_score_recompute",
        topk_indices_global=topk_indices_global,
        current_stream=current_stream,
    )


# =============================================================================
# Dense path
# =============================================================================


def _dense_score_recompute(
    q: torch.Tensor,  # (bs, seqlen_q, n_heads_q, head_dim)
    kv: torch.Tensor,  # (bs, seqlen_k, n_heads_kv, head_dim) 4D, n_heads_kv=1
    weights_or_lse: torch.Tensor,  # (bs, n_heads_q, seqlen_q) — transposed to (B,H,S)
    is_index_scores: bool,
    softmax_scale: float,
    out: Optional[torch.Tensor],
    denom_out: Optional[torch.Tensor],
    num_threads: int,
    nvtx_range_name: str,
    ratio: int = 1,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile + launch the dense 3-WG score kernel."""
    compute_capability = _get_device_capability()
    assert compute_capability == 9, f"SM90 kernel on compute capability {compute_capability}"
    assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
    assert num_threads == _DENSE_NUM_THREADS, f"SM90 dense score is 3-WG-only (num_threads={_DENSE_NUM_THREADS}); got {num_threads}."

    q, kv, weights_or_lse = _validate_and_prepare_common(q, kv, weights_or_lse, is_index_scores)

    batch_size, seqlen_q, num_head, head_dim = q.shape
    _, seqlen_k, num_head_kv, _ = kv.shape
    assert num_head > num_head_kv and num_head % num_head_kv == 0, f"MQA required: num_head={num_head}, num_head_kv={num_head_kv}"
    qhead_per_kvhead = num_head // num_head_kv

    tile_m, num_head_tiles = _compute_tile_m(qhead_per_kvhead)

    if out is None:
        out = torch.zeros((batch_size, seqlen_q, seqlen_k), dtype=torch.float32, device=q.device)
    else:
        out = out if out.is_contiguous() else out.contiguous()
    if denom_out is None:
        denom_out = torch.zeros((batch_size, seqlen_q), dtype=torch.float32, device=q.device)
    else:
        denom_out = denom_out if denom_out.is_contiguous() else denom_out.contiguous()

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = _resolve_stream(current_stream)

    # Dense path never consumes topk_idxs / topk_length; pass topk_max = seqlen_k
    # so the kernel iterates the full KV sequence like the legacy fused path did.
    topk_max = seqlen_k

    compile_key = (
        dtype,
        head_dim,
        qhead_per_kvhead,
        tile_m,
        _TILE_N,
        _KV_STAGE,
        num_threads,
        _SWAP_AB,
        topk_max,
        is_index_scores,
        softmax_scale,
        False,
        num_head_tiles,
        False,
        False,
        ratio,  # has_topk_length, is_sparse, output_log_probs
    )

    cache = _dense_score_recompute.compile_cache
    if compile_key not in cache:
        q_cute = to_cute_tensor(q)
        kv_cute = to_cute_tensor(kv)
        out_cute = to_cute_tensor(out)
        weights_cute = to_cute_tensor(weights_or_lse)
        denom_cute = to_cute_tensor(denom_out)

        kernel_obj = DenseScoreRecomputeSm90(
            dtype,
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=tile_m,
            tile_n=_TILE_N,
            num_threads=num_threads,
            swap_AB=_SWAP_AB,
            topk_max=topk_max,
            is_index_scores=is_index_scores,
            softmax_scale=softmax_scale,
            has_topk_length=False,
            num_head_tiles=num_head_tiles,
            ratio=ratio,
        )

        cache[compile_key] = cute.compile(
            kernel_obj,
            q_cute,
            kv_cute,
            None,  # mTopkIdxs (sparse-only)
            current_stream,
            out_cute,
            weights_cute,
            None,  # mTopkLength (sparse-only)
            denom_cute,
            options=compile_options(),
        )

    with torch.cuda.nvtx.range(nvtx_range_name):
        cache[compile_key](
            q,
            kv,
            None,
            current_stream,
            out,
            weights_or_lse,
            None,
            denom_out,
        )
    return out, denom_out


def _dense_score_recompute_varlen(
    q: torch.Tensor,  # (total_q, n_heads_q, head_dim)
    kv: torch.Tensor,  # (total_k, n_heads_kv, head_dim)
    weights_or_lse: torch.Tensor,  # (total_q, n_heads_q)
    is_index_scores: bool,
    softmax_scale: float,
    out: Optional[torch.Tensor],
    denom_out: Optional[torch.Tensor],
    num_threads: int,
    nvtx_range_name: str,
    ratio: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """THD packed dense score via per-batch BSHD SM90 launches.

    The SM90 dense score kernel is BSHD-native. This adapter preserves the
    public THD API by slicing each batch's packed Q/K/per-head tensors,
    launching the existing BSHD kernel, then copying results back into the
    packed ``(total_q, max_seqlen_k)`` output layout.
    """
    compute_capability = _get_device_capability()
    assert compute_capability == 9, f"SM90 kernel on compute capability {compute_capability}"
    assert ratio >= 1, f"ratio must be >= 1, got {ratio}"
    assert q.ndim == 3 and kv.ndim == 3 and weights_or_lse.ndim == 2
    q, kv, weights_or_lse = _validate_and_prepare_common(q, kv, weights_or_lse, is_index_scores)
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()
    assert cu_seqlens_q.is_cuda and cu_seqlens_k.is_cuda

    total_q = q.shape[0]
    if max_seqlen_k is None:
        k_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = int(k_lens.max().item())
    if out is None:
        out = torch.zeros((total_q, int(max_seqlen_k)), dtype=torch.float32, device=q.device)
    else:
        out = out if out.is_contiguous() else out.contiguous()
    if denom_out is None:
        denom_out = torch.empty((total_q,), dtype=torch.float32, device=q.device)
    else:
        denom_out = denom_out if denom_out.is_contiguous() else denom_out.contiguous()

    cu_q_host = cu_seqlens_q.detach().cpu().tolist()
    cu_k_host = cu_seqlens_k.detach().cpu().tolist()
    with torch.cuda.nvtx.range(nvtx_range_name + "_thd"):
        for b in range(len(cu_q_host) - 1):
            qs, qe = cu_q_host[b], cu_q_host[b + 1]
            ks, ke = cu_k_host[b], cu_k_host[b + 1]
            sq_b = qe - qs
            sk_b = ke - ks
            if sq_b == 0 or sk_b == 0:
                continue
            q_b = q[qs:qe].unsqueeze(0).contiguous()
            kv_b = kv[ks:ke].unsqueeze(0).contiguous()
            per_head_bhs = weights_or_lse[qs:qe].unsqueeze(0).transpose(1, 2).contiguous()
            out_b, denom_b = _dense_score_recompute(
                q_b,
                kv_b,
                per_head_bhs,
                is_index_scores=is_index_scores,
                softmax_scale=softmax_scale,
                out=None,
                denom_out=None,
                num_threads=num_threads,
                nvtx_range_name=nvtx_range_name + "_bshd_slice",
                ratio=ratio,
                current_stream=current_stream,
            )
            out[qs:qe, :sk_b].copy_(out_b[0])
            denom_out[qs:qe].copy_(denom_b[0])
            if sk_b < out.shape[1]:
                out[qs:qe, sk_b:].fill_(float("-inf"))
    return out, denom_out


_dense_score_recompute.compile_cache = {}


def dense_indexer_score_recompute(
    q_indexer: torch.Tensor,  # BSHD (bs, seqlen_q, n_heads_q, head_dim) or THD (total_q, n_heads_q, head_dim)
    k_indexer: torch.Tensor,  # BSHD (bs, seqlen_k, n_heads_kv, head_dim) or THD (total_k, n_heads_kv, head_dim)
    weights: torch.Tensor,  # BSH (bs, seqlen_q, n_heads_q) or TH (total_q, n_heads_q)
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    num_threads: int = _DENSE_NUM_THREADS,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense indexer backward score over full KV:
        S[b,q,t] = sm_scale * sum_h [ReLU(Q_h · K_t^T) · W_{b,q,h}]
        denom    = logsumexp(S, dim=-1)
    Returns (scores, lse_denom).

    sm_scale is applied to the fp32 head-reduced score inside the kernel
    (preserves precision vs pre-multiplying onto bf16 weights on the host).
    Defaults to 1.0 (identity).
    """
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("THD requires both cu_seqlens_q and cu_seqlens_k")
        return _dense_score_recompute_varlen(
            q_indexer,
            k_indexer,
            weights,
            is_index_scores=True,
            softmax_scale=sm_scale,
            out=out,
            denom_out=denom_out,
            num_threads=num_threads,
            nvtx_range_name="dense_indexer_score_recompute",
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            current_stream=current_stream,
        )
    w_bhs = weights.transpose(1, 2).contiguous()
    return _dense_score_recompute(
        q_indexer,
        k_indexer,
        w_bhs,
        is_index_scores=True,
        softmax_scale=sm_scale,
        out=out,
        denom_out=denom_out,
        num_threads=num_threads,
        nvtx_range_name="dense_indexer_score_recompute",
        ratio=ratio,
        current_stream=current_stream,
    )


def dense_attn_score_recompute(
    q_attn: torch.Tensor,  # BSHD (bs, seqlen_q, n_heads_q, head_dim) or THD (total_q, n_heads_q, head_dim)
    k_attn: torch.Tensor,  # BSHD (bs, seqlen_k, n_heads_kv, head_dim) or THD (total_k, n_heads_kv, head_dim)
    lse: torch.Tensor,  # BSH (bs, seqlen_q, n_heads_q) or TH (total_q, n_heads_q) float32
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    num_threads: int = _DENSE_NUM_THREADS,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense attention backward score over full KV:
        P[b,q,h,t] = exp(Q_h · K_t^T · scale − LSE[b,q,h])
        S[b,q,t]   = sum_h P[b,q,h,t]
        denom      = sum(S, dim=-1)   (L1-norm)
    Returns (scores, l1norm_denom).
    """
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("THD requires both cu_seqlens_q and cu_seqlens_k")
        return _dense_score_recompute_varlen(
            q_attn,
            k_attn,
            lse,
            is_index_scores=False,
            softmax_scale=softmax_scale,
            out=out,
            denom_out=denom_out,
            num_threads=num_threads,
            nvtx_range_name="dense_attn_score_recompute",
            ratio=ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            current_stream=current_stream,
        )
    lse_bhs = lse.transpose(1, 2).contiguous()
    return _dense_score_recompute(
        q_attn,
        k_attn,
        lse_bhs,
        is_index_scores=False,
        softmax_scale=softmax_scale,
        out=out,
        denom_out=denom_out,
        num_threads=num_threads,
        nvtx_range_name="dense_attn_score_recompute",
        ratio=ratio,
        current_stream=current_stream,
    )


__all__ = [
    "sparse_indexer_score_recompute",
    "sparse_attn_score_recompute",
    "dense_indexer_score_recompute",
    "dense_attn_score_recompute",
    "to_cute_tensor",
    "maybe_contiguous",
    "torch2cute_dtype_map",
]
