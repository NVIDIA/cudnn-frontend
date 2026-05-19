"""
Indexer Forward Interface — CuTe DSL backend.

Wraps IndexerForwardSm100 (DSL kernel) with compile caching, TMA padding,
and torch.Tensor ↔ cute.Tensor conversion.
"""

from __future__ import annotations

from typing import Optional

import torch

import cutlass
import cutlass.cute as cute

from .indexer_fwd_sm100 import IndexerForwardSm100
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_major as _get_device_capability,
    maybe_contiguous as _maybe_contiguous,
    resolve_stream,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor as _to_cute_tensor

# Module-level compile cache
_compile_cache: dict = {}


def indexer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Indexer QK forward pass using CuTe DSL kernel.

    Computes S_sum = sm_scale * sum_h [(Q @ K^T).relu() * W] with
    bottom-right aligned ratio causal mask.
    sm_scale is applied to the fp32 head-reduced score inside the kernel
    (higher precision than pre-multiplying onto bf16 W on the host).

    Args:
        q: BSHD ``(bs, seqlen_q, n_heads_q, head_dim)`` or THD
           ``(total_q, n_heads_q, head_dim)`` [BF16]
        k: BSHD ``(bs, seqlen_k, n_heads_kv, head_dim)`` or THD
           ``(total_k, n_heads_kv, head_dim)`` [BF16]
        w: BSH ``(bs, seqlen_q, n_heads_q)`` or TH ``(total_q, n_heads_q)`` [BF16]
        ratio: compression ratio (int), default 4
        qhead_per_kv_head: auto inferred if None
        out: optional output tensor. BSHD: ``(bs, seqlen_q, seqlen_k)``.
             THD: ``(total_q, max_seqlen_k)`` with local-K columns.
        sm_scale: scalar applied to fp32 score post head-reduce; default 1.0

    Returns:
        S_sum: BSHD ``(bs, seqlen_q, seqlen_k)`` or THD
               ``(total_q, max_seqlen_k)`` [FP32]
    """
    q, k, w = [_maybe_contiguous(t) for t in (q, k, w)]

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    assert is_varlen_q == is_varlen_k, "THD input requires both cu_seqlens_q and cu_seqlens_k"
    is_varlen = is_varlen_q
    if is_varlen:
        assert cu_seqlens_q is not None and cu_seqlens_k is not None, "THD input requires both cu_seqlens_q and cu_seqlens_k"
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1, f"{name} must be 1D"
            assert t.stride(0) == 1, f"{name} must be contiguous"
            assert t.is_cuda, f"{name} must be on CUDA device"
        assert q.ndim == 3, f"THD q must be 3D (total_q, n_heads_q, head_dim), got {q.ndim}D"
        assert k.ndim == 3, f"THD k must be 3D (total_k, n_heads_kv, head_dim), got {k.ndim}D"
        assert w.ndim == 2, f"THD w must be 2D (total_q, n_heads_q), got {w.ndim}D"
        total_q, n_heads_q, head_dim = q.shape
        total_k, n_heads_kv, head_dim_k = k.shape
        bs = cu_seqlens_q.shape[0] - 1
        assert cu_seqlens_k.shape == (bs + 1,), "cu_seqlens_k must have shape (batch_size + 1,)"
        assert cu_seqlens_q.shape == (bs + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
        assert head_dim == head_dim_k, f"q head_dim ({head_dim}) != k head_dim ({head_dim_k})"
        assert w.shape == (total_q, n_heads_q), f"THD w shape must be ({total_q}, {n_heads_q}), got {tuple(w.shape)}"
        if qhead_per_kv_head is None:
            qhead_per_kv_head = n_heads_q // n_heads_kv
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")
        seqlen_q_dim = int(max_seqlen_q)
        seqlen_k_dim = int(max_seqlen_k)
        device = q.device
        out_shape = (total_q, seqlen_k_dim)
        out_buf_shape = None
    else:
        if qhead_per_kv_head is None:
            qhead_per_kv_head = q.shape[2] // k.shape[2]

        bs, seqlen_q_dim, n_heads_q, head_dim = q.shape
        _, seqlen_k_dim, n_heads_kv, _ = k.shape
        device = q.device
        if seqlen_q_dim > seqlen_k_dim * ratio:
            raise ValueError(f"seqlen_q ({seqlen_q_dim}) must be <= seqlen_k * ratio " f"({seqlen_k_dim * ratio})")
        out_shape = (bs, seqlen_q_dim, seqlen_k_dim)
        out_buf_shape = None

    # TMA S2G requires globalStride aligned to 16 bytes.
    # For FP32, seqlen_k must be a multiple of 4 elements (4 × 4B = 16B).
    TMA_ALIGN_ELEMS = 4
    seqlen_k_padded = (seqlen_k_dim + TMA_ALIGN_ELEMS - 1) // TMA_ALIGN_ELEMS * TMA_ALIGN_ELEMS
    need_pad = seqlen_k_padded != seqlen_k_dim
    out_orig = out

    if need_pad:
        out_buf_shape = (total_q, seqlen_k_padded) if is_varlen else (bs, seqlen_q_dim, seqlen_k_padded)
        out_buf = torch.empty(out_buf_shape, dtype=torch.float32, device=device)
        out = out_buf[:, :seqlen_k_dim] if is_varlen else out_buf[:, :, :seqlen_k_dim]
    elif out is None:
        out = torch.empty(out_shape, dtype=torch.float32, device=device)
    else:
        assert out.shape == out_shape, f"out must have shape {out_shape}, got {tuple(out.shape)}"
        assert out.dtype == torch.float32 and out.is_cuda

    # sm_scale participates in compile_key because cutlass.Float32 is baked
    # as a constant into the compiled kernel, so distinct scale values need
    # distinct cache entries.
    compile_key = (
        q.dtype,
        head_dim,
        qhead_per_kv_head,
        ratio,
        m_block_size,
        n_block_size,
        num_threads,
        q_stage,
        kv_stage,
        float(sm_scale),
        is_varlen,
        seqlen_q_dim,
        seqlen_k_dim,
    )

    if compile_key not in _compile_cache:
        q_cute = _to_cute_tensor(q)
        k_cute = _to_cute_tensor(k)
        w_cute = _to_cute_tensor(w)
        out_cute = _to_cute_tensor(out)
        cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None

        kernel_obj = IndexerForwardSm100(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv_head,
            ratio=ratio,
            is_varlen=is_varlen,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            kv_stage=kv_stage,
        )

        current_stream = resolve_stream()
        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q_dim)
        max_k_arg = cutlass.Int32(seqlen_k_dim)

        _compile_cache[compile_key] = cute.compile(
            kernel_obj,
            q_cute,
            k_cute,
            w_cute,
            out_cute,
            n_heads_kv,
            max_q_arg,
            max_k_arg,
            scale_arg,
            cu_q_cute,
            cu_k_cute,
            current_stream,
            options=compile_options(),
        )

    # Init to -inf: skipped causal n-blocks and masked positions stay -inf
    out.fill_(float("-inf"))
    current_stream = resolve_stream()
    scale_arg = cutlass.Float32(sm_scale)
    max_q_arg = cutlass.Int32(seqlen_q_dim)
    max_k_arg = cutlass.Int32(seqlen_k_dim)
    with torch.cuda.nvtx.range("indexer_fwd_kernel"):
        _compile_cache[compile_key](
            q,
            k,
            w,
            out,
            n_heads_kv,
            max_q_arg,
            max_k_arg,
            scale_arg,
            cu_seqlens_q if is_varlen else None,
            cu_seqlens_k if is_varlen else None,
            current_stream,
        )

    if out_orig is not None and out.data_ptr() != out_orig.data_ptr():
        out_orig.copy_(out)
        return out_orig
    return out
