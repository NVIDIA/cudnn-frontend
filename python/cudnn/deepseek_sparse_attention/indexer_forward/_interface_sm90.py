"""Indexer forward interface for the SM90 CuTe DSL backend."""

from __future__ import annotations

from typing import Optional

import torch

import cutlass
import cutlass.cute as cute

from .indexer_fwd_sm90 import IndexerForwardSm90
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    maybe_contiguous as _maybe_contiguous,
    resolve_stream,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
    to_cute_tensor as _to_cute_tensor,
)

_SUPPORTED_QHPKV = (32, 64)
_compile_cache: dict = {}

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _validate_common(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> None:
    assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
    assert k.dtype == torch.bfloat16, f"k must be bfloat16, got {k.dtype}"
    assert w.dtype == torch.bfloat16, f"w must be bfloat16, got {w.dtype}"
    assert q.is_cuda and k.is_cuda and w.is_cuda, "q, k, w must be CUDA tensors"


def indexer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    current_stream=None,
) -> torch.Tensor:
    """Indexer QK forward pass using the direct SM90 CuTe DSL port."""
    _validate_common(q, k, w)
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    if is_varlen_q != is_varlen_k:
        raise ValueError("THD input requires both cu_seqlens_q and cu_seqlens_k")
    is_varlen = is_varlen_q

    q = _maybe_contiguous(q)
    k = _maybe_contiguous(k)
    w = _maybe_contiguous(w)

    if is_varlen:
        assert cu_seqlens_q is not None and cu_seqlens_k is not None
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1, f"{name} must be 1D"
            assert t.is_cuda, f"{name} must be on CUDA"
            assert t.stride(0) == 1, f"{name} must be contiguous"
        assert q.ndim == 3, f"THD q must be 3D, got {q.ndim}D"
        assert k.ndim == 3, f"THD k must be 3D, got {k.ndim}D"
        assert w.ndim == 2, f"THD w must be 2D, got {w.ndim}D"
        total_q, n_heads_q, head_dim = q.shape
        _, n_heads_kv, head_dim_k = k.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        assert cu_seqlens_k.shape == (batch_size + 1,)
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")
        seqlen_q_dim = int(max_seqlen_q)
        seqlen_k_dim = int(max_seqlen_k)
        out_shape = (total_q, seqlen_k_dim)
    else:
        assert q.ndim == 4, f"q must be 4D BSHD, got {q.ndim}D"
        assert k.ndim == 4, f"k must be 4D BSHD, got {k.ndim}D"
        assert w.ndim == 3, f"w must be 3D BSH, got {w.ndim}D"
        batch_size, seqlen_q_dim, n_heads_q, head_dim = q.shape
        kb, seqlen_k_dim, n_heads_kv, head_dim_k = k.shape
        assert kb == batch_size, f"q batch ({batch_size}) != k batch ({kb})"
        assert w.shape == (
            batch_size,
            seqlen_q_dim,
            n_heads_q,
        ), f"w shape must be {(batch_size, seqlen_q_dim, n_heads_q)}, got {tuple(w.shape)}"
        if seqlen_q_dim > seqlen_k_dim * ratio:
            raise ValueError(f"seqlen_q ({seqlen_q_dim}) must be <= seqlen_k * ratio ({seqlen_k_dim * ratio})")
        out_shape = (batch_size, seqlen_q_dim, seqlen_k_dim)

    assert head_dim == head_dim_k, f"q head_dim ({head_dim}) != k head_dim ({head_dim_k})"
    assert head_dim == 128, f"head_dim must be 128, got {head_dim}"
    assert n_heads_kv == 1, f"SM90 direct fwd currently supports num_head_kv == 1, got {n_heads_kv}"
    assert n_heads_q % n_heads_kv == 0
    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    assert qhead_per_kv_head == n_heads_q // n_heads_kv
    assert qhead_per_kv_head in _SUPPORTED_QHPKV, f"qhead_per_kv_head must be one of {_SUPPORTED_QHPKV}, got {qhead_per_kv_head}"

    if out is None:
        out = torch.empty(out_shape, dtype=torch.float32, device=q.device)
    else:
        assert out.shape == out_shape, f"out must have shape {out_shape}, got {tuple(out.shape)}"
        assert out.dtype == torch.float32 and out.is_cuda
        assert out.is_contiguous(), "out must be contiguous"

    # SM90 TMA store descriptors require the row stride in bytes to be 16B aligned.
    # The old C++ BSHD path uses TMA store when that descriptor is legal; otherwise
    # we keep the same warp-scalar writeback used by its varlen path.
    use_tma_store = (not is_varlen) and (out.stride(1) * out.element_size()) % 16 == 0

    compile_key = (
        q.dtype,
        int(head_dim),
        int(qhead_per_kv_head),
        int(ratio),
        bool(is_varlen),
        bool(use_tma_store),
    )
    current_stream = resolve_stream(current_stream)
    if compile_key not in _compile_cache:
        q_cute = _to_cute_tensor(q)
        k_cute = _to_cute_tensor(k)
        w_cute = _to_cute_tensor(w)
        out_cute = _to_cute_tensor(out)
        cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        kernel_obj = IndexerForwardSm90(
            torch2cute_dtype_map[q.dtype],
            head_dim=int(head_dim),
            qhead_per_kvhead=int(qhead_per_kv_head),
            ratio=int(ratio),
            is_varlen=is_varlen,
            use_tma_store=use_tma_store,
        )
        _compile_cache[compile_key] = cute.compile(
            kernel_obj,
            q_cute,
            k_cute,
            w_cute,
            out_cute,
            cutlass.Int32(int(n_heads_kv)),
            cutlass.Int32(int(seqlen_q_dim)),
            cutlass.Int32(int(seqlen_k_dim)),
            cutlass.Float32(float(sm_scale)),
            cu_q_cute,
            cu_k_cute,
            current_stream,
            options=compile_options(),
        )

    out.fill_(float("-inf"))
    with torch.cuda.nvtx.range("indexer_fwd_kernel_sm90_direct_dsl"):
        _compile_cache[compile_key](
            q,
            k,
            w,
            out,
            cutlass.Int32(int(n_heads_kv)),
            cutlass.Int32(int(seqlen_q_dim)),
            cutlass.Int32(int(seqlen_k_dim)),
            cutlass.Float32(float(sm_scale)),
            cu_seqlens_q if is_varlen else None,
            cu_seqlens_k if is_varlen else None,
            current_stream,
        )
    return out


__all__ = ["indexer_fwd"]
