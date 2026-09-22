# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Indexer Forward Interface — CuTe DSL backend.

Wraps IndexerForwardSm100 (DSL kernel) with compile caching and
torch.Tensor ↔ cute.Tensor conversion. The dense path binds the caller's
``out`` view directly (no staging buffer, no copy-back) and compiles the
unused LSE slot out.
"""

from __future__ import annotations

from typing import Optional

import torch

import cutlass
import cutlass.cute as cute

from .indexer_fwd_sm100 import IndexerForwardSm100
from .indexer_fwd_sm100_mxfp8 import IndexerForwardSm100Mxfp8
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import (
    ceil_div as _ceil_div,
    maybe_contiguous as _maybe_contiguous,
    resolve_stream,
    torch_stream_context as _torch_stream_context,
    validate_q_causal_offsets,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor as _to_cute_tensor


def _packed_mxfp8_scale_shape(
    *,
    bs: int,
    seqlen: int,
    n_heads_kv: int,
    sf_groups: int,
    pack_q_heads: int = 1,
) -> tuple[int, int, int]:
    mn = seqlen * pack_q_heads
    return (
        bs * n_heads_kv,
        _ceil_div(mn, 128) * 128,
        _ceil_div(sf_groups, 4) * 4,
    )


def _validate_thd_mxfp8_scale_contract(
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor],
    cu_seqlens_k_scale_padded: Optional[torch.Tensor],
    *,
    bs: int,
    n_heads_kv: int,
    sf_groups: int,
    device: torch.device,
) -> None:
    if cu_seqlens_q_scale_padded is None or cu_seqlens_k_scale_padded is None:
        raise ValueError("THD MXFP8 requires Q/K scale padded cu_seqlens")

    for prefix, name in (
        (cu_seqlens_q_scale_padded, "cu_seqlens_q_scale_padded"),
        (cu_seqlens_k_scale_padded, "cu_seqlens_k_scale_padded"),
    ):
        if prefix.dtype != torch.int32 or prefix.ndim != 1 or prefix.shape[0] != bs + 1 or not prefix.is_contiguous() or prefix.device != device:
            raise ValueError(f"{name} must be contiguous int32 shape ({bs + 1},) on {device}")

    sf_padded = _ceil_div(sf_groups, 4) * 4
    for scale, name in ((q_scale, "q_scale"), (k_scale, "k_scale")):
        if scale.ndim != 3:
            raise ValueError(f"{name} must be a 3D packed scale tensor")
        if scale.shape[0] != n_heads_kv or scale.shape[1] % 128 != 0 or scale.shape[2] != sf_padded:
            raise ValueError(f"THD {name} must have shape ({n_heads_kv}, multiple_of_128, " f"{sf_padded}), got {tuple(scale.shape)}")
        if scale.device != device:
            raise ValueError("q_scale/k_scale must be on the same device as q/k")


def _validate_indexer_qhead_per_kv_head(qhead_per_kv_head: int, precision: str) -> None:
    supported = (32, 64)
    if qhead_per_kv_head not in supported:
        raise ValueError(f"precision={precision!r} indexer requires " f"qhead_per_kv_head=32 or 64, got {qhead_per_kv_head}")


# The score kernel binds ``out`` through a 16-byte-aligned dynamic layout, so
# the fp32 row stride must be a multiple of 4 elements (and cover ceil4(S_k)).
TMA_ALIGN_ELEMS = 4


def padded_seqlen_k(seqlen_k: int) -> int:
    return _ceil_div(int(seqlen_k), TMA_ALIGN_ELEMS) * TMA_ALIGN_ELEMS


def _validate_out_view(
    out: torch.Tensor,
    out_shape: tuple[int, ...],
    seqlen_k_dim: int,
    device: torch.device,
) -> None:
    seqlen_k_padded = padded_seqlen_k(seqlen_k_dim)
    if out.dtype != torch.float32 or not out.is_cuda or out.device != device:
        raise ValueError(f"out must be a float32 CUDA tensor on {device}, got {out.dtype} on {out.device}")
    if tuple(out.shape) != out_shape:
        raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}")
    strides = tuple(out.stride())
    if (
        strides[-1] != 1
        or strides[-2] % TMA_ALIGN_ELEMS != 0
        or strides[-2] < seqlen_k_padded
        or any(s % TMA_ALIGN_ELEMS for s in strides[:-2])
        or out.data_ptr() % 16
    ):
        padded_shape = (*out_shape[:-1], seqlen_k_padded)
        raise ValueError(
            f"out must be a view with a {seqlen_k_padded}-element (16-byte-aligned) row stride; "
            f"allocate {padded_shape} fp32 and pass out[..., :{seqlen_k_dim}]; got strides {strides}"
        )


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
    q_causal_offsets: Optional[torch.Tensor] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    is_compressed_logits: bool = False,
    topk: int = 0,
    microbatch_rows: int = -1,
    topk_indices_global: bool = True,
    cand_buffer: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    return_softmax: Optional[bool] = None,
    softmax_out: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    current_stream=None,
):
    """
    Indexer QK forward pass using CuTe DSL kernel.

    Computes S_sum = sm_scale * sum_h [(Q @ K^T).relu() * W] with a
    ratio causal mask against compressed-KV positions. With
    ``is_compressed_logits=True``, writes only causal-valid scores to compact
    scratch storage and returns ``(topk_indices, topk_logits, topk_softmax[, lse])``
    by default without materializing the dense score tensor. Pass
    ``return_softmax=False`` to omit the fused top-k softmax.
    ``deterministic=True`` makes exact-value ties at the K-th boundary select
    the smallest local KV indices; the selected set is reproducible while slot
    order remains unspecified.
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
        out: optional dense-score output (must be ``None`` for the compressed
             path). BSHD: ``(bs, seqlen_q, seqlen_k)``; THD:
             ``(total_q, max_seqlen_k)`` with local-K columns. ``None``
             allocates on ``current_stream`` and returns a contiguous result.
             A caller ``out`` whose row stride is a multiple of 4 fp32 elements
             and at least ``ceil4(seqlen_k)`` (e.g. ``buf[..., :seqlen_k]`` of a
             ``(..., ceil4(seqlen_k))`` allocation) is bound directly; any other
             layout of the logical shape is staged and copied back (identity
             kept). Strided q/k/w/q_scale/k_scale are made contiguous here.
             ``_indexer_fwd_bound`` is the strict, allocation-free entry.
        sm_scale: scalar applied to fp32 score post head-reduce; default 1.0
        precision: ``"bf16"`` for the existing BF16 kernel or ``"mxfp8"``
            for the SM100 MXFP8 kernel.
        q_scale: blockscaled-packed E8M0 scale tensor for MXFP8 Q.
        k_scale: blockscaled-packed E8M0 scale tensor for MXFP8 K.
        cu_seqlens_q_scale_padded: required THD MXFP8 prefix for scale storage.
            Values are consumed as provided; each Q span must cover its logical
            sequence, and ``span * qhead_per_kv_head`` must be a multiple of
            128 packed-MN rows. Any span satisfying those requirements is valid.
        cu_seqlens_k_scale_padded: required THD MXFP8 prefix for scale storage.
            Values are consumed as provided; each K span must cover its logical
            sequence and be a multiple of 128 tokens. Any span satisfying those
            requirements is valid. Both scale prefixes must be ``None`` for BSHD.
        sf_vec_size: scale vector size for MXFP8, currently must be 32.
        q_causal_offsets: optional int32 CUDA tensor of shape ``(batch,)``.
            Each entry is the global uncompressed token index for local q[0].

    Returns:
        S_sum: BSHD ``(bs, seqlen_q, seqlen_k)`` or THD
               ``(total_q, max_seqlen_k)`` [FP32]
    """
    current_stream = resolve_stream(current_stream)
    precision = precision.lower()
    if precision not in ("bf16", "mxfp8"):
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")
    if num_threads != 384:
        raise ValueError(f"SM100 indexer_fwd only supports num_threads=384, got {num_threads}")
    if q_stage != 2:
        raise ValueError(f"SM100 indexer_fwd only supports q_stage=2, got {q_stage}")
    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio}")

    if is_compressed_logits:
        if topk <= 0 or topk > 2048:
            raise ValueError("is_compressed_logits=True requires topk in (0, 2048], " f"got {topk}")
        if out is not None:
            raise ValueError("out is a dense-score buffer and cannot be used with is_compressed_logits=True")
        if kv_stage != 4:
            raise ValueError(f"SM100 compressed indexer forward only supports kv_stage=4, got {kv_stage}")
        from ._compressed_top_k_sm100 import (
            _indexer_fwd_compress_topk_thd,
            indexer_fwd_compress_topk,
        )

        # All eager allocations, copies, both kernel launches, and the optional
        # local-to-global conversion must use the caller-selected stream.
        # The compressed path is a torch-op layer (no APIBase engine), so it
        # keeps the strided-input .contiguous() convenience.
        with _torch_stream_context(current_stream):
            q, k, w = [_maybe_contiguous(t, current_stream) for t in (q, k, w)]
            q_scale = _maybe_contiguous(q_scale, current_stream)
            k_scale = _maybe_contiguous(k_scale, current_stream)
            if cu_seqlens_q is not None or cu_seqlens_k is not None:
                if cu_seqlens_q is None or cu_seqlens_k is None:
                    raise ValueError("THD compressed-logits top-k requires both cu_seqlens_q and cu_seqlens_k")
                return _indexer_fwd_compress_topk_thd(
                    q,
                    k,
                    w,
                    topk,
                    ratio=ratio,
                    qhead_per_kv_head=qhead_per_kv_head,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                    sm_scale=sm_scale,
                    microbatch_rows=microbatch_rows,
                    precision=precision,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
                    cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
                    sf_vec_size=sf_vec_size,
                    topk_indices_global=topk_indices_global,
                    cand_buffer=cand_buffer,
                    out_indices=out_indices,
                    out_logits=out_logits,
                    cand_batch_offsets=cand_batch_offsets,
                    q_causal_offsets=q_causal_offsets,
                    return_softmax=return_softmax,
                    softmax_out=softmax_out,
                    return_lse=return_lse,
                    lse_out=lse_out,
                    deterministic=deterministic,
                )
            else:
                if cand_batch_offsets is not None:
                    raise ValueError("cand_batch_offsets is only used by THD compressed top-k; " "BSHD computes its candidate offsets internally")
                return indexer_fwd_compress_topk(
                    q,
                    k,
                    w,
                    topk,
                    ratio=ratio,
                    qhead_per_kv_head=qhead_per_kv_head,
                    m_block_size=m_block_size,
                    n_block_size=n_block_size,
                    sm_scale=sm_scale,
                    microbatch_rows=microbatch_rows,
                    precision=precision,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    sf_vec_size=sf_vec_size,
                    topk_indices_global=topk_indices_global,
                    cand_buffer=cand_buffer,
                    out_indices=out_indices,
                    out_logits=out_logits,
                    return_lse=return_lse,
                    lse_out=lse_out,
                    return_softmax=return_softmax,
                    softmax_out=softmax_out,
                    q_causal_offsets=q_causal_offsets,
                    deterministic=deterministic,
                )

    # Dense convenience layer (torch-op semantics; ``IndexerForward`` binds
    # ``_indexer_fwd_bound`` directly and declines all of this in check_support).
    q, k, w = [_maybe_contiguous(t, current_stream) for t in (q, k, w)]
    q_scale = _maybe_contiguous(q_scale, current_stream)
    k_scale = _maybe_contiguous(k_scale, current_stream)
    out_shape, seqlen_k_dim = _dense_out_shape(q, k, cu_seqlens_q, max_seqlen_k)
    out_orig = out
    if out is not None:
        if out.dtype != torch.float32 or not out.is_cuda or out.device != q.device:
            raise ValueError(f"out must be a float32 CUDA tensor on {q.device}, got {out.dtype} on {out.device}")
        if tuple(out.shape) != out_shape:
            raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}")
        try:
            _validate_out_view(out, out_shape, seqlen_k_dim, q.device)
        except ValueError:
            out = None  # not bindable: staged through a padded buffer and copied back
    need_pad = padded_seqlen_k(seqlen_k_dim) != seqlen_k_dim
    if out is None:
        with _torch_stream_context(current_stream):
            out = torch.empty((*out_shape[:-1], padded_seqlen_k(seqlen_k_dim)), dtype=torch.float32, device=q.device)[..., :seqlen_k_dim]
    _indexer_fwd_bound(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        out=out,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        current_stream=current_stream,
    )
    return _finish_dense_out(out, out_orig, need_pad=need_pad, current_stream=current_stream)


def _dense_out_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    max_seqlen_k: Optional[int],
) -> tuple[tuple[int, ...], int]:
    if cu_seqlens_q is not None:
        if max_seqlen_k is None:
            raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")
        return (int(q.shape[0]), int(max_seqlen_k)), int(max_seqlen_k)
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"BSHD q/k must be 4D, got q {tuple(q.shape)} k {tuple(k.shape)}")
    return (int(q.shape[0]), int(q.shape[1]), int(k.shape[1])), int(k.shape[1])


def _finish_dense_out(
    out: torch.Tensor,
    out_orig: Optional[torch.Tensor],
    *,
    need_pad: bool,
    current_stream=None,
) -> torch.Tensor:
    """Copy a staged result back into the caller's ``out`` (identity kept) or make an allocated padded view contiguous."""
    with _torch_stream_context(current_stream):
        if out_orig is not None and out_orig is not out:
            out_orig.copy_(out)
            return out_orig
        if out_orig is None and need_pad:
            return out.contiguous()
    return out


def _indexer_fwd_bound(
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
    q_causal_offsets: Optional[torch.Tensor] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    current_stream=None,
) -> torch.Tensor:
    """Strict dense entry (what ``IndexerForward.execute`` calls): binds a validated padded-stride ``out``
    directly; never allocates, never copies, raises on strided inputs or an unbindable ``out``."""
    current_stream = resolve_stream(current_stream)
    precision = precision.lower()
    if precision not in ("bf16", "mxfp8"):
        raise ValueError(f"precision must be 'bf16' or 'mxfp8', got {precision!r}")
    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio}")

    if precision == "bf16":
        for tensor, name in ((q, "q"), (k, "k"), (w, "w")):
            assert tensor.dtype == torch.bfloat16, f"{name} must be bfloat16, got {tensor.dtype}"
            assert tensor.is_cuda, f"{name} must be on CUDA device"
        if q_scale is not None or k_scale is not None or cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
            raise ValueError("q_scale, k_scale, and scale padded cu_seqlens are only valid " "with precision='mxfp8'")
    else:
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
            raise TypeError("precision='mxfp8' requires q and k to be torch.float8_e4m3fn")
        if w.dtype != torch.bfloat16:
            raise TypeError("precision='mxfp8' requires w to be torch.bfloat16")
        if q_scale is None or k_scale is None:
            raise ValueError("precision='mxfp8' requires q_scale and k_scale")
        if q_scale.dtype != torch.float8_e8m0fnu or k_scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("precision='mxfp8' requires q_scale and k_scale to be " "torch.float8_e8m0fnu")
        if sf_vec_size != 32:
            raise ValueError("precision='mxfp8' currently requires sf_vec_size=32")

    # The kernel addresses these natively through a unit innermost stride;
    # IndexerForward.check_support() declines anything else (Rule 2 / R5).
    for tensor, name in ((q, "q"), (k, "k"), (w, "w"), (q_scale, "q_scale"), (k_scale, "k_scale")):
        if tensor is not None and tensor.stride(-1) != 1:
            raise ValueError(f"indexer_fwd reads {name} with a unit innermost stride natively; got strides {tuple(tensor.stride())}")

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
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")
        seqlen_q_dim = int(max_seqlen_q)
        seqlen_k_dim = int(max_seqlen_k)
        device = q.device
        out_shape = (total_q, seqlen_k_dim)
    else:
        bs, seqlen_q_dim, n_heads_q, head_dim = q.shape
        _, seqlen_k_dim, n_heads_kv, _ = k.shape
        device = q.device
        if seqlen_q_dim > seqlen_k_dim * ratio:
            raise ValueError(f"seqlen_q ({seqlen_q_dim}) must be <= seqlen_k * ratio " f"({seqlen_k_dim * ratio})")
        out_shape = (bs, seqlen_q_dim, seqlen_k_dim)

    if n_heads_kv != 1:
        raise ValueError("SM100 indexer forward currently requires n_heads_kv=1 (MQA); " f"got n_heads_kv={n_heads_kv}")
    expected_qhead_per_kv_head = n_heads_q // n_heads_kv
    if qhead_per_kv_head is None:
        qhead_per_kv_head = expected_qhead_per_kv_head
    elif qhead_per_kv_head != expected_qhead_per_kv_head:
        raise ValueError(f"qhead_per_kv_head ({qhead_per_kv_head}) must equal " f"n_heads_q // n_heads_kv ({expected_qhead_per_kv_head})")
    _validate_indexer_qhead_per_kv_head(qhead_per_kv_head, precision)

    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(bs), q.device, stream=current_stream)

    max_q_tokens_per_tile = 4 if head_dim == 128 and qhead_per_kv_head == 32 else 2
    if precision == "bf16" and m_block_size // qhead_per_kv_head > max_q_tokens_per_tile:
        if m_block_size == 128:
            m_block_size = qhead_per_kv_head * 2
        else:
            raise ValueError(
                f"SM100 indexer_fwd supports at most {max_q_tokens_per_tile} q tokens per tile; got "
                f"m_block_size={m_block_size}, qhead_per_kv_head={qhead_per_kv_head}"
            )
    if m_block_size % qhead_per_kv_head != 0:
        raise ValueError(f"m_block_size ({m_block_size}) must be divisible by " f"qhead_per_kv_head ({qhead_per_kv_head})")

    if precision == "mxfp8":
        assert q_scale is not None and k_scale is not None
        sf_groups = _ceil_div(head_dim, sf_vec_size)
        if is_varlen:
            _validate_thd_mxfp8_scale_contract(
                q_scale,
                k_scale,
                cu_seqlens_q_scale_padded,
                cu_seqlens_k_scale_padded,
                bs=bs,
                n_heads_kv=n_heads_kv,
                sf_groups=sf_groups,
                device=device,
            )
        else:
            if cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
                raise ValueError("BSHD MXFP8 requires scale padded cu_seqlens to be None")
            q_shape = _packed_mxfp8_scale_shape(
                bs=bs,
                seqlen=seqlen_q_dim,
                n_heads_kv=n_heads_kv,
                sf_groups=sf_groups,
                pack_q_heads=qhead_per_kv_head,
            )
            k_shape = _packed_mxfp8_scale_shape(
                bs=bs,
                seqlen=seqlen_k_dim,
                n_heads_kv=n_heads_kv,
                sf_groups=sf_groups,
            )
            if tuple(q_scale.shape) != q_shape:
                raise ValueError(f"q_scale packed shape must be {q_shape}, got {tuple(q_scale.shape)}")
            if tuple(k_scale.shape) != k_shape:
                raise ValueError(f"k_scale packed shape must be {k_shape}, got {tuple(k_scale.shape)}")

    if out is None:
        raise ValueError(
            f"indexer_fwd requires out: allocate {(*out_shape[:-1], padded_seqlen_k(seqlen_k_dim))} fp32 and pass "
            f"out[..., :{seqlen_k_dim}] (indexer_forward_wrapper allocates it for you)"
        )
    _validate_out_view(out, out_shape, seqlen_k_dim, device)

    if precision == "mxfp8":
        assert q_scale is not None and k_scale is not None
        compile_key = (
            "mxfp8",
            q.dtype,
            k.dtype,
            w.dtype,
            q_scale.dtype,
            k_scale.dtype,
            head_dim,
            qhead_per_kv_head,
            ratio,
            m_block_size,
            n_block_size,
            num_threads,
            is_varlen,
            sf_vec_size,
            q_causal_offsets is not None,
        )

        if compile_key not in _compile_cache:
            q_cute = _to_cute_tensor(q)
            k_cute = _to_cute_tensor(k)
            w_cute = _to_cute_tensor(w)
            q_scale_cute = _to_cute_tensor(q_scale)
            k_scale_cute = _to_cute_tensor(k_scale)
            out_cute = _to_cute_tensor(out)
            denom_cute = None  # compute_lse=False: dead slot, compiled out (R3)
            cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
            cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
            cu_q_scale_cute = _to_cute_tensor(cu_seqlens_q_scale_padded, leading_dim=0) if is_varlen else None
            cu_k_scale_cute = _to_cute_tensor(cu_seqlens_k_scale_padded, leading_dim=0) if is_varlen else None
            q_offsets_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

            kernel_obj = IndexerForwardSm100Mxfp8(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                k_block_size=64,
                kv_stage=24,
                ratio=ratio,
                is_varlen=is_varlen,
                sf_vec_size=sf_vec_size,
                compute_lse=False,
                is_compressed_logits=False,
            )

            scale_arg = cutlass.Float32(sm_scale)
            max_q_arg = cutlass.Int32(seqlen_q_dim)
            max_k_arg = cutlass.Int32(seqlen_k_dim)

            _compile_cache[compile_key] = cute.compile(
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

        with _torch_stream_context(current_stream):
            out.fill_(float("-inf"))
        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q_dim)
        max_k_arg = cutlass.Int32(seqlen_k_dim)
        with torch.cuda.nvtx.range("indexer_fwd_mxfp8_kernel"):
            _compile_cache[compile_key](
                q,
                k,
                w,
                q_scale,
                k_scale,
                out,
                None,
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
        return out

    head_dim_padded = (head_dim + 15) // 16 * 16
    k_block_size = 64 if head_dim_padded % 64 == 0 else head_dim_padded
    compile_key = (
        "bf16",
        q.dtype,
        head_dim,
        qhead_per_kv_head,
        ratio,
        m_block_size,
        n_block_size,
        k_block_size,
        kv_stage,
        num_threads,
        is_varlen,
        q_causal_offsets is not None,
    )

    if compile_key not in _compile_cache:
        q_cute = _to_cute_tensor(q)
        k_cute = _to_cute_tensor(k)
        w_cute = _to_cute_tensor(w)
        out_cute = _to_cute_tensor(out)
        denom_cute = None  # compute_lse=False: dead slot, compiled out (R3)
        cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        q_offsets_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None

        kernel_obj = IndexerForwardSm100(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv_head,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            k_block_size=k_block_size,
            kv_stage=kv_stage,
            ratio=ratio,
            is_varlen=is_varlen,
            compute_lse=False,
            is_compressed_logits=False,
        )

        scale_arg = cutlass.Float32(sm_scale)
        max_q_arg = cutlass.Int32(seqlen_q_dim)
        max_k_arg = cutlass.Int32(seqlen_k_dim)

        _compile_cache[compile_key] = cute.compile(
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

    # Init to -inf: skipped causal n-blocks and masked positions stay -inf
    with _torch_stream_context(current_stream):
        out.fill_(float("-inf"))
    scale_arg = cutlass.Float32(sm_scale)
    max_q_arg = cutlass.Int32(seqlen_q_dim)
    max_k_arg = cutlass.Int32(seqlen_k_dim)
    with torch.cuda.nvtx.range("indexer_fwd_kernel"):
        _compile_cache[compile_key](
            q,
            k,
            w,
            out,
            None,
            scale_arg,
            max_q_arg,
            max_k_arg,
            cu_seqlens_q if is_varlen else None,
            cu_seqlens_k if is_varlen else None,
            q_causal_offsets,
            current_stream,
        )
    return out
