# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch training composition over the native DSA APIBase wrappers.

This allocating semantic adapter deliberately lives above the kernel execute
contract. Metadata normalization and optional H16/H32-to-H64 forward padding
are explicit parts of this API; no external forward provider is loaded.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Callable, Optional

import torch
from cudnn.api_base import TupleDict


def _validate_contract(q, kv, indices, attn_sink, topk_length, softmax_scale):
    for name, tensor in (("q", q), ("kv", kv), ("indices", indices), ("attn_sink", attn_sink), ("topk_length", topk_length)):
        if tensor is None and name in ("attn_sink", "topk_length"):
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on Q's device {q.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2:
        raise ValueError("expected Q [S_q,H,D], KV [S_kv,D], indices [S_q,K]")
    sq, heads, dim = q.shape
    if sq <= 0 or kv.shape[0] <= 0 or indices.shape[1] <= 0:
        raise ValueError("sequence extents and physical Top-K must be positive")
    if heads not in (16, 32, 64, 128) or dim not in (512, 576) or (heads, dim) == (128, 576):
        raise NotImplementedError("native training supports H16/H32/H64 D512/D576 and H128 D512")
    if kv.shape[1] != dim or indices.shape[0] != sq:
        raise ValueError("KV dimension and index row count must match Q")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or indices.dtype != torch.int32:
        raise TypeError("Q/KV must be BF16 and indices must be INT32")
    if attn_sink is not None and (attn_sink.shape != (heads,) or attn_sink.dtype != torch.float32):
        raise ValueError("attn_sink must be FP32 [H]")
    if topk_length is not None and (topk_length.shape != (sq,) or topk_length.dtype != torch.int32):
        raise ValueError("topk_length must be INT32 [S_q]")
    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, Real)):
        raise TypeError("softmax_scale must be a host real scalar or None")
    scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) != (10, 0):
        raise NotImplementedError("native sparse attention training currently requires SM100")
    from cudnn.frost.buffers import cutedsl_requirement_error

    error = cutedsl_requirement_error("native DSA training")
    if error:
        raise RuntimeError(error)
    return scale


def _normalize_sparse_index_slots(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    s_kv: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mask unsafe slots and bound lengths without changing slot positions."""

    topk = indices.shape[1]
    valid = (indices >= 0) & (indices < s_kv)
    bounded_length = None
    if topk_length is not None:
        bounded_length = topk_length.clamp(min=0, max=topk)
        positions = torch.arange(topk, dtype=torch.int32, device=indices.device).unsqueeze(0)
        valid = valid & (positions < bounded_length.unsqueeze(1))
    return torch.where(valid, indices, -1), bounded_length


def _normalize_cudnn_sparse_metadata(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    s_kv: int,
    *,
    trusted_compact_metadata: bool = False,
    _compactify: Optional[Callable[[torch.Tensor], Any]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalize metadata for the shared native forward/backward contract.

    The semantic API treats negative or ``>= S_kv`` indices as invalid. Current
    cuDNN DSA backward only guards negative sentinels without lengths, and it
    assumes every entry in a compact active prefix is valid when lengths are
    present.  Normalize high sentinels to ``-1`` in both cases.  For the
    length form, also mask the inactive suffix, compact valid entries to the
    front, and derive a new length.  This is asynchronous GPU work: there is
    no device-to-host validation or synchronization in the adapter.

    Out-of-contract length values are clamped to the physical ``[0, K]``
    envelope before masking.  An asynchronous device assert would poison the
    CUDA context on failure, while a strict host check would synchronize the
    hot path; clamping gives the downstream kernels a memory-safe contract.

    ``trusted_compact_metadata`` is the explicit zero-work normalization
    alternative for a producer that already satisfies the narrower cuDNN
    backward contract.  Later launch preparation can still pad an unaligned
    Top-K or make metadata contiguous where the downstream kernel requires it.
    """

    if trusted_compact_metadata:
        # This is an explicit caller contract, not a property inferred from
        # device data.  Inspecting the values here would either synchronize or
        # launch the same metadata kernels this fast path exists to avoid.
        return indices, topk_length

    normalized, _ = _normalize_sparse_index_slots(indices, topk_length, s_kv)
    if topk_length is None:
        return normalized, None

    if _compactify is None:
        # Lazy import preserves the optional CuTe DSL boundary at module import.
        from .indexer_top_k.api import compactify_wrapper

        _compactify = compactify_wrapper
    compact = _compactify(normalized)
    return compact["indices"], compact["topk_length"]


def _pad_indices(indices):
    topk = indices.shape[1]
    padded = (topk + 127) // 128 * 128
    if padded == topk:
        return indices
    return torch.nn.functional.pad(indices, (0, padded - topk), value=-1)


def _native_forward(q, kv, indices, sink, lengths, scale):
    from .sparse_attention_forward.api import sparse_attention_forward_wrapper

    heads = q.shape[1]
    launch_q, launch_sink = q, sink
    if heads < 64:
        launch_q = torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
        launch_sink = torch.nn.functional.pad(sink, (0, 64 - heads), value=-float("inf"))
    result = sparse_attention_forward_wrapper(
        launch_q,
        kv,
        indices,
        attn_sink=launch_sink,
        topk_length=lengths,
        softmax_scale=scale,
    )
    if heads < 64:
        return tuple(result[key][:, :heads].contiguous() for key in ("out", "max_logits", "lse"))
    return result["out"], result["max_logits"], result["lse"]


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, sink, lengths, scale, trusted):
        with torch.cuda.device(q.device):
            safe_indices, safe_lengths = _normalize_cudnn_sparse_metadata(
                indices,
                lengths,
                kv.shape[0],
                trusted_compact_metadata=trusted,
            )
            safe_indices = _pad_indices(safe_indices)
            effective_sink = sink if sink is not None else torch.full((q.shape[1],), -float("inf"), dtype=torch.float32, device=q.device)
            out, max_logits, lse = _native_forward(q, kv, safe_indices, effective_sink, safe_lengths, scale)
        ctx.scale = scale
        ctx.has_lengths = safe_lengths is not None
        if any(ctx.needs_input_grad[i] for i in (0, 1, 3)):
            tensors = [q, kv, out, lse, effective_sink, safe_indices]
            if safe_lengths is not None:
                tensors.append(safe_lengths)
            ctx.save_for_backward(*tensors)
        ctx.mark_non_differentiable(max_logits, lse)
        ctx.set_materialize_grads(False)
        return out, max_logits, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dout, _dmax, _dlse):
        if dout is None:
            return (None,) * 7
        q, kv, out, lse, sink, indices, *rest = ctx.saved_tensors
        from .sparse_attention_backward.api import sparse_attention_backward_wrapper

        with torch.cuda.device(q.device):
            result = sparse_attention_backward_wrapper(
                q,
                kv,
                out,
                dout.contiguous(),
                lse,
                sink,
                indices,
                softmax_scale=ctx.scale,
                topk_length=rest[0] if ctx.has_lengths else None,
            )
        needs = ctx.needs_input_grad
        return (result["dq"] if needs[0] else None, result["dkv"] if needs[1] else None, None, result["d_sink"] if needs[3] else None, None, None, None)


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    trusted_compact_metadata: bool = False,
) -> TupleDict:
    """Native BF16 MQA sparse attention with first-order PyTorch autograd.

    Returns ``out``, ``max_logits`` and KV-only natural-log ``lse``. Only
    ``out`` is differentiable, with respect to Q, KV and optional sink.
    Inputs are contiguous on one SM100 device; launch uses the current
    PyTorch stream. H16/H32 forward is padded to H64, then sliced back.

    Safe mode masks invalid/OOB indices and inactive suffixes; supplied lengths
    are clamped to [0,K] and valid entries compacted before both passes.
    ``trusted_compact_metadata=True`` skips that work only when the producer
    guarantees bounded valid active prefixes, negative inactive suffixes, and
    lengths in [0,K]. Without lengths, all nonnegative indices must be in range.
    Top-K is padded to a 128-slot launch envelope. This allocating composition
    does not change the lower-level APIBase execute contract.
    """
    if not isinstance(trusted_compact_metadata, bool):
        raise TypeError("trusted_compact_metadata must be a bool")
    scale = _validate_contract(q, kv, indices, attn_sink, topk_length, softmax_scale)
    out, max_logits, lse = _SparseAttention.apply(q, kv, indices, attn_sink, topk_length, scale, trusted_compact_metadata)
    return TupleDict(out=out, max_logits=max_logits, lse=lse)


@torch.no_grad()
def sparse_attention_score_recompute(
    q: torch.Tensor,
    kv: torch.Tensor,
    lse: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Return normalized attention targets aligned to original index slots.

    LSE is the KV-only natural-log statistic from native sparse attention.
    Invalid/inactive slots return zero. No gradients are attached to targets.
    """
    scale = _validate_contract(q, kv, indices, None, topk_length, softmax_scale)
    if not isinstance(lse, torch.Tensor) or lse.shape != q.shape[:2] or lse.dtype != torch.float32 or lse.device != q.device or not lse.is_contiguous():
        raise ValueError("lse must be contiguous FP32 [S_q,H] on Q's device")
    from .score_recompute.api import sparse_attn_score_recompute_wrapper

    with torch.cuda.device(q.device):
        safe_indices, _ = _normalize_sparse_index_slots(indices, topk_length, kv.shape[0])
        result = sparse_attn_score_recompute_wrapper(
            q.unsqueeze(0),
            kv.unsqueeze(0),
            lse.unsqueeze(0),
            _pad_indices(safe_indices).unsqueeze(0),
            scale,
            qhead_per_kv_head=q.shape[1],
            topk_length=None,
            topk_indices_global=False,
        )
        target = result["target"].squeeze(0)[:, : indices.shape[1]].contiguous()
    return TupleDict(target=target, indices=safe_indices)


__all__ = ["sparse_attention", "sparse_attention_score_recompute"]
