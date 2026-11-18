# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 packed-THD compact indexer score plus radix Top-K orchestration."""

from __future__ import annotations

from typing import Optional

import torch

from cudnn.deepseek_sparse_attention.indexer_top_k.compress_top_k_sm100 import compress_stage2_topk_varlen
from cudnn.deepseek_sparse_attention.indexer_top_k.local_to_global_dsl import _local_to_global_inplace
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context
from cudnn.frost.capture import is_capturing

from ._compressed_top_k_sm100 import _compress_cand_batch_offsets
from ._interface_sm90 import indexer_fwd as indexer_fwd_sm90


def _validate_output(t, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    if t is None:
        return
    if not t.is_cuda or t.device != device or t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA {dtype} tensor of shape {shape} on {device}")


def _validate_alignment(t: Optional[torch.Tensor], name: str, alignment: int = 16) -> None:
    if t is not None and t.numel() and t.data_ptr() % alignment:
        raise ValueError(f"{name} must be {alignment}-byte aligned")


def _storage_span(t: Optional[torch.Tensor]) -> Optional[tuple[int, int]]:
    if t is None or t.numel() == 0:
        return None
    span_elements = 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    start = t.data_ptr()
    return start, start + span_elements * t.element_size()


def _reject_overlapping_writes(writes, reads) -> None:
    named_writes = [(name, _storage_span(tensor)) for name, tensor in writes]
    named_reads = [(name, _storage_span(tensor)) for name, tensor in reads]
    named_writes = [(name, span) for name, span in named_writes if span is not None]
    named_reads = [(name, span) for name, span in named_reads if span is not None]
    pairs = [(named_writes[i], named_writes[j]) for i in range(len(named_writes)) for j in range(i + 1, len(named_writes))]
    pairs += [(write, read) for write in named_writes for read in named_reads]
    for (lhs_name, (lhs_start, lhs_end)), (rhs_name, (rhs_start, rhs_end)) in pairs:
        if lhs_start < rhs_end and rhs_start < lhs_end:
            raise ValueError(f"SM90 compressed indexer writable buffer {lhs_name} overlaps {rhs_name}")


def indexer_fwd_compress_topk_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    ratio: int,
    qhead_per_kv_head: Optional[int],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    sm_scale: float = 1.0,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    topk_indices_global: bool = True,
    cand_buffer: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    return_softmax: Optional[bool] = None,
    softmax_out: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    stream=None,
    _compiled_score=None,
    _compiled_topk=None,
) -> tuple[torch.Tensor, ...]:
    """Run the validated SM90 packed-THD compact score and Top-K path.

    This first SM90 variant intentionally supports one production geometry:
    ratio=4, D=128, one KV head, and 64 query heads per KV head. Candidate,
    output, softmax and LSE buffers must be caller-owned during graph capture.
    """
    precision = precision.lower()
    if precision not in ("bf16", "fp8"):
        raise NotImplementedError("SM90 compressed indexer forward supports precision='bf16' or 'fp8' only")
    if ratio != 4:
        raise NotImplementedError(f"SM90 compressed indexer forward is validated only for ratio=4, got {ratio}")
    if not 0 < topk <= 2048:
        raise ValueError(f"top_k must be in [1, 2048], got {topk}")
    if q.ndim != 3 or k.ndim != 3 or w.ndim != 2:
        raise ValueError("SM90 compressed indexer forward requires packed THD q(Tq,H,D), k(Tk,Hkv,D), and w(Tq,H)")
    total_q, n_heads_q, head_dim = q.shape
    if head_dim != 128 or k.shape[-1] != 128:
        raise NotImplementedError(f"SM90 compressed indexer forward is validated only for head_dim=128, got q={head_dim}, k={k.shape[-1]}")
    if k.shape[-2] != 1:
        raise NotImplementedError(f"SM90 compressed indexer forward is validated only for one KV head, got {k.shape[-2]}")
    derived_qhpkv = n_heads_q // k.shape[-2]
    if qhead_per_kv_head is None:
        qhead_per_kv_head = derived_qhpkv
    if qhead_per_kv_head != 64 or derived_qhpkv != 64:
        raise NotImplementedError(
            "SM90 compressed indexer forward is initially validated only for qhead_per_kv_head=64; "
            f"got requested={qhead_per_kv_head}, derived={derived_qhpkv}"
        )
    if tuple(w.shape) != (total_q, n_heads_q):
        raise ValueError(f"w must have shape {(total_q, n_heads_q)}, got {tuple(w.shape)}")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() < 2 or cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1D tensors of equal shape (batch_size + 1,)")
    device = q.device
    for tensor, name in ((q, "q"), (k, "k"), (w, "w")):
        if not tensor.is_cuda or tensor.device != device or not tensor.is_contiguous():
            raise NotImplementedError(f"SM90 compressed indexer forward requires contiguous CUDA {name} on {device}")
        _validate_alignment(tensor, name)
    if precision == "bf16":
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or w.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("SM90 BF16 compressed indexer forward requires BF16 q/k and BF16 or FP32 w")
        if q_scale is not None or k_scale is not None:
            raise ValueError("SM90 BF16 compressed indexer forward does not accept q_scale or k_scale")
    elif q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn or w.dtype != torch.bfloat16:
        raise TypeError("SM90 FP8 compressed indexer forward requires E4M3 q/k and BF16 w")
    for tensor, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
        if not tensor.is_cuda or tensor.device != device or tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous int32 CUDA tensor on {device}")
        _validate_alignment(tensor, name)
    if q_causal_offsets is not None and (
        not q_causal_offsets.is_cuda
        or q_causal_offsets.device != device
        or q_causal_offsets.dtype != torch.int32
        or tuple(q_causal_offsets.shape) != (cu_seqlens_q.numel() - 1,)
        or not q_causal_offsets.is_contiguous()
    ):
        raise ValueError("q_causal_offsets must be a contiguous int32 CUDA tensor of shape (batch_size,) on the input device")
    _validate_alignment(q_causal_offsets, "q_causal_offsets")
    if precision == "fp8":
        for tensor, name, shape in (
            (q_scale, "q_scale", (total_q, n_heads_q)),
            (k_scale, "k_scale", (k.shape[0], 1)),
        ):
            if (
                tensor is None
                or not tensor.is_cuda
                or tensor.device != device
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"SM90 fp8 {name} must be a contiguous float32 CUDA tensor of shape {shape} on {device}")
            _validate_alignment(tensor, name)

    launch_stream = resolve_stream(stream, device)
    capturing = is_capturing(launch_stream)
    if capturing and (cand_buffer is None or cand_batch_offsets is None):
        raise ValueError("CUDA Graph capture requires caller-provided cand_buffer and cand_batch_offsets prepared before capture")
    if capturing and (out_indices is None or out_logits is None):
        raise ValueError("CUDA Graph capture requires caller-provided out_indices and out_logits")
    want_softmax = True if return_softmax is None else bool(return_softmax)
    want_softmax = want_softmax or softmax_out is not None
    want_lse = bool(return_lse or lse_out is not None)
    if capturing and want_softmax and softmax_out is None:
        raise ValueError("CUDA Graph capture requires caller-provided softmax_out when softmax is enabled")
    if capturing and want_lse and lse_out is None:
        raise ValueError("CUDA Graph capture requires caller-provided lse_out when LSE is enabled")
    if capturing and topk_indices_global:
        raise ValueError("CUDA Graph capture on SM90 requires topk_indices_global=False to avoid an extra conversion launch")

    out_shape = (total_q, topk)
    _validate_output(out_indices, "out_indices", out_shape, torch.int32, device)
    _validate_output(out_logits, "out_logits", out_shape, torch.float32, device)
    _validate_output(softmax_out, "softmax_out", out_shape, torch.float32, device)
    _validate_output(lse_out, "lse_out", (total_q,), torch.float32, device)
    _validate_alignment(lse_out, "lse_out")
    output_buffers = (
        ("out_indices", out_indices),
        ("out_logits", out_logits),
        ("softmax_out", softmax_out),
        ("lse_out", lse_out),
    )
    input_buffers = (
        ("q", q),
        ("k", k),
        ("w", w),
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("cand_batch_offsets", cand_batch_offsets),
        ("q_causal_offsets", q_causal_offsets),
    )
    _reject_overlapping_writes(output_buffers, input_buffers)

    bs = cu_seqlens_q.numel() - 1
    if cand_batch_offsets is None:
        with torch_stream_context(launch_stream, device):
            cand_batch_offsets = _compress_cand_batch_offsets(cu_seqlens_q, cu_seqlens_k, ratio, q_causal_offsets)
    else:
        if (
            not cand_batch_offsets.is_cuda
            or cand_batch_offsets.device != device
            or cand_batch_offsets.dtype != torch.int64
            or tuple(cand_batch_offsets.shape) != (bs + 1,)
            or not cand_batch_offsets.is_contiguous()
        ):
            raise ValueError(f"cand_batch_offsets must be a contiguous int64 CUDA tensor of shape ({bs + 1},) on {device}")
    _validate_alignment(cand_batch_offsets, "cand_batch_offsets")
    if cand_buffer is None:
        with torch_stream_context(launch_stream, device):
            total_candidates = int(cand_batch_offsets[-1].item())
            cand_buffer = torch.empty(total_candidates, dtype=torch.float32, device=device)
    elif (
        not cand_buffer.is_cuda
        or cand_buffer.device != device
        or cand_buffer.dtype != torch.float32
        or cand_buffer.ndim != 1
        or not cand_buffer.is_contiguous()
        or cand_buffer.data_ptr() % 16 != 0
    ):
        raise ValueError("cand_buffer must be a contiguous, 16-byte-aligned 1D float32 CUDA tensor on the input device")

    _reject_overlapping_writes(
        (("cand_buffer", cand_buffer),),
        input_buffers + (("cand_batch_offsets", cand_batch_offsets),) + output_buffers,
    )
    with torch_stream_context(launch_stream, device):
        if out_indices is None:
            out_indices = torch.empty(out_shape, dtype=torch.int32, device=device)
        if out_logits is None:
            out_logits = torch.empty(out_shape, dtype=torch.float32, device=device)
        if want_softmax and softmax_out is None:
            softmax_out = torch.empty(out_shape, dtype=torch.float32, device=device)
        if want_lse and lse_out is None:
            lse_out = torch.empty((total_q,), dtype=torch.float32, device=device)

    if total_q == 0:
        result = (out_indices, out_logits)
        if want_softmax:
            result += (softmax_out,)
        if want_lse:
            result += (lse_out,)
        return result

    indexer_fwd_sm90(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=sm_scale,
        out=cand_buffer,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        return_lse=want_lse,
        lse_out=lse_out,
        current_stream=launch_stream,
        is_compressed_logits=True,
        cand_batch_offsets=cand_batch_offsets,
        _compiled_kernel=_compiled_score,
    )

    # Keep launch specialization independent of live token counts. 256 threads
    # is the validated long-context SM90 variant; short/ragged batches reuse it.
    block_threads = 256
    stage2 = compress_stage2_topk_varlen(
        cand_buffer,
        cu_seqlens_q,
        cu_seqlens_k,
        cand_batch_offsets,
        total_q,
        max_seqlen_q,
        max_seqlen_k,
        topk,
        ratio,
        block_threads=block_threads,
        stream=launch_stream,
        out_indices=out_indices,
        out_logits=out_logits,
        q_causal_offsets=q_causal_offsets,
        out_softmax=softmax_out,
        return_softmax=want_softmax,
        deterministic=deterministic,
        _compiled_kernel=_compiled_topk,
    )
    if want_softmax:
        indices, logits, softmax = stage2
    else:
        indices, logits = stage2
    if topk_indices_global:
        _local_to_global_inplace(indices, 0, cu_seqlens_q, cu_seqlens_k, launch_stream)
    result = (indices, logits)
    if want_softmax:
        result += (softmax,)
    if want_lse:
        result += (lse_out,)
    return result


__all__ = ["indexer_fwd_compress_topk_sm90"]
