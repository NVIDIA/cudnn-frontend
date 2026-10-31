# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper launch path for the public packed QSA selector."""

import torch

from cudnn._torch_stream import stream_context
from cudnn._tupledict import TupleDict


def select(
    q,
    k,
    cu,
    *,
    topk,
    ratio,
    pos,
    max_q,
    scale,
    weights,
    deterministic,
    ids,
    scores,
    cand,
    offsets,
    stream,
):
    import cutlass
    from cuda.bindings import driver as cuda
    from cudnn.api_base import check_buffers_disjoint
    from cudnn.deepseek_sparse_attention.indexer_forward._compressed_top_k_sm100 import (
        compress_topk_cand_buffer_size_thd,
    )
    from .kernels.select_sm90 import compile_score, compile_topk
    from .selection import _padded_cu_seqlens_k

    total, heads, _ = q.shape
    batch, blocks, _ = k.shape
    if batch < 1:
        raise ValueError("kbar requires B >= 1")
    for name, value in (("cu_seqlens_q", cu), ("q_pos0", pos)):
        if value is not None and not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if weights is not None and weights.dtype == torch.float32 and weights.stride(-1) != 1:
        raise NotImplementedError("FP32 w requires unit last stride")
    with torch.cuda.device(q.device), stream_context(stream, q.device):
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and any(x is None for x in (max_q, cand, offsets, ids, scores)):
            raise ValueError("CUDA-graph capture requires max_seqlen_q, cand_buffer, cand_batch_offsets, block_ids_out and scores_out")
        if max_q is None:
            max_q = int((cu[1:] - cu[:-1]).max().item())
        if type(max_q) is not int or max_q < 0 or max_q // ratio > blocks:
            raise ValueError("max_seqlen_q must be nonnegative and covered by kbar at block_size")
        cu_k = _padded_cu_seqlens_k(batch, blocks, q.device)
        if offsets is None:
            offsets, size = compress_topk_cand_buffer_size_thd(cu, cu_k, ratio, pos)
        else:
            if offsets.shape != (batch + 1,) or offsets.dtype != torch.int64 or offsets.device != q.device or not offsets.is_contiguous():
                raise ValueError("cand_batch_offsets must be contiguous int64 [B + 1] on qi's device")
            size = int(offsets[-1].item()) if cand is None else cand.numel()
        if cand is None:
            cand = torch.empty(size, dtype=torch.float32, device=q.device)
        if cand.dtype != torch.float32 or cand.device != q.device or not cand.is_contiguous() or cand.data_ptr() % 16:
            raise ValueError("cand_buffer must be contiguous, 16-byte aligned float32 on qi's device")
        if cand.numel() < size:
            raise ValueError("cand_buffer is smaller than the compact candidate layout")
        cand = cand.view(-1)
        outputs = []
        for name, value, dtype in (
            ("block_ids_out", ids, torch.int32),
            ("scores_out", scores, torch.float32),
        ):
            if value is None:
                value = torch.empty((total, topk), dtype=dtype, device=q.device)
            if value.shape != (total, topk) or value.dtype != dtype or value.device != q.device or value.stride(-1) != 1:
                raise ValueError(f"{name} must have shape [T, top_k], dtype {dtype}, unit last stride and qi's device")
            outputs.append(value)
        ids, scores = outputs
        check_buffers_disjoint(
            (("cand_buffer", cand), ("block_ids_out", ids), ("scores_out", scores)),
            (("qi", q), ("kbar", k), ("cu_seqlens_q", cu), ("q_pos0", pos), ("w", weights), ("cand_batch_offsets", offsets)),
            what="QSA selector",
        )
        if total:
            dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float8E4M3FN
            wdtype = None if weights is None else (cutlass.BFloat16 if weights.dtype == torch.bfloat16 else cutlass.Float32)
            launch_stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
            compile_score(heads, dtype, wdtype, pos is not None)(
                q,
                k,
                weights,
                cu,
                pos,
                offsets,
                cand,
                float(scale),
                ratio,
                max_q,
                launch_stream,
            )
            compile_topk(topk, ratio, pos is not None, deterministic)(
                cand,
                ids.view(total, 1, topk),
                scores.view(total, 1, topk),
                max_q,
                blocks,
                launch_stream,
                cu,
                cu_k,
                offsets,
                pos,
                None,
                None,
            )
        return TupleDict(block_ids=ids, scores=scores)
