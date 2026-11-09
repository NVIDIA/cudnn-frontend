# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch training API for simplified sparse attention."""

from contextlib import nullcontext

import torch

from cudnn._tupledict import TupleDict
from cudnn.frost.buffers import cutedsl_requirement_error

from .cutedsl.backend import RowMetadata as SimplifiedSparseAttentionMetadata

__all__ = ["SimplifiedSparseAttentionMetadata", "simplified_sparse_attention", "simplified_sparse_attention_reserve"]


def _check_runtime():
    error = cutedsl_requirement_error("simplified_sparse_attention", floor=(4, 8, 0))
    if error:
        raise ImportError(error)
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("simplified_sparse_attention does not support external CUDA graph capture.")


def simplified_sparse_attention(query, key, value, q_idx, k_idx, metadata, topk, *, loss_coeff=0.1, kv_ready_event=None):
    """Return BF16 ``output`` and FP32 ``indexer_loss``, with PyTorch autograd.

    Q is [Tq, Hq, 256], K/V are [Tk, 1, 256], and indexer Q/K are
    [Tq, 128]/[Tk, 128]. All inputs are BF16 on SM100 or SM103.
    Hq is 16, 32, or 96; topk is 512, 1024, or 2048. Metadata supplies
    per-query key bounds, valid rows, and the global loss denominator.
    The loss already includes ``loss_coeff`` and that denominator.

    The call allocates outputs and scratch on the input device's current
    stream. It does not support external CUDA graph capture or guarantee
    bitwise determinism. K/V may be produced on another stream: record
    ``kv_ready_event`` after production and retain their storage until both
    streams finish. Q, indexer inputs, and metadata must already be ready.
    """
    from .cutedsl.backend import sparse_attention

    with torch.cuda.device(query.device) if query.is_cuda else nullcontext():
        _check_runtime()
        output, loss = sparse_attention(query, key, value, q_idx, k_idx, metadata, topk, loss_coeff=loss_coeff, kv_ready_event=kv_ready_event)
    return TupleDict(output=output, indexer_loss=loss)


def simplified_sparse_attention_reserve(max_query_len, max_key_len, topk):
    """Warm filtered-indexer storage on the current device and stream.

    Call before training with the largest rectangular query/key extents.
    Calls within these extents reuse the reserved repair graph and storage.
    This does not enable external CUDA graph capture.
    """
    from .cutedsl.backend import reserve_indexer

    _check_runtime()
    reserve_indexer(max_query_len, max_key_len, topk)
