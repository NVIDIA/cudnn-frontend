# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host metadata checks; no device reads or DSL imports."""

import math

import torch

# (query heads, KV heads) geometries the attention kernels serve: the full 24Q/2KV model and its 12Q/1KV (two-way) and
# 6Q/1KV (four-way, KV head replicated) tensor-parallel shards.
HEAD_GEOMETRIES = ((6, 1), (12, 1), (24, 2))
# The batched backward (bprop_sm100 + bprop_dispatch) fills one 24-column MMA tile with the heads of NQ consecutive
# queries, which share one gathered K/V block union per CTA: two 12-head queries at 12Q/1KV, four 6-head queries at
# 6Q/1KV (NQ * heads per KV head = 24). 24Q/2KV runs the single-query backward (bprop_sm100_base on SM100/SM103,
# bprop_sm107 on SM107).
PAIRED_BACKWARD_COLUMNS = 24
PAIRED_BACKWARD_QUERIES = {(12, 1): 2, (6, 1): 4}
# Compact IDs of the batched backward: the block index in the low BLK_BITS bits, one owner bit per batched query above
# them. At NQ = 4 the owner bits are 28..31, so a code with bit 31 set is a negative int32: readers mask the block with
# (1 << BLK_BITS) - 1 and test single owner bits with (code >> (BLK_BITS + i)) & 1, which the arithmetic shift keeps right.
BLK_BITS = 28


def paired_backward(hq, hkv):
    return (hq, hkv) in PAIRED_BACKWARD_QUERIES


def paired_queries(hq, hkv):
    """Queries batched per CTA by the backward: 2 (12Q/1KV) or 4 (6Q/1KV); 1 for the single-query kernels."""
    return PAIRED_BACKWARD_QUERIES.get((hq, hkv), 1)


def check_tensor(name, tensor, shape, dtype, device, alignment=4):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have dtype {dtype} on {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


def check_inputs(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, block_first, block_last, scale, block_size):
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor with shape [T, Hq, 256]")
    if not isinstance(k, torch.Tensor) or k.ndim != 3:
        raise ValueError("k must have shape [T, Hkv, 256]")
    total, hq, dim = q.shape
    hkv = k.shape[1]
    if total < 1 or dim != 256 or (hq, hkv) not in HEAD_GEOMETRIES:
        raise NotImplementedError("QSA requires T > 0, D = 256, and 6/1, 12/1 or 24/2 query/KV heads")
    if q.numel() > 2**31 - 1:
        raise NotImplementedError("QSA requires T * Hq * D <= INT32_MAX")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("QSA training requires BF16 q, k, v and grad_out")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size != 4:
        raise NotImplementedError("QSA supports block_size=4")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite host scalar")
    if not isinstance(sel_blocks, torch.Tensor) or sel_blocks.ndim != 2:
        raise ValueError("sel_blocks must have shape [T, NB]")
    blocks = sel_blocks.shape[1]
    if blocks < 1 or blocks > 2**31 - 1:
        raise ValueError("NB must be in [1, INT32_MAX]")
    device = q.device
    for name, tensor, heads in (("q", q, hq), ("k", k, hkv), ("v", v, hkv)):
        check_tensor(name, tensor, (total, heads, dim), torch.bfloat16, device, 16)
    check_tensor("sel_blocks", sel_blocks, (total, blocks), torch.uint8, device, 1)
    for name, tensor in (("lo", lo), ("hi", hi), ("blk_base", blk_base), ("tok_base", tok_base)):
        check_tensor(name, tensor, (total,), torch.int32, device)
    for name, tensor in (("block_first", block_first), ("block_last", block_last)):
        check_tensor(name, tensor, (blocks,), torch.int32, device)
    return total, hq, hkv, blocks
