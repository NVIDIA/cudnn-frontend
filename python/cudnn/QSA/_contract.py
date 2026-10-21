# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host metadata checks; no device reads or DSL imports."""

import math

import torch


def check_tensor(name, tensor, shape, dtype, device, alignment=4):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(shape):
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
    if total < 1 or dim != 256 or (hq, hkv) not in ((12, 1), (24, 2)):
        raise NotImplementedError("QSA requires T > 0, D = 256, and 12/1 or 24/2 query/KV heads")
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
        check_tensor(name, tensor, (total, heads, dim), q.dtype, device, 16)
    check_tensor("sel_blocks", sel_blocks, (total, blocks), torch.uint8, device, 1)
    for name, tensor in (("lo", lo), ("hi", hi), ("blk_base", blk_base), ("tok_base", tok_base)):
        check_tensor(name, tensor, (total,), torch.int32, device)
    for name, tensor in (("block_first", block_first), ("block_last", block_last)):
        check_tensor(name, tensor, (blocks,), torch.int32, device)
    return total, hq, hkv, blocks
