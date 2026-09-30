# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def supported_block_size(*, backward: bool = False) -> int:
    if not torch.cuda.is_available():
        pytest.skip("block sparse attention tests require CUDA")
    major, _ = torch.cuda.get_device_capability()
    supported = {9, 10, 11} if backward else {9, 10, 11, 12}
    if major not in supported:
        direction = "backward" if backward else "forward"
        pytest.skip(f"block sparse attention {direction} is not supported on SM{major}x")
    return 64 if major in {9, 12} else 128


def make_fixed_metadata(batch: int, heads: int, seqlen_q: int, seqlen_k: int, block_size: int):
    num_q_blocks = (seqlen_q + block_size - 1) // block_size
    num_kv_blocks = (seqlen_k + block_size - 1) // block_size
    selected = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    q2k = selected.view(1, 1, 1, 2).expand(batch, heads, num_q_blocks, 2).contiguous()
    block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    block_sizes[-1] = seqlen_k - (num_kv_blocks - 1) * block_size
    return q2k, 2, block_sizes


def make_variable_metadata(batch: int, heads: int, seqlen_q: int, seqlen_k: int, block_size: int):
    num_q_blocks = (seqlen_q + block_size - 1) // block_size
    num_kv_blocks = (seqlen_k + block_size - 1) // block_size
    q2k = torch.empty((batch, heads, num_q_blocks, 3), dtype=torch.int32, device="cuda")
    q2k[..., 0, :] = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    q2k[..., 1, :] = torch.tensor([2, 0, 3], dtype=torch.int32, device="cuda")
    q2k_block_nums = torch.empty((batch, heads, num_q_blocks), dtype=torch.int32, device="cuda")
    q2k_block_nums[..., 0] = 1
    q2k_block_nums[..., 1] = 3
    block_sizes = torch.full((num_kv_blocks,), block_size, dtype=torch.int32, device="cuda")
    block_sizes[2] = block_size // 2
    return q2k, q2k_block_nums, block_sizes
