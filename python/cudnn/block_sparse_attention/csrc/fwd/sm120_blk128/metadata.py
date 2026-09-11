# SPDX-FileCopyrightText: Copyright (c) 2026 tiffany940107. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch

SM120_BLK128_LOGICAL_BLOCK_SIZE = 128
SM120_BLK128_BACKEND_BLOCK_SIZE = 64


@dataclass(frozen=True)
class Sm120Blk128LoweredMetadata:
    """Physical blk64 metadata for the SM120 logical-blk128 forward path."""

    q2k_block_index: torch.Tensor
    block_sparse_num: int
    block_sizes: Optional[torch.Tensor]
    q2k_block_nums: Optional[torch.Tensor]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _lower_block_sizes(block_sizes: torch.Tensor) -> torch.Tensor:
    """Split each logical 128-token size into two physical 64-token sizes."""
    first_half = block_sizes.clamp(min=0, max=SM120_BLK128_BACKEND_BLOCK_SIZE)
    second_half = (block_sizes - SM120_BLK128_BACKEND_BLOCK_SIZE).clamp(min=0, max=SM120_BLK128_BACKEND_BLOCK_SIZE)
    return torch.stack((first_half, second_half), dim=-1).flatten(-2).contiguous()


def lower_sm120_blk128_metadata(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor],
    q2k_block_nums: Optional[torch.Tensor],
    *,
    seqlen_q: int,
) -> Sm120Blk128LoweredMetadata:
    """Lower logical blk128 sparse metadata to the native SM120 blk64 ABI.

    Every logical KV block ``j`` becomes physical blocks ``2*j`` and
    ``2*j + 1``. Each logical Q row is repeated for its two physical Q tiles.
    The public wrapper validates all shapes and requires complete logical KV
    blocks before invoking this transform. All operations stay on the metadata
    device and therefore do not introduce a host synchronization.
    """
    physical_index_base = q2k_block_index * 2
    physical_indices = torch.stack((physical_index_base, physical_index_base + 1), dim=-1).flatten(-2)
    physical_q_blocks = _ceil_div(seqlen_q, SM120_BLK128_BACKEND_BLOCK_SIZE)
    physical_indices = physical_indices.repeat_interleave(2, dim=2)[:, :, :physical_q_blocks, :].contiguous()

    physical_block_sizes = _lower_block_sizes(block_sizes) if block_sizes is not None else None
    if q2k_block_nums is not None:
        physical_block_nums = (q2k_block_nums * 2).repeat_interleave(2, dim=2)[:, :, :physical_q_blocks].contiguous()
        physical_block_sparse_num = 0
    else:
        physical_block_nums = None
        physical_block_sparse_num = int(block_sparse_num) * 2

    return Sm120Blk128LoweredMetadata(
        q2k_block_index=physical_indices,
        block_sparse_num=physical_block_sparse_num,
        block_sizes=physical_block_sizes,
        q2k_block_nums=physical_block_nums,
    )
