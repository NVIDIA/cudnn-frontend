# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit scratch layout for the fixed S65536/CP16 HCA execution plan."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Buffer:
    name: str
    dtype: str
    shape: tuple
    offset: int
    nbytes: int


@dataclass(frozen=True)
class WorkspaceLayout:
    rank: int
    group_tokens: int
    groups: int
    keys: int
    compressed_keys: int
    buffers: tuple
    nbytes: int


def workspace_layout(rank):
    if type(rank) is not int or not 0 <= rank < 16:
        raise ValueError("An integer CP16 rank is required")
    group_tokens = 128 if rank in (3, 5, 7, 9, 11, 12, 13, 14, 15) else 32
    groups, n, dim = 4096 // group_tokens, 4096 * 128, 512
    compressed = (rank + 1) * 32
    keys = ((128 + group_tokens + compressed + 63) // 64) * 64
    buffers = []
    offset = 0
    specs = [
        ("packed_keys", "bfloat16", (groups, keys, dim)),
        ("normalization", "float32", (2 * n,)),
        ("sink_partial", "float32", (n,)),
        ("p", "bfloat16", (groups, group_tokens * 128, keys)),
        ("ds", "bfloat16", (groups, group_tokens * 128, keys)),
        ("dk", "float32", (groups, keys, dim)),
        ("dv", "float32", (groups, keys, dim)),
    ]
    for name, dtype, shape in specs:
        offset = ((offset + 255) // 256) * 256
        size = math.prod(shape) * (2 if dtype == "bfloat16" else 4)
        buffers.append(Buffer(name, dtype, shape, offset, size))
        offset += size
    return WorkspaceLayout(rank, group_tokens, groups, keys, compressed, tuple(buffers), ((offset + 255) // 256) * 256)
