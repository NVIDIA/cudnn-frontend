# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit scratch layout for aligned HCA backward."""

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
    local_tokens: int
    cp_size: int
    group_tokens: int
    groups: int
    keys: int
    compressed_keys: int
    buffers: tuple
    nbytes: int

    @property
    def local_kv_rows(self):
        return self.local_tokens + 128

    @property
    def groups_per_rank(self):
        return self.local_tokens // 128

    @property
    def compressed_rows(self):
        return self.cp_size * (self.groups_per_rank + 1)

    @property
    def kv_rows(self):
        return self.local_kv_rows + self.compressed_rows


def workspace_layout(rank, local_tokens=4096, cp_size=16):
    if type(cp_size) is not int or cp_size not in (4, 8, 16):
        raise ValueError("cp_size must be 4, 8, or 16")
    if type(rank) is not int or not 0 <= rank < cp_size:
        raise ValueError("cp_rank must be an integer in [0, cp_size)")
    if type(local_tokens) is not int or local_tokens % 128 or not 8192 <= local_tokens * cp_size <= 131072:
        raise NotImplementedError("Aligned HCA requires 8K-128K tokens and 128-token-aligned CP chunks")
    original = local_tokens == 4096 and cp_size == 16
    group_tokens = 32 if local_tokens * cp_size <= 16384 else 128
    if original:
        group_tokens = 32 if rank not in (3, 5, 7, 9, 11, 12, 13, 14, 15) else 128
    groups, n, dim = local_tokens // group_tokens, local_tokens * 128, 512
    compressed = (rank + 1) * (local_tokens // 128)
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
    return WorkspaceLayout(rank, local_tokens, cp_size, group_tokens, groups, keys, compressed, tuple(buffers), ((offset + 255) // 256) * 256)
