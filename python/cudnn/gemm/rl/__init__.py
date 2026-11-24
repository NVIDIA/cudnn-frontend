# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""BF16 GEMM with fixed per-row arithmetic for rollout and training."""

from typing import NamedTuple


class Schedule(NamedTuple):
    tile_m: int
    tile_n: int
    stages: int
    mma_group: int
    cluster_n: int

    def grid_x(self, m, n):
        return ((m - 1) // self.tile_m + 1) * ((n - 1) // (self.tile_n * self.cluster_n) + 1) * self.mma_group


def chunks_for(n, k):
    """Choose a power of two of equal K16-aligned chunks from the weight shape."""
    if (n + 31) // 32 >= 256:
        return 1
    budget = min(k // (16 * n), 128)
    chunks = 1
    while 2 * chunks <= budget and k % (32 * chunks) == 0 and k // (2 * chunks) >= 256:
        chunks *= 2
    return chunks


def schedule_for(m, n, chunks):
    """Choose placement independently of the arithmetic fixed by the weight shape."""
    if m <= 64:
        tile_m, tile_n, stages, group = 64, 32, 16, 1
        cluster_n = 4 if n >= 128 else 2 if n >= 64 else 1
    else:
        group = 2
        if m >= 256 and n >= 256 and chunks == 1:
            tile_m, tile_n, stages = 256, 128, 8
        else:
            tile_m, tile_n, stages = 128, 64, 16
        cluster_n = 2 if n > tile_n else 1
    return Schedule(tile_m, tile_n, stages, group, cluster_n)
