# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Device algorithm source: pkuwangh/mlsys@72c2384a5c325198d6600906d5835d1ef6faabac
# mlsys/gpu/kernels/wan_vae/mega_wan_vae/kernels/_scheduler.py

"""Persistent M-major scheduling with runtime M extent."""

from dataclasses import dataclass

import cutlass
from cutlass import cute


@dataclass(frozen=True)
class TileSchedule:
    """Compile-time tile geometry; grid Z identifies persistent workers."""

    tiles_n: int
    cluster_m: int

    @cute.jit
    def get_work(self, index, runtime_clusters_m, wide_index: cutlass.Constexpr = False):
        """Decode a cluster index; callers advance it by gridDim.z after each tile."""
        if cutlass.const_expr(wide_index):
            clusters_m = runtime_clusters_m
            divisor = runtime_clusters_m
        else:
            clusters_m = runtime_clusters_m.divisor
            divisor = runtime_clusters_m
        tile_n = index // divisor
        cluster_m = index % divisor
        tile_m = cluster_m * self.cluster_m + cute.arch.block_idx()[0]
        return (
            (
                cute.arch.make_warp_uniform(tile_m),
                cute.arch.make_warp_uniform(tile_n),
                cutlass.Int32(0),
            ),
            cutlass.Boolean(cute.arch.make_warp_uniform(index < clusters_m * self.tiles_n)),
        )
