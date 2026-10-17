# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Fixed M128/N128 dense d128 geometry, split input/O, one score slot."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class CfgBwdD128:
    TILE_M: int = 128
    TILE_N: int = 128
    HEAD_DIM: int = 128
    HEAD_GRANULE: int = 64
    GROUP_COLS: int = 64
    POINTWISE_COLS: int = 32
    TILE_K_HW: int = 16
    THREADS: int = 384
    COMPUTE_WARPS: int = 8
    WARP_LANES: int = 32
    COMPUTE_ARRIVERS: int = 256
    STATS_WARPS: int = 2
    STATS_THREADS: int = 64
    STATS_ARRIVERS: int = 64
    TMA_WARP: int = 8
    MMA_WARP: int = 9
    STATS_WARP: int = 10
    CONTROL_WARP: int = 11
    CTA_MMA: int = 1
    ONE_LANE: int = 1
    INPUT_STAGES: int = 2
    SINGLE_STAGE: int = 1
    BPE: int = 2
    SWIZZLE_BYTES: int = 128
    SMEM_LAYOUT: int = 2
    DESC_VERSION: int = 0
    TMEM_WORD_BYTES: int = 4
    TMEM_COLS: int = 512
    TMEM_S: int = 0
    TMEM_DP: int = 128
    TMEM_G0: int = 256
    TMEM_G1: int = 384
    RESIDENT0: int = 0
    RESIDENT1: int = 16384
    INPUT: int = 32768
    INPUT_STAGE_ELEMS: int = 32768
    INPUT_OPERAND_ELEMS: int = 16384
    AUX: int = 98304
    SLAB_ELEMS: int = 114688
    STATS_ELEMS: int = 256
    Q_STATS_DELTA: int = 128
    KV_STATS_DELTA: int = 128
    CONTROL_BYTES: int = 260
    DRIVER_RESERVED_BYTES: int = 1024
    MAX_ROUNDED_REGISTERS: int = 168

    @property
    def dynamic_smem_bytes(self):
        return self.SLAB_ELEMS * self.BPE + self.STATS_ELEMS * 4 + self.CONTROL_BYTES

    @property
    def smem_bytes(self):
        return self.dynamic_smem_bytes + self.DRIVER_RESERVED_BYTES

    def __post_init__(self):
        changed = [f.name for f in fields(self) if getattr(self, f.name) != f.default]
        if changed:
            raise ValueError(f"dense N128 geometry is fixed; unsupported overrides: {', '.join(changed)}")
        assert self.TILE_K_HW == 16
        assert self.HEAD_GRANULE * self.BPE == self.SWIZZLE_BYTES
        assert self.INPUT + self.INPUT_STAGES * self.INPUT_STAGE_ELEMS == self.AUX
        assert self.AUX + self.TILE_M * self.HEAD_DIM == self.SLAB_ELEMS
        assert self.INPUT_STAGE_ELEMS == 2 * self.INPUT_OPERAND_ELEMS
        assert self.COMPUTE_WARPS * self.WARP_LANES == self.COMPUTE_ARRIVERS
        assert self.STATS_WARPS * self.WARP_LANES == self.STATS_ARRIVERS
        assert (self.COMPUTE_WARPS + 4) * self.WARP_LANES == self.THREADS
        assert self.dynamic_smem_bytes == 230660 and self.smem_bytes <= 227 * 1024
        assert self.TMEM_G1 + self.HEAD_DIM == self.TMEM_COLS


CFG = CfgBwdD128()
