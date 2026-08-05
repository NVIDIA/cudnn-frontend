# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Llama SDPA flavor config for SM80 — d_qk = d_v = 128, FP16.

Knob defaults below were swept on A100-PCIE-40GB (cudnn-dev-tallship-22-04):
``(tile_m=128, tile_n=64, num_warps=8)`` matches Phase-3 baseline of
~97 % FA-2.8.3 at SQ=8K — keep these unless a perf-node sweep finds a
better point.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int = 128
    D_V: int = 128
    TILE_M: int = 128
    TILE_N: int = 64
    NUM_WARPS: int = 8


CFG = Cfg()
