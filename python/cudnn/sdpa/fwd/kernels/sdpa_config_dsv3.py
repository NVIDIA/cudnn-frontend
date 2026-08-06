# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 SDPA flavor config for SM80 — d_qk = 192, d_v = 128, FP16.

DSv3 is the first SM80 flavor with d_qk != d_v; the same kernel skeleton
in ``prefill_f16_sm80.py`` is reused, with the K vs V row-stride
plumbing already split internally so the asymmetric head dims work
without a fork.

Knob defaults below are an initial guess — actual best (tile_m, tile_n,
num_warps) point on A100 has not been swept yet.  The perf-node sweep
will pin it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int = 192
    D_V: int = 128
    TILE_M: int = 128
    TILE_N: int = 64
    NUM_WARPS: int = 8


CFG = Cfg()
