# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS SDPA flavor config for SM80 — d_qk = d_v = 64, FP16.

Knob defaults below were picked from the A100-PCIE-40GB perf sweep:
``(tile_m=128, tile_n=64, num_warps=8)`` — wait, NO: at d=64 the
SV mma has 8 n_frags instead of 16, so M_BLOCKS=2 per warp fits the
register budget that spills at d=128.  Best point is
``(tile_m=128, tile_n=64, num_warps=4)`` (M_BLOCKS=2 per warp) which
matches FA to within ~5-7 % on H=32 H_kv=2 GQA shapes at SQ ∈ {2K,
4K, 8K}.  For ``--mask causal`` the sweep additionally wants
``--sched lpt_l2 --sched-l2-mib 16`` on top — driver-side override,
not pinned in the config.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    D_QK: int = 64
    D_V: int = 64
    TILE_M: int = 128
    TILE_N: int = 64
    NUM_WARPS: int = 4


CFG = Cfg()
