# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed compile-time constants of the GDN chunk-factor (T_inv) pass (SM100 / SM103); the per-compile attributes live on
``GdnTinvCfg`` in the kernel file.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 64  # chunk size / token tile (the mma M, N of the K K^T GEMM)
    D_K: int = 128  # default key head dim (contraction of the K K^T GEMM); the per-compile cfg carries 64 or 128

    # --- work split (one CTA per SM, persistent over rows_per_cta tile rows x heads_per_cta heads) ---
    MAX_ROWS_PER_CTA: int = 128  # sRows capacity; the host picks rows_per_cta <= this for one wave of CTAs

    # --- warp assignments (12 warps total) ---
    COMPUTE_GROUP_WARP_IDS: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7))  # pair inverses; group g takes pairs g, g + 2, ...
    TMA_K_WARP_ID: int = 8  # K tile loads
    TCGEN05_MMA_WARP_ID: int = 9  # sole tcgen05 issuer: the pair's [K0;K1] @ [K0;K1]^T
    EPILOGUE_WARP_ID: int = 10  # tile stores
    LOAD_GATE_WARP_ID: int = 11  # gate cumsum + beta loads

    # --- register split ---
    LAUNCH_REGS: int = 168  # the DSL's per-thread register cap at launch; setmaxregister redistributes it per role
    NUM_REGS_OTHER: int = 96

    THREADS_PER_WARP: int = 32

    # --- SMEM stage counts ---
    SMEM_K_STAGES: int = 4  # two-box K stages (32 KB each for bf16/fp16) kept in flight by the TMA warp
    SMEM_TILE_STAGES: int = 4  # 8 KB tile stages per compute group (finished tiles awaiting the store warp)
    SMEM_GATE_STAGES: int = 4  # gate cumsum / beta stages per compute group (1 KB each), filled by the gate warp

    # --- TMEM stage counts ---
    TMEM_ACC_STAGES: int = 4  # 128-column accumulator stages (pair GEMM results awaiting their compute group)

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
