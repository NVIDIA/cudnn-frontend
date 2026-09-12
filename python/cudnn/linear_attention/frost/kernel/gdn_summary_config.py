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

"""Fixed compile-time constants of the GDN fused state-summary (H + M) kernel (SM100 / SM103); the per-compile attributes
live on ``GdnSummaryCfg`` in the kernel file.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 64  # chunk size / token tile (the mma N or K of every GEMM)
    D_K: int = 128  # default key head dim (contraction of the K*state GEMM, output dim of the KV update); per-compile 64 or 128
    D_V: int = 128  # default value head dim (rows of chain H); per-compile 64 or 128

    # --- warp assignments (12 warps total) ---
    CHAIN_M_WARP_IDS: Tuple[int, ...] = (0, 1, 2, 3)  # transition chain epilogues (identity seed, V = 0)
    CHAIN_H_WARP_IDS: Tuple[int, ...] = (4, 5, 6, 7)  # state chain epilogues (zero / initial_state seed, consumes V)
    LOAD_GATE_WARP_ID: int = 8  # gate chunk loads
    TMA_WARP_ID: int = 9  # K / V TMA loads + chunk-factor bulk loads
    TCGEN05_MMA_WARP_ID: int = 10  # sole tcgen05 issuer: both chains' KS / U / KV per chunk; TMEM lifecycle
    REGISTER_POOL_WARP_ID: int = 11  # setmaxnreg.dec only: the CTA launches at 168 regs/thread and this warp's share feeds the chain groups

    # --- register split (12 warps launched at 168 regs/thread: 4 x 24 + 8 x 240 = 2016 = 12 x 168) ---
    NUM_REGS_CHAIN: int = 240
    NUM_REGS_OTHER: int = 24

    THREADS_PER_WARP: int = 32

    CLUSTER_SHAPE_MNK: Tuple[int, int, int] = (1, 1, 1)

    # --- SMEM stage counts ---
    SMEM_SCHEDULER_STAGES: int = 2
    SMEM_K_STAGES: int = 3
    SMEM_V_STAGES: int = 3
    SMEM_T_INV_STAGES: int = 3
    SMEM_GATE_STAGES: int = 3

    # --- TMEM stage counts ---
    TMEM_STATE_ACC_STAGES: int = 1
    TMEM_STATE_INPUT_STAGES: int = 1

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
