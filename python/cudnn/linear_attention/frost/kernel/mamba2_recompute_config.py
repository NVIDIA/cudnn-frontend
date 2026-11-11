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

"""
Fixed constants of the Mamba-2 recompute (state / checkpoint-only) kernel: warp roles, register split, SMEM / TMEM
stage counts.  The per-compile knobs (dtypes, initial / final state, checkpoints) are ``Mamba2RecomputeCfg`` fields.

Target arch: SM100 / SM103 / SM107.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 128

    # --- warp assignments (8 warps total) ---
    COMPUTE_GROUP_1_WARP_IDS: Tuple[int, ...] = (0, 1, 2, 3)
    LOAD_GATE_WARP_ID: int = 4
    TMA_KV_WARP_ID: int = 5
    TCGEN05_MMA_WARP_ID: int = 6
    EPILOGUE_WARP_ID: int = 7

    # --- register split ---
    NUM_REGS_COMPUTE_GROUP_1: int = 256
    NUM_REGS_OTHER: int = 40

    THREADS_PER_WARP: int = 32

    CLUSTER_SHAPE_MNK: Tuple[int, int, int] = (1, 1, 1)

    # --- SMEM stage counts ---
    SMEM_SCHEDULER_STAGES: int = 2
    SMEM_K_STAGES: int = 4
    SMEM_V_STAGES: int = 2
    SMEM_STATE_STAGES: int = 1
    SMEM_U_STAGES: int = 1
    SMEM_GATE_STAGES: int = 3

    # --- TMEM stage counts ---
    TMEM_STATE_ACC_STAGES: int = 1

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
