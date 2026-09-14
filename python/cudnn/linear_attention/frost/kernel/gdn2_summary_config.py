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

"""GDN-2 fused state-summary kernel config: fixed compile-time constants of the BT=16 schedule that runs the H
recurrence (state from zero or an initial state) and the M recurrence (identity seed, zero values: the piece
transition) in lockstep on one 16-warp CTA per (piece, head).  Derived SMEM / TMEM sizes and offsets are stamped
by ``build_cfg`` in ``gdn2_summary_f16.py``.  Target arch: Blackwell SM100 / SM103.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 16

    # --- warp assignments (16 warps = 512 threads) ---
    COMPUTE_GROUP_0_WARP_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    COMPUTE_GROUP_1_WARP_IDS: Tuple[int, ...] = (8, 9, 10, 11)
    SUPER_MMA_WARP_ID: int = 12
    TCGEN05_MMA_WARP_ID: int = 13
    TMA_WARP_ID: int = 14
    SUPER_MMA_TWIN_WARP_ID: int = 15

    # --- register split ---
    NUM_REGS_COMPUTE_GROUP_0: int = 160
    NUM_REGS_COMPUTE_GROUP_1: int = 136
    NUM_REGS_OTHER: int = 56

    THREADS_PER_WARP: int = 32

    BUFFER_ALIGN_BYTES: int = 1024

    # --- SMEM / TMEM ring stage counts ---
    SMEM_RAW_STAGES: int = 5
    SMEM_SCHEDULER_STAGES: int = 8
    SMEM_DECAY_STAGES: int = 2
    SMEM_INTERMEDIATE_STAGES: int = 2
    QK_SCALE_READY_STAGES: int = 4

    CLUSTER_SHAPE_MNK: Tuple[int, int, int] = (1, 1, 1)


CFG = Cfg()
