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
Fixed constants of the Mamba-2 backward summary (state-gradient-only) kernel: warp roles, SMEM / TMEM stage counts.
The per-compile knobs (dtypes, the d_final_state port) are ``Mamba2BpropSummaryCfg`` fields.

Target arch: SM100 / SM103 / SM107.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 128

    # --- warp assignments (11 warps total, one register budget: no setmaxnreg) ---
    COMPUTE_GROUP_1_WARP_IDS: Tuple[int, ...] = (0, 1, 2, 3)
    COMPUTE_GROUP_2_WARP_IDS: Tuple[int, ...] = (4, 5, 6, 7)
    LOAD_GATE_WARP_ID: int = 8
    TMA_QDO_WARP_ID: int = 9
    TCGEN05_MMA_WARP_ID: int = 10

    THREADS_PER_WARP: int = 32

    CLUSTER_SHAPE_MNK: Tuple[int, int, int] = (1, 1, 1)
    SMEM_SCHEDULER_STAGES: int = 2

    # --- SMEM stage counts ---
    SMEM_Q_STAGES: int = 1
    SMEM_DO_STAGES: int = 2
    SMEM_GATE_STAGES: int = 2

    # --- TMEM stage counts ---
    TMEM_DSTATE_ACC_STAGES: int = 1

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
