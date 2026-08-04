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

"""Kimi Delta Attention (KDA) Cutlass DSL backward kernel config — STUB.

The FROST KDA backward is not implemented yet (see ``kda_bprop_f16.py``);
these constants mirror the prefill config's tile shape for when the backward
kernel lands.

Target arch: Blackwell SM100 (GB200) / SM103 (GB300).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    B_T: int = 16
    D_K: int = 128
    D_V: int = 128


CFG = Cfg()
