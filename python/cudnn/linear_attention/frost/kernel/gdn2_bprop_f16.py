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

"""Gated DeltaNet v2 (GDN-2) Cutlass DSL backward kernel — STUB.

The FROST GDN-2 backward is not implemented yet.  Because the GDN-2 chunk
size is small, the design recomputes the forward per-chunk states inside the
backward (no H store in the prefill kernel).  Until the backward kernel
lands, this module raises ``NotImplementedError``.

Target arch: Blackwell SM100 (GB200) / SM103 (GB300).
"""

from __future__ import annotations


def get_workspace_size(B: int, HQ: int, HV: int) -> int:
    return 0


def chunk_gdn2_bwd_sm100(*args, **kwargs) -> None:
    """Not implemented — the FROST GDN-2 backward kernel is a stub."""
    raise NotImplementedError("FROST GDN-2 backward is not implemented yet (recompute-in-bprop kernel is a stub).")
