# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level CuTe/DLPack helpers shared by the sm_100 norm kernels.

Mirrors the role of ``cudnn.sdpa.utils``: DLPack -> cute.Tensor conversion and a
couple of small launch-side helpers. The heavy device-side primitives (cp.async
staging, block reductions) live in each pass's ``kernels/_common_sm100.py`` and
build on ``cudnn.frost.tile_dsl`` + ``cutlass.primitives``.
"""

from __future__ import annotations

from cutlass.cute.runtime import from_dlpack


def dyn(t):
    """Wrap a torch tensor as a dynamic-layout cute tensor (16-byte aligned)."""
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()
