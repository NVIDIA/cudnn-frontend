# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted for cuDNN Frontend from FlashInfer's
# flashinfer/gemm/batch_invariant/blocks/blockscaled_mx_mixed.py (e2d163fb3d46).
"""MXFP8 activations and packed MXFP4 weights on Blackwell and Rubin."""

from __future__ import annotations

import cutlass

from .arithmetic import SF_VEC
from .blockscaled import E8M0_DTYPE, BlockScaledPlan, BlockScaledPlanSM107

E4M3 = cutlass.Float8E4M3FN
E2M1 = cutlass.Float4E2M1FN
#: The sm_100 instruction's 8-bit smem container of a sub-byte operand (``kind::mxf8f6f4``).
CONTAINER = cutlass.Int8


class BlockScaledPlanMXW48(BlockScaledPlan):
    """e4m3 activations x e2m1 weights, UE8M0 scales per 32 on both: the sm_100 / sm_103 ``kind::mxf8f6f4`` instruction at
    32 elements per step; the e2m1 weight unpacked by the TMA into the 8-bit container the instruction reads (one element
    per byte: the instruction's sub-byte operand format, :data:`CONTAINER`)."""

    IN_FMT = "mxfp4w8a"
    A_DT = E4M3
    B_DT = E2M1
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp4w8a"]
    B_SMEM_DT = CONTAINER
    B_TMA_INTERNAL = CONTAINER


class BlockScaledPlanMXW48K64(BlockScaledPlanSM107):
    """e4m3 activations x e2m1 weights, UE8M0 scales per 32: the sm_107 ``kind::mxf8f6f4`` instruction at 64 elements per step
    (two scales per row and step); the packed e2m1 operand read directly (the sm_107 plan's TMA box and smem layout at
    4 bits per element)."""

    IN_FMT = "mxfp4w8a_k64"
    INST_K = 64
    A_DT = E4M3
    B_DT = E2M1
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp4w8a_k64"]


PLANS = {"mxfp4w8a": BlockScaledPlanMXW48, "mxfp4w8a_k64": BlockScaledPlanMXW48K64}
