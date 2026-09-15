# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subchannel-scaled dSwiGLU-backward grouped GEMM for MoE workloads."""

from .api import (
    GroupedGemmDswigluSubchannelScaledSm100,
    grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100,
)

__all__ = [
    "GroupedGemmDswigluSubchannelScaledSm100",
    "grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100",
]
