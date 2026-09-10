# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subchannel-scaled (second-level SFA2) grouped GEMM weight gradient for MoE workloads."""

from .api import (
    GroupedGemmWgradSubchannelScaledSm100,
    grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100,
)

__all__ = [
    "GroupedGemmWgradSubchannelScaledSm100",
    "grouped_gemm_wgrad_subchannel_scaled_wrapper_sm100",
]
