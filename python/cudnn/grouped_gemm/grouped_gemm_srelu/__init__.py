# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Grouped GEMM SReLU Kernel Module

This module provides the forward grouped GEMM with SReLU activation
for MoE (Mixture of Experts) workloads on SM100+ GPUs.
"""

from .api import (
    GroupedGemmSreluSm100,
    grouped_gemm_srelu_wrapper_sm100,
)

__all__ = [
    "GroupedGemmSreluSm100",
    "grouped_gemm_srelu_wrapper_sm100",
]
