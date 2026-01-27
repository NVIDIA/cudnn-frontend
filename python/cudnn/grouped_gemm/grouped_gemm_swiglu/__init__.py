# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Grouped GEMM SwiGLU Kernel Module

This module provides the forward grouped GEMM with SwiGLU activation
for MoE (Mixture of Experts) workloads on SM100+ GPUs.
"""

from .api import (
    GroupedGemmSwigluSm100,
    grouped_gemm_swiglu_wrapper_sm100,
)

__all__ = [
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
]
