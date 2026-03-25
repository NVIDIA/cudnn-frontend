# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Discrete-weight Grouped GEMM GLU Kernel Module

This module provides forward and backward discrete-weight grouped GEMM with GLU
activation (SwiGLU/GeGLU) for MoE (Mixture of Experts) workloads on SM100+ GPUs.
"""

from .discrete_grouped_gemm_swiglu import (
    DiscreteGroupedGemmSwigluSm100,
    discrete_grouped_gemm_swiglu_wrapper_sm100,
)

from .discrete_grouped_gemm_dswiglu import (
    DiscreteGroupedGemmDswigluSm100,
    discrete_grouped_gemm_dswiglu_wrapper_sm100,
)

__all__ = [
    "DiscreteGroupedGemmSwigluSm100",
    "discrete_grouped_gemm_swiglu_wrapper_sm100",
    "DiscreteGroupedGemmDswigluSm100",
    "discrete_grouped_gemm_dswiglu_wrapper_sm100",
]
