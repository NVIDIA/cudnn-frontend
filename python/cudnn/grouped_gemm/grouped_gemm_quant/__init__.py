# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Grouped GEMM Quant Kernel Module

This module provides the contiguous grouped GEMM with output quantization
for MoE (Mixture of Experts) workloads on SM100+ GPUs.
Used for FC2 (forward down-projection) and dFC1 (backward FC1 GEMMs).
"""

from .api import (
    GroupedGemmQuantSm100,
    grouped_gemm_quant_wrapper_sm100,
)

__all__ = [
    "GroupedGemmQuantSm100",
    "grouped_gemm_quant_wrapper_sm100",
]
