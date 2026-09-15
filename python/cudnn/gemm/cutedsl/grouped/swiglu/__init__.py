# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    "grouped_gemm_swiglu_jax_sm100",
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
]


def __getattr__(name):
    if name == "grouped_gemm_swiglu_jax_sm100":
        from .jax_api import grouped_gemm_swiglu_jax_sm100

        return grouped_gemm_swiglu_jax_sm100
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
