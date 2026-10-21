# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    "grouped_gemm_srelu_jax_forward",
]


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "grouped_gemm_srelu_jax_forward":
        from .jax_api import grouped_gemm_srelu_jax_forward

        return grouped_gemm_srelu_jax_forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
