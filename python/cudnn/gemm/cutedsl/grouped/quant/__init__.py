# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    "grouped_gemm_quant_jax_forward",
]


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "grouped_gemm_quant_jax_forward":
        from .jax_api import grouped_gemm_quant_jax_forward

        return grouped_gemm_quant_jax_forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
