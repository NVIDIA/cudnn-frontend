# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch exports for discrete-weight grouped GEMM operations.

JAX APIs are exported through :mod:`cudnn.jax`.
"""

from ..common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "discrete_grouped_gemm_swiglu": (
            "DiscreteGroupedGemmSwigluSm100",
            "discrete_grouped_gemm_swiglu_wrapper_sm100",
        ),
        "discrete_grouped_gemm_dswiglu": (
            "DiscreteGroupedGemmDswigluSm100",
            "discrete_grouped_gemm_dswiglu_wrapper_sm100",
        ),
    },
    submodules=(
        "discrete_grouped_gemm_dswiglu",
        "discrete_grouped_gemm_swiglu",
    ),
)
