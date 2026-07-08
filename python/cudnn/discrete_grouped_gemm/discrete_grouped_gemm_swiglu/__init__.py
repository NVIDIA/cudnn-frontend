# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API exports for discrete grouped GEMM + SwiGLU."""

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "api": (
            "DiscreteGroupedGemmSwigluSm100",
            "discrete_grouped_gemm_swiglu_wrapper_sm100",
        ),
    },
    submodules=("api",),
)
