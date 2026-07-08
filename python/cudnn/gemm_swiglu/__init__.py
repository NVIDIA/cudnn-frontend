# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API and framework-neutral GEMM + SwiGLU operation exports."""

from .._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": ("GemmSwigluSm100Op",),
        "api": (
            "GemmSwigluSm100",
            "gemm_swiglu_wrapper_sm100",
        ),
    },
    submodules=("api", "op"),
)
