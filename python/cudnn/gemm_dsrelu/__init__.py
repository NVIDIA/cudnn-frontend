# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API and framework-neutral GEMM + dsReLU exports."""

from ..common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": ("GemmDsreluSm100Op",),
        "api": (
            "GemmDsreluSm100",
            "gemm_dsrelu_wrapper_sm100",
        ),
    },
    submodules=("api", "op"),
)
