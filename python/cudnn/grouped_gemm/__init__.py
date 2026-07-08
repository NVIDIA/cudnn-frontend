# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API exports for the grouped GEMM operation family."""

from .._operation_api import make_operation_api

_OPERATIONS = (
    "swiglu",
    "dswiglu",
    "quant",
    "srelu",
    "dsrelu",
    "glu",
    "glu_hadamard",
    "dglu",
    "wgrad",
)

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        f"grouped_gemm_{operation}.api": (
            f"GroupedGemm{''.join(part.capitalize() for part in operation.split('_'))}Sm100",
            f"grouped_gemm_{operation}_wrapper_sm100",
        )
        for operation in _OPERATIONS
    },
    submodules=tuple(f"grouped_gemm_{operation}" for operation in _OPERATIONS),
)
