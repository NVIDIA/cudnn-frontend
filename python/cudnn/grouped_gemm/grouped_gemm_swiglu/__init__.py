# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Grouped GEMM SwiGLU APIs for SM100."""

from ..._operation_api import make_operation_api

_API_EXPORTS = (
    "GroupedGemmSwigluSm100",
    "grouped_gemm_swiglu_wrapper_sm100",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
