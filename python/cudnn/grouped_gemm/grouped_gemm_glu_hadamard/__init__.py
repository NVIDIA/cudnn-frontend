# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy API exports for grouped GEMM + GLU + Hadamard."""

from ..._operation_api import make_operation_api

_API_EXPORTS = (
    "GroupedGemmGluHadamardSm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
