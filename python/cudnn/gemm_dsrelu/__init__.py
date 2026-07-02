# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy API exports for dense GEMM + squared-ReLU backward."""

from .._operation_api import make_operation_api

_API_EXPORTS = (
    "GemmDsreluSm100",
    "gemm_dsrelu_wrapper_sm100",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
