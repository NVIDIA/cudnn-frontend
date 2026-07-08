# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={"api": ("GroupedGemmDsreluSm100", "grouped_gemm_dsrelu_wrapper_sm100")},
    submodules=("api", "jax"),
)
