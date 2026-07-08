# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from ..._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={"api": ("GroupedGemmDswigluSm100", "grouped_gemm_dswiglu_wrapper_sm100")},
    submodules=("api", "jax"),
)
