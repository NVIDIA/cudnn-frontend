# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from ..._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "api": (
            "GroupedGemmGluHadamardSm100",
            "grouped_gemm_glu_hadamard_wrapper_sm100",
        )
    },
    submodules=("api", "jax"),
)
