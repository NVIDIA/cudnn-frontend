# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy framework adapters for RMSNorm + RHT + amax."""

from .._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "kernel": (
            "RMSNormRHTAmaxKernel",
            "best_num_threads",
            "pick_rows_per_cta",
        ),
        "api": (
            "RmsNormRhtAmaxSm100",
            "rmsnorm_rht_amax_wrapper_sm100",
        ),
    },
    submodules=("api", "jax", "kernel"),
)
