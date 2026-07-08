# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy Torch API and framework-neutral operation/kernel exports.

JAX APIs are exported through :mod:`cudnn.jax`.
"""

from ..common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": (
            "RmsNormRhtAmaxSm100Op",
            "best_num_threads",
            "pick_rows_per_cta",
        ),
        "kernel": ("RMSNormRHTAmaxKernel",),
        "api": (
            "RmsNormRhtAmaxSm100",
            "rmsnorm_rht_amax_wrapper_sm100",
        ),
    },
    submodules=("api", "kernel", "op"),
)
