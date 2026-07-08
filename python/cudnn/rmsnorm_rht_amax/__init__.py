# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy API exports for RMSNorm + RHT + amax.

Unqualified symbols always come from the Torch ``api.py``. The JAX API is
available explicitly through the sibling ``jax`` namespace.
"""

from .._operation_api import make_operation_api


_API_EXPORTS = (
    "RmsNormRhtAmaxSm100",
    "best_num_threads",
    "pick_rows_per_cta",
    "rmsnorm_rht_amax_sm100",
    "rmsnorm_rht_amax_wrapper_sm100",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
