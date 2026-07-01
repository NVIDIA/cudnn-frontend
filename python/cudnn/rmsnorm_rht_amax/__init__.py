# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy framework exports for RMSNorm + RHT + amax.

``api.py`` is always the Torch implementation and ``jax.py`` is always the
JAX implementation. The unqualified package API prefers Torch when available
and otherwise exposes the JAX API.
"""

from .._framework_api import make_framework_api


_TORCH_EXPORTS = (
    "RmsNormRhtAmaxSm100",
    "best_num_threads",
    "pick_rows_per_cta",
    "rmsnorm_rht_amax_sm100",
    "rmsnorm_rht_amax_wrapper_sm100",
)
_JAX_EXPORTS = (
    "RmsNormRhtAmaxResult",
    "rmsnorm_rht_amax_sm100",
)

__all__, __getattr__ = make_framework_api(
    globals(),
    torch_exports=_TORCH_EXPORTS,
    jax_exports=_JAX_EXPORTS,
)
