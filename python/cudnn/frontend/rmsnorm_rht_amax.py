# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Semantic catalog entry for RMSNorm + RHT + amax."""

from __future__ import annotations

from typing import Any, Optional

from ._registry import FrontendTarget, TargetBinding, frontend_operation

_COMMON_PARAMETER_NAMES = {
    "x": "x",
    "weight": "weight",
    "eps": "eps",
    "num_threads": "num_threads",
    "rows_per_cta": "rows_per_cta",
}
_COMMON_OUTPUT_NAMES = {
    "output": "output",
    "amax": "amax",
}


@frontend_operation(
    name="rmsnorm_rht_amax_sm100",
    targets={
        FrontendTarget.TORCH: TargetBinding(
            module="cudnn.rmsnorm_rht_amax.api",
            symbol="rmsnorm_rht_amax_wrapper_sm100",
            parameter_map={
                **_COMMON_PARAMETER_NAMES,
                "x": "x_tensor",
                "weight": "w_tensor",
            },
            output_map={
                **_COMMON_OUTPUT_NAMES,
                "output": "o_tensor",
                "amax": "amax_tensor",
            },
            target_only_parameters=("current_stream",),
        ),
        FrontendTarget.JAX: TargetBinding(
            module="cudnn.jax.rmsnorm_rht_amax",
            symbol="rmsnorm_rht_amax_sm100",
            parameter_map=_COMMON_PARAMETER_NAMES,
            output_map=_COMMON_OUTPUT_NAMES,
        ),
    },
    api_anchors=(
        "cudnn.rmsnorm_rht_amax.api:RmsNormRhtAmaxSm100",
        "cudnn.rmsnorm_rht_amax.api:rmsnorm_rht_amax_wrapper_sm100",
    ),
    kernel_anchors=("cudnn.rmsnorm_rht_amax.kernel:RMSNormRHTAmaxKernel.kernel",),
    output_names=("output", "amax"),
    parity_case="rmsnorm_rht_amax",
)
def _rmsnorm_rht_amax_sm100_contract(
    x: Any,
    weight: Any,
    *,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
) -> None:
    """Framework-neutral semantic parameters and defaults for this operation."""


__all__ = []
