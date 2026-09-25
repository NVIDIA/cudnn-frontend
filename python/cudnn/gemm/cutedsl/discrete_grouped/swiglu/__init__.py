# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .api import (
    DiscreteGroupedGemmSwigluSm100,
    discrete_grouped_gemm_swiglu_wrapper_sm100,
)

__all__ = [
    "DiscreteGroupedGemmSwigluSm100",
    "discrete_grouped_gemm_swiglu_wrapper_sm100",
    "discrete_grouped_gemm_swiglu_jax_sm100",
]


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "discrete_grouped_gemm_swiglu_jax_sm100":
        from .jax_api import discrete_grouped_gemm_swiglu_jax_sm100

        return discrete_grouped_gemm_swiglu_jax_sm100
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
