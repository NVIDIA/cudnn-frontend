# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .api import (
    GroupedGemmDswigluSm100,
    grouped_gemm_dswiglu_wrapper_sm100,
)

__all__ = [
    "grouped_gemm_dswiglu_jax_sm100",
    "GroupedGemmDswigluSm100",
    "grouped_gemm_dswiglu_wrapper_sm100",
]


def __getattr__(name):
    if name == "grouped_gemm_dswiglu_jax_sm100":
        from .jax_api import grouped_gemm_dswiglu_jax_sm100

        return grouped_gemm_dswiglu_jax_sm100
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
