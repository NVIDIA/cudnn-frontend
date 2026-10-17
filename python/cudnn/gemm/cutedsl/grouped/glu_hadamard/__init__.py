# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .api import (
    GroupedGemmGluHadamardSm100,
    grouped_gemm_glu_hadamard_wrapper_sm100,
)

__all__ = [
    "GroupedGemmGluHadamardSm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
    "grouped_gemm_glu_hadamard_jax_forward",
]


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "grouped_gemm_glu_hadamard_jax_forward":
        from .jax_api import grouped_gemm_glu_hadamard_jax_forward

        return grouped_gemm_glu_hadamard_jax_forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
