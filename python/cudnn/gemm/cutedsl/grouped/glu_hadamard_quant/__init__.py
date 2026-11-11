# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from .api import (
    GroupedGemmGluHadamardQuantSm100,
    grouped_gemm_glu_hadamard_quant_wrapper_sm100,
)

__all__ = [
    "GroupedGemmGluHadamardQuantSm100",
    "grouped_gemm_glu_hadamard_quant_wrapper_sm100",
    "grouped_gemm_glu_hadamard_quant_jax_forward",
]


def __getattr__(name):
    # Lazy: the jax entry point imports jax/cutlass.jax, which must not be pulled in
    # for torch-only users.
    if name == "grouped_gemm_glu_hadamard_quant_jax_forward":
        from .jax_api import grouped_gemm_glu_hadamard_quant_jax_forward

        return grouped_gemm_glu_hadamard_quant_jax_forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
