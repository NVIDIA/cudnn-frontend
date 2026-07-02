# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lazy exports for grouped GEMM operation packages."""

from importlib import import_module

_SYMBOLS = {
    "GroupedGemmSwigluSm100": (".grouped_gemm_swiglu", "GroupedGemmSwigluSm100"),
    "grouped_gemm_swiglu_wrapper_sm100": (".grouped_gemm_swiglu", "grouped_gemm_swiglu_wrapper_sm100"),
    "GroupedGemmDswigluSm100": (".grouped_gemm_dswiglu", "GroupedGemmDswigluSm100"),
    "grouped_gemm_dswiglu_wrapper_sm100": (".grouped_gemm_dswiglu", "grouped_gemm_dswiglu_wrapper_sm100"),
    "GroupedGemmQuantSm100": (".grouped_gemm_quant", "GroupedGemmQuantSm100"),
    "grouped_gemm_quant_wrapper_sm100": (".grouped_gemm_quant", "grouped_gemm_quant_wrapper_sm100"),
    "GroupedGemmSreluSm100": (".grouped_gemm_srelu", "GroupedGemmSreluSm100"),
    "grouped_gemm_srelu_wrapper_sm100": (".grouped_gemm_srelu", "grouped_gemm_srelu_wrapper_sm100"),
    "GroupedGemmDsreluSm100": (".grouped_gemm_dsrelu", "GroupedGemmDsreluSm100"),
    "grouped_gemm_dsrelu_wrapper_sm100": (".grouped_gemm_dsrelu", "grouped_gemm_dsrelu_wrapper_sm100"),
    "GroupedGemmGluSm100": (".grouped_gemm_glu", "GroupedGemmGluSm100"),
    "grouped_gemm_glu_wrapper_sm100": (".grouped_gemm_glu", "grouped_gemm_glu_wrapper_sm100"),
    "GroupedGemmGluHadamardSm100": (".grouped_gemm_glu_hadamard", "GroupedGemmGluHadamardSm100"),
    "grouped_gemm_glu_hadamard_wrapper_sm100": (".grouped_gemm_glu_hadamard", "grouped_gemm_glu_hadamard_wrapper_sm100"),
    "GroupedGemmDgluSm100": (".grouped_gemm_dglu", "GroupedGemmDgluSm100"),
    "grouped_gemm_dglu_wrapper_sm100": (".grouped_gemm_dglu", "grouped_gemm_dglu_wrapper_sm100"),
    "GroupedGemmWgradSm100": (".grouped_gemm_wgrad", "GroupedGemmWgradSm100"),
    "grouped_gemm_wgrad_wrapper_sm100": (".grouped_gemm_wgrad", "grouped_gemm_wgrad_wrapper_sm100"),
}


def __getattr__(name):
    if name in _SYMBOLS:
        module_name, symbol_name = _SYMBOLS[name]
        value = getattr(import_module(module_name, __name__), symbol_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value


__all__ = list(_SYMBOLS)
