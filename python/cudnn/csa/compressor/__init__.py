# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy public surface for the fused CSA/HCA compressor kernels."""

from importlib import import_module

_SYMBOLS = {
    "CSACompressorForward": (".api", "CSACompressorForward"),
    "CSACompressorBackward": (".api", "CSACompressorBackward"),
    "csa_compressor_forward_wrapper": (".api", "csa_compressor_forward_wrapper"),
    "csa_compressor_backward_wrapper": (".api", "csa_compressor_backward_wrapper"),
    "csa_compressor_forward_jax_sm100": (".jax_api", "csa_compressor_forward_jax_sm100"),
    "csa_compressor_backward_jax_sm100": (".jax_api", "csa_compressor_backward_jax_sm100"),
}

__all__ = list(_SYMBOLS)


def __getattr__(name):
    if name in _SYMBOLS:
        module_name, symbol_name = _SYMBOLS[name]
        value = getattr(import_module(module_name, package=__name__), symbol_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
