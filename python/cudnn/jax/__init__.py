# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX integration for frontend-only CuTe DSL operations."""

from importlib import import_module
from typing import Any

_SYMBOLS = {
    "IndexerForwardResult": (".indexer_forward", "IndexerForwardResult"),
    "IndexerTopKResult": (".indexer_top_k", "IndexerTopKResult"),
    "RmsNormRhtAmaxResult": (".rmsnorm_rht_amax", "RmsNormRhtAmaxResult"),
    "indexer_forward_wrapper": (".indexer_forward", "indexer_forward_wrapper"),
    "indexer_top_k_wrapper": (".indexer_top_k", "indexer_top_k_wrapper"),
    "rmsnorm_rht_amax_sm100": (".rmsnorm_rht_amax", "rmsnorm_rht_amax_sm100"),
}
_DSA_SYMBOLS = frozenset(("indexer_forward_wrapper", "indexer_top_k_wrapper"))


def _load_symbol(name: str) -> Any:
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name: str) -> Any:
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _DSANamespace:
    """JAX counterparts of the existing ``cudnn.DSA`` wrapper names."""

    def __getattr__(self, name: str) -> Any:
        if name in _DSA_SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"JAX DSA has no attribute {name!r}")


DSA = _DSANamespace()

__all__ = ["DSA", *_SYMBOLS]
