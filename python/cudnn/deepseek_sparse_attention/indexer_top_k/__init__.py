# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy DSA Top-K exports; selecting one API does not import sibling kernels."""

from importlib import import_module

_SYMBOLS = {
    "IndexerTopK": (".api", "IndexerTopK"),
    "indexer_top_k_wrapper": (".api", "indexer_top_k_wrapper"),
    "local_to_global_wrapper": (".api", "local_to_global_wrapper"),
    "compactify_wrapper": (".api", "compactify_wrapper"),
    "IndexerTopKVarlen": (".varlen_api", "IndexerTopKVarlen"),
    "indexer_top_k_varlen_wrapper": (".varlen_api", "indexer_top_k_varlen_wrapper"),
}


def __getattr__(name):
    if name not in _SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _SYMBOLS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


__all__ = list(_SYMBOLS)
