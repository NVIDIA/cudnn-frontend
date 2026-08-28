# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variant-neutral, index-driven sparse attention.

This package hosts the generic sparse-attention ops parameterized by
per-query index lists (token / block / micro-block granularity, per-layer /
per-KV-head-group / per-head index scope). Architecture-specific pipelines
(indexers, top-k, compressors) live in their own packages
(:mod:`cudnn.deepseek_sparse_attention`, :mod:`cudnn.native_sparse_attention`,
:mod:`cudnn.csa`); mask-driven block sparsity lives in
:mod:`cudnn.block_sparse_attention`.
"""

from importlib import import_module

_SYMBOLS = {
    "SparseAttentionForward": (".forward", "SparseAttentionForward"),
    "sparse_attention_forward_wrapper": (".forward", "sparse_attention_forward_wrapper"),
}


def _load_symbol(name):
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*_SYMBOLS.keys()]
