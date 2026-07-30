# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

_SYMBOLS = {
    "block_sparse_attention_forward": (".api", "block_sparse_attention_forward"),
    "block_sparse_attention_backward": (".api", "block_sparse_attention_backward"),
}


def _load_symbol(name):
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    if name == "BSA":
        return BSA
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class BSANamespace:
    def __getattr__(self, name):
        if name in _SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"BSA has no attribute {name!r}")


BSA = BSANamespace()

__all__ = ["BSA", *_SYMBOLS.keys()]
