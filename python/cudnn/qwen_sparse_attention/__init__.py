# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

_SYMBOLS = {
    "SparseAttentionBackward": (".sparse_attention_backward", "SparseAttentionBackward"),
    "sparse_attention_backward_wrapper": (".sparse_attention_backward", "sparse_attention_backward_wrapper"),
}


def _load_symbol(name):
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    if name == "QSA":
        return QSA
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class QSANamespace:
    def __getattr__(self, name):
        if name in _SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"QSA has no attribute {name!r}")


QSA = QSANamespace()

__all__ = ["QSA", *_SYMBOLS.keys()]
