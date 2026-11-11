# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

# Lazy exports (PEP 562): ``api`` imports torch, which the JAX path must not require.
_SYMBOLS = {
    "IndexerTopK": ".api",
    "indexer_top_k_wrapper": ".api",
    "local_to_global_wrapper": ".api",
    "compactify_wrapper": ".api",
}

__all__ = list(_SYMBOLS)


def __getattr__(name):
    if name in _SYMBOLS:
        value = getattr(import_module(_SYMBOLS[name], package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
