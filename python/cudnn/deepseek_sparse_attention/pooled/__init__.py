# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal pooled DSA plans; public functions live in cudnn.ops.dsa."""

from importlib import import_module

_SYMBOLS = {
    "ExpandIndices": ".indices",
    "expand_indices_wrapper": ".indices",
    "PoolCacheUpdate": ".cache",
    "pool_cache_update_wrapper": ".cache",
    "IndexerForward": ".forward",
    "indexer_forward_wrapper": ".forward",
}


def __getattr__(name):
    if name in _SYMBOLS:
        value = getattr(import_module(_SYMBOLS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"PooledDSA has no attribute {name!r}")


__all__ = []
