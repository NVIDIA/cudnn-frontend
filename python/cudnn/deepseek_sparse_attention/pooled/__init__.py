# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled DSA plans; public forward functions live in cudnn.ops.dsa."""

from importlib import import_module

from .backward import (
    pooled_expand_compact,
    pooled_indexer_backward,
    pooled_indexer_select,
    pooled_indexer_target,
    pooled_keys_forward,
    pooled_sparse_attention_backward,
    pooled_sparse_attention_backward_with_teacher,
    pooled_teacher_recompute,
    unpool_backward,
)

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


__all__ = [
    "pooled_expand_compact",
    "pooled_indexer_backward",
    "pooled_indexer_select",
    "pooled_indexer_target",
    "pooled_keys_forward",
    "pooled_sparse_attention_backward",
    "pooled_sparse_attention_backward_with_teacher",
    "pooled_teacher_recompute",
    "unpool_backward",
]
