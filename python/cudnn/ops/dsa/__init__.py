# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek sparse attention operations."""

from importlib import import_module

_LAZY_EXPORTS = {
    "pooled_cache_update_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_cache_update_forward"),
    "pooled_indexer_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_indexer_forward"),
    "pooled_expand_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_expand_forward"),
    "pooled_select_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_select_forward"),
    "pooled_select_workspace_size": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_select_workspace_size"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _LAZY_EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
