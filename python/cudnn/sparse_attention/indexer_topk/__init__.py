# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.sparse_attention.indexer_topk: fused indexer scoring + top-k selection."""

from typing import Any

_LAZY_EXPORTS = {
    "IndexerTopK": ("cudnn.sparse_attention.indexer_topk.api", "IndexerTopK"),
    "indexer_topk_wrapper": ("cudnn.sparse_attention.indexer_topk.api", "indexer_topk_wrapper"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
