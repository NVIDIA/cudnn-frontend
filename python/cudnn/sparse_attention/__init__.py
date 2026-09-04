# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.sparse_attention: variant-neutral, index-driven sparse attention.

The generic ops parameterized by per-query index lists (token / block /
micro-block granularity; per-layer / per-KV-head-group / per-head index
scope). Architecture-specific pipelines (indexers, top-k, compressors) live
in their own packages (``cudnn.deepseek_sparse_attention``,
``cudnn.native_sparse_attention``, ``cudnn.csa``); mask-driven block sparsity
lives in ``cudnn.block_sparse_attention``.

Components live one level down (``cudnn.sparse_attention.fwd``,
``cudnn.sparse_attention.indexer_topk``, later ``.bwd``), mirroring
``cudnn.sdpa``; the family level lazily re-exports the high-level wrappers.
"""

from typing import Any

_LAZY_EXPORTS = {
    "SparseAttentionForward": ("cudnn.sparse_attention.fwd", "SparseAttentionForward"),
    "sparse_attention_forward_wrapper": ("cudnn.sparse_attention.fwd", "sparse_attention_forward_wrapper"),
    "IndexerTopK": ("cudnn.sparse_attention.indexer_topk", "IndexerTopK"),
    "indexer_topk_wrapper": ("cudnn.sparse_attention.indexer_topk", "indexer_topk_wrapper"),
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
