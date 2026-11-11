# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simplified sparse attention with a trainable indexer."""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import SimplifiedSparseAttentionMetadata, dense_indexer_kl, simplified_sparse_attention, simplified_sparse_attention_reserve
    from .jax_api import simplified_sparse_attention_jax

__all__ = [
    "SimplifiedSparseAttentionMetadata",
    "dense_indexer_kl",
    "simplified_sparse_attention",
    "simplified_sparse_attention_reserve",
    "simplified_sparse_attention_jax",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".jax_api" if name == "simplified_sparse_attention_jax" else ".api"
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
