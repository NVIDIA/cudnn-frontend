# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek sparse attention operations."""

from importlib import import_module

_LAZY_EXPORTS = {
    "compressor_forward_jax": ("cudnn.csa.compressor.jax_api", "compressor_forward_jax"),
    "compressor_backward_jax": ("cudnn.csa.compressor.jax_api", "compressor_backward_jax"),
    "indexer_forward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "indexer_forward_jax"),
    "indexer_backward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "indexer_backward_jax"),
    "dense_indexer_backward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "dense_indexer_backward_jax"),
    "indexer_top_k_forward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "indexer_top_k_forward_jax"),
    "sparse_attention_forward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "sparse_attention_forward_jax"),
    "sparse_attention_backward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "sparse_attention_backward_jax"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _LAZY_EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
