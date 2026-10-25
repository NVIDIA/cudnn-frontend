# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek sparse attention operations."""

from importlib import import_module

_LAZY_EXPORTS = {
    "compressor_forward": ("cudnn.csa", "csa_compressor_forward_wrapper"),
    "compressor_backward": ("cudnn.csa", "csa_compressor_backward_wrapper"),
    "indexer_forward": ("cudnn.deepseek_sparse_attention", "indexer_forward_wrapper"),
    "indexer_backward": ("cudnn.deepseek_sparse_attention", "indexer_backward_wrapper"),
    "dense_indexer_backward": ("cudnn.deepseek_sparse_attention", "dense_indexer_backward_wrapper"),
    "top_k_forward": ("cudnn.deepseek_sparse_attention", "indexer_top_k_wrapper"),
    "sparse_attention_forward": ("cudnn.deepseek_sparse_attention", "sparse_attention_forward_wrapper"),
    "sparse_attention_backward": ("cudnn.deepseek_sparse_attention", "sparse_attention_backward_wrapper"),
    "compressor_forward_jax": ("cudnn.csa.compressor.jax_api", "compressor_forward_jax"),
    "compressor_backward_jax": ("cudnn.csa.compressor.jax_api", "compressor_backward_jax"),
    "indexer_forward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "indexer_forward_jax"),
    "indexer_backward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "indexer_backward_jax"),
    "dense_indexer_backward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "dense_indexer_backward_jax"),
    "top_k_forward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "top_k_forward_jax"),
    "sparse_attention_forward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "sparse_attention_forward_jax"),
    "sparse_attention_backward_jax": ("cudnn.deepseek_sparse_attention.jax_api", "sparse_attention_backward_jax"),
    "pooled_cache_update_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_cache_update_forward"),
    "pooled_indexer_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_indexer_forward"),
    "pooled_expand_forward": ("cudnn.deepseek_sparse_attention.pooled.api", "pooled_expand_forward"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _LAZY_EXPORTS[name]
    try:
        value = getattr(import_module(module), symbol)
    except ImportError as error:
        from cudnn import _optional_dependency_message

        raise ImportError(_optional_dependency_message(name, error)) from error
    globals()[name] = value
    return value
