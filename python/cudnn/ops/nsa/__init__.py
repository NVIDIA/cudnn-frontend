# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native sparse attention (NSA) ops.

Public symbols resolve on first attribute access so importing ``cudnn.ops.nsa``
requires neither PyTorch nor JAX.
"""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "selection_attention_forward": ("cudnn.native_sparse_attention.selection", "selection_attention_wrapper"),
    "compression_attention_forward": ("cudnn.native_sparse_attention.compression", "compression_attention_wrapper"),
    "sliding_window_attention_forward": ("cudnn.native_sparse_attention.sliding_window_attention", "sliding_window_attention_wrapper"),
    "top_k_reduction_forward": ("cudnn.native_sparse_attention.top_k", "topk_reduction_wrapper"),
    "selection_attention_forward_jax": ("cudnn.native_sparse_attention.jax_api", "selection_attention_forward"),
    "compression_attention_forward_jax": ("cudnn.native_sparse_attention.jax_api", "compression_attention_forward"),
    "top_k_reduction_forward_jax": ("cudnn.native_sparse_attention.jax_api", "top_k_reduction_forward"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        value = getattr(importlib.import_module(module_name), attr_name)
    except ImportError as error:
        from cudnn import _optional_dependency_message

        raise ImportError(_optional_dependency_message(name, error)) from error

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
