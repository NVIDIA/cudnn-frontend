# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.sparse_attention.fwd: the sparse-attention forward direction."""

from typing import Any

_LAZY_EXPORTS = {
    "SparseAttentionForward": ("cudnn.sparse_attention.fwd.api", "SparseAttentionForward"),
    "sparse_attention_forward_wrapper": ("cudnn.sparse_attention.fwd.api", "sparse_attention_forward_wrapper"),
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
