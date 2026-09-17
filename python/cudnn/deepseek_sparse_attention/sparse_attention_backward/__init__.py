# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".api", package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SparseAttentionBackward", "sparse_attention_backward_wrapper"]
