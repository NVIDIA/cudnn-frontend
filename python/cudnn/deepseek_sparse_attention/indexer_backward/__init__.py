# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

__all__ = [
    "DenseIndexerBackward",
    "IndexerBackward",
    "dense_indexer_backward_wrapper",
    "indexer_backward_wrapper",
]


def __getattr__(name):
    # Lazy so the kernel modules (used by the JAX bindings) import without torch.
    if name in __all__:
        value = getattr(import_module(".api", package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
