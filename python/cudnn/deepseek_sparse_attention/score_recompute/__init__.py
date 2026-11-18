# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

__all__ = [
    "SparseIndexerScoreRecompute",
    "sparse_indexer_score_recompute_wrapper",
    "SparseAttnScoreRecompute",
    "sparse_attn_score_recompute_wrapper",
    "DenseIndexerScoreRecompute",
    "dense_indexer_score_recompute_wrapper",
    "DenseAttnScoreRecompute",
    "dense_attn_score_recompute_wrapper",
]


def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".api", package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
