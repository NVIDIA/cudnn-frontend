# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX integration for frontend-only CuTe DSL operations.

Importing this namespace is the explicit opt-in boundary for the JAX optional
dependencies. The base :mod:`cudnn` package does not import JAX or CuTe DSL.
"""

from importlib import import_module
from types import SimpleNamespace


try:
    import_module("jax")
    import_module("cutlass.jax")
except ModuleNotFoundError as exc:
    if exc.name in {"jax", "cutlass", "cutlass.jax"}:
        raise ImportError(
            "cudnn.jax requires the JAX optional dependencies; install them "
            "with 'pip install nvidia-cudnn-frontend[jax]'"
        ) from exc
    raise

from .indexer_forward import IndexerForwardResult, indexer_forward_wrapper
from .indexer_top_k import IndexerTopKResult, indexer_top_k_wrapper
from .rmsnorm_rht_amax import RmsNormRhtAmaxResult, rmsnorm_rht_amax_sm100


DSA = SimpleNamespace(
    indexer_forward_wrapper=indexer_forward_wrapper,
    indexer_top_k_wrapper=indexer_top_k_wrapper,
)

__all__ = [
    "DSA",
    "IndexerForwardResult",
    "IndexerTopKResult",
    "RmsNormRhtAmaxResult",
    "indexer_forward_wrapper",
    "indexer_top_k_wrapper",
    "rmsnorm_rht_amax_sm100",
]
