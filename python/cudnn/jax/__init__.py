# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX facade for co-located frontend-only operation APIs.

Importing :mod:`cudnn.jax` is the explicit dependency boundary for JAX and
CuTe DSL. Dependencies are validated once here, public wrapper modules are then
imported, and architecture-specific kernel modules remain deferred until an
operation is traced.
"""

from importlib import import_module
from types import SimpleNamespace

_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"


def _require_dependencies(module_names: list[str]) -> None:
    for module_name in module_names:
        try:
            import_module(module_name)
        except ModuleNotFoundError as error:
            missing_name = error.name
            if missing_name is None or not (module_name == missing_name or module_name.startswith(f"{missing_name}.")):
                raise
            raise ImportError(f"cudnn.jax requires the {module_name!r} module. Install the " f"JAX integration with `{_INSTALL_HINT}`.") from error


_require_dependencies(["jax", "cutlass"])

import cutlass.jax
import jax

if not cutlass.jax.is_available():
    minimum_version_info = tuple(cutlass.jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION)
    minimum_version = ".".join(str(part) for part in minimum_version_info)
    installed_version = getattr(jax, "__version__", "unknown")
    reason = f"CUTLASS JAX support is unavailable with JAX {installed_version}; " f"the minimum supported JAX version is {minimum_version}."
    raise ImportError(f"cudnn.jax cannot be imported because {reason} Install the JAX " f"integration with `{_INSTALL_HINT}`.")

from ..deepseek_sparse_attention.indexer_forward.jax import (
    IndexerForwardResult,
    indexer_forward_wrapper,
)
from ..deepseek_sparse_attention.indexer_top_k.jax import (
    IndexerTopKResult,
    indexer_top_k_wrapper,
)
from ..deepseek_sparse_attention.score_recompute.jax import (
    SparseAttnScoreRecomputeResult,
    SparseIndexerScoreRecomputeResult,
    sparse_attn_score_recompute_wrapper,
    sparse_indexer_score_recompute_wrapper,
)
from ..rmsnorm_rht_amax.jax import (
    RmsNormRhtAmaxResult,
    rmsnorm_rht_amax_sm100,
)

DSA = SimpleNamespace(
    indexer_forward_wrapper=indexer_forward_wrapper,
    indexer_top_k_wrapper=indexer_top_k_wrapper,
    sparse_indexer_score_recompute_wrapper=sparse_indexer_score_recompute_wrapper,
    sparse_attn_score_recompute_wrapper=sparse_attn_score_recompute_wrapper,
)

__all__ = [
    "DSA",
    "IndexerForwardResult",
    "IndexerTopKResult",
    "RmsNormRhtAmaxResult",
    "SparseAttnScoreRecomputeResult",
    "SparseIndexerScoreRecomputeResult",
    "indexer_forward_wrapper",
    "indexer_top_k_wrapper",
    "rmsnorm_rht_amax_sm100",
    "sparse_attn_score_recompute_wrapper",
    "sparse_indexer_score_recompute_wrapper",
]
