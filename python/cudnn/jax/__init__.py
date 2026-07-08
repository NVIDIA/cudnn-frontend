# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Public optional-dependency boundary for JAX operation APIs.

JAX and CUTLASS are validated eagerly at this boundary. Individual operation
adapters remain lazy so importing :mod:`cudnn.jax` does not import every CuTe
kernel or any Torch API module.
"""

from importlib import import_module
from typing import Any

_INSTALL_HINT = "pip install 'nvidia-cudnn-frontend[jax]'"
_OPTIONAL_DEPENDENCY_PREFIXES = ("cuda", "cutlass", "jax")

try:
    import cutlass.jax as _cutlass_jax
    import jax as _jax
except ModuleNotFoundError as error:
    missing_name = error.name
    if missing_name is None or not any(missing_name == prefix or missing_name.startswith(f"{prefix}.") for prefix in _OPTIONAL_DEPENDENCY_PREFIXES):
        raise
    raise ImportError(f"cuDNN JAX APIs require the optional JAX dependencies. Install them with `{_INSTALL_HINT}`.") from error

if not _cutlass_jax.is_available():
    minimum_version = ".".join(str(part) for part in _cutlass_jax.CUTE_DSL_MIN_SUPPORTED_JAX_VERSION)
    installed_version = getattr(_jax, "__version__", "unknown")
    raise ImportError(
        f"CUTLASS JAX support is unavailable with JAX {installed_version}; "
        f"the minimum supported JAX version is {minimum_version}. "
        f"Install the optional JAX dependencies with `{_INSTALL_HINT}`."
    )

from .._jax import JaxApiBase, JaxTensorDesc, TupleDict, disable_device_compatibility_checks  # noqa: E402

_OPERATION_EXPORTS = {
    "RmsNormRhtAmaxSm100": ("..rmsnorm_rht_amax.jax", "RmsNormRhtAmaxSm100"),
    "rmsnorm_rht_amax_sm100": ("..rmsnorm_rht_amax.jax", "rmsnorm_rht_amax_sm100"),
    "SparseAttentionBackward": ("..deepseek_sparse_attention.sparse_attention_backward.jax", "SparseAttentionBackward"),
    "sparse_attention_backward_wrapper": (
        "..deepseek_sparse_attention.sparse_attention_backward.jax",
        "sparse_attention_backward_wrapper",
    ),
    "IndexerForward": ("..deepseek_sparse_attention.indexer_forward.jax", "IndexerForward"),
    "indexer_forward_wrapper": ("..deepseek_sparse_attention.indexer_forward.jax", "indexer_forward_wrapper"),
    "IndexerTopK": ("..deepseek_sparse_attention.indexer_top_k.jax", "IndexerTopK"),
    "indexer_top_k_wrapper": ("..deepseek_sparse_attention.indexer_top_k.jax", "indexer_top_k_wrapper"),
    "local_to_global_wrapper": ("..deepseek_sparse_attention.indexer_top_k.jax", "local_to_global_wrapper"),
    "compactify_wrapper": ("..deepseek_sparse_attention.indexer_top_k.jax", "compactify_wrapper"),
    "SparseIndexerScoreRecompute": ("..deepseek_sparse_attention.score_recompute.jax", "SparseIndexerScoreRecompute"),
    "sparse_indexer_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "sparse_indexer_score_recompute_wrapper",
    ),
    "SparseAttnScoreRecompute": ("..deepseek_sparse_attention.score_recompute.jax", "SparseAttnScoreRecompute"),
    "sparse_attn_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "sparse_attn_score_recompute_wrapper",
    ),
    "DenseIndexerScoreRecompute": ("..deepseek_sparse_attention.score_recompute.jax", "DenseIndexerScoreRecompute"),
    "dense_indexer_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "dense_indexer_score_recompute_wrapper",
    ),
    "DenseAttnScoreRecompute": ("..deepseek_sparse_attention.score_recompute.jax", "DenseAttnScoreRecompute"),
    "dense_attn_score_recompute_wrapper": (
        "..deepseek_sparse_attention.score_recompute.jax",
        "dense_attn_score_recompute_wrapper",
    ),
    "IndexerBackward": ("..deepseek_sparse_attention.indexer_backward.jax", "IndexerBackward"),
    "indexer_backward_wrapper": ("..deepseek_sparse_attention.indexer_backward.jax", "indexer_backward_wrapper"),
    "DenseIndexerBackward": ("..deepseek_sparse_attention.indexer_backward.jax", "DenseIndexerBackward"),
    "dense_indexer_backward_wrapper": ("..deepseek_sparse_attention.indexer_backward.jax", "dense_indexer_backward_wrapper"),
}

_DSA_EXPORTS = frozenset(name for name in _OPERATION_EXPORTS if name not in {"RmsNormRhtAmaxSm100", "rmsnorm_rht_amax_sm100"})


def _load_operation(name: str) -> Any:
    module_name, symbol_name = _OPERATION_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = value
    return value


class _DSANamespace:
    """Lazy JAX view of the DeepSeek sparse-attention operation family."""

    def __getattr__(self, name: str) -> Any:
        if name not in _DSA_EXPORTS:
            raise AttributeError(f"cudnn.jax.DSA has no attribute {name!r}")
        value = _load_operation(name)
        setattr(self, name, value)
        return value

    def __dir__(self) -> list[str]:
        return sorted((*vars(self), *_DSA_EXPORTS))


DSA = _DSANamespace()


def __getattr__(name: str) -> Any:
    if name in _OPERATION_EXPORTS:
        return _load_operation(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *_OPERATION_EXPORTS))


__all__ = [
    "DSA",
    "JaxApiBase",
    "JaxTensorDesc",
    "TupleDict",
    "disable_device_compatibility_checks",
    *_OPERATION_EXPORTS,
]
