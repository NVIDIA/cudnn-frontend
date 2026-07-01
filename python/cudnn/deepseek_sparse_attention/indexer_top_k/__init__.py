"""Lazy framework exports for the DSA indexer top-K operation.

``api.py`` is always the Torch implementation and ``jax.py`` is always the
JAX implementation. The unqualified package API prefers Torch when available
and otherwise exposes the JAX API.
"""

from ..._framework_api import make_framework_api


_TORCH_EXPORTS = (
    "IndexerTopK",
    "indexer_top_k_wrapper",
    "local_to_global_wrapper",
    "compactify_wrapper",
)
_JAX_EXPORTS = ("IndexerTopKResult", "indexer_top_k_wrapper")

__all__, __getattr__ = make_framework_api(
    globals(),
    torch_exports=_TORCH_EXPORTS,
    jax_exports=_JAX_EXPORTS,
)
