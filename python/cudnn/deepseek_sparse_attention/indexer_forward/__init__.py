"""Lazy framework exports for the DeepSeek indexer-forward operation.

``api.py`` is always the Torch implementation and ``jax.py`` is always the
JAX implementation. The unqualified package API prefers Torch when available
and otherwise exposes the JAX API.
"""

from ..._framework_api import make_framework_api


_TORCH_EXPORTS = ("IndexerForward", "indexer_forward_wrapper")
_JAX_EXPORTS = ("IndexerForwardResult", "indexer_forward_wrapper")

__all__, __getattr__ = make_framework_api(
    globals(),
    torch_exports=_TORCH_EXPORTS,
    jax_exports=_JAX_EXPORTS,
)
