"""Lazy API exports for the DeepSeek indexer-forward operation.

Unqualified symbols always come from the Torch ``api.py``. The JAX API is
available explicitly through the sibling ``jax`` namespace.
"""

from ..._operation_api import make_operation_api


_API_EXPORTS = ("IndexerForward", "indexer_forward_wrapper")

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
