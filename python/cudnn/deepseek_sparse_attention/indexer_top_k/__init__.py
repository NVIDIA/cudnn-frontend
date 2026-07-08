"""Lazy API exports for the DSA indexer top-K operation.

Unqualified symbols always come from the Torch ``api.py``. The JAX API is
available explicitly through the sibling ``jax`` namespace.
"""

from ..._operation_api import make_operation_api


_API_EXPORTS = (
    "IndexerTopK",
    "indexer_top_k_wrapper",
    "local_to_global_wrapper",
    "compactify_wrapper",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
