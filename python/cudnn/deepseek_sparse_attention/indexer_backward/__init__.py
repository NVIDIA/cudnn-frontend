"""Lazy API exports for DSA indexer backward."""

from ..._operation_api import make_operation_api

_API_EXPORTS = (
    "DenseIndexerBackward",
    "IndexerBackward",
    "dense_indexer_backward_wrapper",
    "indexer_backward_wrapper",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
