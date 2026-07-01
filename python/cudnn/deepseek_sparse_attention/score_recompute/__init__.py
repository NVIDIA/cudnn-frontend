"""Lazy API exports for the DSA score-recompute operations.

Unqualified symbols always come from the Torch ``api.py``. The JAX API is
available explicitly through the sibling ``jax`` namespace.
"""

from ..._operation_api import make_operation_api

_API_EXPORTS = (
    "SparseIndexerScoreRecompute",
    "sparse_indexer_score_recompute_wrapper",
    "SparseAttnScoreRecompute",
    "sparse_attn_score_recompute_wrapper",
    "DenseIndexerScoreRecompute",
    "dense_indexer_score_recompute_wrapper",
    "DenseAttnScoreRecompute",
    "dense_attn_score_recompute_wrapper",
)

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
