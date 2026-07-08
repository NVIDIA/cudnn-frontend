"""Lazy Torch API and framework-neutral score-recompute exports.

JAX APIs are exported through :mod:`cudnn.jax`.
"""

from ..._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "config": (
            "DenseScoreKernelConfig",
            "SparseScoreKernelConfig",
            "dispatch_sparse_attn_tile_params",
            "resolve_dense_score_kernel_config",
            "resolve_dense_score_smem_config",
            "resolve_sparse_score_kernel_config",
            "resolve_sparse_score_smem_config",
        ),
        "op": (
            "DenseScoreRecomputeOp",
            "DenseScoreSm90Config",
            "SparseScoreRecomputeOp",
            "SparseScoreSm90Config",
        ),
        "api": (
            "SparseIndexerScoreRecompute",
            "sparse_indexer_score_recompute_wrapper",
            "SparseAttnScoreRecompute",
            "sparse_attn_score_recompute_wrapper",
            "DenseIndexerScoreRecompute",
            "dense_indexer_score_recompute_wrapper",
            "DenseAttnScoreRecompute",
            "dense_attn_score_recompute_wrapper",
        ),
    },
    submodules=("api", "config", "op"),
)
