"""Lazy Torch API and framework-neutral indexer top-K exports.

JAX APIs are exported through :mod:`cudnn.jax`.
"""

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": ("IndexerTopKOp", "bucket_num_cols"),
        "api": (
            "IndexerTopK",
            "compactify_wrapper",
            "indexer_top_k_wrapper",
            "local_to_global_wrapper",
        ),
    },
    submodules=("api", "op"),
)
