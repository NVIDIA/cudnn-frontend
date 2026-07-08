"""Lazy Torch API and framework-neutral indexer-forward exports.

JAX APIs are exported through :mod:`cudnn.jax`.
"""

from ..._operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "op": (
            "IndexerForwardOp",
            "SUPPORTED_COMPUTE_CAPABILITIES",
            "TMA_ALIGN_ELEMENTS",
        ),
        "api": ("IndexerForward", "indexer_forward_wrapper"),
    },
    submodules=("api", "op"),
)
