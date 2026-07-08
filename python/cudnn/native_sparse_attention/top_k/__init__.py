"""Lazy Torch top-K API exports."""

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={"api": ("TopKReduction", "topk_reduction_wrapper")},
    submodules=("api", "jax"),
)
