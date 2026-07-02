"""Lazy API exports for NSA top-K reduction."""

from ..._operation_api import make_operation_api

_API_EXPORTS = ("TopKReduction", "topk_reduction_wrapper")

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
