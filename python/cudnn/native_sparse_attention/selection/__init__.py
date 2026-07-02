"""Lazy API exports for NSA selection attention."""

from ..._operation_api import make_operation_api

_API_EXPORTS = ("SelectionAttention", "selection_attention_wrapper")

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
