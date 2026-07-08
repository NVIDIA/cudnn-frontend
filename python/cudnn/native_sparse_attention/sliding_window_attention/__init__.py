"""Lazy API exports for NSA sliding-window attention."""

from ..._operation_api import make_operation_api

_API_EXPORTS = ("SlidingWindowAttention", "sliding_window_attention_wrapper")

__all__, __getattr__ = make_operation_api(
    globals(),
    exports=_API_EXPORTS,
)
