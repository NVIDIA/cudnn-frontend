"""Lazy Torch sliding-window-attention API exports."""

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={"api": ("SlidingWindowAttention", "sliding_window_attention_wrapper")},
    submodules=("api", "jax"),
)
