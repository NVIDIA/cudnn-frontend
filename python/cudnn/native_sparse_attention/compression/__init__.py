"""Lazy Torch compression-attention API exports."""

from ...common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={"api": ("CompressionAttention", "compression_attention_wrapper")},
    submodules=("api", "jax"),
)
