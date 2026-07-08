"""Lazy Torch namespace for native sparse-attention operations."""

from typing import Any

from ..common.operation_api import make_operation_api

__all__, __getattr__, __dir__ = make_operation_api(
    globals(),
    exports={
        "selection": ("SelectionAttention", "selection_attention_wrapper"),
        "compression": ("CompressionAttention", "compression_attention_wrapper"),
        "sliding_window_attention": (
            "SlidingWindowAttention",
            "sliding_window_attention_wrapper",
        ),
        "top_k": ("TopKReduction", "topk_reduction_wrapper"),
    },
)
_OPERATION_EXPORTS = frozenset(__all__)


class NSANamespace:
    def __getattr__(self, name: str) -> Any:
        if name not in _OPERATION_EXPORTS:
            raise AttributeError(f"NSA has no attribute {name!r}")
        value = __getattr__(name)
        setattr(self, name, value)
        return value

    def __dir__(self) -> list[str]:
        return sorted((*vars(self), *_OPERATION_EXPORTS))


NSA = NSANamespace()

# Preserve the historical root contract: operations are accessed through NSA.
__all__ = ["NSA"]
