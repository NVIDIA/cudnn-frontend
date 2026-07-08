"""Lazy public namespace for Native Sparse Attention operations."""

from importlib import import_module

_SYMBOLS = {
    "SelectionAttention": (".selection", "SelectionAttention"),
    "selection_attention_wrapper": (".selection", "selection_attention_wrapper"),
    "CompressionAttention": (".compression", "CompressionAttention"),
    "compression_attention_wrapper": (
        ".compression",
        "compression_attention_wrapper",
    ),
    "SlidingWindowAttention": (
        ".sliding_window_attention",
        "SlidingWindowAttention",
    ),
    "sliding_window_attention_wrapper": (
        ".sliding_window_attention",
        "sliding_window_attention_wrapper",
    ),
    "TopKReduction": (".top_k", "TopKReduction"),
    "topk_reduction_wrapper": (".top_k", "topk_reduction_wrapper"),
}


def _load_symbol(name):
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    if name == "NSA":
        return NSA
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class NSANamespace:
    def __getattr__(self, name):
        if name in _SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"NSA has no attribute {name!r}")


NSA = NSANamespace()

__all__ = ["NSA"]
