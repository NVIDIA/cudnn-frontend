"""Torch API exports for the DSA indexer top-K kernels.

The exports are lazy so framework-neutral kernel modules can be imported by
the optional JAX frontend without importing Torch.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "IndexerTopK",
    "indexer_top_k_wrapper",
    "local_to_global_wrapper",
    "compactify_wrapper",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".api", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
