"""Lazy PyTorch exports for the DeepSeek indexer-forward operation.

Keeping these exports lazy lets the optional JAX adapter import the shared
CuTe DSL kernel without importing PyTorch.
"""

from importlib import import_module
from typing import Any

__all__ = ["IndexerForward", "indexer_forward_wrapper"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(".api", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
