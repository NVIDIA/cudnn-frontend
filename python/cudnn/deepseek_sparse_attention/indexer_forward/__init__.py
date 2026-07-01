"""Lazy framework exports for the DeepSeek indexer-forward operation.

``api.py`` is always the Torch implementation and ``jax.py`` is always the
JAX implementation. The unqualified package API prefers Torch when available
and otherwise exposes the JAX API.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any


_TORCH_EXPORTS = ("IndexerForward", "indexer_forward_wrapper")
_JAX_EXPORTS = ("IndexerForwardResult", "indexer_forward_wrapper")

if find_spec("torch") is not None:
    _DEFAULT_API = ".api"
    __all__ = list(_TORCH_EXPORTS)
elif find_spec("jax") is not None:
    _DEFAULT_API = ".jax"
    __all__ = list(_JAX_EXPORTS)
else:
    _DEFAULT_API = None
    __all__ = []


def __getattr__(name: str) -> Any:
    if name in {"api", "jax"}:
        value = import_module(f".{name}", __name__)
    elif _DEFAULT_API is not None and name in __all__:
        value = getattr(import_module(_DEFAULT_API, __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
