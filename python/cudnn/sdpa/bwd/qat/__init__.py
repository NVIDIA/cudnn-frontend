# SPDX-License-Identifier: Apache-2.0

"""NVFP4 quantization-aware SDPA backward."""

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "Nvfp4AttentionQatBackward": (".api", "Nvfp4AttentionQatBackward"),
    "nvfp4_attention_qat_backward": (".api", "nvfp4_attention_qat_backward"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public QAT symbols on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    value = getattr(importlib.import_module(module, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    """List both materialized globals and lazy public exports."""
    return sorted(set(globals()) | set(__all__))
