# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-dependent cuDNN operation helpers.

Public symbols resolve on first attribute access so importing ``cudnn.ops``
does not require PyTorch.
"""

import importlib
import sys
from types import ModuleType
from typing import Any

_OPTIONAL_DEPENDENCY_INSTALL_HINT = "Install with 'pip install nvidia-cudnn-frontend[cutedsl]'"

_LAZY_EXPORTS = {
    "causal_conv1d": (".causal_conv1d", "causal_conv1d"),
    "causal_conv1d_nwh": (".causal_conv1d", "causal_conv1d_nwh"),
    "b2b_causal_conv1d": (".causal_conv1d", "b2b_causal_conv1d"),
    "causal_conv1d_update": ("._causal_conv1d_update", "causal_conv1d_update"),
    "fft_causal_conv1d": (".fft_causal_conv1d", "fft_causal_conv1d"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        value = getattr(importlib.import_module(module_name, __name__), attr_name)
    except ImportError as error:
        from cudnn import _optional_dependency_message

        raise ImportError(_optional_dependency_message(name, error)) from error

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _OpsModule(ModuleType):
    """Keep callable exports stable after importing a same-named implementation."""

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        target = _LAZY_EXPORTS.get(name)
        if target is not None and isinstance(value, ModuleType):
            module_name, attr_name = target
            if value.__name__ == __name__ + module_name:
                value = getattr(value, attr_name)
                super().__setattr__(name, value)
        return value


# Python binds submodules on their parent package, bypassing __getattr__. Resolve
# those bindings on attribute access without importing any optional dependency.
sys.modules[__name__].__class__ = _OpsModule
