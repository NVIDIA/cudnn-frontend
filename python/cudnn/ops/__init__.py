# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-dependent cuDNN operation helpers.

Public symbols resolve on first attribute access so importing ``cudnn.ops``
does not require PyTorch.
"""

import importlib
from typing import Any

_OPTIONAL_DEPENDENCY_INSTALL_HINT = "Install with 'pip install nvidia-cudnn-frontend[cutedsl]'"

_LAZY_EXPORTS = {
    "causal_conv1d": (".causal_conv1d", "causal_conv1d"),
    "causal_conv1d_nwh": (".causal_conv1d", "causal_conv1d_nwh"),
    "b2b_causal_conv1d": (".causal_conv1d", "b2b_causal_conv1d"),
    "fft_causal_conv1d": (".fft_causal_conv1d", "fft_causal_conv1d"),
    "Nvfp4BlockScaleQuantizer": (".nvfp4", "Nvfp4BlockScaleQuantizer"),
    "Nvfp4BlockScaleDequantizer": (".nvfp4", "Nvfp4BlockScaleDequantizer"),
    "nvfp4_block_scale_quantize": (".nvfp4", "nvfp4_block_scale_quantize"),
    "nvfp4_block_scale_dequantize": (".nvfp4", "nvfp4_block_scale_dequantize"),
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
        raise ImportError(f"{name} requires optional dependencies. {_OPTIONAL_DEPENDENCY_INSTALL_HINT}: {error}") from error

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
