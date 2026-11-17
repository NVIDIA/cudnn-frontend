# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.norm.frost: the FROST (CuTe DSL) normalization decode steps -- ``AddNormQuantDecode`` (the residual add, the RMSNorm and the
quantization of the following projection's input in one launch: bf16 in; bf16, FP8 E4M3 or NVFP4 out; decode-time row counts; SM100-class
parts).  Lazy exports with no DSL gate at attribute access: the entry is constructible, and declines by type, on a host without the DSL."""

from typing import Any

_LAZY_EXPORTS = {
    "AddNormQuantDecode": ("cudnn.norm.frost.add_norm_quant_decode", "AddNormQuantDecode"),
    "Cfg": ("cudnn.norm.frost.kernel.add_norm_quant_decode_config", "Cfg"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
