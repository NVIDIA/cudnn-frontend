# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.gemm.frost.projection_decode: the FP8 split-K projection for decode rows ``Fp8ProjectionDecode`` (E4M3 weight x E4M3 activation
-> bf16, 1 .. 32 rows, the weight stream issued ahead of the dependent-launch wait; SM100-class parts).  Lazy exports: nothing here
imports torch or the CuTe DSL at import time."""

from typing import Any

_LAZY_EXPORTS = {
    "Fp8ProjectionDecode": ("cudnn.gemm.frost.projection_decode.fp8_projection_decode", "Fp8ProjectionDecode"),
    "Form": ("cudnn.gemm.frost.projection_decode.kernel.fp8_projection_decode_config", "Form"),
    "Geometry": ("cudnn.gemm.frost.projection_decode.kernel.fp8_projection_decode_config", "Geometry"),
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
