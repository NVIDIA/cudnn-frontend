# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.gemm.frost.swiglu_decode: the gate / up product of an FP4 MLP for decode rows with SiLU(gate) x up and the FP4 re-quantisation
in one launch, ``SwigluFp4Decode`` (block-scaled FP4 weight x FP4 activation -> FP4 codes with E4M3 block scales, 1 .. 64 rows, the
weight stream issued ahead of the dependent-launch wait; SM100-class parts).  Lazy exports: nothing here imports torch or the CuTe DSL
at import time."""

from typing import Any

_LAZY_EXPORTS = {
    "Geometry": ("cudnn.gemm.frost.swiglu_decode.kernel.swiglu_fp4_decode_config", "Geometry"),
    "Cfg": ("cudnn.gemm.frost.swiglu_decode.kernel.swiglu_fp4_decode_config", "Cfg"),
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
