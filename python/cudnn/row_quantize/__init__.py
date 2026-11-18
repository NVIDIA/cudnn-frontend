# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The row quantization operation family: decode steps that quantize the rows of a bf16 slab for the multiply that consumes them
(``cudnn.row_quantize.frost``).  Lazy exports: nothing here imports torch or the CuTe DSL at ``import cudnn`` time."""

from typing import Any

_LAZY_EXPORTS = {
    "RowQuantizeFp4Decode": ("cudnn.row_quantize.frost.row_quantize_fp4_decode", "RowQuantizeFp4Decode"),
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
