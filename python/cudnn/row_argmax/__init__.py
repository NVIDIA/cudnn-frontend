# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The row argmax operation family: decode steps that pick, per row of a slab, the column of the row maximum
(``cudnn.row_argmax.frost``).  Lazy exports: nothing here imports torch or the CuTe DSL at ``import cudnn`` time."""

from typing import Any

_LAZY_EXPORTS = {
    "RowArgmaxDecode": ("cudnn.row_argmax.frost.row_argmax_decode", "RowArgmaxDecode"),
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
