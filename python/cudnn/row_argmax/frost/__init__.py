# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.row_argmax.frost: the FROST (CuTe DSL) row argmax decode steps -- ``RowArgmaxDecode`` (per row of a bf16 slab the lowest
column of the row maximum, or the entry of an id table at that column, in one launch; decode-time row counts; SM100-class parts).
Lazy exports with no DSL gate at attribute access: the entry is constructible, and declines by type, on a host without the DSL."""

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
