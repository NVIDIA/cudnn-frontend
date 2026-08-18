# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention.cutile: cuTile GDN / KDA implementations.

``GdnCuTileEngine`` (a router ``BaseEngine``) executes single-node GDN /
GDN_BWD graphs on the chunked cuTile kernels in ``kernels/gdn``.
``KdaCuTileEngine`` does the same for KDA / KDA_BWD graphs on
``kernels/kda``.
"""

from typing import Any

_LAZY_EXPORTS = {
    "GdnCuTileEngine": ("cudnn.linear_attention.cutile.gdn_engine", "GdnCuTileEngine"),
    "KdaCuTileEngine": ("cudnn.linear_attention.cutile.kda_engine", "KdaCuTileEngine"),
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
