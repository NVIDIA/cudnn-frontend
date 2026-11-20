# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attention preparation operation family: the decode step that normalizes and rotates a token's query and key head rows, copies
its gate rows and writes its keys and values into paged pools (``cudnn.attention_prepare.frost``).  Lazy exports: nothing here
imports torch or the CuTe DSL at ``import cudnn`` time."""

from typing import Any

_LAZY_EXPORTS = {
    "AttentionPrepareDecode": ("cudnn.attention_prepare.frost.prepare_decode", "AttentionPrepareDecode"),
    "AttentionPrepareGeometry": ("cudnn.attention_prepare.frost.kernel.prepare_decode_config", "AttentionPrepareGeometry"),
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
