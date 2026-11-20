# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.attention_prepare.frost: the FROST (CuTe DSL) attention-preparation decode step -- ``AttentionPrepareDecode`` (per token the
RMSNorm of every query and key head row, the rotation of its rotary part by the token's position, the gate rows copied beside the
queries and the keys and values written into paged pools, in one launch: bf16 in and out; decode-time token counts; SM100-class
parts).  Lazy exports with no DSL gate at attribute access: the entry is constructible, and declines by type, on a host without the DSL."""

from typing import Any

_LAZY_EXPORTS = {
    "AttentionPrepareDecode": ("cudnn.attention_prepare.frost.prepare_decode", "AttentionPrepareDecode"),
    "AttentionPrepareGeometry": ("cudnn.attention_prepare.frost.kernel.prepare_decode_config", "AttentionPrepareGeometry"),
    "Cfg": ("cudnn.attention_prepare.frost.kernel.prepare_decode_config", "Cfg"),
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
