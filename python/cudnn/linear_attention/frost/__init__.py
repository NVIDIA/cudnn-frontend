# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention.frost: the FROST linear-attention engines —
Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet v2 on the SM100
chunked kernels built on Cutlass primitives. All three serve forward and
backward on SM100/SM103 and rank ahead of the cuTile fallbacks, except
GDN-2, which does not have a cuTile fallback."""

# Lazy: importing one family's engine must not drag its neighbours in.
import importlib
from typing import Any

_LAZY_EXPORTS = {
    "GdnFrostEngine": (".gdn_engine", "GdnFrostEngine"),
    "Gdn2FrostEngine": (".gdn2_engine", "Gdn2FrostEngine"),
    "KdaFrostEngine": (".kda_engine", "KdaFrostEngine"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    value = getattr(importlib.import_module(module, __name__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["GdnFrostEngine", "KdaFrostEngine", "Gdn2FrostEngine"]
