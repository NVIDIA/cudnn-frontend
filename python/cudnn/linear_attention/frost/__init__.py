# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention.frost: the FROST linear-attention engines —
Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet v2 on the SM100
chunked kernels built on Cutlass primitives. ``GdnFrostEngine`` is the
default GDN engine on SM100/SM103; ``KdaFrostEngine`` and ``Gdn2FrostEngine``
are forward-only (their backward kernels are stubs — KDA gradients run on
``KdaCuTileEngine``)."""

# Lazy: importing one family's engine must not drag its neighbours in. The
# manifest's factories tolerate a missing optional dependency PER ENGINE, and
# eager imports here would have made one bad import cost all three.
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
