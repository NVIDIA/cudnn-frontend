# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cudnn.linear_attention: the linear-attention operation family (GDN, KDA, GDN-2)."""

from typing import Any

_LAZY_EXPORTS = {
    "gated_delta_net": ("cudnn.linear_attention.ops", "gated_delta_net"),
    "kimi_delta_attention": ("cudnn.linear_attention.ops", "kimi_delta_attention"),
    "gated_delta_net_v2": ("cudnn.linear_attention.ops", "gated_delta_net_v2"),
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


# --- family factories -------------------------------------------------------
# One factory per FAMILY (a kind of graph), returning every engine that serves
# it. A missing optional dependency must cost only the engine that needs it,
# never the whole family — the manifest's own ImportError guard is family-wide,
# so per-engine tolerance has to live here.
def _collect(ids, *specs):
    """Build the engines named in ``ids``, tolerating a missing dependency.

    ``ids`` is ``{name: engine_id}`` from engines/manifest.py, the single source
    of engine ids. A missing optional dependency must cost only the engine that
    needs it, never the whole family -- the manifest's own ImportError guard is
    family-wide, so per-engine tolerance has to live here.
    """
    import importlib
    import logging

    out = []
    for name, module, attr in specs:
        if name not in ids:
            continue
        try:
            engine = getattr(importlib.import_module(module), attr)()
        except ImportError as exc:
            logging.getLogger(__name__).info("engine %s is unavailable in this environment: %s", name, exc)
            continue
        engine.engine_id = ids[name]
        out.append(engine)
    return out


def GdnEngines(ids):
    """The GDN family: frost first, cuTile second (candidate order, not rank)."""
    return _collect(
        ids,
        ("gdn_frost", "cudnn.linear_attention.frost.gdn_engine", "GdnFrostEngine"),
        ("gdn_cutile", "cudnn.linear_attention.cutile.gdn_engine", "GdnCuTileEngine"),
    )


def KdaEngines(ids):
    """The KDA family."""
    return _collect(
        ids,
        ("kda_frost", "cudnn.linear_attention.frost.kda_engine", "KdaFrostEngine"),
        ("kda_cutile", "cudnn.linear_attention.cutile.kda_engine", "KdaCuTileEngine"),
    )


def Gdn2Engines(ids):
    """The GDN-2 family."""
    return _collect(ids, ("gdn2_frost", "cudnn.linear_attention.frost.gdn2_engine", "Gdn2FrostEngine"))
