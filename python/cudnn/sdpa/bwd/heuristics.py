# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDPA-backward family heuristics: list each eligible engine's plan WITH the
tiles it will build, so a recorded ``(engine_id, knobs)`` pins the kernel.

``(kind, facts, offered) -> [PlanConfig]``, backend-blind and graph-blind like
``sdpa.fwd.heuristics.recommend``. The backward engines have no runner-up
candidates: every eligible row contributes exactly one entry, for ``"A"`` and
``"FALLBACK"`` alike. What the entry spells is what the lowering would pick
when handed no knobs:

- a row whose kernel takes a tile choice (``Capabilities.tile_ms`` /
  ``tile_ns`` non-empty) gets its default pair as ``TILE_M`` / ``TILE_N`` --
  the SM120 kernel's per-head-dim table, or the sole point of a fixed domain;
- a row with no tile axis lists no knobs: its geometry is not a tuning
  decision, so ``{}`` is the complete record.

Replaying the listed knobs through ``create_execution_plan`` builds the same
kernel the automatic pick builds today, and keeps building it if the default
table moves in a later release.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Tuple

from cudnn.engines.base import PlanConfig

from .engines import ENGINE_SPECS, EngineSpec, SdpaBwdKnobs, mismatch


def _sm120_default_tiles(facts) -> Optional[Tuple[int, int]]:
    """The SM120 f16 kernel's default (q_tile, kv_tile) for these head dims
    (``config_sm120.DEFAULT_TILES``, keyed by the padded d_qk the kernel runs at)."""
    from .config_sm120 import DEFAULT_TILES, padded_head_dims

    pads = padded_head_dims(int(facts.d_qk), int(facts.d_v))
    return None if pads is None else DEFAULT_TILES[pads[0]]


# Rows whose kernel resolves an unset tile from a table of its own. A row with a
# single-point tile domain needs no entry (the sole point is the default); a row
# without a tile axis needs none either. test_sdpa_bwd_heuristics guards that
# every row with a wider domain is listed here.
_DEFAULT_TILE_RESOLVERS: Dict[str, Callable[[object], Optional[Tuple[int, int]]]] = {
    "sdpa_bwd_sm120": _sm120_default_tiles,
}


def default_knobs(spec: EngineSpec, facts) -> Optional[SdpaBwdKnobs]:
    """The knobs the lowering would resolve for ``facts`` when handed none:
    ``None`` for a row with no tile axis, else a complete ``SdpaBwdKnobs``."""
    caps = spec.capabilities
    if not caps.tile_ms and not caps.tile_ns:
        return None
    resolver = _DEFAULT_TILE_RESOLVERS.get(spec.name)
    if resolver is not None:
        tiles = resolver(facts)
        if tiles is None:
            return None
        return SdpaBwdKnobs(tile_m=tiles[0], tile_n=tiles[1])
    if len(caps.tile_ms) == 1 and len(caps.tile_ns) == 1:
        return SdpaBwdKnobs(tile_m=next(iter(caps.tile_ms)), tile_n=next(iter(caps.tile_ns)))
    raise NotImplementedError(f"{spec.name} advertises a tile domain but no default-tile resolver is registered for it")


def _eligible(facts, offered: Dict[str, int]) -> Iterator[Tuple[int, EngineSpec]]:
    """(engine_id, spec) for each offered row whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """One complete entry per eligible row, in ``ENGINE_SPECS`` order; ``kind``
    does not change the answer (no runner-up candidates to thin out)."""
    out: List[PlanConfig] = []
    for engine_id, spec in _eligible(facts, offered):
        knobs = default_knobs(spec, facts)
        if knobs is None or mismatch(spec.capabilities, facts, knobs) is None:
            out.append(PlanConfig(engine_id, knobs))
    return out
