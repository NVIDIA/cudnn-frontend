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


def _leads(spec: EngineSpec, facts) -> bool:
    """Whether a default row's plans go ahead of the backend's block.

    ``sdpa_bwd_sm100_d128`` (measured 2026-10-26 against cuDNN 9.28.0.38, 216 cases at 32k tokens: d64/d128, GQA
    32/8 16/2 and MHA, S 1k-16k, causal / mask-free / SWA1024, BSHD and THD at equal and half-to-full lengths): THD at
    d128 wins 0.79-0.95 (one 1.07) -> ahead; dense BSHD d128 is parity (0.93-1.27), a window loses ~2x and dense D64
    loses 1.7-2.3x -> behind the backend. Every other row keeps ours-first."""
    if spec.name == "sdpa_bwd_sm100_d128":
        return facts.thd and facts.window_left is None and (facts.d_qk, facts.d_v) == (128, 128)
    return True


def propose(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """The manifest hook: :func:`recommend`'s plans, the backend's block placed after the rows that lead
    (:func:`_leads`) and before the rest. ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` set keeps ours first everywhere."""
    ours = recommend(kind, facts, offered)
    from cudnn.engines.manifest import opt_in_engines_enabled

    if not ours or opt_in_engines_enabled():
        return ours
    from cudnn.engines.heuristics import BACKEND

    spec_of = {offered[s.name]: s for s in ENGINE_SPECS if s.name in offered}
    lead = [p for p in ours if _leads(spec_of[p.engine_id], facts)]
    trail = [p for p in ours if p not in lead]
    return lead + [BACKEND] + trail if trail else lead


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """One complete entry per eligible row, in ``ENGINE_SPECS`` order; ``kind``
    does not change the answer (no runner-up candidates to thin out)."""
    out: List[PlanConfig] = []
    for engine_id, spec in _eligible(facts, offered):
        knobs = default_knobs(spec, facts)
        if knobs is None or mismatch(spec.capabilities, facts, knobs) is None:
            out.append(PlanConfig(engine_id, knobs))
    return out
