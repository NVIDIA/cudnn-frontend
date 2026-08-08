# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""How the SDPA-forward family ranks plans for a graph.

``engines/heuristics.rank`` hands over the parsed facts, this family's engine
ids, and the backend's entries tagged by mode; what :func:`recommend` returns is
``graph.plans``, position for position. The whole comparison is here because
that is the only place it can be made — a cell cannot see its siblings, and
neither side of the FROST/backend split can place the other.

Per mode:

- **A** — candidates worth running, best guess first. A cell whose capability
  row declares one point per axis contributes one entry; a cell with a real
  knob domain names its choice first and the runners-up behind it, for a caller
  that autotunes. :func:`_sm120_tiles` is the worked example of such a rule.
- **FALLBACK** — the config expected to build where mode A's choice may not.
  Nothing here is chosen for speed.
- **OPENSOURCE** — mode A without the backend's recommendation: these cells ARE
  the open-source implementation and the backend's engines are not. Combine it
  to measure coverage — ``[OPENSOURCE, A, FALLBACK]`` tries every FROST config
  first and still has the backend behind it, so a graph that runs on a backend
  plan is one FROST does not cover.
- **B** — answered as A: it asks for a wider search than A, and this family has
  none to give.

To add a rule for a cell: write the function, list the cell in
``_TILE_RULE_CELLS``, and put the measurement behind it in the commit. A cell
absent from that set falls back to its capability row's sole point per axis,
which is the honest answer when nobody has timed it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cudnn

from cudnn.engines.base import PlanConfig
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, Capabilities, SdpaFwdKnobs, mismatch

# Cells timed against the backend's own kernel and found SLOWER: the backend's
# mode-A entries lead and the cell is the second choice.
#
# A cell absent from this set has either measured faster or not been timed at
# all, and both lead. That is deliberate but it is NOT a measurement -- it is
# the order this dispatch has always had. Moving a cell in here is an
# experiment, not an edit. Coverage does not depend on it: a caller who wants
# FROST tried first asks for heur_mode.OPENSOURCE.
_MEASURED_BEHIND: frozenset = frozenset()

# Cells whose (tile_m, tile_n) choice _sm120_tiles makes.
_TILE_RULE_CELLS = frozenset({"sdpa_fwd_prefill_sm120"})


def _sm120_tiles(facts) -> Tuple[int, int]:
    """(tile_m, tile_n) for the SM120 SDPA-forward prefill cell.

    ``tile_n=128`` unconditionally -- fastest across the f16 sweep, by 2-4%
    rather than a uniform margin.

    ``tile_m=64`` when the grid cannot fill the machine AND each CTA has enough
    KV tiles for the extra Q-tile loop to amortize: halving the Q tile doubles
    the CTA count, which only pays while SMs sit idle, and adds loop overhead
    that only long sequences absorb. Worth 1.5x at 64 CTAs. The 1.5x SM-count
    bound is where the two stop trading evenly: 240 CTAs still want the finer
    tile on this part, 320 want the coarser one by 1.19x.

    A causal mask halves the work per CTA, so the machine empties sooner and
    the finer Q tile keeps paying further out -- folded in as a halved
    effective grid.

    Regret against the best of the enumerated domain: 1.009 geomean, 1.054
    worst case. A tile rule is a property of the KERNEL it was measured on, not
    of the hardware: an earlier revision of this template staged P through SMEM
    and wanted ``tile_n=64``; the shfl path removed that traffic and the
    optimum moved.
    """
    sm_count = facts.device_sm_count or 0
    if sm_count <= 0:
        return 128, 128
    grid = -(-facts.s_q // 128) * facts.h_q * facts.b
    if facts.causal:
        grid //= 2
    kv_tiles = -(-facts.s_kv // 128)
    fine = grid * 2 <= sm_count or (grid * 2 <= 3 * sm_count and kv_tiles >= 12)
    return (64 if fine else 128), 128


def _sole(values):
    """The only value on an axis, or None where the row declares no domain."""
    return next(iter(values)) if len(values) == 1 else None


def _knobs(caps: Capabilities, tile_m, tile_n) -> SdpaFwdKnobs:
    """A knob request for one point.

    A field stays None only on an axis whose capability row declares no domain
    — the engine then has no say to honour. It never means "engine, pick for
    me": that reading is what let the same choice be made twice, once here and
    once inside the adapter.
    """
    return SdpaFwdKnobs(sched_policy=_sole(caps.sched_policies), tile_m=tile_m, tile_n=tile_n, cga=_sole(caps.cgas))


def _eligible(facts, offered: Dict[str, int]):
    """(engine_id, spec) for each offered cell whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


def _admissible(caps: Capabilities, facts, knobs: SdpaFwdKnobs) -> bool:
    return mismatch(caps, facts, knobs) is None


def _mode_a(facts, offered: Dict[str, int], mode) -> List[PlanConfig]:
    """Candidates worth timing, best guess first."""
    out = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        if spec.name not in _TILE_RULE_CELLS:
            # No rule measured for this cell: its capability row has one point
            # per axis, so there is nothing to choose between anyway.
            knobs = _knobs(caps, _sole(caps.tile_ms), _sole(caps.tile_ns))
            if _admissible(caps, facts, knobs):
                out.append(PlanConfig(engine_id, knobs, mode=mode))
            continue
        best = _sm120_tiles(facts)
        # The guess first, then the rest of the domain as autotune candidates:
        # the rule's regret is small but not zero, so the runners-up are worth
        # offering to a caller who measures.
        ordered = sorted(
            ((m, n) for m in caps.tile_ms for n in caps.tile_ns),
            key=lambda mn: (mn != best, mn[1] != best[1], -mn[0]),
        )
        for tile_m, tile_n in ordered:
            knobs = _knobs(caps, tile_m, tile_n)
            if _admissible(caps, facts, knobs):
                out.append(PlanConfig(engine_id, knobs, mode=mode))
    return out


def _mode_fallback(facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Configs expected to build where mode A's choice may not.

    TODO: today this is the smallest tile the row admits — the config that asks
    least of the device, which is the one thing a fallback must be. Once a cell
    has features its largest tiles cannot serve, this becomes the handful of
    configs that between them cover the whole plane, chosen from measurements.
    """
    out = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        knobs = _knobs(caps, min(caps.tile_ms, default=None), min(caps.tile_ns, default=None))
        if _admissible(caps, facts, knobs):
            out.append(PlanConfig(engine_id, knobs, mode=cudnn.heur_mode.FALLBACK))
    return out


def _leads(offered: Dict[str, int], plans: List[PlanConfig]) -> bool:
    """Whether this family's mode-A plans outrank the backend's. See _MEASURED_BEHIND."""
    behind = {offered[name] for name in _MEASURED_BEHIND if name in offered}
    return bool(plans) and not all(cfg.engine_id in behind for cfg in plans)


def recommend(modes: List[Any], facts, offered: Dict[str, int], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The ranked plan list for this graph, mode by mode in the caller's order.

    Each mode contributes a block and the blocks concatenate, so asking for
    ``[A, FALLBACK]`` puts every tuned candidate — both sides' — ahead of every
    fallback. A plan repeated across modes keeps its first position: building
    the same config twice only costs the caller a JIT compile.
    """
    # An untagged backend entry is the delegating one: candidates C++ holds but
    # never exposes as plans, which Graph::build_plans tries BEFORE its own
    # engine_configs. It belongs to no mode and must keep the lead, or an
    # OPENSOURCE caller gets a native kernel instead of the OSS one.
    out: List[PlanConfig] = [c for c in backend_plans if c.mode is None]
    for mode in modes:
        if mode == cudnn.heur_mode.OPENSOURCE:
            out += _mode_a(facts, offered, cudnn.heur_mode.A)
        elif mode in (cudnn.heur_mode.A, cudnn.heur_mode.B):
            ours = _mode_a(facts, offered, mode)
            theirs = [c for c in backend_plans if c.mode == mode]
            out += (ours + theirs) if _leads(offered, ours) else (theirs + ours)
        elif mode == cudnn.heur_mode.FALLBACK:
            out += _mode_fallback(facts, offered) + [c for c in backend_plans if c.mode == mode]

    seen, ranked = set(), []
    for cfg in out:
        key = (cfg.engine_id, repr(cfg.knobs), cfg.cpp_index)
        if key not in seen:
            seen.add(key)
            ranked.append(cfg)
    return ranked
