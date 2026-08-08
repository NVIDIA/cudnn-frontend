# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""How the SDPA-forward family ranks plans for a graph.

``engines/heuristics.rank`` hands over the parsed facts, this family's engine
ids, and the backend's entries tagged by mode; what :func:`recommend` returns is
``graph.plans``, position for position. The whole comparison is here because
that is the only place it can be made — a cell cannot see its siblings, and
neither side of the FROST/backend split can place the other.

Per mode:

- **A** — candidates that are all worth running, best guess first. A caller who
  does not autotune takes entry 0; one who does builds the first few and times
  them. The tile rule (``config_sm120.tile_choice``) picks the guess.
- **FALLBACK** — configs that are expected to build wherever mode A might not,
  ordered cheapest-resource first. Nothing here is chosen for speed.

``heur_mode.OPENSOURCE`` is mode A without the backend's recommendation: these
cells ARE the open-source implementation, and the backend's engines are not.
Combine it to measure coverage -- ``[OPENSOURCE, A, FALLBACK]`` tries every
FROST config first and still has the backend behind it, so a graph that runs on
a backend plan is one FROST does not cover.

``heur_mode.B`` is answered as A: it asks for a wider search than A, and this
family has none to give.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cudnn

from cudnn.engines.base import PlanConfig
from cudnn.sdpa.fwd.config_sm120 import tile_choice
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, Capabilities, SdpaFwdKnobs, mismatch

# Cells whose tile choice has measurements behind it (see
# docs/fe-oss-apis/attention/sdpa-fp8-sm120.md). Named rather than derived from
# the capability row: a rule belongs to the kernel it was measured on, and a new
# cell inheriting one by accident of its knob domain is how a tile rule
# outlives its evidence.
_TILE_RULE_CELLS = frozenset({"sdpa_fwd_prefill_sm120", "sdpa_fwd_prefill_sm120_fp8"})

# Cells timed against the backend's own kernel and found SLOWER: the backend's
# mode-A entries lead and the cell is the second choice. sm120 fp8 is not here
# because it measures 1.20-1.83x the backend's native fp8 fprop across 28
# shapes (docs/fe-oss-apis/attention/sdpa-fp8-sm120.md).
#
# A cell absent from this set has either measured faster or not been timed at
# all; both lead, which is deliberate. Coverage does not depend on it -- a
# caller who wants FROST tried first asks for heur_mode.OPENSOURCE.
_MEASURED_BEHIND: frozenset = frozenset()


def _sole(values):
    """The only value on an axis, or None where the row offers a choice."""
    return next(iter(values)) if len(values) == 1 else None


def _knobs(caps: Capabilities, tile_m, tile_n) -> SdpaFwdKnobs:
    """A knob request for one point. A field stays None only where the row
    declares no domain for that axis — the engine then has no say to honour."""
    return SdpaFwdKnobs(sched_policy=_sole(caps.sched_policies), tile_m=tile_m, tile_n=tile_n, cga=_sole(caps.cgas))


def _admissible(caps: Capabilities, facts, knobs: SdpaFwdKnobs) -> bool:
    return mismatch(caps, facts, knobs) is None


def _eligible(facts, offered: Dict[str, int]):
    """(engine_id, spec) for each offered cell whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


def _mode_a(facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Candidates worth timing, best guess first."""
    out = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        if spec.name not in _TILE_RULE_CELLS:
            # No rule measured for this cell: its capability row has one point
            # per axis, so there is nothing to choose between anyway.
            knobs = _knobs(caps, _sole(caps.tile_ms), _sole(caps.tile_ns))
            if _admissible(caps, facts, knobs):
                out.append(PlanConfig(engine_id, knobs, mode=cudnn.heur_mode.A))
            continue
        best = tile_choice(facts.s_q, facts.s_kv, facts.h_q, facts.b, facts.device_sm_count or 0, facts.causal)
        # The guess first, then the rest of the domain as autotune candidates:
        # the rule's regret is small but not zero, so the runner-up is worth
        # offering to a caller who measures.
        ordered = sorted(
            ((m, n) for m in caps.tile_ms for n in caps.tile_ns),
            key=lambda mn: (mn != best, mn[1] != best[1], -mn[0]),
        )
        for tile_m, tile_n in ordered:
            knobs = _knobs(caps, tile_m, tile_n)
            if _admissible(caps, facts, knobs):
                out.append(PlanConfig(engine_id, knobs, mode=cudnn.heur_mode.A))
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
    [A, FALLBACK] puts every tuned candidate — both sides' — ahead of every
    fallback. A plan repeated across modes keeps its first position: building
    the same config twice only costs the caller a JIT compile.
    """
    a_modes = (cudnn.heur_mode.A, cudnn.heur_mode.B)
    # An untagged backend entry is the delegating one: candidates C++ holds but
    # never exposes as plans, which Graph::build_plans tries BEFORE its own
    # engine_configs. It belongs to no mode and must keep the lead, or an
    # OPENSOURCE caller gets a native kernel instead of the OSS one.
    out: List[PlanConfig] = [c for c in backend_plans if c.mode is None]
    for mode in modes:
        if mode == cudnn.heur_mode.OPENSOURCE:
            # Mode A without the backend's recommendation: the caller asked for
            # an open-source implementation and the backend's engines are not
            # one. Nothing to place, so the measurements do not come into it.
            out += _mode_a(facts, offered)
        elif mode in a_modes:
            # B asks for a wider search than A and this family has none to give,
            # so it answers as A does. The backend answered B on its own terms.
            ours = _mode_a(facts, offered)
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
