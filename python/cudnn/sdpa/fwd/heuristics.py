# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""How the SDPA-forward family ranks plans for a graph.

``engines/heuristics.rank`` hands over the parsed facts, this family's engine
ids, and the backend's entries tagged by mode; what :func:`recommend` returns is
``graph.plans``, position for position. The whole comparison is here because
that is the only place it can be made — a cell cannot see its siblings, and
neither side of the FROST/backend split can place the other.

Per mode:

- **A** — candidates worth running, best guess first. Today every cell declares
  one point per knob axis, so a cell contributes one entry; a cell that offers
  a real domain names its choice here, and the runners-up behind it for a
  caller that autotunes.
- **FALLBACK** — the config expected to build where mode A's choice may not.
  Nothing here is chosen for speed.
- **OPENSOURCE** — mode A without the backend's recommendation: these cells ARE
  the open-source implementation and the backend's engines are not. Combine it
  to measure coverage — ``[OPENSOURCE, A, FALLBACK]`` tries every FROST config
  first and still has the backend behind it, so a graph that runs on a backend
  plan is one FROST does not cover.
- **B** — answered as A: it asks for a wider search than A, and this family has
  none to give.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


def _sole(values):
    """The only value on an axis, or None where the row declares no domain (or
    offers a choice this family has no rule to make yet)."""
    return next(iter(values)) if len(values) == 1 else None


def _knobs(caps: Capabilities) -> SdpaFwdKnobs:
    """The knob request for a cell's declared point.

    A field stays None only on an axis whose capability row declares no domain
    — the engine then has no say to honour. It never means "engine, pick for
    me": that reading is what let the same choice be made twice, once here and
    once inside the adapter.
    """
    return SdpaFwdKnobs(sched_policy=_sole(caps.sched_policies), tile_m=_sole(caps.tile_ms), tile_n=_sole(caps.tile_ns), cga=_sole(caps.cgas))


def _eligible(facts, offered: Dict[str, int]):
    """(engine_id, spec) for each offered cell whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


def _candidates(facts, offered: Dict[str, int], mode) -> List[PlanConfig]:
    """One entry per eligible cell, at the config its capability row declares."""
    out = []
    for engine_id, spec in _eligible(facts, offered):
        knobs = _knobs(spec.capabilities)
        if mismatch(spec.capabilities, facts, knobs) is None:
            out.append(PlanConfig(engine_id, knobs, mode=mode))
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
            out += _candidates(facts, offered, cudnn.heur_mode.A)
        elif mode in (cudnn.heur_mode.A, cudnn.heur_mode.B):
            ours = _candidates(facts, offered, mode)
            theirs = [c for c in backend_plans if c.mode == mode]
            out += (ours + theirs) if _leads(offered, ours) else (theirs + ours)
        elif mode == cudnn.heur_mode.FALLBACK:
            out += _candidates(facts, offered, mode) + [c for c in backend_plans if c.mode == mode]

    seen, ranked = set(), []
    for cfg in out:
        key = (cfg.engine_id, repr(cfg.knobs), cfg.cpp_index)
        if key not in seen:
            seen.add(key)
            ranked.append(cfg)
    return ranked
