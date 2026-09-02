# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend heuristics: produce the ranked plan list for a graph.

``create_execution_plans([heur_mode.A, ...])`` collects the inputs — the parsed
facts, the engine ids on offer, and the backend's own entries tagged with the
mode that produced each — and assembles ``graph.plans``.

Two layers, deliberately separated:

- The FAMILY hook is :func:`recommend`-shaped: ``(kind, facts, offered) ->
  [PlanConfig]`` — pure, backend-blind, import-light. It answers one question:
  which of MY engines serve these facts, with which complete knob assignments,
  best first. It never sees the backend, modes, or another family.

- PLACEMENT lives HERE (:func:`_assemble`), once for every family: python
  proposals lead the backend's entries inside each mode block. That is a
  standing assumption, not a measurement — an OSS engine that loses to the
  backend gets fixed or pulled, not demoted; and an autotune (build ALL) pass
  measures every entry regardless of order, so the order only decides the
  default winner. The delegating entry, dedup, and the mode strip are all
  placement bookkeeping and stay out of the families.

An engine answers two questions only: can I serve this graph
(``check_support``) and compile me this config (``build_plan``).

A family that declares no hook falls back to one default plan per accepting
engine, ahead of the backend's entries. Every python engine belongs to a
family — the manifest is the only way one exists — so a graph has a family's
proposals or only the backend's.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from .base import BaseEngine, PlanConfig, decline_types

_LOG = logging.getLogger("cudnn.engines.heuristics")


def default_modes() -> List[Any]:
    """The modes assumed when the caller named none — the backend's own default."""
    import cudnn

    return [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]


def accepts(engine: BaseEngine, graph) -> bool:
    """Whether ``engine`` will serve ``graph``, declines being routing not error."""
    try:
        engine.check_support(graph)
    except decline_types() as exc:
        _LOG.debug("engine %s declined the graph: %s", engine.name, exc)
        return False
    return True


def _unranked(graph, engines: List[BaseEngine], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The list for a family with no hook: accepting engines, then the backend."""
    return [PlanConfig(e.engine_id, None) for e in engines if accepts(e, graph)] + [_strip(c) for c in backend_plans]


def _strip(cfg: PlanConfig) -> PlanConfig:
    """A final-list entry: (engine_id, knobs[, cpp_index]) — the mode tag is
    assembly bookkeeping and never reaches ``graph.plans``."""
    if cfg.mode is None and cfg.cpp_index is None:
        return cfg
    return PlanConfig(cfg.engine_id, cfg.knobs, cpp_index=cfg.cpp_index)


def _assemble(modes: List[Any], recommend: Callable[[str], List[PlanConfig]], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The final ranked list: mode block by mode block in the caller's order,
    python proposals leading the backend's entries inside each block.

    ``recommend(kind)`` is the family's hook already bound to (facts, offered):
    ``kind`` is ``"A"`` (candidates worth timing, best first — also the answer
    to B, which asks for a wider search the families have none to give) or
    ``"FALLBACK"`` (the config expected to build where A's choice may not).

    An untagged backend entry is the delegating one: OSS candidates C++ holds
    but never exposes as plans, so it cannot be enumerated. It belongs to no
    mode, and it is NOT a pure OSS entry — Graph::build_plans tries the OSS
    engine and, if that one declines, falls through to the native
    engine_configs already enqueued. So it leads the BACKEND's entries but not
    ours: ahead of our OPENSOURCE block it would answer an OSS-coverage
    question with a native kernel.

    Asking for ``[A, FALLBACK]`` puts every tuned candidate — both sides' —
    ahead of every fallback. A plan repeated across blocks keeps its first
    position. Identity is (engine, knobs): cpp_index is only WHERE one backend
    query put a plan, so keying on it would let one config both modes return
    through as two entries — and an autotuner would build and time it twice.
    """
    import cudnn

    delegating = [c for c in backend_plans if c.mode is None]
    out: List[PlanConfig] = []
    for mode in modes:
        if mode == cudnn.heur_mode.OPENSOURCE:
            out += recommend("A") + delegating
        elif mode in (cudnn.heur_mode.A, cudnn.heur_mode.B):
            out += recommend("A") + delegating + [c for c in backend_plans if c.mode == mode]
        elif mode == cudnn.heur_mode.FALLBACK:
            out += recommend("FALLBACK") + delegating + [c for c in backend_plans if c.mode == mode]
    # A delegate with no mode asked for it (the backend has engines but exposed
    # no plans) would otherwise be dropped.
    out += delegating

    seen, ranked = set(), []
    for cfg in out:
        key = (cfg.engine_id, repr(cfg.knobs))
        if key not in seen:
            seen.add(key)
            ranked.append(_strip(cfg))
    return ranked


def rank(graph, engines: List[BaseEngine], backend_plans: List[PlanConfig], modes: Optional[List[Any]] = None) -> List[PlanConfig]:
    """The ranked plan list for ``graph`` — what ``create_execution_plans`` stores.

    ``engines`` are this graph's python candidates and ``backend_plans`` the
    backend's own entries, each already tagged with its ``mode``.
    """
    from . import manifest

    modes = list(modes) if modes else default_modes()
    family = manifest.family_for(graph)
    recommend = manifest.resolve_heuristics(family) if family is not None else None
    if recommend is None:
        return _unranked(graph, engines, backend_plans)

    analyzer = manifest.resolve_analyzer(family)
    facts = graph._facts_for(analyzer) if analyzer is not None else None
    if facts is None:
        # The family claims the graph by node type but its analyzer cannot
        # express it. Nothing to recommend on; the backend serves it.
        return [_strip(c) for c in backend_plans]

    offered = {e.name: e.engine_id for e in engines}
    plans = _assemble(modes, lambda kind: list(recommend(kind, facts, offered)), backend_plans)
    own = set(offered.values())
    for cfg in plans:
        from .engine_ids import is_python_engine

        if is_python_engine(cfg.engine_id) and cfg.engine_id not in own:
            raise ValueError(f"heuristics for {family.name} returned python engine_id {cfg.engine_id}, which the family does not own or offer")
    return plans
