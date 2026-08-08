# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend heuristics: produce the ranked plan list for a graph.

``create_execution_plans([heur_mode.A, ...])`` collects the inputs — the parsed
facts, the engine ids on offer, and the backend's own entries tagged with the
mode that produced each — and hands all of it to the graph's family. What the
family returns IS ``graph.plans``, position for position.

One function, everything in view. Ranking is a comparison, so whoever ranks has
to see both sides: an engine cannot see its siblings, and a family that only
saw its own engines could not decide whether the backend belongs in front of
them. That is why there is no per-engine ``propose_plans`` and no second
merge step after this one — splitting the decision is what forced the previous
design to concatenate and call it ranking.

An engine answers two questions only: can I serve this graph
(``check_support``) and compile me this config (``build_plan``).

A family that declares no ``heuristics`` hook falls back to one default plan
per accepting engine, ahead of the backend's entries. Every python engine
belongs to a family — the manifest is the only way one exists — so a graph has
a family's opinion or only the backend's.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from .base import BaseEngine, PlanConfig, decline_types

if TYPE_CHECKING:
    from .._pygraph import pygraph

_LOG = logging.getLogger("cudnn.engines.heuristics")


def default_modes() -> List[Any]:
    """The modes assumed when the caller named none — the backend's own default."""
    import cudnn

    return [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]


def accepts(engine: BaseEngine, graph: "pygraph") -> bool:
    """Whether ``engine`` will serve ``graph``, declines being routing not error."""
    try:
        engine.check_support(graph)
    except decline_types() as exc:
        _LOG.debug("engine %s declined the graph: %s", engine.name, exc)
        return False
    return True


def _unranked(graph: "pygraph", engines: List[BaseEngine], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """The ranking for a family with no heuristics hook: accepting engines, then the backend."""
    return [PlanConfig(e.engine_id, None) for e in engines if accepts(e, graph)] + list(backend_plans)


def rank(graph: "pygraph", engines: List[BaseEngine], backend_plans: List[PlanConfig], modes: Optional[List[Any]] = None) -> List[PlanConfig]:
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
        # express it. Nothing to rank its engines on; the backend serves it.
        return list(backend_plans)

    offered = {e.name: e.engine_id for e in engines}
    plans = list(recommend(modes, facts, offered, list(backend_plans)))
    own = set(offered.values())
    for cfg in plans:
        from .engine_ids import is_python_engine

        if is_python_engine(cfg.engine_id) and cfg.engine_id not in own:
            raise ValueError(f"heuristics for {family.name} returned python engine_id {cfg.engine_id}, which the family does not own or offer")
    return plans
