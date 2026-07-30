# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router: builds the ranked execution-plan list at plan-creation time.

Implements the dispatch stage of the Python API unification proposal:

    Python Graph API -> create_execution_plans() -> Router -> ranked plan list
                                                             (one flat (engine_id,
                                                              knobs) list mixing
                                                              python DSLs + backend)

Routing happens at ``create_execution_plans()`` time, NOT at graph construction,
so graph building stays backend-agnostic (lazy lowering). The Router returns a
flat list of ``PlanConfig(engine_id, knobs)``: Python engines (ids in the
reserved high region) whose ``check_support()`` accepts the graph, plus AT MOST
ONE backend delegating entry (``BACKEND_HEURISTIC_ENGINE_ID``). Dispatch on each
plan's id (``is_python_engine``) decides whether to run via the Python registry
or lower to the cuDNN C++ backend.

WHAT THIS MR SUPPORTS (the enforced contract — ``create_execution_plans()``
validates the final Router output, whatever the Router implementation):

* python entries must name engines registered on the graph;
* the only legal non-python entry is ONE backend delegating sentinel — the
  backend's own plans stay behind it, addressed via the classic at-index APIs
  (a separate, backend-owned index space);
* an empty plan list is rejected (there is no legal empty planning state).

Concrete backend engine configs as first-class routed entries
(``PlanConfig(cudnn_engine_id, knobs)`` interleaved with python plans) are NOT
representable in this MR: they need a typed plan representation and a build
path via cpp ``create_execution_plan(engine_id, knobs)`` — that is the
heuristics/autotune follow-up MR's job, not one extra lowering branch. What IS
already decided and stable here: routed indices never shift (the sentinel never
expands in place), and the backend engine set is discovered per graph at plan
time (get_engine_count / get_engine_and_knobs_at_index on the lowered graph) —
never statically enumerated in frontend code, because it varies by backend
version.

Policy remains pluggable at three levels: subclass ``Router`` and override
``plan()``; pass per-graph via ``pygraph(router=...)`` / ``set_router()``
(before planning); or swap the process-wide ``default_router``. ``plan()`` may
return any ordering/mix of the representable entries — python-first,
backend-first, interleaved, conditional on the graph. The current default is a
placeholder concat.
"""

from typing import TYPE_CHECKING, List

from .base import BaseEngine, PlanConfig
from .engine_ids import BACKEND_HEURISTIC_ENGINE_ID

if TYPE_CHECKING:
    from .._pygraph import pygraph


class Router:
    """Default policy: python engines that support the graph, then the backend."""

    def plan(self, graph: "pygraph", backends: List[BaseEngine]) -> List[PlanConfig]:
        """Return the ranked candidate plan list for ``graph``.

        Python engines are included (by ascending ``engine_id``, a stable order)
        when their ``check_support(graph)`` does not raise. A backend DECLINES
        only via ``NotImplementedError`` or ``cudnn.cudnnGraphNotSupportedError``
        (the classic unsupported-graph signal); any other exception is a bug in
        the engine and propagates to the caller instead of silently falling back.
        """
        import cudnn

        decline = (NotImplementedError, cudnn.cudnnGraphNotSupportedError)
        plans: List[PlanConfig] = []
        for engine in sorted(backends, key=lambda e: e.engine_id):
            try:
                proposals = engine.propose_plans(graph)
            except decline:
                continue
            for pc in proposals:
                if pc.engine_id != engine.engine_id:  # no identity injection
                    raise ValueError(f"engine {engine.name!r} proposed a plan with foreign engine_id {pc.engine_id}")
            plans.extend(proposals)

        # The backend side is ONE delegating entry by design: the frontend owns
        # only its python-engine id segment and must work against any (incl.
        # future) backend version, so the backend's engine set can never be
        # statically enumerated here — it is discovered per graph at plan time
        # via the backend's own heuristics/query API (get_engine_and_knobs_at_
        # index on the lowered graph) when a caller wants to expand or autotune.
        plans.append(PlanConfig(BACKEND_HEURISTIC_ENGINE_ID))
        return plans


# Process-wide default. Assign a Router subclass to change global policy, or pass
# one to pygraph(router=...) / graph.set_router(...) per graph.
default_router = Router()
