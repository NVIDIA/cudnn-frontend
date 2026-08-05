# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router: builds the ranked execution-plan list at create_execution_plans() time.

    Python Graph API -> create_execution_plans() -> Router -> one ranked list
                                                             (graph.plans)

The list is flat ``PlanConfig(engine_id, knobs)`` entries mixing python engines
and backend engines in any order. Position is rank; ``engine_id`` is identity
and the key the build walk dispatches on — the two are independent, so an
engine from anywhere can sit anywhere in the list.

Candidates come from ``engines.manifest`` (the library's own static table) plus
anything the caller added with ``register_backend()``. The backend's own ranked
entries come from ``graph.backend_plan_entries()``, which is [] when the backend
declined the graph or is not installed. Merging the two sides is
``engines.heuristics.heuristics_sort``'s job — that is the seam a real cost
model replaces.

Policy is pluggable at three levels: subclass ``Router`` and override ``plan()``;
pass one per graph (``pygraph(router=...)`` / ``graph.set_router()``); or swap
the process-wide ``default_router``. A Router places the backend's entries by
calling ``graph.backend_plan_entries()`` (answered once per graph) and putting
the result where it wants — there is no placeholder to expand, so the list a
Router returns is the list, position for position.
"""

import logging
from typing import TYPE_CHECKING, List

from . import heuristics
from .base import BaseEngine, PlanConfig

if TYPE_CHECKING:
    from .._pygraph import pygraph

_LOG = logging.getLogger("cudnn.engines.router")


def decline_types():
    """The exception types that mean "this engine does not serve this graph"."""
    import cudnn

    return (NotImplementedError, cudnn.cudnnGraphNotSupportedError)


class Router:
    """Default policy: every claiming python engine, ranked against the backend."""

    def python_plans(self, graph: "pygraph", engines: List[BaseEngine]) -> List[PlanConfig]:
        """Proposals from the python engines that claim ``graph``, in candidate order.

        An engine declines with ``NotImplementedError`` /
        ``cudnnGraphNotSupportedError`` only; any other exception is a bug in the
        engine and propagates instead of silently costing the user a kernel.
        """
        decline = decline_types()
        out: List[PlanConfig] = []
        for engine in engines:
            try:
                proposals = engine.propose_plans(graph)
            except decline as exc:
                _LOG.debug("engine %s declined the graph: %s", engine.name, exc)
                continue
            for cfg in proposals:
                lo, hi = engine.owned_id_range
                if not lo <= cfg.engine_id < hi:  # no identity injection
                    raise ValueError(f"engine {engine.name!r} proposed a plan with foreign engine_id {cfg.engine_id}")
            out.extend(proposals)
        return out

    def plan(self, graph: "pygraph", engines: List[BaseEngine]) -> List[PlanConfig]:
        # Through the module, not a bound name: heuristics.heuristics_sort is the
        # seam a real ranking policy replaces, and it must be swappable in a
        # live process (tests, experiments) without touching this file.
        return heuristics.heuristics_sort(graph, self.python_plans(graph, engines), graph.backend_plan_entries())


# Process-wide default. Assign a Router subclass to change global policy, or pass
# one to pygraph(router=...) / graph.set_router(...) per graph.
default_router = Router()
