"""Router: builds the ranked execution-plan list at plan-creation time.

Implements the dispatch stage of the Python API unification proposal:

    Python Graph API -> create_execution_plans() -> Router -> ranked plan list
                                                             (one flat (engine_id,
                                                              knobs) list mixing
                                                              python DSLs + cuDNN)

Routing happens at ``create_execution_plans()`` time, NOT at graph construction,
so graph building stays backend-agnostic (lazy lowering). The Router returns a
flat list of ``PlanConfig(engine_id, knobs)`` — Python engines (ids in the
reserved high region) whose ``check_support()`` accepts the graph, plus the
cuDNN side. Dispatch on each plan's id (``is_python_engine``) decides whether to
run via the Python registry or lower to the cuDNN C++ backend.

Contract for the future heuristics MR (ranking policy is intentionally NOT
decided here — only the flexibility to decide it later):

1. Policy is pluggable at three levels: subclass ``Router`` and override
   ``plan()``; pass per-graph via ``pygraph(router=...)`` / ``set_router()``;
   or swap the process-wide ``default_router``.
2. ``plan()`` may return ANY ordering/mix — python-first, cuDNN-first,
   conditional on the graph — the lifecycle dispatches purely on each entry's
   id (``is_python_engine``). The current default is a placeholder concat.
3. "Query both": the Router receives the graph and may trigger lowering to ask
   the loaded backend's own heuristics (get_engine_count /
   get_engine_and_knobs_at_index on the lowered graph). Backend engine sets
   vary by backend version and MUST be discovered per graph at plan time —
   never statically enumerated in frontend code.
4. Specific backend entries: ``PlanConfig(engine_id>=0, knobs)`` can carry a
   concrete cuDNN engine config in the same list. Honoring it at build time
   (cpp ``create_execution_plan(engine_id, knobs)`` instead of the heuristics
   path) is the designated extension point in ``pygraph._lower_cudnn_plan``.
   ``select_plan(i)`` + ``get_execution_plan_count()`` already expose the
   ranked list for autotune-style selection.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from .base import BaseEngine
from .engine_ids import CUDNN_HEURISTIC_ENGINE_ID

if TYPE_CHECKING:
    from ..pygraph import NativeGraph


@dataclass
class PlanConfig:
    """One candidate execution plan: an engine id + its knobs.

    ``engine_id`` lives in the shared flat id space (``engine_ids``); knobs are
    engine-specific tuning (cuDNN knob dict, or a python engine's config). The
    plan's source is derived from the id via ``is_python_engine`` — no separate
    field, so cuDNN and python plans are interchangeable in the ranked list.
    """

    engine_id: int
    knobs: Any = None


class Router:
    """Default policy: python engines that support the graph, then cuDNN."""

    def plan(self, graph: "NativeGraph", backends: List[BaseEngine]) -> List[PlanConfig]:
        """Return the ranked candidate plan list for ``graph``.

        Python engines are included (by ascending ``engine_id``, a stable order)
        when their ``check_support(graph)`` does not raise; the cuDNN side is
        appended as a single heuristics entry. A backend declines by raising
        ``NotImplementedError`` / ``ValueError`` / ``RuntimeError``.
        """
        plans: List[PlanConfig] = []
        for engine in sorted(backends, key=lambda e: e.engine_id):
            try:
                engine.check_support(graph)
            except (NotImplementedError, ValueError, RuntimeError):
                continue
            plans.append(PlanConfig(engine.engine_id, getattr(engine, "default_knobs", None)))

        # The cuDNN side is ONE delegating entry by design: the frontend owns
        # only its python-engine id segment and must work against any (incl.
        # future) backend version, so the backend's engine set can never be
        # statically enumerated here — it is discovered per graph at plan time
        # via the backend's own heuristics/query API (get_engine_and_knobs_at_
        # index on the lowered graph) when a caller wants to expand or autotune.
        plans.append(PlanConfig(CUDNN_HEURISTIC_ENGINE_ID))
        return plans


# Process-wide default. Assign a Router subclass to change global policy, or pass
# one to NativeGraph(router=...) / graph.set_router(...) per graph.
default_router = Router()
