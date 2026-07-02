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

Phase 1: the cuDNN side is a single "let cuDNN heuristics pick" entry appended
after the python plans. That entry is later replaced by the true per-engine
cuDNN configs (read via get_engine_and_knobs_at_index) and the concat becomes a
real heuristics-driven ranking.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from .base import BaseEngine
from .engine_ids import CUDNN_HEURISTIC_ENGINE_ID

if TYPE_CHECKING:
    from ..graph_native import NativeGraph


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

        # Phase 1: cuDNN as one "heuristics decides" entry. TODO: replace with the
        # true per-engine cuDNN configs + a real heuristics-driven ranking merge.
        plans.append(PlanConfig(CUDNN_HEURISTIC_ENGINE_ID))
        return plans


# Process-wide default. Assign a Router subclass to change global policy, or pass
# one to NativeGraph(router=...) / graph.set_router(...) per graph.
default_router = Router()
