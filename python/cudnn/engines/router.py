"""Router: selects an execution backend at plan-creation time.

Implements the dispatch stage of the Python API unification proposal:

    Python Graph API -> create_execution_plans() -> Router -> selected backend
                                                             (QDSL / CTM / Triton /
                                                              reference / cuDNN Graph)

Routing happens at ``create_execution_plans()`` time, NOT at graph
construction, so graph building stays backend-agnostic (lazy lowering). The
default policy tries each registered backend's ``check_support()`` in ascending
``priority`` order and returns the first that accepts the graph; ``None`` means
"no native backend accepted — fall back to the cuDNN Graph backend".

Custom policies (cost model, user pin, benchmark-driven) subclass ``Router``
and override ``select()``.
"""

from typing import TYPE_CHECKING, List, Optional

from .base import BaseEngine

if TYPE_CHECKING:
    from ..graph_native import NativeGraph


class Router:
    """Default backend-selection policy: first-supporting, by priority."""

    def select(self, graph: "NativeGraph", candidates: List[BaseEngine]) -> Optional[BaseEngine]:
        """Return the backend to run ``graph``, or ``None`` for the cuDNN path.

        Candidates are tried in ascending ``priority`` order; the first whose
        ``check_support(graph)`` does not raise is selected. A backend declines
        by raising ``NotImplementedError`` / ``ValueError`` / ``RuntimeError``.
        """
        for engine in sorted(candidates, key=lambda e: getattr(e, "priority", 100)):
            try:
                engine.check_support(graph)
            except (NotImplementedError, ValueError, RuntimeError):
                continue
            return engine
        return None


# Process-wide default. Assign a Router subclass to change global routing policy,
# or pass one to NativeGraph(router=...) / graph.set_router(...) per graph.
default_router = Router()
