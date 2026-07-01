"""Base class for NativeGraph execution backends (engines).

This module defines the abstract interface every execution backend must
implement. A backend is one of the interchangeable implementations the Router
dispatches to (Python DSLs, a naive reference, the cuDNN Graph backend, ...) —
see ``docs/python_native_graph_router.md``.

Create a custom backend by subclassing ``BaseEngine`` and implementing
``execute()``; override ``check_support()`` so the Router can decide whether
this backend can run a given graph.

Example:
    class MyEngine(BaseEngine):
        name = "my_engine"
        priority = 50  # lower is tried first by the Router

        def check_support(self, graph):
            for node in graph.nodes:
                if node.node_type != NodeType.MATMUL:
                    raise NotImplementedError(...)

        def execute(self, graph, tensor_data):
            ...  # write results into caller-provided output buffers
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..graph_native import NativeGraph


class BaseEngine(ABC):
    """Abstract base class for graph execution backends.

    A backend executes the operations defined in a NativeGraph. Different
    backends use different implementations (PyTorch reference, cuTile, other
    Python-DSL fusion engines, ...). The Router picks one at
    ``create_execution_plans()`` time by trying each candidate's
    ``check_support()`` in ascending ``priority`` order.

    Attributes:
        name: Human-readable identifier.
        priority: Router ordering hint — lower is preferred / tried first.
            Reference/fallback backends should use a large value.
    """

    name: str = "base"
    priority: int = 100

    def __init__(self):
        pass

    def check_support(self, graph: "NativeGraph") -> None:
        """Raise if this backend cannot execute ``graph``.

        Called by the Router during ``create_execution_plans()``. Raise
        ``NotImplementedError`` / ``ValueError`` / ``RuntimeError`` (unsupported
        op, layout, or hardware) to decline the graph; the Router then tries the
        next candidate, falling back to the cuDNN backend if none accept.

        Default: accept everything (subclasses should narrow this).
        """
        _ = graph

    def get_workspace_size(self) -> int:
        """Workspace bytes this backend needs (default 0)."""
        return 0

    @abstractmethod
    def execute(
        self,
        graph: "NativeGraph",
        tensor_data: Dict[int, Any],
    ) -> None:
        """Execute the whole graph.

        ``tensor_data`` maps tensor UIDs (inputs + outputs) to their device
        data. The backend writes results directly into the caller-provided
        output buffers (matching cuDNN's execution model).
        """
        raise NotImplementedError(f"Engine '{self.name}' must implement execute()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, priority={self.priority})"
