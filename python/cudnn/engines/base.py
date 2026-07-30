# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend (engine) contract for the Python graph: plan -> compile -> execute.

A backend is one of the interchangeable implementations the Router dispatches
to (Python DSLs, a naive reference, the cuDNN Graph backend, ...). The
lifecycle mirrors a real JIT/DSL engine:

  1. ``propose_plans(graph)``  -> candidate ``PlanConfig`` entries (one per
     configuration the engine wants ranked; decline the whole graph by raising
     ``NotImplementedError`` / ``cudnn.cudnnGraphNotSupportedError``).
  2. ``build_plan(graph, plan)`` -> a ``CompiledPlan`` — the expensive JIT step,
     run ONCE per (graph, selected plan) at ``graph.build_plans()`` time; the
     compiled artifact lives on the graph, so one engine instance is safely
     reusable across graphs.
  3. ``CompiledPlan.execute(graph, tensor_data, ctx)`` — hot path. The
     ``ExecutionContext`` carries the caller's handle / stream / workspace /
     dynamic-shape overrides explicitly; engines must not hard-code a stream or
     silently allocate hidden workspace.

Simple eager engines only implement ``execute()`` — the default ``build_plan``
wraps it in a trivial ``CompiledPlan``.

Example:
    class MyEngine(BaseEngine):
        name = "my_engine"
        engine_id = PYTHON_ENGINE_ID_BASE + 7  # stable id it owns

        def check_support(self, graph):
            for node in graph.nodes:
                if node.node_type != NodeType.MATMUL:
                    raise NotImplementedError(...)

        def execute(self, graph, tensor_data, ctx):
            ...  # write results into caller-provided output buffers
"""

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from .engine_ids import PYTHON_ENGINE_ID_BASE  # noqa: F401 — re-exported for engine authors

if TYPE_CHECKING:
    from ..pygraph import pygraph


@dataclass(frozen=True)
class PlanConfig:
    """One candidate execution plan: an engine id + its knobs.

    ``engine_id`` lives in the shared flat id space (``engine_ids``); knobs are
    engine-specific tuning (cuDNN knob dict, or a python engine's config). The
    plan's source is derived from the id via ``is_python_engine`` — no separate
    field, so cuDNN and python plans are interchangeable in the ranked list.
    One engine may propose several plans differing only in knobs.
    """

    engine_id: int
    knobs: Any = None


@dataclass(frozen=True)
class ExecutionContext:
    """Runtime context passed to a compiled plan at execute time.

    Everything an engine may need is explicit here — no engine should reach
    into private graph state, hard-code a stream, or allocate hidden workspace.
    ``stream`` is resolved from the handle when available (classic
    ``cudnn.set_stream(handle, ...)`` semantics).
    """

    handle: Any = None
    stream: Any = None
    workspace: Any = None
    override_uids: Any = None
    override_shapes: Any = None
    override_strides: Any = None


class CompiledPlan:
    """A compiled (graph, plan) artifact. Subclass for real JIT engines."""

    def get_workspace_size(self) -> int:
        """Workspace bytes this plan needs at execute time (default 0)."""
        return 0

    def execute(self, graph: "pygraph", tensor_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        raise NotImplementedError


class _EagerPlan(CompiledPlan):
    """Default CompiledPlan for simple eager engines (delegates to engine.execute)."""

    def __init__(self, engine: "BaseEngine", plan: PlanConfig):
        self.engine = engine
        self.plan = plan

    def get_workspace_size(self) -> int:
        return self.engine.get_workspace_size()

    def execute(self, graph, tensor_data, ctx: ExecutionContext) -> None:
        self.engine.execute(graph, tensor_data, ctx)


class BaseEngine(ABC):
    """Abstract base class for python graph execution backends.

    Attributes:
        name: Human-readable identifier.
        engine_id: Stable id in the shared flat engine-id space, in the reserved
            Python region (>= PYTHON_ENGINE_ID_BASE). Subclasses MUST declare it;
            the base default (None) is rejected at register_backend().
        default_knobs: Optional default tuning knobs for this engine's plan.
    """

    name: str = "base"
    # Subclasses MUST declare a stable id in the reserved python region; the
    # base intentionally has none so a forgotten override fails at registration
    # instead of silently colliding with another engine.
    engine_id: Any = None
    default_knobs: Any = None

    def __init__(self):
        pass

    def check_support(self, graph: "pygraph") -> None:
        """Raise to decline ``graph``.

        Decline ONLY via ``NotImplementedError`` or
        ``cudnn.cudnnGraphNotSupportedError`` (the classic unsupported-graph
        signal); any other exception is treated as an engine bug and propagates.
        Default: accept everything (subclasses should narrow this).
        """
        _ = graph

    def propose_plans(self, graph: "pygraph") -> List[PlanConfig]:
        """Candidate plans for ``graph``, in this engine's preference order.

        Default: one plan with ``default_knobs`` when ``check_support`` accepts.
        Engines with several viable configurations override this to expose them
        to ranking/autotune (each entry's knobs reach ``build_plan`` verbatim).
        """
        self.check_support(graph)
        return [PlanConfig(self.engine_id, self.default_knobs)]

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: "ExecutionContext" = None) -> CompiledPlan:
        """Compile ``graph`` for ``plan`` (the expensive step; run once per
        graph/plan at build_plans() time). ``ctx`` carries the build context —
        handle and stream when available — so device-specific AoT compilers get
        their inputs explicitly instead of reading private graph state. Default
        wraps eager ``execute()``."""
        return _EagerPlan(self, plan)

    def get_workspace_size(self) -> int:
        """Workspace bytes for eager engines (compiled plans report their own)."""
        return 0

    def execute(self, graph: "pygraph", tensor_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        """Eager execution hook (used by the default ``build_plan``).

        ``tensor_data`` maps tensor UIDs (inputs + outputs) to their device
        data. Write results directly into the caller-provided output buffers,
        on ``ctx.stream`` when set.
        """
        raise NotImplementedError(f"Engine '{self.name}' must implement execute() or build_plan()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, engine_id={self.engine_id})"
