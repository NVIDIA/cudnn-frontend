# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend (engine) contract for the Python graph: claim -> compile -> execute.

A backend is one of the interchangeable implementations the ranked plan list
dispatches to (Python DSLs, a naive reference, the cuDNN Graph backend, ...). The
lifecycle mirrors a real JIT/DSL engine:

  1. ``check_support(graph)`` -> accept, or decline the whole graph by raising
     ``NotImplementedError`` / ``cudnn.cudnnGraphNotSupportedError``. An engine
     does NOT propose its own plans: which configs to try, in what order, and
     where the backend's own entries belong is one comparison across every
     candidate, which no engine can make from the inside. That lives in
     ``engines/heuristics.py`` and the family hook it dispatches to.
  2. ``build_plan(graph, plan)`` -> a ``CompiledPlan`` — the expensive JIT step,
     run ONCE per (graph, selected plan) at ``graph.build_plans()`` time; the
     compiled artifact lives on the graph, so one engine instance is safely
     reusable across graphs.
  3. ``CompiledPlan.execute(graph, uid_to_data, ctx)`` — hot path.
     ``uid_to_data`` is the caller's variant pack (tensor uid -> device
     buffer, exactly as the classic backend receives it); the
     ``ExecutionContext`` carries the caller's handle / stream / workspace /
     dynamic-shape overrides explicitly; engines must not hard-code a stream
     or silently allocate hidden workspace. Engines that address buffers by
     port name call ``resolve_node_buffers(graph, uid_to_data)`` (see below).

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

        def execute(self, graph, uid_to_data, ctx):
            node_buffers = resolve_node_buffers(graph, uid_to_data)
            ...  # write results into caller-provided output buffers
"""

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from .engine_ids import PYTHON_ENGINE_ID_BASE  # noqa: F401 — re-exported for engine authors

if TYPE_CHECKING:
    from ..pygraph import pygraph


def decline_types():
    """The exception types that mean "this engine does not serve this graph".

    ImportError counts: an engine whose optional dependency is absent cannot
    serve the graph, and since lowering imports are deferred past check_support
    that only becomes visible at build time.
    """
    import cudnn

    return (NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError)


@dataclass(frozen=True)
class PlanConfig:
    """One candidate execution plan: an engine id + its knobs.

    ``engine_id`` lives in the shared flat id space (``engine_ids``); knobs are
    engine-specific tuning (cuDNN knob dict, or a python engine's config). The
    plan's source is derived from the id via ``is_python_engine`` — no separate
    field, so cuDNN and python plans are interchangeable in the ranked list.
    One engine may propose several plans differing only in knobs.

    ``cpp_index`` is set only on backend entries: the position this plan holds
    in the lowered graph's own plan list, so building it is one
    ``build_plan_at_index`` instead of a rebuild from (engine_id, knobs).

    ``mode`` is the heuristic mode that produced the entry. Ranking needs it:
    "the backend's mode-A entries ahead of ours, its fallbacks behind" cannot
    be said about a list whose entries do not remember where they came from.
    """

    engine_id: int
    knobs: Any = None
    cpp_index: Any = None
    mode: Any = None


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


@dataclass(frozen=True)
class NodeBuffers:
    """Per-node ``{port_name: caller buffer}`` maps, the result of
    ``resolve_node_buffers``. Only WIRED, NON-VIRTUAL ports
    appear, and every one is guaranteed a buffer (a missing buffer raises at
    resolution). Torch tensors arrive detached — both DLPack and
    ``__cuda_array_interface__`` refuse to export ``requires_grad`` tensors,
    and graph-level gradients are the backward nodes' contract, never
    autograd tracing through an engine. Virtual intermediates carry no
    caller buffers; engines that chain them across nodes key their own
    scratch by ``node.inputs[port].uid``."""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


def resolve_node_buffers(graph: "pygraph", uid_to_data: Dict[int, Any]) -> Dict[Any, NodeBuffers]:
    """Join each node's wired ports with the caller's variant pack into
    per-node ``NodeBuffers``. Strict: every non-virtual port must have a
    buffer. Torch tensors are detached here (DLPack and
    ``__cuda_array_interface__`` refuse ``requires_grad`` export; graph-level
    gradients belong to the backward nodes, not to autograd tracing through
    engines)."""

    def resolve(node, ports, direction):
        bufs = {}
        for port, t in ports.items():
            if t is None:
                continue
            b = uid_to_data.get(t.uid)
            if b is None:
                if t.is_virtual:
                    continue  # engine-internal intermediate
                raise ValueError(f"node {node.name!r}: no buffer for {direction} port {port!r} (tensor {t.name!r})")
            bufs[port] = b.detach() if hasattr(b, "detach") else b
        return bufs

    return {node: NodeBuffers(resolve(node, node.inputs, "input"), resolve(node, node.outputs, "output")) for node in graph.nodes}


class CompiledPlan:
    """A compiled (graph, plan) artifact. Subclass for real JIT engines."""

    def get_workspace_size(self) -> int:
        """Workspace bytes this plan needs at execute time (default 0)."""
        return 0

    def execute(self, graph: "pygraph", uid_to_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        raise NotImplementedError


class _EagerPlan(CompiledPlan):
    """Default CompiledPlan for simple eager engines (delegates to engine.execute)."""

    def __init__(self, engine: "BaseEngine", plan: PlanConfig):
        self.engine = engine
        self.plan = plan

    def get_workspace_size(self) -> int:
        return self.engine.get_workspace_size()

    def execute(self, graph, uid_to_data, ctx: ExecutionContext) -> None:
        self.engine.execute(graph, uid_to_data, ctx)


class BaseEngine(ABC):
    """Abstract base class for python graph execution backends.

    Attributes:
        name: Human-readable identifier.
        engine_id: Stable id in the shared flat engine-id space, in the reserved
            Python region (>= PYTHON_ENGINE_ID_BASE). Subclasses MUST declare it;
            the base default (None) is rejected when the manifest builds it.
        behavior_notes / numerical_notes: what this engine's plans are, in the
            same vocabulary the backend's plans answer in, so
            deselect_behavior_notes(...) and friends mean one thing across the
            ranked list. Declared, not inferred — an engine that says nothing
            is simply never filtered out by a note.
    """

    name: str = "base"
    # Subclasses MUST declare a stable id in the reserved python region; the
    # base intentionally has none so a forgotten override fails at registration
    # instead of silently colliding with another engine.
    engine_id: Any = None
    behavior_notes: tuple = ()
    numerical_notes: tuple = ()

    def __init__(self):
        pass

    # An engine answers for exactly ONE id: the one its manifest slot handed it.
    # (The id_end / owned_id_range block this used to declare was for engines
    # registered at runtime, which no longer exist.)

    def check_support(self, graph: "pygraph") -> None:
        """Raise to decline ``graph``.

        Decline ONLY via ``NotImplementedError`` or
        ``cudnn.cudnnGraphNotSupportedError`` (the classic unsupported-graph
        signal); any other exception is treated as an engine bug and propagates.
        Default: accept everything (subclasses should narrow this).
        """
        _ = graph

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

    def execute(self, graph: "pygraph", uid_to_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        """Eager execution hook (used by the default ``build_plan``).

        ``uid_to_data`` maps tensor UIDs (inputs + outputs) to their device
        data; call ``resolve_node_buffers`` to address buffers by port name.
        Write results directly into the caller-provided output buffers, on
        ``ctx.stream`` when set.
        """
        raise NotImplementedError(f"Engine '{self.name}' must implement execute() or build_plan()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, engine_id={self.engine_id})"
