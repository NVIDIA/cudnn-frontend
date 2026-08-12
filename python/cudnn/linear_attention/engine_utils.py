# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-side helpers shared by the linear-attention backends: check_support
dtype gates, the explicit-workspace carver, and the compiled-plan wrapper."""

from __future__ import annotations

from cudnn.engines.base import CompiledPlan, NodeBuffers, bind_ports

from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace


def _dtype_name(dt) -> str:
    import cudnn

    return {cudnn.data_type.HALF: "float16", cudnn.data_type.BFLOAT16: "bfloat16", cudnn.data_type.FLOAT: "float32"}[dt]


def _require_dtype(engine: str, node, port: str, want, *, out: bool = False) -> None:
    import cudnn  # noqa: F401 — `want` members come from cudnn.data_type

    t = (node.outputs if out else node.inputs).get(port)
    if t is None:
        return
    got = t.get_data_type()
    if got is None:
        return  # unset (e.g. inferred outputs): the kernel validates the buffer
    wanted = want if isinstance(want, tuple) else (want,)
    if got not in wanted:
        names = "/".join(w.name for w in wanted)
        raise NotImplementedError(f"{engine}: '{port}' must be {names} (the kernel-native dtype; no staging), got {got}")


def _require_state_pair(engine: str, node) -> None:
    """The kernels require matching initial/final state dtypes."""
    s0 = node.inputs.get("initial_state")
    fs = node.outputs.get("final_state")
    if s0 is None or fs is None:
        return
    a, b = s0.get_data_type(), fs.get_data_type()
    if a is not None and b is not None and a != b:
        raise NotImplementedError(f"{engine}: initial_state and final_state dtypes must match (got {a} vs {b})")


class _FrostPlan(CompiledPlan):
    """A compiled linear-attention kernel, driven from the normalized pack.

    The port-to-slot join is a property of the graph, so it happens once and is
    kept; only the addresses change between executes. What the kernel receives
    is built from the pack, never the caller's object — the geometry it is
    checked against and the geometry it runs on are then the same reading.
    """

    takes_variant_pack = True

    def __init__(self, compiled):
        self._compiled = compiled
        self._ports = None
        self._name = type(compiled).__name__

    def get_workspace_size(self) -> int:
        return self._compiled.workspace_bytes()

    def execute(self, graph, variant_pack, ctx) -> None:
        ports = self._ports
        if ports is None:
            ports = self._ports = bind_ports(graph, variant_pack)
        _check_contiguous(variant_pack, ports)
        node_buffers = {}
        for node, slots in ports.items():
            names = list(slots.inputs) + list(slots.outputs)
            views = variant_pack.operands(list(slots.inputs.values()) + list(slots.outputs.values()))
            split = len(slots.inputs)
            node_buffers[node] = NodeBuffers(dict(zip(names[:split], views[:split])), dict(zip(names[split:], views[split:])))
        required = self._compiled.workspace_bytes()
        workspace = Workspace.over(variant_pack, required, self._name) if required else None
        self._compiled(node_buffers, workspace=workspace, stream=ctx.stream)


def _check_contiguous(variant_pack, ports) -> None:
    """Contiguity gate over the whole pack, decided from the strides it holds.

    The dim and stride were taken from the caller's object once, at
    normalization; probing each buffer again cost 8.6 us apiece — nine per GDN
    forward — to learn what the pack already knows. The scan itself is in the
    native pack, 0.24 us for eight operands, so only naming the offender costs
    anything and that happens once, on the way to raising.

    One gate for every kernel rather than a call per compiled callable naming
    its own ports: the rule was the same list every time, and a port added to a
    node but forgotten there would have gone unchecked.
    """
    ok, offender = variant_pack.all_contiguous()
    if ok:
        return
    for node, slots in ports.items():
        for direction in (slots.inputs, slots.outputs):
            for port, slot in direction.items():
                if slot == offender:
                    raise ValueError(f"cudnn.frost {node.name!r}: buffer for {port!r} must be contiguous (buffers pass straight to the kernel)")
    raise ValueError(f"cudnn.frost: the buffer at variant-pack slot {offender} must be contiguous")


_pinned_engines = None  # e.g. ("gdn_cutile",) -- set by a suite, None => the manifest decides


def pin_engines(names):
    """Force planning onto the named engines for this process, or None to stop.

    A suite that means to validate ONE implementation says which, by name. It
    is not a way to add an engine: every engine is in the manifest, and this
    only narrows which of them a plan may land on.
    """
    global _pinned_engines
    previous = _pinned_engines
    _pinned_engines = tuple(names) if names else None
    return previous


def apply_pin(graph):
    """Select the pinned engine's plan, if a pin is in force.

    Runs after create_execution_plans(), so it selects from the ranked list the
    heuristics produced rather than replacing them.
    """
    if not _pinned_engines:
        return
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    index = next((i for i, n in enumerate(names) if any(n == p or n.startswith(p + "[") for p in _pinned_engines)), None)
    if index is None:
        raise AssertionError(f"pinned engines {list(_pinned_engines)} produced no plan; plans={names}")
    graph.select_plan(index)
