# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-side helpers shared by the linear-attention backends: check_support
dtype gates, the explicit-workspace carver, and the compiled-plan wrapper."""

from __future__ import annotations

from cudnn.engines.base import CompiledPlan, resolve_node_buffers

from cudnn.frost import buffers


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
    def __init__(self, compiled):
        self._compiled = compiled

    def get_workspace_size(self) -> int:
        return self._compiled.workspace_bytes()

    def execute(self, graph, uid_to_data, ctx) -> None:
        node_buffers = resolve_node_buffers(graph, uid_to_data)
        self._compiled(node_buffers, workspace=getattr(ctx, "workspace", None), stream=getattr(ctx, "stream", None))


def _check_contiguous(plan_name: str, **bufs) -> None:
    """Contiguity gate over the caller's buffers (pass-through, no staging)."""
    for name, b in bufs.items():
        if b is None:
            continue
        _ptr, shape, strides, _dtype, _dev = buffers.probe(b)
        if not buffers.is_contiguous(shape, strides):
            raise ValueError(f"{plan_name}: buffer for {name!r} must be contiguous (buffers pass straight to the kernel)")


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
