# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST SDPA-backward engines: one BaseEngine per capability cell.

Listed in ``cudnn/engines/manifest.py`` as ONE row owning the
``FROST_SDPA_BWD_ID_BASE`` block, so ``FrostSdpaBwdEngines()`` returns the whole
family and a graph containing an sdpa_backward() node reaches them through the
ordinary lifecycle — no registration call. The row is opt-in
(``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``) until these engines have the arch
coverage to serve graphs unasked.

The capability table, the probe and the lowering stay in ``engines.py``
(``ENGINE_SPECS`` / ``analyze_for`` / ``build``); this file is only the engine
contract around them.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from cudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph

    from .engines import EngineSpec


def _check_workspace(workspace, required: int, name: str) -> None:
    """A FROST executor carves its scratch out of the CALLER's workspace: no
    hidden per-execute allocation, stable pointers, CUDA-graph friendly."""
    if workspace is None:
        raise ValueError(f"{name} needs a {required}-byte workspace; execute() got none — allocate graph.get_workspace_size() bytes and pass it")
    available = workspace.numel() * workspace.element_size() if hasattr(workspace, "numel") else len(workspace)
    if available < required:
        raise ValueError(f"{name} needs a {required}-byte workspace; the buffer provides {available}")


class _FrostSdpaBwdPlan(CompiledPlan):
    """A compiled SDPA-backward executor plus the graph binding it was compiled for."""

    def __init__(self, name: str, compiled: Any):
        self._name = name
        self._compiled = compiled
        # The kernel is bound to specific graph tensors; the variant pack the
        # graph API hands us covers every IO tensor of the graph, so key the
        # kernel's own operands out of it by uid (uids are eager and unique).
        self._tensors = list(compiled.binding.bound_tensors())
        # A bound tensor's uid is fixed once the graph is frozen, so read them
        # here rather than re-walking the list on every execute.
        self._uids = [t.get_uid() for t in self._tensors]
        self._workspace_bytes = int(getattr(compiled, "workspace_bytes", 0) or 0)

    def get_workspace_size(self) -> int:
        return self._workspace_bytes

    def execute(self, graph: "pygraph", uid_to_data, ctx: ExecutionContext) -> None:
        # Keyed by IR tensor object: that is the binding's own identity, and the
        # only key resolve_variant_pack() accepts for an auto-assigned uid.
        pack = {}
        for t, uid in zip(self._tensors, self._uids):
            buf = uid_to_data.get(uid)
            if buf is None:
                missing = [t.get_name() or uid for t, uid in zip(self._tensors, self._uids) if uid_to_data.get(uid) is None]
                raise ValueError(f"{self._name}: the variant pack is missing buffers for {missing}")
            pack[t] = buf
        required = self._workspace_bytes
        if required:
            _check_workspace(ctx.workspace, required, self._name)
            self._compiled(pack, ctx.workspace, stream=ctx.stream)
        else:
            self._compiled(pack, stream=ctx.stream)


class FrostSdpaBwdEngine(BaseEngine):
    """One SDPA-backward capability cell (arch x geometry).

    Wraps a single :class:`~cudnn.sdpa.bwd.engines.EngineSpec`: ``name`` is the
    spec's shipped name and ``engine_id`` is its fixed offset in the family's id
    block (see :func:`FrostSdpaBwdEngines`).
    """

    def __init__(self, spec: "EngineSpec", engine_id: int):
        super().__init__()
        self._spec = spec
        self.name = spec.name
        self.engine_id = engine_id

    def _decline_reason(self, graph: "pygraph", knobs) -> Optional[str]:
        from .engines import analyze_for

        try:
            _, reason = analyze_for(self._spec, graph, knobs)
        except ValueError as exc:
            # ValueError is the analyzer's internal "cannot express this graph";
            # at the engine boundary that is a decline, not a user error.
            return str(exc)
        return reason

    def check_support(self, graph: "pygraph") -> None:
        reason = self._decline_reason(graph, None)
        if reason is not None:
            raise NotImplementedError(f"{self.name}: {reason}")

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from .engines import build

        knobs = plan.knobs if plan is not None else None
        try:
            return _FrostSdpaBwdPlan(self.name, build(self._spec, graph, knobs))
        except (NotImplementedError, ValueError, ImportError) as exc:
            # ImportError: the DSL adapter resolves at build time now (support
            # checks must not pay for it), so a missing cutedsl extra surfaces
            # HERE rather than making the family vanish at import. It is a
            # decline -- the walk moves on and the backend serves the graph.
            raise NotImplementedError(f"{self.name}: {exc}") from exc


def FrostSdpaBwdEngines(ids) -> List[FrostSdpaBwdEngine]:
    """The SDPA-backward engines the manifest asked for, in ENGINE_SPECS order.

    ``ids`` is ``{name: engine_id}`` from engines/manifest.py — the single
    source of engine ids. A spec absent from it is one the manifest is not
    offering (still opt-in gated), so it is simply not built; a spec that has
    no slot AT ALL is caught by test_dispatch, not at runtime.
    """
    from .engines import ENGINE_SPECS

    engines = []
    for spec in ENGINE_SPECS:
        if spec.name in ids:
            engines.append(FrostSdpaBwdEngine(spec, ids[spec.name]))
    return engines
