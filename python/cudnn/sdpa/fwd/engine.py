# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST SDPA-forward engines: one BaseEngine per capability cell.

Listed in ``cudnn/engines/manifest.py`` as ONE row owning the
``FROST_SDPA_FWD_ID_BASE`` block, so ``FrostSdpaFwdEngines()`` returns the whole
family and a graph containing an sdpa() node reaches them through the ordinary
lifecycle — no registration call. The row is opt-in
(``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``) until these engines have the arch
coverage to serve graphs unasked.

The capability table, the probe and the lowering are unchanged and stay in
``engines.py`` (``ENGINE_SPECS`` / ``analyze_for`` / ``build``); this file is
only the engine contract around them.
"""

from typing import TYPE_CHECKING, Any, List, Optional

from cudnn import behavior_note
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


class _FrostSdpaFwdPlan(CompiledPlan):
    """A compiled SDPA-forward executor plus the graph binding it was compiled for."""

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
        # Prepared launch (plan-time argument template, positional tvm-ffi call): binds the graph's
        # normalized VariantPack; without one the plan is handed the raw uid map as before.
        self._prepared = getattr(compiled, "prepared", None)
        self._default_stream = getattr(compiled, "default_stream", None)  # the caller's current stream when the handle carries none
        self.takes_variant_pack = self._prepared is not None
        self._stream_handles: dict = {}

    def get_workspace_size(self) -> int:
        return self._workspace_bytes

    def execute(self, graph: "pygraph", uid_to_data, ctx: ExecutionContext) -> None:
        if self._prepared is not None:
            pack = uid_to_data  # a VariantPack (takes_variant_pack)
            if pack is None:
                raise ValueError(f"{self._name}: the graph could not normalize the variant pack for this plan")
            ws_ptr = 0
            if self._workspace_bytes:
                ws_ptr, nbytes = pack.workspace, pack.workspace_bytes
                if not ws_ptr:
                    raise ValueError(
                        f"{self._name} requires a {self._workspace_bytes}-byte workspace but execute() received none; allocate graph.get_workspace_size() bytes"
                    )
                if nbytes and nbytes < self._workspace_bytes:  # 0: a bare address, size unknown
                    raise ValueError(
                        f"{self._name}: needs a {self._workspace_bytes}-byte workspace, got {nbytes} bytes (size it with graph.get_workspace_size())"
                    )
            raw_stream = ctx.stream
            if raw_stream is None:
                # No handle stream: the caller's current stream, as the tensor path's _get_default_stream does
                # (a legacy-stream launch would run eagerly inside a CUDA-graph capture and leave the graph empty).
                cu_stream = self._default_stream() if self._default_stream is not None else None
                stream_int = int(cu_stream) if cu_stream is not None else 0
            else:
                cu_stream = self._stream_handles.get(raw_stream)
                if cu_stream is None:
                    from cuda.bindings import driver as _drv

                    cu_stream = self._stream_handles[raw_stream] = _drv.CUstream(raw_stream)
                stream_int = int(raw_stream)
            self._prepared.execute(pack, ws_ptr, cu_stream, stream_int)
            return
        # The executor's operands out of the graph-wide variant pack, keyed by
        # IR tensor identity (the binding's own key).
        resolved = {}
        for t, uid in zip(self._tensors, self._uids):
            buf = uid_to_data.get(uid)
            if buf is None:
                missing = [t.get_name() or uid for t, uid in zip(self._tensors, self._uids) if uid_to_data.get(uid) is None]
                raise ValueError(f"{self._name}: the variant pack is missing buffers for {missing}")
            resolved[id(t)] = buf
        required = self._workspace_bytes
        if required:
            _check_workspace(ctx.workspace, required, self._name)
            self._compiled.execute_resolved(resolved, ctx.workspace, stream=ctx.stream)
        else:
            self._compiled.execute_resolved(resolved, stream=ctx.stream)

    def _launch_sequence(self):
        """Run the kernel over probe buffers and record what it launches.

        The executor wants a pack keyed by the binding's own tensor objects, so
        that is all this supplies; export itself is shared (CompiledPlan). The
        probe buffers carry each tensor's graph strides, which for SDPA are a
        BHSD view over BSHD storage rather than anything row-major.

        INCOMPLETE, and it will say so rather than write a bad artifact. One
        parameter is left: ``problem_size``, a tuple of six int32 passed as a
        SINGLE argument, which the payload has no kind for. Adding one means a
        tvm-ffi container built once at load -- building it per call would add
        host cost to the path whose cheapness is the point.

        Everything else about this plan does export: the recorder carves the
        ``sinks`` / ``seq_kv`` / ``o_desc`` dummies (zero-filled stand-ins for
        optional ports the graph did not ask for) out of the engine workspace,
        and binds the keyword-called kernel positionally.
        """
        from cudnn.engines.cutedsl_aot import record_launch_sequence

        required = self.get_workspace_size()

        def run(buffers, workspace, stream):
            pack = {t: buffers[t.get_uid()] for t in self._tensors}
            if required:
                self._compiled(pack, workspace, stream=stream)
            else:
                self._compiled(pack, stream=stream)

        return record_launch_sequence(run, self._tensors, required)


class FrostSdpaFwdEngine(BaseEngine):
    """One SDPA-forward capability cell (arch x phase x geometry x quantization).

    Wraps a single :class:`~cudnn.sdpa.fwd.engines.EngineSpec`: ``name`` is the
    spec's shipped name and ``engine_id`` is its fixed offset in the family's id
    block (see :func:`FrostSdpaFwdEngines`).
    """

    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

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

    # Public knob vocabulary (BaseEngine contract): the native SdpaFwdKnobs
    # travels inside PlanConfig; callers see {cudnn.knob_type: int}.
    def knobs_to_public(self, knobs) -> dict:
        from .engines import SdpaFwdKnobs

        if knobs is None:
            return {}
        if isinstance(knobs, dict):
            return dict(knobs)
        if isinstance(knobs, SdpaFwdKnobs):
            return knobs.to_public()
        return super().knobs_to_public(knobs)

    def knobs_from_public(self, public: dict):
        from .engines import SdpaFwdKnobs

        return SdpaFwdKnobs.from_public(public) if public else None

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from .engines import build

        knobs = plan.knobs if plan is not None else None
        if isinstance(knobs, dict):  # a replayed public record, not yet converted
            knobs = self.knobs_from_public(knobs)
        try:
            return _FrostSdpaFwdPlan(self.name, build(self._spec, graph, knobs))
        except (NotImplementedError, ValueError, ImportError) as exc:
            # ImportError: the DSL adapter resolves at build time now (support
            # checks must not pay for it), so a missing cutedsl extra surfaces
            # HERE rather than making the family vanish at import. It is a
            # decline -- the walk moves on and the backend serves the graph.
            raise NotImplementedError(f"{self.name}: {exc}") from exc


def FrostSdpaFwdEngines(ids) -> List[FrostSdpaFwdEngine]:
    """The SDPA-forward engines the manifest asked for, in ENGINE_SPECS order.

    ``ids`` is ``{name: engine_id}`` from engines/manifest.py — the single
    source of engine ids. A spec absent from it is one the manifest is not
    offering (still opt-in gated), so it is simply not built; a spec that has
    no slot AT ALL is caught by test_dispatch, not at runtime.
    """
    from .engines import ENGINE_SPECS

    engines = []
    for spec in ENGINE_SPECS:
        if spec.name in ids:
            engines.append(FrostSdpaFwdEngine(spec, ids[spec.name]))
    return engines
