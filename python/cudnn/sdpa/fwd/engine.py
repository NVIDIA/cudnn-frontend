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
from cudnn.engines.engine_ids import FROST_SDPA_FWD_ID_BASE

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

    def get_workspace_size(self) -> int:
        return int(getattr(self._compiled, "workspace_bytes", 0) or 0)

    def execute(self, graph: "pygraph", uid_to_data, ctx: ExecutionContext) -> None:
        # Keyed by IR tensor object: that is the binding's own identity, and the
        # only key resolve_variant_pack() accepts for an auto-assigned uid.
        pack = {}
        missing = []
        for t in self._tensors:
            buf = uid_to_data.get(t.get_uid())
            if buf is None:
                missing.append(t.get_name() or t.get_uid())
            else:
                pack[t] = buf
        if missing:
            raise ValueError(f"{self._name}: the variant pack is missing buffers for {missing}")
        required = self.get_workspace_size()
        if required:
            _check_workspace(ctx.workspace, required, self._name)
            self._compiled(pack, ctx.workspace, stream=ctx.stream)
        else:
            self._compiled(pack, stream=ctx.stream)


class FrostSdpaFwdEngine(BaseEngine):
    """One SDPA-forward capability cell (arch x phase x geometry x quantization).

    Wraps a single :class:`~cudnn.sdpa.fwd.engines.EngineSpec`: ``name`` is the
    spec's shipped name and ``engine_id`` is its fixed offset in the family's id
    block (see :func:`FrostSdpaFwdEngines`).
    """

    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def __init__(self, spec: "EngineSpec", offset: int):
        super().__init__()
        self._spec = spec
        self.name = spec.name
        self.engine_id = FROST_SDPA_FWD_ID_BASE + offset

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

    def propose_plans(self, graph: "pygraph") -> List[PlanConfig]:
        # One plan, no knobs: nothing proposes a tuning request today, so the
        # engines run at their capability row's default tile/schedule. A knob
        # search (SdpaFwdKnobs over Capabilities.tile_ms/tile_ns/cgas) becomes
        # several entries here; each one's knobs reach build_plan verbatim.
        self.check_support(graph)
        return [PlanConfig(self.engine_id, self.default_knobs)]

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from .engines import build

        knobs = plan.knobs if plan is not None else None
        try:
            return _FrostSdpaFwdPlan(self.name, build(self._spec, graph, knobs))
        except (NotImplementedError, ValueError) as exc:
            raise NotImplementedError(f"{self.name}: {exc}") from exc


# engine_id = FROST_SDPA_FWD_ID_BASE + offset. An offset is FIXED FOREVER: an
# autotune result is (engine_id, knobs) and must replay across versions.
# Appending a spec takes the next free offset; offsets are never reordered or
# reused. Keyed by the spec's shipped name rather than by its position in
# ENGINE_SPECS, because that position is the PREFERENCE order and may change.
_ID_OFFSETS = {
    "sdpa_fwd_prefill_sm100_d128": 0,
    "sdpa_fwd_prefill_sm100_d256": 1,
    "sdpa_fwd_prefill_sm100_d512": 2,
    "sdpa_fwd_prefill_sm100_d128_mxfp8": 3,
    "sdpa_fwd_prefill_sm100_d128_fp8": 4,
    "sdpa_fwd_prefill_sm120": 5,
}


def FrostSdpaFwdEngines() -> List[FrostSdpaFwdEngine]:
    """The SDPA-forward engine family, in ENGINE_SPECS (= preference) order."""
    from .engines import ENGINE_SPECS

    engines = []
    for spec in ENGINE_SPECS:
        if spec.name not in _ID_OFFSETS:
            raise KeyError(f"engine spec {spec.name!r} has no engine-id offset; allocate the next free one in engine._ID_OFFSETS (never reuse)")
        engines.append(FrostSdpaFwdEngine(spec, _ID_OFFSETS[spec.name]))
    return engines
