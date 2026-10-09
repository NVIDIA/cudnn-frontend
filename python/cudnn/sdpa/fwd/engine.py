# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST SDPA-forward engines: one BaseEngine per capability cell.

Listed in ``cudnn/engines/manifest.py`` as ONE row owning the
``FROST_SDPA_FWD_ID_BASE`` block, so ``FrostSdpaFwdEngines()`` returns the whole
family and a graph containing an sdpa() node reaches them through the ordinary
lifecycle — no registration call. The SM100, SM107, SM120 and SM90 f16/bf16 slots, the
SM100 per-tensor FP8 slot and the SM107 MXFP8 slot are default candidates, ranked against
the backend per measured shard or qualification verdict (``placement.py``); the other
slots stay opt-in (``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``) until they have the arch
coverage and the qualification to serve graphs unasked.

The capability table, the probe and the lowering are unchanged and stay in
``engines.py`` (``ENGINE_SPECS`` / ``analyze_for`` / ``build``); this file is
only the engine contract around them.
"""

from typing import TYPE_CHECKING, List, Optional

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from cudnn.sdpa._plan import _FrostSdpaPlan, _check_workspace

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph

    from .engines import EngineSpec


class _FrostSdpaFwdPlan(_FrostSdpaPlan):
    def _execute_tensor(self, uid_to_data, ctx):
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
