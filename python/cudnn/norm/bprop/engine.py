# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseEngine wrapper for the sm_100 norm-backward engines.

The manifest (``cudnn/engines/manifest.py``) names :func:`FrostNormBwdEngines`; this
module is the thin adapter between that framework and the capability/lowering
logic in :mod:`cudnn.norm.bprop.engines`. The split mirrors
``cudnn.sdpa.bprop.engine` / `.engines``: specs and capabilities there,
plan/engine protocol here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig

if TYPE_CHECKING:
    from cudnn.norm.bprop.engines import EngineSpec


class _FrostNormPlan(CompiledPlan):
    """Wraps the closure returned by the lowering: it takes the variant pack and
    performs the launch, so there is no workspace and nothing to stage."""

    def __init__(self, name: str, run):
        self._name = name
        self._run = run
        # The lowering sizes its scratch at build time; 0 means the kernels this
        # graph routes to carve nothing, which is most of them.
        self._ws_bytes = int(getattr(run, "workspace_bytes", 0) or 0)

    def get_workspace_size(self) -> int:
        return self._ws_bytes

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        from cudnn.norm.utils import Workspace, on_stream

        # Both the kernel launches and the output copies follow torch's current
        # stream, so the handle's stream has to be made current for the whole
        # call -- see on_stream().
        ws = Workspace(getattr(ctx, "workspace", None)) if self._ws_bytes else None
        with on_stream(getattr(ctx, "stream", None)):
            self._run(variant_pack, ws)


class FrostNormBwdEngine(BaseEngine):
    """One shipped norm-backward engine spec, bound to its manifest id."""

    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def __init__(self, spec: "EngineSpec", engine_id: int):
        super().__init__()
        self._spec = spec
        self.name = spec.name
        self.engine_id = engine_id

    def _decline_reason(self, graph, knobs) -> Optional[str]:
        from cudnn.norm.bprop.engines import analyze_for, mismatch

        from cudnn.frost.buffers import cutedsl_primitives_error, cutedsl_requirement_error

        # Rule 7: decline at CHECK time, not when lowering imports the kernels.
        error = cutedsl_requirement_error(self.name)
        if error is not None:
            return error
        facts = analyze_for(graph)
        if facts is None:
            return "graph is not a single norm node this engine serves"
        if facts.scale_t is None:
            # layernorm_backward / rmsnorm_backward take an OPTIONAL scale upstream
            # (#188); without it the node has no DScale or DBias either. These
            # kernels read gamma per channel rather than assuming ones, and
            # synthesising a gamma here would allocate inside execute (Rule 1), so
            # the graph goes to an engine that serves it natively.
            return "backward without a scale input is not served"
        if not facts.uniform_dtype:
            # These kernels read scale/bias/DY at the I/O dtype; a mixed-precision
            # graph would be read wrong rather than merely be slow.
            return "scale/bias/DY must share the input dtype; mixed-precision parameters are not supported"

        reason = mismatch(self._spec.capabilities, facts, knobs)
        if reason is not None:
            return reason
        # Last, because it imports the DSL (~1 s): only reached when the engine
        # would otherwise accept, where lowering is about to import it anyway.
        return cutedsl_primitives_error(self.name)

    def check_support(self, graph) -> None:
        reason = self._decline_reason(graph, None)
        if reason is not None:
            raise NotImplementedError(f"{self.name}: {reason}")

    def build_plan(self, graph, plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from cudnn.norm.bprop.engines import build

        knobs = getattr(plan, "knobs", None)
        reason = self._decline_reason(graph, knobs)
        if reason is not None:
            raise NotImplementedError(f"{self.name}: {reason}")
        try:
            return _FrostNormPlan(self.name, build(self._spec, graph, knobs))
        except NotImplementedError:
            raise
        except Exception as exc:
            raise NotImplementedError(f"{self.name}: {exc}") from exc


def FrostNormBwdEngines(ids) -> List[FrostNormBwdEngine]:
    """The norm-backward engines the manifest asked for, in ENGINE_SPECS order.

    ``ids`` is ``{name: engine_id}`` from engines/manifest.py -- the single
    source of engine ids. A spec absent from it is one the manifest is not
    offering, so it is simply not built.
    """
    from .engines import ENGINE_SPECS

    return [FrostNormBwdEngine(spec, ids[spec.name]) for spec in ENGINE_SPECS if spec.name in ids]
