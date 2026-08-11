# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST GEMM engine: JIT-compiled fused matmul chains for cuDNN graphs.

Listed in ``cudnn/engines/manifest.py``, so a graph containing a matmul reaches
it through the ordinary lifecycle — no registration call. The row is opt-in
(``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``) until this engine has the arch
coverage to serve graphs unasked. The analysis and codegen are unchanged (``graph_analyzer`` /
``compiler``); this file is only the engine contract around them.
"""

from typing import TYPE_CHECKING, List

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from cudnn.frost.workspace import Workspace

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph


class _FrostGemmPlan(CompiledPlan):
    """A compiled fused-GEMM kernel plus the graph binding it was compiled for."""

    takes_variant_pack = True

    def __init__(self, compiled):
        self._compiled = compiled
        # Keyed by tensor OBJECT, not uid: one tensor can occupy two operand
        # roles (matmul(A, A)), and resolve_variant_pack treats a repeated uid
        # as ambiguous.
        self._tensors = list(compiled.binding.bound_tensors())
        self._slots = None

    def get_workspace_size(self) -> int:
        return int(getattr(self._compiled, "workspace_bytes", 0) or 0)

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        slots = self._slots
        if slots is None:
            try:
                slots = self._slots = [variant_pack.slot(t.get_uid()) for t in self._tensors]
            except KeyError as exc:
                raise ValueError(f"frost_gemm: tensor uid {exc} is bound by the kernel but is not an operand of this graph") from exc
        # The kernel reads its M/N/K off these, so they must be the pack's --
        # which carry the shape this execute runs, override_shapes included.
        views = variant_pack.views(slots)
        required = self.get_workspace_size()
        # Scratch is carved from the CALLER's workspace: stable pointers, so a
        # plan stays safe to capture in a CUDA graph.
        extra = (Workspace.over(variant_pack, required, "frost_gemm"),) if required else ()
        run_resolved = getattr(self._compiled, "run_resolved", None)
        if run_resolved is not None:
            # Which bound tensor holds which operand was settled at build, so
            # resolve_variant_pack's by-object / by-uid / by-name tables have
            # no question left to answer.
            run_resolved({id(t): v for t, v in zip(self._tensors, views)}, *extra, stream=ctx.stream)
        else:
            self._compiled(dict(zip(self._tensors, views)), *extra, stream=ctx.stream)


class FrostGemmEngine(BaseEngine):
    """Fused matmul chains (dense, block-scaled, MoE grouped) on SM100/SM103."""

    name = "frost_gemm"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph: "pygraph") -> None:
        from .compiler import probe_supported
        from .graph_analyzer import _graph_has_gemm

        if not _graph_has_gemm(graph):
            raise NotImplementedError("frost_gemm: the graph has no matmul / moe_grouped_matmul node")
        try:
            probe_supported(graph)
        except (NotImplementedError, ValueError) as exc:
            # ValueError is the analyzer's internal "cannot express this graph";
            # at the engine boundary that is a decline, not a user error.
            raise NotImplementedError(f"frost_gemm: {exc}") from exc

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from .graph_analyzer import build_gemm_plan

        try:
            return _FrostGemmPlan(build_gemm_plan(graph))
        except (NotImplementedError, ValueError) as exc:
            raise NotImplementedError(f"frost_gemm: {exc}") from exc


def FrostGemmEngines(ids):
    """The gemm engines the manifest asked for, with the ids it assigned.

    ``ids`` is ``{name: engine_id}`` from engines/manifest.py, the single source
    of engine ids -- an engine does not carry one of its own.
    """
    out = []
    for cls in (FrostGemmEngine,):
        if cls.name in ids:
            engine = cls()
            engine.engine_id = ids[cls.name]
            out.append(engine)
    return out
