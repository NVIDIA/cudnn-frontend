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

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph


class _FrostGemmPlan(CompiledPlan):
    """A compiled fused-GEMM kernel plus the graph binding it was compiled for."""

    def __init__(self, compiled):
        self._compiled = compiled
        # Keyed by tensor OBJECT, not uid: one tensor can occupy two operand
        # roles (matmul(A, A)), and resolve_variant_pack treats a repeated uid
        # as ambiguous.
        self._tensors = list(compiled.binding.bound_tensors())

    def get_workspace_size(self) -> int:
        return int(getattr(self._compiled, "workspace_bytes", 0) or 0)

    def execute(self, graph, uid_to_data, ctx: ExecutionContext) -> None:
        pack, missing = {}, []
        for t in self._tensors:
            buf = uid_to_data.get(t.get_uid())
            if buf is None:
                missing.append(t.get_name() or t.get_uid())
            else:
                pack[t] = buf
        if missing:
            raise ValueError(f"frost_gemm: the variant pack is missing buffers for {missing}")
        required = self.get_workspace_size()
        if required:
            _check_workspace(ctx.workspace, required)
            self._compiled(pack, ctx.workspace, stream=ctx.stream)
        else:
            self._compiled(pack, stream=ctx.stream)


def _check_workspace(workspace, required: int) -> None:
    """A FROST executor carves its scratch out of the CALLER's workspace: no
    hidden per-execute allocation, stable pointers, CUDA-graph friendly."""
    if workspace is None:
        raise ValueError(f"frost_gemm needs a {required}-byte workspace; execute() got none — allocate graph.get_workspace_size() bytes and pass it")
    available = workspace.numel() * workspace.element_size() if hasattr(workspace, "numel") else len(workspace)
    if available < required:
        raise ValueError(f"frost_gemm needs a {required}-byte workspace; the buffer provides {available}")


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
