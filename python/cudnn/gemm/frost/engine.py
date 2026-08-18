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
        self._operand_indices = None
        # The one launch path a dense or block-scale kernel has: the closure the
        # recipe is captured into. A graph it cannot serve was declined at
        # check_support, so there is nothing to fall back TO -- None here means
        # MoE, which takes the variant-pack dict and its own workspace below.
        self._lowered = getattr(compiled, "lowered", None)
        self._launch = self._lowered

    def get_workspace_size(self) -> int:
        return int(getattr(self._compiled, "workspace_bytes", 0) or 0)

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        indices = self._operand_indices
        if indices is None:
            try:
                indices = self._operand_indices = [variant_pack.index_of(t.get_uid()) for t in self._tensors]
            except KeyError as exc:
                raise ValueError(f"frost_gemm: tensor uid {exc} is bound by the kernel but is not an operand of this graph") from exc
        # The kernel reads its M/N/K off these, so they must be the pack's --
        # which carry the shape this execute runs, override_shapes included.
        operands = variant_pack.operands(indices)
        required = self.get_workspace_size()
        if required:
            # Scratch is carved from the CALLER's workspace: stable pointers, so
            # a plan stays safe to capture in a CUDA graph. Only the MoE
            # launchers need it, and they take the variant-pack dict.
            self._compiled(dict(zip(self._tensors, operands)), Workspace.over(variant_pack, required, "frost_gemm"), stream=ctx.stream)
            return
        launch = self._launch
        if launch is not None:
            # Which bound tensor holds which operand was settled at build, so
            # the buffers arrive in that order and the launcher indexes them.
            # Which AXIS ORDER each one arrived in is a per-call fact only the
            # pack knows, since a bare address wears the graph's layout. None
            # means every operand here is the caller's own.
            graph_order = None
            borrowed = variant_pack.graph_described
            if borrowed:
                flags = tuple(i in borrowed for i in indices)
                graph_order = flags if any(flags) else None
            launch(operands, graph_order, stream=ctx.stream)
        else:
            self._compiled(dict(zip(self._tensors, operands)), stream=ctx.stream)


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
        from cudnn.frost.device import build_device

        # Bake the plan for the device of the handle the graph carries (via ctx),
        # not whatever CUDA device is current at build time. A foreign raw-int
        # handle (or none) carries no device -> None -> classic current-device.
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        try:
            with build_device(device):
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
