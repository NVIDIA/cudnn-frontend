# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BaseEngine wrapper for the sm_100 norm-forward engines.

The manifest (``cudnn/engines/manifest.py``) names :func:`FrostNormFwdEngines`; this
module is the thin adapter between that framework and the capability/lowering
logic in :mod:`cudnn.norm.fprop.engines`. The split mirrors
``cudnn.sdpa.fprop.engine` / `.engines``: specs and capabilities there,
plan/engine protocol here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from cudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig

if TYPE_CHECKING:
    from cudnn.norm.fprop.engines import EngineSpec


class _FrostNormPlan(CompiledPlan):
    """Wraps the closure returned by the lowering: it takes the variant pack and
    performs the launch, so there is no workspace and nothing to stage."""

    def __init__(self, name: str, run):
        self._name = name
        self._run = run

    def get_workspace_size(self) -> int:
        return 0

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        self._run(variant_pack)


class FrostNormFwdEngine(BaseEngine):
    """One shipped norm-forward engine spec, bound to its manifest id."""

    def __init__(self, spec: "EngineSpec", engine_id: int):
        super().__init__()
        self._spec = spec
        self.name = spec.name
        self.engine_id = engine_id

    def _decline_reason(self, graph, knobs) -> Optional[str]:
        from cudnn.norm.fprop.engines import analyze_for, mismatch

        facts = analyze_for(graph)
        if facts is None:
            return "graph is not a single norm node this engine serves"
        return mismatch(self._spec.capabilities, facts, knobs)

    def check_support(self, graph) -> None:
        reason = self._decline_reason(graph, None)
        if reason is not None:
            raise NotImplementedError(f"{self.name}: {reason}")

    def build_plan(self, graph, plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from cudnn.norm.fprop.engines import build

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


def FrostNormFwdEngines(ids) -> List[FrostNormFwdEngine]:
    """The norm-forward engines the manifest asked for, in ENGINE_SPECS order.

    ``ids`` is ``{name: engine_id}`` from engines/manifest.py -- the single
    source of engine ids. A spec absent from it is one the manifest is not
    offering, so it is simply not built.
    """
    from .engines import ENGINE_SPECS

    return [FrostNormFwdEngine(spec, ids[spec.name]) for spec in ENGINE_SPECS if spec.name in ids]
