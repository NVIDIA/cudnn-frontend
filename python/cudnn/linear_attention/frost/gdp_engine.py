# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDP engine: GDP / GDP_BWD nodes on the ``num_householder`` sub-token expansion (gate on sub-token 0, readout on
sub-token ``n - 1``).  ``gdn_engine.py`` builds the plans on the shared GDN kernels, routing the d_v = 64 backward (n > 1) to
the token-domain fork ``kernel/gdp_bprop_v64_f16.py``; the bprop summary reads compact q / dO at every d_v."""

from __future__ import annotations

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate, summary_support_gates
from .gdn_engine import build_gdn, build_gdn_summary, gdn_support_gates
from .kernel.gdp_bprop_v64_config import MAX_EXPAND_NUM


class GdpFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDP graphs (THD layout)."""

    name = "gdp_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("GdpFrostEngine", facts, "GDP")
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"GdpFrostEngine: q/k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"GdpFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        if facts.is_bwd and facts.d_v == 64 and facts.num_householder > MAX_EXPAND_NUM[facts.d_qk]:
            raise NotImplementedError(
                f"GdpFrostEngine: the d_v = 64 backward keeps 2 * num_householder gate stages in shared memory: num_householder must be "
                f"<= {MAX_EXPAND_NUM[facts.d_qk]} at q/k head dim {facts.d_qk}, got {facts.num_householder}"
            )
        gdn_support_gates("GdpFrostEngine", facts)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return FrostLaPlan(build_gdn(graph))


class GdpSummaryFrostEngine(BaseEngine):
    """FROST summary backend for single-node GDP_SUMMARY and
    GDP_SUMMARY_BWD graphs on the ``num_householder``-expanded timeline."""

    name = "gdp_summary_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("GdpSummaryFrostEngine", facts, "GDP_SUMMARY")
        if facts.d_qk not in (64, 128):
            raise NotImplementedError(f"GdpSummaryFrostEngine: k head dim must be 64 or 128, got {facts.d_qk}")
        if facts.d_v not in (64, 128):
            raise NotImplementedError(f"GdpSummaryFrostEngine: v head dim must be 64 or 128, got {facts.d_v}")
        checkpoint = facts.checkpoint_every_n_tokens
        if checkpoint and checkpoint % 64 != 0:
            raise NotImplementedError(f"GdpSummaryFrostEngine: checkpoint_every_n_tokens must be a positive multiple of 64 (got {checkpoint})")
        summary_support_gates("GdpSummaryFrostEngine", facts, graph)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return FrostLaPlan(build_gdn_summary(graph))
