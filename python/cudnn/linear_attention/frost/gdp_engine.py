# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDP engine: GDP / GDP_BWD nodes on the ``num_householder`` sub-token expansion (gate on sub-token 0, readout on
sub-token ``n - 1``).  ``gdn_engine.py`` builds the plans on the shared GDN kernels, routing the d_v = 64 backward (n > 1) to
the token-domain fork ``kernel/gdp_bprop_v64_f16.py``; the bprop summary reads compact q / dO at every d_v."""

from __future__ import annotations

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device
from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate, summary_support_gates
from .gdn_engine import build_gdn, build_gdn_summary, gdn_support_gates


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
        gdn_support_gates("GdpFrostEngine", facts)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
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
        summary_support_gates("GdpSummaryFrostEngine", facts, graph)

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_gdn_summary(graph))
