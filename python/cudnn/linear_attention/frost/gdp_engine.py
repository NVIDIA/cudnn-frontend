# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST GDP engine: GDP/GDP_BWD nodes on the unmodified GDN kernels via the
``num_householder`` sub-token expansion (gate on sub-token 0, readout on
sub-token ``n - 1``); compiled plans shared with ``gdn_engine.py``."""

from __future__ import annotations

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan

from cudnn.frost.device import build_device
from ..graph_analyzer import analyze
from .engine import FrostLaPlan, frost_la_gate
from .gdn_engine import build_gdn, gdn_support_gates


class GdpFrostEngine(BaseEngine):
    """FROST chunked-kernel backend for single-node GDP graphs (THD layout)."""

    name = "gdp_frost"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph) -> None:
        facts = graph._facts_for(analyze)
        frost_la_gate("GdpFrostEngine", facts, "GDP", d_v_allowed=(64, 128))
        gdn_support_gates("GdpFrostEngine", facts)
        if facts.safe_gate:
            raise NotImplementedError("GdpFrostEngine: safe_gate has no exact neutral fill on the expanded sub-token rows")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        with build_device(device):
            return FrostLaPlan(build_gdn(graph))
