# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the initial FROST convolution weight-gradient family behind backend plans."""

from cudnn.engines.base import PlanConfig
from cudnn.engines.heuristics import BACKEND

from .graph_analyzer import WgradGraph, analyze_wgrad


def analyze_facts(graph):
    """Return supported weight-gradient structure, or leave the graph to cuDNN."""
    try:
        return analyze_wgrad(graph)
    except NotImplementedError:
        return None


def propose(kind: str, facts: WgradGraph, offered: dict[str, int]) -> list[PlanConfig]:
    """Rank the weight-gradient plan after the backend's plans."""
    engine_id = offered.get("frost_conv_wgrad")
    return [BACKEND, PlanConfig(engine_id)] if engine_id is not None else []
