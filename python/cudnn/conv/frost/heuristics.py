# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep newly covered C96 convolutions behind the backend's ranked plans."""

from cudnn.engines.base import PlanConfig
from cudnn.engines.heuristics import BACKEND

from .graph_analyzer import ConvGraph, analyze


def analyze_facts(graph):
    """Return supported graph structure, or leave an unmatched graph to cuDNN."""
    try:
        return analyze(graph)
    except NotImplementedError:
        return None


def propose(kind: str, facts: ConvGraph, offered: dict[str, int]) -> list[PlanConfig]:
    engine_id = offered.get("frost_conv")
    if engine_id is None:
        return []
    ours = [PlanConfig(engine_id)]
    # C96 extends coverage; it has no demonstrated complete-operation speed
    # advantage. Explicit selection and OPENSOURCE still expose this plan.
    # Preserve the existing preference for all previously supported channels.
    return [BACKEND] + ours if facts.image.dim[1] == 96 else ours
