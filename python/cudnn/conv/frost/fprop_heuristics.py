# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank the FROST forward-convolution family, keeping C96 behind backend plans."""

from cudnn import data_type
from cudnn.engines.base import PlanConfig
from cudnn.engines.heuristics import BACKEND

from .graph_analyzer import FpropGraph, analyze_fprop


def analyze_facts(graph):
    """Return supported graph structure, or leave an unmatched graph to cuDNN."""
    try:
        return analyze_fprop(graph)
    except NotImplementedError:
        return None


def propose(kind: str, facts: FpropGraph, offered: dict[str, int]) -> list[PlanConfig]:
    """Keep BF16 C96 backend-first while preserving existing dtype and channel preferences."""
    engine_id = offered.get("frost_conv_fprop")
    if engine_id is None:
        return []
    ours = [PlanConfig(engine_id)]
    # C96 extends coverage; it has no demonstrated complete-operation speed
    # advantage. Explicit selection and OPENSOURCE still expose this plan.
    # Preserve the existing preference for all previously supported channels.
    new_c96 = facts.image.dim[1] == 96 and facts.image.data_type == facts.output.data_type == data_type.BFLOAT16
    return [BACKEND] + ours if new_c96 else ours
