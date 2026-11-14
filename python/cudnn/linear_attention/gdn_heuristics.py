# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN family heuristics: rank the offered engines for one graph.

Same job and same limits as ``kda_heuristics``: ORDERING ONLY, and only on
Hopper. Slot order is identity, not preference -- ``gdn_cutile`` holds slot 1
because it shipped first, so without this hook it wins every sm90 GDN graph
even where the dedicated sm90 engine at slot 4 serves the graph and is faster.

Eligibility stays with ``check_support``: ``engines/heuristics.rank`` only
offers engines that accept the graph, and ``build_plans()`` walks past a
decline, so ranking an engine first can cost a probe but never a wrong answer.

Outside Hopper the order is exactly the slot order this replaced, so no
non-Hopper graph changes engine.
"""

from __future__ import annotations

from typing import Dict, List

from cudnn.engines.base import PlanConfig

from .kda_heuristics import HOPPER_SM, _current_sm

# Hopper preference, fastest first. gdn_frost never accepts an sm90 graph, so the
# only real choice here is between the dedicated engine and gdn_cutile.
_HOPPER_FIRST = ("gdn_hopper_cuda",)


def _ordered_names(offered: Dict[str, int]) -> List[str]:
    """Offered engine names, best first."""
    names = list(offered)
    if _current_sm() != HOPPER_SM:
        return names
    promoted = [n for n in _HOPPER_FIRST if n in offered]
    return promoted + [n for n in names if n not in promoted]


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """One default plan per offered engine, best first (``kind`` does not change the answer)."""
    return [PlanConfig(offered[name], None) for name in _ordered_names(offered)]
