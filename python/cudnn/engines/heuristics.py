# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend heuristics: rank python and backend plans into ONE list.

``create_execution_plans()`` collects both sides and hands them here. The
returned order IS the order the build walk tries.

PLACEHOLDER: claiming engines first, then the backend. Real ranking — per
operation, over a cost model that can compare a CuTe tile config against a cuDNN
engine — replaces the body of :func:`heuristics_sort` and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from .base import PlanConfig

if TYPE_CHECKING:
    from .._pygraph import pygraph


def heuristics_sort(graph: "pygraph", python_plans: List[PlanConfig], backend_plans: List[PlanConfig]) -> List[PlanConfig]:
    """Rank the two sides into one list. Either side may be empty.

    Which python engines are in the list at all is the manifest's decision
    (``EngineFamily.offered``, where ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` is
    read); a plan that reaches here has already been admitted.

    Placeholder ranking, deliberately. The facts plumbing this reads FROM is
    already in place and unused: ``graph._facts_for(analyzer)`` holds the record
    the family's analyzer produced (planning attached it before this ran), and
    ``manifest.family_for(graph)`` names the family whose vocabulary it is
    in. A real policy has two layers — order the family's own engines using
    those facts, then merge that against the backend's entries on a common
    currency (predicted time), which is the only comparison that has to work
    across families. Neither is written yet; the seam is here so that writing
    them does not mean re-plumbing the graph.
    """
    return python_plans + backend_plans
