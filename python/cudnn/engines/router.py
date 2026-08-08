# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router: the plumbing that feeds the heuristics at create_execution_plans() time.

    Python Graph API -> create_execution_plans() -> Router -> heuristics -> graph.plans

Deliberately dumb. It gathers the inputs — the graph's python engine candidates
and the backend's own entries — calls ``heuristics.rank``, and checks that what
came back names engines this graph can dispatch to. It makes no ordering
decision of its own: policy lives in ``engines/heuristics.py`` and the family
hook it dispatches to, so there is exactly one place to change how plans are
ranked.

The list is flat ``PlanConfig(engine_id, knobs)`` entries mixing python engines
and backend engines in any order. Position is rank; ``engine_id`` is identity
and the key the build walk dispatches on — the two are independent, so an
engine from anywhere can sit anywhere in the list.

Candidates come from ``engines.manifest`` (the library's own static table) plus
anything the caller added with ``register_backend()``. The backend's own entries
come from ``graph.backend_plan_entries()``, which is [] when the backend
declined the graph or is not installed.
"""

import logging
from typing import TYPE_CHECKING, List

from . import heuristics
from .base import BaseEngine, PlanConfig

if TYPE_CHECKING:
    from .._pygraph import pygraph

_LOG = logging.getLogger("cudnn.engines.router")


def decline_types():
    """The exception types that mean "this engine does not serve this graph".

    ImportError counts: an engine whose optional dependency is absent cannot
    serve the graph, and since lowering imports are deferred past check_support
    that only becomes visible at build time.
    """
    import cudnn

    return (NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError)


def plan(graph: "pygraph", engines: List[BaseEngine]) -> List[PlanConfig]:
    """The ranked plan list for ``graph``.

    Through the module, not a bound name: ``heuristics.rank`` is the seam a
    ranking policy replaces, and it must be swappable in a live process (tests,
    experiments) without touching this file.
    """
    return heuristics.rank(graph, engines, graph.backend_plan_entries(), graph._backend_heuristics)
