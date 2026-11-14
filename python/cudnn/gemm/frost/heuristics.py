# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family heuristics for ``frost_gemm``: the facts callable and the proposal callable the engine manifest names (``engines/manifest.py``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cudnn.engines.base import PlanConfig

from .knobs import GemmKnobs

_LOG = logging.getLogger("cudnn.gemm.frost.heuristics")

ENGINE = "frost_gemm"


@dataclass(frozen=True)
class GemmFacts:
    """What the ranking and the engine share about one GEMM graph."""

    chain: Any  # fusion_ir.FusionChain
    dynamic_shapes: bool  # the graph declared dynamic shapes (split-K is skipped)
    shapes_fixed: bool = False  # neither dynamic nor override-shape enabled
    alternatives: tuple = ()


def analyze_facts(graph) -> Optional[GemmFacts]:
    """The manifest's ``analyzer`` hook: facts for a GEMM graph, or None when
    the FROST analyzer cannot express it (the backend then serves it alone)."""
    from .compiler import _graph_dynamic_shapes, _graph_shapes_fixed
    from .graph_analyzer import _graph_has_gemm, analyze

    has_gather = any(node.node_type.name == "GATHER" for node in graph.nodes)
    if not _graph_has_gemm(graph) and not has_gather:
        return None
    try:
        if has_gather:
            from .gather_execution import available_routes

            chains = tuple(route.chain for route in available_routes(graph))
        else:
            chains = (analyze(graph),)
    except (NotImplementedError, ValueError, KeyError) as exc:
        _LOG.debug("frost_gemm analyzer declined the graph: %s", exc)
        return None
    return GemmFacts(chain=chains[0], alternatives=chains[1:], dynamic_shapes=_graph_dynamic_shapes(graph), shapes_fixed=_graph_shapes_fixed(graph))


def recommend(kind: str, facts: GemmFacts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Ordered, supported plans carrying exact public-replay knobs."""
    engine_id = offered.get(ENGINE)
    if engine_id is None:
        return []
    from .compiler import plan_configs, probe_chain, probe_cutedsl
    from .planning import MAX_PLAN_CONFIGS

    try:
        probe_cutedsl()
    except (NotImplementedError, ValueError, KeyError) as exc:
        _LOG.debug("frost_gemm proposes nothing (%s): %s", kind, exc)
        return []
    plans = {}
    for chain in (facts.chain, *facts.alternatives):
        if chain is None:
            plans.setdefault(None, PlanConfig(engine_id))
            continue
        try:
            configs = plan_configs(chain, dynamic_shapes=facts.dynamic_shapes, shapes_fixed=facts.shapes_fixed)
        except (NotImplementedError, ValueError, KeyError) as exc:
            _LOG.debug("frost_gemm skips route (%s): %s", kind, exc)
            continue
        for config in configs:
            try:
                probe_chain(chain, config)
                knobs = GemmKnobs.from_config(config)
            except (NotImplementedError, ValueError, KeyError) as exc:
                _LOG.debug("frost_gemm skips candidate %s (%s): %s", config.name, kind, exc)
                continue
            plans.setdefault(knobs, PlanConfig(engine_id, knobs))
            if len(plans) == MAX_PLAN_CONFIGS:
                return list(plans.values())
    return list(plans.values())
