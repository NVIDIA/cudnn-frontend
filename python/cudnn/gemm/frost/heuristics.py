# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family heuristics for ``frost_gemm``: the facts callable and the proposal
callable the engine manifest names (``engines/manifest.py``).

Same shape as the SDPA family (``cudnn.sdpa.fwd.heuristics``): the analyzer
turns the graph into facts once, ``recommend`` proposes complete
``PlanConfig(engine_id, GemmKnobs)`` entries from those facts alone, and every
listed plan carries the exact tile config the engine will build -- so a recorded
``(engine_id, knobs)`` replays the same kernel after the automatic pick changes.

The ordinary automatic strategy remains first. An explicit common-parent
BF16 SwiGLU graph can also propose the paired SM100 specialization. These are
candidates for measurement; their order does not claim a universal ranking.
"""

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
    pair: Any = None  # explicit common-parent SwiGLU facts, when supported
    fc2: Any = None  # direct small-row BF16 projection, when supported


def analyze_facts(graph) -> Optional[GemmFacts]:
    """The manifest's ``analyzer`` hook: facts for a GEMM graph, or None when
    the FROST analyzer cannot express it (the backend then serves it alone)."""
    from .compiler import _graph_dynamic_shapes
    from .graph_analyzer import _graph_has_gemm, analyze

    if not _graph_has_gemm(graph):
        return None
    try:
        chain = analyze(graph)
    except (NotImplementedError, ValueError, KeyError) as exc:
        _LOG.debug("frost_gemm analyzer declined the graph: %s", exc)
        return None
    dynamic = _graph_dynamic_shapes(graph)
    from .moe_pair import analyze_pair

    try:
        pair = analyze_pair(graph, dynamic_shapes=dynamic)
    except (NotImplementedError, ValueError, KeyError):
        pair = None
    from .moe_fc2_pair import analyze_fc2

    try:
        fc2 = analyze_fc2(graph, dynamic_shapes=dynamic)
    except (NotImplementedError, ValueError, KeyError):
        fc2 = None
    return GemmFacts(chain=chain, dynamic_shapes=dynamic, pair=pair, fc2=fc2)


def recommend(kind: str, facts: GemmFacts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Ordered candidate plans for ``facts`` (``kind`` is ``"A"`` or ``"FALLBACK"``).

    Each candidate uses its engine's support gate and shared public knobs.
    Keep the existing automatic strategy first; add the paired specialization
    only for graphs declaring its supported parent relationship.
    """
    from .compiler import plan_config, probe_chain, probe_cutedsl

    proposals = []
    engine_id = offered.get(ENGINE)
    if engine_id is not None:
        try:
            probe_cutedsl()
            config = plan_config(facts.chain, dynamic_shapes=facts.dynamic_shapes)
            probe_chain(facts.chain, config)
            proposals.append(PlanConfig(engine_id, GemmKnobs.from_config(config)))
        except (NotImplementedError, ValueError, KeyError) as exc:
            _LOG.debug("frost_gemm proposes nothing (%s): %s", kind, exc)
    paired_id = offered.get("frost_moe_swiglu_pair")
    if paired_id is not None and facts.pair is not None:
        from .moe_pair import device_params, pair_knobs

        try:
            device_params()
            proposals.append(PlanConfig(paired_id, pair_knobs()))
        except (NotImplementedError, ValueError, KeyError) as exc:
            _LOG.debug("paired MoE proposes nothing (%s): %s", kind, exc)
    fc2_id = offered.get("frost_moe_fc2_pair")
    if fc2_id is not None and facts.fc2 is not None:
        from .moe_pair import device_params
        from .moe_fc2_pair import Fc2Knobs

        try:
            device_params()
            # Keep the historical default first; tuning may select either depth.
            proposals.extend(PlanConfig(fc2_id, Fc2Knobs(ab_stages=stages)) for stages in (12, 6))
        except (NotImplementedError, ValueError, KeyError) as exc:
            _LOG.debug("paired FC2 proposes nothing (%s): %s", kind, exc)
    return proposals
