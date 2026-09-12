# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family heuristics for ``frost_gemm``: the facts callable and the proposal
callable the engine manifest names (``engines/manifest.py``).

Same shape as the SDPA family (``cudnn.sdpa.fwd.heuristics``): the analyzer
turns the graph into facts once, ``recommend`` proposes complete
``PlanConfig(engine_id, GemmKnobs)`` entries from those facts alone, and every
listed plan carries the exact tile config the engine will build -- so a recorded
``(engine_id, knobs)`` replays the same kernel after the automatic pick changes.

The proposal set is deliberately ONE entry today: the automatic strategy the
engine picked before this module existed, now spelled out. Widening it
(neighbouring tiles, cluster shapes, split-K) is a change confined to
:func:`recommend`; the engine, the analyzer and the vocabulary stay as they are.
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
    return GemmFacts(chain=chain, dynamic_shapes=_graph_dynamic_shapes(graph))


def recommend(kind: str, facts: GemmFacts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Ordered candidate plans for ``facts`` (``kind`` is ``"A"`` or ``"FALLBACK"``).

    One candidate: the automatic tile strategy, gated exactly as
    ``check_support`` gates it (cutedsl floor + the chain-level probes), so a
    plan is listed only when the engine will build it. Its knobs are the
    config's canonical name in the shared vocabulary.
    """
    engine_id = offered.get(ENGINE)
    if engine_id is None:
        return []
    from .compiler import plan_config, probe_chain, probe_cutedsl

    try:
        probe_cutedsl()
        config = plan_config(facts.chain, dynamic_shapes=facts.dynamic_shapes)
        probe_chain(facts.chain, config)
        knobs = GemmKnobs.from_config(config)
    except (NotImplementedError, ValueError, KeyError) as exc:
        _LOG.debug("frost_gemm proposes nothing (%s): %s", kind, exc)
        return []
    return [PlanConfig(engine_id, knobs)]
