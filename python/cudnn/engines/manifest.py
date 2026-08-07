# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engines this version ships — the library's own static table.

No registration call: the library knows which engines it was built with, so
discovery is the library's job. ``register_backend()`` survives only as the
out-of-tree escape hatch.

A family may still be OPT-IN (``opt_in=True``): offered only when
``CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1``. That is a maturity gate, per family,
not an architecture switch — an engine graduates by flipping one field once it
has the arch coverage and the benchmarks to justify serving graphs by default.
Engines that are the ONLY implementation of their op (GDN/KDA/GDN2 have no
backend lowering at all) are never gated; gating them would just delete the op.

This module has ZERO imports of engine code (families are strings and ints), so
``import cudnn`` never pays an engine's import cost just to know the engine
exists — importing the CuTe DSL alone costs ~1.2 s, while the frost opsets
themselves are ~10-30 ms once it is resident.

Candidate selection is two-stage:

1. The coarse key (this file, microseconds). The graph's node-type names are
   matched against each family's ``anchors`` / ``closed_under`` / SM range. A
   GEMM family is not imported for an SDPA graph, and an SM100 family is not
   imported on SM90.
2. The engine's own ``check_support()`` — imported only now — decides.

Stage 1 is a NECESSARY condition, never the verdict: two families may both
claim a graph and both stay in the ranked list (a python engine's claim is "I
execute this whole graph", so claims compete rather than conflict — see
frost/README.md). Keeping it a filter rather than a central pattern matcher is
deliberate: an authoritative table would be a second matcher to maintain, and
every time an engine widened its envelope someone would have to remember to
widen the table too.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .engine_ids import FAMILY_BLOCK, FROST_GEMM_ID_BASE, FROST_SDPA_BWD_ID_BASE, FROST_SDPA_FWD_ID_BASE, GDN2_ID_BASE, GDN_ID_BASE, KDA_ID_BASE

_LOG = logging.getLogger("cudnn.engines.manifest")

# Opt-in for engines that are not enabled by default yet. Read live so it can be
# toggled per process / test.
_ENABLE_ENV = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"


def opt_in_engines_enabled() -> bool:
    """Whether opt-in (``opt_in=True``) manifest families are offered."""
    return os.environ.get(_ENABLE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class EngineFamily:
    """One family of in-tree engines, described without importing them.

    A family is a KIND OF GRAPH (roughly what the backend calls an operation-graph
    mode, at a granularity of our choosing), not a group of engines that happen to
    ship together. Every graph belongs to exactly one family or to none, so the
    engines within a family compete and engines across families never do.

    The family owns: the id block its engines draw from, the arch range, the
    maturity gate, and the vocabulary its graphs are described in. Two families
    MAY share an analyzer (SDPA forward and backward do) — sharing a description
    is their choice; what is fixed is that a graph is never claimed by two.
    """

    engine_id: int  # first id of the FAMILY_BLOCK-wide block this family owns
    name: str
    module: str
    factory: str
    sm_lo: int = 0  # 0 => architecture-independent
    sm_hi: int = 10_000
    opt_in: bool = False  # True => only offered with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
    # ("module", "callable") producing this family's facts from a graph, or None
    # while a family still reads the graph inside its engines.
    analyzer: Optional[Tuple[str, str]] = None

    @property
    def id_end(self) -> int:
        return self.engine_id + FAMILY_BLOCK

    def owns(self, engine_id: int) -> bool:
        return self.engine_id <= engine_id < self.id_end

    def offered(self, sm: Optional[int]) -> bool:
        """Whether this family is available at all — maturity gate and arch.

        Classification already decided the family; this only says whether it is
        on offer. ``sm is None`` means the probe could not answer (no cuda-python
        in the image, no device yet), NOT "wrong arch": filtering on an unknown
        would silently delete every arch-gated engine and leave ops that have no
        backend lowering with nothing to run. The engine's own check_support()
        re-checks the arch anyway.
        """
        if self.opt_in and not opt_in_engines_enabled():
            return False
        if self.sm_lo == 0 or sm is None:
            return True
        return self.sm_lo <= sm <= self.sm_hi


# --- classification: which FAMILY does a graph belong to ---------------------
# A partition, not N competing claims: one node type names exactly one family,
# so "two families claimed this graph" is not a case that can arise and is not
# an invariant anyone has to test. Names, not enum members, so this file still
# imports no engine code.
_FAMILY_OF_NODE = {
    "MATMUL": "frost_gemm",
    "MATMUL_FP8": "frost_gemm",
    "MOE_GROUPED_MATMUL": "frost_gemm",
    "SDPA": "frost_sdpa_fwd",
    "SDPA_FP8": "frost_sdpa_fwd",
    "SDPA_MXFP8": "frost_sdpa_fwd",
    "SDPA_BWD": "frost_sdpa_bwd",
    "GDN": "gdn",
    "GDN_BWD": "gdn",
    "KDA": "kda",
    "KDA_BWD": "kda",
    "GDN2": "gdn2",
    "GDN2_BWD": "gdn2",
}

# Node types that are not an op family of their own -- they attach to whatever
# family the graph's anchor node named. Listing them here is only a cheap way to
# skip an import the family would decline anyway; the engine's check_support()
# is what actually decides, so a type missing from this set costs one wasted
# import, never a wrong answer.
_ATTACHABLE = frozenset({"POINTWISE", "REDUCTION", "BLOCK_SCALE_QUANTIZE", "BLOCK_SCALE_DEQUANTIZE"})


# ---------------------------------------------------------------------------
# The manifest, one entry per family. Ids are pre-release (engine_ids.py), so
# blocks may still be re-cut; once shipped, a slot is fixed forever because an
# autotune result is (engine_id, knobs).
# ---------------------------------------------------------------------------
MANIFEST: Tuple[EngineFamily, ...] = (
    EngineFamily(
        GDN_ID_BASE,
        "gdn",
        "cudnn.linear_attention",
        "GdnEngines",
        sm_lo=90,  # union over the family's engines; each re-checks in check_support()
    ),
    EngineFamily(
        KDA_ID_BASE,
        "kda",
        "cudnn.linear_attention",
        "KdaEngines",
        sm_lo=90,
    ),
    EngineFamily(
        GDN2_ID_BASE,
        "gdn2",
        "cudnn.linear_attention",
        "Gdn2Engines",
        sm_lo=100,
        sm_hi=103,
    ),
    EngineFamily(
        FROST_GEMM_ID_BASE,
        "frost_gemm",
        "cudnn.gemm.frost.engine",
        "FrostGemmEngine",
        sm_lo=100,
        sm_hi=103,
        opt_in=True,
    ),
    EngineFamily(
        FROST_SDPA_FWD_ID_BASE,
        "frost_sdpa_fwd",
        "cudnn.sdpa.fwd.engine",
        "FrostSdpaFwdEngines",
        sm_lo=100,
        opt_in=True,
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
    EngineFamily(
        FROST_SDPA_BWD_ID_BASE,
        "frost_sdpa_bwd",
        "cudnn.sdpa.bwd.engine",
        "FrostSdpaBwdEngines",
        # TODO: widen when an SM100/SM80 spec lands
        sm_lo=120,
        sm_hi=121,
        opt_in=True,
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
)


_INSTANCES: Dict[int, Any] = {}


def graph_node_types(graph) -> frozenset:
    """The graph's node-type names — the classification key. Pure python IR."""
    return frozenset(node.node_type.name for node in graph.nodes)


def family_for(graph, sm: Optional[int]) -> Optional[EngineFamily]:
    """The one family this graph belongs to, or None.

    Classification is a lookup, so "which family" has exactly one answer by
    construction. A graph whose node types name two different families (a matmul
    and an sdpa in one graph) belongs to neither: no in-tree family serves that
    shape, and the backend is the only candidate.

    Returns None equally when the family exists but is not on offer here (opt-in
    withheld, wrong arch) — the caller's next move is the same either way.
    """
    named = {_FAMILY_OF_NODE[n] for n in graph_node_types(graph) if n in _FAMILY_OF_NODE}
    if len(named) != 1:
        return None
    name = named.pop()
    family = next(f for f in MANIFEST if f.name == name)
    return family if family.offered(sm) else None


def resolve_analyzer(family: EngineFamily):
    """The family's facts callable, or None when it declares no analyzer.

    Importing it is the caller's decision, not this module's: keeping
    ``analyzer`` a pair of strings is what lets the coarse key stay
    import-free. validate() resolves it and attaches the record to the graph;
    the family's engines then read that same record back.
    """
    if family.analyzer is None:
        return None
    import importlib

    module, attr = family.analyzer
    try:
        return getattr(importlib.import_module(module), attr)
    except ImportError as exc:
        # Same contract as instantiate(): a missing optional dependency makes
        # the family absent, not the graph unplannable. Importing an analyzer
        # pulls in its package (cudnn.sdpa.__init__ -> cuda.bindings, cutlass),
        # so without this a planning call raises instead of falling back to the
        # backend.
        _LOG.info("analyzer for %s is unavailable in this environment: %s", family.name, exc)
        return None


def instantiate(family: EngineFamily):
    """Import the family's module and build its engine(s) — cached per family.

    A factory may return one engine or a list of them (a family that exposes
    several ids out of its id block).

    A MISSING dependency (the CuTe DSL is not installed) makes the engine
    absent — that is a supported configuration, logged at info. Anything else
    is a bug in the engine and is logged at WARNING with its traceback: an
    engine that quietly vanishes from every plan list is the failure mode this
    design exists to remove, so it must be loud even though planning continues
    without it.
    """
    cached = _INSTANCES.get(family.engine_id)
    if cached is not None:
        return cached
    import importlib

    try:
        module = importlib.import_module(family.module)
        made = getattr(module, family.factory)()
    except ImportError as exc:
        _LOG.info("engine %s is unavailable in this environment: %s", family.name, exc)
        made = []
    except Exception:  # noqa: BLE001 — never let one bad engine take down planning
        _LOG.warning("engine %s failed to load and will not be offered", family.name, exc_info=True)
        made = []
    engines = list(made) if isinstance(made, (list, tuple)) else [made]
    for engine in engines:
        lo, hi = engine.owned_id_range
        if not (family.engine_id <= lo and hi <= family.id_end):
            raise ValueError(
                f"engine {engine.name!r} claims ids [{lo}, {hi}), which is not contained in the "
                f"[{family.engine_id}, {family.id_end}) block the manifest reserves for {family.name!r}"
            )
    _INSTANCES[family.engine_id] = engines
    return engines


def engines_for(graph, sm: Optional[int]):
    """Every in-tree engine of the graph's family, in candidate order."""
    family = family_for(graph, sm)
    return list(instantiate(family)) if family is not None else []
