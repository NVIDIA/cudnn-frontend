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

from .engine_ids import FROST_GEMM_ID_BASE, FROST_SDPA_BWD_ID_BASE, FROST_SDPA_FWD_ID_BASE, LINEAR_ATTENTION_ID_BASE

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

    A family is the unit everything else is scoped to: the node-type envelope
    it serves, the id block its engines draw from, the arch range, the maturity
    gate, and the facts vocabulary its graphs are described in. Splitting a
    family (fp8 SDPA out of SDPA, say) is an implementation choice, not a new
    concept — it costs one more entry here and nothing elsewhere.
    """

    engine_id: int  # first id of the block this family owns
    name: str
    module: str
    factory: str
    anchors: frozenset  # graph must contain at least one of these node types
    closed_under: Optional[frozenset] = None  # every node type must be in here; defaults to anchors
    id_hi: Optional[int] = None  # exclusive end of the id block it owns
    sm_lo: int = 0  # 0 => architecture-independent
    sm_hi: int = 10_000
    opt_in: bool = False  # True => only offered with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1
    # ("module", "callable") producing this family's facts from a graph, or None
    # while a family still reads the graph inside its engines. Family-scoped on
    # purpose: an SDPA fact means nothing to a GEMM engine, so there is no union
    # type to widen and no shared record to keep two readers in sync with.
    analyzer: Optional[Tuple[str, str]] = None

    @property
    def id_end(self) -> int:
        return self.id_hi if self.id_hi is not None else self.engine_id + 1

    def owns(self, engine_id: int) -> bool:
        return self.engine_id <= engine_id < self.id_end

    def matches(self, node_types: frozenset, sm: Optional[int]) -> bool:
        if self.opt_in and not opt_in_engines_enabled():
            return False
        if not (node_types & self.anchors):
            return False
        if not (node_types <= (self.closed_under or self.anchors)):
            return False
        if self.sm_lo == 0 or sm is None:
            # sm is None means the probe could not answer (no cuda-python in the
            # image, no device yet), NOT "this is the wrong arch". Filtering on
            # an unknown would silently delete every arch-gated engine and leave
            # ops that have no backend lowering with nothing to run — exactly
            # the vanishing-engine failure this design exists to remove. The
            # engine's own check_support() re-checks the arch anyway.
            return True
        return self.sm_lo <= sm <= self.sm_hi


# --- node-type groups (names, so this file imports no enum) -----------------
_GEMM_ANCHOR = frozenset({"MATMUL", "MATMUL_FP8", "MOE_GROUPED_MATMUL"})
# The closure is a PROMISE that the family serves every node type in it. RESHAPE
# is not in it because nothing in gemm/frost consumes a RESHAPE node: with it
# here, matmul -> reshape matched, the analyzer compiled only the matmul, and
# execute then asked for a buffer the caller never binds ('mm::C').
_GEMM_CLOSURE = _GEMM_ANCHOR | frozenset({"POINTWISE", "REDUCTION", "BLOCK_SCALE_QUANTIZE", "BLOCK_SCALE_DEQUANTIZE"})
_SDPA_FWD = frozenset({"SDPA", "SDPA_FP8", "SDPA_MXFP8"})
_SDPA_BWD = frozenset({"SDPA_BWD"})
_GDN = frozenset({"GDN", "GDN_BWD"})
_GDN2 = frozenset({"GDN2", "GDN2_BWD"})
_KDA = frozenset({"KDA", "KDA_BWD"})


# ---------------------------------------------------------------------------
# The manifest. Append-only: a shipped engine_id is never renumbered (an
# autotune result is (engine_id, knobs) and must replay across versions).
# ---------------------------------------------------------------------------
MANIFEST: Tuple[EngineFamily, ...] = (
    EngineFamily(
        LINEAR_ATTENTION_ID_BASE + 1,
        "gdn_frost",
        "cudnn.linear_attention.frost.gdn_engine",
        "GdnFrostEngine",
        _GDN,
        sm_lo=100,
        sm_hi=103,
    ),
    EngineFamily(
        LINEAR_ATTENTION_ID_BASE + 2,
        "gdn_cutile",
        "cudnn.linear_attention.cutile.gdn_engine",
        "GdnCuTileEngine",
        _GDN,
        sm_lo=90,
    ),
    EngineFamily(
        LINEAR_ATTENTION_ID_BASE + 3,
        "kda_frost",
        "cudnn.linear_attention.frost.kda_engine",
        "KdaFrostEngine",
        _KDA,
        sm_lo=100,
        sm_hi=103,
    ),
    EngineFamily(
        LINEAR_ATTENTION_ID_BASE + 4,
        "kda_cutile",
        "cudnn.linear_attention.cutile.kda_engine",
        "KdaCuTileEngine",
        _KDA,
        sm_lo=90,
    ),
    EngineFamily(
        LINEAR_ATTENTION_ID_BASE + 5,
        "gdn2_frost",
        "cudnn.linear_attention.frost.gdn2_engine",
        "Gdn2FrostEngine",
        _GDN2,
        sm_lo=100,
        sm_hi=103,
    ),
    EngineFamily(
        FROST_GEMM_ID_BASE + 0,
        "frost_gemm",
        "cudnn.gemm.frost.engine",
        "FrostGemmEngine",
        _GEMM_ANCHOR,
        _GEMM_CLOSURE,
        id_hi=FROST_GEMM_ID_BASE + 100,
        sm_lo=100,
        sm_hi=103,
        opt_in=True,
    ),
    EngineFamily(
        FROST_SDPA_FWD_ID_BASE + 0,
        "frost_sdpa_fwd",
        "cudnn.sdpa.fwd.engine",
        "FrostSdpaFwdEngines",
        _SDPA_FWD,
        id_hi=FROST_SDPA_FWD_ID_BASE + 100,
        sm_lo=100,
        opt_in=True,
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
    EngineFamily(
        FROST_SDPA_BWD_ID_BASE + 0,
        "frost_sdpa_bwd",
        "cudnn.sdpa.bwd.engine",
        "FrostSdpaBwdEngines",
        _SDPA_BWD,
        id_hi=FROST_SDPA_BWD_ID_BASE + 100,
        # TODO: widen when an SM100/SM80 spec lands
        sm_lo=120,
        sm_hi=121,
        opt_in=True,
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
)


_INSTANCES: Dict[int, Any] = {}


def graph_node_types(graph) -> frozenset:
    """The graph's node-type names — the coarse key. Pure python IR, no imports."""
    return frozenset(node.node_type.name for node in graph.nodes)


def candidate_families(node_types: frozenset, sm: Optional[int]) -> Tuple[EngineFamily, ...]:
    """Manifest families whose coarse key matches, in manifest order."""
    return tuple(family for family in MANIFEST if family.matches(node_types, sm))


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
    return getattr(importlib.import_module(module), attr)


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
    """Every in-tree engine whose coarse key matches ``graph``, in manifest order."""
    node_types = graph_node_types(graph)
    out = []
    for family in candidate_families(node_types, sm):
        out.extend(instantiate(family))
    return out
