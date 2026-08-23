# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engines this version ships — the library's own static table.

No registration call: the library knows which engines it was built with, so
discovery is the library's job, and this table is the ONLY way a python engine
exists. An engine id decodes to a family and a slot here
(:func:`engine_for_id`), so nothing has to be registered for one to be named,
built, or replayed.

This module has ZERO imports of engine code (families are strings and ints), so
``import cudnn`` never pays an engine's import cost just to know the engine
exists.

Dispatch is two stages:

1. CLASSIFY (this file, microseconds). ``_ANCHOR_NODE_TO_FAMILY`` maps a node type to
   the one family that serves that kind of graph, so a graph has 0 or 1
   families and engines across families never compete. Arch range and per-engine
   maturity then say which of that family's engines are on offer.
2. The engine's own ``check_support()`` — imported only now — decides whether it
   can serve this particular graph.

Stage 1 is a NECESSARY condition, never the verdict. It deliberately does NOT
describe what an engine can do: that judgment lives in check_support(), and a
coarser copy of it here would be a second thing to maintain and a place to lie.
``closed_under`` was exactly that copy, and it lied about RESHAPE.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .engine_ids import FAMILY_BLOCK, FROST_GEMM_ID_BASE, FROST_SDPA_BWD_ID_BASE, FROST_SDPA_FWD_ID_BASE, GDN2_ID_BASE, GDN_ID_BASE, KDA_ID_BASE

_LOG = logging.getLogger("cudnn.engines.manifest")

# Opt-in for engines that are not enabled by default yet. Read live so it can be
# toggled per process / test.
_ENABLE_ENV = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"


def opt_in_engines_enabled() -> bool:
    """Whether opt-in engines (``EngineSlot.opt_in``) are offered."""
    return os.environ.get(_ENABLE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class EngineSlot:
    """One engine's place in its family's id block.

    A shipped slot is fixed forever (an autotune result is (engine_id, knobs)):
    append the next free one, never reorder or reuse. ``opt_in`` is per engine
    so one implementation can graduate while a sibling matures; it lives here
    rather than on the engine class because the gate must answer without
    importing the engine.
    """

    slot: int
    opt_in: bool = False


@dataclass(frozen=True)
class EngineFamily:
    """One family of in-tree engines, described without importing them.

    A family is a KIND OF GRAPH (roughly the backend's operation-graph mode, at
    a granularity of our choosing), not a group of engines that ship together.
    Every graph belongs to exactly one or to none, so engines within a family
    compete and engines across families never do. Two families may share an
    analyzer (SDPA forward and backward do); what is fixed is that a graph is
    never claimed by two.

    ``slots`` is the single source of every python engine id -- engines are
    handed theirs by :func:`instantiate` and declare none of their own.
    """

    engine_id: int  # first id of the FAMILY_BLOCK-wide block this family owns
    name: str
    module: str
    factory: str
    slots: Mapping[str, EngineSlot] = field(default_factory=dict)
    # ("module", "callable") producing this family's facts from a graph, or None
    # while a family still reads the graph inside its engines.
    analyzer: Optional[Tuple[str, str]] = None
    # ("module", "callable") ranking (engine_id, knobs) for this family, given
    # its facts and the backend's entries. The family is the smallest scope that
    # can rank -- an engine cannot see its siblings. None falls back to one
    # default plan per accepting engine, ahead of the backend's.
    heuristics: Optional[Tuple[str, str]] = None

    @property
    def id_end(self) -> int:
        return self.engine_id + FAMILY_BLOCK

    def owns(self, engine_id: int) -> bool:
        return self.engine_id <= engine_id < self.id_end

    def offered_ids(self) -> Dict[str, int]:
        """``{engine name: engine id}`` for the engines on offer.

        Maturity only -- no arch range. Whether an engine suits a device is the
        engine's own check_support(); a coarser copy here lied twice before it
        was deleted.
        """
        enabled = opt_in_engines_enabled()
        return {name: self.engine_id + s.slot for name, s in self.slots.items() if enabled or not s.opt_in}


# Node types that NAME a family -- not a list of what a family can serve.
# POINTWISE, REDUCTION, anything added tomorrow are absent on purpose and
# ignored when classifying, so `matmul + pointwise` is a gemm graph. Names, not
# enum members, so this file imports no engine code.
_ANCHOR_NODE_TO_FAMILY = {
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
        slots={"gdn_frost": EngineSlot(0), "gdn_cutile": EngineSlot(1)},
        analyzer=("cudnn.linear_attention.graph_analyzer", "analyze"),
    ),
    EngineFamily(
        KDA_ID_BASE,
        "kda",
        "cudnn.linear_attention",
        "KdaEngines",
        slots={"kda_frost": EngineSlot(0), "kda_cutile": EngineSlot(1)},
        analyzer=("cudnn.linear_attention.graph_analyzer", "analyze"),
    ),
    EngineFamily(
        GDN2_ID_BASE,
        "gdn2",
        "cudnn.linear_attention",
        "Gdn2Engines",
        slots={"gdn2_frost": EngineSlot(0)},
        analyzer=("cudnn.linear_attention.graph_analyzer", "analyze"),
    ),
    EngineFamily(
        FROST_GEMM_ID_BASE,
        "frost_gemm",
        "cudnn.gemm.frost.engine",
        "FrostGemmEngines",
        slots={"frost_gemm": EngineSlot(0, opt_in=True)},
    ),
    EngineFamily(
        FROST_SDPA_FWD_ID_BASE,
        "frost_sdpa_fwd",
        "cudnn.sdpa.fwd.engine",
        "FrostSdpaFwdEngines",
        # Slots are FIXED FOREVER; append the next free one, never reorder.
        # RETIRED (one engine per arch x dtype family absorbed the per-head-dim
        # rows; kernel-flavor choice moved into the lowering — never reuse):
        #   0 sm100_d128, 1 sm100_d256, 2 sm100_d512, 3 sm100_d128_mxfp8,
        #   4 sm100_d128_fp8, 6 sm100_d192_d128, 9 sm100_d192_d128_fp8,
        #   10 sm100_d192_d128_mxfp8
        slots={
            "sdpa_fwd_prefill_sm120": EngineSlot(5, opt_in=True),
            "sdpa_fwd_prefill_sm120_fp8": EngineSlot(7, opt_in=True),
            "sdpa_fwd_prefill_sm80": EngineSlot(8, opt_in=True),
            "sdpa_fwd_prefill_sm100": EngineSlot(11, opt_in=True),
            "sdpa_fwd_prefill_sm100_mxfp8": EngineSlot(12, opt_in=True),
            "sdpa_fwd_prefill_sm100_fp8": EngineSlot(13, opt_in=True),
            "sdpa_fwd_prefill_sm107_fp8": EngineSlot(14, opt_in=True),
        },
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
        heuristics=("cudnn.sdpa.fwd.heuristics", "recommend"),
    ),
    EngineFamily(
        FROST_SDPA_BWD_ID_BASE,
        "frost_sdpa_bwd",
        "cudnn.sdpa.bwd.engine",
        "FrostSdpaBwdEngines",
        slots={
            "sdpa_bwd_sm120": EngineSlot(0, opt_in=True),
            "sdpa_bwd_sm80": EngineSlot(1, opt_in=True),
        },
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
)


_INSTANCES: Dict[int, Any] = {}


def graph_node_types(graph) -> frozenset:
    """The graph's node-type names — the classification key. Pure python IR."""
    return frozenset(node.node_type.name for node in graph.nodes)


def family_for(graph) -> Optional[EngineFamily]:
    """The one family this graph belongs to, or None.

    A pure property of the graph: no ``sm``, no environment. Availability is a
    separate question (``EngineFamily.offered_ids``). Naming two families (a
    matmul and an sdpa together) means neither, and the backend is the only
    candidate.

    Does NOT decide coverage: an unlisted node type is ignored, so
    ``matmul -> reshape`` still classifies as gemm and the analyzer declines it.
    """
    named = {_ANCHOR_NODE_TO_FAMILY[n] for n in graph_node_types(graph) if n in _ANCHOR_NODE_TO_FAMILY}
    if len(named) != 1:
        return None
    name = named.pop()  # once: a generator would re-pop on every iteration
    return next(f for f in MANIFEST if f.name == name)


def _resolve(family: EngineFamily, ref: Optional[Tuple[str, str]], what: str):
    """Import a ("module", "callable") declaration, or None when absent.

    Importing is the caller's decision, not this module's: keeping these
    declarations pairs of strings is what lets the coarse key stay import-free.
    A missing optional dependency makes the hook absent, not the graph
    unplannable -- importing one pulls in its package (cudnn.sdpa.__init__ ->
    cuda.bindings, cutlass), so without this a planning call raises instead of
    falling back to the backend.
    """
    if ref is None:
        return None
    import importlib

    module, attr = ref
    try:
        return getattr(importlib.import_module(module), attr)
    except ImportError as exc:
        _LOG.info("%s for %s is unavailable in this environment: %s", what, family.name, exc)
        return None


def resolve_heuristics(family: EngineFamily):
    """The family's proposal callable, or None when it declares none.

    The contract is ``recommend(kind, facts, offered) -> [PlanConfig]`` — pure
    and backend-blind; placement against the backend's entries happens once
    for every family in ``engines/heuristics._assemble``.
    """
    return _resolve(family, family.heuristics, "heuristics")


def resolve_analyzer(family: EngineFamily):
    """The family's facts callable, or None when it declares no analyzer.

    Planning resolves it and attaches the record to the frozen graph; the
    family's heuristics and engines then read that same record back.
    """
    return _resolve(family, family.analyzer, "analyzer")


def instantiate(family: EngineFamily, ids: Dict[str, int]):
    """Import the family's module and build the engines named in ``ids``.

    The factory is HANDED the ids its engines are to use. Engines do not carry
    an id of their own, so one cannot claim a number the manifest did not give
    it, and a gated engine is simply absent from ``ids`` rather than built and
    filtered afterwards.

    A MISSING dependency (the CuTe DSL is not installed) makes the family absent
    — a supported configuration, logged at info. Anything else is a bug in the
    engine and is logged at WARNING with its traceback: an engine that quietly
    vanishes from every plan list is the failure mode this design exists to
    remove, so it must be loud even though planning continues without it.
    """
    key = (family.engine_id, tuple(sorted(ids.items())))
    cached = _INSTANCES.get(key)
    if cached is not None:
        return cached
    import importlib

    try:
        module = importlib.import_module(family.module)
        made = getattr(module, family.factory)(dict(ids))
    except ImportError as exc:
        _LOG.info("family %s is unavailable in this environment: %s", family.name, exc)
        made = []
    except Exception:  # noqa: BLE001 — never let one bad family take down planning
        _LOG.warning("family %s failed to load and will not be offered", family.name, exc_info=True)
        made = []
    engines = list(made) if isinstance(made, (list, tuple)) else [made]
    for engine in engines:
        if engine.engine_id != ids.get(engine.name):
            raise ValueError(f"engine {engine.name!r} answers for id {engine.engine_id}, not the {ids.get(engine.name)} the manifest assigned it")
    _INSTANCES[key] = engines
    return engines


def engines_for(graph):
    """Every in-tree engine of the graph's family, in candidate order."""
    family = family_for(graph)
    if family is None:
        return []
    ids = family.offered_ids()  # availability is a separate question from kind
    return list(instantiate(family, ids)) if ids else []


def engine_for_id(engine_id: int):
    """The engine that owns ``engine_id``, or None.

    An engine id is fully decodable from this table: the family owning the id
    block, then the slot within it. Nothing has to be registered first, which is
    what lets create_execution_plan() replay an autotune result on a fresh graph
    -- including an engine that is not a candidate for THAT graph, where the
    replay is a deliberate pin rather than a routing decision.
    """
    for family in MANIFEST:
        if not family.owns(engine_id):
            continue
        ids = family.offered_ids()
        if engine_id not in ids.values():
            return None  # a real slot, but gated off in this process
        return next((e for e in instantiate(family, ids) if e.engine_id == engine_id), None)
    return None
