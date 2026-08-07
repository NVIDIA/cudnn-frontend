# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The engines this version ships — the library's own static table.

No registration call: the library knows which engines it was built with, so
discovery is the library's job. ``register_backend()`` is the out-of-tree
escape hatch and nothing else uses it.

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

    The SLOT is fixed forever: an autotune result is (engine_id, knobs), so a
    number that has shipped must keep meaning the same engine. Adding an engine
    takes the next free slot; slots are never reordered or reused.

    ``opt_in`` is per ENGINE, not per family: maturity is a property of one
    implementation. A family's half-precision engines can graduate while its
    fp8 engine is still maturing, which one flag per family made impossible.
    It lives here rather than on the engine class because the whole point of
    the gate is to know what to offer WITHOUT importing the engine.
    """

    slot: int
    opt_in: bool = False


@dataclass(frozen=True)
class EngineFamily:
    """One family of in-tree engines, described without importing them.

    A family is a KIND OF GRAPH (roughly what the backend calls an operation-graph
    mode, at a granularity of our choosing), not a group of engines that happen to
    ship together. Every graph belongs to exactly one family or to none, so the
    engines within a family compete and engines across families never do.

    The family owns: the id block its engines draw from, the arch range, and the
    vocabulary its graphs are described in. Two families MAY share an analyzer
    (SDPA forward and backward do) — sharing a description is their choice; what
    is fixed is that a graph is never claimed by two.

    ``slots`` is the SINGLE SOURCE of every python engine id. Engines do not
    declare their own: ``instantiate()`` hands each factory the ids its engines
    are to use, so an engine cannot claim a number it was not given, and the
    whole id space is readable here rather than reconstructed from four files.
    """

    engine_id: int  # first id of the FAMILY_BLOCK-wide block this family owns
    name: str
    module: str
    factory: str
    slots: Mapping[str, EngineSlot] = field(default_factory=dict)
    sm_lo: int = 0  # 0 => architecture-independent
    sm_hi: int = 10_000
    # ("module", "callable") producing this family's facts from a graph, or None
    # while a family still reads the graph inside its engines.
    analyzer: Optional[Tuple[str, str]] = None

    @property
    def id_end(self) -> int:
        return self.engine_id + FAMILY_BLOCK

    def owns(self, engine_id: int) -> bool:
        return self.engine_id <= engine_id < self.id_end

    def offered_ids(self, sm: Optional[int]) -> Dict[str, int]:
        """``{engine name: engine id}`` for the engines on offer here.

        ``sm is None`` means the probe could not answer (no cuda-python in the
        image, no device yet), NOT "wrong arch": filtering on an unknown would
        silently delete every arch-gated engine and leave ops that have no
        backend lowering with nothing to run. The engine's own check_support()
        re-checks the arch anyway.
        """
        if self.sm_lo and sm is not None and not (self.sm_lo <= sm <= self.sm_hi):
            return {}
        enabled = opt_in_engines_enabled()
        return {name: self.engine_id + s.slot for name, s in self.slots.items() if enabled or not s.opt_in}


# --- classification: which FAMILY does a graph belong to ---------------------
# ANCHOR nodes only: these are the node types that NAME a family, not a list of
# what a family can serve. Everything else -- POINTWISE, REDUCTION, a block-scale
# quantize, a node type added tomorrow -- is absent on purpose and ignored when
# classifying, so `matmul + pointwise` is a gemm graph. Whether the family can
# serve the whole graph is its analyzer's judgment; a coarser copy of that
# judgment here is what closed_under was, and it lied about RESHAPE.
#
# One anchor names exactly one family, and family_for() is a function, so "two
# families claimed this graph" is not a case that can arise. Names, not enum
# members, so this file still imports no engine code.
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
        sm_lo=90,  # union over the family's engines; each re-checks in check_support()
    ),
    EngineFamily(
        KDA_ID_BASE,
        "kda",
        "cudnn.linear_attention",
        "KdaEngines",
        slots={"kda_frost": EngineSlot(0), "kda_cutile": EngineSlot(1)},
        sm_lo=90,
    ),
    EngineFamily(
        GDN2_ID_BASE,
        "gdn2",
        "cudnn.linear_attention",
        "Gdn2Engines",
        slots={"gdn2_frost": EngineSlot(0)},
        sm_lo=100,
        sm_hi=103,
    ),
    EngineFamily(
        FROST_GEMM_ID_BASE,
        "frost_gemm",
        "cudnn.gemm.frost.engine",
        "FrostGemmEngines",
        slots={"frost_gemm": EngineSlot(0, opt_in=True)},
        sm_lo=100,
        sm_hi=103,
    ),
    EngineFamily(
        FROST_SDPA_FWD_ID_BASE,
        "frost_sdpa_fwd",
        "cudnn.sdpa.fwd.engine",
        "FrostSdpaFwdEngines",
        # Slots are FIXED FOREVER; append the next free one, never reorder.
        slots={
            "sdpa_fwd_prefill_sm100_d128": EngineSlot(0, opt_in=True),
            "sdpa_fwd_prefill_sm100_d256": EngineSlot(1, opt_in=True),
            "sdpa_fwd_prefill_sm100_d512": EngineSlot(2, opt_in=True),
            "sdpa_fwd_prefill_sm100_d128_mxfp8": EngineSlot(3, opt_in=True),
            "sdpa_fwd_prefill_sm100_d128_fp8": EngineSlot(4, opt_in=True),
            "sdpa_fwd_prefill_sm120": EngineSlot(5, opt_in=True),
            "sdpa_fwd_prefill_sm100_d192_d128": EngineSlot(6, opt_in=True),
        },
        sm_lo=100,
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
    EngineFamily(
        FROST_SDPA_BWD_ID_BASE,
        "frost_sdpa_bwd",
        "cudnn.sdpa.bwd.engine",
        "FrostSdpaBwdEngines",
        slots={"sdpa_bwd_sm120": EngineSlot(0, opt_in=True)},
        # TODO: widen when an SM100/SM80 spec lands
        sm_lo=120,
        sm_hi=121,
        analyzer=("cudnn.sdpa.graph_analyzer", "analyze"),
    ),
)


_INSTANCES: Dict[int, Any] = {}


def graph_node_types(graph) -> frozenset:
    """The graph's node-type names — the classification key. Pure python IR."""
    return frozenset(node.node_type.name for node in graph.nodes)


def family_for(graph) -> Optional[EngineFamily]:
    """The one family this graph belongs to, or None.

    A PURE property of the graph. No ``sm``, no environment: what kind of graph
    this is cannot depend on which machine is asking or on which engines happen
    to be built. Whether that family has an engine to offer here is a separate
    question — ``EngineFamily.offered_ids(sm)`` — and conflating the two made
    "not that kind of graph" indistinguishable from "no engine for it here".

    Classification is single-valued by construction: this is a function, so
    "two families claimed this graph" is not a case that can arise. A graph
    whose node types name two different families (a matmul and an sdpa in one
    graph) belongs to neither, and the backend is the only candidate.

    Note what this does NOT decide: a node type absent from the table is
    ignored, so ``matmul -> reshape`` still classifies as gemm. Whether a family
    can serve the WHOLE graph is its analyzer's judgment, deliberately — a
    coarser copy of that judgment here is what ``closed_under`` was, and it
    lied about RESHAPE.
    """
    named = {_ANCHOR_NODE_TO_FAMILY[n] for n in graph_node_types(graph) if n in _ANCHOR_NODE_TO_FAMILY}
    if len(named) != 1:
        return None
    name = named.pop()  # once: a generator would re-pop on every iteration
    return next(f for f in MANIFEST if f.name == name)


def resolve_analyzer(family: EngineFamily):
    """The family's facts callable, or None when it declares no analyzer.

    Importing it is the caller's decision, not this module's: keeping
    ``analyzer`` a pair of strings is what lets the coarse key stay
    import-free. Planning resolves it and attaches the record to the frozen
    graph; the family's engines then read that same record back.
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


def engines_for(graph, sm: Optional[int]):
    """Every in-tree engine of the graph's family, in candidate order."""
    family = family_for(graph)
    if family is None:
        return []
    ids = family.offered_ids(sm)  # availability is a separate question from kind
    return list(instantiate(family, ids)) if ids else []
