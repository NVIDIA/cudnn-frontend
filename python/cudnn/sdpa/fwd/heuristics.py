# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SDPA-forward family's proposals: which cells, with which knob sets.

:func:`recommend` is the family's ENTIRE heuristic surface — the PURE core:
``(kind, facts, offered) -> [PlanConfig]``. Backend-blind, graph-blind,
import-light. For every offered cell whose capability row admits the facts,
the cell's rule emits an ORDERED list of COMPLETE knob assignments (every
axis the row declares a domain for carries a concrete value; ``None`` only on
undeclared axes), each re-validated through ``mismatch(caps, facts, knobs)``
— a set is honored or never listed. The same engine appears once per
surviving set. Standalone callers (wrappers, autotuners) invoke this directly
with a hand-built :class:`~cudnn.sdpa.graph_analyzer.SdpaGraphFacts`; nothing
here touches the backend, a graph object, or heuristic modes.

``kind`` is ``"A"`` (candidates worth timing, best guess first, runners-up
behind for a caller that autotunes) or ``"FALLBACK"`` (the config expected to
build where A's choice may not — nothing chosen for speed).

Everything else about the final list — mode blocks, the backend's entries,
the delegating entry, dedup, the mode strip — is PLACEMENT, and placement is
not a family opinion: it lives once in ``engines/heuristics._assemble``,
under the standing assumption that these proposals lead the backend's entries
(an OSS engine measured behind the backend gets fixed or pulled, not
demoted).

Cross-ENGINE order within a proposal batch is ``ENGINE_SPECS`` declaration
order. Today that is unambiguous in practice — co-eligible cells are the
envelope-overlap family, which all lower to the same kernel — and the seam
for a real ranking, when one is measured, is a score stage here in
:func:`recommend`, not a new layer.

To add a rule for a cell: write a generator (:func:`_sm120_tiles` is the
worked example), register the cell in ``_TILE_RULE_CELLS`` (or grow a new
axis via the five-part checklist in ``engines.py``), put the measurement in
the commit. A cell with no rule runs its row's sole point per axis, which is
the honest answer while nobody has timed it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Tuple

import cudnn

from cudnn.engines.base import PlanConfig
from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from cudnn.sdpa.fwd.config_sm100 import cga_tile_m, pack_gqa_supported
from cudnn.sdpa.fwd.config_sm120 import FP8_HEAD_TILE_GRANULE, HEAD_TILE_GRANULE, SMEM_CAPACITY_BYTES, smem_bytes
from cudnn.sdpa.fwd.engines import ENGINE_SPECS, Capabilities, EngineSpec, SdpaFwdKnobs, _band_covers_kv_tail, mismatch

# Cells whose (tile_m, tile_n) choice _sm120_tiles makes.
_TILE_RULE_CELLS = frozenset({"sdpa_fwd_prefill_sm120", "sdpa_fwd_prefill_sm120_fp8"})

# Cap on complete knob sets emitted per engine per kind. The combiner grows
# Σ|axis runners|, never the cartesian product; this bound keeps the plan list
# legible and an autotune-ALL pass affordable even as axes accumulate.
_MAX_SETS_PER_ENGINE = 6

# The causal-balancing budget for the CLC/static LPT_L2 policy on SM100/SM120:
# LPT_L2's block-cyclic head grouping only pays when ONE head's K+V working set
# can actually stay L2-resident.
_SM100_L2_BUDGET_BYTES = 50 * 1024 * 1024

# The SM80 kernels' L2 grouping budget is a per-flavor MiB table fed to the
# template (sched_l2_mib); the adapter owns that table. For POINT ORDERING all
# that matters here is that SM80's measured primary for causal is LPT_L2.


# ---------------------------------------------------------------------------
# axis generators — each returns an ORDERED candidate list, best first
# ---------------------------------------------------------------------------


def _sole(values):
    """The only value on an axis, or None where the row declares no domain."""
    return next(iter(values)) if len(values) == 1 else None


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# --- KV split (see choose_split_kv) ----------------------------------------
# Largest split considered; past this the reduction outgrows the parallelism.
_SPLIT_KV_MAX = 16
# A split thinner than this is prologue/epilogue dominated.
_SPLIT_KV_MIN_TILES = 2
# What a CTA-tile costs beyond its KV loop (Q load, prologue, epilogue), in
# units of one KV tile. Empirical: re-measure if the per-tile fixed cost moves.
_SPLIT_KV_CTA_COST = 21.0


def choose_split_kv(
    *,
    q_tiles: int,
    heads_q: int,
    batch: int,
    kv_tiles: int,
    sm_count: int,
    ctas_per_tile: int = 1,
    max_split: int = _SPLIT_KV_MAX,
) -> int:
    """How many KV chunks to cut each Q tile into; 1 = do not split.

    A prefill launch is ``q_tiles * heads_q * batch`` independent tiles, each
    walking the whole KV loop.  When that product is below the SM count the chip
    idles however long the loop is; splitting multiplies the tile count by ``s``
    and divides each tile's KV work by it, then pays one reduction over the
    partials.

    A CTA holds its tile for the whole loop, so a launch costs whole WAVES.
    Minimise, over powers of two:

        waves(s) = ceil(base_ctas * s / sm_count)
        cost(s)  = waves(s) * (ceil(kv_tiles / s) + CTA_COST)

    CTA_COST is what a tile re-pays whatever its loop length, so it sits inside
    the wave term -- once per CTA-tile, not once per split.

    What falls out: an under-full launch splits until the wave is full; an
    over-full one with a partial-wave tail splits FINER to smooth it, even past
    the SM count; an exactly balanced one (base_ctas = k * sm_count) has no tail
    and never splits.

    Powers of two only, because ``split_kv`` is a TemplateParams field and so a
    kernel-module cache key -- an unrestricted choice mints a compiled
    specialization per shape.

    Returns 1 when there is nothing to split or nothing beats not splitting.
    Bounded by ``max_split``, by ``kv_tiles`` (more splits than tiles would
    leave some provably empty) and by ``_SPLIT_KV_MIN_TILES``.
    """
    if min(q_tiles, heads_q, batch, kv_tiles, sm_count, ctas_per_tile) <= 0:
        return 1
    base_ctas = q_tiles * heads_q * batch * ctas_per_tile
    if kv_tiles <= 1:
        return 1

    best_split = 1
    best_cost = float(_ceil_div(base_ctas, sm_count) * (kv_tiles + _SPLIT_KV_CTA_COST))
    split = 2
    while split <= min(max_split, kv_tiles):
        # Every split must stay thick enough to amortise its own prologue and
        # epilogue. The chunking hands the remainder to the leading splits, so
        # the THINNEST gets floor(kv_tiles / split) -- that is what must clear.
        if kv_tiles // split < _SPLIT_KV_MIN_TILES:
            break
        waves = _ceil_div(base_ctas * split, sm_count)
        cost = waves * (_ceil_div(kv_tiles, split) + _SPLIT_KV_CTA_COST)
        if cost < best_cost:
            best_split, best_cost = split, cost
        split <<= 1
    return best_split


def _sm120_tiles(caps: Capabilities, facts) -> Tuple[int, int]:
    """(tile_m, tile_n) for the SM120 SDPA-forward prefill cell.

    ``tile_m=64`` when the grid cannot fill the machine AND each CTA has enough
    KV tiles to amortize the extra Q-tile loop; a causal mask counts as a
    halved grid because it halves the work per CTA. ``tile_n`` is the largest
    that fits SMEM: 128 is fastest, but a wide head has no room for it (D>=208
    in half, further out in FP8 -- its KV tile is a byte per element).

    Fit is a KERNEL property, so ``config_sm120.smem_bytes`` is the one
    implementation and the adapter's check calls it too. So is the tuning: an
    earlier revision of this template staged P through SMEM and wanted
    ``tile_n=64``. Re-measure when a kernel changes -- the sweeps are in PR #528
    (f16: regret 1.009 geomean, 1.054 worst) and PR #509 (fp8: 1.0046 geomean,
    1.039 worst over 30 seeded cells).

    Read those worst cases with care. Most cells here are within the ~1%
    run-to-run floor of each other, so a single sweep's worst cell is often
    noise: an unseeded run of the SAME code reported 1.155 at B1xH16xS2048
    causal, which the seeded repeat shows as a tie. What survives repetition is
    that the misses cluster on CAUSAL shapes, where the triangular mask shifts
    the per-CTA balance in a way `grid` alone does not capture.
    """
    sm_count = facts.device_sm_count or 0
    grid = -(-facts.s_q // 128) * facts.h_q * facts.b
    if facts.causal:
        grid //= 2
    kv_tiles = -(-facts.s_kv // 128)
    fine = sm_count > 0 and (grid * 2 <= sm_count or (grid * 2 <= 3 * sm_count and kv_tiles >= 12))
    tile_m = 64 if fine else 128
    # FP8 stages a byte per KV element but still writes O in half, so the two
    # SMEM terms size differently -- see config_sm120.smem_bytes. The kernel
    # stages ENVELOPE head tiles (actual dims round up to the granule), so
    # the fit check must round the same way or a tile offered here declines
    # at build -- and a knob-carried tile skips the adapter's fallback.
    qkv_itemsize, o_itemsize = (1, 2) if facts.is_fp8 else (2, 2)
    granule = FP8_HEAD_TILE_GRANULE if facts.is_fp8 else HEAD_TILE_GRANULE
    d_qp = -(-facts.d_qk // granule) * granule
    d_vp = -(-facts.d_v // granule) * granule
    fits = [n for n in sorted(caps.tile_ns, reverse=True) if smem_bytes(d_qp, d_vp, tile_m, n, qkv_itemsize, o_itemsize) <= SMEM_CAPACITY_BYTES]
    return tile_m, (fits[0] if fits else min(caps.tile_ns))


def _tile_points(spec: EngineSpec, facts) -> List[Tuple[Optional[int], Optional[int]]]:
    """Ordered (tile_m, tile_n) candidates: the rule's best guess first, then
    the rest of the SMEM-fitting domain for a caller that autotunes. Configs
    the kernel cannot fit are not runners-up — they would sit in the list only
    to decline at build."""
    caps = spec.capabilities
    if spec.name not in _TILE_RULE_CELLS:
        # No rule measured for this cell: its capability row has one point per
        # axis, so there is nothing to choose between anyway.
        return [(_sole(caps.tile_ms), _sole(caps.tile_ns))]
    best = _sm120_tiles(caps, facts)
    qkv_itemsize, o_itemsize = (1, 2) if facts.is_fp8 else (2, 2)
    # Same envelope rounding as _sm120_tiles / the adapter's SMEM check.
    granule = FP8_HEAD_TILE_GRANULE if facts.is_fp8 else HEAD_TILE_GRANULE
    d_qp = -(-facts.d_qk // granule) * granule
    d_vp = -(-facts.d_v // granule) * granule
    domain = [(m, n) for m in caps.tile_ms for n in caps.tile_ns if smem_bytes(d_qp, d_vp, m, n, qkv_itemsize, o_itemsize) <= SMEM_CAPACITY_BYTES]
    return sorted(domain or [best], key=lambda mn: (mn != best, mn[1] != best[1], -mn[0]))


def _sched_points(caps: Capabilities, facts) -> List[Optional[int]]:
    """Ordered scheduler-policy candidates.

    The PRIMARY reproduces what each adapter's internal derivation historically
    chose for the graph path, so promoting the decision into the ranked list
    changes nothing for a caller that builds the first plan; the remaining
    domain follows for autotune. This is the one causal LPT/LPT_L2 oracle on
    the graph path — the adapters keep a None-input derivation only for
    standalone wrapper users who bypass ranking.
    """
    domain = caps.sched_policies
    if len(domain) <= 1:
        return [_sole(domain)]
    causal_ish = facts.causal or facts.right_band_widening
    if caps.sm_hi == 80:
        # SM80's measured choices (see the adapter's flavor table): causal
        # always groups for L2; pure SWA prefers plain LPT only in the band
        # where the window walk is long enough to imbalance rows.
        if causal_ish:
            primary = SCHED_LPT_L2
        elif facts.window_left is not None and facts.window_left >= 0:
            primary = SCHED_LPT if 1024 <= facts.s_kv <= 16384 else SCHED_NATURAL
        else:
            primary = SCHED_NATURAL
    elif causal_ish:
        # SM100/SM120: balance the triangular load; pick the LPT variant by
        # whether one head's K+V working set fits the L2 budget.
        elem = 1 if (facts.is_fp8 or facts.is_mxfp8) else 2
        one_head_bytes = int(facts.s_kv) * (int(facts.d_qk) + int(facts.d_v)) * elem
        primary = SCHED_LPT_L2 if one_head_bytes <= _SM100_L2_BUDGET_BYTES else SCHED_LPT
    else:
        primary = SCHED_NATURAL
    order = {SCHED_LPT_L2: (SCHED_LPT, SCHED_NATURAL), SCHED_LPT: (SCHED_LPT_L2, SCHED_NATURAL), SCHED_NATURAL: (SCHED_LPT, SCHED_LPT_L2)}
    runners = [p for p in order[primary] if p in domain]
    # A mask-free graph gains nothing from either LPT remap — the grid is
    # already balanced — so don't spend autotune slots on them.
    if not causal_ish and facts.window_left is None:
        runners = []
    return [primary if primary in domain else _sole(domain) or SCHED_NATURAL] + runners


# --- pack_gqa (GQA head packing) --------------------------------------------


def _pack_gqa_wins(facts, tile_q: int) -> bool:
    """Pack when the Q sequence cannot fit in a single tile, then we can further
    apply split_kv on top of GQA packing.

    TODO: we may enhance this heuristic logic in the future by considering more
    factors such as the device SM count.
    """
    return facts.s_q < tile_q


def _pack_gqa_tile_q(caps: Capabilities, facts, tile_m: Optional[int]) -> int:
    """The Q rows one grid tile covers, for :func:`_pack_gqa_wins`.

    The SM100 family runs CGA tiles — ``cga_tile_m`` of the flavor the envelope
    lowering picks for the graph's head dims (d128/d192: 512; d256/d512:
    256). SM120 launches one CTA per tile, so it is ``tile_m`` itself.
    """
    if caps.sm_lo >= 120 and caps.sm_hi < 130:
        return tile_m or 128
    if facts.d_qk <= 128 and facts.d_v <= 128:
        return cga_tile_m(128)
    if facts.d_qk <= 192 and facts.d_v <= 128:
        return cga_tile_m(192)
    if facts.d_qk <= 256 and facts.d_v <= 256:
        return cga_tile_m(256)
    return cga_tile_m(512)


def _pack_gqa_points(caps: Capabilities, facts, tile_m: int) -> Tuple[bool, ...]:
    """The pack_gqa axis, best first: ``(True, False)`` when packing wins,
    ``(False, True)`` when it is only eligible, ``(False,)`` when it is not."""
    if not (True in caps.pack_gqas and not facts.thd and facts.h_q != facts.h_kv and pack_gqa_supported(facts.h_q, facts.h_kv, tile_m)):
        return (False,)
    if _pack_gqa_wins(facts, _pack_gqa_tile_q(caps, facts, tile_m)):
        return (True, False)
    return (False, True)


def _split_points(caps: Capabilities, facts, tile_m: Optional[int], tile_n: Optional[int], cga: Optional[int], pack_g: int = 1) -> List[Optional[int]]:
    """Ordered split-KV candidates for the chosen tile geometry.

    ``pack_g`` is the pack_gqa group of the set the split rides (1 =
    unpacked): packing multiplies each head-group's Q rows by G and divides
    the head count by it, so the wave-cost model must see the PACKED launch
    — the packed grid is smaller, which is exactly when splitting pays.

    The value comes from :func:`choose_split_kv`'s wave-cost model, fed the
    facts-level launch geometry (``tile_m*cga`` rows per tile — the recommend
    tier's approximation of the kernel Cfg's exact ``TILES_Q*TILE_M*CTA_MMA``).
    The generator respects the split path's structural limits (dense-only, no
    sink — mismatch() enforces the same, so an emitted >1 never reaches a
    kernel that cannot honor it).

    The split point is deliberately a RUNNER-UP behind no-split until sweeps
    justify flipping the default: first-build behavior stays exactly what this
    dispatch has always done, and autotune / select_plan reach the split plan
    today.
    """
    domain = caps.split_kvs
    if len(domain) <= 1:
        return [_sole(domain)]
    no_split = 1 if 1 in domain else min(domain)
    if facts.thd or facts.has_sink or facts.padded or facts.seq_q_trim:
        return [no_split]
    if caps.skv_tail_via_padding and facts.s_kv % (caps.skv_tile or 128) != 0 and not _band_covers_kv_tail(facts):
        # This S_kv would be served through the synthesized KV-tail padding,
        # which the split cannot ride (mismatch declines the same combination).
        return [no_split]
    if (facts.is_fp8 or facts.is_mxfp8) and facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
        # The combine reduces partials in half precision; reducing QUANTIZED
        # partials would lose what the split is meant to be neutral about.
        return [no_split]
    sm_count = facts.device_sm_count or 0
    if sm_count <= 0:
        return [no_split]
    rows_per_tile = (tile_m or 128) * (cga or 1)
    split = choose_split_kv(
        q_tiles=_ceil_div(facts.s_q * pack_g, rows_per_tile),
        heads_q=facts.h_q // pack_g,
        batch=facts.b,
        kv_tiles=_ceil_div(facts.s_kv, tile_n or 128),
        sm_count=sm_count,
        ctas_per_tile=cga or 1,
        max_split=max(domain),
    )
    # Snap the model's power-of-two answer down into the declared domain.
    usable = [s for s in sorted(domain) if 1 < s <= split]
    if not usable:
        return [no_split]
    return [no_split, usable[-1]]


def _softmax_points(caps: Capabilities) -> List[Optional[int]]:
    """Softmax-precision candidates.

    FLOAT when the row serves it, else the row's sole point. HALF is NEVER
    proposed: it changes numerics (f16x2 exponent), so it is reachable only
    by explicit request — auto-proposing it is the CUDNN_SOFTMAX_PRECISION
    environment-knob failure mode this vocabulary exists to avoid. Flipping the
    Rubin-FP8 default to HALF is a separate, evidence-carrying change.
    """
    if cudnn.data_type.FLOAT in caps.softmax_precisions:
        return [cudnn.data_type.FLOAT]
    sole = _sole(caps.softmax_precisions)
    # A HALF-only row still never gets HALF proposed — same numerics rule.
    return [None if sole == cudnn.data_type.HALF else sole]


def _cga_points(caps: Capabilities) -> List[Optional[int]]:
    """CGA candidates — every row today declares a single honest point."""
    return [_sole(caps.cgas)]


# ---------------------------------------------------------------------------
# the combiner — complete assignments, Σ growth, never the cartesian product
# ---------------------------------------------------------------------------


def _knob_sets(spec: EngineSpec, facts) -> List[SdpaFwdKnobs]:
    """The cell's ordered COMPLETE knob assignments.

    The baseline takes the best value on every axis; runners-up deviate on ONE
    axis at a time in impact order (tiles, sched, pack_gqa, split) with the
    other axes held at their best, capped at ``_MAX_SETS_PER_ENGINE``. Two
    axes carry a structural coupling: a packed set rides the largest tile
    that admits the ratio, and a split set rides the plain scheduler. Axis
    interactions the kernels cannot serve are the generators'/mismatch's job
    — nothing here multiplies domains together.
    """
    caps = spec.capabilities
    tiles = _tile_points(spec, facts)
    scheds = _sched_points(caps, facts)
    cga = _cga_points(caps)[0]
    # pack_gqa couples with the tile axis: a packed set rides the LARGEST
    # SMEM-fitting tile that admits the ratio — packing fixes the underfill
    # small tiles exist for, and a bigger tile admits a bigger G (G must
    # divide it) — while an unpacked set keeps the tile rule's choice.
    pack_tile = next(
        ((m, n) for m, n in sorted(tiles, key=lambda mn: (-(mn[0] or 0), mn[1] != 128)) if True in _pack_gqa_points(caps, facts, m or 128)),
        None,
    )
    packed_first = pack_tile is not None and _pack_gqa_points(caps, facts, pack_tile[0] or 128)[0]
    unpacked_pack = False if True in caps.pack_gqas else _sole(caps.pack_gqas)
    base_tile = pack_tile if packed_first else tiles[0]
    # The split model sees the launch geometry of the set it rides — the
    # packed grid when the baseline packs.
    splits = _split_points(caps, facts, base_tile[0], base_tile[1], cga, pack_g=(facts.h_q // facts.h_kv) if packed_first else 1)
    base = SdpaFwdKnobs(
        sched_policy=scheds[0],
        tile_m=base_tile[0],
        tile_n=base_tile[1],
        cga=cga,
        pack_gqa=True if packed_first else unpacked_pack,
        split_kv=splits[0],
        softmax_precision=_softmax_points(caps)[0],
    )
    out = [base]
    for tile_m, tile_n in tiles[1:]:
        # A packed baseline's tile runners keep the packing, so tiles the
        # ratio cannot ride are skipped — emitted, they would only spend
        # set-cap slots for mismatch to drop.
        if base.pack_gqa is True and True not in _pack_gqa_points(caps, facts, tile_m or 128):
            continue
        out.append(replace(base, tile_m=tile_m, tile_n=tile_n))
    for policy in scheds[1:]:
        out.append(replace(base, sched_policy=policy))
    # The opposite pack_gqa leg, riding its own tile (packed: the largest
    # admitting tile; unpacked: the tile rule's best).
    if pack_tile is not None:
        if packed_first:
            out.append(replace(base, pack_gqa=unpacked_pack, tile_m=tiles[0][0], tile_n=tiles[0][1]))
        else:
            out.append(replace(base, pack_gqa=True, tile_m=pack_tile[0], tile_n=pack_tile[1]))
    for split in splits[1:]:
        # Split sets ride the plain scheduler: the SM120 config bars a split
        # under the LPT remaps, and in the underfilled regime a split targets
        # the LPT balancing is moot — the split itself levels the grid.
        out.append(replace(base, split_kv=split, sched_policy=SCHED_NATURAL if SCHED_NATURAL in caps.sched_policies else base.sched_policy))
    seen, unique = set(), []
    for knobs in out:
        if knobs not in seen:
            seen.add(knobs)
            unique.append(knobs)
    return unique[:_MAX_SETS_PER_ENGINE]


def _fallback_knobs(caps: Capabilities) -> SdpaFwdKnobs:
    """The config expected to build where the tuned choice may not.

    Today this is the smallest tile the row admits with the plain scheduler
    and no split — the config that asks least of the device, which is the one
    thing a fallback must be.
    """
    return SdpaFwdKnobs(
        sched_policy=SCHED_NATURAL if SCHED_NATURAL in caps.sched_policies else _sole(caps.sched_policies),
        tile_m=min(caps.tile_ms, default=None),
        tile_n=min(caps.tile_ns, default=None),
        cga=_sole(caps.cgas),
        pack_gqa=False if False in caps.pack_gqas else _sole(caps.pack_gqas),
        split_kv=1 if 1 in caps.split_kvs else _sole(caps.split_kvs),
        softmax_precision=_sole(caps.softmax_precisions),
    )


def _eligible(facts, offered: Dict[str, int]) -> Iterator[Tuple[int, EngineSpec]]:
    """(engine_id, spec) for each offered cell whose capability row admits ``facts``."""
    for spec in ENGINE_SPECS:
        engine_id = offered.get(spec.name)
        if engine_id is not None and mismatch(spec.capabilities, facts, None) is None:
            yield engine_id, spec


# ---------------------------------------------------------------------------
# recommend — the pure, backend-blind core (also the standalone entry point)
# ---------------------------------------------------------------------------


def recommend(kind: str, facts, offered: Dict[str, int]) -> List[PlanConfig]:
    """Ordered candidate plans for ``facts`` — no backend, no graph, no modes.

    ``kind`` is ``"A"`` (candidates worth timing, best guess first) or
    ``"FALLBACK"`` (least-demanding configs). Every returned entry carries a
    complete knob assignment validated through ``mismatch(caps, facts, knobs)``
    — honored-or-never-listed — and NO mode. Standalone callers (wrappers,
    autotuners) use this directly: build a ``SdpaGraphFacts``, pass the
    family's ``offered_ids()``, run or time the sets in order.
    """
    out: List[PlanConfig] = []
    for engine_id, spec in _eligible(facts, offered):
        caps = spec.capabilities
        sets = _knob_sets(spec, facts) if kind == "A" else [_fallback_knobs(caps)]
        for knobs in sets:
            if mismatch(caps, facts, knobs) is None:
                out.append(PlanConfig(engine_id, knobs))
    return out


# Placement — mode blocks, the backend's entries, the delegating entry, dedup,
# the mode strip — is NOT this family's business: it happens once for every
# family in ``engines/heuristics._assemble``, with these proposals leading the
# backend's entries inside each block by standing assumption.
