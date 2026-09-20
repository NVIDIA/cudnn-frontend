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
from math import gcd
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import cudnn

from cudnn.engines.base import PlanConfig
from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)
from cudnn.sdpa.fwd.config_sm100 import (
    TemplateParams as Sm100TemplateParams,
    cga_tile_m,
    cga_ctas,
    d192_square_br_as_tl,
    d256_square_br_as_tl,
    decode_d256_q_tile,
    pack_gqa_group_size,
    pack_gqa_supported,
)
from cudnn.sdpa.fwd.config_sm120 import D512_FLAVOR, FP8_HEAD_TILE_GRANULE, HEAD_TILE_GRANULE, SMEM_CAPACITY_BYTES, pick_flavor, smem_bytes, tile_domain
from cudnn.sdpa.fwd.engines import (
    ENGINE_SPECS,
    Capabilities,
    EngineSpec,
    SdpaFwdKnobs,
    _selected_d_shape,
    _synth_kv_padding,
    effective_cgas,
    effective_sched_policies,
    mismatch,
    pack_gqa_partial,
)

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
# Rubin, causal, NO GQA (h_q == h_kv): the LPT-vs-NATURAL crossover in grid WAVES
# (work items per persistent 2-CTA cluster).  Perf node, kernel-level, d192x128
# FP8 H128 causal, NATURAL = 1.00: LPT 1.00 / 1.16 / 0.93 / 0.87 / 0.92 and
# LPT_L2 0.90 / - / 1.00 / - / 0.99 at S = 2K / 4K / 8K / 16K / 32K, i.e. LPT
# pays at 10-19 waves and costs from 39 waves on, LPT_L2 never pays there.
# 256 = the 2-CTA flavors' q rows per cluster; 106 clusters = the 212-SM part / 2.
_SM107_CGA_Q_ROWS = 256
_SM107_CLUSTERS = 106
_SM107_NO_GQA_LPT_MAX_WAVES = 24

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
# A split thinner than this is prologue/epilogue dominated.
_SPLIT_KV_MIN_TILES = 2
# What a CTA-tile costs beyond its KV loop (Q load, prologue, epilogue), in
# units of one KV tile. Empirical: re-measure if the per-tile fixed cost moves.
_SPLIT_KV_CTA_COST = 21.0
# What ONE split's partials cost the combine pass, per wave of combine blocks,
# in units of one KV tile of main-kernel work. The combine's own occupancy
# (blocks/SM of sm100/split_combine) is ABSORBED into this coefficient: it is
# one fixed kernel, so blocks/SM is a constant, and folding it in keeps a
# cuOccupancy query -- which would need a compiled CUfunction -- off the
# planning path. Empirical: re-measure if sm100/split_combine changes.
#
# Re-measured for the fp32 partials the SM100 split kernels now write (was 0.2,
# fitted when partials were half). Widening them turned out to cost the combine
# almost nothing -- 1.05x on a 148-row x 512-split sweep, because the pass is
# not purely bandwidth-bound -- so the move is NOT a consequence of the extra
# bytes. It corrects a coefficient that was too large for the shapes this model
# is asked about: the split kernel also stopped staging O through SMEM and TMA,
# which made splitting cheaper on the main-kernel side.
#
# Two independent fits agree. Timing the combine directly against one KV tile of
# main-kernel work, with the partials large enough to live in HBM rather than
# L2, gives 0.037 (f16) / 0.039 (f32). Minimising regret over the end-to-end
# sweep in test_split_kv_heuristic._B300_FIT gives an optimum PLATEAU of
# [0.04, 0.155] -- every value in it makes the same 12 choices. 0.1 is that
# plateau's midpoint, so it is the value furthest from flipping either way.
_SPLIT_KV_COMBINE_COST = 0.1


class _SplitKvLaunch(NamedTuple):
    q_tiles: int
    heads_q: int
    kv_tiles: int
    ctas_per_tile: int


def split_kv_candidates(*, sm_count: int, kv_tiles: int) -> List[int]:
    """The splits worth scoring on this device, ascending, always starting at 1.

    THE single split-KV list -- what a row can BUILD is a separate boolean
    (``Capabilities.split_kv_supported``), so this is free to be device-derived
    rather than a hand-maintained per-row literal.

    Powers of two from 1 up to ``2**ceil(log2(sm_count))``: you never need more
    CTA-tiles than the machine has SMs, so that is where the occupancy argument
    for splitting runs out. Rounding UP rather than down offers the first
    over-subscribing point and lets the cost model reject it on the wave term,
    instead of the bound pre-judging it. Powers of two because ``split_kv`` is a
    TemplateParams field and so a kernel-module cache key -- an unrestricted
    choice mints a compiled specialization per shape.

    Also bounded by ``kv_tiles // _SPLIT_KV_MIN_TILES``. The chunking hands the
    remainder to the LEADING splits, so the thinnest gets ``floor(kv_tiles/s)``
    tiles, and ``floor(kv_tiles/s) >= m`` is exactly ``s <= floor(kv_tiles/m)``.
    On a short KV that bound binds first.

    No workspace bound here: the partial slabs grow with s, but so does the
    combine term in :func:`choose_split_kv`, and it grows with the Q rows --
    which is what makes the slabs big in the first place. The model self-limits;
    a caller needing a hard ceiling has ``deselect_workspace_greater_than``.
    """
    if sm_count <= 0 or kv_tiles <= 0:
        return [1]
    hi = 1 << max(0, (sm_count - 1).bit_length())  # 2**ceil(log2(sm_count))
    hi = min(hi, max(1, kv_tiles // _SPLIT_KV_MIN_TILES))
    out, s = [], 1
    while s <= hi:
        out.append(s)
        s <<= 1
    return out


def choose_split_kv(
    *,
    q_tiles: int,
    heads_q: int,
    batch: int,
    kv_tiles: int,
    sm_count: int,
    combine_rows: int,
    ctas_per_tile: int = 1,
    candidates: Optional[List[int]] = None,
    unsplit_launch: Optional[_SplitKvLaunch] = None,
) -> int:
    """How many KV chunks to cut each Q tile into; 1 = do not split.

    A prefill launch is ``q_tiles * heads_q * batch`` independent tiles, each
    walking the whole KV loop.  When that product is below the SM count the chip
    idles however long the loop is; splitting multiplies the tile count by ``s``
    and divides each tile's KV work by it, then pays one reduction over the
    partials.

    Splitting runs TWO kernels, so the model is two LATENCIES summed -- each one
    (sequential rounds) x (what one round costs).  Both terms must be latency:
    mixing in an aggregate-work term would double-count the parallelism the wave
    factor has already divided out.

        waves(s)      = ceil(base_ctas * s / sm_count)      # main grid
        combine_waves = ceil(combine_rows / sm_count)       # combine grid, NO s
        cost(s)       = waves(s)      * (ceil(kv_tiles / s) + CTA_COST)
                      + combine_waves * (s * COMBINE_COST)

    CTA_COST is what a tile re-pays whatever its loop length, so it sits INSIDE
    the wave term -- once per CTA-tile, not once per split.  COMBINE_COST is
    outside it: the combine is a separate launch whose grid is ``(S_q, H, B)``
    (sm100/split_combine), one block per output row and independent of ``s`` --
    only the per-block work grows with ``s``, since each block reduces ``s``
    partials.  Hence ``combine_rows`` (= S_q * H_q * B) and not ``base_ctas``.

    Why the combine term matters: ``s`` reaches the first term ONLY through
    ``waves(s)``, a step function.  Between wave boundaries a larger split is
    free there while the loop term keeps falling, so without a second term the
    model always takes the largest split that fits the current wave.  That is
    harmless while the candidate list stops at 4 and a runaway once it does not.

    What falls out: an under-full launch splits until the wave is full; an
    over-full one with a partial-wave tail splits FINER to smooth it, even past
    the SM count; an exactly balanced one (base_ctas = k * sm_count) has no tail
    and never splits; and a long-S_q chunk splits less than a short one, because
    its combine has more rows to reduce.

    ``q_tiles`` / ``heads_q`` / ``kv_tiles`` / ``ctas_per_tile`` describe the
    split leg. ``unsplit_launch`` may describe a different complete no-split
    assignment, as happens when D192 uses CGA1 unsplit but requires CGA2 for a
    split. It defaults to the split geometry, preserving the ordinary case.

    Returns 1 when there is nothing to split or nothing beats not splitting.
    ``candidates`` defaults to :func:`split_kv_candidates` for the device.
    """
    split_launch = _SplitKvLaunch(q_tiles, heads_q, kv_tiles, ctas_per_tile)
    if unsplit_launch is None:
        unsplit_launch = split_launch
    if min(*split_launch, *unsplit_launch, batch, sm_count) <= 0:
        return 1
    if kv_tiles <= 1:
        return 1
    if candidates is None:
        candidates = split_kv_candidates(sm_count=sm_count, kv_tiles=kv_tiles)
    # The combine reads every partial of every output row, so its grid is sized
    # by the rows; max(1, ...) because a decode-shaped launch has fewer rows
    # than SMs and still pays one wave.
    combine_waves = max(1, _ceil_div(max(0, combine_rows), sm_count))

    best_split, best_cost = 1, None
    for split in candidates:
        if split < 1 or split > kv_tiles:
            continue
        # Every split must stay thick enough to amortise its own prologue and
        # epilogue. The chunking hands the remainder to the leading splits, so
        # the THINNEST gets floor(kv_tiles / split) -- that is what must clear.
        if split > 1 and kv_tiles // split < _SPLIT_KV_MIN_TILES:
            continue
        launch = unsplit_launch if split == 1 else split_launch
        base_ctas = launch.q_tiles * launch.heads_q * batch * launch.ctas_per_tile
        waves = _ceil_div(base_ctas * split, sm_count)
        cost = waves * (_ceil_div(launch.kv_tiles, split) + _SPLIT_KV_CTA_COST) + combine_waves * split * _SPLIT_KV_COMBINE_COST
        if best_cost is None or cost < best_cost:
            best_split, best_cost = split, cost
    return best_split


# --- KV split on the d256 decode tile (see choose_decode_tile_split_kv) ------
# sm100/decode_d256_f16.py is not the machine choose_split_kv was fitted on: one
# cta_group::1 CTA per (KV-head group, batch, split) unit streams 128-key K/V
# tiles (128 KiB each at d256 half) with about one tile of fixed cost, and the
# kernel is HBM-bound rather than MMA-bound. Its costs, in units of one KV tile
# streamed by a lone CTA (1.75 us on B200):
#
# The share of the device's SMs whose concurrent K/V streams saturate HBM. Past
# it every CTA slows in proportion to the stream count (the bandwidth is
# shared), which for an HBM-bound kernel is the same statement as "waves".
# Fitted on B200 (148 SMs): 64 CTAs x 32 tiles ran 56.9 us (1.78 us/tile,
# per-CTA-bound), 128 CTAs x 16 tiles 45 us (1.57x per tile), 256 CTAs x 32
# tiles 184 us (3.2x per tile) -- all consistent with ~81 CTAs.
_DECODE_TILE_SATURATION_FRACTION = 0.55
# What a CTA costs beyond its KV loop (Q^T load, prologue, a 16/32-row epilogue).
_DECODE_TILE_CTA_COST = 1.0
# A KV tile's cost to a lone CTA on the 32-column tile (S_q x G in (16, 32]:
# two softmax column groups over the same 128 TMEM lanes, issue-bound where
# the 16-column tile is bandwidth-bound): 64 CTAs x 32 tiles ran 90.2 us
# against the 16-column tile's 57.1 us. HBM saturates at the same byte rate,
# so a slower stream also needs proportionally more CTAs to reach it. That
# tile is compiled but not routed today (the adapter keeps S_q x G in (16, 32]
# on the prefill tile, config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS); its fit
# stays here so routing it is a one-constant change.
_DECODE_TILE_WIDE_Q_TILE_COST = 1.6
# The shared sm100/split_combine pass over a decode-shaped grid: one combine
# wave, ~6 us measured at b=32 x 32 heads.
_DECODE_TILE_COMBINE_COST = 3.5
# What the two-launch split path costs an EAGER caller on the host per
# graph.execute beyond the one-launch path: a second CuTe-DSL launch plus the
# partial-slab carving. Measured 62 -> 92 us on the b=32 x 2 KV heads x 4096
# keys serving shape (Python launch path, B200 host), against a 6 us GPU
# saving. A CUDA-graph replay pays none of it, but a plan cannot know whether
# it will be captured, so the default CHARGES it -- a split leads only where
# its GPU saving also covers the eager caller's extra host time -- and the
# captured caller's optimum (this term at 0) is listed as the runner-up plan.
_DECODE_TILE_SPLIT_LAUNCH_COST = 17.0


def _decode_tile_cost(split: int, *, units: int, kv_tiles: int, sm_count: int, tile_cost: float, launch_cost: float) -> float:
    streams = units * split
    per_tile = max(tile_cost, streams / (_DECODE_TILE_SATURATION_FRACTION * sm_count))
    cost = (_ceil_div(kv_tiles, split) + _DECODE_TILE_CTA_COST) * per_tile
    if split > 1:
        cost += _DECODE_TILE_COMBINE_COST + launch_cost
    return cost


def choose_decode_tile_split_kv(
    *,
    units: int,
    kv_tiles: int,
    sm_count: int,
    q_tile: int = 16,
    launch_cost: float = _DECODE_TILE_SPLIT_LAUNCH_COST,
) -> int:
    """How many KV chunks the d256 decode tile cuts each unit into; 1 = do not split.

    ``units`` is the launch's CTA count before splitting -- batch x KV-head
    groups (a packed group is one unit, an unpacked head one each);
    ``kv_tiles`` the 128-key tiles of the declared S_kv; ``q_tile`` the tile's
    N extent (16 or 32 packed Q rows, :func:`config_sm100.decode_d256_q_tile`).

        streams(s)  = units * s                                   # concurrent K/V streams
        per_tile(s) = max(TILE_COST(q_tile), streams(s) / (SATURATION * sm_count))
        cost(s)     = (ceil(kv_tiles / s) + CTA_COST) * per_tile(s)
                    + [s > 1] * (COMBINE_COST + launch_cost)

    A lone CTA streams a KV tile at TILE_COST (1 on the 16-column tile, 1.6 on
    the issue-bound 32-column one); once the streams saturate HBM every CTA
    slows with their count instead.  What falls out: an under-full launch
    splits until its streams saturate HBM; a launch already past it never
    splits (the bytes are the same, only the fixed costs grow); in between,
    the split must SAVE more than it ADDS.  With the default ``launch_cost``
    the added part includes the eager caller's second host launch, so on the
    16-column tile the b=32 x 2 KV heads x 4096-key serving shape (6 us of GPU
    saving for 30 us of host time) stays unsplit while a small batch or a
    long KV (b=8; s_kv=16384 at b=32) still splits, and the 32-column tile --
    whose lone CTA is slow enough that the same shape saves ~32 us -- would
    split (it is not routed today; see _DECODE_TILE_WIDE_Q_TILE_COST).
    ``launch_cost=0`` is the optimum of a caller replaying a captured CUDA
    graph.

    Candidates come from :func:`split_kv_candidates`; ties go to the smaller
    split. Returns 1 for degenerate inputs.
    """
    if min(units, kv_tiles, sm_count) <= 0:
        return 1
    tile_cost = _DECODE_TILE_WIDE_Q_TILE_COST if q_tile > 16 else 1.0
    kw = dict(units=units, kv_tiles=kv_tiles, sm_count=sm_count, tile_cost=tile_cost, launch_cost=launch_cost)
    best, best_cost = 1, _decode_tile_cost(1, **kw)
    for split in split_kv_candidates(sm_count=sm_count, kv_tiles=kv_tiles):
        if split <= 1:
            continue
        cost = _decode_tile_cost(split, **kw)
        if cost < best_cost:
            best, best_cost = split, cost
    return best


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
    preferred_tile_m = 64 if fine else 128
    # FP8 stages a byte per KV element but still writes O in half, so the two
    # SMEM terms size differently -- see config_sm120.smem_bytes. The kernel
    # stages ENVELOPE head tiles (actual dims round up to the granule), so
    # the fit check must round the same way or a tile offered here declines
    # at build -- and a knob-carried tile skips the adapter's fallback.
    qkv_itemsize, o_itemsize = (1, 2) if facts.is_fp8 else (2, 2)
    granule = FP8_HEAD_TILE_GRANULE if facts.is_fp8 else HEAD_TILE_GRANULE
    d_qp = -(-facts.d_qk // granule) * granule
    d_vp = -(-facts.d_v // granule) * granule
    # Prefer the selected query tile, then larger query and KV tiles.
    candidate_tiles = sorted(
        tile_domain(facts.d_qk, facts.d_v, facts.is_fp8),
        key=lambda tile: (tile[0] == preferred_tile_m, tile[0], tile[1]),
        reverse=True,
    )
    for m, n in candidate_tiles:
        if m not in caps.tile_ms or n not in caps.tile_ns:
            continue
        if smem_bytes(d_qp, d_vp, m, n, qkv_itemsize, o_itemsize) <= SMEM_CAPACITY_BYTES:
            return m, n
    # Nothing fits: fall back to the smallest tile the kernel table admits for
    # the row, so the decline comes from the SMEM check rather than at build.
    admitted = [(m, n) for m, n in candidate_tiles if m in caps.tile_ms and n in caps.tile_ns] or candidate_tiles
    return min(admitted, key=lambda tile: (tile[1], tile[0]))


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
    tile_choices = tile_domain(facts.d_qk, facts.d_v, facts.is_fp8)
    domain = [
        (m, n)
        for m in caps.tile_ms
        for n in caps.tile_ns
        if (m, n) in tile_choices and smem_bytes(d_qp, d_vp, m, n, qkv_itemsize, o_itemsize) <= SMEM_CAPACITY_BYTES
    ]
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
    # The FLAVOR's domain, not the row-wide floor: a `sched_policies_by_d_shape`
    # claim (the Rubin f16 / FP8 (256, 256) LPT entries) must reach the ranking,
    # or LPT is only ever honoured when a caller REQUESTS the knob and is never
    # proposed for the first plan -- which is the whole point of claiming it.
    domain = effective_sched_policies(caps, facts)
    if len(domain) <= 1:
        return [_sole(domain)]
    if facts.thd and SCHED_NATURAL in domain:
        # A ragged batch carries its own scheduler: it walks the LIVE units
        # through batch_remap over a machine-sized grid. The LPT decodes map a
        # linear tile id onto a dense rectangular tile space, so ranking them
        # here would hand THD a decode built for a geometry it does not have --
        # and spend autotune slots on it. Same exclusion the adapters apply to
        # their standalone-wrapper derivation.
        return [SCHED_NATURAL]
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
    elif causal_ish and _sm120_d512_windowed(caps, facts):
        # Sliding window on the d512 flavor: a unit's K/V working set is a few
        # tiles whichever head it belongs to, so LPT_L2's per-KV-group batching
        # has nothing to protect in L2 and its row order only costs; the plain
        # heads-fastest LPT walk is the faster one for packed and unpacked units.
        primary = SCHED_LPT
    elif causal_ish and 107 <= caps.sm_lo < 120 and int(facts.h_q) == int(facts.h_kv):
        # Rubin (cc 10.7-11.x) without GQA: every KV head is read by exactly one Q head, so
        # LPT_L2 has no K/V sharing to group -- and it is not free (-10 % at
        # S=2K, see the constants above).  Plain LPT balances the triangle
        # while the grid is a few waves and costs at many, so choose by wave
        # count.  GQA shapes keep the L2 rule below (llama d128 H64/8 on the
        # same node: LPT_L2 +4..8 % over NATURAL at every S).  Rubin only until
        # the SM100 line is measured the same way.
        # The bound is the arch LINE, not `>= 107`: the SM120 rows sit at sm_lo=120 and
        # keep the SM100/SM120 L2-budget rule below (the wave-count constants above
        # are Rubin's cluster count and CGA rows, unmeasured on GeForce Blackwell).
        waves = (int(facts.b) * int(facts.h_q) * -(-int(facts.s_q) // _SM107_CGA_Q_ROWS)) / _SM107_CLUSTERS
        primary = SCHED_LPT if waves <= _SM107_NO_GQA_LPT_MAX_WAVES else SCHED_NATURAL
    elif (
        causal_ish
        and caps.sm_lo == 100
        and not (facts.is_fp8 or facts.is_mxfp8)
        and _selected_d_shape(caps, facts) == (128, 128)
        and 1 in effective_cgas(caps, facts)
        and _d128_decode_tile_fits(caps, facts)
    ):
        # The d128 decode tile (bottom-right causal MTP, S_q * PACK_G <= 128):
        # one Q tile per (packed head, batch), so every unit walks the same
        # per-batch KV range and LPT has nothing to balance; LPT_L2's head
        # grouping groups nothing when the packed head IS the KV head (only
        # the G / PACK_G packed heads of a partially packed group).  Measured on
        # B200 (b=32, H=64/4, S_q=4, S_kv=4096, page 16): NATURAL 120.0 us vs
        # LPT_L2 125.4 us on the prefill tile; the decode tile keeps the order.
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
    # The primary may be outside a row's DOMAIN (the SM107 f16 rows and the
    # d256 / d512 flavors carry no SCHED_LPT_L2 -- their kernels do not thread
    # its decode inputs).  Fall back along the
    # SAME preference order rather than to NATURAL: dropping straight to NATURAL
    # cost a causal Rubin FP8 graph the LPT load-balancing win, and listed
    # NATURAL twice ([0, 1, 0]), burning an autotune slot on a duplicate plan.
    # When the primary IS in domain this is byte-identical to the old form
    # (order[primary] never contains primary).
    chosen = next((p for p in (primary, *order[primary]) if p in domain), SCHED_NATURAL)
    runners = [p for p in order[primary] if p in domain and p != chosen]
    # A mask-free graph gains nothing from either LPT remap — the grid is
    # already balanced — so don't spend autotune slots on them.
    if not causal_ish and facts.window_left is None:
        runners = []
    return [chosen, *runners]


def select_d192_auto_knobs(
    params: Sm100TemplateParams,
    *,
    pertensor: bool,
    s_q: int,
    s_kv: int,
) -> tuple[int, int]:
    """Select the measured D192 scheduler and CGA defaults.

    This function is shared by graph heuristics and standalone callers. It
    chooses public knobs only; lowering canonicalizations and private codegen
    fields are derived separately in ``config_sm100``.
    """

    if params.split_kv > 1:
        return SCHED_NATURAL, 2

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    mxfp8 = fp8 and not pertensor
    window_left = params.window_left
    window_right = params.window_right
    top_left = not params.bottom_right or d192_square_br_as_tl(params, s_q=s_q, s_kv=s_kv)

    mx_dense_mid_causal_cga1 = (
        mxfp8
        and not params.thd_varlen
        and window_left is None
        and window_right == 0
        and top_left
        and 4096 < s_kv <= 8192
        and (params.dtype_qkv == DTYPE_E5M2 or s_q >= 4096)
    )
    sched_policy = SCHED_NATURAL if mx_dense_mid_causal_cga1 else params.sched_policy

    pt_cga1 = (
        pertensor
        and not params.thd_varlen
        and (
            window_left is not None
            or (params.dtype_qkv == DTYPE_E5M2 and window_right is None)
            or (params.dtype_qkv == DTYPE_E4M3 and params.dtype_o in (DTYPE_E4M3, DTYPE_E5M2) and window_left is None and window_right == 0 and top_left)
        )
    )

    mx_cga1 = False
    if mxfp8:
        masked = window_right is not None
        sliding = window_left is not None
        if params.thd_varlen:
            if params.dtype_qkv == DTYPE_E5M2 and not masked:
                mx_cga1 = True
            elif masked and sliding:
                min_s_kv = 4096 if params.dtype_qkv == DTYPE_E4M3 else 2048
                mx_cga1 = s_kv >= min_s_kv
            elif masked:
                mx_cga1 = s_kv >= 2048
        elif masked:
            mx_cga1 = params.dtype_qkv == DTYPE_E4M3 or sliding or s_kv <= 4096
    mx_cga1 = mx_cga1 or mx_dense_mid_causal_cga1
    return sched_policy, 1 if pt_cga1 or mx_cga1 else 2


def select_d256_auto_knobs(
    params: Sm100TemplateParams,
    *,
    pertensor: bool,
    s_q: int,
    s_kv: int,
) -> tuple[int, int]:
    """Select the measured D256 scheduler and fixed FP8 CTA1 geometry."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if not fp8:
        return params.sched_policy, 2
    if params.split_kv > 1 or params.thd_varlen:
        return SCHED_NATURAL, 1

    no_mask = params.window_left is None and params.window_right is None and not params.seq_kv_lens_present
    if no_mask:
        return SCHED_NATURAL, 1

    top_left = not params.bottom_right or d256_square_br_as_tl(params, s_q=s_q, s_kv=s_kv)
    pt_lpt_l2 = (
        pertensor
        and params.window_left is None
        and params.window_right is not None
        and not params.seq_kv_lens_present
        and top_left
        and _ceil_div(s_q, 256) >= 16
    )
    if pt_lpt_l2:
        return params.sched_policy, 1
    return SCHED_LPT, 1


def select_d512_auto_knobs(params: Sm100TemplateParams) -> tuple[int, int]:
    """Select the measured D512 scheduler and fixed MXFP8 CTA1 geometry."""

    if params.thd_varlen:
        return SCHED_NATURAL, 1
    if params.window_right is not None:
        return SCHED_LPT, 1
    return params.sched_policy, 1


def _sm100_params_from_facts(facts, *, split_kv: int, sched_policy: int) -> Sm100TemplateParams:
    dtype_codes = {
        cudnn.data_type.FP8_E4M3: DTYPE_E4M3,
        cudnn.data_type.FP8_E5M2: DTYPE_E5M2,
        cudnn.data_type.BFLOAT16: DTYPE_BF16,
        cudnn.data_type.HALF: DTYPE_FP16,
    }
    return Sm100TemplateParams(
        dtype_qkv=dtype_codes[facts.dtype],
        dtype_o=dtype_codes.get(facts.dtype_o, -1),
        window_left=facts.window_left,
        window_right=(facts.right_bound if facts.right_band_widening else 0 if facts.causal else None),
        bottom_right=facts.bottom_right,
        seq_kv_lens_present=facts.padded,
        seq_q_lens_present=facts.seq_q_trim,
        sched_policy=sched_policy,
        thd_varlen=facts.thd,
        split_kv=split_kv,
    )


# The d128 decode tile's Q rows per CTA (config_sm100.CfgD128Decode: TILES_Q=1 x
# TILE_M=128 x CTA_MMA=1).  The SM100 f16 row's (128, 128) flavor has two
# tiles behind TILE_CGA_M: cga2 is the prefill pipeline (512 rows per
# cluster), cga1 the decode tile (sm100/decode_d128_f16.py).
_D128_DECODE_TILE_ROWS = 128


def _d128_decode_tile_fits(caps: Capabilities, facts, pack_gqa: Optional[bool] = None) -> bool:
    """Whether one d128 decode tile covers a KV head's live Q rows.

    ``S_q * pack_g <= 128`` with ``pack_g`` the CANDIDATE's own packing:
    ``pack_gqa=True`` is the packed leg (one unit carries the packed group
    ``p = Cfg.PACK_G`` -- the whole GQA group ``G`` when it divides the tile,
    else its largest divisor that does (partial PackGQA: 96/8 packs 4) --
    ``p`` rows per token), ``False`` the unpacked one (one head, ``S_q`` rows),
    and ``None`` -- the graph-level question -- reads as the packed leg when
    the row can pack this graph, which is the leg a decode-shaped graph
    proposes first.  The fit is a property of the candidate, not the graph:
    at ``S_q * p > 128 >= S_q`` the packed leg keeps the prefill tile while
    the unpacked runner-up rides the decode tile.  Dense only: the decode tile
    has no THD leg.  Measured on B200 (b=32, H=64/4, d128, S_kv=4096, page 16,
    bf16): the prefill tile at cga2 119 us, the decode tile 49 us -- see the
    kernel docstring; at S_q * G > 128 the prefill tile's second sub-tile is
    live and cga2's collective MMA halves per-CTA K/V traffic, so the rule
    stops there rather than at a measured crossover.
    """
    if facts.thd:
        return False
    if pack_gqa is None:
        pack_gqa = _pack_gqa_eligible(caps, facts, _D128_DECODE_TILE_ROWS)
    # The kernel's HEADS_PER_TILE for this leg (1 unpacked): the launch the
    # split model sees, and the rows one unit really carries.
    pack_g = _pack_gqa_group(caps, facts, _D128_DECODE_TILE_ROWS, pack_gqa)
    return facts.s_q * pack_g <= _D128_DECODE_TILE_ROWS


def _auto_sched_cga(spec: EngineSpec, facts, *, split_kv: int, sched_policy: int, pack_gqa: Optional[bool] = None) -> tuple[int, Optional[int]]:
    """``pack_gqa`` is the candidate's packing where the width depends on it
    (the d128 f16 SM100 flavor, :func:`_d128_decode_tile_fits`); ``None`` asks
    the graph-level question.  The other flavors' rules ignore it."""
    caps = spec.capabilities
    domain = effective_cgas(caps, facts, split_kv)
    selected_shape = _selected_d_shape(caps, facts)
    if selected_shape == (128, 128) and domain == frozenset({1, 2}) and not (facts.is_fp8 or facts.is_mxfp8):
        # The f16 SM100 row: cga1 = the decode tile when one of its 128-row
        # tiles covers the head's Q rows, else the cga2 prefill pipeline.
        return sched_policy, (1 if _d128_decode_tile_fits(caps, facts, pack_gqa) else 2)
    if selected_shape == (256, 256) and any(shape == selected_shape for shape, _ in caps.cgas_by_d_shape):
        params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
        selected_sched, selected_cga = select_d256_auto_knobs(params, pertensor=facts.is_fp8, s_q=facts.s_q, s_kv=facts.s_kv)
        if selected_cga not in domain:
            raise ValueError(f"D256 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
        return selected_sched, selected_cga
    if selected_shape == (512, 512) and facts.is_mxfp8 and caps.sm_lo == 100:
        params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
        selected_sched, selected_cga = select_d512_auto_knobs(params)
        if selected_cga not in domain:
            raise ValueError(f"D512 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
        return selected_sched, selected_cga
    if selected_shape != (192, 128) or not any(shape == (192, 128) for shape, _ in caps.cgas_by_d_shape):
        return sched_policy, _sole(domain)
    params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
    selected_sched, selected_cga = select_d192_auto_knobs(params, pertensor=facts.is_fp8, s_q=facts.s_q, s_kv=facts.s_kv)
    if selected_cga not in domain:
        raise ValueError(f"D192 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
    return selected_sched, selected_cga


# --- pack_gqa (GQA head packing) --------------------------------------------


def _pack_gqa_wins(facts, tile_q: int) -> bool:
    """Pack when the Q sequence cannot fit in a single tile, then we can further
    apply split_kv on top of GQA packing.

    TODO: we may enhance this heuristic logic in the future by considering more
    factors such as the device SM count.
    """
    return facts.s_q < tile_q


def _pack_gqa_tile_q(caps: Capabilities, facts, tile_m: Optional[int], cga: Optional[int] = None) -> int:
    """The Q rows one grid tile covers, for :func:`_pack_gqa_wins`.

    The SM100 family runs CGA tiles. D192 accepts CGA1 and CGA2, so callers must
    pass the CGA of the complete assignment they are evaluating. SM120 launches
    one CTA per tile, so it is ``tile_m`` itself.
    """
    if caps.sm_lo >= 120 and caps.sm_hi < 130:
        return tile_m or 128
    if facts.d_qk <= 128 and facts.d_v <= 128:
        return cga_tile_m(128, cga)
    if facts.d_qk <= 192 and facts.d_v <= 128:
        return cga_tile_m(192, cga)
    if facts.d_qk <= 256 and facts.d_v <= 256:
        return cga_tile_m(256, cga)
    return cga_tile_m(512, cga)


def _sm120_d512_windowed(caps: Capabilities, facts) -> bool:
    """A sliding-window graph on the SM120 d512 flavor.

    Two rules key on it. Pack the GQA group into the Q tile: a packed unit holds
    ``tile_m / G`` tokens, so its key span is that many tokens plus the window
    instead of ``tile_m`` plus the window, fewer K/V tiles through the L2->SMEM
    path and less masked-out MMA at the same DRAM bytes. And walk the units with
    plain LPT (see :func:`_sched_points`). Without a window the decode rule
    alone decides the packing. The FP8 flavor takes only the LPT walk: its
    64-key tile covers a 64-token unit's window in as many tiles as a packed
    one-token unit, so packing saves no MMA there (measured 8-11% slower).
    """
    return caps.sm_lo >= 120 and caps.sm_hi < 130 and facts.window_left is not None and pick_flavor(facts.d_qk, facts.d_v, fp8=facts.is_fp8) == D512_FLAVOR


def _d256_decode_tile_selected(caps: Capabilities, facts, pack_g: int) -> bool:
    """Whether the SM100 f16/bf16 row lowers this graph onto the d256 decode tile
    (sm100/decode_d256_f16.py) -- the twin of ``SdpaFwdDslSm100._decode_q_tile``:
    the (256, 256) flavor, half inputs, dense (not THD), S_q x packed heads
    within the tile's N extent, ``pack_g`` being the DECODE tile's group
    (:func:`_decode_tile_pack_g`).  Rubin has its own row (no decode tile)."""
    return (
        caps.sm_lo == 100
        and caps.sm_hi < 107
        and not facts.is_fp8
        and not facts.is_mxfp8
        and not facts.thd
        and _selected_d_shape(caps, facts) == (256, 256)
        and decode_d256_q_tile(facts.s_q, pack_g) > 0
    )


def _decode_tile_pack_g(facts, pack_g: int) -> int:
    """The heads the DECODE tile packs per token for a set whose prefill-tile
    group is ``pack_g`` (:func:`_pack_gqa_group`; 1 = unpacked): the whole GQA
    ratio.  sm100/decode_d256_f16.py packs ``HEADS_PER_TILE = QH_PER_KH`` --
    one unit per (KV head, batch) with every head of the group in its 16-column
    tile (96/8: 12 live rows, four zero-filled) -- and has no partial form, so
    the prefill tile's ``gcd(G, 128)`` (4 for 96/8, partial PackGQA) is not
    this launch's geometry: fed that, S_q = 2 x 96/8 (24 packed rows, the
    prefill tile's graph -- the adapter counts S_q x G) would read as an
    8-row decode launch and be costed with the decode model."""
    return (facts.h_q // facts.h_kv) if pack_g > 1 else 1


def _pack_gqa_eligible(caps: Capabilities, facts, tile_m: int) -> bool:
    """Whether a packed set can be built at ``tile_m``: the row offers packing,
    the batch is dense, the graph carries no fused epilogue gate (its per-head
    gate tile cannot address a packed tile's interleaved rows -- mismatch()
    declines the same pair), there is a group to pack and the ratio divides
    the tile -- or, on a flavor with partial PackGQA, shares a factor with it
    (96/8 packs 4 of its 12 heads; 24/8 has nothing to pack and stays unpacked)."""
    return (
        True in caps.pack_gqas
        and not facts.thd
        and not facts.has_epilogue_gate
        and facts.h_q != facts.h_kv
        and pack_gqa_supported(facts.h_q, facts.h_kv, tile_m, partial=pack_gqa_partial(caps, facts))
    )


def _pack_gqa_group(caps: Capabilities, facts, tile_m: Optional[int], packed: Optional[bool]) -> int:
    """The heads one packed Q tile row-group holds for a ``pack_gqa=packed``
    set: 1 unpacked, else ``Cfg.PACK_G`` -- the whole ratio G when it divides
    the tile, its largest divisor that does on a partial-PackGQA flavor (96/8
    -> 4).  The launch geometry the wave-cost model must see is the PACKED
    one: ``h_q // p`` packed heads of ``s_q * p`` rows each -- feeding it G
    where the kernel packs p (96/8: 8 heads instead of 24) shrinks the
    apparent grid 3x and over-proposes the split at mid batch sizes."""
    if not packed:
        return 1
    return pack_gqa_group_size(facts.h_q // facts.h_kv, tile_m or 128, partial=pack_gqa_partial(caps, facts))


def _pack_gqa_points(caps: Capabilities, facts, tile_m: int, cga: Optional[int] = None) -> Tuple[bool, ...]:
    """The pack_gqa axis, best first: ``(True, False)`` when packing wins,
    ``(False, True)`` when it is only eligible, ``(False,)`` when it is not."""
    if not _pack_gqa_eligible(caps, facts, tile_m):
        return (False,)
    if _pack_gqa_wins(facts, _pack_gqa_tile_q(caps, facts, tile_m, cga)) or (_sm120_d512_windowed(caps, facts) and not facts.is_fp8):
        return (True, False)
    return (False, True)


def _sm100_f16(caps: Capabilities, facts) -> bool:
    return caps.sm_lo == 100 and caps.sm_hi < 107 and facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16)


def _swa_kv_tiles(facts, *, token_span: int, tile_n: int) -> int:
    """Maximum issued KV range of this candidate's Q clusters under SWA.

    Mirrors the range, not the element mask: the last Q cluster retains its
    full token span, including padding, just as compute_kv_loop_bounds does.
    Distinct Q-start residues repeat after tile_n/gcd(token_span,tile_n)
    clusters. Within one residue, both unclipped bounds move by whole tiles;
    their clipped intersection is largest at one of the two integer points
    nearest its center. Thus planning never scans an unbounded Q sequence.
    """
    kv_tiles = _ceil_div(facts.s_kv, tile_n)
    if facts.window_left is None:
        return kv_tiles
    causal = facts.causal or facts.right_band_widening
    right = facts.right_bound if facts.right_band_widening else 0
    if facts.padded and facts.bottom_right:
        # Device KV lengths change the diagonal's tile residue. Bound every
        # possible alignment without reading a device value on the host.
        return min(kv_tiles, _ceil_div(token_span + facts.window_left + right + tile_n - 1, tile_n)) if causal else kv_tiles
    diagonal = facts.s_kv - facts.s_q if facts.bottom_right else 0
    if not causal:
        return max(0, kv_tiles - max(0, (diagonal - facts.window_left) // tile_n))
    q_tiles = _ceil_div(facts.s_q, token_span)
    period = tile_n // gcd(token_span, tile_n)
    step = period * token_span // tile_n
    longest = 0
    for residue in range(min(period, q_tiles)):
        lo = (residue * token_span + diagonal - facts.window_left) // tile_n
        hi = _ceil_div(residue * token_span + diagonal + token_span + right, tile_n)
        last = (q_tiles - 1 - residue) // period
        center = max(0, min(last, (kv_tiles - lo - hi) // (2 * step)))
        for index in (center, min(last, center + 1)):
            longest = max(longest, min(kv_tiles, hi + index * step) - max(0, lo + index * step))
    return longest


def _split_launch(caps: Capabilities, facts, tile_m, tile_n, cga, pack_g: int) -> _SplitKvLaunch:
    rows = _pack_gqa_tile_q(caps, facts, tile_m, cga)
    kv_tiles = _ceil_div(facts.s_kv, tile_n or 128)
    ctas = cga or 1
    if _sm100_f16(caps, facts):
        kv_tiles = _swa_kv_tiles(facts, token_span=rows // pack_g, tile_n=tile_n or 128)
        ctas = cga_ctas(_selected_d_shape(caps, facts)[0], cga)
    return _SplitKvLaunch(_ceil_div(facts.s_q * pack_g, rows), facts.h_q // pack_g, kv_tiles, ctas)


def _split_points(
    caps: Capabilities,
    facts,
    tile_m: Optional[int],
    tile_n: Optional[int],
    cga: Optional[int],
    pack_g: int = 1,
    *,
    unsplit_knobs: Optional[SdpaFwdKnobs] = None,
) -> List[Optional[int]]:
    """Ordered split-KV candidates for the chosen tile geometry.

    ``pack_g`` is the packed group of the set the split rides (1 = unpacked;
    :func:`_pack_gqa_group` -- the kernel's ``Cfg.PACK_G``, which is the GQA
    ratio G when it divides the tile and a proper divisor of it under partial
    PackGQA): packing multiplies each packed head's Q rows by ``pack_g`` and
    divides the head count by it, so the wave-cost model must see the PACKED
    launch — the packed grid is smaller, which is exactly when splitting pays.

    The value comes from :func:`choose_split_kv`'s wave-cost model, fed the
    EXACT launch geometry via :func:`_pack_gqa_tile_q` — the Q rows one grid
    tile covers, which on SM100 is the cluster's ``TILES_Q*TILE_M*CTA_MMA``.
    ``unsplit_knobs`` lets the no-split candidate carry a different complete
    geometry; D192 requires this because split-KV is CGA2-only while its tuned
    unsplit assignment may use CGA1. The generator respects structural limits
    (dense-only, no sink — mismatch() enforces the same, so an emitted >1
    never reaches a kernel that cannot honor it).

    A split the model asks for LEADS, with no-split behind it as the runner-up
    — so a plain ``build_plans()`` runs the split, and autotune / select_plan
    can still reach the unsplit plan. Emitting it the other way round meant the
    default build never used the split the model had just computed.
    """
    no_split = 1
    if not caps.split_kv_supported:
        return [no_split]
    # Paged KV is padded by construction and the split composes with the
    # per-batch lengths (it IS the decode lever there) — see mismatch().
    if facts.thd or facts.has_sink or (facts.padded and not facts.has_paged_kv) or facts.seq_q_trim:
        return [no_split]
    if facts.has_epilogue_gate:
        # The fused O * sigmoid(G) epilogue lives in the unsplit kernel; the
        # combine would write the un-gated O (mismatch declines the same pair,
        # so this is hygiene: never PROPOSE a knob the row cannot honour).
        return [no_split]
    if _synth_kv_padding(caps, facts):
        # This S_kv would be served through the synthesized KV-tail padding,
        # which the split cannot ride (mismatch declines the same combination,
        # through the same predicate the lowering uses). A paged graph never
        # takes that path — its per-batch lengths bound the walk on device —
        # so a declared max that is not a tile multiple keeps its split.
        return [no_split]
    # A quantized O is a legal split target: the partials stay WIDER than the
    # O dtype whatever it is, and the combine performs the only cast down to it.
    sm_count = facts.device_sm_count or 0
    if sm_count <= 0:
        return [no_split]
    decode_pack_g = _decode_tile_pack_g(facts, pack_g)
    if _d256_decode_tile_selected(caps, facts, decode_pack_g):
        # The decode tile is a different machine from the one the prefill model
        # below was fitted on (one cta_group::1 CTA per (KV-head group, batch,
        # split) unit, HBM-bound, ~1 tile of fixed cost -- not cga2 clusters
        # with a 21-tile cost), so it has its own model.  The LEADING entry is
        # the choice that also pays for the split path's second host launch
        # (charged as if serialized with the GPU work -- SUPPORT_MATRIX_TRACKER.md
        # footnote d has the measured eager and replay numbers); the captured
        # caller's optimum, when it differs, is the runner-up; no-split closes
        # the list as usual.
        geometry = dict(
            units=facts.b * (facts.h_q // decode_pack_g),
            kv_tiles=_swa_kv_tiles(facts, token_span=decode_d256_q_tile(facts.s_q, decode_pack_g) // decode_pack_g, tile_n=tile_n or 128),
            sm_count=sm_count,
            q_tile=decode_d256_q_tile(facts.s_q, decode_pack_g),
        )
        eager = choose_decode_tile_split_kv(**geometry)
        captured = choose_decode_tile_split_kv(**geometry, launch_cost=0.0)
        points = [eager]
        for split in (captured, no_split):
            if split not in points:
                points.append(split)
        return points
    split_launch = _split_launch(caps, facts, tile_m, tile_n, cga, pack_g)
    unsplit_launch = None
    if unsplit_knobs is not None:
        unsplit_pack_g = _pack_gqa_group(caps, facts, unsplit_knobs.tile_m, unsplit_knobs.pack_gqa)
        unsplit_launch = _split_launch(caps, facts, unsplit_knobs.tile_m, unsplit_knobs.tile_n, unsplit_knobs.cga, unsplit_pack_g)
    split = choose_split_kv(
        q_tiles=split_launch.q_tiles,
        heads_q=split_launch.heads_q,
        batch=facts.b,
        kv_tiles=split_launch.kv_tiles,
        sm_count=sm_count,
        # The combine's grid is (S_q, H, B) — the REAL head count, not the
        # packed one: packing folds heads into Q rows for the main kernel, but
        # the combine still reduces one block per (row, head, batch) of the
        # graph's own output.
        combine_rows=facts.s_q * facts.h_q * facts.b,
        ctas_per_tile=split_launch.ctas_per_tile,
        unsplit_launch=unsplit_launch,
    )
    if split <= 1:
        return [no_split]
    return [split, no_split]


# NOTE: the softmax accumulator precision is NOT a knob axis. It changes
# numerics (the Rubin f16x2 exponent arm), so it is the
# sdpa(softmax_precision=) op attribute: a graph FACT that engines.mismatch
# gates against Capabilities.softmax_precisions. Heuristics never propose it —
# auto-proposing it was the CUDNN_SOFTMAX_PRECISION environment-knob failure
# mode this vocabulary exists to avoid.


# ---------------------------------------------------------------------------
# the combiner — complete assignments, Σ growth, never the cartesian product
# ---------------------------------------------------------------------------


def _knob_sets(spec: EngineSpec, facts) -> List[SdpaFwdKnobs]:
    """The cell's ordered COMPLETE knob assignments.

    The baseline takes the best value on every axis; runners-up deviate on ONE
    geometry at a time (tiles, CGA, sched, pack_gqa, split), recomputing split
    for SM100 f16 geometry runners, capped at ``_MAX_SETS_PER_ENGINE``. Two
    axes carry a structural coupling: a packed set rides the largest tile
    that admits the ratio, and a split set rides the plain scheduler. Axis
    interactions the kernels cannot serve are the generators'/mismatch's job
    — nothing here multiplies domains together.
    """
    caps = spec.capabilities
    tiles = _tile_points(spec, facts)
    generic_scheds = _sched_points(caps, facts)
    unpacked_pack = False if True in caps.pack_gqas else _sole(caps.pack_gqas)
    # A split set rides the plain scheduler: the SM120 config bars a split under
    # the LPT remaps, and in the underfilled regime a split targets, LPT
    # balancing is moot — the split itself levels the grid. The coupling is
    # structural, so it binds whichever leg leads; it cannot live only on the
    # runner-up loop or a leading split would inherit the derived LPT policy.
    plain_sched = SCHED_NATURAL if SCHED_NATURAL in caps.sched_policies else generic_scheds[0]

    unsplit_sched, _ = _auto_sched_cga(spec, facts, split_kv=1, sched_policy=generic_scheds[0])
    scheds = [unsplit_sched] + [policy for policy in generic_scheds if policy != unsplit_sched]

    def _pack_choice(cga: Optional[int]):
        # A packed set rides the largest fitting tile that admits the ratio. The
        # decision uses this leg's actual CGA span; D192 CGA1 and CGA2 cover a
        # different number of Q rows.
        pack_tile = next(
            ((m, n) for m, n in sorted(tiles, key=lambda mn: (-(mn[0] or 0), mn[1] != 128)) if True in _pack_gqa_points(caps, facts, m or 128, cga)),
            None,
        )
        packed_first = pack_tile is not None and _pack_gqa_points(caps, facts, pack_tile[0] or 128, cga)[0]
        return (pack_tile if packed_first else tiles[0]), packed_first, pack_tile

    def _leg(split_value: int) -> SdpaFwdKnobs:
        seed_sched = plain_sched if split_value > 1 else scheds[0]
        sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=split_value, sched_policy=seed_sched)
        base_tile, packed_first, _ = _pack_choice(cga)
        pack_gqa = True if packed_first else unpacked_pack
        if pack_gqa is not True:
            # The width above was judged on the packed leg's rows (the graph-
            # level question); a leg that runs unpacked is re-judged on one
            # head's S_q rows -- the d128 decode-tile fit belongs to the
            # candidate, not the graph.  _pack_choice does not move under the
            # new width: it depends on cga only through _pack_gqa_wins, which
            # is settled the same way at either width for any S_q this can
            # change (an eligible group that does not win has S_q >= 512).
            sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=split_value, sched_policy=seed_sched, pack_gqa=False)
        return SdpaFwdKnobs(
            sched_policy=sched_policy,
            tile_m=base_tile[0],
            tile_n=base_tile[1],
            cga=cga,
            pack_gqa=pack_gqa,
            split_kv=split_value,
        )

    unsplit_leg = _leg(1)
    split_leg = _leg(2)
    split_pack_g = _pack_gqa_group(caps, facts, split_leg.tile_m, split_leg.pack_gqa)
    splits = _split_points(
        caps,
        facts,
        split_leg.tile_m,
        split_leg.tile_n,
        split_leg.cga,
        pack_g=split_pack_g,
        unsplit_knobs=unsplit_leg,
    )

    def _resplit(knobs: SdpaFwdKnobs) -> SdpaFwdKnobs:
        # A runner's changed packing/tile/CGA is a different launch, not just
        # a label on the baseline's split. Keep other architecture policies
        # unchanged until they have their own geometry validation.
        if not _sm100_f16(caps, facts):
            return knobs

        def leg(split):
            policy, auto_cga = _auto_sched_cga(spec, facts, split_kv=split, sched_policy=plain_sched if split > 1 else scheds[0], pack_gqa=knobs.pack_gqa)
            cga = knobs.cga if knobs.cga in effective_cgas(caps, facts, split) else auto_cga
            return replace(knobs, sched_policy=policy, cga=cga, split_kv=split)

        unsplit, split = leg(1), leg(2)
        points = _split_points(
            caps,
            facts,
            split.tile_m,
            split.tile_n,
            split.cga,
            pack_g=_pack_gqa_group(caps, facts, split.tile_m, split.pack_gqa),
            unsplit_knobs=unsplit,
        )
        return leg(points[0])

    base = _leg(splits[0])
    out = [base]
    for tile_m, tile_n in tiles[1:]:
        # A packed baseline's tile runners keep the packing, so tiles the
        # ratio cannot ride are skipped — emitted, they would only spend
        # set-cap slots for mismatch to drop.
        if base.pack_gqa is True and True not in _pack_gqa_points(caps, facts, tile_m or 128, base.cga):
            continue
        out.append(_resplit(replace(base, tile_m=tile_m, tile_n=tile_n)))
    if _sm100_f16(caps, facts):
        # Explicit widths can select a different supported template even when
        # the auto-ranking prefers another one. Cost each with its own split.
        for cga in sorted(effective_cgas(caps, facts, base.split_kv)):
            if cga != base.cga:
                out.append(_resplit(replace(base, cga=cga)))
    # Scheduler runners ride an UNSPLIT leg: a split set is pinned to the plain
    # scheduler above, so an LPT runner is only a candidate without one.
    sched_host = unsplit_leg
    for policy in scheds[1:]:
        out.append(replace(sched_host, sched_policy=policy))
    # The opposite pack_gqa leg, riding its own tile (packed: the largest
    # admitting tile; unpacked: the tile rule's best) and its own CGA width:
    # on the d128 f16 SM100 flavor a packed leg that overflows one 128-row
    # tile (S_q * G > 128 >= S_q) keeps the prefill tile while its unpacked
    # runner-up fits the decode tile.  The other flavors' width rules do not
    # read the packing, so this returns base.cga there.
    _, _, pack_tile = _pack_choice(base.cga)
    if pack_tile is not None:
        if base.pack_gqa is True:
            _, unpacked_cga = _auto_sched_cga(spec, facts, split_kv=base.split_kv, sched_policy=base.sched_policy, pack_gqa=False)
            out.append(_resplit(replace(base, pack_gqa=unpacked_pack, tile_m=tiles[0][0], tile_n=tiles[0][1], cga=unpacked_cga)))
        else:
            _, packed_cga = _auto_sched_cga(spec, facts, split_kv=base.split_kv, sched_policy=base.sched_policy, pack_gqa=True)
            out.append(_resplit(replace(base, pack_gqa=True, tile_m=pack_tile[0], tile_n=pack_tile[1], cga=packed_cga)))
    for split in splits[1:]:
        out.append(_leg(split))
    seen, unique = set(), []
    for knobs in out:
        if knobs not in seen:
            seen.add(knobs)
            unique.append(knobs)
    return unique[:_MAX_SETS_PER_ENGINE]


def _fallback_knobs(spec: EngineSpec, facts) -> SdpaFwdKnobs:
    """The config expected to build where the tuned choice may not.

    Today this is the smallest tile the row admits with the plain scheduler
    and no split — the config that asks least of the device, which is the one
    thing a fallback must be.
    """
    caps = spec.capabilities
    sched_policy = SCHED_NATURAL if SCHED_NATURAL in caps.sched_policies else _sole(caps.sched_policies)
    pack_gqa = False if False in caps.pack_gqas else _sole(caps.pack_gqas)
    # An unpacked fallback is judged on one head's rows (the d128 decode tile
    # is also the least-demanding width: one CTA, no cluster); a row that only
    # packs leaves the graph-level question to the rule.
    sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=1, sched_policy=sched_policy, pack_gqa=False if pack_gqa is False else None)
    return SdpaFwdKnobs(
        sched_policy=sched_policy,
        tile_m=min(caps.tile_ms, default=None),
        tile_n=min(caps.tile_ns, default=None),
        cga=cga,
        pack_gqa=pack_gqa,
        split_kv=1,  # the fallback never splits: least-demanding means one kernel, no partial workspace
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
        sets = _knob_sets(spec, facts) if kind == "A" else [_fallback_knobs(spec, facts)]
        for knobs in sets:
            if mismatch(caps, facts, knobs) is None:
                out.append(PlanConfig(engine_id, knobs))
    return out


# Placement — mode blocks, the backend's entries, the delegating entry, dedup,
# the mode strip — is NOT this family's business: it happens once for every
# family in ``engines/heuristics._assemble``, with these proposals leading the
# backend's entries inside each block by standing assumption.
