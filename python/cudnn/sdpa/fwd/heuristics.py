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
    d192_square_br_as_tl,
    d256_square_br_as_tl,
    pack_gqa_supported,
)
from cudnn.sdpa.fwd.config_sm120 import FP8_HEAD_TILE_GRANULE, HEAD_TILE_GRANULE, SMEM_CAPACITY_BYTES, smem_bytes
from cudnn.sdpa.fwd.engines import (
    ENGINE_SPECS,
    Capabilities,
    EngineSpec,
    SdpaFwdKnobs,
    _band_covers_kv_tail,
    _selected_d_shape,
    effective_cgas,
    mismatch,
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
# (blocks/SM of split_combine_sm100) is ABSORBED into this coefficient: it is
# one fixed kernel, so blocks/SM is a constant, and folding it in keeps a
# cuOccupancy query -- which would need a compiled CUfunction -- off the
# planning path. Empirical: re-measure if split_combine_sm100 changes.
_SPLIT_KV_COMBINE_COST = 0.2


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
    (split_combine_sm100), one block per output row and independent of ``s`` --
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


def _auto_sched_cga(spec: EngineSpec, facts, *, split_kv: int, sched_policy: int) -> tuple[int, Optional[int]]:
    caps = spec.capabilities
    domain = effective_cgas(caps, facts, split_kv)
    selected_shape = _selected_d_shape(caps, facts)
    if selected_shape == (256, 256) and any(shape == selected_shape for shape, _ in caps.cgas_by_d_shape):
        params = _sm100_params_from_facts(facts, split_kv=split_kv, sched_policy=sched_policy)
        selected_sched, selected_cga = select_d256_auto_knobs(params, pertensor=facts.is_fp8, s_q=facts.s_q, s_kv=facts.s_kv)
        if selected_cga not in domain:
            raise ValueError(f"D256 heuristic selected cga={selected_cga} outside the declared domain {sorted(domain)}")
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


def _pack_gqa_points(caps: Capabilities, facts, tile_m: int, cga: Optional[int] = None) -> Tuple[bool, ...]:
    """The pack_gqa axis, best first: ``(True, False)`` when packing wins,
    ``(False, True)`` when it is only eligible, ``(False,)`` when it is not."""
    if not (True in caps.pack_gqas and not facts.thd and facts.h_q != facts.h_kv and pack_gqa_supported(facts.h_q, facts.h_kv, tile_m)):
        return (False,)
    if _pack_gqa_wins(facts, _pack_gqa_tile_q(caps, facts, tile_m, cga)):
        return (True, False)
    return (False, True)


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

    ``pack_g`` is the pack_gqa group of the set the split rides (1 =
    unpacked): packing multiplies each head-group's Q rows by G and divides
    the head count by it, so the wave-cost model must see the PACKED launch
    — the packed grid is smaller, which is exactly when splitting pays.

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
    if facts.thd or facts.has_sink or facts.padded or facts.seq_q_trim:
        return [no_split]
    if caps.skv_tail_via_padding and facts.s_kv % (caps.skv_tile or 128) != 0 and not _band_covers_kv_tail(facts):
        # This S_kv would be served through the synthesized KV-tail padding,
        # which the split cannot ride (mismatch declines the same combination).
        return [no_split]
    if (facts.is_fp8 or facts.is_mxfp8) and facts.dtype_o not in (
        cudnn.data_type.HALF,
        cudnn.data_type.BFLOAT16,
    ):
        # The combine reduces partials in half precision; reducing QUANTIZED
        # partials would lose what the split is meant to be neutral about.
        return [no_split]
    sm_count = facts.device_sm_count or 0
    if sm_count <= 0:
        return [no_split]
    rows_per_tile = _pack_gqa_tile_q(caps, facts, tile_m, cga)
    split_launch = _SplitKvLaunch(
        q_tiles=_ceil_div(facts.s_q * pack_g, rows_per_tile),
        heads_q=facts.h_q // pack_g,
        kv_tiles=_ceil_div(facts.s_kv, tile_n or 128),
        ctas_per_tile=cga or 1,
    )
    unsplit_launch = None
    if unsplit_knobs is not None:
        unsplit_pack_g = (facts.h_q // facts.h_kv) if unsplit_knobs.pack_gqa else 1
        unsplit_launch = _SplitKvLaunch(
            q_tiles=_ceil_div(
                facts.s_q * unsplit_pack_g,
                _pack_gqa_tile_q(caps, facts, unsplit_knobs.tile_m, unsplit_knobs.cga),
            ),
            heads_q=facts.h_q // unsplit_pack_g,
            kv_tiles=_ceil_div(facts.s_kv, unsplit_knobs.tile_n or 128),
            ctas_per_tile=unsplit_knobs.cga or 1,
        )
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
        sched_policy, cga = _auto_sched_cga(
            spec,
            facts,
            split_kv=split_value,
            sched_policy=plain_sched if split_value > 1 else scheds[0],
        )
        base_tile, packed_first, _ = _pack_choice(cga)
        return SdpaFwdKnobs(
            sched_policy=sched_policy,
            tile_m=base_tile[0],
            tile_n=base_tile[1],
            cga=cga,
            pack_gqa=True if packed_first else unpacked_pack,
            split_kv=split_value,
            softmax_precision=_softmax_points(caps)[0],
        )

    unsplit_leg = _leg(1)
    split_leg = _leg(2)
    split_pack_g = (facts.h_q // facts.h_kv) if split_leg.pack_gqa else 1
    splits = _split_points(
        caps,
        facts,
        split_leg.tile_m,
        split_leg.tile_n,
        split_leg.cga,
        pack_g=split_pack_g,
        unsplit_knobs=unsplit_leg,
    )
    base = _leg(splits[0])
    out = [base]
    for tile_m, tile_n in tiles[1:]:
        # A packed baseline's tile runners keep the packing, so tiles the
        # ratio cannot ride are skipped — emitted, they would only spend
        # set-cap slots for mismatch to drop.
        if base.pack_gqa is True and True not in _pack_gqa_points(caps, facts, tile_m or 128, base.cga):
            continue
        out.append(replace(base, tile_m=tile_m, tile_n=tile_n))
    # Scheduler runners ride an UNSPLIT leg: a split set is pinned to the plain
    # scheduler above, so an LPT runner is only a candidate without one.
    sched_host = unsplit_leg
    for policy in scheds[1:]:
        out.append(replace(sched_host, sched_policy=policy))
    # The opposite pack_gqa leg, riding its own tile (packed: the largest
    # admitting tile; unpacked: the tile rule's best).
    _, _, pack_tile = _pack_choice(base.cga)
    if pack_tile is not None:
        if base.pack_gqa is True:
            out.append(replace(base, pack_gqa=unpacked_pack, tile_m=tiles[0][0], tile_n=tiles[0][1]))
        else:
            out.append(replace(base, pack_gqa=True, tile_m=pack_tile[0], tile_n=pack_tile[1]))
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
    sched_policy, cga = _auto_sched_cga(spec, facts, split_kv=1, sched_policy=sched_policy)
    return SdpaFwdKnobs(
        sched_policy=sched_policy,
        tile_m=min(caps.tile_ms, default=None),
        tile_n=min(caps.tile_ns, default=None),
        cga=cga,
        pack_gqa=False if False in caps.pack_gqas else _sole(caps.pack_gqas),
        split_kv=1,  # the fallback never splits: least-demanding means one kernel, no partial workspace
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
        sets = _knob_sets(spec, facts) if kind == "A" else [_fallback_knobs(spec, facts)]
        for knobs in sets:
            if mismatch(caps, facts, knobs) is None:
                out.append(PlanConfig(engine_id, knobs))
    return out


# Placement — mode blocks, the backend's entries, the delegating entry, dedup,
# the mode strip — is NOT this family's business: it happens once for every
# family in ``engines/heuristics._assemble``, with these proposals leading the
# backend's entries inside each block by standing assumption.
