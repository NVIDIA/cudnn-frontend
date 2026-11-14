# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel-facing configuration for the FROST SM90 SDPA forward templates.

Two templates ship: the dedicated D128 template (``sm90/prefill_d128_f16.py``,
issue #378), a 128 x 128 tile for the exact (128, 128) dense declarations it
serves, and the D512 body (``sm90/prefill_d512_f16.py``, with its native
128/256/512-column storage and the exact dense D64 body beside it) on a 64 x 64
tile for every head dim up to 512 and every declared feature. A pinned tile
names the template; with none pinned ``pick_flavor`` chooses once, from the head
dims and the declared features (``NOT_SERVED``). The frozen ``TemplateParams``
record contains only graph declarations and selected tuning knobs. Runtime
lengths, pointers, and packed token totals stay out of compile keys.
Graph-adapter declaration logic lives with the adapter; this module keeps the
small interface consumed by the native D64/D128/D256/D512 kernels.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Optional

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL

SCALE_POSITIVE = 1
SCALE_ZERO = 0
SCALE_NEGATIVE = -1


# Maximum envelope. The adapter selects native 128/256/512-column storage and
# contraction; exact dense D64 keeps its separate body. GMEM maps carry the actual
# extents, with zero-filled loads and clipped stores at head tails. Eight elements
# satisfy the TMA 16-byte global-stride rule for both half types.
D_TILE = 512
D_ALIGN = 8
# Sequence tile geometry shared by the native head widths.
TILE_M = 64
TILE_N = 64

# Kernel flavors: the SM90 templates a declaration lowers onto, keyed by (D_QK, D_V),
# smallest first. D512 is the D512 body: every head-dim pair up to its tile
# (head_dims_mismatch), on its native 128/256/512-column storage (head_tile) or, for
# the exact dense D64 subset, the D64 body (native_d64_mismatch). D128 (issue #378) is
# the dedicated 128 x 128 template; it serves exact (128, 128) dense declarations outside
# NOT_SERVED. The adapter, the engine row and the tests share these spellings.
D512_FLAVOR = (D_TILE, D_TILE)
D128_FLAVOR = (128, 128)
FLAVORS = (D128_FLAVOR, D512_FLAVOR)
# Each flavor's fixed CTA tile, (tile_m, tile_n): Q rows by keys. A pinned tile names the
# flavor whose tile it is (a 128 tile the D128 template, a 64 tile the D512 body), and the
# adapter refuses a tile no flavor of the declaration has; the engine row states the same
# (Capabilities.tile_ms_by_d_shape / tile_ns_by_d_shape).
FLAVOR_TILES = {D512_FLAVOR: (TILE_M, TILE_N), D128_FLAVOR: (128, 128)}
# Native-range floor per flavor, in the engine row's d_envelope_floors vocabulary: D128
# serves head dims in (127, 128], i.e. exactly (128, 128). D512 has no template floor: it
# serves every pair up to its tile (the engine row's floors are ROW decisions,
# engines._sm90_spec).
FLAVOR_FLOORS = {D128_FLAVOR: 127}
# What a native flavor does not serve, by declared feature (declared_features' vocabulary),
# with the label its refusal names. pick_flavor never selects a flavor for a declaration that
# names one of its features, so such a declaration runs on the D512 body, which serves them
# all; validate_params(flavor=...) refuses them with these labels, in this order.
D128_NOT_SERVED = {
    "thd": "THD (ragged) layouts",
    "paged": "paged KV",
    "split_kv": "split-KV",
    "sink": "attention sinks",
    "window_left": "a sliding window",
    "window_right": "a widened right band",
    "lengths": "per-batch lengths",
    "pack_gqa": "PackGQA",
}
NOT_SERVED = {D128_FLAVOR: D128_NOT_SERVED}

# The route's cutoff on L, a causal plan's mean KV tiles per Q tile: from L 48 every causal D128 plan of the walk's rule
# goes one tile per CTA, whatever kind the rule picks (d128_causal_route). On H100 SXM GPUs (#1159's harness, bf16,
# ratios medians of per-round ratios, the band the larger of 1% and the twins' gap), an earlier rule routed the paired
# walks from this L, on 161 timings of the walk against the one-tile kernel. The walk beat it on all 54 at L 40.5 or
# less; from L 48.5 to 80.5 it won five (by 1.2% to 2.6%) and tied eight, and from L 88.5 up it lost 12 of 23 (by 1.0%
# to 2.3%), tied ten and won one; the walk of b8 h32/8 S_q 2048 S_kv 16384 (L 120.5) ran 0.988 of the one-tile kernel.
# At the same S_q and S_kv a later run timed the one-tile kernel 1.4% faster than the walk of b5 h12/12 and 1.2% faster
# the NATURAL stride of b1 h20/20 (both beyond the band), and 0.3% and 0.9% slower than the walks of b1 h32/8 and b1
# h64/4 (inside it). Every one-tile timing here ran the causal order a routed plan keeps (LPT_L2: no arm pinned one,
# and the graph route proposed it), so heuristics._sched_points proposes that order for a routed plan.
D128_ROUTE_MAX_L = 48

# The D128 template's 128 x 128 set leads the SM90 row's proposals only from this many query rows per request
# (heuristics._sm90_d128_knob_sets). Below it the template's 128-row Q tile is at most half full, and the D512 body's
# 64-row tile and its packed decode tile ran faster: on 258 training cells of a frozen H100 SXM study (cuDNN 9.27, bf16,
# exact dense (128, 128), the repo drivers' timing with an L2 flush), the D512 body's own plan order ran 1.2 to 2.5
# times the template's speed at S_q 16 or less (geomean by head layout and mask, up to 29 times on one GQA decode
# cell) and 0.93 to 1.46 times at S_q 17 to 64, while the template ran 1.1 to 2.4 times the D512 body's speed at S_q
# 65 to 256.
D128_LEAD_MIN_SQ = 65


# The SCHED_LPT_L2 budget of the SM90 templates (D512 and D128 read it through kernels/sm90/_common_hopper; api_dsl and
# heuristics keep their own copies): here, so that the D128 planner below reads the same value without the DSL.
SCHED_L2_BUDGET_BYTES = 50 * 1024 * 1024


# The plan-time rule of a persistent causal record (causal_walk_plan, and causal_stride_snake for the stride) reads the
# seven constants below. Each is a speed choice: every value computes the same O and Stats bit for bit. The rule lives
# here, host arithmetic with no template, no torch and no DSL, so that the adapter's route (d128_causal_route) and the
# graph route's proposal read it without loading the template; prefill_d128_f16 imports every name below and keeps it
# as its own attribute. A changed value never reuses a compiled artifact: the persistent cache's environment carries a
# digest of the whole cudnn package's source (frost/compiled_cache.source_tree_digest).
# The light-row count light_rows, the number of each (head, batch) group's lightest Q tiles that go out one at a time
# after the pairs: -1 lets the rule choose it, and keep the stride by the grid where the stride balances the plan at
# least as well; n >= 0 forces light_rows = min(n, T) on the paired walk of a plan with more items than CTAs (0: pairs
# only; T: single items only, heaviest first). A plan with at most one item per CTA keeps light_rows = T at every value.
CAUSAL_WALK_LIGHT_ROWS: int = -1
# The rule's light_rows stops where a round of single items would spread its K and V reads past the L2 budget of the
# LPT_L2 order (work_of's): such a round reads up to min(ceil(CTAs / qh_per_kh) + 1, h_kv * b) KV heads, each to its
# row's last KV tile. False lets light_rows reach T.
CAUSAL_WALK_L2_CAP: bool = True
# An item's fixed cost in KV tiles, which the rule adds to each item's KV tile count. On an H100 SXM a development copy
# of the persistent loop took about 3.2 us of each item outside its KV steps and about 1.7 us per KV step at S = 1024
# (an in-kernel clock probe and a fit to kernel times), so 2; 1 and 3 choose the same light_rows as 2 on 26 causal
# shapes from b1 to b66 at S = 512 to 16384.
CAUSAL_WALK_PHI: int = 2
# Odd rounds go to the CTAs in reverse, so a CTA that took one of the heaviest units of a round takes one of the
# lightest of the next. False deals every round from CTA 0.
CAUSAL_WALK_SNAKE: bool = True
# The paired walk needs a head's Q tiles spread over enough CTAs: a plan with fewer Q tiles per (head, batch) group than
# CAUSAL_WALK_MIN_TILES takes the stride by the grid instead (causal_walk_plan returns None). With T <= 8 a group's
# pairs sit on at most four CTAs, each running two of its Q tiles one after the other, so the second reads its K and V
# back from HBM; the stride in NATURAL order runs all of a group's Q tiles in one round on neighbouring CTAs, which
# read them once (Nsight Compute: 1.00 to 1.03 times the read-once floor against 1.26 to 2.16 for the LPT_L2 stride
# shipped before; timed on 69 shapes, the walk at T 2 to 4 ran 0.98 of cuDNN where the stride ran 1.05 to 1.07). -1 lets
# the walk's rule decide at every T.
CAUSAL_WALK_MIN_TILES: int = 9
# The cut above applies only to a plan of at most CAUSAL_WALK_CUT_MAX_QH Q heads per KV head (qh_per_kh); a plan with
# more goes to the walk's rule at every T. Its argument is K and V read once, and with 16 Q heads per KV head the 16
# heads that share a KV head already read it close together in either order. A timing run compared the stride the cut
# keeps with the walk the rule picks without it on 20 such shapes of 56 fresh frozen ones (#1159's harness, H100
# SXM, ratios medians of per-round ratios, the band the larger of 1% and the twins' gap): 1.00 to 1.14 at qh_per_kh 8
# or less (12 shapes, 9 above the band, none below), 0.985 to 1.004 at 16 (8 shapes, 4 below the band, none above).
CAUSAL_WALK_CUT_MAX_QH: int = 8
# The stride by the grid's deal (causal_stride_snake): round k gives items k * n to k * n + n - 1 to CTAs 0 to n - 1, n
# the grid; under the snake deal odd rounds go to the CTAs in reverse (stride_next), so a CTA that took a group's
# heaviest Q tile in one round takes a light one in the next, while each round still holds the same items. "auto"
# takes the deal whose busiest CTA carries less (item costs as the walk's rule counts them, the straight deal on a tie);
# "straight" and "snake" force one. The planner's cost predicted the measured snake-over-straight ratio within about 2%
# on 19 causal shapes (0.878 against 0.881 on the shape where the straight deal won). Bit for bit either way.
STRIDE_DEAL: str = "auto"


class CausalWalk(NamedTuple):
    """The paired walk's compile-time constants (causal_walk_plan): plain ints and bools.

    Units in walk order. For each (head, batch) group g = head + h_q * batch in turn, ``pairs`` units, the j-th holding
    Q tiles T - 1 - j and light_rows + j; when T - light_rows is odd, groups 2k and 2k + 1 share one unit of their
    middle tile light_rows + pairs, right after group 2k + 1's pairs, and an odd group count leaves the last group's
    middle alone, the last heavy unit. Then the light rows, one item per unit: tile light_rows - 1 of every group, then
    light_rows - 2, ..., tile 0. Round k deals units k * ctas to k * ctas + ctas - 1 to CTAs 0 to ctas - 1, in reverse
    when k is odd under snake.
    """

    q_tiles: int
    groups: int
    light_rows: int
    pairs: int  # pair units per group, (q_tiles - light_rows) // 2
    middles: bool  # q_tiles - light_rows is odd
    lone_middle: int  # 1 when the last group's middle unit holds one item (middles and an odd group count), else 0
    heavy_units: int  # the pair and middle units, which come first
    units: int
    ctas: int  # min(units, the persistent CTA count): every CTA has a unit
    snake: bool


def causal_walk_layout(q_tiles: int, groups: int, light_rows: int, persistent_ctas: int, snake: bool) -> CausalWalk:
    """The paired walk over ``groups`` groups of ``q_tiles`` Q tiles with ``light_rows`` light rows (CausalWalk)."""
    pairs, middles = divmod(q_tiles - light_rows, 2)
    heavy = groups * pairs + middles * ((groups + 1) // 2)
    units = heavy + groups * light_rows
    lone = int(bool(middles) and groups % 2 == 1)
    return CausalWalk(q_tiles, groups, light_rows, pairs, bool(middles), lone, heavy, units, min(units, persistent_ctas), snake)


def causal_kv_tiles(shape: tuple[int, int, int, int, int], bottom_right: bool) -> list[int]:
    """Each Q tile's KV tile count on a causal plan: work_record's iterations (compute_kv_loop_bounds' right bound) in
    plain ints. Tile t walks min(N_kv, t + 1 + ceil(diag / KV_TILE)) tiles, diag = S_kv - S_q under bottom-right and 0
    under top-left; a partial last Q tile walks for all of its rows."""
    _, _, _, s_q, s_kv = shape
    q_tile, kv_tile = FLAVOR_TILES[D128_FLAVOR]
    diag = s_kv - s_q if bottom_right else 0
    return [max(0, min(-(-s_kv // kv_tile), -(-(t * q_tile + q_tile + diag) // kv_tile))) for t in range(-(-s_q // q_tile))]


def _rounds_makespan(costs: list[int], ctas: int, snake: bool) -> int:
    """The largest CTA load when ``costs`` (one per unit, in walk order) go out in rounds of min(ctas, len(costs))
    units, each round from CTA 0 or, under snake, odd rounds in reverse."""
    n = min(ctas, len(costs))
    padded = costs + [0] * (-len(costs) % n)
    if not snake:
        return max(sum(padded[cta::n]) for cta in range(n))
    # Under snake, CTA c takes column c of the even rounds and column n - 1 - c of the odd ones (units 2n - 1 - c,
    # 4n - 1 - c and so on): one strided slice each, summed in C, about twice as fast as summing the rows' columns.
    return max(sum(padded[cta :: 2 * n]) + sum(padded[2 * n - 1 - cta :: 2 * n]) for cta in range(n))


def _walk_unit_costs(walk: CausalWalk, item_cost: list[int]) -> list[int]:
    """Each unit's cost in walk order (CausalWalk): the sum of its items' ``item_cost`` (indexed by Q tile)."""
    q_tiles, groups, light, pairs = walk.q_tiles, walk.groups, walk.light_rows, walk.pairs
    pair_costs = [item_cost[q_tiles - 1 - j] + item_cost[light + j] for j in range(pairs)]
    if walk.middles:
        middle = item_cost[light + pairs]
        costs = (pair_costs + pair_costs + [2 * middle]) * (groups // 2) + (pair_costs + [middle] if groups % 2 else [])
    else:
        costs = pair_costs * groups
    for tile in range(light - 1, -1, -1):
        costs += [item_cost[tile]] * groups
    return costs


def _lpt_l2_bytes(s_kv: int, row_bytes: int) -> int:
    """work_of's LPT_L2 budget: a power of two of whole KV groups within SCHED_L2_BUDGET_BYTES; 0 is one group."""
    groups = SCHED_L2_BUDGET_BYTES // max(1, s_kv * row_bytes)
    return (1 << groups.bit_length() - 1) * s_kv * row_bytes if groups > 1 else 0


def _stride_item_costs(shape, sched_policy: int, qh_per_kh: int, item_cost: list[int], row_bytes: int) -> list[int]:
    """Each item's cost in the stride walk's order, the one-tile grid's order under ``sched_policy`` (work_of): NATURAL
    Q tile fastest, then head, then batch; LPT the heaviest row of every (head, batch) first; LPT_L2 the same row by row
    inside each block of whole KV groups (lpt_l2_tile_coords' blocks)."""
    b, h_q, _, _, s_kv = shape
    if sched_policy == SCHED_NATURAL:
        return item_cost * (h_q * b)
    row_items = [h_q * b]
    if sched_policy == SCHED_LPT_L2:
        n_groups = h_q // qh_per_kh * b
        active = min(max(1, _lpt_l2_bytes(s_kv, row_bytes) // max(1, s_kv * row_bytes)), n_groups)
        active = -(-n_groups // -(-n_groups // active))
        blocks = -(-n_groups // active)
        row_items = [(n_groups - k * active if k == blocks - 1 else active) * qh_per_kh for k in range(blocks)]
    costs = []
    for per_row in row_items:
        for tile in range(len(item_cost) - 1, -1, -1):
            costs += [item_cost[tile]] * per_row
    return costs


def _stride_deal(costs: list[int], ctas: int, deal: str) -> tuple[int, bool]:
    """``(largest CTA load, snake)`` of the stride by the grid over ``costs`` (one per item, in the stride's order)
    under ``deal`` (STRIDE_DEAL): "auto" takes the snake deal only where its largest load is strictly smaller."""
    if deal not in ("auto", "straight", "snake"):
        raise ValueError(f"prefill_d128_f16_sm90: STRIDE_DEAL must be 'auto', 'straight' or 'snake'; got {deal!r}")
    straight = _rounds_makespan(costs, ctas, False) if deal != "snake" else None
    snaked = _rounds_makespan(costs, ctas, True) if deal != "straight" else None
    if snaked is not None and (straight is None or snaked < straight):
        return snaked, True
    return straight, False


@lru_cache(maxsize=64)
def causal_walk_plan(
    shape: tuple[int, int, int, int, int],
    bottom_right: bool,
    sched_policy: int,
    qh_per_kh: int,
    persistent_ctas: int,
    storage_bytes: int,
    *,
    light_rows: int = CAUSAL_WALK_LIGHT_ROWS,
    l2_cap: bool = CAUSAL_WALK_L2_CAP,
    phi: int = CAUSAL_WALK_PHI,
    snake: bool = CAUSAL_WALK_SNAKE,
    min_tiles: int = CAUSAL_WALK_MIN_TILES,
    cut_max_qh: int = CAUSAL_WALK_CUT_MAX_QH,
    deal: str = STRIDE_DEAL,
) -> Optional[CausalWalk]:
    """The paired walk of a persistent causal record, or None for the stride by the grid: the plan-time rule, every
    input a compile-key one (the declared shape, the mask, sched_policy, the GQA ratio, the CTA count, the storage
    width) or a constant above. Cached (its answer is an immutable CausalWalk or None), so repeated plans reuse it.

    A plan with at most one item per CTA takes light_rows = T, one item per CTA, and a forced ``light_rows`` its walk.
    Below ``min_tiles`` Q tiles per group the stride, where ``qh_per_kh`` is at most ``cut_max_qh``
    (CAUSAL_WALK_MIN_TILES, CAUSAL_WALK_CUT_MAX_QH). Otherwise, with each item costing its KV tiles plus ``phi``, the
    rule takes the light_rows in [0, cap] whose walk has the smallest largest CTA load (the smaller on a tie), cap the
    largest light_rows whose rounds of single items stay within the L2 budget (the LPT_L2 order's, or one KV group's
    bytes where that is 0), and keeps the stride where the stride's largest load, under the deal ``deal`` picks
    (_stride_deal), is smaller. Costs are exact integer sums over the units, at most T + 1 walks of at most T * b * h_q
    items.
    """
    b, h_q, _, _, s_kv = shape
    kv_tiles = causal_kv_tiles(shape, bottom_right)
    q_tiles, groups = len(kv_tiles), h_q * b
    if q_tiles * groups <= persistent_ctas:
        return causal_walk_layout(q_tiles, groups, q_tiles, persistent_ctas, snake)
    if light_rows >= 0:
        return causal_walk_layout(q_tiles, groups, min(light_rows, q_tiles), persistent_ctas, snake)
    if q_tiles < min_tiles and qh_per_kh <= cut_max_qh:
        return None
    item_cost = [tiles + phi for tiles in kv_tiles]
    row_bytes = 2 * D128_FLAVOR[0] * storage_bytes
    top = q_tiles
    if l2_cap:
        budget = _lpt_l2_bytes(s_kv, row_bytes) or s_kv * row_bytes
        heads = min(-(-persistent_ctas // qh_per_kh) + 1, h_q // qh_per_kh * b)
        top = next((tile for tile, tiles in enumerate(kv_tiles) if heads * tiles * FLAVOR_TILES[D128_FLAVOR][1] * row_bytes > budget), q_tiles)
    best_load, best = None, None
    for light in range(top + 1):
        walk = causal_walk_layout(q_tiles, groups, light, persistent_ctas, snake)
        load = _rounds_makespan(_walk_unit_costs(walk, item_cost), persistent_ctas, snake)
        if best is None or load < best_load:
            best_load, best = load, walk
    stride_load, _ = _stride_deal(_stride_item_costs(shape, sched_policy, qh_per_kh, item_cost, row_bytes), persistent_ctas, deal)
    return best if best_load <= stride_load else None


@lru_cache(maxsize=64)
def causal_stride_snake(
    shape: tuple[int, int, int, int, int],
    bottom_right: bool,
    sched_policy: int,
    qh_per_kh: int,
    persistent_ctas: int,
    storage_bytes: int,
    *,
    phi: int = CAUSAL_WALK_PHI,
    deal: str = STRIDE_DEAL,
) -> bool:
    """Whether the stride by the grid of a persistent causal record deals odd rounds in reverse (stride_next): the
    stride's deal (STRIDE_DEAL) over the items in the stride's order (sched_policy, _stride_item_costs), each costing
    its KV tiles plus ``phi`` as in causal_walk_plan, on min(items, ``persistent_ctas``) CTAs. Every input a compile-key
    one, and cached as causal_walk_plan."""
    kv_tiles = causal_kv_tiles(shape, bottom_right)
    item_cost = [tiles + phi for tiles in kv_tiles]
    row_bytes = 2 * D128_FLAVOR[0] * storage_bytes
    _, snaked = _stride_deal(_stride_item_costs(shape, sched_policy, qh_per_kh, item_cost, row_bytes), persistent_ctas, deal)
    return snaked


# The route's name for the planner's KV tile count (causal_kv_tiles above): one definition.
d128_causal_kv_tiles = causal_kv_tiles


def d128_causal_route(shape, bottom_right: bool, ctas: int) -> bool:
    """Whether a causal D128 plan of the walk's rule goes one tile per CTA (api_dsl.D128_CAUSAL_WALK_ROUTE): a plan of
    at most ``ctas`` work items (Q tiles times heads times batch), on which both kernels run one item per CTA, or one
    whose L, its mean KV tiles per Q tile (d128_causal_kv_tiles), is at least D128_ROUTE_MAX_L, whatever kind the rule
    picks. ``shape`` is the declared (b, h_q, h_kv, S_q, S_kv), ``bottom_right`` the mask's alignment as the template
    reads it, and ``ctas`` the persistent CTA count compile() gives the record (the device's SM count). No template, no
    planner and no torch: the walk's search takes seconds on the largest plans, and the graph route's proposal
    (heuristics._sm90_d128_causal_walks) reads the same decision from this import-light module."""
    b, h_q, _, _, _ = shape
    kv_tiles = d128_causal_kv_tiles(shape, bottom_right)
    return len(kv_tiles) * h_q * b <= ctas or sum(kv_tiles) >= D128_ROUTE_MAX_L * len(kv_tiles)


# The causal claim counter of a persistent causal D128 record (TemplateParams.causal_claim): 0 runs the static
# schedules (the stride, the paired walk), 1 hands every item after a CTA's first out from an Int32 counter in the
# caller's workspace with the items numbered group-major and each (head, batch) group's Q tiles in NATURAL order. A
# dynamic hand-out keeps a head's Q tiles running together at any plan size (Nsight Compute: K and V read 1.000 times
# the read-once floor where a static schedule drifted to 1.24 to 1.46); the counter needs D128_CLAIM_WORKSPACE_BYTES of
# caller workspace, zeroed on the launch stream before every launch. Handing out a group's heaviest Q tile first was
# also timed: never faster than NATURAL (by 0 to 1.2%), so it is not built.
D128_CLAIM_OFF = 0
D128_CLAIM_NATURAL = 1
D128_CLAIM_WORKSPACE_BYTES = 128


def d128_claim_coords(item: int, q_tiles: int, heads: int) -> tuple[int, int, int]:
    """``(Q tile, head, batch)`` of work item ``item`` on a causal claim-counter record (causal_claim 1): items are
    numbered group-major, batch then head then Q tile, so a group's Q tiles are consecutive and are handed out within a
    few claims of each other. The template's work_of decodes the same arithmetic in the kernel; this is its host form
    for tests."""
    return item % q_tiles, (item // q_tiles) % heads, item // (q_tiles * heads)


def declared_features(*, thd=False, sink=False, window_left=None, window_right=None, lengths=False, pack_gqa=False, paged=False, split_kv=1) -> frozenset[str]:
    """The features one declaration names, in the vocabulary of ``NOT_SERVED``.

    ``window_left`` is the sliding-window offset and ``window_right`` a widened right bound,
    each None when absent; a ``window_right`` of 0 is plain causal and names nothing.
    ``lengths`` is any per-batch Q or KV length tensor, ``paged`` a paged K/V pool and
    ``split_kv`` the KV partition count (1, unsplit, names nothing).
    """
    named = (
        ("thd", thd),
        ("paged", paged),
        ("split_kv", split_kv != 1),
        ("sink", sink),
        ("window_left", window_left is not None),
        ("window_right", bool(window_right)),
        ("lengths", lengths),
        ("pack_gqa", pack_gqa),
    )
    return frozenset(feature for feature, on in named if on)


def native_d64_mismatch(
    *,
    thd=False,
    sink=False,
    pack_gqa=False,
    q_lengths=False,
    kv_lengths=False,
    window_left=None,
    window_right=None,
    bottom_right=False,
    s_q=0,
    s_kv=0,
    split_kv=1,
):
    """The bounded native D64 leg, shared by graph and standalone admission."""
    if split_kv != 1:
        return "native SM90 D64 does not support split-KV"
    if thd or sink or pack_gqa or q_lengths or kv_lengths or window_left is not None or window_right is not None:
        return "native SM90 D64 serves dense, unpadded, unpacked Q/K/V without sinks or window bands"
    if bottom_right and s_q > s_kv:
        return "unpadded bottom-right S_q > S_kv is not yet supported on SM90"
    return None


def head_dims_mismatch(d_qk, d_v) -> str | None:
    """Why ``(d_qk, d_v)`` is outside the D envelope, or None; D_QK != D_V is served."""
    for value in (d_qk, d_v):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 < value <= D_TILE or value % D_ALIGN:
            return (
                f"SM90 SDPA serves positive D_QK/D_V multiples of {D_ALIGN} up to its {D_TILE}-column tile "
                f"(TMA 16-byte global-stride rule); graph has D_QK={d_qk}/D_V={d_v}"
            )
    return None


def pick_flavor(d_qk: int, d_v: int, flavors: tuple[tuple[int, int], ...] = (D512_FLAVOR,), declared: frozenset[str] = frozenset()) -> tuple[int, int]:
    """The template that serves a ``(d_qk, d_v)`` declaration naming ``declared``, among ``flavors``.

    The smallest native flavor in ``flavors`` that covers the pair above its floor
    (``FLAVOR_FLOORS``) and serves every declared feature (``NOT_SERVED``), else D512,
    the D512 body. Total: an out-of-envelope pair still answers D512, and
    ``head_dims_mismatch`` declines it, so the adapter's existing NotImplementedError is
    unchanged. ``flavors`` is the caller's pool (the flavors with a template,
    ``api_dsl._SM90_KERNEL_FILES``, as ``api_dsl._pick_flavor``'s ``candidates`` on the
    SM100 line); the default pool is D512 alone, so every declaration selects D512.
    ``declared`` is ``declared_features`` of the declaration.
    """
    for flavor in FLAVORS:
        if flavor == D512_FLAVOR or flavor not in flavors or declared & NOT_SERVED.get(flavor, {}).keys():
            continue
        if d_qk <= flavor[0] and d_v <= flavor[1] and min(d_qk, d_v) > FLAVOR_FLOORS.get(flavor, 0):
            return flavor
    return D512_FLAVOR


def d128_decline(d_qk: int, d_v: int, declared: frozenset[str] = frozenset()) -> str | None:
    """Why the D128 template (SM90's 128 x 128 tile) does not serve a ``(d_qk, d_v)``
    declaration naming ``declared``, or None: the refusal a pinned 128 tile meets where
    ``pick_flavor`` keeps the declaration on the D512 body. The head dims first (exactly
    (128, 128), ``FLAVOR_FLOORS``), then the first ``NOT_SERVED`` feature in table order."""
    if pick_flavor(d_qk, d_v, FLAVORS) != D128_FLAVOR:
        return f"SM90's 128 x 128 tile (the D128 template) serves exact (128, 128) head dims; graph has D_QK={d_qk}/D_V={d_v}"
    for feature, label in D128_NOT_SERVED.items():
        if feature in declared:
            return f"SM90's 128 x 128 tile (the D128 template) does not serve {label}"
    return None


def d128_graph_decline(facts, *, pack_gqa: bool = False, split_kv: int = 1) -> str | None:
    """Why the SM90 row's 128 x 128 tile, the dedicated D128 template (#378), cannot run this graph with this packing
    and split, or None: :func:`d128_decline` on the graph's declared features, then the two declines that are the
    row's alone: S_q == 1 (decode is outside the scope #378 accepted; the D512 body keeps it, while the direct adapter
    serves it on D128), and the adapter's pending unpadded bottom-right S_q > S_kv, declined as native_d64_mismatch
    declines it for D64 so the row ranks no D128 plan that cannot build. ``facts`` is the graph's
    ``graph_analyzer.SdpaGraphFacts``. It lives here, beside the decline it extends, so the row's placement can ask it
    without importing the engines module."""
    declared = declared_features(
        thd=facts.thd,
        sink=facts.has_sink,
        window_left=facts.window_left,
        window_right=(facts.right_bound or 1) if facts.right_band_widening else None,
        lengths=bool(
            facts.seq_q_trim
            or facts.padded
            or facts.seq_q_t is not None
            or facts.cu_seq_q_t is not None
            or facts.seq_kv_t is not None
            or facts.cu_seq_kv_t is not None
        ),
        pack_gqa=pack_gqa,
        paged=facts.has_paged_kv,
        split_kv=split_kv,
    )
    why = d128_decline(facts.d_qk, facts.d_v, declared)
    if why is not None:
        return why
    if facts.s_q == 1:
        return "s_q == 1 (decode) is outside the SM90 D128 template's scope; the 64 x 64 tile serves it"
    if facts.causal and facts.bottom_right and facts.s_q > facts.s_kv:
        return "unpadded bottom-right S_q > S_kv is not yet supported on SM90"
    return None


def dense_bind_strides(shape: tuple, strides: tuple, elem_bytes: int) -> tuple | None:
    """SM90 binds every covering dense permutation with TMA-aligned live strides."""
    from cudnn.sdpa.graph_analyzer import dense_layout_ok

    if not dense_layout_ok(shape, strides):
        return None
    bs, hs, ss = (int(shape[3]) if size == 1 else int(stride) for size, stride in zip(shape[:3], strides))
    if any(stride * elem_bytes % 16 for stride in (bs, hs, ss)):
        return None
    return bs, ss, hs


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph choices that change the traced SM90 kernel specialization."""

    dtype_qkv: int = DTYPE_FP16
    causal: bool = False  # Top-left on dense unpadded S_q == S_kv is spelled bottom-right by the adapter.
    thd_varlen: bool = False
    has_lse: bool = False
    pack_gqa: bool = False
    qh_per_kh: int = 1
    sched_policy: int = SCHED_NATURAL
    scale_mode: int = SCALE_POSITIVE
    # Dense (B,) length reads; THD sets seq_kv_lens_present, its metadata operand.
    seq_q_lens_present: bool = False
    seq_kv_lens_present: bool = False
    # Diagonal band, appended so every earlier specialization keeps its defaults.
    # `window_left` is the sliding-window offset W (cuDNN left bound - 1): key k is visible
    # from row q when k >= q + delta - W. `window_right` is a widened right bound R > 0
    # (k <= q + delta + R); plain causal stays `causal=True`. `bottom_right=None` is the
    # original alignment, bottom-right exactly when causal; SdpaFwdDslSm90.check_support canonicalizes it.
    window_left: int | None = None
    window_right: int | None = None
    bottom_right: bool | None = None
    # Per-Q-head FP32 sink logits: an extra bound operand in these specializations only.
    has_sink: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the Stats store
    # scales the natural-log LSE by log2(e).
    stats_log2: bool = False
    # THD caller-buffer origins: per port in (Q, K, V, O, Stats) order, ``(bits, multiplier)`` of the
    # port's bound ragged offsets (int32 / int64, and the offset unit in elements), or None for a port
    # bound without offsets, which keeps the compact prefix. None for the whole field keeps the packed
    # layout. The setup launch turns each offset into a token origin, so padded THD (gaps between
    # sequences, TE's cu_seqlens_padded) is read and written where the caller put it.
    thd_origins: tuple | None = None
    split_kv: int = 1
    # Native contraction/storage envelope; independent of runtime lengths.
    head_tile: int = 512
    # Dense queries with native TMA loads from independently mapped K/V pages.
    page_size: int = 0
    # Overlap switch of the D128 template (sm90/prefill_d128_f16.py), appended so every earlier specialization
    # keeps its defaults: False renders the v1 loop. intra_wg_overlap issues QK(t) and PV(t-1) as two WGMMA groups in
    # flight together inside each compute warpgroup, and computes the same O and Stats bit for bit, up to the sign of a
    # zero O element. SdpaFwdDslSm90 sets the shipped value (api_dsl.D128_INTRA_WG_OVERLAP) for the D128 flavor only;
    # D512 refuses it.
    intra_wg_overlap: bool = False
    # Persistent switch of the D128 template, appended after the overlap one so every earlier specialization keeps its
    # default (False: one work tile per CTA). persistent launches min(work items, persistent_ctas) CTAs that each loop
    # over work items, striding by the grid (on a paired walk, causal_walk 1, min(walk units, persistent_ctas) CTAs,
    # each walking its dealt units), with the pipeline states carried across items, so the producer loads the next
    # item's first K tile and Q while the current item finishes (persistent_ctas is a compile() argument the adapter
    # sizes to the SM count). It requires intra_wg_overlap, so the v1 loop stays the non-persistent reference, and it
    # computes the one-tile-per-CTA overlap kernel's O and Stats bit for bit. SdpaFwdDslSm90 sets the shipped value
    # (api_dsl.D128_PERSISTENT) on non-causal D128 plans, and on causal ones only as api_dsl.D128_PERSISTENT_CAUSAL_WALK
    # says, less the plans api_dsl.D128_CAUSAL_WALK_ROUTE routes one tile per CTA, those of at most one item per CTA or
    # of L, the mean KV tiles per Q tile, at least D128_ROUTE_MAX_L above (causal_walk below; at that constant's
    # default, 2, every other causal plan is persistent). D512 refuses it.
    persistent: bool = False
    # Causal walk of the D128 persistent loop, appended after the persistent switch so every earlier specialization
    # keeps its default (0: the stride by the grid over the sched_policy order). 1 is the paired walk: of each (head,
    # batch)'s T Q tiles, tiles light_rows to T - 1 go out in pairs of equal KV work on an aligned causal staircase
    # (tile T - 1 - j with tile light_rows + j; two groups' middle tiles pair up when T - light_rows is odd), then tiles
    # light_rows - 1 down to 0 one at a time, in rounds dealt to the CTAs in alternating directions, with light_rows set
    # at compile time from the declared shape and CTA count (or the stride kept below the template's
    # CAUSAL_WALK_MIN_TILES Q tiles per group on a plan of at most its CAUSAL_WALK_CUT_MAX_QH Q heads per KV head, and
    # where it balances the plan better, dealing odd rounds in reverse where that balances it better). It requires
    # persistent and causal, and computes the one-tile-per-CTA overlap kernel's O and Stats bit for bit.
    # SdpaFwdDslSm90 sets it from api_dsl.D128_PERSISTENT_CAUSAL_WALK on causal D128 plans only (1 at that constant's
    # default, 2), with sched_policy NATURAL unless the caller pins one, and clears it (with persistent) on a plan
    # api_dsl.D128_CAUSAL_WALK_ROUTE routes one tile per CTA. D512 refuses it.
    causal_walk: int = 0
    # Causal claim counter of the D128 persistent loop, appended after the causal walk so every earlier specialization
    # keeps its default (0, D128_CLAIM_OFF: the static schedules above). 1 (D128_CLAIM_NATURAL): CTA c's first item is
    # c, and every later one is the CTA count plus a ticket the producer warp takes from an Int32 counter at the meta
    # pointer (word 0 of the caller's workspace, which the launcher zeroes on the launch stream before every launch),
    # the items numbered group-major (d128_claim_coords). It requires persistent and causal, causal_walk 0 and
    # sched_policy NATURAL, and computes the one-tile-per-CTA overlap kernel's O and Stats bit for bit. SdpaFwdDslSm90
    # sets it from api_dsl.D128_CAUSAL_CLAIM on causal persistent records the route keeps, above
    # api_dsl.D128_CLAIM_MIN_ROUNDS rounds of items per CTA. D512 refuses it.
    causal_claim: int = 0


# The ports of ``TemplateParams.thd_origins``, in order.
THD_ORIGIN_PORTS = ("q", "k", "v", "o", "stats")


def thd_origin_maps(b: int) -> int:
    """Tensor maps of a caller-origin THD plan: per-sequence O in ``0..B-1``, the dead-unit pad at
    ``B``, then per-sequence K at ``B+1..2B`` and V at ``2B+1..3B``."""
    return 3 * b + 1


def thd_origins_off(b: int) -> int:
    """Int32 word offset of the Int64 ``[5, B]`` token origins, right after the maps."""
    from cudnn.frost.tile_dsl.thd import TENSOR_MAP_QWORDS, THD_MAPS_OFF

    return THD_MAPS_OFF(b) + thd_origin_maps(b) * TENSOR_MAP_QWORDS * 2


def thd_meta_words(b: int, origins: bool) -> int:
    """Int32 words of the THD metadata operand: the shared layout with its ``B + 3`` maps or, with
    caller-buffer origins, the ``3B + 1`` maps and then the origins."""
    from cudnn.frost.tile_dsl.thd import THD_MAPS_META_WORDS

    return thd_origins_off(b) + 2 * len(THD_ORIGIN_PORTS) * b if origins else THD_MAPS_META_WORDS(b)


# A band bound at or beyond S_q + S_kv never masks a key inside the declared envelopes, and
# every band bound is added to Int32 token coordinates in the kernel.
BAND_COORDINATE_LIMIT = 2**30


def validate_params(params: TemplateParams, flavor: tuple[int, int] = D512_FLAVOR) -> None:
    """Validate the immutable specialization before constructing a kernel.

    ``flavor`` names the template asking. D512 (the default) runs the checks below
    unchanged, messages included, and refuses the D128 template's overlap, persistent
    and causal walk switches. D128 first refuses its ``NOT_SERVED`` features (the
    declarations pick_flavor keeps on D512, so the adapter never reaches this refusal),
    then persistence without intra-warpgroup overlap, then a causal walk other than 0
    and 1 and the paired walk outside a persistent causal record, then a claim counter
    other than 0, 1 and 2 or outside a persistent causal NATURAL record without the
    walk, then runs the same checks under its own name.
    """
    if flavor == D128_FLAVOR:
        declared = declared_features(
            thd=params.thd_varlen,
            sink=params.has_sink,
            window_left=params.window_left,
            window_right=params.window_right,
            lengths=params.seq_q_lens_present or params.seq_kv_lens_present,
            pack_gqa=params.pack_gqa,
            paged=params.page_size != 0,
            split_kv=params.split_kv,
        )
        for feature, label in D128_NOT_SERVED.items():
            if feature in declared:
                raise ValueError(f"SM90 D128 does not serve {label}; the D512 body (tile 64) serves such declarations")
    elif flavor != D512_FLAVOR:
        raise ValueError(f"SM90 has no {flavor} template; flavors: {sorted(FLAVORS)}")
    if flavor == D512_FLAVOR and params.intra_wg_overlap:
        raise ValueError("SM90 D512 has no intra-warpgroup overlap switch; it is a D128 template field")
    if flavor == D512_FLAVOR and params.persistent:
        raise ValueError("SM90 D512 has no persistent switch; it is a D128 template field")
    if flavor == D512_FLAVOR and params.causal_walk:
        raise ValueError("SM90 D512 has no causal walk switch; it is a D128 template field")
    if params.persistent and not params.intra_wg_overlap:
        raise ValueError("SM90 D128 persistent requires intra_wg_overlap; the v1 loop stays the non-persistent reference")
    if type(params.causal_walk) is not int or params.causal_walk not in (0, 1):
        raise ValueError(f"SM90 D128 causal_walk must be 0 (the stride by the grid) or 1 (the paired walk); got {params.causal_walk!r}")
    if params.causal_walk and not (params.persistent and params.causal):
        raise ValueError("SM90 D128 causal_walk=1 requires persistent and causal: it orders a causal plan's persistent walk")
    if flavor == D512_FLAVOR and params.causal_claim:
        raise ValueError("SM90 D512 has no causal claim counter; it is a D128 template field")
    if type(params.causal_claim) is not int or params.causal_claim not in (D128_CLAIM_OFF, D128_CLAIM_NATURAL):
        raise ValueError(
            f"SM90 D128 causal_claim must be {D128_CLAIM_OFF} (static) or {D128_CLAIM_NATURAL} (the counter, NATURAL); got {params.causal_claim!r}"
        )
    if params.causal_claim and not (params.persistent and params.causal and not params.causal_walk and params.sched_policy == SCHED_NATURAL):
        raise ValueError(
            "SM90 D128 causal_claim requires persistent and causal with causal_walk 0 and sched_policy NATURAL: the counter "
            "replaces the static schedules of a causal plan's persistent walk and numbers its items itself"
        )
    name = f"SM90 D{flavor[0]}"
    tile_m = FLAVOR_TILES[flavor][0]
    if type(params.page_size) is not int or params.page_size < 0:
        raise ValueError("SM90 page_size must be a nonnegative integer")
    if params.page_size:
        p = params.page_size
        if p % 8 or (p < TILE_N and TILE_N % p) or (p >= TILE_N and p % TILE_N):
            raise ValueError("SM90 page_size must be divisible by 8 and divide 64 or be a multiple of 64")
        if params.thd_varlen or not params.seq_kv_lens_present:
            raise ValueError("SM90 paged KV requires dense Q and device KV lengths")
    if type(params.head_tile) is not int or params.head_tile not in (128, 256, 512):
        raise ValueError("SM90 head_tile must be 128, 256 or 512")
    if type(params.split_kv) is not int or not 1 <= params.split_kv <= 65535:
        raise ValueError("SM90 split_kv must be an integer in [1, 65535]")
    if params.thd_varlen and params.split_kv > 1:
        raise ValueError("SM90 split-KV requires dense Q")
    if params.dtype_qkv not in (DTYPE_FP16, DTYPE_BF16):
        raise ValueError(f"{name} requires uniform FP16 or BF16 Q/K/V/O")
    from .config_sm100 import pack_gqa_supported

    if params.qh_per_kh < 1 or (params.pack_gqa and not pack_gqa_supported(params.qh_per_kh, 1, tile_m, partial=True)):
        raise ValueError(f"{name} PackGQA requires a positive head ratio sharing a factor with tile_m={tile_m}")
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{name} supports only NATURAL/LPT/LPT_L2 single-tile scheduling")
    if params.thd_varlen and params.sched_policy != SCHED_NATURAL:
        raise ValueError(f"{name} THD uses the natural single-tile work decoder")
    if params.scale_mode not in (SCALE_POSITIVE, SCALE_ZERO, SCALE_NEGATIVE):
        raise ValueError(f"{name} scale mode must be positive, zero, or negative")
    if params.thd_varlen:
        if not params.seq_kv_lens_present:
            raise ValueError(f"{name} thd_varlen requires seq_kv_lens_present (the THD metadata tensor)")
        if params.seq_q_lens_present:
            raise ValueError(f"{name} seq_q_lens_present is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
    if params.thd_origins is not None:
        if not params.thd_varlen or len(params.thd_origins) != len(THD_ORIGIN_PORTS):
            raise ValueError(f"{name} thd_origins is THD-only and lists Q, K, V, O and Stats")
        for port, entry in zip(THD_ORIGIN_PORTS, params.thd_origins):
            if entry is not None and (len(entry) != 2 or entry[0] not in (32, 64) or type(entry[1]) is not int or entry[1] < 1):
                raise ValueError(f"{name} thd_origins[{port}] must be (32 or 64, multiplier >= 1)")
        if params.thd_origins[4] is not None and not params.has_lse:
            raise ValueError(f"{name} Stats origins require has_lse")
    if params.causal and params.window_right is not None:
        raise ValueError(f"{name} plain causal is causal=True; window_right holds only a widened right bound")
    for bound_name, bound, low in (("window_left", params.window_left, 0), ("window_right", params.window_right, 1)):
        if bound is not None and (type(bound) is not int or not low <= bound < BAND_COORDINATE_LIMIT):
            raise ValueError(f"{name} {bound_name} must be an integer in [{low}, 2**30)")
    if params.bottom_right is not None:
        if params.window_left is None and params.window_right is None and not params.causal:
            raise ValueError(f"{name} diagonal alignment needs a band bound")
        if params.bottom_right == params.causal:
            raise ValueError(f"{name} bottom_right repeats the alignment causal implies; the canonical spelling is None")
