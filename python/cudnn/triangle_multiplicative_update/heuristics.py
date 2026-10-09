# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The triangle multiplicative update's performance knobs and their MEASURED defaults -- the one place the API reads a tile or a tile order from.

Every axis here is performance-only: a plan computes the same function (bitwise, the tests pin it) under any value of its domain, so a caller
or an autotuner may pick freely.  Numerics modes (``affine_fold``, ``fuse_residual`` of the output stage) are NOT knobs: they change the
function and stay op attributes the API never sets.

  ``k2_tile``        the FROST GEMM tile config of the contraction (:data:`K2_TILE_DOMAIN`, catalog names): the 2-CTA 256 x 256 pair tile,
                     the same with the A operand multicast across 2 / 4 CTAs, the 1-CTA 128 x 128 tile, the same with B multicast across 2.
  ``k1_pass_order``  the projection stage's tile order (``"interleaved"``: both passes of a row block adjacent, the second ``x`` read an L2 hit;
                     ``"major"``: every row block of pass 0 before pass 1).
  ``k3_prologue``    the output stage's MN-major LayerNorm form (``"mn_major"`` in place; ``"transpose"`` into the K-major layout).
  ``clc_stages``     the two LayerNorm-prologue stages' scheduler ring depth.
  ``k_block``        the two LayerNorm-prologue stages' row tile: k's per 256-row pair tile (:data:`K_BLOCK_DOMAIN`; the substrate's rule
                     ``default_k_block`` when a record leaves it unset -- 256 from N = 256, else the next power of two >= N).  At N = 384 the
                     rule's 256 pads 25 % of every pair tile's rows; 128 pads none (the lever measured in the perf campaign).
  ``ln_pass_form``   the two stages' LayerNorm pass form (:data:`LN_PASS_FORMS`: ``"registers"`` = the row held in registers between the passes;
                     ``"smem"`` / ``"smem_rolled"`` = the row re-read from the SMEM stage in every pass, unrolled / rolled chunk loops -- the
                     rolled form has no K = 256 spills; bitwise the same output).
  ``epi_math``       the two stages' gated-epilogue arithmetic form (:data:`EPI_MATHS`: ``"scalar"`` / ``"packed"`` fp32x2 pairs; bitwise).

:func:`recommend` is the table -- the best MEASURED configuration per geometry on the 212-SM Rubin perf-node class, keyed by hidden width ``c``,
N bucket, batch bucket and direction: a bucket table (width, N) carries the complete knob set, and a short refinement table overrides ONE axis
where a cell of the measured grid (B in {1, 4}, both directions) beat the bucket's pick decisively in every pass that measured it.  A geometry
nobody timed takes the nearest measured bucket's pick (N: the first floor at or below it; B: 4 from B >= 4, else 1), never a guess outside the
domain.  :func:`resolve` merges a caller's ``k2_tile`` / ``knobs`` over it -- a :class:`TriMulKnobs` is a PARTIAL record: an axis left ``None``
takes the table's value, so ``TriMulKnobs(clc_stages=4)`` changes the ring depth and keeps the geometry's measured tile, tile order and
prologue.  :func:`validate_knobs` is the typed vocabulary check ``check_support`` runs on the resolved record.

The vocabulary is not mirrored into the graph API's ``KnobType_t`` band: this API is a standalone cuEquivariance-signature surface, not a graph
engine, so no autotune record of the form ``(engine_id, {knob_type: int})`` exists for it.  Should it become a graph op, the four axes join the
frontend-only knob band (append-only) and the dataclass converts at that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from cudnn.gemm.frost.ln_prologue import A0_PROLOGUES, EPI_MATHS, LN_PASS_FORMS, MIN_K_BLOCK_MN_MAJOR, PAIR_TILE_ROWS, PASS_ORDERS
from cudnn.gemm.frost.ln_prologue import default_k_block as _substrate_default_k_block

from .kernels import contraction as k2

#: The contraction tile configs the heuristics may pick and ``check_support`` admits (every one a FROST GEMM catalog name; every one measured).
K2_TILE_DOMAIN: Tuple[str, ...] = (
    k2.DESIGN_TILE_CONFIG,
    k2.MULTICAST_A2_TILE_CONFIG,
    k2.MULTICAST_A4_TILE_CONFIG,
    k2.SMALL_N_TILE_CONFIG,
    k2.SMALL_N_MULTICAST_B2_TILE_CONFIG,
)
#: The scheduler ring depths of the LayerNorm-prologue stages that were measured (the substrate admits 1..8).
CLC_STAGES_DOMAIN: Tuple[int, ...] = (1, 2, 4)
#: The row tiles (k's per 256-row pair tile) BOTH stages serve for every N the API admits: the output stage's MN-major box needs >= 64, the pair
#: tile is 256 -- the powers of two between (the projection stage alone admits 8).
K_BLOCK_DOMAIN: Tuple[int, ...] = tuple(1 << i for i in range(MIN_K_BLOCK_MN_MAJOR.bit_length() - 1, PAIR_TILE_ROWS.bit_length()))
#: The batch sizes the table was measured at; :func:`batch_bucket` maps any B onto the nearest one below (B >= 4 -> 4, else 1).
BATCH_BUCKETS: Tuple[int, ...] = (1, 4)
#: The c = 128 geometry's knob values below N = 512 (the API's K1 tile-order pin reads the first); :func:`recommend` is the table that decides.
DEFAULT_K1_PASS_ORDER = "interleaved"
DEFAULT_K3_PROLOGUE = "mn_major"
DEFAULT_CLC_STAGES = 2
#: The substrate forms every bucket pick carries today (the substrate's own defaults): the re-read / packed forms are selectable through
#: ``TriMulKnobs(ln_pass_form=..., epi_math=...)`` and flip here per width only with a measured A/B/A on the perf-node class.
_LN_FORM_DEFAULT = "registers"
_EPI_MATH_DEFAULT = "scalar"


@dataclass(frozen=True)
class TriMulKnobs:
    """One knob assignment -- COMPLETE when every axis is set (what a plan runs; :func:`recommend` returns one), PARTIAL when an axis is ``None``
    (a caller's override: the unset axes take the measured table's values in :func:`resolve`).  Performance only: the output is the same function
    under every member of every domain."""

    k2_tile: Optional[str] = None
    k1_pass_order: Optional[str] = None
    k3_prologue: Optional[str] = None
    clc_stages: Optional[int] = None
    k_block: Optional[int] = None  # appended (2026-10-09): the row tile of the two LayerNorm-prologue stages
    #: appended axes (the two LayerNorm-prologue stages share them): the LayerNorm pass form and the epilogue arithmetic form
    ln_pass_form: Optional[str] = None
    epi_math: Optional[str] = None

    def is_complete(self) -> bool:
        return None not in (self.k2_tile, self.k1_pass_order, self.k3_prologue, self.clc_stages, self.k_block, self.ln_pass_form, self.epi_math)

    def merged_over(self, base: "TriMulKnobs") -> "TriMulKnobs":
        """This record's SET axes over ``base`` (the unset ones keep the base's values)."""
        return TriMulKnobs(
            k2_tile=self.k2_tile if self.k2_tile is not None else base.k2_tile,
            k1_pass_order=self.k1_pass_order if self.k1_pass_order is not None else base.k1_pass_order,
            k3_prologue=self.k3_prologue if self.k3_prologue is not None else base.k3_prologue,
            clc_stages=self.clc_stages if self.clc_stages is not None else base.clc_stages,
            k_block=self.k_block if self.k_block is not None else base.k_block,
            ln_pass_form=self.ln_pass_form if self.ln_pass_form is not None else base.ln_pass_form,
            epi_math=self.epi_math if self.epi_math is not None else base.epi_math,
        )


def validate_knobs(knobs: TriMulKnobs) -> None:
    """Typed vocabulary check (``ValueError`` names the axis and its domain); a performance knob outside its domain is a malformed declaration.
    An axis left ``None`` is a partial record's "take the table's value" and passes; the record a plan runs is complete (``resolve`` fills it)."""
    if not isinstance(knobs, TriMulKnobs):
        raise TypeError(f"knobs must be a TriMulKnobs, got {type(knobs).__name__}")
    if knobs.k2_tile is not None and knobs.k2_tile not in K2_TILE_DOMAIN:
        raise ValueError(f"k2_tile must be one of the measured contraction tile configs {K2_TILE_DOMAIN}, got {knobs.k2_tile!r}")
    if knobs.k1_pass_order is not None and knobs.k1_pass_order not in PASS_ORDERS:
        raise ValueError(f"k1_pass_order must be one of {PASS_ORDERS}, got {knobs.k1_pass_order!r}")
    if knobs.k3_prologue is not None and knobs.k3_prologue not in A0_PROLOGUES:
        raise ValueError(f"k3_prologue must be one of {A0_PROLOGUES}, got {knobs.k3_prologue!r}")
    if knobs.clc_stages is not None and knobs.clc_stages not in CLC_STAGES_DOMAIN:
        raise ValueError(f"clc_stages must be one of {CLC_STAGES_DOMAIN}, got {knobs.clc_stages!r}")
    if knobs.k_block is not None and (isinstance(knobs.k_block, bool) or knobs.k_block not in K_BLOCK_DOMAIN):
        raise ValueError(f"k_block must be one of {K_BLOCK_DOMAIN} (k's per 256-row pair tile of the two LayerNorm-prologue stages), got {knobs.k_block!r}")
    if knobs.ln_pass_form is not None and knobs.ln_pass_form not in LN_PASS_FORMS:
        raise ValueError(f"ln_pass_form must be one of {LN_PASS_FORMS}, got {knobs.ln_pass_form!r}")
    if knobs.epi_math is not None and knobs.epi_math not in EPI_MATHS:
        raise ValueError(f"epi_math must be one of {EPI_MATHS}, got {knobs.epi_math!r}")


# THE TABLE, part 1 -- the BUCKET picks.  Per hidden width ``c``: descending N floors, each with the COMPLETE knob set measured best over the bucket's
# four cells (B in {1, 4} x both directions; the first floor <= N wins; below the last floor the last entry serves).  Measured on the 212-SM Rubin
# perf-node class (lock 2376 MHz, sampled 2364), with a control pair per cell, in three passes: a discovery pass varying one axis at a time over the
# previous defaults, a verification pass with the composed table as the base re-measuring every alternative, and a table-of-record pass with the
# refined table as the base.  An axis flips only where its alternative beat the base in every cell of the bucket by more than max(0.5 %, 2 x the
# control-pair spread); the tile is the one with the smallest worst-case regret over the bucket when it beats the base tile by that margin.  Two
# interactions the one-axis pass could not see came out of the verification pass: at c = 128 from N >= 512 the projection stage's pass-MAJOR order
# wins +1.3..+1.5 % once the ring is 1 deep and the output prologue is the transpose form (interleaved had won +4 % standalone at the 2-deep ring),
# and at N <= 128 the transpose prologue adds +3.1..+4.3 % on top of the 1-CTA tile (three of four cells decisive, the fourth +0.84 % under a 1.19 %
# control spread -- adopted by judgment).
_PICKS: Dict[int, Tuple[Tuple[int, TriMulKnobs], ...]] = {
    128: (
        (1024, TriMulKnobs(k2_tile=k2.MULTICAST_A4_TILE_CONFIG, k1_pass_order="major", k3_prologue="transpose", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (768, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="transpose", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (512, TriMulKnobs(k2_tile=k2.MULTICAST_A2_TILE_CONFIG, k1_pass_order="major", k3_prologue="transpose", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (384, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (256, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=2, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (129, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (0, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="interleaved", k3_prologue="transpose", clc_stages=2, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
    ),
    256: (
        (1024, TriMulKnobs(k2_tile=k2.MULTICAST_A4_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (768, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (512, TriMulKnobs(k2_tile=k2.MULTICAST_A2_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (384, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (256, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (129, TriMulKnobs(k2_tile=k2.DESIGN_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=1, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
        (0, TriMulKnobs(k2_tile=k2.SMALL_N_TILE_CONFIG, k1_pass_order="major", k3_prologue="mn_major", clc_stages=2, ln_pass_form=_LN_FORM_DEFAULT, epi_math=_EPI_MATH_DEFAULT)),
    ),
}

# THE TABLE, part 2 -- the per-CELL refinements: (N floor of the bucket, batch bucket, direction or None = both) -> ONE axis of the bucket pick
# overridden per row.  A refinement exists where one single-axis alternative beat the bucket pick DECISIVELY (> max(0.5 %, 2 x the control-pair
# spread)) in that cell of the table-of-record pass AND with the same sign in every earlier pass that measured both knob sets, and whose gain
# REPRODUCED when the refined table was measured as the base; the bucket keeps its pick where the other batch or direction lost.  Keys may repeat
# across rows only for different axes (pinned by the tests).  Margins: +X % = t_bucket_pick / t_refined - 1 (the pass that found it; as the base).
#   c = 128, N <= 128, B = 1, incoming:       ring depth 1   +1.87 % (as base +1.75 %; the outgoing twin -0.34 %, B = 4 -0.5..-0.6 %)
#   c = 128, 256 <= N < 384, B >= 4:          ring depth 1   +1.95 / +1.87 % (as base +1.97 / +2.01 %; B = 1 loses 1.5 %), and ON TOP of it
#                                             pass-MAJOR K1  +1.26 / +1.33 % (the ring-depth interaction of the N >= 512 buckets: at depth 1 the
#                                             adjacent-pass order loses what the shared x read gains; at depth 2 the two orders tied)
#   c = 128, 384 <= N < 512, B = 1:           ring depth 4   +1.02 / +1.04 % (as base +0.93 / +1.04 %; B = 4 loses 1.9 %)
#   c = 256, N <= 128, B >= 4:                ring depth 1   +0.53 / +0.57 % (as base +0.59 / +0.71 %; B = 1 loses 3.4 %)
# Not adopted: the 1-CTA tile with B multicast at c = 128, 256 <= N < 384, B = 1 -- +0.5..+0.6 % over the pair tile as an arm in three passes but
# +0.26 % inside a 0.53 % control spread when measured as the base (a light-load cell at the 2352 MHz step); the pair tile stays, the residue is
# at most 0.6 % on those two cells.
_REFINEMENTS: Dict[int, Tuple[Tuple[int, int, Optional[str], TriMulKnobs], ...]] = {
    128: (
        (0, 1, "incoming", TriMulKnobs(clc_stages=1)),
        (256, 4, None, TriMulKnobs(clc_stages=1)),
        (256, 4, None, TriMulKnobs(k1_pass_order="major")),
        (384, 1, None, TriMulKnobs(clc_stages=4)),
    ),
    256: ((0, 4, None, TriMulKnobs(clc_stages=1)),),
}


def _nearest_width(c: int) -> int:
    widths = sorted(_PICKS)
    return min(widths, key=lambda w: (abs(w - int(c)), w))


def batch_bucket(batch: int) -> int:
    """The measured batch bucket a batch size falls in: the largest member of :data:`BATCH_BUCKETS` at or below it (1 below 4)."""
    b = int(batch)
    return max((x for x in BATCH_BUCKETS if x <= b), default=BATCH_BUCKETS[0])


def _bucket_floor(n_res: int, table: Tuple[Tuple[int, TriMulKnobs], ...]) -> Tuple[int, TriMulKnobs]:
    for floor, knobs in table:
        if n_res >= floor:
            return floor, knobs
    return table[-1]


def k2_tile_for(n_res: int, channels: int = 128, batch: int = 1, direction: str = "outgoing") -> str:
    """The contraction's tile config for ``(N, c, B, direction)`` from the measured table."""
    return recommend(n_res, channels, batch, direction).k2_tile


def default_k_block(n_res: int) -> int:
    """The row tile a bare declaration runs at ``n_res``: the substrate's rule (``cudnn.gemm.frost.ln_prologue.default_k_block``: 256 from N = 256,
    else the next power of two >= N) held inside :data:`K_BLOCK_DOMAIN` (below N = 33 the API declines before any launch; the clamp keeps the record in
    the vocabulary there) -- EXCEPT where the 128-row block computes fewer rows than the 256 one (``N mod 256`` in 1..128 from N = 256: at N = 384 the
    rule's 256 pads 25 % of every pair tile, 128 pads none), measured on the 212-SM Rubin perf-node class as the faster tile there (the lever's record:
    N = 384, c = 128 and 256, B = 1 and 4, both directions; the per-CTA TMA box is the same as 256's, only the scheduler walks fewer tiles)."""
    n = int(n_res)
    kb = max(_substrate_default_k_block(n), K_BLOCK_DOMAIN[0])
    if n >= PAIR_TILE_ROWS and -(-n // 128) * 128 < -(-n // PAIR_TILE_ROWS) * PAIR_TILE_ROWS:
        kb = 128
    return kb


def recommend(n_res: int, channels: int = 128, batch: int = 1, direction: str = "outgoing") -> TriMulKnobs:
    """The measured default knob set for this geometry: the (width, N bucket) pick with the (batch bucket, direction) refinements applied, the
    row tile from :func:`default_k_block` unless a pick or refinement sets it -- complete (every axis set)."""
    if direction not in k2.DIRECTIONS:
        raise ValueError(f"direction must be one of {k2.DIRECTIONS}, got {direction!r}")
    n_res = int(n_res)
    width = _nearest_width(channels)
    floor, knobs = _bucket_floor(n_res, _PICKS[width])
    bb = batch_bucket(batch)
    for r_floor, r_batch, r_direction, override in _REFINEMENTS.get(width, ()):
        if r_floor == floor and r_batch == bb and (r_direction is None or r_direction == direction):
            knobs = override.merged_over(knobs)
    if knobs.k_block is None:
        knobs = replace(knobs, k_block=default_k_block(n_res))
    return knobs


def resolve(n_res: int, channels: int, batch: int, direction: str, *, k2_tile: Optional[str] = None, knobs: Optional[TriMulKnobs] = None) -> TriMulKnobs:
    """The knob set a plan runs: :func:`recommend` for the geometry, then the caller's ``knobs`` (its SET axes) and ``k2_tile`` (one axis) over
    it -- always a complete record.  No validation here: ``check_support`` runs :func:`validate_knobs` so an out-of-vocabulary name is a typed
    decline at the usual place (a ``knobs`` of the wrong type raises there too)."""
    out = recommend(n_res, channels, batch, direction)
    if knobs is not None and isinstance(knobs, TriMulKnobs):
        out = knobs.merged_over(out)
    elif knobs is not None:
        return knobs  # not a TriMulKnobs: handed through for validate_knobs to raise the TypeError
    if k2_tile is not None:
        out = replace(out, k2_tile=str(k2_tile))
    return out


__all__ = [
    "BATCH_BUCKETS",
    "CLC_STAGES_DOMAIN",
    "DEFAULT_CLC_STAGES",
    "DEFAULT_K1_PASS_ORDER",
    "DEFAULT_K3_PROLOGUE",
    "K2_TILE_DOMAIN",
    "K_BLOCK_DOMAIN",
    "TriMulKnobs",
    "batch_bucket",
    "default_k_block",
    "k2_tile_for",
    "recommend",
    "resolve",
    "validate_knobs",
]
