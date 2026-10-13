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
                     rolled form has no K = 256 spills; ``"smem_rolled_pf"`` = the rolled form with the next chunk's loads pipelined one
                     iteration ahead; all bitwise the same output.  ``"smem_rolled_tree"`` = the pipelined form with four partial sums per
                     statistics pass: a different fp32 summation order (:data:`LN_PASS_FORMS_REORDERED`), admitted as a caller's knob, never
                     a pick of this table).
  ``epi_math``       the two stages' gated-epilogue arithmetic form (:data:`EPI_MATHS`: ``"scalar"`` / ``"packed"`` fp32x2 pairs; bitwise).
  ``min_blocks_per_mp``  the two stages' ``.minnctapersm`` launch hint (:data:`MIN_BLOCKS_PER_MP_DOMAIN`: 0 = the rendering as shipped, its
                     register split dropped by ptxas; 1 = the hint under which ptxas honours the split -- bitwise the same output).  An OVERRIDE
                     axis only: ``None`` (the table never sets it) means the per-RENDERING rule :func:`register_split_hint`, applied at compile
                     time with each stage's bias flag, because the fact it answers -- does this rendering spill at the flat allocation? -- is a
                     property of (kernel, width, bias arm, LayerNorm / epilogue form), not of N or B.
  ``ln_once``        the two LayerNorm-prologue stages' work item (:data:`LN_ONCE_FORMS`: ``""`` = one GEMM tile, the row block's A rows re-streamed
                     and re-normalised per (pass, N tile); ``"block"`` = the whole row block held resident in SMEM and LayerNormed ONCE across its
                     passes x N tiles; ``"passes"`` = (row block, one N tile) across the passes -- the finer persistent tail for small N; bitwise the
                     same output).  The output stage runs one pass, so it takes ``"block"`` at c = 256 only (two N tiles) and keeps today's item
                     elsewhere (the substrate's typed decline of a one-tile item, resolved in the API's stage-parameter builder, stated); a geometry
                     axis of the table (the persistent tail is a function of N and B), OFF in every pick until its A/B/A lands a per-bucket default.

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

from cudnn.gemm.frost.ln_prologue import (
    A0_PROLOGUES,
    EPI_MATHS,
    EPI_MATHS_PACKED,
    LN_ONCE_FORMS,
    LN_PASS_FORMS,
    MIN_BLOCKS_PER_MP_DOMAIN,
    MIN_K_BLOCK_MN_MAJOR,
    PAIR_TILE_ROWS,
    PASS_ORDERS,
)
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
#: The c = 128 geometry's knob values below N = 256 (the API's K1 tile-order pin reads the first; from N = 256 the c = 128 picks take the pass-major
#: order with the LayerNorm-once item); :func:`recommend` is the table that decides.
DEFAULT_K1_PASS_ORDER = "interleaved"
DEFAULT_K3_PROLOGUE = "mn_major"
DEFAULT_CLC_STAGES = 2


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
    #: appended (2026-10-13): the two stages' ``.minnctapersm`` launch hint -- an OVERRIDE axis the table never sets: ``None`` = the per-rendering
    #: rule :func:`register_split_hint` (keyed by each stage's bias flag at compile time), so it is not one of the geometry axes ``is_complete`` names
    min_blocks_per_mp: Optional[int] = None
    #: appended (2026-10-13, wave 2): the two stages' LayerNorm-once-per-row-block work item (:data:`LN_ONCE_FORMS`; "" = today's per-tile item) -- a
    #: GEOMETRY axis (the persistent tail it coarsens is a function of N and B), so ``is_complete`` names it and every pick carries a value
    ln_once: Optional[str] = None

    def is_complete(self) -> bool:
        return None not in (self.k2_tile, self.k1_pass_order, self.k3_prologue, self.clc_stages, self.k_block, self.ln_pass_form, self.epi_math, self.ln_once)

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
            min_blocks_per_mp=self.min_blocks_per_mp if self.min_blocks_per_mp is not None else base.min_blocks_per_mp,
            ln_once=self.ln_once if self.ln_once is not None else base.ln_once,
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
    if knobs.min_blocks_per_mp is not None and (isinstance(knobs.min_blocks_per_mp, bool) or knobs.min_blocks_per_mp not in MIN_BLOCKS_PER_MP_DOMAIN):
        raise ValueError(
            f"min_blocks_per_mp must be one of {MIN_BLOCKS_PER_MP_DOMAIN} (the two LayerNorm-prologue stages' .minnctapersm launch hint: 0 = the rendering "
            f"as shipped, 1 = ptxas honours the body's register split), got {knobs.min_blocks_per_mp!r}"
        )
    if knobs.ln_once is not None and knobs.ln_once not in LN_ONCE_FORMS:
        raise ValueError(
            f"ln_once must be one of {LN_ONCE_FORMS} (the two LayerNorm-prologue stages' work item: '' = one GEMM tile, 'block' = the row block "
            f"LayerNormed once across its passes x N tiles, 'passes' = across its passes at one N tile), got {knobs.ln_once!r}"
        )


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
#
# The two substrate FORMS (the LayerNorm pass form and the epilogue arithmetic form; both bitwise the register / scalar rendering) were measured as
# a separate pass on the two LayerNorm-prologue kernels STANDALONE (the projection stage K1 and the output stage K3, graph replay, 100-400 x 3,
# control pair, every arm bitwise equal to the base) over N in {128, 217, 256, 384, 512, 768, 1024} x B in {1, 4} per width; the per-width signs:
#   c = 256: the row RE-READ with ROLLED chunk loops (``ln_pass_form="smem_rolled"``) is the whole register-residency story -- K1 +178..+199 %
#            and K3 +140..+146 % over the register form on every N >= 256 cell of both passes (K1 N = 512: 683 -> 234 us; the K = 256 renderings
#            spilled 240 / 241 and 499 / 504 words per lane, the rolled form 0 / 0 at REG 112-114), and the packed epilogue ON TOP of it adds
#            +1.4..+2.3 % (K1) / +0.4..+0.9 % (K3) more, outside every such cell's control spread -- although packed ALONE loses 1.1..2.9 % against
#            the register form (the epilogue is not the binding stage until the spills are gone).  Every c = 256 pick: rolled + packed.
#   c = 128: the packed epilogue (``epi_math="packed"``) wins or ties on every cell of both kernels (K1 +4.8..+37 %; K3 +2.0..+11.9 % with two
#            neutral cells, +0.0 / -0.1 % at N 217 / 384 B = 1 on the second pass: no spills here, half the ALU issue of the gated epilogue); the
#            rolled re-read form is a per-BUCKET sign on K1 (K3 runs the transpose prologue at c = 128, under which the substrate serves the
#            register form only, so the knob reaches the projection stage alone): over the packed-only pick it is +3.7 / +3.7 % at N = 256 B = 1
#            (-1.8 / -1.7 % at B = 4: the refinement below), +17.1 / +12.4 % at N = 384 B = 1 (the second pass at the 128-row tile the API runs
#            there) and +9.7 / +2.1 % at B = 4, -1.8 / -1.4 % at N = 512 B = 1 (+0.7 / +0.7 % at B = 4: the second refinement below), +0.1 /
#            +0.1 % at N = 768 (neutral: the register form stays) and +0.68 / +0.66 % at N = 1024 B = 1 (the 0.5 % threshold met only barely)
#            with +1.3 % at B = 4 (both cells of the bucket, two passes: the rolled form), and a LOSS of 3.8 / 4.0 and 8.6 / 6.5 % at N = 217 /
#            128 (three SMEM passes with nothing to buy) -- two passes on two nodes of the class, the same sign on every cell.  The unrolled re-read form
#            ("smem") is not in the table: -4.5..-12.7 % at c = 128 and a third of the rolled form's gain at c = 256 (its normalise pass
#            still spills half).
#
# The PIPELINED rolled form (`ln_pass_form="smem_rolled_pf"`: the rolled re-read with chunk (q + 1)'s loads carried one loop iteration ahead -- the same
# words into the same additions in the same order, bitwise), measured STANDALONE on both LayerNorm-prologue kernels over the heuristics' complete pick
# (graph replay, 100 x 3, control pair <= 0.4 %, every arm bitwise the pick, the 212-SM Rubin class at the 2376 lock; bias off AND on):
#   c = 256: K1 +2.9 / +2.9 / +3.0 / +3.2 % (bias off) and +5.2 / +5.1 / +5.2 / +5.6 % (bias on) at N 512 x 1 / 384 x 1 / 256 x 4 / 1024 x 1, K3
#            +32.3 / +30.2 / +32.5 / +33.5 % (bias off: the shipped rolled K3 rendering spills at REG 80 without the biases, 298 -> 226 us at N 512)
#            and +21.9 / +20.2 / +21.9 / +22.5 % (bias on, no spill in either: 278 -> 228 us) -- every cell of every measured bucket, the same sign in
#            both bias arms, outside 10x the control spread.  Every c = 256 pick: pipelined rolled + packed.
#   c = 128: MIXED by cell and by bias arm on K1 (the only stage the knob reaches there; K3's transpose prologue keeps the register form): -1.4 % (off) /
#            +0.6 % (on) at N 1024 x 1, +0.9 / +2.6 % at N 384 x 1, and -11 / -0.4 % against the REGISTER picks at N 512 x 1 and 256 x 4 (the register
#            form + packed epilogue leads every rolled form there by 9-11 % without the biases) -- no bucket flips; the register / rolled picks stand.
# The tree form (`"smem_rolled_tree"`: the pipelined form with four partial sums per statistics pass -- a different fp32 ORDER, LN_PASS_FORMS_REORDERED)
# adds +0.2..+3.5 % on top (K3 c 256 +25.0..+25.3 % vs the pick with the biases, +32.5..+36.0 % without; K1 c 256 +5.3..+6.1 %) and is NEVER a pick of
# this table: a caller's knob, reported with its oracle distance.
#
# The LayerNorm-ONCE work item (`ln_once="block"`: the row block's A operand(s) resident in SMEM across its passes x N tiles, LayerNormed once; the
# per-tile B ring unchanged; bitwise the per-tile item), measured STANDALONE on both stages over the heuristics' complete pick (graph replay 300 x 3,
# control pair <= 0.2 %, every arm bitwise the pick, the 212-SM Rubin class at the 2376 lock; bias off AND on; the busy clock sampled per cell, the
# 1.0-1.5 %-under-the-lock windows read as LOWER bounds):
#   c = 256: K1 +118 / +118 / +139 / +98 / +72 % (bias off) and +105 / +105 / +120 / +88 / +64 % (bias on) at N 512 x 1 / 256 x 4 / 1024 x 1 / 384 x 1 /
#            256 x 1 (219 -> 101 us at 512 x 1 off); K3 +60 / +60 / +68 / +48 / +31 % (off) and +60 / +60 / +68 / +49 / +32 % (on) (223 -> 139 us) --
#            every cell of every bucket, both bias arms, decisively; the small-N tail of the coarser item (256 x 1: 2.4 item-waves) costs less than the
#            LayerNorm it removes.  Every c = 256 pick: ln_once="block" (the "passes" item, LN 4 -> 2 per block, takes roughly half the gain: K1 +46..+80 %).
#   c = 128: K1 only (the output stage's one-tile item is the substrate's typed decline there), on the buckets whose pick is the pass-MAJOR order:
#            +43 / +44 / +53 / +53 % (bias off) and +45 / +45 / +49 / +53 % (bias on) at N 512 x 1 / 256 x 4 / 1024 x 4 / 768 x 1 (65 -> 45 us at 512 x 1
#            off; 1000 x 3, CLEAN windows) -> ln_once="block" on the 512 / 768 / 1024 floors and on the (256, B >= 4) refinement (whose order is major);
#            the buckets picking the INTERLEAVED order (256 x 1, 384) cannot take the lever as is (the pass must live inside the item: a typed decline);
#            their candidate ln_once="block" + the major order, measured against the interleaved pick WITH the major order alone as the control: K1
#            standalone +16.7 / +15.0 % at 256 x 1, +30.4 / +32.4 % at 384 x 1, +44.5 / +41.9 % at 384 x 4 (bias off / on) where the major order alone
#            reads -0.3..-2.6 % (the lever carries it); the whole call +8.1 / +8.1 % at 256 x 128 x 1 and +12.5 / +13.4 % at 384 x 128 x 1 -> the 256
#            and 384 floors take ("major", ln_once="block"); the 129 and 0 floors (N < 256: 0.6-1.2 tile-waves at B 1, the coarser item most exposed)
#            stay as shipped until measured.
def _pick(k2_tile: str, k1_pass_order: str, k3_prologue: str, clc_stages: int, ln_pass_form: str, epi_math: str, ln_once: str = "") -> TriMulKnobs:
    """One COMPLETE bucket pick in axis order (the row tile ``k_block`` is left to :func:`default_k_block`; the LayerNorm-once item ``ln_once`` is OFF
    unless a bucket's A/B/A set it)."""
    return TriMulKnobs(
        k2_tile=k2_tile,
        k1_pass_order=k1_pass_order,
        k3_prologue=k3_prologue,
        clc_stages=clc_stages,
        ln_pass_form=ln_pass_form,
        epi_math=epi_math,
        ln_once=ln_once,
    )


# THE BIAS-PATH EPILOGUE FORM (`epi_math="packed_biasfold"`: the gate bias staged halved and folded into the tanh argument's fma, -16 FADD2 per
# 32-channel subtile per lane on the renderings WITH biases; OUTPUT-BITWISE the packed form; byte-identical to it WITHOUT biases), measured as a
# whole-call A/B/A on one exclusive GPU of the 212-SM Rubin perf-node class (lock 2376; in-process arm + control twins, slots shuffled, graph
# replay >= 100 x 3, the arm bitwise the base on every cell, the fp64 oracle within its element budget).  Its sign follows the projection
# stage's RENDERING, not N or B: a WIN wherever K1 runs the `registers` LayerNorm form (the epilogue-bound rendering), a LOSS of ~0.7 % wherever
# K1 runs a rolled form -- so the pick below follows the LayerNorm form per (bucket, batch): the 768 bucket (both batches) and the 512 bucket
# at B = 1 take the fold, the (512, B >= 4) refinement (rolled K1) keeps `packed`, the (256, B >= 4) refinement (register K1) takes the fold;
# the 1024 and 384 buckets (rolled K1), the 256 bucket at B = 1 (interleaved rolled, neutral) and the unmeasured N < 256 buckets keep `packed`.
# the fold vs the shipped pick (whole call, bias ON, one exclusive GPU of the 212-SM Rubin perf-node class): 512:128:1 +1.08 % outgoing / +0.98 % incoming, 768:128:1 +1.62 %, 256:128:4 +1.07 %, 768:128:4 +1.73 % (register-form K1: WIN); 1024:128:1 -0.68 %, 512:128:4 -0.68 %, 1024:128:4 -0.63 %, 384:128:4 -0.91 % (rolled-form K1: LOSS -- the CUPTI split puts it in K1, 260 -> 268 us at 1024:128:1, K3 gains 222 -> 218); 256:128:1 +0.06 %, 384:128:1 +0.23 % (interleaved: NEUTRAL); the 11 bias-OFF twins -0.12..+0.07 % (byte-identical renderings)
# the hoisted-load half alone (packed_hoist) -0.16..+0.47 % everywhere: neutral
_PICKS: Dict[int, Tuple[Tuple[int, TriMulKnobs], ...]] = {
    128: (
        (1024, _pick(k2.MULTICAST_A4_TILE_CONFIG, "major", "transpose", 1, "smem_rolled", "packed", ln_once="block")),
        (768, _pick(k2.DESIGN_TILE_CONFIG, "major", "transpose", 1, "registers", "packed_biasfold", ln_once="block")),
        (512, _pick(k2.MULTICAST_A2_TILE_CONFIG, "major", "transpose", 1, "registers", "packed_biasfold", ln_once="block")),
        (384, _pick(k2.SMALL_N_TILE_CONFIG, "major", "transpose", 1, "smem_rolled", "packed", ln_once="block")),
        (256, _pick(k2.DESIGN_TILE_CONFIG, "major", "transpose", 2, "smem_rolled", "packed", ln_once="block")),
        (129, _pick(k2.DESIGN_TILE_CONFIG, "interleaved", "transpose", 1, "registers", "packed")),
        (0, _pick(k2.SMALL_N_TILE_CONFIG, "interleaved", "transpose", 2, "registers", "packed")),
    ),
    256: (
        (1024, _pick(k2.MULTICAST_A4_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", ln_once="block")),
        (768, _pick(k2.DESIGN_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", ln_once="block")),
        (512, _pick(k2.MULTICAST_A2_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", ln_once="block")),
        (384, _pick(k2.SMALL_N_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", ln_once="block")),
        (256, _pick(k2.DESIGN_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", ln_once="block")),
        (129, _pick(k2.DESIGN_TILE_CONFIG, "major", "mn_major", 1, "smem_rolled_pf", "packed", ln_once="block")),
        (0, _pick(k2.SMALL_N_TILE_CONFIG, "major", "mn_major", 2, "smem_rolled_pf", "packed", ln_once="block")),
    ),
}

# THE TABLE, part 2 -- the per-CELL refinements: (N floor of the bucket, batch bucket, direction or None = both) -> ONE axis of the bucket pick
# overridden per row.  A refinement exists where one single-axis alternative beat the bucket pick DECISIVELY (> max(0.5 %, 2 x the control-pair
# spread)) in that cell of the table-of-record pass AND with the same sign in every earlier pass that measured both knob sets, and whose gain
# REPRODUCED when the refined table was measured as the base; the bucket keeps its pick where the other batch or direction lost.  Keys may repeat
# across rows only for different axes (pinned by the tests).  Margins: +X % = t_bucket_pick / t_refined - 1 (the pass that found it; as the base).
#   c = 128, N <= 128, B = 1, incoming:       ring depth 1   +1.87 % (as base +1.75 %; the outgoing twin -0.34 %, B = 4 -0.5..-0.6 %)
#   c = 128, 256 <= N < 384, B >= 4:          ring depth 1   +1.95 / +1.87 % (as base +1.97 / +2.01 %; B = 1 loses 1.5 %), and ON TOP of it
#                                             the REGISTER LN form: the rolled re-read form loses 1.8 / 1.7 % against the packed-only pick on
#                                             this cell (standalone K1, two passes; the same work as N = 512 B = 1, where it loses 1.8 / 1.4 %)
#                                             while B = 1 gains 3.7 / 3.7 %.  (The pass-MAJOR order this cell had as a refinement, +1.26 / +1.33 %
#                                             over the bucket's interleaved pick, is the BUCKET's pick since the LayerNorm-once item landed:
#                                             +43.8 / +45.2 % standalone K1 here, bias off / on, with the major order the item requires.)
#   c = 128, 384 <= N < 512, B = 1:           ring depth 4   +1.02 / +1.04 % (as base +0.93 / +1.04 %; B = 4 loses 1.9 %)
#   c = 128, 512 <= N < 768, B >= 4:          ROLLED LN form +0.7 / +0.7 % over the packed-only pick (standalone K1, two passes; B = 1 loses
#                                             1.8 / 1.4 % -- the sign follows the amount of work: 512 x 4 computes the rows of 1024 x 1, where
#                                             the rolled form wins too)
#   c = 256, N <= 128, B >= 4:                ring depth 1   +0.53 / +0.57 % (as base +0.59 / +0.71 %; B = 1 loses 3.4 %)
#   c = 128, 256 <= N < 384, B >= 4:          the BIAS-PATH epilogue fold (`packed_biasfold`) +1.07 % on the whole call with the biases (this
#                                             cell's K1 is the register form; the B = 1 cell's interleaved rolled K1 is neutral, +0.06 %)
#   c = 128, 512 <= N < 768, B >= 4:          `packed` kept over the bucket's fold pick: -0.68 % with the biases on this cell's ROLLED K1
#                                             (B = 1, the register form, +1.08 / +0.98 % outgoing / incoming)
# Not adopted: the 1-CTA tile with B multicast at c = 128, 256 <= N < 384, B = 1 -- +0.5..+0.6 % over the pair tile as an arm in three passes but
# +0.26 % inside a 0.53 % control spread when measured as the base (a light-load cell at the 2352 MHz step); the pair tile stays, the residue is
# at most 0.6 % on those two cells.
_REFINEMENTS: Dict[int, Tuple[Tuple[int, int, Optional[str], TriMulKnobs], ...]] = {
    128: (
        (0, 1, "incoming", TriMulKnobs(clc_stages=1)),
        (256, 4, None, TriMulKnobs(clc_stages=1)),
        (256, 4, None, TriMulKnobs(ln_pass_form="registers")),
        (384, 1, None, TriMulKnobs(clc_stages=4)),
        (512, 4, None, TriMulKnobs(ln_pass_form="smem_rolled")),
        (256, 4, None, TriMulKnobs(epi_math="packed_biasfold")),
        (512, 4, None, TriMulKnobs(epi_math="packed")),
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


# THE RENDERING RULE -- the `.minnctapersm` launch hint per (stage, width, bias arm, LayerNorm form, epilogue form).  Not a geometry axis: the fact it
# answers is whether THIS rendering spills at the flat register allocation ptxas picks when the body's setmaxnreg split is dropped (no `.maxnreg`,
# no `.minnctapersm`: C7508).  The host SASS of every production rendering (sm_107a trace-compile, nvdisasm) says where: every bias-OFF rendering
# is allocated EXACTLY 80 registers (= 65536 / (2 x 384 threads): ptxas's two-CTAs-per-SM target, reached with a few spills when the body's demand
# sits near it) while every bias-ON one is allocated its demand (97-114) -- and the one rendering whose spills reach the hot path is the output
# stage at c = 256 with the rolled LayerNorm form and no output biases: REG 80 / STACK 24, 5 STL / 7 LDL = three swizzled staging-store addresses
# and two row-decode constants reloaded per tile ahead of the epilogue's TMEM loads (69.8 k local requests in ncu), +7.0..+8.0 % on the kernel
# against its bias-ON twin on all nine c = 256 cells of the matrix, +3.0..+4.0 % on the whole call.  The hint lifts C7508 (the split honoured:
# USETMAXREG 6, the epilogue + LayerNorm warps at 232, the producers at 24; the cubin byte-identical to a `.maxnreg 168` build); a row enters this
# table only with its A/B/A on the 212-SM Rubin perf-node class (the same protocol as the geometry table), and the default 0 keeps every other
# production rendering's cubin byte-identical to the one measured before the axis existed.
# Keyed (stage, width, has_bias, ln_pass_form, epi_math, pass_order, ln_once) -- the projection stage's tile ORDER is part of the rendering (the
# `interleaved` K1 renderings carry 4 STL / 13 LDL of role-prologue values UNDER the hint where the `major` ones carry none); the output stage has no
# order (None); the LayerNorm WORK ITEM (`ln_once`) is part of it too (below: every row names the per-tile item).
# MEASURED ROWS (the A/B/A of the rule as it lands; the bias-OFF rows first, then the bias-ON rows of the c 128 register-form renderings that run
# the bias-path epilogue fold -- a bias-ON rendering is allocated its demand (no spill either way), and the honoured split still pays there):
# the projection stage at c 128 with the pass-MAJOR tile order (the heuristics' pick from N = 512 and at 256 B 4): whole-call A/B/A on one exclusive GPU of the 212-SM Rubin perf-node class (lock 2376), in-process arm + control twins, 100-400 x 3 graph replay, two passes with the same sign -- the register form +1.6 / +1.5 / +1.7 / +1.8 % at 512:128:1 / 256:128:4 / 768:128:1 / 768:128:4 (its 1 / 1 STACK-8 spill gone; K1 itself +3.8 %), the rolled form +1.1..+1.2 % at 1024:128:1 / 512:128:4 / 1024:128:4 (REG 80 flat -> the 232 / 24 split, no spill either way; K1 +2.5 %);
# NOT a row: the INTERLEAVED rolled projection renderings (N 256 B 1, 384) -0.1..-0.7 % (4 / 13 role-prologue spills under the hint), the output stage at c 256 bias off -0.1 / -0.1 / +0.0 % (its 5 / 7 epilogue spills vanish, the kernel time does not move: 298 vs 297 us -- the bias-off deficit against the bias-on body is elsewhere), the bias-ON renderings NOT named below (the identity controls +0.0 %; the rolled-form K1 bias-on cells +0.33 / +0.35 % under the hint = inside the bar).
# the hint rows (whole call, bias ON, the same node class): the K1-only row +2.57 / +2.39 / +2.52 / +2.52 % at 512:128:1 / 768:128:1 / 256:128:4 / 768:128:4 (the CUPTI split: K1 71.8 -> 68.1 us = -5.2 %, K3 unchanged), +0.23 / +0.26 % on the rolled-form K1 cells 1024:128:1 / 512:128:4 (inside the bar), -1.15 % on the interleaved 384:128:4 (role-prologue spills under the split, as at bias off); the K3-only row -0.10..+0.11 % on every cell (neutral: not a row); both stages hinted together +2.37..+2.61 % = the K1 row's gain
# THE WORK ITEM IS PART OF THE KEY (the LayerNorm-once item `ln_once`, measured 2026-10-13 on the landed picks, the same node class and protocol):
# under `ln_once="block"` the item's +4 live words in the 24-register producer roles turn the honoured split into spills the per-tile item never
# had (host SASS, sm_107a: the c 128 register form 0 / 0 -> 4 STL / 13 LDL at STACK 16, the rolled form 1 / 1 -> 9 / 15 at STACK 24, the c 256
# pf forms 2 / 2 -> 9 / 16 (K1) and 5 / 6 -> 25 / 26 at STACK 144 (K3)), and the whole call does not pay for them: the rule's hint vs
# `min_blocks_per_mp=0` over the landed picks (one process per knob set, in-process control twins, graph replay, the fp64 oracle on every cell;
# the three cells unhinted under both as the identity control: +0.01 / -0.31 / +0.11 %) -- the UNHINTED call is faster by 1.10 / 1.28 / 0.69 /
# 1.15 / 0.46 % at 512:128:1 / 768:128:1 / 1024:128:4 / 256:128:4 / 384:128:1 bias off (the register AND the rolled form) and within the noise at
# 512:128:1 / 768:128:1 / 256:128:4 bias on (-0.15 / +0.17 / -0.15 %); standalone the rolled K1 loses 2.5 / 3.2 % hinted at 1024:4 / 256:1 and the
# c 256 forms lose 6.1-7.0 % (K1) and 12.0 % (K3) (the standalone register-form K1 reads +0.7..+1.1 % hinted -- a `want_stats=True` rendering the
# API never builds; the whole call is the landing evidence, as for every row above).  So every row names the PER-TILE item (`ln_once=""`): a bare
# declaration whose pick carries the item (every c 128 bucket from N = 256, every c 256 bucket) runs UNHINTED -- the renderings the pins measured
# at 0 / 0 (register) and 1 / 1 (rolled); a caller's `TriMulKnobs(ln_once="")` keeps the measured hint; "passes" has no row (unmeasured -> as shipped).
_REGISTER_SPLIT_HINTS: Dict[Tuple[str, int, bool, str, str, Optional[str], str], int] = {
    ("k1", 128, False, "registers", "packed", "major", ""): 1,
    ("k1", 128, False, "smem_rolled", "packed", "major", ""): 1,
    # under the block item this rendering runs unhinted (above); there the fold still beats `packed` +0.73 / +0.93 % on 768:128:1 / 768:128:4 and ties on
    # 512:128:1 outgoing / incoming / 256:128:4 (+0.14 / +0.43 / -0.17 %; both arms unhinted, 212-SM Rubin class) -- the pick stands; `packed` + a
    # block-keyed hint 1 read +0.68 / +0.79 / +0.51 % on 512:128:1 outgoing / incoming / 768:128:1 over it (two passes, not yet reproduced as the base)
    ("k1", 128, True, "registers", "packed_biasfold", "major", ""): 1,
}


def register_split_hint(
    stage: str, channels: int, has_bias: bool, ln_pass_form: str, epi_math: str, pass_order: Optional[str] = None, ln_once: str = ""
) -> int:
    """The ``.minnctapersm`` hint a bare declaration hands the ``stage`` (``"k1"`` = the projection, ``"k3"`` = the output stage) rendering at this
    hidden width, bias arm, LayerNorm / epilogue form, (K1) tile order and LayerNorm work item (``ln_once``; the stage's OWN item -- the output stage's
    is ``""`` where the substrate declines a one-tile item): the MEASURED rows of :data:`_REGISTER_SPLIT_HINTS`, else 0 (the rendering as shipped;
    every row names the per-tile item -- under the LayerNorm-once item the honoured split spills and the whole call runs faster unhinted).  A
    caller's ``TriMulKnobs(min_blocks_per_mp=...)`` overrides it for both stages."""
    if stage not in ("k1", "k3"):
        raise ValueError(f"stage must be 'k1' or 'k3', got {stage!r}")
    order = (str(pass_order) if pass_order is not None else None) if stage == "k1" else None
    # the rows are keyed by RENDERING: the three packed epilogue forms differ only under `has_bias` and render ONE cubin without biases, so a
    # bias-off lookup reads the family's row (`EPI_MATHS_PACKED`); a bias-on rendering is hinted only by a row naming its EXACT form
    epm = "packed" if (not has_bias and str(epi_math) in EPI_MATHS_PACKED) else str(epi_math)
    return _REGISTER_SPLIT_HINTS.get((stage, _nearest_width(channels), bool(has_bias), str(ln_pass_form), epm, order, str(ln_once or "")), 0)


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
    "MIN_BLOCKS_PER_MP_DOMAIN",
    "TriMulKnobs",
    "batch_bucket",
    "default_k_block",
    "k2_tile_for",
    "recommend",
    "register_split_hint",
    "resolve",
    "validate_knobs",
]
