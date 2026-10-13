# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The LayerNorm-prologue gated dual GEMM substrate -- host side (params, geometry, build, launch).

What it computes, per pair row ``r`` of a ``[rows, K]`` bf16 activation ``z`` (``K`` = the LayerNorm width = the GEMM's K):

    x      = LayerNorm_K(z[r, :]; gamma, beta, eps)                      fp32 statistics, rounded ONCE to bf16 (the MMA operand)
    acc    = x @ W^T                                                       W = [2 * c_out, K] bf16, fp32 accumulation
    out[r, n] = sigmoid(acc[r, gate(n)] + b_g[n]) * (acc[r, value(n)] + b_p[n]) * mask[r]      n < c_out, fp32, rounded once to bf16

with the VALUE column of output channel ``n`` and its GATE column in the SAME 256-wide GEMM N tile: the weight rows are
INTERLEAVED per 128-channel block as ``[W_p[128j : 128j + 128]; W_g[128j : 128j + 128]]`` (:func:`interleave_value_gate`), so
every N tile is ``[value block | gate block]`` and the epilogue pairs local columns ``n`` and ``128 + n`` thread-locally.
``mask[r]`` is a per-row scalar applied in fp32 BEFORE the bf16 rounding; a zero mask row stores EXACTLY ``+0.0`` (a select, never
``x * 0`` -- ``sdpa-invariants.md`` S 2) and a 256-row tile whose rows are all masked skips its MMA by a warp vote (bitwise the same
output).  Several ``passes`` (the triangle multiplicative update's ``a`` and ``b`` projections) share one activation and take
their own weight / bias / output plane; the per-row ``(mean, rstd)`` of the LayerNorm go to GMEM once (pass 0) for a backward.

The kernel BODY is the rendered sm_107a FROST GEMM template (``gemm/frost/sm100/kernel_templates/sm100_matmul.py``,
``CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma``: a cta_group::2 pair computing 256 pair rows x 256 columns, K
streamed in 64-wide stages, persistent CLC scheduler) plus two arms -- a LayerNorm PROLOGUE on the A stages in SMEM (4 extra
warps, one pair row per lane) and the gated-pairing EPILOGUE with a channel-major (transposed) TMA store.  The body is
``sm107_dual_gemm.py`` in THIS package (:func:`body_path`; one rendered body per arch, named ``sm<NNN>_dual_gemm.py`` like the
arch trees' templates -- the sm_100a body of a later step goes beside it, and the host picks by arch); its module docstring carries
the barrier table with the per-warp arrival ledger, the SMEM table with swizzle + why, and the row-block tiling below.  A rendering
for one arch is not a registry template, which is why it does not sit in a ``kernel_templates/`` directory (the template inventory
sweeps and the arch-family resolver define those files).  Consumers: the triangle multiplicative update's K1
(``cudnn.triangle_multiplicative_update.kernels.trimul_proj``, ``pairing = "columns"``) and its K3
(``kernels.trimul_tail``, ``pairing = "accumulators"``: two GEMMs of N = c_out each, the value from a LayerNormed MN-major operand,
the gate from a LayerNormed K-major one, the pairing ACROSS the two accumulators, a pair-major output -- :func:`run_ln_dual_gemm_tail`);
the triangle-attention block's projection later -- they import THIS module, never copy the kernel.

Row-block tiling (``row_block_geometry``).  The caller's rows are ``(bi, k)`` pairs -- ``bi`` = a batch-and-outer index in
``[0, bn)``, ``k`` = the inner index in ``[0, n_res)`` -- stored pair-major in ``z`` (row ``bi * n_res + k``) and CHANNEL-major in
the output: ``out[pass][c][bi][k]`` with the ``k`` axis padded to ``n_pad`` (a multiple of 8 so the TMA store's row stride is
16-byte aligned; the kernel writes the ``k`` in ``[n_res, n_pad)`` as zeros).  A 256-row pair tile covers ``r_i`` consecutive
``bi`` x ``kb`` consecutive ``k`` with ``r_i * kb == 256`` and ``kb`` a power of two (``kb = 256`` for ``n_res >= 256``, else the
next power of two >= ``max(n_res, 8)``), so the A load is one 3-D TMA box ``(64 channels, min(kb, 128) k, 128 // min(kb, 128) bi)``
per CTA, the store the matching ``(k, bi, 32 channels)`` box, and a row's ``(bi, k)`` are a shift and a mask of its tile row.
The last ``k`` block of a ``bi`` and the last ``bi`` block are partial: TMA zero-fills the loads, clips the stores, and the
kernel treats such rows as masked (``mask = 0``, no statistics).  ``kb`` is a RUNTIME operand of the compiled kernel (``lkb`` /
``n_kb`` / ``kbc`` / ``rc`` in the launch's ``geom`` tuple, dynamic TMA box dims), so a caller may override the rule through
``k_block`` on ``row_block_geometry`` / the two launchers -- a performance knob: every row's arithmetic is independent of the
tile it sits in, so the output is bitwise the same under every admitted ``kb``.  Why one would: at ``n_res = 384`` the rule's
``kb = 256`` pads 25 % of every pair tile's rows (the second k block holds 128 valid k of 256); ``kb = 128`` pads none.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Optional

#: The LayerNorm width (= K) is streamed in 64-element stages; the pairing needs 128-channel value / gate blocks.
K_STAGE = 64
CHANNEL_BLOCK = 128
PAIR_TILE_ROWS = 256
#: Output ``k`` padding: the channel-major TMA store needs a 16-byte-aligned row stride (bf16 -> 8 elements).
K_PAD_MULTIPLE = 8
MIN_K_BLOCK = 8

DIAG_ARMS = ("", "one_pass_stats", "no_ln_norm", "no_ln", "no_epi_math")
#: Tile order (performance only; the same function, bitwise): ``"major"`` = the pass is the grid's z axis (every row block of
#: pass 0 before pass 1); ``"interleaved"`` = the pass folded into the scheduler's M axis so the two tiles reading one row block's
#: z rows are adjacent (the second pass's z load is an L2 hit -- the HBM-side saving of a shared z load without resident weights).
PASS_ORDERS = ("major", "interleaved")
#: ``"columns"`` = ONE GEMM of N = 2 c_out whose 256-wide tiles pair value columns [0, 128) with gate columns [128, 256) (K1);
#: ``"accumulators"`` = TWO GEMMs of N = c_out (value = operand 0's, gate = operand 1's accumulator), the pairing across them (K3).
PAIRINGS = ("columns", "accumulators")
A0_MAJORS = ("k", "mn")
#: The MN-major operand's LayerNorm form (``pairing = "accumulators"`` only): ``"mn_major"`` normalises in place in the MN-major stage and the
#: MMA reads it MN-major; ``"transpose"`` writes the normalised row K-major into the same stage (after a 4-warp barrier) and the MMA reads K-major.
A0_PROLOGUES = ("mn_major", "transpose")
#: The MN-major TMA box is 64 rows of ONE ``bi``: a pair tile's ``kb`` must be >= 64 (``n_res > 32``) in the accumulators arm.
MIN_N_RES_MN_MAJOR = 33
#: The same constraint on the LAUNCHED ``kb`` (a caller's ``k_block`` below 64 cannot serve the accumulators arm; the columns arm admits 8).
MIN_K_BLOCK_MN_MAJOR = 64
#: The LayerNorm warps' pass form (performance only; bitwise the same operand): ``"registers"`` = the row's ``K / 2`` packed words stay in
#: registers between the statistics passes and the normalise pass (the residency that spills from K = 256 up at the flat entry allocation);
#: ``"smem"`` = every pass re-reads the row's 16-byte chunks from the (unchanged) SMEM stage at the same swizzled offsets -- no residency;
#: ``"smem_rolled"`` = the same with the per-stage chunk loops as runtime loops (the code-size lever against the instruction-cache stall);
#: ``"smem_rolled_pf"`` = the rolled form with a 2-deep software pipeline over every chunk loop (the next chunk's loads issued before the
#: current chunk's arithmetic; the same words into the same additions in the same order: bitwise the register form);
#: ``"smem_rolled_tree"`` = the pipelined form with FOUR partial sums per statistics pass (word w of every chunk accumulates into partial w,
#: combined once per pass as (t0 + t1) + (t2 + t3)) -- a 2-deep dependent chain per chunk instead of 8, at a DIFFERENT fp32 summation
#: order: the one member of the vocabulary that is NOT bitwise the register form (:data:`LN_PASS_FORMS_REORDERED`; the fp64 oracle's
#: bound decides, never a silent default).
LN_PASS_FORMS = ("registers", "smem", "smem_rolled", "smem_rolled_pf", "smem_rolled_tree")
#: The pass forms whose fp32 summation ORDER differs from the register form's: the same function to the oracle's bound, not bitwise --
#: a bitwise pin excludes them, a heuristics default never names them without a decision on the measured oracle distance.
LN_PASS_FORMS_REORDERED = ("smem_rolled_tree",)
#: The gated epilogue's arithmetic form (performance only; the same fp32 operations per element): ``"scalar"`` = one element at a time;
#: ``"packed"`` = channel PAIRS on the packed fp32x2 ALU (``add.f32x2`` / ``mul.f32x2`` / ``fma.rn.f32x2``: half the ALU issue per element).
EPI_MATHS = ("scalar", "packed")
#: The ``.minnctapersm`` launch hint (performance only; bitwise the same output): 0 = no directive (the rendering as shipped: ptxas drops the
#: body's ``setmaxnreg`` split and allocates one flat count per warp), 1 = the hint under which ptxas honours the split -- the kernel is one CTA
#: per SM by its SMEM footprint either way (:class:`LnDualGemmParams`).  2 or more would cap the allocation at 80 registers: refused.
MIN_BLOCKS_PER_MP_DOMAIN = (0, 1)
#: The rendered kernel bodies, one per arch family (``sm<NNN>_dual_gemm.py`` beside this file); ``build_ln_dual_gemm`` picks by name.
BODIES = {"sm107": "sm107_dual_gemm.py"}


def body_path(arch: str = "sm107") -> str:
    """The rendered body for ``arch`` (a typed decline for an arch without one -- the sm_100a body is a later step)."""
    if arch not in BODIES:
        raise ValueError(f"the LayerNorm-prologue dual GEMM has a rendered body for {sorted(BODIES)} only, not {arch!r}")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), BODIES[arch])


@dataclass(frozen=True)
class LnDualGemmParams:
    """Compile-time shape of one LayerNorm-prologue gated dual GEMM artifact (the template's ``FROST_TEMPLATE_PARAMS``).

    ``k``: the LayerNorm width = GEMM K (a multiple of 64; the kernel keeps every K stage of a tile resident, so at most
    ``ab_stages * 64`` -- 576 on the sm_107a rendering).  ``c_out``: output channels per pass (a multiple of 128; the GEMM's
    N is ``2 * c_out``).  ``passes``: weight / bias / output planes sharing one activation.  ``has_bias``: the projection
    biases are present (else folded out at the ABI).  ``want_stats``: write ``(mean, rstd)`` per row.  ``has_mask``: a per-row
    mask tensor is bound (else every valid row has mask 1).  ``skip_masked_tiles``: the warp-vote skip of fully masked tiles
    (OFF renders the control arm the bitwise pin compares against).  ``eps``: the LayerNorm epsilon.  ``clc_stages``: the CLC
    scheduler ring depth (performance only: every value computes the same function).  ``pass_order``: the tile order
    (:data:`PASS_ORDERS`; performance only, bitwise the same output).
    """

    k: int = 128
    c_out: int = 128
    passes: int = 2
    has_bias: bool = True
    want_stats: bool = True
    has_mask: bool = True
    skip_masked_tiles: bool = True
    eps: float = 1e-5
    #: CLC scheduler ring depth.  The template's 1-deep ring lets the TMA / LayerNorm warps run only one tile ahead of the
    #: epilogue (the next tile id is fetched once EVERY consumer has read the previous one); with 2 K stages per tile the
    #: per-tile fixed costs (prologue, epilogue) ARE the pipeline, so the ring depth is the first lever (measured per node).
    clc_stages: int = 2
    #: DIAGNOSTIC arms for relief measurements (never a production setting; the oracle tests pass only with ""):
    #: "one_pass_stats" (sum / sum-of-squares in one pass), "no_ln_norm" (statistics, no normalise write-back), "no_ln" (the LN
    #: warps only forward the stages), "no_epi_math" (the epilogue stores zeros without reading TMEM).
    diag: str = ""
    #: Tile order (:data:`PASS_ORDERS`): "major" = the rendering's z-axis pass; "interleaved" = the pass folded into the
    #: scheduler's M axis (passes must be a power of two) -- the shared-z-load form; the output is bitwise the same.  The default stays
    #: the rendering's "major" on purpose: the sign is PER WIDTH (measured on the 212-SM Rubin perf class, 400 x 3 iterations, CLEAN
    #: windows, at the 2-deep ring: K = 128 interleaved +3.85 .. +4.13 %, K = 256 interleaved -0.76 .. -0.80 %; with a 1-deep ring the
    #: whole-call sign at K = 128 reverses from N >= 512), so a consumer picks it per geometry (the
    #: triangle multiplicative update's heuristics module does) rather than inheriting one value for every K.
    pass_order: str = "major"
    #: :data:`PAIRINGS`.  "accumulators" (K3): operand 0 = the VALUE operand (``a0_major``), operand 1 = the GATE operand (K-major), both
    #: LayerNormed with their own (gamma, beta) over the SAME K; N = ``c_out`` per GEMM; the output is pair-major ``[rows, c_out]``;
    #: ``has_mask`` / ``skip_masked_tiles`` must be False and ``passes`` 1.
    pairing: str = "columns"
    #: operand 0's GMEM major (:data:`A0_MAJORS`): "mn" = its rows are MN-major (the K3 ``y`` rows: the inner index contiguous), "k" = K-major.
    a0_major: str = "k"
    #: the MN-major LayerNorm form (:data:`A0_PROLOGUES`; accumulators arm only).  Performance only: both forms compute the same function.
    a0_prologue: str = "mn_major"
    #: NUMERICS MODE (an op attribute, NOT a performance knob: it changes the function, so no heuristic or autotuner may set it; OFF by
    #: default).  Accumulators arm: operand 0's LayerNorm is the PURE ``(x - mean) * rstd`` -- the caller folded gamma into the value weights
    #: and beta into the value bias (:func:`fold_output_affine`): a second bf16 rounding of the weights and the bias, measured ~2.3x the
    #: unfolded max-abs error.
    affine_fold: bool = False
    #: NUMERICS MODE (an op attribute, not a knob; OFF by default).  Accumulators arm: ``out = bf16(residual[row, :] + gate * value)`` in ONE
    #: rounding (the fused block form); the unfused API rounds twice.
    fuse_residual: bool = False
    #: the TMA warp prefetches the tile's A boxes into L2 before waiting for a ring slot (performance only).
    prefetch_a_l2: bool = False
    #: :data:`LN_PASS_FORMS` -- how the LayerNorm warps hold the row between their passes (performance only; bitwise the same operand for every
    #: form outside :data:`LN_PASS_FORMS_REORDERED`, whose partial sums change the fp32 summation order -- the oracle's bound, not a bitwise pin).  The
    #: default stays the rendering's register residency; a consumer picks the re-read form per width (its value is in the spill counts).
    #: Under ``pairing = "accumulators"`` the re-read forms require ``a0_prologue = "mn_major"`` (the transpose form's K-major writes overlap
    #: the channel rows every lane still re-reads -- the register form keeps all reads before its LN_SYNC barrier, the re-read form cannot).
    ln_pass_form: str = "registers"
    #: :data:`EPI_MATHS` -- the gated epilogue's arithmetic form (performance only).
    epi_math: str = "scalar"
    #: The ``.minnctapersm`` hint handed to the kernel launch (the DSL's ``launch(..., min_blocks_per_mp=)``; performance only, bitwise the same
    #: output).  0 = no directive = the rendering as shipped: ptxas DROPS the body's six ``setmaxnreg`` requests (neither ``.maxnreg`` nor
    #: ``.minnctapersm`` -> C7508) and every warp runs at one flat allocation, which its occupancy heuristic may set at 80 = 65536 / (2 x 384)
    #: -- two CTAs per SM -- with a few spills when the body's demand sits near it (the bias-OFF accumulators-arm c_out 256 rolled rendering:
    #: REG 80, STACK 24, 5 / 7 spills reloaded per tile on the epilogue's hot path, +7-8 % on the kernel; its bias-ON twin is allocated 112
    #: with no spill).  1 = the hint that lifts C7508: ptxas honours the split (USETMAXREG 6 -- epilogue + LN warps 232, producers 24; the
    #: cubin is byte-identical to a ``.maxnreg 168`` build).  The ~309 KiB SMEM footprint makes the kernel one CTA per SM either way, so 1
    #: states a fact; 2 or more would cap the allocation at 80 with the split impossible (refused).  A consumer sets it per RENDERING where
    #: measured (the triangle multiplicative update's ``heuristics.register_split_hint``), never library-wide: the default keeps every
    #: production cubin byte-identical.
    min_blocks_per_mp: int = 0

    @property
    def dual(self) -> bool:
        return self.pairing == "accumulators"

    @property
    def n_out(self) -> int:
        """The GEMM N of the artifact: 2 c_out (value | gate interleaved) in the columns arm, c_out per GEMM in the accumulators arm."""
        return self.c_out if self.dual else 2 * self.c_out

    @property
    def num_a_operands(self) -> int:
        return 2 if self.dual else 1

    @property
    def k_stages(self) -> int:
        return self.k // K_STAGE


def validate_ln_dual_gemm_params(p: LnDualGemmParams) -> None:
    """Typed declines for every value a caller can trip (never a module-level assert -- engine contract S 7)."""
    if not isinstance(p, LnDualGemmParams):
        raise TypeError(f"expected LnDualGemmParams, got {type(p).__name__}")
    if p.k <= 0 or p.k % K_STAGE != 0:
        raise ValueError(f"the LayerNorm width K must be a positive multiple of {K_STAGE} (one TMA / MMA stage), got {p.k}")
    if p.c_out <= 0 or p.c_out % CHANNEL_BLOCK != 0:
        raise ValueError(f"c_out must be a positive multiple of {CHANNEL_BLOCK} (one value block + one gate block per 256-wide N tile), got {p.c_out}")
    if p.passes < 1:
        raise ValueError(f"passes must be >= 1, got {p.passes}")
    if not (p.eps > 0.0):
        raise ValueError(f"eps must be > 0, got {p.eps}")
    if not (1 <= p.clc_stages <= 8):
        raise ValueError(f"clc_stages must be in [1, 8], got {p.clc_stages}")
    if p.diag not in DIAG_ARMS:
        raise ValueError(f"diag must be one of {DIAG_ARMS}, got {p.diag!r}")
    if p.pass_order not in PASS_ORDERS:
        raise ValueError(f"pass_order must be one of {PASS_ORDERS}, got {p.pass_order!r}")
    if p.pass_order == "interleaved" and (p.passes & (p.passes - 1)):
        raise ValueError(f"pass_order='interleaved' folds the pass into the tile index with a shift and a mask: passes must be a power of two, got {p.passes}")
    if p.pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {p.pairing!r}")
    if p.a0_major not in A0_MAJORS:
        raise ValueError(f"a0_major must be one of {A0_MAJORS}, got {p.a0_major!r}")
    if p.a0_prologue not in A0_PROLOGUES:
        raise ValueError(f"a0_prologue must be one of {A0_PROLOGUES}, got {p.a0_prologue!r}")
    if p.ln_pass_form not in LN_PASS_FORMS:
        raise ValueError(f"ln_pass_form must be one of {LN_PASS_FORMS}, got {p.ln_pass_form!r}")
    if p.epi_math not in EPI_MATHS:
        raise ValueError(f"epi_math must be one of {EPI_MATHS}, got {p.epi_math!r}")
    if isinstance(p.min_blocks_per_mp, bool) or p.min_blocks_per_mp not in MIN_BLOCKS_PER_MP_DOMAIN:
        raise ValueError(
            f"min_blocks_per_mp must be 0 (no .minnctapersm directive: the rendering as shipped, the register split dropped by ptxas) or 1 (the hint "
            f"under which ptxas honours the body's setmaxnreg split); 2 or more would cap the allocation at 65536 / (2 x 384) = 80 registers with the "
            f"split impossible -- got {p.min_blocks_per_mp!r}"
        )
    if p.dual:
        if p.has_mask or p.skip_masked_tiles:
            raise ValueError("pairing='accumulators' has no output mask: has_mask and skip_masked_tiles must be False")
        if p.passes != 1:
            raise ValueError(f"pairing='accumulators' has one output: passes must be 1, got {p.passes}")
        if p.a0_major != "mn":
            raise ValueError("pairing='accumulators' is written for an MN-major operand 0 (the channel-major y rows); a0_major must be 'mn'")
        if p.k > 6 * K_STAGE:
            raise ValueError(f"pairing='accumulators' keeps every K stage of a tile resident in a 6-deep ring: K must be <= {6 * K_STAGE}, got {p.k}")
        if p.ln_pass_form != "registers" and p.a0_prologue != "mn_major":
            raise ValueError(
                f"ln_pass_form={p.ln_pass_form!r} re-reads operand 0's MN-major stage in every LayerNorm pass; a0_prologue='transpose' overwrites the "
                "channel rows other lanes still read (the register form reads everything before its LN_SYNC barrier) -- use a0_prologue='mn_major'"
            )
    else:
        if p.a0_major != "k":
            raise ValueError("pairing='columns' (the single K-major operand) takes a0_major='k'")
        if p.affine_fold or p.fuse_residual:
            raise ValueError("affine_fold / fuse_residual belong to the accumulators arm (pairing='accumulators')")


@dataclass(frozen=True)
class RowBlockGeometry:
    """The pair-tile decomposition of ``(bn, n_res)`` rows -- every number the kernel's host call needs."""

    n_res: int
    bn: int
    n_pad: int  #: output k extent (a multiple of 8)
    kb: int  #: k's per 256-row pair tile (power of two)
    lkb: int  #: log2(kb)
    r_i: int  #: bi's per pair tile (= 256 // kb)
    n_kb: int  #: k blocks per bi row
    n_ib: int  #: bi blocks (ceil(bn / r_i))
    kbc: int  #: k's per CTA box (min(kb, 128))
    rc: int  #: bi's per CTA box (128 // kbc)

    @property
    def pair_tiles(self) -> int:
        return self.n_ib * self.n_kb

    @property
    def m_sched(self) -> int:
        """The flat M the template's scheduler walks: one 256-row pair tile per (bi block, k block)."""
        return self.pair_tiles * PAIR_TILE_ROWS

    @property
    def rows_computed(self) -> int:
        return self.m_sched

    @property
    def rows_valid(self) -> int:
        return self.bn * self.n_res


def default_k_block(n_res: int) -> int:
    """The rule: ``kb = 256`` for ``n_res >= 256`` (one bi per pair tile), else the next power of two >= max(n_res, 8)."""
    n_res = int(n_res)
    if n_res >= PAIR_TILE_ROWS:
        return PAIR_TILE_ROWS
    kb = MIN_K_BLOCK
    while kb < n_res:
        kb *= 2
    return kb


def validate_k_block(k_block) -> int:
    """A caller's ``k_block``: a power of two in ``[MIN_K_BLOCK, PAIR_TILE_ROWS]`` (ValueError names the domain -- a malformed declaration)."""
    if isinstance(k_block, bool) or not isinstance(k_block, int):
        raise ValueError(f"k_block must be an int power of two in [{MIN_K_BLOCK}, {PAIR_TILE_ROWS}], got {k_block!r}")
    if k_block < MIN_K_BLOCK or k_block > PAIR_TILE_ROWS or k_block & (k_block - 1):
        raise ValueError(f"k_block must be a power of two in [{MIN_K_BLOCK}, {PAIR_TILE_ROWS}] (k's per 256-row pair tile), got {k_block}")
    return k_block


def row_block_geometry(n_res: int, bn: int, k_block: Optional[int] = None) -> RowBlockGeometry:
    """The pair-tile decomposition: ``kb`` = :func:`default_k_block` (``None``) or the caller's :func:`validate_k_block`-checked ``k_block``
    (a performance knob -- the same bits under every admitted value; the accumulators arm additionally needs ``kb >= MIN_K_BLOCK_MN_MAJOR``)."""
    n_res, bn = int(n_res), int(bn)
    if n_res < 1 or bn < 1:
        raise ValueError(f"n_res and bn must be >= 1, got n_res={n_res}, bn={bn}")
    kb = default_k_block(n_res) if k_block is None else validate_k_block(k_block)
    lkb = kb.bit_length() - 1
    r_i = PAIR_TILE_ROWS // kb
    n_kb = -(-n_res // kb)
    n_ib = -(-bn // r_i)
    kbc = min(kb, PAIR_TILE_ROWS // 2)
    rc = (PAIR_TILE_ROWS // 2) // kbc
    n_pad = -(-n_res // K_PAD_MULTIPLE) * K_PAD_MULTIPLE
    return RowBlockGeometry(n_res=n_res, bn=bn, n_pad=n_pad, kb=kb, lkb=lkb, r_i=r_i, n_kb=n_kb, n_ib=n_ib, kbc=kbc, rc=rc)


def interleave_value_gate(w_value, w_gate):
    """``[c_out, K]`` value and gate weights -> the ``[2 * c_out, K]`` row order the kernel's pairing expects:
    per 128-channel block ``j``: ``W_p[128j : 128j + 128]`` then ``W_g[128j : 128j + 128]``.  Works on any tensor library
    with ``reshape`` / ``cat`` (torch); the inverse is :func:`deinterleave_value_gate`.  A COPY: do it once at plan time."""
    import torch

    if w_value.shape != w_gate.shape or w_value.ndim != 2:
        raise ValueError(f"value / gate weights must both be [c_out, K], got {tuple(w_value.shape)} and {tuple(w_gate.shape)}")
    c_out, k = w_value.shape
    if c_out % CHANNEL_BLOCK != 0:
        raise ValueError(f"c_out must be a multiple of {CHANNEL_BLOCK}, got {c_out}")
    v = w_value.reshape(c_out // CHANNEL_BLOCK, CHANNEL_BLOCK, k)
    g = w_gate.reshape(c_out // CHANNEL_BLOCK, CHANNEL_BLOCK, k)
    return torch.cat([v, g], dim=1).reshape(2 * c_out, k)


def interleave_value_gate_bias(b_value, b_gate):
    """The ``[c_out]`` value and gate biases in the interleaved ``[2 * c_out]`` order of :func:`interleave_value_gate`."""
    return interleave_value_gate(b_value.reshape(-1, 1), b_gate.reshape(-1, 1)).reshape(-1)


@dataclass(frozen=True)
class LnDualGemmPlan:
    """One compiled artifact.  Plan-time only; :func:`run_ln_dual_gemm` launches."""

    params: LnDualGemmParams
    module: Any
    launch: Any  # the cute-compiled ``_host``


@lru_cache(maxsize=None)
def build_ln_dual_gemm(params: LnDualGemmParams, arch: str = "sm107") -> LnDualGemmPlan:
    """Load the ``arch`` body specialized for ``params`` and cute-compile it.  Shapes (rows, bn, n_res) are runtime.

    The CuTe-DSL gates run HERE, not only in the API layer above (the substrate is a public module: the kernel test modules and the
    triangle-attention projection build on it): below the DSL floor, or on a DSL without the sm_107a target, the failure is a typed
    ``NotImplementedError`` naming the version -- never the DSL's own ``KeyError: 'sm_107a'`` out of ``cute.compile``.  The target
    check is unconditional while the only rendered body is the sm_107a one."""
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
    from cudnn.frost.template_loader import load_template

    msg = cutedsl_requirement_error("the LayerNorm-prologue dual GEMM") or cutedsl_arch_requirement_error((10, 7))
    if msg:
        raise NotImplementedError(msg)
    validate_ln_dual_gemm_params(params)
    mod = load_template(body_path(arch), params, tag="ln_prologue_dual_gemm")
    return LnDualGemmPlan(params=params, module=mod, launch=mod.compile())


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def run_ln_dual_gemm(
    plan: LnDualGemmPlan,
    z,  # [bn * n_res, K] bf16 contiguous (pair-major rows)
    w,  # [passes, 2 * c_out, K] bf16 contiguous, rows interleaved per pass (interleave_value_gate)
    out,  # [passes, c_out, bn, n_pad] bf16 contiguous (channel-major planes; k padded to n_pad)
    ln_weight,  # [K] bf16
    ln_bias,  # [K] bf16
    *,
    n_res: int,
    bn: int,
    bias=None,  # [passes, 2 * c_out] bf16 interleaved, required iff params.has_bias
    mask=None,  # [bn * n_res] bf16 per-row mask, required iff params.has_mask
    stats=None,  # [bn * n_res, 2] fp32 (mean, rstd), required iff params.want_stats
    stream,
    k_block: Optional[int] = None,  # k's per 256-row pair tile (row_block_geometry's rule when None; performance only, bitwise the same)
) -> RowBlockGeometry:
    """Launch.  No allocation, no conversion; every check is cheap host arithmetic.  Returns the geometry used."""
    import torch
    from cuda.bindings import driver as cuda

    p = plan.params
    _check(not p.dual, "this artifact was compiled with pairing='accumulators'; launch it through run_ln_dual_gemm_tail")
    g = row_block_geometry(n_res, bn, k_block)
    rows = g.rows_valid
    bf16 = torch.bfloat16
    _check(
        z.dtype == bf16 and z.is_contiguous() and tuple(z.shape) == (rows, p.k),
        f"z must be a contiguous bf16 [{rows}, {p.k}] tensor, got {tuple(z.shape)} {z.dtype}",
    )
    _check(
        w.dtype == bf16 and w.is_contiguous() and tuple(w.shape) == (p.passes, p.n_out, p.k),
        f"w must be a contiguous bf16 [{p.passes}, {p.n_out}, {p.k}] tensor (rows interleaved per 128-channel block), got {tuple(w.shape)} {w.dtype}",
    )
    _check(
        out.dtype == bf16 and out.is_contiguous() and tuple(out.shape) == (p.passes, p.c_out, bn, g.n_pad),
        f"out must be a contiguous bf16 [{p.passes}, {p.c_out}, {bn}, {g.n_pad}] tensor (channel-major, k padded to {g.n_pad}), got {tuple(out.shape)} {out.dtype}",
    )
    for name, t_ in (("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        _check(
            t_.dtype == bf16 and t_.is_contiguous() and tuple(t_.shape) == (p.k,),
            f"{name} must be a contiguous bf16 [{p.k}] tensor, got {tuple(t_.shape)} {t_.dtype}",
        )
    if p.has_bias:
        _check(bias is not None, "this artifact was compiled with has_bias=True; bias must be bound (no silent fallback)")
        _check(
            bias.dtype == bf16 and bias.is_contiguous() and tuple(bias.shape) == (p.passes, p.n_out),
            f"bias must be a contiguous bf16 [{p.passes}, {p.n_out}] tensor (interleaved like the weight rows), got {tuple(bias.shape)} {bias.dtype}",
        )
    else:
        _check(bias is None, "this artifact was compiled with has_bias=False; bias must be None")
    if p.has_mask:
        _check(mask is not None, "this artifact was compiled with has_mask=True; mask must be bound")
        _check(
            mask.dtype == bf16 and mask.is_contiguous() and tuple(mask.shape) == (rows,),
            f"mask must be a contiguous bf16 [{rows}] tensor, got {tuple(mask.shape)} {mask.dtype}",
        )
    else:
        _check(mask is None, "this artifact was compiled with has_mask=False; mask must be None")
    if p.want_stats:
        _check(stats is not None, "this artifact was compiled with want_stats=True; stats must be bound")
        _check(
            stats.dtype == torch.float32 and stats.is_contiguous() and tuple(stats.shape) == (rows, 2),
            f"stats must be a contiguous fp32 [{rows}, 2] tensor, got {tuple(stats.shape)} {stats.dtype}",
        )
    else:
        _check(stats is None, "this artifact was compiled with want_stats=False; stats must be None")
    # Alignment the kernel assumes (a contiguous but OFFSET view passes every shape check and would fault or read the wrong half-word):
    # TMA operands and the 16-byte parameter loads at 16 B, the mask's aligned 32-bit word loads at 4 B, the (mean, rstd) v2 stores at 8 B.
    for name, t_ in (("z", z), ("w", w), ("out", out)):
        _check(t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (a TMA operand)")
    for name, t_ in (("ln_weight", ln_weight), ("ln_bias", ln_bias), ("bias", bias)):
        _check(t_ is None or t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (read with 16-byte vector loads)")
    _check(mask is None or mask.data_ptr() % 4 == 0, "mask must be 4-byte aligned (read as aligned 32-bit words)")
    _check(stats is None or stats.data_ptr() % 8 == 0, "stats must be 8-byte aligned (written as 8-byte (mean, rstd) pairs)")
    geom = (g.m_sched, p.n_out, p.k, p.passes, g.n_res, g.bn, g.n_pad, g.lkb, g.n_kb, g.kbc, g.rc)
    plan.launch(
        geom,
        z.view(-1),
        None,
        w.view(-1),
        None,
        out.view(-1),
        ln_weight,
        ln_bias,
        None,
        None,
        bias.view(-1) if bias is not None else None,
        mask if mask is not None else None,
        stats.view(-1) if stats is not None else None,
        None,
        cuda.CUstream(int(stream)),
    )
    return g


def fold_output_affine(w_value, b_value, ln_weight, ln_bias):
    """The affine fold of ``LayerNorm(y) @ W^T + b`` (the ``affine_fold`` knob): ``W' = bf16(W * gamma[None, :])``,
    ``b' = bf16(b + W @ beta)`` so that ``(y - mean) * rstd @ W'^T + b'`` equals the unfolded function up to the SECOND bf16 rounding
    of the weights and of the bias (the knob's numerics class).  fp64 arithmetic, bf16 results; ``b_value`` may be None (then ``W @ beta``
    alone).  A COPY at plan time."""
    import torch

    w64 = w_value.double()
    g64 = ln_weight.double()
    be64 = ln_bias.double()
    if w64.ndim != 2 or w64.shape[1] != g64.numel() or be64.numel() != g64.numel():
        raise ValueError(f"fold_output_affine: W must be [c_out, K] with K == len(gamma) == len(beta), got {tuple(w64.shape)}, {g64.numel()}, {be64.numel()}")
    w_folded = (w64 * g64[None, :]).to(torch.bfloat16).contiguous()
    b_fold = w64 @ be64
    if b_value is not None:
        b_fold = b_fold + b_value.double()
    return w_folded, b_fold.to(torch.bfloat16).contiguous()


def run_ln_dual_gemm_tail(
    plan: LnDualGemmPlan,
    a_value,  # [c_out_in = K, bn, n_pad] bf16 contiguous: operand 0, CHANNEL-major (its pair rows are MN-major)
    a_gate,  # [bn * n_res, K] bf16 contiguous: operand 1, pair-major rows (K-major)
    w_value,  # [c_out, K] bf16 contiguous
    w_gate,  # [c_out, K] bf16 contiguous
    out,  # [bn * n_res, c_out] bf16 contiguous (pair-major)
    ln_w_value,  # [K] bf16 (gamma of operand 0; ignored by the kernel under affine_fold, still bound)
    ln_b_value,  # [K] bf16
    ln_w_gate,  # [K] bf16 (gamma of operand 1)
    ln_b_gate,  # [K] bf16
    *,
    n_res: int,
    bn: int,
    bias=None,  # [2 * c_out] bf16 interleaved per 128-block [b_value; b_gate] (interleave_value_gate_bias), required iff params.has_bias
    stats=None,  # [bn * n_res, 2] fp32 (mean, rstd) of operand 0's LayerNorm, required iff params.want_stats
    residual=None,  # [bn * n_res, c_out] bf16 pair-major, required iff params.fuse_residual
    stream,
    k_block: Optional[int] = None,  # k's per 256-row pair tile (row_block_geometry's rule when None; >= MIN_K_BLOCK_MN_MAJOR here; bitwise the same)
) -> RowBlockGeometry:
    """Launch the accumulators arm (K3).  No allocation, no conversion; every check is cheap host arithmetic.  Returns the geometry."""
    import torch
    from cuda.bindings import driver as cuda

    p = plan.params
    _check(p.dual, "this artifact was compiled with pairing='columns'; launch it through run_ln_dual_gemm")
    g = row_block_geometry(n_res, bn, k_block)
    _check(
        g.n_res >= MIN_N_RES_MN_MAJOR,
        f"the accumulators arm loads the MN-major operand in 64-row boxes of one bi: n_res must be >= {MIN_N_RES_MN_MAJOR} (kb >= 64), got {n_res}",
    )
    _check(
        g.kb >= MIN_K_BLOCK_MN_MAJOR,
        f"the accumulators arm loads the MN-major operand in 64-row boxes of one bi: k_block must be >= {MIN_K_BLOCK_MN_MAJOR}, got {g.kb}",
    )
    rows = g.rows_valid
    bf16 = torch.bfloat16
    _check(
        a_value.dtype == bf16 and a_value.is_contiguous() and tuple(a_value.shape) == (p.k, bn, g.n_pad),
        f"a_value must be a contiguous bf16 [{p.k}, {bn}, {g.n_pad}] tensor (channel-major, the inner index padded to {g.n_pad}), got {tuple(a_value.shape)} {a_value.dtype}",
    )
    _check(
        a_gate.dtype == bf16 and a_gate.is_contiguous() and tuple(a_gate.shape) == (rows, p.k),
        f"a_gate must be a contiguous bf16 [{rows}, {p.k}] tensor (pair-major rows), got {tuple(a_gate.shape)} {a_gate.dtype}",
    )
    for name, t_ in (("w_value", w_value), ("w_gate", w_gate)):
        _check(
            t_.dtype == bf16 and t_.is_contiguous() and tuple(t_.shape) == (p.c_out, p.k),
            f"{name} must be a contiguous bf16 [{p.c_out}, {p.k}] tensor, got {tuple(t_.shape)} {t_.dtype}",
        )
    _check(
        out.dtype == bf16 and out.is_contiguous() and tuple(out.shape) == (rows, p.c_out),
        f"out must be a contiguous bf16 [{rows}, {p.c_out}] tensor (pair-major), got {tuple(out.shape)} {out.dtype}",
    )
    for name, t_ in (("ln_w_value", ln_w_value), ("ln_b_value", ln_b_value), ("ln_w_gate", ln_w_gate), ("ln_b_gate", ln_b_gate)):
        _check(
            t_.dtype == bf16 and t_.is_contiguous() and tuple(t_.shape) == (p.k,),
            f"{name} must be a contiguous bf16 [{p.k}] tensor, got {tuple(t_.shape)} {t_.dtype}",
        )
    if p.has_bias:
        _check(bias is not None, "this artifact was compiled with has_bias=True; bias must be bound (no silent fallback)")
        _check(
            bias.dtype == bf16 and bias.is_contiguous() and tuple(bias.shape) == (2 * p.c_out,),
            f"bias must be a contiguous bf16 [{2 * p.c_out}] tensor (interleaved [value block; gate block] per 128 channels), got {tuple(bias.shape)} {bias.dtype}",
        )
    else:
        _check(bias is None, "this artifact was compiled with has_bias=False; bias must be None")
    if p.want_stats:
        _check(stats is not None, "this artifact was compiled with want_stats=True; stats must be bound")
        _check(
            stats.dtype == torch.float32 and stats.is_contiguous() and tuple(stats.shape) == (rows, 2),
            f"stats must be a contiguous fp32 [{rows}, 2] tensor, got {tuple(stats.shape)} {stats.dtype}",
        )
    else:
        _check(stats is None, "this artifact was compiled with want_stats=False; stats must be None")
    if p.fuse_residual:
        _check(residual is not None, "this artifact was compiled with fuse_residual=True; residual must be bound")
        _check(
            residual.dtype == bf16 and residual.is_contiguous() and tuple(residual.shape) == (rows, p.c_out),
            f"residual must be a contiguous bf16 [{rows}, {p.c_out}] tensor (pair-major), got {tuple(residual.shape)} {residual.dtype}",
        )
    else:
        _check(residual is None, "this artifact was compiled with fuse_residual=False; residual must be None")
    for name, t_ in (("a_value", a_value), ("a_gate", a_gate), ("w_value", w_value), ("w_gate", w_gate), ("out", out), ("residual", residual)):
        _check(t_ is None or t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (a TMA operand or a 16-byte vector load)")
    for name, t_ in (("ln_w_value", ln_w_value), ("ln_b_value", ln_b_value), ("ln_w_gate", ln_w_gate), ("ln_b_gate", ln_b_gate), ("bias", bias)):
        _check(t_ is None or t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (read with 16-byte vector loads)")
    _check(stats is None or stats.data_ptr() % 8 == 0, "stats must be 8-byte aligned (written as 8-byte (mean, rstd) pairs)")
    geom = (g.m_sched, p.n_out, p.k, p.passes, g.n_res, g.bn, g.n_pad, g.lkb, g.n_kb, g.kbc, g.rc)
    plan.launch(
        geom,
        a_value.view(-1),
        a_gate.view(-1),
        w_value.view(-1),
        w_gate.view(-1),
        out.view(-1),
        ln_w_value,
        ln_b_value,
        ln_w_gate,
        ln_b_gate,
        bias,
        None,
        stats.view(-1) if stats is not None else None,
        residual.view(-1) if residual is not None else None,
        cuda.CUstream(int(stream)),
    )
    return g
