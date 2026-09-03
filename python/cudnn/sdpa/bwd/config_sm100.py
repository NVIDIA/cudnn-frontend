# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the FROST SM100 SDPA-backward stage-2 flavor (d512).

Stage 2 of the three-stage backward: it consumes Q / K / V / dO / LSE / do_dot
and produces the ``S`` and ``dS`` workspaces that stage 3's three GEMMs reduce
into dV / dK / dQ.

Geometry is fixed here; the per-graph compile-time parameters (dtype, mask,
padding, scheduler policy) arrive as a :class:`TemplateParams` built by the
adapter in :mod:`cudnn.sdpa.bwd.api_dsl`. Nothing in this module reads an
environment variable.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier by the engine's ``Capabilities`` —
a ``ValueError`` from here means that row has a gap.

Layout rationale: the
Rubin SM107 backward's 320 KiB single-CTA layout does not fit SM100's 227 KiB.
This flavor instead forks the *forward* d512 SM100 kernel's cga4x1 role split —
sub-group 0 runs BMM1 (Q.K^T -> S_acc) and the softmax, sub-group 1 runs BMM2
(dO.V^T -> dS_acc) and the dS epilogue — which gives each sub-group its own
227 KiB of SMEM and its own 512 TMEM columns. That is what makes the operand
resident in TMEM on BOTH sides (Q on sg0, dO on sg1) and restores the
accumulator double-buffer the Rubin kernel had to drop.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stage-3 causal K-trim modes (see MatmulTemplateParams.causal_mode).
CAUSAL_K_NONE = 0
CAUSAL_K_LO = 1
CAUSAL_K_HI = 2
from typing import Optional, Tuple

from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E5M2,
    DTYPE_FP16,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
)

# SM100 usable dynamic SMEM per CTA (228 KiB physical, 227 KiB usable).  Mirrors
# ``sdpa.fwd.config_sm100._SM100_MAX_DYN_SMEM``; duplicated rather than imported
# so the backward pass does not depend on the forward's config module.
_SM100_MAX_DYN_SMEM = 227 * 1024

# TMEM columns on Blackwell.  576 is Rubin, and needs ``is_exclusive=True``,
# which is fenced out of the public cutlass-dsl wheel — do not raise this.
_SM100_TMEM_COLS = 512


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph compile-time parameters threaded into the stage-2 template.

    Shapes are deliberately absent: they ride the template's per-shape
    ``compile()`` cache. Batch and head are absent for the same reason *and*
    because they are host-loop coordinates — the kernel takes ``head_base`` /
    ``batch_base`` as runtime Int32 arguments.
    """

    dtype_qkv: int = DTYPE_BF16
    # Mask band. ``window_right`` set => causal; ``window_left`` set => SWA.
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    bottom_right: bool = False
    seq_kv_lens_present: bool = False
    seq_q_lens_present: bool = False
    sched_policy: int = SCHED_NATURAL
    # Tuning knob: halves of the fp32 S tile shipped sg0 -> sg1 per kv step.
    # 2 keeps the cross-sub-group ring 2-deep at an identical byte footprint;
    # 1 ships the whole tile and hard-stalls sg0 on sg1.
    xfer_halves: int = 2


@dataclass(frozen=True)
class MatmulTemplateParams:
    """Per-GEMM compile-time parameters for the stage-3 gradient GEMMs.

    Lives HERE rather than in the kernel file because a template loaded through
    ``frost.template_loader`` is executed before it is registered in
    ``sys.modules``, and a module-scope ``@dataclass`` needs its own module to be
    importable while the decorator runs. Same reason ``TemplateParams`` above is
    defined here and not in the stage-2 template.

    Only the operand-major pair varies: the three stage-3 GEMMs share one tile
    config, and the rendered bodies for the different majors are textually
    identical apart from ten constants (see the kernel's ``_MAJOR_CONSTS``).

        dV = S^T.dO, dK = dS^T.Q -> a_is_m_major=True   (kv is contiguous and is M)
        dQ = dS.K                -> a_is_m_major=False  (kv is contiguous and is K)

    ``b_is_n_major`` is True for all three: D is contiguous in BSHD and D is N.
    """

    a_is_m_major: bool = False
    b_is_n_major: bool = True
    # Causal K-trim.  Stage 2 under a causal mask leaves the tiles above the
    # diagonal UNWRITTEN, so this is a correctness requirement before it is an
    # optimization: a stage-3 GEMM must not read them.
    #
    #   CAUSAL_K_NONE  dense -- full K range.
    #   CAUSAL_K_LO    dV = S^T.dO and dK = dS^T.Q.  Output row is kv; S[q,kv]
    #                  is written only for q >= the stage-2 block containing kv,
    #                  so K (= q) STARTS there.
    #   CAUSAL_K_HI    dQ = dS.K.  Output row is q; dS[q,kv] is written only for
    #                  kv < the end of q's stage-2 block, so K (= kv) ENDS there.
    #
    # ``causal_gran`` is stage 2's write granularity in elements -- its
    # ``TILE_M * CTA_MMA`` (256), because its kv bound is taken over the whole
    # cluster q span.  Rounding to it is what makes "never read an unwritten
    # tile" exact rather than approximate.
    causal_mode: int = CAUSAL_K_NONE
    causal_gran: int = 0
    # How far the non-zero band extends PAST the plain kv <= q diagonal, in
    # elements.  Two things push it out and they add:
    #   * `diagonal_band_right_bound` (band widening) -- kv <= q + W;
    #   * bottom-right alignment      -- kv <= q + (S_kv - S_q).
    # Stage 2 writes those columns, so a trim that ignores them cuts away real
    # data. Negative is legal (bottom-right with S_kv < S_q pulls the band IN).
    causal_shift: int = 0
    # Epilogue store vector, in BYTES.  The output N is the head dim, and the
    # epilogue stores N in `vec_bytes_epi / sizeof(out)` element chunks -- so 32
    # requires d % 16 == 0 and 16 only d % 8 == 0.  Rendering the upstream
    # template at an N % 8 shape changes THIS CONSTANT AND NOTHING ELSE (the
    # bodies are textually identical), which is why it is a plain knob here.
    # Use `vec_bytes_epi_for(d, bpe)` rather than setting it by hand.
    vec_bytes_epi: int = 32


def vec_bytes_epi_for(d: int, bpe: int = 2) -> int:
    """Widest legal stage-3 epilogue store vector for a head dim of ``d``.

    The epilogue writes the output's N (= the head dim) in
    ``vec_bytes_epi / bpe`` element chunks, and the compiled artifact declares
    that as a symbolic divisibility -- so a mismatched d fails at CALL time with
    *"expected to be divisible by 16"*, not at build time.  32 B is the fast
    path and needs ``d % 16 == 0``; 16 B relaxes that to ``d % 8 == 0``.
    """
    per32 = 32 // bpe
    return 32 if d % per32 == 0 else 16


def _validate_params(params: TemplateParams) -> None:
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(
            f"SM100 SDPA bwd d512: dtype_qkv must be DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); "
            f"got {params.dtype_qkv}. FP8/MXFP8 backward is not implemented on this arch."
        )
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"SM100 SDPA bwd d512: window_left must be non-negative; got {params.window_left}")
    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"SM100 SDPA bwd d512: window_right must be non-negative; got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("SM100 SDPA bwd d512: bottom_right alignment requires a causal upper bound (window_right)")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError("SM100 SDPA bwd d512: seq_q_lens_present requires seq_kv_lens_present (padding mask)")
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"SM100 SDPA bwd d512: sched_policy must be one of NATURAL/LPT/LPT_L2; got {params.sched_policy}")
    if params.xfer_halves not in (1, 2):
        raise ValueError(f"SM100 SDPA bwd d512: xfer_halves must be 1 or 2; got {params.xfer_halves}")


def _mask_flags_from(params: TemplateParams) -> int:
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.seq_kv_lens_present:
        flags |= MASK_PADDED
    return flags


def bpe(dtype: int) -> int:
    return 1 if dtype <= DTYPE_E5M2 else 2


@dataclass(frozen=True)
class CfgBwdD512:
    """Stage-2 geometry: d_qk = d_v = 512, SM100, cga4x1 role split.

    Identifiers are keyed by op geometry, not by the model this was tuned for
    (frost-engine-contract.md §8).
    """

    TILE_M: int = 128  # q rows per CTA
    TILE_N: int = 128  # kv cols per kernel tile (collective MMA N)
    TILE_K: int = 512  # d_qk
    TILE_O: int = 512  # d_v

    DTYPE_QKV: int = DTYPE_BF16
    BPE: int = 2

    # cga4x1 = two sub-groups of two CTAs.  sg0: BMM1 + softmax; sg1: BMM2 + dS.
    CGA_M: int = 4
    CGA_N: int = 1
    CTA_MMA: int = 2

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    DO_SWZ_BYTES: int = 128
    S_SWZ_BYTES: int = 128

    # f16/bf16 is 1-chunk only on SM10x; 2-chunk (32) is silently wrong.
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    STAGES_KV: int = 2  # K ring on sg0 / V ring on sg1 (SMEM-cap driven)
    STAGES_ACC: int = 2  # S_acc / dS_acc parity slots (TMEM-cap driven)
    XFER_HALVES: int = 2  # fp32 S halves shipped sg0 -> sg1
    SCHEDULER_STAGES: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    SOFTMAX_WG_WARPS: int = 4
    CORRECTION_WARPS: int = 0

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 0
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    TOTAL_WARPS: int = 8
    THREADS_PER_CTA: int = 8 * 32
    SOFTMAX_WG0_BASE: int = 0
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TMASTG_WARP_ID: int = 6
    SCHED_WARP_ID: int = 7

    # --- mbarrier arrival counts ---------------------------------------
    # P3: an init count must equal the EXACT sum of producer arrivals per
    # phase, and the constant belongs next to its derivation.  Note
    # ``.arrive()`` fires on every LANE of the calling warp, not once per warp.
    ONE_LANE: int = 1  # a single elect_sync'd lane
    ONE_WARP: int = 32  # an un-elected THREAD arrive from one warp
    COMPUTE_LANES: int = 4 * 32  # SOFTMAX_WG_WARPS * 32, the compute warp group
    # *_acc_empty: every lane of the compute WG on BOTH CTAs of the pair
    # arrive_on_peer's the pair leader.
    ACC_EMPTY_ARRIVERS: int = 4 * 32 * 2  # COMPUTE_LANES * CTA_MMA

    # Scheduler ring.  Each calling warp delivers exactly ONE arrive to each
    # CTA of the cluster, so this is the count of (warp, CTA) instances that
    # call read_tile_id_arrive.  EVERY warp role except the scheduler warp
    # itself calls it, on every CTA of the cluster:
    #     (SOFTMAX_WG_WARPS + TMA-LDG + TMA-STG + MMA) * CGA_SIZE
    #   = (4 + 1 + 1 + 1) * 4 = 28
    # The MMA term counts BOTH arms: the non-leader is a quiet warp but it
    # still runs the persistent loop and must stay in step, so it arrives too.
    #
    # The forward d512's 25 is NOT comparable and must not be copied: its
    # TMA-STG runs on sg1 only, and only its sg1 non-leader MMA warp arrives.
    #
    # Getting this too LOW is the dangerous direction: the barrier completes
    # before every role has read the payload, so the ring advances a tile early
    # every iteration and the kernel wedges intermittently -- it does not fail
    # cleanly (P3).  An init of 26 against 28 arrivers cost a debugging session.
    READ_TILE_ARRIVERS: int = 28

    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    BOTTOM_RIGHT: int = 0
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL


# ---------------------------------------------------------------------------
# Resource derivations — the single source of truth for both the validator
# below and the kernel's allocation block.  Never inline these numbers.
# ---------------------------------------------------------------------------


def operand_bytes(cfg: CfgBwdD512) -> Tuple[int, int, int, int]:
    """(Q, dO, K-per-stage, V-per-stage) SMEM bytes for ONE CTA.

    K and V are split along the collective MMA-N across the CTA pair, so each
    CTA holds ``TILE_N / CTA_MMA`` rows of them; Q and dO are whole-block.
    """
    q = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    do = cfg.TILE_M * cfg.TILE_O * cfg.BPE
    k = (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE
    v = (cfg.TILE_N * cfg.TILE_O // cfg.CTA_MMA) * cfg.BPE
    return q, do, k, v


def xfer_bytes(cfg: CfgBwdD512) -> int:
    """fp32 S staged for the cross-sub-group ship, summed over the ring.

    fp32 and not the io dtype on purpose: the Rubin reference never transfers S
    at all (it multiplies the fp32 register copy straight into dS), so shipping
    a rounded S across the role split would be an accuracy regression the
    reference does not have.  Halving the tile keeps the ring 2-deep
    at an identical footprint.
    """
    return cfg.XFER_HALVES * cfg.TILE_M * (cfg.TILE_N // cfg.XFER_HALVES) * 4


def s_tma_iters(cfg: CfgBwdD512) -> int:
    """Workspace TMA subtiles per stored tile.

    One row of a subtile must be exactly one swizzle atom, so this is
    ``(TILE_N * BPE) // S_SWZ_BYTES`` -- 2 at bf16/TILE_N=128/128 B.  The
    validator pins ``XFER_HALVES`` to it: one shipped fp32 half is exactly one
    stored subtile is exactly one softmax chunk.
    """
    return (cfg.TILE_N * cfg.BPE) // cfg.S_SWZ_BYTES


def cast_bytes(cfg: CfgBwdD512) -> int:
    """io-dtype staging for one workspace tile (S on sg0, dS on sg1).

    Separate from the fp32 xfer buffer so the fp32 ring slot is released on
    cast completion instead of on ``tma_store_wait``.
    """
    return cfg.TILE_M * cfg.TILE_N * cfg.BPE


def smem_bytes(cfg: CfgBwdD512) -> Tuple[int, int]:
    """(sg0, sg1) SMEM tensor bytes per CTA.

    Each sub-group's operand slab is a max-union alias, not a sum: the Q (resp.
    dO) staging buffer is dead once the UTCCP has moved it to TMEM, so the K
    (resp. V) ring reuses those exact bytes behind the alias-seam barrier.
    """
    q, do, k, v = operand_bytes(cfg)
    sg0_alias = max(q, cfg.STAGES_KV * k)
    sg1_alias = max(do, cfg.STAGES_KV * v)
    common = xfer_bytes(cfg) + cast_bytes(cfg)
    return sg0_alias + common, sg1_alias + common


def tmem_cols(cfg: CfgBwdD512) -> Tuple[int, int]:
    """(sg0, sg1) TMEM columns.

    An accumulator is fp32 and TILE_N wide => TILE_N columns per parity slot.
    A 16-bit operand packs 2 elements per 32-bit column, so [TILE_M x d] costs
    ``d * BPE / 4`` columns -- 256 at d=512, which is exactly what leaves room
    for the two accumulator slots under the 512-column cap.
    """
    acc = cfg.STAGES_ACC * cfg.TILE_N
    return acc + (cfg.TILE_K * cfg.BPE) // 4, acc + (cfg.TILE_O * cfg.BPE) // 4


def _validate_cfg_d512(cfg: CfgBwdD512) -> None:
    """Consistency checks on the (mostly hardcoded) stage-2 d512 geometry."""
    sg0_smem, sg1_smem = smem_bytes(cfg)
    sg0_tmem, sg1_tmem = tmem_cols(cfg)
    checks = (
        # --- register split: all four are hardware constraints -------------
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "bwd d512: MMA/TMALDG/TMASTG/Scheduler regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "bwd d512: register budget over 512"),
        (
            cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0,
            "bwd d512: per-role regs must be multiples of 8",
        ),
        (24 <= cfg.SOFTMAX_REGS <= 256 and 24 <= cfg.MMA_REGS <= 256, "bwd d512: per-warp regs must be within [24, 256]"),
        # --- geometry ------------------------------------------------------
        (cfg.TILE_M == 128 and cfg.TILE_N == 128, "bwd d512: TILE_M = TILE_N = 128"),
        (cfg.TILE_K == 512 and cfg.TILE_O == 512, "bwd d512: d_qk = d_v = 512"),
        (cfg.CGA_M == 4 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2, "bwd d512: cga4x1 / CTA_MMA=2 only"),
        (cfg.CGA_M // cfg.CTA_MMA == 2, "bwd d512 (role split): exactly two sub-groups (CGA_M / CTA_MMA == 2)"),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 0, "bwd d512 (role split): one softmax warpgroup, no correction warp"),
        (cfg.TILE_N % cfg.XFER_HALVES == 0, "bwd d512: XFER_HALVES must divide TILE_N"),
        # A four-way alignment that the store layout, the ship ring and the
        # softmax register budget all depend on: one fp32 S half == one softmax
        # chunk == one workspace TMA subtile == one 128 B swizzle atom per row.
        # (TILE_N * BPE) // S_SWZ_BYTES is the subtile count; at TILE_N=128,
        # BPE=2, 128 B swizzle that is 2, and XFER_HALVES=2 matches it.  The
        # forward asserts the same coincidence as P_TMA_ITERS == P_XFER_HALVES.
        (
            cfg.XFER_HALVES == (cfg.TILE_N * cfg.BPE) // cfg.S_SWZ_BYTES,
            f"bwd d512: XFER_HALVES ({cfg.XFER_HALVES}) must equal the workspace TMA subtile count "
            f"({(cfg.TILE_N * cfg.BPE) // cfg.S_SWZ_BYTES}) -- one ship half must be exactly one stored subtile",
        ),
        # --- MMA: bf16/fp16 is 1-chunk only on SM10x -----------------------
        (
            cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16,
            "bwd d512 f16/bf16: TILE_K_HW must be 16 (1-chunk only on SM10x -- 2-chunk silently wrong)",
        ),
        (cfg.BPE == bpe(cfg.DTYPE_QKV), "bwd d512: BPE must match DTYPE_QKV"),
        # The workspace dtype IS the io dtype.  An earlier Rubin revision tied
        # it to an independent output dtype and produced a silent mismatch
        # against the stage-3 GEMMs, which read the workspace as the io dtype.
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "bwd d512: workspace/io dtype must be BF16 or FP16"),
        # --- swizzle -------------------------------------------------------
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.DO_SWZ_BYTES == 128,
            "bwd d512: Q/K/V/dO swizzle must all be 128B",
        ),
        # --- caps ----------------------------------------------------------
        (
            sg0_smem <= _SM100_MAX_DYN_SMEM,
            f"bwd d512: sg0 SMEM {sg0_smem / 1024:.1f} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (
            sg1_smem <= _SM100_MAX_DYN_SMEM,
            f"bwd d512: sg1 SMEM {sg1_smem / 1024:.1f} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (
            sg0_tmem <= _SM100_TMEM_COLS,
            f"bwd d512: sg0 TMEM carve {sg0_tmem} over the {_SM100_TMEM_COLS}-column Blackwell cap",
        ),
        (
            sg1_tmem <= _SM100_TMEM_COLS,
            f"bwd d512: sg1 TMEM carve {sg1_tmem} over the {_SM100_TMEM_COLS}-column Blackwell cap",
        ),
        # --- masks ---------------------------------------------------------
        (not (cfg.BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)), "bwd d512: bottom_right requires a causal band"),
        (
            cfg.READ_TILE_ARRIVERS == (cfg.SOFTMAX_WG_WARPS + 3) * cfg.CGA_M * cfg.CGA_N,
            f"bwd d512: READ_TILE_ARRIVERS ({cfg.READ_TILE_ARRIVERS}) must equal "
            f"{(cfg.SOFTMAX_WG_WARPS + 3) * cfg.CGA_M * cfg.CGA_N} "
            "= (softmax warps + TMA-LDG + TMA-STG + MMA) * CGA_SIZE -- every role but the scheduler, on every CTA",
        ),
        (cfg.COMPUTE_LANES == cfg.SOFTMAX_WG_WARPS * 32, "bwd d512: COMPUTE_LANES must be SOFTMAX_WG_WARPS * 32"),
        # v1 scope, deliberately narrow so the engine row cannot over-promise:
        # dense only, natural (grid-mapped) schedule.  Masks and persistence are
        # Phase 4 / Phase 5; each needs its own capabilities row + reject test.
        # Causal is implemented (tile-level triangular skip + a post-exp2
        # per-cell mask on the diagonal tiles).  SWA and padding are NOT, and
        # neither is bottom-right alignment: under bottom-right with
        # S_kv < S_q the leading q tiles get an EMPTY kv range, and this
        # kernel has no empty-tile path -- every cross-CTA ring would have to
        # fire a bookkeeping arrive so the four CTAs stay in step.  Top-left
        # causal always keeps kv_right >= 1, which is why it needs none.
        # Causal (top-left and bottom-right) and SWA are implemented: the tile
        # bounds come from the shared `compute_kv_loop_bounds` and the per-cell
        # mask from `apply_mask_chunk`, both driven by MASK_FLAGS.
        #
        # An EMPTY kv range (which bottom-right and SWA both admit) needs no
        # special path here: every ring except the per-tile operand one is
        # per-kv-iteration, so a zero-trip loop fires nothing, and
        # `_residual_depth(0, stages)` is 0. The three conditions that make that
        # safe all hold -- all four CTAs derive IDENTICAL bounds (they are keyed
        # on the cluster's q span), `read_tile_id_arrive` sits ABOVE the kv loop,
        # and the cast/xfer buffers are not aliased into the operand slab.
        #
        # Padding (`seq_kv_lens`) is still out: it needs the PER-BATCH kv length
        # at the bounds and mask sites, and this kernel threads only the scalar.
        # Padding IS implemented, for a UNIFORM length: the engine rounds the
        # compile shape up to the tile and passes the real S_q / S_kv, and the
        # kernel masks the tail (kv side through apply_mask_chunk, q side by
        # zeroing the row). That is what makes a sequence length which is not a
        # multiple of the tile work at all. A PER-BATCH seq_len still needs the
        # per-batch value threaded to the bounds and mask sites.
        # LPT / LPT_L2 need lpt_tile_coords and its L2-residency model, which
        # bwd/kernels/_common_sm100.py deliberately does not copy.
        (cfg.SCHEDULER_POLICY == SCHED_NATURAL, "bwd d512 v1: only SCHED_NATURAL is implemented (LPT/LPT_L2 need the L2 tile-coord model)"),
        (cfg.ACC_EMPTY_ARRIVERS == cfg.COMPUTE_LANES * cfg.CTA_MMA, "bwd d512: ACC_EMPTY_ARRIVERS must be COMPUTE_LANES * CTA_MMA"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512(params: TemplateParams) -> CfgBwdD512:
    _validate_params(params)
    b = bpe(params.dtype_qkv)
    cfg = CfgBwdD512(
        DTYPE_QKV=params.dtype_qkv,
        BPE=b,
        XFER_HALVES=params.xfer_halves,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        BOTTOM_RIGHT=int(params.bottom_right),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SCHEDULER_POLICY=params.sched_policy,
    )
    _validate_cfg_d512(cfg)
    return cfg
