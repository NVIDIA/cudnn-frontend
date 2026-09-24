# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the FROST SM107 (Rubin) SDPA-backward d256 flavors.

Two kernel BODIES share this module, one per dtype family, and each is its own
pipeline rather than a dtype arm of the other:

* ``kernels/sm107/bprop_d256_f16.py`` (bf16 / fp16): natural MMA order
  ``Q.K -> dO.V -> P.dO``, P written back INTO the S accumulator (no P ring, no
  ``s_acc_empty``), BMM1 K-split (the K back-half is UTCCP'd into the spare
  TMEM columns and the freed SMEM hosts the second ``dO_dv`` stage).
* ``kernels/sm107/bprop_d256_fp8.py`` (per-tensor FP8 E4M3): lookahead MMA
  order ``Q.K[i+1]`` ahead of ``S.dO[i]``, a 2-stage fp8 P ring in TMEM, three
  3-deep Q / dO rings.

Both compute dV in-kernel and store dS to a GMEM workspace; dK and dQ are the
``bprop_matmul_blackwell`` GEMMs over that workspace, and the FP8 chain writes
its dS as **bf16** so the bf16 GEMM renderings consume it unchanged (the fp8
GEMM arm is a follow-up).  GQA is folded by ``bprop_chain_common.dkv_reduce``.

**Why this is a separate module from** :mod:`cudnn.sdpa.bwd.config_sm100`: the
same reason the forward has one (:mod:`cudnn.sdpa.fwd.config_sm107`).  The
Blackwell d512 backward is a cga4x1 role split with 8 warps; these Rubin bodies
are a single cga2 pair with 12 warps, a different TMEM map (576 columns) and a
327 KiB SMEM layout.  The two overlap in the NAME of a few fields and in
nothing else, and routing one body through the other's ``make_cfg`` is how the
forward d256 FP8 port got a scheduler-ring init count no warp population could
reach -- a 100 %-GPU hang at every shape with no fault and nothing for a
sanitizer to find.  A config value is a claim about a body; the body cannot
check it at run time, so the claims are pinned here, per body.

Provenance: every value is the one the two bodies were written and validated
against (dV / dK / dQ ``cos = 1.0`` on Rubin), keyed here by op geometry
(``d256``) per the engine contract.  **Field names are the pre-port bodies'
own, verbatim** (``STAGES_dO``, ``SWA_WINDOW``, ``CAUSAL_BOTTOM_RIGHT``, ...),
so a kernel port can reference ``CFG.<name>`` without a rename pass; the
FROST-only additions carry NEW names (``IS_FP8``, ``DTYPE_DS`` / ``BPE_DS``,
``IDESC_K_DIM``, ``K_SPLIT_UTCCP``, ``XFER_STAGES``, ``STATS_STAGES``,
``READ_TILE_ARRIVERS_TOT``, ``SOFT_X_CTA_MMA``, ``MMA_COMMIT_ARRIVES``,
``SEQ_KV_LENS_PRESENT``).  Two dead knobs of the pre-port bodies
(``SPLIT_PIPELINE``, ``V2_PIPELINE``) and the unused ``dK_SWZ_BYTES`` are NOT
carried -- the ports delete their (already dead) uses.

**Dtype codes are ``tile_dsl.constants.DTYPE_*``** (E4M3 = 0, E5M2 = 1,
BF16 = 2, FP16 = 3), NOT the pre-port bodies' private 0/1/2.  The f16 body's
``STORAGE_DTYPE = Float16 if DTYPE_QKV == 0 else BFloat16`` dispatch and the
``DTYPE_O in (1, 2)`` grad-dtype dispatch must be rewritten against
``DTYPE_FP16`` / ``DTYPE_BF16`` in the port; the fp8 body's ``DTYPE_QKV == 0``
happens to coincide with ``DTYPE_E4M3`` and stays correct.

What this module is the single source of truth for (never re-literal these in
a kernel):

1. the SMEM buffer table in DECLARATION ORDER (:func:`smem_layout`) -> the byte
   tally the launcher uses (:func:`kernel_smem_bytes`), the per-CTA cap
   predicate, and the tcgen05 descriptor version (:func:`desc_version`, from
   the ``build()`` ROOTS of that table -- rules/mma-tma-matrix.md S6);
2. the TMEM column map (:func:`tmem_layout`, 576 columns, ``is_exclusive``);
3. the 12-warp register split and warp ids;
4. the mbarrier lane constants and the scheduler-ring arriver count
   (:func:`read_tile_arrivers_tot`, derived from the body's call sites);
5. the ring depths the bodies were validated at.

All configuration errors raise :class:`ValueError`.  Anything a *user-built
graph* could trip must be rejected earlier, by the engine row's
``Capabilities`` / the adapter's ``check_support`` -- a ``ValueError`` from here
means those have a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
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

# The per-graph record is SHARED with the SM100 backward so the two adapters
# spell dtype / band / padding / schedule identically; the Rubin body's extra
# knobs are appended below (see TemplateParams).
from cudnn.sdpa.bwd.config_sm100 import TemplateParams as _BwdTemplateParams

__all__ = [
    "TemplateParams",
    "CfgBwdD256",
    "BufferElems",
    "TmemLayout",
    "SmemSlab",
    "FAMILY_F16",
    "FAMILY_FP8",
    "SMEM_CAP_BYTES",
    "SMEM_SCAFFOLD_BYTES",
    "SMEM_USABLE_BYTES",
    "TCGEN05_V0_ADDR_LIMIT",
    "TMEM_TOTAL_COLS",
    "TMEM_IS_EXCLUSIVE",
    "REG_BUDGET_PER_CTA",
    "reg_entry_pool",
    "make_cfg_d256_bwd",
    "bpe",
    "buffer_elems",
    "tmem_layout",
    "smem_layout",
    "smem_bytes",
    "kernel_smem_bytes",
    "desc_roots",
    "desc_version",
    "mbar_stage_counts",
    "scaffold_bytes_declared",
    "read_tile_arrivers_tot",
    "validate_head_chunk",
    "q_pad_rows",
    "kv_pad_rows",
    "ds_workspace_bytes",
    "launch_grid",
]


# ---------------------------------------------------------------------------
# Rubin architecture constants
# ---------------------------------------------------------------------------

# Rubin SM10.7 oversized per-CTA SMEM cap (``CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_
# SHARED_MEMORY_PER_BLOCK`` = 334848 B, CUDA 13.4+).  Reaching past 227 KiB
# requires the launcher to set ALLOW_OVERSIZED_SHARED_MEMORY; both bodies do
# (the f16 layout is 322 KiB of slabs).
SMEM_CAP_BYTES = 327 * 1024

# Everything that is NOT one of the 1024-B-aligned operand / staging slabs:
# the ~45 mbarriers (16-B-aligned Int64 arrays), the scheduler's two rings +
# 16-B response slots, the TMEM-pointer word.  The DECLARED tally
# (:func:`scaffold_bytes_declared`) is ~0.6 KiB on both bodies; 2 KiB is the
# budget it must stay under, so an added mbarrier ring surfaces here as a
# raise instead of as the last slab silently overflowing the cap.
SMEM_SCAFFOLD_BYTES = 2 * 1024

# What the slabs may use.  This is deliberately NOT the forward's 320 KiB
# ``SMEM_USABLE_BYTES``: that figure is one kernel's self-imposed guard, and
# the f16 backward body is a MEASURED 322 KiB-of-slabs layout that launched
# and validated on Rubin -- a 320 KiB ceiling would reject a kernel that runs.
# The honest bound is the device cap minus the declared scaffolding.
SMEM_USABLE_BYTES = SMEM_CAP_BYTES - SMEM_SCAFFOLD_BYTES

# 14-bit ``start_address`` window of a version-0 tcgen05 SMEM descriptor.  Any
# ``SmemTile.desc()`` ROOT (a ring stage start, or a ``.shifted()`` UTCCP
# source) at or past this wraps to offset 0 and the MMA multiplies the bottom
# of SMEM -- exactly-zero accumulator, no crash (rules/mma-tma-matrix.md S6).
TCGEN05_V0_ADDR_LIMIT = 256 * 1024

# TMEM columns.  Rubin raises the Blackwell 512 to 576, which needs
# ``tmem_alloc(..., is_exclusive=True)`` (tile_dsl/tmem.py); alloc and dealloc
# must both pass ``TmemLayout.TOTAL_COLS``.
TMEM_TOTAL_COLS = 576
TMEM_IS_EXCLUSIVE = True

# Register file: 65 536 x 32-bit per SM, one CTA per SM, so the sum of the
# per-warp counts over the LAUNCHED warps must fit 65536 / 32.
REG_BUDGET_PER_CTA = 65536 // 32  # 2048


def reg_entry_pool(total_warps: int) -> int:
    """The per-CTA register POOL that ``setmaxnreg`` redistributes = what the LAUNCH allocated: ptxas's entry
    count (65536 / 32 / total_warps, rounded DOWN to the 8-register allocation granule) times the warps -- NOT
    the 65536 / 32 register file.  ``setmaxnreg.inc`` can only draw what the ``.dec`` warps released into that
    pool, so a split whose per-warp sum exceeds it leaves the last INCREASE warp parked forever: a HANG at 100 %
    SM utilization at EVERY shape, no fault, no message, and no SASS pin sees it (the split is present and the
    counts look fine).  2026-09-23: the f16 body's 8 x 232 + 4 x 48 = 2048 > 12 x 168 = 2016 hung the very first
    Rubin launch of the port (1 cluster, 1 tile); the pre-port 8 x 232 + 4 x 40 = 2016 balanced exactly, as
    every CUTLASS warp-specialized split does.  Same rule as the "`.maxnreg 128` HANGS" dead end on d512."""
    return (REG_BUDGET_PER_CTA // total_warps) // 8 * 8 * total_warps


_SMEM_SLAB_ALIGN = 1024  # every slab is a 1024-B-aligned cutlass.Array

FAMILY_F16 = "f16"
FAMILY_FP8 = "fp8"
_FAMILIES = (FAMILY_F16, FAMILY_FP8)

_FLAVOR = {FAMILY_F16: "sm107 bwd d256 f16", FAMILY_FP8: "sm107 bwd d256 fp8"}


# ---------------------------------------------------------------------------
# Per-graph record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateParams(_BwdTemplateParams):
    """The SM100 backward record plus the two knobs only these bodies read.

    Inherited (spelled exactly as the SM100 adapter spells them): ``dtype_qkv``,
    ``window_left`` / ``window_right`` (``window_right`` set => causal,
    ``window_left`` set => SWA), ``bottom_right``, ``seq_kv_lens_present``,
    ``seq_q_lens_present``, ``thd_varlen``, ``sched_policy``, ``xfer_halves``.
    ``xfer_halves`` is a d512 role-split tuning knob with no counterpart in
    these bodies and is INERT here.  ``seq_q_lens_present`` and ``thd_varlen``
    are rejected (not implemented in v1).

    A plain :class:`cudnn.sdpa.bwd.config_sm100.TemplateParams` is also
    accepted by :func:`make_cfg_d256_bwd` (the two extras default).
    """

    # Gradient (dV; and via the quantize pass dK / dQ) storage dtype.  -1 =
    # inherit: the io dtype on the f16 family, E4M3 on the fp8 family (the
    # cuDNN ``sdpa_fp8_backward`` contract: FP8 grads + amax).  BF16 / FP16 on
    # the fp8 family is the pre-quantization output used for the bitwise A/B
    # against the pre-port kernel.
    dtype_o: int = -1
    # Informational only: the main kernel is sink-agnostic (dQ/dK/dV are already
    # sink-correct through the sink-aware LSE the forward wrote); it gates the
    # separate dsink reduction launch in the adapter.  No body effect.
    has_sink: bool = False


def bpe(dtype: int) -> int:
    """Bytes per storage element: FP8 codes 1, BF16 / FP16 2."""
    return 1 if dtype <= DTYPE_E5M2 else 2


def _is_fp8_code(dtype: int) -> bool:
    return dtype <= DTYPE_E5M2


# ---------------------------------------------------------------------------
# The Cfg record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgBwdD256:
    """Geometry of the Rubin d256 backward bodies (d_qk = d_v = 256, cga2).

    Defaults exist so the dataclass is constructible in a doc / test context;
    :func:`make_cfg_d256_bwd` sets every field explicitly and validates.
    """

    # --- Tile shape ---------------------------------------------------------
    TILE_M: int = 128  # kv rows per CTA (M of every collective MMA is TILE_M * CTA_MMA = 256)
    TILE_N: int = 128  # q cols per q-tile (inner loop)
    TILE_K: int = 256  # d_qk
    TILE_O: int = 256  # d_v

    # --- Dtype (tile_dsl.constants.DTYPE_* codes) ---------------------------
    DTYPE_QKV: int = DTYPE_BF16
    DTYPE_O: int = DTYPE_BF16  # dV (in-kernel) grad storage dtype
    BPE: int = 2
    BPE_O: int = 2
    # FROST-only: 1 on the fp8 family, 0 on f16 (the bodies' dtype dispatch).
    IS_FP8: int = 0
    # FROST-only: dS GMEM-workspace dtype.  f16: the io dtype (the stage-3 GEMMs
    # read the workspace as the io dtype -- a mismatch is silent garbage).  fp8:
    # BF16, so the bf16 GEMM renderings consume it unchanged.  EVERY dS-ring
    # constant (SMEM bytes, TMA box, subtile count, swizzle row) derives from
    # BPE_DS, never from BPE: with an e4m3 io and a bf16 dS the two geometries
    # differ, and a dS walk borrowed from BPE passes every f16 test and fails
    # only fp8.
    DTYPE_DS: int = DTYPE_BF16
    BPE_DS: int = 2

    # --- Cluster (single cga2 pair; both CTAs collaborate on every MMA) ------
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    # --- Per-tensor swizzle (all 128 B at d=256) ----------------------------
    Q_SWZ_BYTES: int = 128
    dO_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    P_SWZ_BYTES: int = 128  # the dS ring's store_swizzled pattern (Swizzle(3,4,3))
    dS_SWZ_BYTES: int = 128  # the dS ring's TMA-store descriptor swizzle
    dV_SWZ_BYTES: int = 128  # dV epilogue staging + TMA-store descriptor swizzle

    # --- MMA k-step (arch-sensitive) + idesc k_dim --------------------------
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    # FROST-only: the ``Tcgen05InstrDesc.build(k_dim=)`` value every idesc in
    # the body must pass.  Rubin dense FP8 = K64 + k_dim=1 (arch-OPPOSITE of
    # Blackwell); f16 = K16 + the default 0.  A mismatch scrambles accumulator
    # ROWS silently (rules/mma-tma-matrix.md S1).
    IDESC_K_DIM: int = 0
    # FROST-only: 1 = the f16 BMM1 K-split (K back-half UTCCP'd to TMEM, the
    # freed SMEM aliased by dO_dv stage 1).  Decides the TMEM RSVD purpose and
    # the combined [sdOdv_s0 | K | V] SMEM slab.
    K_SPLIT_UTCCP: int = 0

    # --- Pipeline depths ----------------------------------------------------
    STAGES_Q: int = 2
    STAGES_dO: int = 2  # sdO -- BMM1 dP B-operand ring
    STAGES_dO_DV: int = 2  # sdO_dv -- BMM2 dV B-operand ring (f16: stage 1 = K back-half alias)
    STAGES_KV: int = 1  # K, V loaded ONCE per kv-block
    STAGES_TMEM_S: int = 1  # S / dP accumulators single-buffered
    STAGES_TMEM_P: int = 1  # f16: P inside S_acc; fp8: 2-stage fp8 P ring in TMEM [512..576)
    # FROST-only (module constants in the pre-port bodies): dS SMEM ring depth
    # and the lse / do_dot prefetch ring depth.
    XFER_STAGES: int = 1
    STATS_STAGES: int = 2
    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2

    # --- Warp specialization (8 compute warps = 2 wg x 4, lane = kv row) ----
    SOFTMAX_WARPGROUPS: int = 2
    SOFTMAX_WG_WARPS: int = 4
    CORRECTION_WARPS: int = 0

    # --- Register budget ----------------------------------------------------
    SOFTMAX_REGS: int = 232
    CORRECTION_REGS: int = 0
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    # --- Mask / sink (compile-time; the TRANSPOSE of the forward) -----------
    # lane = kv row, inner loop = q tile: a fixed kv-block bounds WHICH q-tiles
    # attend and the per-cell mask zeroes P on masked (kv, q) cells; P = 0 makes
    # dV / dS / dK / dQ inherit the mask.
    MASK_FLAGS: int = MASK_NONE
    SWA_WINDOW: int = 0  # window_left (keys q - W .. q); 0 when MASK_SWA is unset
    CAUSAL_BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0  # informational only (no main-kernel effect)
    # FROST-only: per-batch kv lengths are threaded (MASK_PADDED).
    SEQ_KV_LENS_PRESENT: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL  # 0/1/2 = natural-3D / lpt / lpt_l2 (flat 1-D grid)

    # --- Warp layout (per CTA) ----------------------------------------------
    #   0..3  softmax wg0 (q[0:64])   4..7  softmax wg1 (q[64:128])
    #   8 MMA   9 TMALDG   10 TMASTG (dS store + stats prefetch)   11 scheduler
    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    # --- mbarrier lane constants (P3) ---------------------------------------
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_WG_LANES: int = 128  # one wg
    SOFTMAX_LANES: int = 256  # both wgs (every compute lane of one CTA)
    # FROST-only: every compute lane of BOTH CTAs arriving on the pair leader
    # (the LEADER-scope ``s_acc_empty`` / ``dp_empty`` / ``p_ready`` /
    # ``dv_acc_empty`` / ``tmem_dealloc`` counts).
    SOFT_X_CTA_MMA: int = 512
    # FROST-only: one ``tcgen05.commit`` multicast = one arrive per target CTA.
    MMA_COMMIT_ARRIVES: int = 1
    # FROST-only: per-CTA init count of the scheduler's ``mb_read_tile_id``;
    # see read_tile_arrivers_tot() for the derivation naming the body.
    READ_TILE_ARRIVERS_TOT: int = 21

    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 64


# ---------------------------------------------------------------------------
# Derived element counts / TMA geometry -- the pre-port bodies' module constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferElems:
    """The pre-port bodies' module-level buffer / TMA constants, by name.

    Bind them at module level in a kernel (``_B = buffer_elems(CFG)``,
    ``qBufferElems = _B.qBufferElems`` ...) instead of re-deriving.  Every dS
    figure is BPE_DS-driven (see ``CfgBwdD256.DTYPE_DS``).
    """

    _M_PER_CTA: int  # per-CTA q rows of the Q / dO N-split (TILE_N // CTA_MMA)
    qBufferElems: int  # one Q ring stage (per CTA): _M_PER_CTA x TILE_K
    dOBufferElems: int  # one dO ring stage (per CTA): _M_PER_CTA x TILE_O (== TILE_N x TILE_O / CTA_MMA for sdO_dv)
    kBufferElems: int  # K (per CTA): TILE_M x TILE_K
    vBufferElems: int  # V (per CTA): TILE_M x TILE_O
    pBufferElems: int  # a [kv, q] tile: TILE_M x TILE_N
    dSBufferElems: int  # one dS ring stage: TILE_M x TILE_N (elements of DTYPE_DS)
    dVBufferElems: int  # dV epilogue staging: TILE_M x TILE_O (elements of DTYPE_O)
    K_BACK_OFF_ELEMS: int  # f16 K-split: the K back-half (d_qk[128:256]) offset inside sK
    Q_BACK_OFF_ELEMS: int  # f16 K-split: the Q back-half offset inside a sQ stage
    _DODV_STAGE_ELEMS: int  # sdO_dv ring stride (f16: dOBufferElems + K_BACK_OFF_ELEMS so stage 1 == sK_back)
    STATS_SLOT_ELEMS: int  # fp32 elems per stats ring slot: lse[TILE_N] | do_dot[TILE_N]
    STATS_LSE_OFF: int
    STATS_DOT_OFF: int
    _SMX_CHUNK: int  # q cols per softmax wg (TILE_N // SOFTMAX_WARPGROUPS)
    _LDTM_NUM: int  # tcgen05.ld.32x32b.xN granularity of the S / dP reads
    _DSOFT_CHUNK: int  # dSoftmax register-pressure chunk
    _KV_BLOCK_ROWS: int  # kv rows per cga2 pair (TILE_M * CTA_MMA)
    CGA_SIZE: int
    # TMA subtile walks (inner bytes / swizzle bytes).
    TMA_QK_ITERS: int
    TMA_VO_ITERS: int
    TMA_QK_GRANU_ELEMS: int
    TMA_VO_GRANU_ELEMS: int
    TMA_QK_SG1_ITERS: int  # the BT=true dO_dv / Q views (d-split across the pair)
    TMA_VO_SG1_ITERS: int
    TMA_QK_SG1_GRANU_ELEMS: int
    TMA_VO_SG1_GRANU_ELEMS: int
    DV_D_BLOCK: int  # dV store subtile width in d_v elems (dV_SWZ_BYTES / BPE_O)
    TMA_DV_ITERS: int
    TMA_DV_GRANU_ELEMS: int
    DV_BLOCK_SLAB: int  # TILE_M * DV_D_BLOCK
    P_TMA_ITERS: int  # dS store subtiles per row: TILE_N * BPE_DS / dS_SWZ_BYTES
    P_D_BLOCK: int  # q cols per dS store subtile
    P_BLOCK_BYTES: int  # (pre-port misnomer, kept) ELEMS per dS subtile = TILE_M * P_D_BLOCK
    pXferBytes: int  # bytes of one dS ring stage = TILE_M * TILE_N * BPE_DS
    # Byte transactions the expect_tx sites arm (cga2 tensor TMAs route both
    # peers' bytes to the LEADER's mbar -> x CTA_MMA; the dV store is per-CTA).
    qTmaTransactionBytes: int
    dOTmaTransactionBytes: int
    kTmaTransactionBytes: int
    vTmaTransactionBytes: int
    dVTmaTransactionBytes: int
    # tcgen05 SMEM descriptor leading / stride byte offsets.
    LEADING_BYTE_OFFSET_QK: int
    STRIDE_BYTE_OFFSET_QK: int
    LEADING_BYTE_OFFSET_dO: int
    STRIDE_BYTE_OFFSET_dO: int
    LEADING_BYTE_OFFSET_P: int
    STRIDE_BYTE_OFFSET_P: int
    LEADING_BYTE_OFFSET_dS: int
    STRIDE_BYTE_OFFSET_dS: int
    LEADING_BYTE_OFFSET_dV: int
    STRIDE_BYTE_OFFSET_dV: int
    LEADING_BYTE_OFFSET_Q_SG1: int  # BT=true B operand: K x swz (B_PC_COLS // 8 > 8 heuristic)
    STRIDE_BYTE_OFFSET_Q_SG1: int
    LEADING_BYTE_OFFSET_dO_SG1: int
    STRIDE_BYTE_OFFSET_dO_SG1: int
    # SmemTile.layout codes (128 B -> 2, 64 B -> 4, 32 B -> 6).
    SMEM_LAYOUT_Q: int
    SMEM_LAYOUT_dO: int
    SMEM_LAYOUT_K: int
    SMEM_LAYOUT_V: int
    SMEM_LAYOUT_P: int
    SMEM_LAYOUT_dS: int
    SMEM_LAYOUT_dV: int


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}


def buffer_elems(cfg: CfgBwdD256) -> BufferElems:
    m_per_cta = cfg.TILE_N // cfg.CTA_MMA
    q_elems = m_per_cta * cfg.TILE_K
    do_elems = m_per_cta * cfg.TILE_O
    k_elems = cfg.TILE_M * cfg.TILE_K
    v_elems = cfg.TILE_M * cfg.TILE_O
    k_back = k_elems // 2
    tma_qk_iters = (cfg.TILE_K * cfg.BPE) // cfg.Q_SWZ_BYTES
    tma_vo_iters = (cfg.TILE_O * cfg.BPE) // cfg.dO_SWZ_BYTES
    tma_qk_sg1_iters = ((cfg.TILE_K // cfg.CTA_MMA) * cfg.BPE) // cfg.Q_SWZ_BYTES
    tma_vo_sg1_iters = ((cfg.TILE_O // cfg.CTA_MMA) * cfg.BPE) // cfg.dO_SWZ_BYTES
    dv_d_block = cfg.dV_SWZ_BYTES // cfg.BPE_O
    p_tma_iters = (cfg.TILE_N * cfg.BPE_DS) // cfg.dS_SWZ_BYTES
    p_d_block = cfg.TILE_N // p_tma_iters
    return BufferElems(
        _M_PER_CTA=m_per_cta,
        qBufferElems=q_elems,
        dOBufferElems=do_elems,
        kBufferElems=k_elems,
        vBufferElems=v_elems,
        pBufferElems=cfg.TILE_M * cfg.TILE_N,
        dSBufferElems=cfg.TILE_M * cfg.TILE_N,
        dVBufferElems=cfg.TILE_M * cfg.TILE_O,
        K_BACK_OFF_ELEMS=k_back,
        Q_BACK_OFF_ELEMS=q_elems // 2,
        _DODV_STAGE_ELEMS=(do_elems + k_back) if cfg.K_SPLIT_UTCCP else do_elems,
        STATS_SLOT_ELEMS=2 * cfg.TILE_N,
        STATS_LSE_OFF=0,
        STATS_DOT_OFF=cfg.TILE_N,
        _SMX_CHUNK=cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS,
        _LDTM_NUM=64,
        _DSOFT_CHUNK=64,
        _KV_BLOCK_ROWS=cfg.TILE_M * cfg.CTA_MMA,
        CGA_SIZE=cfg.CGA_M * cfg.CGA_N,
        TMA_QK_ITERS=tma_qk_iters,
        TMA_VO_ITERS=tma_vo_iters,
        TMA_QK_GRANU_ELEMS=cfg.TILE_K // tma_qk_iters,
        TMA_VO_GRANU_ELEMS=cfg.TILE_O // tma_vo_iters,
        TMA_QK_SG1_ITERS=tma_qk_sg1_iters,
        TMA_VO_SG1_ITERS=tma_vo_sg1_iters,
        TMA_QK_SG1_GRANU_ELEMS=(cfg.TILE_K // cfg.CTA_MMA) // tma_qk_sg1_iters,
        TMA_VO_SG1_GRANU_ELEMS=(cfg.TILE_O // cfg.CTA_MMA) // tma_vo_sg1_iters,
        DV_D_BLOCK=dv_d_block,
        TMA_DV_ITERS=(cfg.TILE_O * cfg.BPE_O) // cfg.dV_SWZ_BYTES,
        TMA_DV_GRANU_ELEMS=dv_d_block,
        DV_BLOCK_SLAB=cfg.TILE_M * dv_d_block,
        P_TMA_ITERS=p_tma_iters,
        P_D_BLOCK=p_d_block,
        P_BLOCK_BYTES=cfg.TILE_M * p_d_block,
        pXferBytes=cfg.TILE_M * cfg.TILE_N * cfg.BPE_DS,
        qTmaTransactionBytes=q_elems * cfg.BPE * cfg.CTA_MMA,
        dOTmaTransactionBytes=do_elems * cfg.BPE * cfg.CTA_MMA,
        kTmaTransactionBytes=k_elems * cfg.BPE * cfg.CTA_MMA,
        vTmaTransactionBytes=v_elems * cfg.BPE * cfg.CTA_MMA,
        dVTmaTransactionBytes=cfg.TILE_M * cfg.TILE_O * cfg.BPE_O,
        LEADING_BYTE_OFFSET_QK=0,
        STRIDE_BYTE_OFFSET_QK=8 * cfg.Q_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dO=0,
        STRIDE_BYTE_OFFSET_dO=8 * cfg.dO_SWZ_BYTES,
        LEADING_BYTE_OFFSET_P=0,
        STRIDE_BYTE_OFFSET_P=8 * cfg.P_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dS=0,
        STRIDE_BYTE_OFFSET_dS=8 * cfg.dS_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dV=0,
        STRIDE_BYTE_OFFSET_dV=8 * cfg.dV_SWZ_BYTES,
        LEADING_BYTE_OFFSET_Q_SG1=cfg.TILE_N * cfg.Q_SWZ_BYTES,
        STRIDE_BYTE_OFFSET_Q_SG1=8 * cfg.Q_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dO_SG1=cfg.TILE_N * cfg.dO_SWZ_BYTES,
        STRIDE_BYTE_OFFSET_dO_SG1=8 * cfg.dO_SWZ_BYTES,
        SMEM_LAYOUT_Q=_SWZ_ENUM[cfg.Q_SWZ_BYTES],
        SMEM_LAYOUT_dO=_SWZ_ENUM[cfg.dO_SWZ_BYTES],
        SMEM_LAYOUT_K=_SWZ_ENUM[cfg.K_SWZ_BYTES],
        SMEM_LAYOUT_V=_SWZ_ENUM[cfg.V_SWZ_BYTES],
        SMEM_LAYOUT_P=_SWZ_ENUM[cfg.P_SWZ_BYTES],
        SMEM_LAYOUT_dS=_SWZ_ENUM[cfg.dS_SWZ_BYTES],
        SMEM_LAYOUT_dV=_SWZ_ENUM[cfg.dV_SWZ_BYTES],
    )


# ---------------------------------------------------------------------------
# TMEM column map (one alloc per CTA, all 576 columns, is_exclusive=True)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TmemLayout:
    """Column map of the pre-port ``KernelTmemLayout``, by field name.

    Both families: ``S_acc`` [0, 128) fp32 [kv, q]; ``dP`` [128, 256) fp32;
    ``dV_acc`` [256, 512) fp32 [kv, d_v] persistent over the q loop.

    * f16: P (the ``mma_ts`` A operand, ``TILE_N * BPE / 4 = 64`` cols) is
      written back INTO ``S_acc`` at [P_OFF=32, 96) -- wg0 [32, 64), wg1
      [64, 96), each inside its own S-read q-half; [512, 576) = RSVD holds the
      UTCCP'd K back-half (d_qk[128:256], ``(TILE_K / 2) * BPE / 4 = 64`` cols).
    * fp8: P is a ``STAGES_TMEM_P``-deep ring of ``TILE_N * BPE / 4 = 32`` cols
      at [512, 576); RSVD_COLS = 0.
    """

    TOTAL_COLS: int
    S_OFF: int
    S_COLS: int
    dP_OFF: int
    dP_COLS: int
    dV_OFF: int
    dV_COLS: int
    P_OFF: int
    P_COLS: int
    RSVD_OFF: int
    RSVD_COLS: int


def tmem_layout(cfg: CfgBwdD256) -> TmemLayout:
    s_cols = cfg.TILE_N
    dp_cols = cfg.TILE_N
    dv_cols = cfg.TILE_O
    p_cols = (cfg.TILE_N * cfg.BPE) // 4
    dv_off = s_cols + dp_cols
    tail = dv_off + dv_cols  # 512
    if cfg.IS_FP8:
        # fp8 P ring owns the tail; nothing reserved.
        return TmemLayout(
            TOTAL_COLS=TMEM_TOTAL_COLS,
            S_OFF=0,
            S_COLS=s_cols,
            dP_OFF=s_cols,
            dP_COLS=dp_cols,
            dV_OFF=dv_off,
            dV_COLS=dv_cols,
            P_OFF=tail,
            P_COLS=p_cols,
            RSVD_OFF=TMEM_TOTAL_COLS,
            RSVD_COLS=0,
        )
    # f16: P inside S_acc, base column = one wg's P half (P_COLS // SOFTMAX_WARPGROUPS)
    # so that wg1's half starts exactly at the S-read q-half boundary.
    return TmemLayout(
        TOTAL_COLS=TMEM_TOTAL_COLS,
        S_OFF=0,
        S_COLS=s_cols,
        dP_OFF=s_cols,
        dP_COLS=dp_cols,
        dV_OFF=dv_off,
        dV_COLS=dv_cols,
        P_OFF=p_cols // cfg.SOFTMAX_WARPGROUPS,
        P_COLS=p_cols,
        RSVD_OFF=tail,
        RSVD_COLS=((cfg.TILE_K // 2) * cfg.BPE) // 4 if cfg.K_SPLIT_UTCCP else 0,
    )


# ---------------------------------------------------------------------------
# SMEM buffer table in DECLARATION ORDER -- the kernel must declare its
# ``cutlass.Array(space=smem)`` slabs in exactly this order, or the desc-root
# tally below describes a layout the kernel does not have.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmemSlab:
    """One 1024-B-aligned ``cutlass.Array`` of the kernel's SharedStorage.

    ``roots`` are the tcgen05 ``build()`` roots inside it -- every ring stage
    start of a descriptor-read tile, plus the UTCCP ``.shifted()`` source --
    each as ``(label, absolute byte offset)``.  Lane-written / TMA-stored
    staging (dS ring, stats ring, the dV alias) has none: no descriptor reads
    it, so it may sit past the 256 KiB line under a version-0 descriptor.
    """

    name: str
    offset: int
    nbytes: int
    roots: Tuple[Tuple[str, int], ...] = ()


def _align_slab(nbytes: int) -> int:
    return (nbytes + _SMEM_SLAB_ALIGN - 1) // _SMEM_SLAB_ALIGN * _SMEM_SLAB_ALIGN


def smem_layout(cfg: CfgBwdD256) -> Tuple[SmemSlab, ...]:
    """The SharedStorage in declaration order, with every descriptor root.

    fp8 body (``sQ | sdO | sdOdv | sExcl[K | V] | sStats | sdS``):
      Q ring 3 x 16 = 48 KiB | dO ring 48 | dO_dv ring 48 | K 32 + V 32 = 64
      (the dV staging ALIASES it post-loop: max(K + V, dV @ BPE_O)) | stats 2 |
      dS ring 3 x 32 (bf16) = 96  -> 306 KiB.  Every root < 208 KiB.

    f16 body (``sQ | sdO | sCombined[sdOdv_s0 | K | V] | sStats | sdS``):
      Q ring 2 x 32 = 64 | dO ring 64 | dO_dv stage 0 32 + K 64 + V 64 = 160
      (dO_dv stage 1 aliases the K back-half freed by the UTCCP -- the ring
      stride is ``dOBufferElems + K_BACK_OFF_ELEMS``; dV staging aliases
      K + V post-loop) | stats 2 | dS ring 1 x 32 = 32  -> 322 KiB.  The last
      root is V at 224 KiB; sStats (288) and sdS (290) sit past the 256 KiB
      line but nothing descriptor-reads them.
    """
    b = buffer_elems(cfg)
    slabs = []
    off = 0

    def _push(name, nbytes, roots=()):
        nonlocal off
        slab = SmemSlab(name, off, _align_slab(nbytes), tuple((lbl, off + r) for lbl, r in roots))
        slabs.append(slab)
        off += slab.nbytes

    q_stage = b.qBufferElems * cfg.BPE
    do_stage = b.dOBufferElems * cfg.BPE
    k_bytes = b.kBufferElems * cfg.BPE
    v_bytes = b.vBufferElems * cfg.BPE
    dv_bytes = b.dVBufferElems * cfg.BPE_O
    kv_alias = max(k_bytes + v_bytes, dv_bytes)

    _push("sQ", cfg.STAGES_Q * q_stage, [(f"sQ[{s}]", s * q_stage) for s in range(cfg.STAGES_Q)])
    _push("sdO", cfg.STAGES_dO * do_stage, [(f"sdO[{s}]", s * do_stage) for s in range(cfg.STAGES_dO)])
    if cfg.K_SPLIT_UTCCP:
        # [sdOdv_s0 | K | V]; sdO_dv[1] == sK_back (alias), UTCCP source = sK_back.
        k_off = do_stage
        k_back_off = k_off + b.K_BACK_OFF_ELEMS * cfg.BPE
        roots = [("sdO_dv[0]", 0), ("sK", k_off), ("sK_back(UTCCP src)", k_back_off), ("sV", k_off + k_bytes)]
        if cfg.STAGES_dO_DV > 1:
            roots.append(("sdO_dv[1](=sK_back alias)", b._DODV_STAGE_ELEMS * cfg.BPE))
        _push("sCombined[sdOdv_s0|K|V](+sdV alias)", do_stage + kv_alias, roots)
    else:
        _push("sdOdv", cfg.STAGES_dO_DV * do_stage, [(f"sdO_dv[{s}]", s * do_stage) for s in range(cfg.STAGES_dO_DV)])
        _push("sExcl[K|V](+sdV alias)", kv_alias, [("sK", 0), ("sV", k_bytes)])
    _push("sStats", cfg.STATS_STAGES * b.STATS_SLOT_ELEMS * 4)
    _push("sdS", cfg.XFER_STAGES * b.dSBufferElems * cfg.BPE_DS)
    return tuple(slabs)


def smem_bytes(cfg: CfgBwdD256) -> int:
    """Slab bytes per CTA (the ``cutlass.Array`` allocations), scaffolding excluded."""
    return sum(s.nbytes for s in smem_layout(cfg))


def kernel_smem_bytes(cfg: CfgBwdD256) -> int:
    """Per-CTA SMEM the launcher must provision: slabs + the scaffold budget."""
    return smem_bytes(cfg) + SMEM_SCAFFOLD_BYTES


def desc_roots(cfg: CfgBwdD256) -> Tuple[Tuple[str, int], ...]:
    """Every tcgen05 SMEM-descriptor ``build()`` root, ``(label, byte offset)``."""
    return tuple(r for s in smem_layout(cfg) for r in s.roots)


def _needs_desc_v1(cfg: CfgBwdD256) -> bool:
    """True when any descriptor root starts at or past the 14-bit v0 window."""
    return any(off >= TCGEN05_V0_ADDR_LIMIT for _, off in desc_roots(cfg))


def desc_version(cfg: CfgBwdD256) -> int:
    """The ``desc_version=`` every ``SmemTile`` in the kernel must take (bind it
    ONCE as the module constant ``DESC_VERSION``; never a per-tile literal)."""
    return 1 if _needs_desc_v1(cfg) else 0


# ---------------------------------------------------------------------------
# Scaffolding: the mbarrier inventory + scheduler + TMEM pointer
# ---------------------------------------------------------------------------


def mbar_stage_counts(cfg: CfgBwdD256) -> Dict[str, int]:
    """``Bars`` field -> stage count, in the pre-port bodies' order.

    The kernel's ``Bars`` must allocate exactly these (the barrier-inspector
    audits init counts against this list).  ``mb_s_acc_empty`` exists only on
    the fp8 body (the f16 body's in-order MMA covers the S WAR); ``mb_k_utccp_done``
    only on the f16 body (the K-split alias seam).
    """
    rings = {
        "mb_q_full": cfg.STAGES_Q,
        "mb_q_empty": cfg.STAGES_Q,
        "mb_do_full": cfg.STAGES_dO,
        "mb_do_empty": cfg.STAGES_dO,
        "mb_dodv_full": cfg.STAGES_dO_DV,
        "mb_dodv_empty": cfg.STAGES_dO_DV,
        "mb_k_full": cfg.STAGES_KV,
        "mb_k_empty": cfg.STAGES_KV,
        "mb_v_full": cfg.STAGES_KV,
        "mb_v_empty": cfg.STAGES_KV,
    }
    if cfg.K_SPLIT_UTCCP:
        rings["mb_k_utccp_done"] = 1
    rings["mb_s_acc_full"] = cfg.STAGES_TMEM_S
    if cfg.IS_FP8:
        rings["mb_s_acc_empty"] = cfg.STAGES_TMEM_S
    rings.update(
        {
            "mb_dp_full": cfg.STAGES_TMEM_S,
            "mb_dp_empty": cfg.STAGES_TMEM_S,
            "mb_p_ready": cfg.STAGES_TMEM_P,
            "mb_stats_full": cfg.STATS_STAGES,
            "mb_stats_empty": cfg.STATS_STAGES,
            "mb_ds_smem_full": cfg.XFER_STAGES,
            "mb_ds_smem_empty": cfg.XFER_STAGES,
            "mb_dv_ready": 1,
            "mb_dv_acc_empty": 1,
            "mb_dv_stg_full": 1,
            "mb_dv_stg_empty": 1,
            "mb_tmem_dealloc": 1,
        }
    )
    return rings


_MBAR_ARRAY_ALIGN = 16  # cutlass.Array(Int64, n, alignment=16)
_SCHED_RESPONSE_WORDS = 8  # Int32 words per scheduler stage (16-B CLC response + slack)
_TMEM_PTR_BYTES = 16  # cutlass.Array(Int32, 1, alignment=16)


def scaffold_bytes_declared(cfg: CfgBwdD256) -> int:
    """Bytes the non-slab allocations declare: mbarriers (8 B each, arrays
    16-B aligned), the scheduler's two rings + response slots, the TMEM
    pointer.  An upper bound on what the DSL lays out; pinned under
    ``SMEM_SCAFFOLD_BYTES`` by the validator."""

    def _arr(n_qwords):
        return (8 * n_qwords + _MBAR_ARRAY_ALIGN - 1) // _MBAR_ARRAY_ALIGN * _MBAR_ARRAY_ALIGN

    mbars = sum(_arr(n) for n in mbar_stage_counts(cfg).values())
    sched = 2 * _arr(cfg.SCHEDULER_STAGES) + cfg.SCHEDULER_STAGES * _SCHED_RESPONSE_WORDS * 4
    return mbars + sched + _TMEM_PTR_BYTES


# ---------------------------------------------------------------------------
# Scheduler-ring arriver count -- derived from the body's call sites
# ---------------------------------------------------------------------------


def read_tile_arrivers_tot(cfg: CfgBwdD256) -> int:
    """Per-CTA init count of ``mb_read_tile_id``.

    ``read_tile_id_arrive(mb, CGA_SIZE)`` lands exactly ONE arrive on EVERY
    CTA of the cluster per calling warp, so each CTA's count is the number of
    warps calling it cluster-wide.  In BOTH bodies the persistent-loop warps
    that call it are: the ``SOFTMAX_WARPGROUPS * SOFTMAX_WG_WARPS`` compute
    warps (``_softmax_warp_group``), the TMA-LDG warp (``_tmaldg_warp``) and
    the TMA-STG warp (``_tmastg_warp``) -- on every CTA -- plus the MMA warp
    on the pair LEADER only (``_mma_warp``; the follower runs
    ``_mma_warp_quiet``, which has no persistent loop and no arrive).  The
    scheduler warp never arrives on its own ring.

        leader   : 8 + MMA + TMALDG + TMASTG = 11
        follower : 8 +       TMALDG + TMASTG = 10
        cluster  : 21 == CTA_MMA * (softmax + 2) + 1

    Too LOW is the dangerous direction: the ring completes before every role
    has read the payload and advances a tile early; too HIGH is an
    unreachable count = a hang at every shape.  Both are silent (P3).
    """
    compute = cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
    return cfg.CTA_MMA * (compute + 2) + 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


def _validate_params(flavor: str, family: str, params: _BwdTemplateParams) -> None:
    """Guard the record a Rubin d256 backward body can express.  Every raise
    here must also be a Capabilities decline -- reaching it is an engine-row bug."""
    dtype_o = getattr(params, "dtype_o", -1)
    if params.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: dtype_qkv must be a tile_dsl DTYPE_* code (E4M3=0 E5M2=1 BF16=2 FP16=3); got {params.dtype_qkv}")
    if family == FAMILY_FP8:
        if params.dtype_qkv != DTYPE_E4M3:
            raise ValueError(
                f"{flavor}: the fp8 body is E4M3-only (dtype_qkv={DTYPE_E4M3}); got {params.dtype_qkv}"
                + (" -- E5M2 is not implemented in this body" if params.dtype_qkv == DTYPE_E5M2 else " -- a half-precision io belongs to the f16 body")
            )
        if dtype_o not in (-1, DTYPE_E4M3, DTYPE_BF16, DTYPE_FP16):
            raise ValueError(
                f"{flavor}: dtype_o must be -1 (inherit -> E4M3, the fp8 graph contract), DTYPE_E4M3, DTYPE_BF16 or DTYPE_FP16 "
                f"(the pre-quantization output for the bitwise A/B); got {dtype_o}"
            )
    else:
        if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
            raise ValueError(
                f"{flavor}: the f16 body takes DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); got {params.dtype_qkv} -- an FP8 io belongs to the fp8 body"
            )
        if dtype_o not in (-1, DTYPE_BF16, DTYPE_FP16):
            raise ValueError(f"{flavor}: dtype_o must be -1 (inherit the io dtype), DTYPE_BF16 or DTYPE_FP16 on the f16 body; got {dtype_o}")
    # --- band -----------------------------------------------------------------
    if params.window_left is not None and params.window_left <= 0:
        raise ValueError(
            f"{flavor}: SWA requires window_left > 0 (got {params.window_left}); the transposed q-tile trim widens the "
            f"q range by SWA_WINDOW and a zero or negative window is not a sliding window"
        )
    if params.window_right is not None and params.window_right != 0:
        raise ValueError(
            f"{flavor}: window_right must be 0 when set (plain causal diagonal); got {params.window_right}. Right-band widening "
            f"is not implemented in this body -- its q-tile trim and per-cell mask know only kv <= q (+ the bottom-right shift)"
        )
    if params.bottom_right and params.window_right is None:
        raise ValueError(f"{flavor}: bottom_right alignment requires a causal band (window_right set)")
    # --- padding / THD ------------------------------------------------------------
    if params.seq_q_lens_present:
        raise ValueError(f"{flavor}: seq_q_lens_present is not implemented -- the body threads only the per-batch kv length (seq_kv_lens)")
    if params.thd_varlen:
        raise ValueError(f"{flavor}: thd_varlen is not implemented -- this body has no THD/varlen leg (dense BSHD only)")
    # --- schedule -----------------------------------------------------------------
    if params.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: sched_policy must be one of NATURAL/LPT/LPT_L2 (0/1/2); got {params.sched_policy}")


def _mask_flags_from(params: _BwdTemplateParams) -> int:
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.seq_kv_lens_present:
        flags |= MASK_PADDED
    return flags


def _validate_cfg_d256_bwd(cfg: CfgBwdD256, flavor: str) -> None:
    """Every predicate the bodies hardcode.  Each message names the failure
    SIGNATURE so the next reader recognises it."""
    b = buffer_elems(cfg)
    tm = tmem_layout(cfg)
    compute_warps = cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
    reg_total = (
        compute_warps * cfg.SOFTMAX_REGS + cfg.CORRECTION_WARPS * cfg.CORRECTION_REGS + cfg.MMA_REGS + cfg.TMALDG_REGS + cfg.TMASTG_REGS + cfg.SCHEDULER_REGS
    )
    # --- register split: all four are hardware constraints (12-warp form) -----
    _check(
        [
            (
                cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS,
                f"{flavor}: MMA/TMALDG/TMASTG/SCHEDULER regs must be equal (HW equality constraint); got "
                f"{cfg.MMA_REGS}/{cfg.TMALDG_REGS}/{cfg.TMASTG_REGS}/{cfg.SCHEDULER_REGS} -- a violation is wrong results or a crash with no diagnostic",
            ),
            (
                reg_total <= reg_entry_pool(cfg.TOTAL_WARPS),
                f"{flavor}: register split {reg_total} over the {cfg.TOTAL_WARPS}-warp ENTRY pool {reg_entry_pool(cfg.TOTAL_WARPS)} "
                f"(= {reg_entry_pool(cfg.TOTAL_WARPS) // cfg.TOTAL_WARPS}/thread at launch; the 65536/32 = {REG_BUDGET_PER_CTA} register file is NOT the bound) "
                f"({compute_warps} softmax @ {cfg.SOFTMAX_REGS} + 4 service @ {cfg.MMA_REGS}) -- setmaxnreg.inc can only take what .dec released: "
                f"the last softmax warp parks forever = a HANG at 100 % SM utilization at every shape, no fault, no message",
            ),
            (
                all(r % 8 == 0 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS, cfg.CORRECTION_REGS)),
                f"{flavor}: every per-role register count must be a multiple of 8 (setmaxnreg granularity)",
            ),
            (
                all(24 <= r <= 256 for r in (cfg.MMA_REGS, cfg.SOFTMAX_REGS, cfg.OTHER_REGS)),
                f"{flavor}: per-warp register counts must be within 24..256",
            ),
            (
                cfg.OTHER_REGS == cfg.MMA_REGS,
                f"{flavor}: OTHER_REGS is the service-warp count the body's setmaxregister(DECREASE) uses; it must equal MMA_REGS",
            ),
        ]
    )
    # --- warp population + scheduler ring ---------------------------------------
    _check(
        [
            (
                cfg.SOFTMAX_WARPGROUPS == 2 and cfg.SOFTMAX_WG_WARPS == 4 and cfg.CORRECTION_WARPS == 0,
                f"{flavor}: the body is 8 compute warps in 2 warpgroups of 4 (q-halves) and no correction warps; got "
                f"{cfg.SOFTMAX_WARPGROUPS}x{cfg.SOFTMAX_WG_WARPS} + {cfg.CORRECTION_WARPS}",
            ),
            (
                cfg.TOTAL_WARPS == compute_warps + cfg.CORRECTION_WARPS + 4,
                f"{flavor}: TOTAL_WARPS must be compute + correction + 4 service warps; got {cfg.TOTAL_WARPS}",
            ),
            (cfg.THREADS_PER_CTA == cfg.TOTAL_WARPS * 32, f"{flavor}: THREADS_PER_CTA must be TOTAL_WARPS*32"),
            (
                cfg.SOFTMAX_WG0_BASE == 0
                and cfg.SOFTMAX_WG1_BASE == cfg.SOFTMAX_WG_WARPS
                and cfg.MMA_WARP_ID == compute_warps + cfg.CORRECTION_WARPS
                and cfg.TMALDG_WARP_ID == cfg.MMA_WARP_ID + 1
                and cfg.TMASTG_WARP_ID == cfg.MMA_WARP_ID + 2
                and cfg.SCHED_WARP_ID == cfg.MMA_WARP_ID + 3 == cfg.TOTAL_WARPS - 1,
                f"{flavor}: warp ids must be wg0 @0, wg1 @{cfg.SOFTMAX_WG_WARPS}, MMA/TMALDG/TMASTG/SCHED consecutive after the compute warps "
                f"(the body's role dispatch is `warp_idx < compute` then == each id)",
            ),
            (
                cfg.READ_TILE_ARRIVERS_TOT == read_tile_arrivers_tot(cfg),
                f"{flavor}: READ_TILE_ARRIVERS_TOT must be {read_tile_arrivers_tot(cfg)} for this body (got {cfg.READ_TILE_ARRIVERS_TOT}) = "
                f"CTA_MMA * (compute warps + TMALDG + TMASTG) + the leader-only MMA warp; a wrong count is an unreachable (hang at EVERY shape) "
                f"or early-completing (scheduler advances a tile early, wedges past ~12 ring wraps) mbarrier",
            ),
            (
                cfg.CGA_M * cfg.CGA_N == cfg.CTA_MMA,
                f"{flavor}: a single cga2 pair -- the cluster IS the MMA pair (CGA_M*CGA_N == CTA_MMA), no sub-group role split",
            ),
        ]
    )
    # --- mbarrier lane constants (P3) -------------------------------------------
    _check(
        [
            (cfg.ONE_LANE == 1 and cfg.ONE_WARP == 32, f"{flavor}: ONE_LANE/ONE_WARP are 1/32"),
            (cfg.SOFTMAX_WG_LANES == cfg.SOFTMAX_WG_WARPS * 32, f"{flavor}: SOFTMAX_WG_LANES must be SOFTMAX_WG_WARPS*32 (one warpgroup's lanes)"),
            (
                cfg.SOFTMAX_LANES == compute_warps * 32,
                f"{flavor}: SOFTMAX_LANES must be every compute lane of one CTA ({compute_warps}*32); got {cfg.SOFTMAX_LANES}",
            ),
            (
                cfg.SOFT_X_CTA_MMA == cfg.SOFTMAX_LANES * cfg.CTA_MMA,
                f"{flavor}: SOFT_X_CTA_MMA (the LEADER-scope init count) must be SOFTMAX_LANES*CTA_MMA -- every compute lane of BOTH CTAs arrives on the leader",
            ),
            (cfg.MMA_COMMIT_ARRIVES == 1, f"{flavor}: one tcgen05.commit multicast is ONE arrive per target CTA (from one elected lane)"),
        ]
    )
    # --- geometry the bodies hardcode ----------------------------------------------
    _check(
        [
            (cfg.TILE_M == 128 and cfg.TILE_N == 128, f"{flavor}: TILE_M = TILE_N = 128 (kv rows per CTA, q cols per q-tile)"),
            (
                cfg.TILE_K == 256 and cfg.TILE_O == 256,
                f"{flavor}: d_qk = d_v = 256 (got TILE_K={cfg.TILE_K}, TILE_O={cfg.TILE_O}); the body is exact-d, no envelope",
            ),
            (cfg.CGA_M == 2 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2, f"{flavor}: single cga2 sub-group (CGA_M=2, CGA_N=1, CTA_MMA=2) only"),
            (cfg.TILES_Q == 1, f"{flavor}: TILES_Q == 1"),
            (cfg.SCHEDULER_STAGES == 2, f"{flavor}: SCHEDULER_STAGES == 2 (the CLC response ring depth the scheduler warp loop assumes)"),
            (cfg.N_BMM2_CHUNKS * cfg.BMM2_CHUNK_SIZE == cfg.TILE_N, f"{flavor}: N_BMM2_CHUNKS*BMM2_CHUNK_SIZE must equal TILE_N"),
            (
                cfg.TILE_N % cfg.SOFTMAX_WARPGROUPS == 0 and b._SMX_CHUNK % b._LDTM_NUM == 0,
                f"{flavor}: the softmax q-half (TILE_N/SOFTMAX_WARPGROUPS) must be a multiple of the LDTM x{b._LDTM_NUM} granule",
            ),
            (cfg.TILE_N % b._DSOFT_CHUNK == 0, f"{flavor}: TILE_N must be a multiple of the dSoftmax chunk ({b._DSOFT_CHUNK})"),
        ]
    )
    # --- dtype / k-step / idesc -----------------------------------------------------
    _check(
        [
            (cfg.IS_FP8 == int(_is_fp8_code(cfg.DTYPE_QKV)), f"{flavor}: IS_FP8 must follow DTYPE_QKV"),
            (
                cfg.BPE == bpe(cfg.DTYPE_QKV) and cfg.BPE_O == bpe(cfg.DTYPE_O) and cfg.BPE_DS == bpe(cfg.DTYPE_DS),
                f"{flavor}: BPE/BPE_O/BPE_DS must match their dtypes",
            ),
        ]
    )
    if cfg.IS_FP8:
        _check(
            [
                (cfg.DTYPE_QKV == DTYPE_E4M3, f"{flavor}: the fp8 body is E4M3-only (no E5M2 arm); got DTYPE_QKV={cfg.DTYPE_QKV}"),
                (
                    cfg.TILE_K_HW_BMM1 == 64 and cfg.TILE_K_HW_BMM2 == 64 and cfg.IDESC_K_DIM == 1,
                    f"{flavor}: Rubin dense FP8 runs the K=64 path and EVERY idesc must pass k_dim=1 (got TILE_K_HW {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}, "
                    f"IDESC_K_DIM={cfg.IDESC_K_DIM}); a mismatch silently scrambles accumulator ROWS (some exact, some 0.5x-2x, sign flips), passes a 1-tile "
                    f"shape and fails wide K (rules/mma-tma-matrix.md S1 -- arch-OPPOSITE of Blackwell's K32/k_dim=0)",
                ),
                (
                    cfg.BMM2_CHUNK_SIZE == cfg.TILE_K_HW_BMM2,
                    f"{flavor}: the fp8 BMM2 P chunk must be the K=64 hardware k-step (BMM2_CHUNK_SIZE == TILE_K_HW_BMM2)",
                ),
                (
                    cfg.DTYPE_DS == DTYPE_BF16,
                    f"{flavor}: the fp8 chain writes its dS workspace as BF16 so the bf16 stage-3 GEMM renderings consume it unchanged (got DTYPE_DS={cfg.DTYPE_DS}); "
                    f"an fp8 dS needs the fp8 GEMM arm (dequant epilogue + K64 idesc), which is a follow-up",
                ),
                (
                    cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_BF16, DTYPE_FP16),
                    f"{flavor}: DTYPE_O must be E4M3 (the fp8 graph contract) or BF16/FP16 (pre-quantization output); got {cfg.DTYPE_O}",
                ),
                (cfg.K_SPLIT_UTCCP == 0, f"{flavor}: the fp8 body has no BMM1 K-split (its spare TMEM columns hold the fp8 P ring)"),
                # ring depths the body was validated at (the lookahead needs Q[i], Q[i+1] and a prefetch in flight)
                (
                    cfg.STAGES_Q == 3 and cfg.STAGES_dO == 3,
                    f"{flavor}: Q / dO rings are 3-deep in this body (got {cfg.STAGES_Q}/{cfg.STAGES_dO}) -- the Q.K[i+1] lookahead keeps two stages live plus one prefetch; "
                    f"another depth was never validated on this body",
                ),
                (cfg.STAGES_dO_DV == cfg.STAGES_dO, f"{flavor}: the fp8 body drives the dO_dv ring at STAGES_dO (got STAGES_dO_DV={cfg.STAGES_dO_DV})"),
                (
                    cfg.STAGES_TMEM_P == 2 and tm.P_OFF + cfg.STAGES_TMEM_P * tm.P_COLS == tm.TOTAL_COLS,
                    f"{flavor}: the fp8 P ring is exactly 2 stages of {tm.P_COLS} cols filling TMEM [{tm.P_OFF}, {tm.TOTAL_COLS}) (got STAGES_TMEM_P={cfg.STAGES_TMEM_P}); "
                    f"the body indexes P_OFF + slot*P_COLS by the mb_p_ready PipelineState, so a deeper ring reads past the allocation and a shallower one breaks "
                    f"the 'store P[i+1] before dSoftmax[i]' overlap the missing p_empty relies on",
                ),
                (cfg.XFER_STAGES == 3, f"{flavor}: the fp8 dS SMEM ring is 3-deep in this body (got {cfg.XFER_STAGES}); other depths were never validated"),
            ]
        )
    else:
        _check(
            [
                (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), f"{flavor}: the f16 body takes BF16 or FP16; got DTYPE_QKV={cfg.DTYPE_QKV}"),
                (
                    cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16 and cfg.IDESC_K_DIM == 0,
                    f"{flavor}: f16/bf16 TILE_K_HW must be 16 with the default idesc k_dim=0 (got {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}, IDESC_K_DIM={cfg.IDESC_K_DIM}); "
                    f"the 2-chunk (K=32) f16 path is silently wrong on SM10x (rules/mma-tma-matrix.md S2)",
                ),
                (
                    cfg.DTYPE_DS == cfg.DTYPE_QKV,
                    f"{flavor}: the dS workspace dtype IS the io dtype (got DTYPE_DS={cfg.DTYPE_DS}, DTYPE_QKV={cfg.DTYPE_QKV}); the stage-3 GEMMs read the workspace "
                    f"as the io dtype and a mismatch is silent garbage gradients",
                ),
                (cfg.DTYPE_O in (DTYPE_BF16, DTYPE_FP16), f"{flavor}: DTYPE_O must be BF16 or FP16 on the f16 body; got {cfg.DTYPE_O}"),
                (cfg.K_SPLIT_UTCCP == 1, f"{flavor}: the f16 body is the BMM1 K-split body (K back-half UTCCP'd to TMEM); K_SPLIT_UTCCP must be 1"),
                # ring depths the body was validated at (327 KiB cap: 2-deep Q/dO, 1-stage dS)
                (
                    cfg.STAGES_Q == 2 and cfg.STAGES_dO == 2,
                    f"{flavor}: Q / dO rings are 2-deep in this body (got {cfg.STAGES_Q}/{cfg.STAGES_dO}); a third 32 KiB stage of either does not fit the 327 KiB cap",
                ),
                (
                    cfg.STAGES_dO_DV == 2 and b.dOBufferElems == b.K_BACK_OFF_ELEMS,
                    f"{flavor}: the dO_dv ring is 2-deep with stage 1 ALIASING the K back-half freed by the UTCCP (got STAGES_dO_DV={cfg.STAGES_dO_DV}); the alias is exact "
                    f"only when one dO_dv stage ({b.dOBufferElems} elems) equals half of K ({b.K_BACK_OFF_ELEMS} elems)",
                ),
                (
                    cfg.STAGES_TMEM_P == 1 and tm.P_OFF + tm.P_COLS <= tm.S_COLS,
                    f"{flavor}: f16 P is a single buffer INSIDE S_acc ([{tm.P_OFF}, {tm.P_OFF + tm.P_COLS}) must fit [0, {tm.S_COLS})); got STAGES_TMEM_P={cfg.STAGES_TMEM_P}",
                ),
                (
                    tm.P_OFF + tm.P_COLS // cfg.SOFTMAX_WARPGROUPS == cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS,
                    f"{flavor}: wg1's P half must begin exactly at the S-read q-half boundary (col {cfg.TILE_N // cfg.SOFTMAX_WARPGROUPS}) so each wg's P write stays inside its own S read",
                ),
                (
                    tm.RSVD_OFF + tm.RSVD_COLS == tm.TOTAL_COLS and tm.RSVD_COLS == ((cfg.TILE_K // 2) * cfg.BPE) // 4,
                    f"{flavor}: TMEM [{tm.RSVD_OFF}, {tm.TOTAL_COLS}) must hold exactly the UTCCP'd K back-half ((TILE_K/2)*BPE/4 = {((cfg.TILE_K // 2) * cfg.BPE) // 4} cols); got RSVD_COLS={tm.RSVD_COLS}",
                ),
                (
                    cfg.XFER_STAGES == 1,
                    f"{flavor}: the f16 dS SMEM ring is 1-deep in this body (got {cfg.XFER_STAGES}); a second 32 KiB stage does not fit the 327 KiB cap",
                ),
            ]
        )
    # --- ring depths common to both -----------------------------------------------------
    _check(
        [
            (cfg.STAGES_KV == 1, f"{flavor}: K and V are loaded ONCE per kv-block (STAGES_KV == 1); got {cfg.STAGES_KV}"),
            (
                cfg.STAGES_TMEM_S == 1,
                f"{flavor}: S and dP accumulators are single-buffered (STAGES_TMEM_S == 1); cross-iteration overlap is the MMA order, not a parity ring",
            ),
            (
                cfg.STATS_STAGES == 2,
                f"{flavor}: the lse/do_dot prefetch ring is 2-deep (got {cfg.STATS_STAGES}); the TMA-STG warp's prefetch-one-ahead assumes it",
            ),
        ]
    )
    # --- TMEM -----------------------------------------------------------------------------
    _check(
        [
            (
                tm.TOTAL_COLS == TMEM_TOTAL_COLS,
                f"{flavor}: TMEM alloc is the full {TMEM_TOTAL_COLS}-column Rubin carve (is_exclusive=True); got {tm.TOTAL_COLS}",
            ),
            (
                tm.S_OFF == 0 and tm.dP_OFF == tm.S_COLS and tm.dV_OFF == tm.dP_OFF + tm.dP_COLS,
                f"{flavor}: TMEM must be S | dP | dV back to back from column 0",
            ),
            (
                tm.S_COLS + tm.dP_COLS + tm.dV_COLS + (cfg.STAGES_TMEM_P * tm.P_COLS if cfg.IS_FP8 else tm.RSVD_COLS) == tm.TOTAL_COLS,
                f"{flavor}: TMEM column map must sum to {TMEM_TOTAL_COLS}: S {tm.S_COLS} + dP {tm.dP_COLS} + dV {tm.dV_COLS} + "
                f"{'P ring ' + str(cfg.STAGES_TMEM_P) + 'x' + str(tm.P_COLS) if cfg.IS_FP8 else 'RSVD ' + str(tm.RSVD_COLS)}",
            ),
            (tm.P_COLS == (cfg.TILE_N * cfg.BPE) // 4, f"{flavor}: the P A-operand is TILE_N*BPE/4 = {(cfg.TILE_N * cfg.BPE) // 4} TMEM cols; got {tm.P_COLS}"),
        ]
    )
    # --- swizzles: one unit with the descriptors that read them -----------------------------
    _check(
        [
            (
                cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.dO_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128,
                f"{flavor}: Q/K/dO/V swizzle must be 128 B (d=256 rows are 256 B fp8 / 512 B f16, and the bodies' subtile walks derive from 128)",
            ),
            (
                cfg.P_SWZ_BYTES == cfg.dS_SWZ_BYTES == 128,
                f"{flavor}: the dS ring's store_swizzled pattern (P_SWZ_BYTES) and its TMA-store descriptor swizzle (dS_SWZ_BYTES) are ONE unit and must both be 128 B; "
                f"bytes stored under one swizzle and read back under another give cos ~ 0.006 with no crash",
            ),
            (cfg.dV_SWZ_BYTES == 128, f"{flavor}: the dV staging store_swizzled and its TMA-store descriptor are 128 B"),
            ((cfg.TILE_N * cfg.BPE_DS) % cfg.dS_SWZ_BYTES == 0, f"{flavor}: a dS row (TILE_N*BPE_DS bytes) must be whole swizzle atoms"),
            ((cfg.TILE_O * cfg.BPE_O) % cfg.dV_SWZ_BYTES == 0, f"{flavor}: a dV row (TILE_O*BPE_O bytes) must be whole swizzle atoms"),
        ]
    )
    # --- masks ------------------------------------------------------------------------------
    _check(
        [
            (not (cfg.CAUSAL_BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)), f"{flavor}: bottom-right alignment requires a causal band (MASK_CAUSAL)"),
            (
                bool(cfg.MASK_FLAGS & MASK_SWA) == (cfg.SWA_WINDOW > 0),
                f"{flavor}: MASK_SWA <=> SWA_WINDOW > 0 (got MASK_FLAGS={cfg.MASK_FLAGS}, SWA_WINDOW={cfg.SWA_WINDOW})",
            ),
            (
                bool(cfg.MASK_FLAGS & MASK_PADDED) == bool(cfg.SEQ_KV_LENS_PRESENT),
                f"{flavor}: MASK_PADDED <=> SEQ_KV_LENS_PRESENT (got MASK_FLAGS={cfg.MASK_FLAGS}, SEQ_KV_LENS_PRESENT={cfg.SEQ_KV_LENS_PRESENT}); a padded mask without the "
                f"per-batch kv lengths masks against the padded total and every sequence attends the whole pad",
            ),
            (cfg.SCHEDULER_POLICY in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2), f"{flavor}: SCHEDULER_POLICY must be 0/1/2"),
        ]
    )
    # --- SMEM: the per-CTA cap, and the scaffold budget -----------------------------------------
    slabs = smem_layout(cfg)
    used = smem_bytes(cfg)
    if used > SMEM_USABLE_BYTES:
        tally = " | ".join(f"{s.name} {s.nbytes // 1024} KiB @{s.offset // 1024}" for s in slabs)
        raise ValueError(
            f"{flavor}: SMEM slabs {used // 1024} KiB + {SMEM_SCAFFOLD_BYTES // 1024} KiB scaffolding exceed the {SMEM_CAP_BYTES // 1024} KiB Rubin oversized "
            f"per-CTA cap ({tally}). Overflowing it does NOT fail cleanly -- the last slab declared clobbers the scaffolding / wraps -- so it must raise here"
        )
    declared = scaffold_bytes_declared(cfg)
    if declared > SMEM_SCAFFOLD_BYTES:
        raise ValueError(
            f"{flavor}: declared scaffolding ({declared} B of mbarriers + scheduler + tmem ptr) exceeds the {SMEM_SCAFFOLD_BYTES} B budget the SMEM cap check "
            f"reserves; raise SMEM_SCAFFOLD_BYTES (and re-check the slabs) rather than letting the tally lie"
        )
    for lbl, off in desc_roots(cfg):
        if not (0 <= off < used):
            raise ValueError(f"{flavor}: descriptor root {lbl} at {off} B lies outside the {used} B slab layout -- the declaration-order model is inconsistent")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_cfg_d256_bwd(params: _BwdTemplateParams, dtype_family: str) -> CfgBwdD256:
    """Build and validate the Cfg for one body.

    ``dtype_family`` names the BODY (``FAMILY_F16`` / ``FAMILY_FP8``), passed by
    the kernel template itself; a record whose ``dtype_qkv`` belongs to the
    other body raises, so a template loaded against the wrong family fails at
    load time instead of tracing the wrong dtype dispatch.
    """
    if dtype_family not in _FAMILIES:
        raise ValueError(f"sm107 bwd d256: dtype_family must be one of {_FAMILIES}; got {dtype_family!r}")
    flavor = _FLAVOR[dtype_family]
    _validate_params(flavor, dtype_family, params)
    is_fp8 = dtype_family == FAMILY_FP8
    dtype_o = getattr(params, "dtype_o", -1)
    if dtype_o < 0:
        dtype_o = DTYPE_E4M3 if is_fp8 else params.dtype_qkv
    dtype_ds = DTYPE_BF16 if is_fp8 else params.dtype_qkv
    mask_flags = _mask_flags_from(params)
    tile_k_hw = 64 if is_fp8 else 16
    # Register split.  The pool setmaxnreg redistributes is the LAUNCH allocation, 12 x 168 = 2016 (reg_entry_pool),
    # so the per-warp sum must not exceed it -- 8 x 232 + 4 x 48 = 2048 (the whole register file) HUNG the first
    # Rubin launch (2026-09-23): the four DEALLOCs release 4 x 120 = 480, the eight ALLOCs ask 8 x 64 = 512, the
    # last softmax warp parks forever.  The pre-port 8 x 232 + 4 x 40 = 2016 balanced exactly but the bf16 f16 builds
    # then spill ONE 4-byte slot in the 40-register MMA warp (the per-thread shared-storage base the sleeping
    # wait-retry loops reload: 1 STL / 8-11 LDL on sm_107a; the fp16 build fits).  f16 therefore moves 8 registers
    # from the softmax warps to the service warps: 8 x 224 + 4 x 56 = 2016, spill-free on both sides (SASS pin
    # test_sm107_register_split_spills_and_drains_sass_pins).  fp8 keeps the pre-port 232 / 40: its dense AND causal
    # sm_107a builds are spill-free (0 STL / 0 LDL, the same pin's fp8-dense / fp8-causal rows, 2026-09-24).
    softmax_regs = 232 if is_fp8 else 224
    service_regs = 40 if is_fp8 else 56
    cfg = CfgBwdD256(
        TILE_M=128,
        TILE_N=128,
        TILE_K=256,
        TILE_O=256,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=bpe(params.dtype_qkv),
        BPE_O=bpe(dtype_o),
        IS_FP8=int(is_fp8),
        DTYPE_DS=dtype_ds,
        BPE_DS=bpe(dtype_ds),
        CGA_M=2,
        CGA_N=1,
        CTA_MMA=2,
        Q_SWZ_BYTES=128,
        dO_SWZ_BYTES=128,
        K_SWZ_BYTES=128,
        V_SWZ_BYTES=128,
        P_SWZ_BYTES=128,
        dS_SWZ_BYTES=128,
        dV_SWZ_BYTES=128,
        TILE_K_HW_BMM1=tile_k_hw,
        TILE_K_HW_BMM2=tile_k_hw,
        IDESC_K_DIM=1 if is_fp8 else 0,
        K_SPLIT_UTCCP=0 if is_fp8 else 1,
        STAGES_Q=3 if is_fp8 else 2,
        STAGES_dO=3 if is_fp8 else 2,
        STAGES_dO_DV=3 if is_fp8 else 2,
        STAGES_KV=1,
        STAGES_TMEM_S=1,
        STAGES_TMEM_P=2 if is_fp8 else 1,
        XFER_STAGES=3 if is_fp8 else 1,
        STATS_STAGES=2,
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        SOFTMAX_WARPGROUPS=2,
        SOFTMAX_WG_WARPS=4,
        CORRECTION_WARPS=0,
        SOFTMAX_REGS=softmax_regs,
        CORRECTION_REGS=0,
        MMA_REGS=service_regs,
        TMALDG_REGS=service_regs,
        TMASTG_REGS=service_regs,
        SCHEDULER_REGS=service_regs,
        OTHER_REGS=service_regs,
        MASK_FLAGS=mask_flags,
        SWA_WINDOW=params.window_left or 0,
        CAUSAL_BOTTOM_RIGHT=int(params.bottom_right),
        HAS_SINK=int(getattr(params, "has_sink", False)),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        L2_SIZE_MIB=60,
        SCHEDULER_POLICY=params.sched_policy,
        TOTAL_WARPS=12,
        THREADS_PER_CTA=12 * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=4,
        MMA_WARP_ID=8,
        TMALDG_WARP_ID=9,
        TMASTG_WARP_ID=10,
        SCHED_WARP_ID=11,
        ONE_LANE=1,
        ONE_WARP=32,
        SOFTMAX_WG_LANES=4 * 32,
        SOFTMAX_LANES=2 * 4 * 32,
        SOFT_X_CTA_MMA=2 * 4 * 32 * 2,
        MMA_COMMIT_ARRIVES=1,
        READ_TILE_ARRIVERS_TOT=2 * (2 * 4 + 2) + 1,
        N_BMM2_CHUNKS=128 // 64,
        BMM2_CHUNK_SIZE=64,
    )
    _validate_cfg_d256_bwd(cfg, flavor)
    return cfg


# ---------------------------------------------------------------------------
# Shape-time helpers for the adapter / the kernel's compile()
# ---------------------------------------------------------------------------


def validate_head_chunk(h_q: int, h_kv: int, qh_chunk: int) -> None:
    """The dS workspace is head-chunked (``qh_chunk`` heads per launch, runtime
    ``head_base``).  A chunk must be a divisor of ``H_q`` AND whole GQA groups
    (a multiple of ``H_q // H_kv``), or the per-Q-head dK/dV partial fold over a
    chunk is incomplete and a KV head's gradient is summed from a partial
    group."""
    if h_kv < 1 or h_q < 1 or h_q % h_kv != 0:
        raise ValueError(f"sm107 bwd d256: H_q ({h_q}) must be a positive multiple of H_kv ({h_kv})")
    group = h_q // h_kv
    if qh_chunk < 1 or h_q % qh_chunk != 0:
        raise ValueError(f"sm107 bwd d256: qh_chunk ({qh_chunk}) must be a positive divisor of H_q ({h_q})")
    if qh_chunk % group != 0:
        raise ValueError(
            f"sm107 bwd d256: qh_chunk ({qh_chunk}) must be a multiple of the GQA group H_q/H_kv = {group}, or the dK/dV fold over a chunk is partial"
        )


def q_pad_rows(cfg: CfgBwdD256) -> int:
    """S_q must be padded to the q-tile (the inner loop walks whole q-tiles)."""
    return cfg.TILE_N


def kv_pad_rows(cfg: CfgBwdD256) -> int:
    """S_kv must be padded to the cga2 pair's kv block (TILE_M * CTA_MMA)."""
    return cfg.TILE_M * cfg.CTA_MMA


def ds_workspace_bytes(cfg: CfgBwdD256, batch: int, qh_chunk: int, s_q_pad: int, s_kv_pad: int) -> int:
    """Bytes of the ``[batch, qh_chunk, S_kv, S_q]`` dS workspace one launch writes
    (kv-major: the layout that lets dK = dS.Q read it un-permuted)."""
    if s_q_pad % q_pad_rows(cfg) or s_kv_pad % kv_pad_rows(cfg):
        raise ValueError(
            f"sm107 bwd d256: workspace extents must be padded to q {q_pad_rows(cfg)} / kv {kv_pad_rows(cfg)} rows; got S_q={s_q_pad}, S_kv={s_kv_pad}"
        )
    return batch * qh_chunk * s_kv_pad * s_q_pad * cfg.BPE_DS


def launch_grid(cfg: CfgBwdD256, batch: int, qh_chunk: int, s_kv_pad: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """``(grid, cluster)`` for one launch: natural = ``(kv_blocks * CGA_M, qh_chunk, B)``;
    LPT / LPT_L2 = the flat 1-D grid the body's tile decode expects."""
    kv_blocks = -(-s_kv_pad // kv_pad_rows(cfg))
    if cfg.SCHEDULER_POLICY == SCHED_NATURAL:
        grid = (kv_blocks * cfg.CGA_M, qh_chunk, batch)
    else:
        grid = (kv_blocks * qh_chunk * batch * cfg.CGA_M, 1, 1)
    return grid, (cfg.CGA_M, cfg.CGA_N, 1)
