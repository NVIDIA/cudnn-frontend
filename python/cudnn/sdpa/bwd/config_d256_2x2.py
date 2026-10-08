# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the FROST d=256 bf16/fp16 SDPA-backward body on the **2x2 datapath**
(``kernels/bprop_d256_2x2_f16.py``: ``tcgen05.mma.cta_group::2`` with the collective M = 128, 64 kv rows per CTA
per sub-block).

One body, two PROFILES, selected by the appended :class:`cudnn.sdpa.bwd.config_sm107.TemplateParams` field
``datapath_2x2_profile`` (0 = the shipped 4x1 bodies, which never call this module):

* ``PROFILE_SM100`` (1): ONE 64-row sub-block per CTA -> a 128-row kv block per cga2 pair; 2-deep S / dP TMEM parity
  rings and a 2-deep SMEM P ring; 210 KiB of slabs under the SM100 227 KiB cap; descriptor version 0.  The SM100
  (cc 10.0-10.6) ``sdpa_bwd_sm100_d256`` row, and the bit-identical Rubin bring-up configuration.
* ``PROFILE_SM107_INTERLEAVED`` (2): TWO 64-row sub-blocks per CTA (its 128 contiguous rows of a 256-row kv block),
  warpgroup g owns sub-block g, so Q / dO / dO_dv are loaded ONCE per 256-row block per q tile (today's operand reuse);
  1-deep S / dP / P rings per sub-block, 322 KiB of slabs under the Rubin 327 KiB cap; descriptor version 1 (the P ring
  sits at 256 KiB).  The Rubin twin behind ``api_dsl_sm107.BWD_D256_2X2``.

Why a separate module from :mod:`cudnn.sdpa.bwd.config_sm107`: that module's ``smem_layout`` / ``desc_version`` /
``tmem_layout`` describe the 4x1 slab table (the K-split UTCCP alias, the TMEM P alias, 576 exclusive columns); the
2x2 body has none of those and ITS roots decide ITS descriptor version.  The record vocabulary is shared --
:class:`CfgBwdD256x2` appends to :class:`cudnn.sdpa.bwd.config_sm107.CfgBwdD256` -- so a kernel port references
``CFG.<name>`` with the same spellings, and every 4x1 helper (``make_cfg_d256_bwd`` and its validator) is untouched.

Geometry vocabulary (section 4 of the design): ``TILE_M`` = 64 kv rows per CTA **per sub-block** (= the MMA's
``m_per_cta``, so ``TILE_M * CTA_MMA`` is still the collective M as on the 4x1 record), ``ROWS_PER_CTA = TILE_M *
KV_SUBBLOCKS`` (the contiguous rows one CTA owns), ``KV_BLOCK_ROWS = ROWS_PER_CTA * CTA_MMA`` (the pair's block, the
scheduler's unit), ``STAGE3_GRAN_ROWS`` = 256 (the kv WRITE pair the stage-3 GEMMs trim at: both 128-row blocks of a
pair walk the identical q range, so the (256, 256) renderings and the no-zero-fill contract are unchanged).

The 2x2 accumulator atom (``probe_layout.log``): a 64 x N fp32 D lands as row m -> TMEM lane ``m % 64 + 64 * (n //
(N/2))``, column ``n % (N/2)``: S / dP (N = 128) take 64 columns, dV (N = 256) 128.  Compute warp ``w``: lane quadrant
``qd = w & 3`` -> kv row ``32 * (qd & 1) + t`` of q-column half ``qd >> 1``; warpgroup ``g = w >> 2`` -> sub-block
``g // SUBBLOCK_WGS`` and column sub-chunk ``g % SUBBLOCK_WGS`` of ``COLS_PER_LANE`` columns.

Every value a kernel constant or an mbarrier init count depends on is derived HERE as a formula and pinned by
``test/python/sdpa/frost/test_sdpa_bwd_config_d256_2x2.py``.  Configuration errors raise :class:`ValueError`; anything
a user-built graph could trip is rejected earlier by the engine row's ``Capabilities`` / the adapter's
``check_support``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16, MASK_CAUSAL, MASK_PADDED, MASK_SWA, SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from cudnn.sdpa.bwd import config_sm107 as _c
from cudnn.sdpa.bwd.config_sm100 import TemplateParams as _BwdTemplateParams
from cudnn.sdpa.bwd.config_sm107 import FAMILY_F16, CfgBwdD256, SmemSlab, TemplateParams, bpe, reg_entry_pool, validate_head_chunk

__all__ = [
    "PROFILE_OFF",
    "PROFILE_SM100",
    "PROFILE_SM107_INTERLEAVED",
    "PROFILES",
    "CfgBwdD256x2",
    "BufferElems2x2",
    "TmemLayout2x2",
    "FAMILY_F16",
    "TemplateParams",
    "MMA_M",
    "SUB_ROWS",
    "TMEM_ALLOC_COLS",
    "SMEM_CAP_BYTES_SM100",
    "SMEM_CAP_BYTES_SM107",
    "SMEM_SCAFFOLD_BYTES",
    "make_cfg_d256_2x2",
    "buffer_elems_2x2",
    "tmem_layout_2x2",
    "smem_layout_2x2",
    "smem_bytes_2x2",
    "kernel_smem_bytes_2x2",
    "desc_roots_2x2",
    "desc_version_2x2",
    "mbar_stage_counts_2x2",
    "mbar_init_counts_2x2",
    "scaffold_bytes_declared_2x2",
    "read_tile_arrivers_tot_2x2",
    "validate_head_chunk",
    "q_pad_rows_2x2",
    "kv_pad_rows_2x2",
    "q_write_tiles_2x2",
    "ds_workspace_bytes_2x2",
    "launch_grid_2x2",
]

# --- the profile axis (TemplateParams.datapath_2x2_profile) -------------------------------------------------------
PROFILE_OFF = 0  # the shipped 4x1 bodies; make_cfg_d256_2x2 REFUSES it
PROFILE_SM100 = 1  # one 64-row sub-block per CTA, 128-row kv block, 227 KiB; SM100 and the Rubin bring-up
PROFILE_SM107_INTERLEAVED = 2  # two sub-blocks per CTA, 256-row kv block, 322 KiB, desc v1; the Rubin twin
PROFILES = (PROFILE_SM100, PROFILE_SM107_INTERLEAVED)
_PROFILE_NAME = {PROFILE_SM100: "sm100 (one 64-row sub-block per CTA)", PROFILE_SM107_INTERLEAVED: "sm107 interleaved (two sub-blocks per CTA)"}

# --- 2x2 datapath constants ---------------------------------------------------------------------------------------
# The collective M of every tcgen05.mma.cta_group::2 on this body (idesc m_dim); MmaDesc(M=MMA_M, cta_group=2) ->
# m_per_cta = SUB_ROWS = 64 kv rows per CTA per sub-block (the 2x2 D atom spans all 128 TMEM lanes).
MMA_M = 128
SUB_ROWS = MMA_M // 2
# One tcgen05.alloc.cta_group::2 of 512 columns per CTA on BOTH arches (a power of two in [32, 512]; 576 / is_exclusive
# is the 4x1 Rubin bodies' and is never exercised here).
TMEM_ALLOC_COLS = 512
# Per-CTA dynamic SMEM caps: SM100 opt-in 227 KiB; Rubin oversized 327 KiB (config_sm107.SMEM_CAP_BYTES).
SMEM_CAP_BYTES_SM100 = 227 * 1024
SMEM_CAP_BYTES_SM107 = _c.SMEM_CAP_BYTES
SMEM_SCAFFOLD_BYTES = _c.SMEM_SCAFFOLD_BYTES
# fp32 accumulator columns per 64-row sub-block tile: S / dP [64 kv x 128 q] = 64, dV [64 kv x 256 d_v] = 128.
_S_COLS = 64
_DV_COLS = 128

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_FLAVOR = "bwd d256 2x2 f16"


# ---------------------------------------------------------------------------
# The Cfg record: the 4x1 vocabulary plus the 2x2 axis (append-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgBwdD256x2(CfgBwdD256):
    """:class:`CfgBwdD256` plus the 2x2 fields.  Inherited fields keep their 4x1 meaning where it applies (dtypes,
    masks, warp roster, register split, mbarrier lane constants); ``TILE_M`` is 64 = kv rows per CTA PER SUB-BLOCK
    (``TILE_M * CTA_MMA`` = the collective MMA M, as before), ``STAGES_SMEM_P`` is the SMEM P ring depth (declared HERE:
    the 4x1 configs carry no SMEM P ring any more), ``STAGES_TMEM_P`` / ``K_SPLIT_UTCCP`` are 0 (no TMEM P, no UTCCP).
    """

    # Which profile this record was built for (1 / 2); 0 never reaches a built record.
    DATAPATH_2X2_PROFILE: int = PROFILE_OFF
    # The SMEM P ring depth, sub-block indexed in the kernel (``N_P_BARS = KV_SUBBLOCKS * STAGES_SMEM_P``): 2 on profile 1,
    # 1 on profile 2.  Owned by this record -- the 4x1 ``CfgBwdD256`` has no SMEM P ring (its MXFP8 body keeps P in TMEM).
    STAGES_SMEM_P: int = 0
    # 64-row sub-blocks per CTA (1 / 2) and the derived row counts.
    KV_SUBBLOCKS: int = 1
    ROWS_PER_CTA: int = SUB_ROWS  # TILE_M * KV_SUBBLOCKS: the contiguous kv rows one CTA owns (its TMA store boxes)
    KV_BLOCK_ROWS: int = 2 * SUB_ROWS  # ROWS_PER_CTA * CTA_MMA: the pair's kv block = the scheduler's unit
    MMA_M: int = MMA_M  # the collective M of every MMA (idesc m_dim); == TILE_M * CTA_MMA
    # The kv WRITE pair: stage 2 rounds every kv block's q range to the range of the STAGE3_GRAN_ROWS-row pair it belongs
    # to (so both 128-row blocks of a pair walk the identical q tiles), the stage-3 GEMMs trim at this granularity
    # (``causal_gran``) and the adapter pads S_kv to it.
    STAGE3_GRAN_ROWS: int = 256
    # Compute-warp ownership: warpgroups per sub-block and q columns per lane (SOFTMAX_WARPGROUPS // KV_SUBBLOCKS, 64 // that).
    SUBBLOCK_WGS: int = 2
    COLS_PER_LANE: int = 32
    # LEADER-scope init count of the per-sub-block fan-ins (s_acc_empty / dp_empty / p_full): every compute lane of
    # THAT sub-block on BOTH CTAs = (SOFTMAX_LANES // KV_SUBBLOCKS) * CTA_MMA.
    L_CNT: int = 512
    TMEM_ALLOC_COLS: int = TMEM_ALLOC_COLS
    # Per-CTA SMEM cap the profile is validated against (227 KiB / 327 KiB).
    SMEM_CAP: int = SMEM_CAP_BYTES_SM100
    # 0 = the NATURAL MMA order (S(i), dP(i), BMM2(i) per q tile); 1 = the lookahead order (S(i+1) between dP(i) and
    # BMM2(i), the fp8 twin's form).  The body binds its module constant MMA_LOOKAHEAD from this field.  Profile 1
    # ships NATURAL: measured on B200 (2026-10-01, B=1 H_q=32 H_kv=2 S=8192 bf16, whole backward, clean CUPTI
    # medians) stage 2 is 3781 us NATURAL vs 4525 us lookahead dense (-16%) and 2020 vs 1974 us causal (+2%); the
    # lookahead's S(i+1) sits in the MMA queue ahead of BMM2(i) and the dV accumulate waits on it every tile.  Profile 2
    # keeps the design's lookahead (unmeasured here: no Rubin board on this box) -- the A/B is the Rubin lane's.
    MMA_LOOKAHEAD: int = 0
    # TMEM columns between sub-block 0's and sub-block 1's regions (profile 2); 0 with one sub-block.
    SUBBLOCK_STRIDE_COLS: int = 0


# ---------------------------------------------------------------------------
# Derived element counts / TMA geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferElems2x2:
    """The body's module-level buffer / TMA / descriptor constants, by name (``_B = buffer_elems_2x2(CFG)``)."""

    _M_PER_CTA: int  # q rows per CTA of the Q / dO N-split (TILE_N // CTA_MMA = 64)
    qBufferElems: int  # one Q ring stage: _M_PER_CTA x TILE_K
    dOBufferElems: int  # one dO ring stage (dP view): _M_PER_CTA x TILE_O
    dOdvBufferElems: int  # one dO_dv ring stage (dV view, BT): TILE_N x (TILE_O // CTA_MMA)
    kSubElems: int  # one K sub-block slab: TILE_M x TILE_K (64 x 256)
    kBufferElems: int  # KV_SUBBLOCKS x kSubElems
    vSubElems: int
    vBufferElems: int
    pSlabElems: int  # one P slab (one sub-block, one ring stage): TILE_M x TILE_N (64 x 128)
    dSBufferElems: int  # one dS ring stage: ROWS_PER_CTA x TILE_N
    dVBufferElems: int  # dV staging (aliases sK): ROWS_PER_CTA x TILE_O
    STATS_SLOT_ELEMS: int
    STATS_LSE_OFF: int
    STATS_DOT_OFF: int
    _KV_BLOCK_ROWS: int
    _KV_WRITE_ROWS: int
    _Q_WRITE_TILES: int
    CGA_SIZE: int
    # TMA subtile walks (128-B swizzle granule = 64 elems at BPE 2)
    TMA_QK_ITERS: int  # Q / K: 4 subtiles of 64 d
    TMA_VO_ITERS: int  # dO / V: 4
    TMA_QK_GRANU_ELEMS: int
    TMA_VO_GRANU_ELEMS: int
    TMA_VO_SG1_ITERS: int  # dO_dv (BT view): (TILE_O // CTA_MMA) / 64 = 2 subtiles of 64 d_v
    TMA_VO_SG1_GRANU_ELEMS: int
    DV_D_BLOCK: int  # 64 d_v cols per dV store subtile (128 B)
    TMA_DV_ITERS: int  # 4
    DV_BLOCK_SLAB: int  # ROWS_PER_CTA x DV_D_BLOCK elems per dV store subtile
    P_TMA_ITERS: int  # dS store subtiles per 128-q row: TILE_N * BPE_DS / 128 = 2
    P_D_BLOCK: int  # 64 q cols per dS store subtile
    DS_BLOCK_SLAB: int  # ROWS_PER_CTA x P_D_BLOCK elems per dS store subtile
    P_SUB_SLAB: int  # TILE_M x 64 elems: one 64-q-column subtile of a P slab (8 KiB)
    # expect_tx bytes (cga2 tensor TMAs deliver BOTH peers' bytes to the leader's mbar -> x CTA_MMA)
    qTmaTransactionBytes: int
    dOTmaTransactionBytes: int
    dOdvTmaTransactionBytes: int
    kTmaTransactionBytes: int
    vTmaTransactionBytes: int
    # tcgen05 SMEM descriptor constants
    LEADING_BYTE_OFFSET_QK: int
    STRIDE_BYTE_OFFSET_QK: int
    LEADING_BYTE_OFFSET_dO_SG1: int  # BT B operand: (TILE_O // CTA_MMA) // 8 > 8 -> leading = K x swizzle = TILE_N * 128
    STRIDE_BYTE_OFFSET_dO_SG1: int
    LEADING_BYTE_OFFSET_P: int
    STRIDE_BYTE_OFFSET_P: int
    SMEM_LAYOUT_SW128: int  # SmemTile.layout code of the 128-B swizzle (every slab on this body)
    P_STORE_ALIGN: int  # store_swizzled alignment of a lane's P / dS segment: COLS_PER_LANE * BPE (64 or 128)


def buffer_elems_2x2(cfg: CfgBwdD256x2) -> BufferElems2x2:
    m_per_cta = cfg.TILE_N // cfg.CTA_MMA
    granu = 128 // cfg.BPE  # 64 elems per 128-B swizzle atom
    q_elems = m_per_cta * cfg.TILE_K
    do_elems = m_per_cta * cfg.TILE_O
    dodv_elems = cfg.TILE_N * (cfg.TILE_O // cfg.CTA_MMA)
    k_sub = cfg.TILE_M * cfg.TILE_K
    v_sub = cfg.TILE_M * cfg.TILE_O
    dv_d_block = cfg.dV_SWZ_BYTES // cfg.BPE_O
    p_tma_iters = (cfg.TILE_N * cfg.BPE_DS) // cfg.dS_SWZ_BYTES
    p_d_block = cfg.TILE_N // p_tma_iters
    return BufferElems2x2(
        _M_PER_CTA=m_per_cta,
        qBufferElems=q_elems,
        dOBufferElems=do_elems,
        dOdvBufferElems=dodv_elems,
        kSubElems=k_sub,
        kBufferElems=cfg.KV_SUBBLOCKS * k_sub,
        vSubElems=v_sub,
        vBufferElems=cfg.KV_SUBBLOCKS * v_sub,
        pSlabElems=cfg.TILE_M * cfg.TILE_N,
        dSBufferElems=cfg.ROWS_PER_CTA * cfg.TILE_N,
        dVBufferElems=cfg.ROWS_PER_CTA * cfg.TILE_O,
        STATS_SLOT_ELEMS=2 * cfg.TILE_N,
        STATS_LSE_OFF=0,
        STATS_DOT_OFF=cfg.TILE_N,
        _KV_BLOCK_ROWS=cfg.KV_BLOCK_ROWS,
        _KV_WRITE_ROWS=cfg.STAGE3_GRAN_ROWS,
        _Q_WRITE_TILES=cfg.STAGE3_GRAN_ROWS // cfg.TILE_N,
        CGA_SIZE=cfg.CGA_M * cfg.CGA_N,
        TMA_QK_ITERS=(cfg.TILE_K * cfg.BPE) // cfg.Q_SWZ_BYTES,
        TMA_VO_ITERS=(cfg.TILE_O * cfg.BPE) // cfg.dO_SWZ_BYTES,
        TMA_QK_GRANU_ELEMS=granu,
        TMA_VO_GRANU_ELEMS=granu,
        TMA_VO_SG1_ITERS=((cfg.TILE_O // cfg.CTA_MMA) * cfg.BPE) // cfg.dO_SWZ_BYTES,
        TMA_VO_SG1_GRANU_ELEMS=granu,
        DV_D_BLOCK=dv_d_block,
        TMA_DV_ITERS=(cfg.TILE_O * cfg.BPE_O) // cfg.dV_SWZ_BYTES,
        DV_BLOCK_SLAB=cfg.ROWS_PER_CTA * dv_d_block,
        P_TMA_ITERS=p_tma_iters,
        P_D_BLOCK=p_d_block,
        DS_BLOCK_SLAB=cfg.ROWS_PER_CTA * p_d_block,
        P_SUB_SLAB=cfg.TILE_M * granu,
        qTmaTransactionBytes=q_elems * cfg.BPE * cfg.CTA_MMA,
        dOTmaTransactionBytes=do_elems * cfg.BPE * cfg.CTA_MMA,
        dOdvTmaTransactionBytes=dodv_elems * cfg.BPE * cfg.CTA_MMA,
        kTmaTransactionBytes=cfg.KV_SUBBLOCKS * k_sub * cfg.BPE * cfg.CTA_MMA,
        vTmaTransactionBytes=cfg.KV_SUBBLOCKS * v_sub * cfg.BPE * cfg.CTA_MMA,
        LEADING_BYTE_OFFSET_QK=0,
        STRIDE_BYTE_OFFSET_QK=8 * cfg.Q_SWZ_BYTES,
        LEADING_BYTE_OFFSET_dO_SG1=cfg.TILE_N * cfg.dO_SWZ_BYTES,
        STRIDE_BYTE_OFFSET_dO_SG1=8 * cfg.dO_SWZ_BYTES,
        LEADING_BYTE_OFFSET_P=0,
        STRIDE_BYTE_OFFSET_P=8 * cfg.P_SWZ_BYTES,
        SMEM_LAYOUT_SW128=_SWZ_ENUM[128],
        P_STORE_ALIGN=cfg.COLS_PER_LANE * cfg.BPE,
    )


# ---------------------------------------------------------------------------
# TMEM column map (one 512-column non-exclusive alloc per CTA)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TmemLayout2x2:
    """Column map.  Sub-block ``s`` owns ``[s * SUB_STRIDE, +SUB_USED)``: S slots at ``S_BASE + slot * S_COLS``, dP slots
    at ``dP_BASE + slot * S_COLS``, dV at ``dV_BASE`` (``DV_COLS`` wide).  Lane rule (the 2x2 D atom): row m -> lane
    ``m + 64 * (n // (N/2))``, column ``n % (N/2)``."""

    TOTAL_COLS: int
    S_COLS: int  # 64 per S / dP slot
    DV_COLS: int  # 128
    STAGES_TMEM_S: int
    S_BASE: int
    dP_BASE: int
    dV_BASE: int
    SUB_USED: int  # columns one sub-block owns
    SUB_STRIDE: int  # offset of sub-block 1 (0 with one sub-block)
    USED_COLS: int
    FREE_COLS: int


def tmem_layout_2x2(cfg: CfgBwdD256x2) -> TmemLayout2x2:
    s_base = 0
    dp_base = cfg.STAGES_TMEM_S * _S_COLS
    dv_base = 2 * cfg.STAGES_TMEM_S * _S_COLS
    sub_used = dv_base + _DV_COLS
    sub_stride = sub_used if cfg.KV_SUBBLOCKS > 1 else 0
    used = cfg.KV_SUBBLOCKS * sub_used
    return TmemLayout2x2(
        TOTAL_COLS=cfg.TMEM_ALLOC_COLS,
        S_COLS=_S_COLS,
        DV_COLS=_DV_COLS,
        STAGES_TMEM_S=cfg.STAGES_TMEM_S,
        S_BASE=s_base,
        dP_BASE=dp_base,
        dV_BASE=dv_base,
        SUB_USED=sub_used,
        SUB_STRIDE=sub_stride,
        USED_COLS=used,
        FREE_COLS=cfg.TMEM_ALLOC_COLS - used,
    )


# ---------------------------------------------------------------------------
# SMEM slab table in DECLARATION ORDER (the kernel declares its arrays in exactly this order)
# ---------------------------------------------------------------------------


def _align_slab(nbytes: int) -> int:
    return (nbytes + 1023) // 1024 * 1024


def smem_layout_2x2(cfg: CfgBwdD256x2) -> Tuple[SmemSlab, ...]:
    """``sQ | sdO | sdOdv | sK | sV | sP | sStats | sdS`` with every tcgen05 ``build()`` root.

    Profile 1: 32 + 32 + 32 + 32 + 32 + 2 x 16 + 2 + 16 = 210 KiB (largest root sP[1] at 176 KiB -> v0).
    Profile 2: 64 + 32 + 32 + 64 + 64 + 2 x 16 + 2 + 32 = 322 KiB (sP at 256 KiB -> v1).
    sdV (the dV staging) ALIASES sK post-loop (``dVBufferElems * BPE_O == kBufferElems * BPE``); sStats / sdS carry no
    descriptor root.
    """
    b = buffer_elems_2x2(cfg)
    slabs = []
    off = 0

    def _push(name, nbytes, roots=()):
        nonlocal off
        slab = SmemSlab(name, off, _align_slab(nbytes), tuple((lbl, off + r) for lbl, r in roots))
        slabs.append(slab)
        off += slab.nbytes

    q_stage = b.qBufferElems * cfg.BPE
    do_stage = b.dOBufferElems * cfg.BPE
    dodv_stage = b.dOdvBufferElems * cfg.BPE
    k_sub = b.kSubElems * cfg.BPE
    v_sub = b.vSubElems * cfg.BPE
    p_slab = b.pSlabElems * cfg.BPE
    _push("sQ", cfg.STAGES_Q * q_stage, [(f"sQ[{s}]", s * q_stage) for s in range(cfg.STAGES_Q)])
    _push("sdO", cfg.STAGES_dO * do_stage, [(f"sdO[{s}]", s * do_stage) for s in range(cfg.STAGES_dO)])
    _push("sdOdv", cfg.STAGES_dO_DV * dodv_stage, [(f"sdO_dv[{s}]", s * dodv_stage) for s in range(cfg.STAGES_dO_DV)])
    _push("sK(+sdV alias)", cfg.KV_SUBBLOCKS * k_sub, [(f"sK[{s}]", s * k_sub) for s in range(cfg.KV_SUBBLOCKS)])
    _push("sV", cfg.KV_SUBBLOCKS * v_sub, [(f"sV[{s}]", s * v_sub) for s in range(cfg.KV_SUBBLOCKS)])
    _push(
        "sP",
        cfg.KV_SUBBLOCKS * cfg.STAGES_SMEM_P * p_slab,
        [(f"sP[{s}][{p}]", (s * cfg.STAGES_SMEM_P + p) * p_slab) for s in range(cfg.KV_SUBBLOCKS) for p in range(cfg.STAGES_SMEM_P)],
    )
    _push("sStats", cfg.STATS_STAGES * b.STATS_SLOT_ELEMS * 4)
    _push("sdS", cfg.XFER_STAGES * b.dSBufferElems * cfg.BPE_DS)
    return tuple(slabs)


def smem_bytes_2x2(cfg: CfgBwdD256x2) -> int:
    return sum(s.nbytes for s in smem_layout_2x2(cfg))


def kernel_smem_bytes_2x2(cfg: CfgBwdD256x2) -> int:
    return smem_bytes_2x2(cfg) + SMEM_SCAFFOLD_BYTES


def desc_roots_2x2(cfg: CfgBwdD256x2) -> Tuple[Tuple[str, int], ...]:
    return tuple(r for s in smem_layout_2x2(cfg) for r in s.roots)


def desc_version_2x2(cfg: CfgBwdD256x2) -> int:
    """The ``desc_version=`` every ``SmemTile`` in the body must take (bound ONCE as its module constant
    ``DESC_VERSION``): 1 iff any descriptor root sits at or past the 14-bit version-0 window (256 KiB)."""
    return 1 if any(off >= _c.TCGEN05_V0_ADDR_LIMIT for _, off in desc_roots_2x2(cfg)) else 0


# ---------------------------------------------------------------------------
# Scaffolding: the mbarrier ledger
# ---------------------------------------------------------------------------


def mbar_stage_counts_2x2(cfg: CfgBwdD256x2) -> Dict[str, int]:
    """``Bars`` field -> stage count, in the body's order.  Sub-block-indexed rings are ``KV_SUBBLOCKS x depth`` deep
    (slot ``s * depth + idx``).  REMOVED vs the 4x1 f16 body: ``mb_k_utccp_done`` (no UTCCP), ``mb_p_ready`` (the TMEM P;
    replaced by the SMEM ring's ``mb_p_full``).  ADDED: ``mb_s_acc_empty`` (the fp8 twin's), ``mb_p_empty``."""
    n_s = cfg.KV_SUBBLOCKS * cfg.STAGES_TMEM_S
    n_p = cfg.KV_SUBBLOCKS * cfg.STAGES_SMEM_P
    return {
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
        "mb_s_full": n_s,
        "mb_s_acc_empty": n_s,
        "mb_dp_full": n_s,
        "mb_dp_empty": n_s,
        "mb_p_full": n_p,
        "mb_p_empty": n_p,
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


def mbar_init_counts_2x2(cfg: CfgBwdD256x2) -> Dict[str, int]:
    """``Bars`` field -> per-CTA init count = the EXACT arrival sum per phase (P3; the body's ledger names the sites):

    * TMA_LOAD ``_full``: 1 (the leader's elected lane's ``expect_tx``; the follower's copy is never armed).
    * MMA_COMMIT: 1 per target CTA (one predicated ``tcgen05.commit`` multicast).
    * LEADER fan-ins of a sub-block (``s_acc_empty`` / ``dp_empty`` / ``p_full``): ``L_CNT`` = that sub-block's compute
      lanes on both CTAs; ``dv_acc_empty`` / ``tmem_dealloc``: every compute lane of both CTAs (``SOFT_X_CTA_MMA``).
    * THREAD: ``stats_full`` one warp (32), ``stats_empty`` / ``ds_smem_full`` / ``dv_stg_full`` every local compute lane
      (256), ``ds_smem_empty`` / ``dv_stg_empty`` the TMA-STG's elected lane (1).
    """
    one, commit, lanes = cfg.ONE_LANE, cfg.MMA_COMMIT_ARRIVES, cfg.SOFTMAX_LANES
    return {
        "mb_q_full": one,
        "mb_q_empty": commit,
        "mb_do_full": one,
        "mb_do_empty": commit,
        "mb_dodv_full": one,
        "mb_dodv_empty": commit,
        "mb_k_full": one,
        "mb_k_empty": commit,
        "mb_v_full": one,
        "mb_v_empty": commit,
        "mb_s_full": commit,
        "mb_s_acc_empty": cfg.L_CNT,
        "mb_dp_full": commit,
        "mb_dp_empty": cfg.L_CNT,
        "mb_p_full": cfg.L_CNT,
        "mb_p_empty": commit,
        "mb_stats_full": cfg.ONE_WARP,
        "mb_stats_empty": lanes,
        "mb_ds_smem_full": lanes,
        "mb_ds_smem_empty": one,
        "mb_dv_ready": commit,
        "mb_dv_acc_empty": cfg.SOFT_X_CTA_MMA,
        "mb_dv_stg_full": lanes,
        "mb_dv_stg_empty": one,
        "mb_tmem_dealloc": cfg.SOFT_X_CTA_MMA,
    }


_MBAR_ARRAY_ALIGN = 16
_SCHED_RESPONSE_WORDS = 8
_TMEM_PTR_BYTES = 16


def scaffold_bytes_declared_2x2(cfg: CfgBwdD256x2) -> int:
    def _arr(n_qwords):
        return (8 * n_qwords + _MBAR_ARRAY_ALIGN - 1) // _MBAR_ARRAY_ALIGN * _MBAR_ARRAY_ALIGN

    mbars = sum(_arr(n) for n in mbar_stage_counts_2x2(cfg).values())
    sched = 2 * _arr(cfg.SCHEDULER_STAGES) + cfg.SCHEDULER_STAGES * _SCHED_RESPONSE_WORDS * 4
    return mbars + sched + _TMEM_PTR_BYTES


def read_tile_arrivers_tot_2x2(cfg: CfgBwdD256x2) -> int:
    """Per-CTA init count of ``mb_read_tile_id``: the warps calling ``read_tile_id_arrive`` cluster-wide = the 8 compute
    warps + TMA-LDG + TMA-STG on every CTA + the MMA warp on the pair leader only (the census is the 4x1 body's)."""
    compute = cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
    return cfg.CTA_MMA * (compute + 2) + 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check(preds) -> None:
    for ok, msg in preds:
        if not ok:
            raise ValueError(msg)


def _validate_cfg_d256_2x2(cfg: CfgBwdD256x2) -> None:
    f = _FLAVOR
    b = buffer_elems_2x2(cfg)
    tm = tmem_layout_2x2(cfg)
    compute_warps = cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_WG_WARPS
    reg_total = compute_warps * cfg.SOFTMAX_REGS + cfg.MMA_REGS + cfg.TMALDG_REGS + cfg.TMASTG_REGS + cfg.SCHEDULER_REGS
    # --- the profile axis and the row geometry it decides -----------------------------------------------------------
    _check(
        [
            (cfg.DATAPATH_2X2_PROFILE in PROFILES, f"{f}: DATAPATH_2X2_PROFILE must be 1 (sm100) or 2 (sm107 interleaved); got {cfg.DATAPATH_2X2_PROFILE}"),
            (
                cfg.KV_SUBBLOCKS == cfg.DATAPATH_2X2_PROFILE,
                f"{f}: KV_SUBBLOCKS is the profile number (1 sub-block on profile 1, 2 on profile 2); got {cfg.KV_SUBBLOCKS}",
            ),
            (
                cfg.TILE_M == SUB_ROWS and cfg.MMA_M == MMA_M and cfg.TILE_M * cfg.CTA_MMA == cfg.MMA_M,
                f"{f}: TILE_M = 64 kv rows per CTA per sub-block, MMA_M = TILE_M * CTA_MMA = 128 (the 2x2 collective M)",
            ),
            (
                cfg.TILE_N == 128 and cfg.TILE_K == 256 and cfg.TILE_O == 256,
                f"{f}: TILE_N = 128 q cols, d_qk = d_v = 256 exactly; got {cfg.TILE_N}/{cfg.TILE_K}/{cfg.TILE_O}",
            ),
            (
                cfg.CGA_M == 2 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2,
                f"{f}: one cga2 pair (CGA_M=2, CGA_N=1, CTA_MMA=2); the 4-CTA multicast arm is a later lane",
            ),
            (cfg.ROWS_PER_CTA == cfg.TILE_M * cfg.KV_SUBBLOCKS, f"{f}: ROWS_PER_CTA must be TILE_M * KV_SUBBLOCKS; got {cfg.ROWS_PER_CTA}"),
            (cfg.KV_BLOCK_ROWS == cfg.ROWS_PER_CTA * cfg.CTA_MMA, f"{f}: KV_BLOCK_ROWS must be ROWS_PER_CTA * CTA_MMA; got {cfg.KV_BLOCK_ROWS}"),
            (
                cfg.STAGE3_GRAN_ROWS == 256 and cfg.STAGE3_GRAN_ROWS % cfg.KV_BLOCK_ROWS == 0 and cfg.STAGE3_GRAN_ROWS % cfg.TILE_N == 0,
                f"{f}: STAGE3_GRAN_ROWS is the 256-row kv write pair (the (256, 256) stage-3 tile's M, whole kv blocks, whole q tiles); got {cfg.STAGE3_GRAN_ROWS}",
            ),
            (
                cfg.SUBBLOCK_WGS == cfg.SOFTMAX_WARPGROUPS // cfg.KV_SUBBLOCKS and cfg.COLS_PER_LANE == (cfg.TILE_N // 2) // cfg.SUBBLOCK_WGS,
                f"{f}: SUBBLOCK_WGS = SOFTMAX_WARPGROUPS // KV_SUBBLOCKS and COLS_PER_LANE = 64 // SUBBLOCK_WGS (32 on profile 1, 64 on profile 2); got {cfg.SUBBLOCK_WGS}/{cfg.COLS_PER_LANE}",
            ),
            (
                cfg.COLS_PER_LANE in (32, 64),
                f"{f}: a lane reads 32 or 64 q columns (tcgen05.ld.32x32b.x32 / .x64, one or two mask keep-words); got {cfg.COLS_PER_LANE}",
            ),
            (
                cfg.L_CNT == (cfg.SOFTMAX_LANES // cfg.KV_SUBBLOCKS) * cfg.CTA_MMA,
                f"{f}: L_CNT (the per-sub-block LEADER-scope fan-in) must be the sub-block's compute lanes x CTA_MMA = {(cfg.SOFTMAX_LANES // cfg.KV_SUBBLOCKS) * cfg.CTA_MMA}; got {cfg.L_CNT}",
            ),
            (cfg.MMA_LOOKAHEAD in (0, 1), f"{f}: MMA_LOOKAHEAD is 0 (NATURAL) or 1 (lookahead); got {cfg.MMA_LOOKAHEAD}"),
        ]
    )
    # --- dtype / k-step --------------------------------------------------------------------------------------------
    _check(
        [
            (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), f"{f}: the body takes BF16 or FP16; got DTYPE_QKV={cfg.DTYPE_QKV}"),
            (
                cfg.DTYPE_O in (DTYPE_BF16, DTYPE_FP16) and cfg.DTYPE_DS == cfg.DTYPE_QKV,
                f"{f}: dV is half precision and the dS workspace IS the io dtype; got DTYPE_O={cfg.DTYPE_O}, DTYPE_DS={cfg.DTYPE_DS}",
            ),
            (
                cfg.BPE == bpe(cfg.DTYPE_QKV) == 2 and cfg.BPE_O == bpe(cfg.DTYPE_O) and cfg.BPE_DS == bpe(cfg.DTYPE_DS),
                f"{f}: BPE/BPE_O/BPE_DS must match their dtypes (2 B)",
            ),
            (cfg.IS_FP8 == 0 and cfg.IS_MXFP8 == 0, f"{f}: an f16 body (no fp8 / mxfp8 arm)"),
            (
                cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16 and cfg.IDESC_K_DIM == 0,
                f"{f}: bf16 / fp16 k-steps are K = 16 with idesc k_dim = 0 on both arches (no K64 seam); got {cfg.TILE_K_HW_BMM1}/{cfg.TILE_K_HW_BMM2}/{cfg.IDESC_K_DIM}",
            ),
            (
                cfg.K_SPLIT_UTCCP == 0 and cfg.STAGES_TMEM_P == 0,
                f"{f}: no UTCCP K-split and no TMEM P (every operand is an SMEM SS operand); got K_SPLIT_UTCCP={cfg.K_SPLIT_UTCCP}, STAGES_TMEM_P={cfg.STAGES_TMEM_P}",
            ),
        ]
    )
    # --- rings and TMEM ----------------------------------------------------------------------------------------------
    _check(
        [
            (cfg.STAGES_KV == 1, f"{f}: K and V are loaded ONCE per kv block (STAGES_KV == 1); got {cfg.STAGES_KV}"),
            (cfg.STAGES_TMEM_S in (1, 2), f"{f}: S / dP are 1- or 2-deep TMEM parity rings; got STAGES_TMEM_S={cfg.STAGES_TMEM_S}"),
            (
                cfg.STAGES_SMEM_P >= 1,
                f"{f}: the SMEM P ring is at least 1 deep (its reuse is the explicit mb_p_empty, never an issue-order invariant); got {cfg.STAGES_SMEM_P}",
            ),
            (cfg.STAGES_Q >= 1 and cfg.STAGES_dO >= 1 and cfg.STAGES_dO_DV >= 1 and cfg.XFER_STAGES >= 1, f"{f}: every operand ring is at least 1 deep"),
            (
                cfg.STATS_STAGES == 2 and cfg.SCHEDULER_STAGES == 2,
                f"{f}: the lse/do_dot prefetch ring and the CLC ring are 2-deep (the scheduler warp loop assumes it)",
            ),
            (
                cfg.TMEM_ALLOC_COLS == TMEM_ALLOC_COLS and tm.TOTAL_COLS == TMEM_ALLOC_COLS,
                f"{f}: one 512-column non-exclusive tcgen05.alloc.cta_group::2 on both arches; got {cfg.TMEM_ALLOC_COLS}",
            ),
            (
                tm.USED_COLS <= tm.TOTAL_COLS,
                f"{f}: TMEM map KV_SUBBLOCKS * (STAGES_TMEM_S * 64 * 2 + 128) = {tm.USED_COLS} must fit the {tm.TOTAL_COLS}-column alloc",
            ),
            (
                cfg.SUBBLOCK_STRIDE_COLS == tm.SUB_STRIDE,
                f"{f}: SUBBLOCK_STRIDE_COLS must be the derived sub-block stride {tm.SUB_STRIDE}; got {cfg.SUBBLOCK_STRIDE_COLS}",
            ),
        ]
    )
    # --- warps / registers / lane constants (the 4x1 census; SASS pins carry over) -----------------------------------
    _check(
        [
            (
                cfg.SOFTMAX_WARPGROUPS == 2 and cfg.SOFTMAX_WG_WARPS == 4 and cfg.CORRECTION_WARPS == 0,
                f"{f}: 8 compute warps in 2 warpgroups of 4, no correction warps",
            ),
            (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, f"{f}: 12 warps / 384 threads per CTA"),
            (
                (cfg.SOFTMAX_WG0_BASE, cfg.SOFTMAX_WG1_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID)
                == (0, 4, 8, 9, 10, 11),
                f"{f}: warp ids wg0 @0, wg1 @4, MMA 8, TMALDG 9, TMASTG 10, SCHED 11",
            ),
            (
                cfg.READ_TILE_ARRIVERS_TOT == read_tile_arrivers_tot_2x2(cfg) == 21,
                f"{f}: READ_TILE_ARRIVERS_TOT must be {read_tile_arrivers_tot_2x2(cfg)}; got {cfg.READ_TILE_ARRIVERS_TOT}",
            ),
            (
                cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS == cfg.OTHER_REGS,
                f"{f}: the four service warps share one register count",
            ),
            (
                reg_total <= reg_entry_pool(cfg.TOTAL_WARPS),
                f"{f}: register split {reg_total} exceeds the {cfg.TOTAL_WARPS}-warp ENTRY pool {reg_entry_pool(cfg.TOTAL_WARPS)} (setmaxnreg.inc can only take what .dec released: a HANG)",
            ),
            (all(r % 8 == 0 and 24 <= r <= 256 for r in (cfg.SOFTMAX_REGS, cfg.OTHER_REGS)), f"{f}: register counts are multiples of 8 in [24, 256]"),
            (
                cfg.ONE_LANE == 1
                and cfg.ONE_WARP == 32
                and cfg.SOFTMAX_WG_LANES == 128
                and cfg.SOFTMAX_LANES == 256
                and cfg.SOFT_X_CTA_MMA == 512
                and cfg.MMA_COMMIT_ARRIVES == 1,
                f"{f}: mbarrier lane constants are the 12-warp body's (1 / 32 / 128 / 256 / 512 / 1)",
            ),
        ]
    )
    # --- masks / schedule ---------------------------------------------------------------------------------------------
    _check(
        [
            (not (cfg.CAUSAL_BOTTOM_RIGHT and not (cfg.MASK_FLAGS & MASK_CAUSAL)), f"{f}: bottom-right alignment requires a causal band"),
            (bool(cfg.MASK_FLAGS & MASK_SWA) == (cfg.SWA_WINDOW > 0), f"{f}: MASK_SWA <=> SWA_WINDOW > 0"),
            (bool(cfg.MASK_FLAGS & MASK_PADDED) == bool(cfg.SEQ_KV_LENS_PRESENT), f"{f}: MASK_PADDED <=> SEQ_KV_LENS_PRESENT"),
            (cfg.SCHEDULER_POLICY in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2), f"{f}: SCHEDULER_POLICY must be 0/1/2"),
        ]
    )
    # --- swizzles -----------------------------------------------------------------------------------------------------
    _check(
        [
            (
                cfg.Q_SWZ_BYTES == cfg.K_SWZ_BYTES == cfg.dO_SWZ_BYTES == cfg.V_SWZ_BYTES == cfg.P_SWZ_BYTES == cfg.dS_SWZ_BYTES == cfg.dV_SWZ_BYTES == 128,
                f"{f}: every slab is 128-B swizzled (the s128b TMA boxes, the SW128 K-major / BT descriptors and the Swizzle(3,4,3) lane stores are ONE unit)",
            ),
            (b.P_STORE_ALIGN in (64, 128), f"{f}: a lane's P / dS segment is a 64-B half row (profile 1) or a 128-B row (profile 2); got {b.P_STORE_ALIGN}"),
        ]
    )
    # --- SMEM -------------------------------------------------------------------------------------------------------
    slabs = smem_layout_2x2(cfg)
    used = smem_bytes_2x2(cfg)
    cap = cfg.SMEM_CAP
    if used + SMEM_SCAFFOLD_BYTES > cap:
        tally = " | ".join(f"{s.name} {s.nbytes // 1024} KiB @{s.offset // 1024}" for s in slabs)
        raise ValueError(
            f"{f}: SMEM slabs {used // 1024} KiB + {SMEM_SCAFFOLD_BYTES // 1024} KiB scaffolding exceed the {cap // 1024} KiB per-CTA cap of profile {cfg.DATAPATH_2X2_PROFILE} ({tally})"
        )
    declared = scaffold_bytes_declared_2x2(cfg)
    if declared > SMEM_SCAFFOLD_BYTES:
        raise ValueError(f"{f}: declared scaffolding ({declared} B) exceeds the {SMEM_SCAFFOLD_BYTES} B budget")
    for lbl, off in desc_roots_2x2(cfg):
        if not (0 <= off < used):
            raise ValueError(f"{f}: descriptor root {lbl} at {off} B lies outside the {used} B slab layout")
    want_cap = SMEM_CAP_BYTES_SM100 if cfg.DATAPATH_2X2_PROFILE == PROFILE_SM100 else SMEM_CAP_BYTES_SM107
    want_v = 0 if cfg.DATAPATH_2X2_PROFILE == PROFILE_SM100 else 1
    _check(
        [
            (cap == want_cap, f"{f}: profile {cfg.DATAPATH_2X2_PROFILE} is validated against the {want_cap // 1024} KiB cap; got SMEM_CAP={cap}"),
            (
                desc_version_2x2(cfg) == want_v,
                f"{f}: profile {cfg.DATAPATH_2X2_PROFILE} renders descriptor version {want_v} (every root under 256 KiB on SM100; the Rubin sP root past it); the layout tally says {desc_version_2x2(cfg)}",
            ),
            (
                b.dVBufferElems * cfg.BPE_O == b.kBufferElems * cfg.BPE,
                f"{f}: the dV staging aliases sK exactly (dVBufferElems * BPE_O == kBufferElems * BPE); got {b.dVBufferElems * cfg.BPE_O} vs {b.kBufferElems * cfg.BPE}",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_cfg_d256_2x2(params: _BwdTemplateParams, dtype_family: str) -> CfgBwdD256x2:
    """Build and validate the Cfg of the 2x2 body for ``params`` (``dtype_family`` must be ``FAMILY_F16``; the
    profile is ``params.datapath_2x2_profile`` and 0 raises -- the 4x1 bodies never call this)."""
    if dtype_family != FAMILY_F16:
        raise ValueError(f"{_FLAVOR}: dtype_family must be FAMILY_F16 ({FAMILY_F16!r}); the 2x2 body has no fp8 / mxfp8 arm; got {dtype_family!r}")
    profile = int(getattr(params, "datapath_2x2_profile", PROFILE_OFF))
    if profile not in PROFILES:
        raise ValueError(
            f"{_FLAVOR}: datapath_2x2_profile must be {PROFILE_SM100} (sm100: one 64-row sub-block per CTA) or {PROFILE_SM107_INTERLEAVED} "
            f"(sm107 interleaved: two sub-blocks per CTA); 0 is the shipped 4x1 body, which never builds this config; got {profile}"
        )
    if getattr(params, "thd_varlen", False):
        # The 4x1 half row serves THD on its packed path; this body has no THD arm (no packed descriptors, no per-sequence
        # tile decode), and the shared validator no longer refuses the field for the family.
        raise ValueError(f"{_FLAVOR}: thd_varlen is not implemented on the 2x2 body (the 4x1 body serves THD / ragged)")
    _c._validate_params(_FLAVOR, FAMILY_F16, params, datapath_2x2=True)
    dtype_o = getattr(params, "dtype_o", -1)
    if dtype_o < 0:
        dtype_o = params.dtype_qkv
    dtype_ds = getattr(params, "dtype_ds", -1)
    if dtype_ds < 0:
        dtype_ds = params.dtype_qkv
    mask_flags = _c._mask_flags_from(params)
    subblocks = profile
    rows_per_cta = SUB_ROWS * subblocks
    stages_tmem_s = 2 if profile == PROFILE_SM100 else 1
    stages_smem_p = 2 if profile == PROFILE_SM100 else 1
    stages_q = 1 if profile == PROFILE_SM100 else 2
    subblock_wgs = 2 // subblocks
    cols_per_lane = 64 // subblock_wgs
    soft_lanes = 2 * 4 * 32
    # Register split per profile, 8 x softmax + 4 x service = 2016 = reg_entry_pool(12).
    #   Profile 1 (32 q columns per compute lane): 176 / 152.  NOT the 4x1 body's 224 / 56: the Q / dO / dO_dv rings are
    #   1-deep, so their slab addresses are STATIC and ptxas hoists every k-step descriptor of the three B operands
    #   (16 + 16 + 8 64-bit values; desc_opaque's mov is transparent to ptxas) into the MMA warp's preamble as kernel
    #   invariants -- at 56 registers it parked them in local memory (70 STL / 79 LDL sm_100a 2026-10-01; 74 / 81 at
    #   sm_107a on the board's ptxas).  The 32-column compute lanes need ~135 registers (max R134 on the causal build).
    #   Pinned 0 / 0 STL / LDL by the sm_100a / sm_103a SASS spill pins (the board's sm_107a build of this profile reads
    #   18 / 43 at 176 / 152 and worse at every other split tried: 208 / 88 -> 57 / 64; it is the A/B arm there, not a row).
    #   Profile 2 (64 q columns per compute lane: twice the S / dP / P registers): 224 / 56, the 4x1 body's split.  Measured
    #   on the Rubin board (sm_107a, internal toolkit ptxas, 2026-10-01, causal): 176 / 152 -> 91 STL / 129 LDL (stack 448),
    #   208 / 88 -> 113 / 135, 216 / 72 and 224 / 56 and 232 / 40 -> 0 / 0 (stack 0).  The 2-deep Q ring keeps the B
    #   descriptors dynamic there, so the MMA warp fits in 56.  Pinned 0 / 0 by the sm_107a profile-2 SASS spill pins.
    softmax_regs, service_regs = (176, 152) if profile == PROFILE_SM100 else (224, 56)
    cfg = CfgBwdD256x2(
        TILE_M=SUB_ROWS,
        TILE_N=128,
        TILE_K=256,
        TILE_O=256,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=bpe(params.dtype_qkv),
        BPE_O=bpe(dtype_o),
        IS_FP8=0,
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
        TILE_K_HW_BMM1=16,
        TILE_K_HW_BMM2=16,
        IDESC_K_DIM=0,
        K_SPLIT_UTCCP=0,
        STAGES_Q=stages_q,
        STAGES_dO=1,
        STAGES_dO_DV=1,
        STAGES_KV=1,
        STAGES_TMEM_S=stages_tmem_s,
        STAGES_TMEM_P=0,
        STAGES_SMEM_P=stages_smem_p,
        XFER_STAGES=1,
        STATS_STAGES=2,
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        SOFTMAX_WARPGROUPS=2,
        SOFTMAX_WG_WARPS=4,
        CORRECTION_WARPS=0,
        # Register split: per profile (see above) -- 176 / 152 on profile 1, 224 / 56 on profile 2.
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
        SOFTMAX_LANES=soft_lanes,
        SOFT_X_CTA_MMA=soft_lanes * 2,
        MMA_COMMIT_ARRIVES=1,
        READ_TILE_ARRIVERS_TOT=2 * (2 * 4 + 2) + 1,
        N_BMM2_CHUNKS=1,
        BMM2_CHUNK_SIZE=128,
        # --- the 2x2 axis ---
        DATAPATH_2X2_PROFILE=profile,
        KV_SUBBLOCKS=subblocks,
        ROWS_PER_CTA=rows_per_cta,
        KV_BLOCK_ROWS=rows_per_cta * 2,
        MMA_M=MMA_M,
        STAGE3_GRAN_ROWS=256,
        SUBBLOCK_WGS=subblock_wgs,
        COLS_PER_LANE=cols_per_lane,
        L_CNT=(soft_lanes // subblocks) * 2,
        TMEM_ALLOC_COLS=TMEM_ALLOC_COLS,
        SMEM_CAP=SMEM_CAP_BYTES_SM100 if profile == PROFILE_SM100 else SMEM_CAP_BYTES_SM107,
        # Profile 1 ships the NATURAL order (measured, see the field); profile 2 keeps the design's lookahead until the Rubin A/B.
        MMA_LOOKAHEAD=0 if profile == PROFILE_SM100 else 1,
        SUBBLOCK_STRIDE_COLS=(2 * stages_tmem_s * _S_COLS + _DV_COLS) if subblocks > 1 else 0,
    )
    _validate_cfg_d256_2x2(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Shape-time helpers for the adapter / the kernel's compile()
# ---------------------------------------------------------------------------


def q_pad_rows_2x2(cfg: CfgBwdD256x2) -> int:
    """S_q must be padded to the q tile (the inner loop walks whole q tiles)."""
    return cfg.TILE_N


def kv_pad_rows_2x2(cfg: CfgBwdD256x2) -> int:
    """S_kv must be padded to the kv WRITE pair (``STAGE3_GRAN_ROWS``, the stage-3 GEMMs' cluster M tile and K-trim
    granularity), NOT to the pair's kv block: both 128-row blocks of a write pair walk the identical q range."""
    return cfg.STAGE3_GRAN_ROWS


def q_write_tiles_2x2(cfg: CfgBwdD256x2) -> int:
    """The q tiles a kv block's masked q range is rounded OUTWARD to (the write pair's, 2)."""
    return cfg.STAGE3_GRAN_ROWS // cfg.TILE_N


def ds_workspace_bytes_2x2(cfg: CfgBwdD256x2, batch: int, qh_chunk: int, s_q_pad: int, s_kv_pad: int) -> int:
    if s_q_pad % q_pad_rows_2x2(cfg) or s_kv_pad % kv_pad_rows_2x2(cfg):
        raise ValueError(
            f"{_FLAVOR}: workspace extents must be padded to q {q_pad_rows_2x2(cfg)} / kv {kv_pad_rows_2x2(cfg)} rows; got S_q={s_q_pad}, S_kv={s_kv_pad}"
        )
    return batch * qh_chunk * s_kv_pad * s_q_pad * cfg.BPE_DS


def launch_grid_2x2(cfg: CfgBwdD256x2, batch: int, qh_chunk: int, s_kv_pad: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """``(grid, cluster)``: NATURAL ``(kv_blocks * CGA_M, qh_chunk, B)`` over ``KV_BLOCK_ROWS``-row blocks; LPT / LPT_L2
    the flat 1-D grid the body's tile decode expects."""
    kv_blocks = -(-s_kv_pad // cfg.KV_BLOCK_ROWS)
    if cfg.SCHEDULER_POLICY == SCHED_NATURAL:
        grid = (kv_blocks * cfg.CGA_M, qh_chunk, batch)
    else:
        grid = (kv_blocks * qh_chunk * batch * cfg.CGA_M, 1, 1)
    return grid, (cfg.CGA_M, cfg.CGA_N, 1)
