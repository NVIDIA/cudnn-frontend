# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM107 2x2 half geometry and descriptor validation; no sibling factory dispatch."""

from dataclasses import dataclass, replace as dataclasses_replace
from typing import Tuple
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.config_sm107 import (
    TmaIters,
    SMEM_USABLE_BYTES,
    TCGEN05_V0_ADDR_LIMIT,
    _validate_params,
    _validate_params_sm100,
    _validate_dtype_k_step,
    _band_fields,
    _check,
    _tma_iters,
    _pack_g,
    resolve_dtype_o,
    bpe,
    tile_k_hw,
    rescale_threshold,
)
from cudnn.frost.tile_dsl.constants import DTYPE_BF16 as _DTYPE_BF16, DTYPE_FP16 as _DTYPE_FP16, SCHED_NATURAL

from cudnn.frost.tile_dsl.constants import DTYPE_FP16, MASK_NONE
from cudnn.sdpa.fwd.config_d512_2x2_common import d512_2x2_geometry_checks, d512_2x2_p_ring_start_bytes, d512_2x2_smem_bytes


@dataclass(frozen=True)
class CfgD512X2:
    TILE_M: int = 64
    TILE_N: int = 128
    TILE_K: int = 512
    TILE_O: int = 512

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    # Cluster (CGA_M, CGA_N, 1) = CGA_M // CTA_MMA cta_group::2 pairs sharing K/V.
    CGA_M: int = 4
    CGA_N: int = 1
    CTA_MMA: int = 2
    # Pairs that share each K/V sub-chunk by TMA multicast: 2 = twin multicast
    # (each CTA issues half of every sub-chunk to itself and its twin c ^ 2),
    # 1 = own-bit loads (the CGA_M=2 bring-up arm).  Always CGA_M // CTA_MMA.
    KV_SHARE: int = 2
    # Q super-tiles (64-row CTA tiles) per cluster: every CTA of the cluster owns
    # its own 64 rows, so the scheduler / bounds helpers decode in CGA_M units
    # (make_sdpa_helpers(kv_shared_cluster=True)); the role-split d512 decodes in
    # CTA_MMA units because its two pairs own the SAME rows.
    Q_SUPERS_PER_CLUSTER: int = 4
    ROWS_PER_CLUSTER: int = 4 * 64

    SPLIT_PIPELINE: int = 1

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    # f16 TILE_K_HW = 16 on SM10x (the K=64 2-chunk form is FP8-only; the f16
    # 2-chunk K=32 path is silently wrong on SM10x -- see _validate_cfg_d512).
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2
    # K ring: 32 KiB d-half sub-chunks (this CTA's 64 kv rows x 256 d cols); two
    # per KV iteration.  V ring: 32 KiB 128-col sub-chunks (128 kv rows x this
    # CTA's d_v [256*cta_in_pair + 128c, +128)); two per KV iteration.  Each
    # ring holds STAGES_*_SUB sub-chunks = one KV tile of buffering at 2.
    STAGES_K_SUB: int = 2
    STAGES_V_SUB: int = 2
    # S parities (TMEM) = P ring slots (SMEM).  BMM1 runs BMM1_LOOKAHEAD
    # iterations ahead of BMM2; the S/P slot reuse is ordered by the MMA
    # thread's own issue order (p_full(i) -> BMM2(i) -> BMM1(i + XFER_STAGES)),
    # which needs XFER_STAGES >= BMM1_LOOKAHEAD + 1 (checked by the validator).
    XFER_STAGES: int = 2
    BMM1_LOOKAHEAD: int = 1
    # STAGES_KV: the role-split name, kept for the shared helpers that read it
    # (max of the two sub-chunk rings, in sub-chunks).
    STAGES_KV: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4

    # setmaxnreg split: 4 softmax warps INCREASE to 192 (64 fp32 S + 32 packed P
    # + mask words), 4 correction warps INCREASE to 208 (two live 64-fp32 epilogue
    # batches), the 4 single warps DECREASE to 40.  Validator: 40 + 208 + 192 <= 512
    # and 32 * (4*192 + 4*208 + 4*40) <= 65536.
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 208
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    # BMM2 is issued per 256-wide (collective) N-block = one 32 KiB V sub-chunk;
    # each N-block has its own bmm2_ready credit.
    BMM2_N_PER_CALL: int = 256
    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 256

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 0
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256

    # --- mbarrier arrival ledger (exact per-phase sums; the kernel's init counts
    # are THESE constants and a host-only test re-derives them) ---
    # read_tile_id_arrive lands ONE arrive per calling warp on EVERY CTA of the
    # cluster: 4 softmax + 4 correction + TMA-LDG + TMA-STG = 10 warps per CTA,
    # plus the MMA warp of each pair LEADER (the quiet non-leader never credits):
    # 10 * CGA_M + CGA_M // CTA_MMA = 42 (CGA_M=4) / 21 (CGA_M=2, = CfgD256's).
    READ_TILE_ARRIVERS: int = 10 * 4 + 4 // 2
    # k/v_empty: one tcgen05.commit per pair LEADER landing on every CTA of the
    # share group (mask 0xF under KV_SHARE=2, the pair mask under KV_SHARE=1).
    KV_EMPTY_ARRIVERS: int = 4 // 2
    # mb_o_full[s]: the 64 lanes of ONE column half (d_v half s // 4) arrive per
    # 8 KiB O subtile -- NOT the 128 lanes of the role-split epilogue.
    O_CHUNK_ARRIVERS: int = 64
    # mb_o_empty: the 32 TMA-STG lanes of this CTA AND of every twin that
    # multicasts V into the aliased sVO (arrive + arrive_on_peer(cta ^ 2)):
    # ONE_WARP * KV_SHARE = 64 (CGA_M=4) / 32 (CGA_M=2).
    O_EMPTY_ARRIVERS: int = 32 * 2
    # mb_p_full / mb_bmm2_ready / mb_empty_mainloop / mb_tmem_dealloc: every lane
    # of the producing warpgroup on BOTH CTAs of the pair.
    PAIR_LANES: int = 128 * 2

    # TMEM: O 64x512 fp32 in the 2x2 atom = 256 cols at [0,256), S parity p =
    # 64 cols at [256 + 64p, +64), alpha[s] at 384+s, tile stats of ring slot s
    # at 386+2s / 387+2s (slot-indexed: a fixed stats pair raced the alpha ring).
    TMEM_COLS: int = 512
    O_TMEM_COLS: int = 256
    S_TMEM_COLS: int = 64
    # tcgen05 SMEM-descriptor version (0 = SM100 format, every operand below 256 KiB).
    DESC_VERSION: int = 0
    # O staging aliases the V ring (today's sg1 O u V protocol: the first V load
    # of a tile waits mb_o_empty).
    OV_ALIAS: int = 1
    # Per-CTA dynamic SMEM cap the budget is checked against (opt-in max).
    SMEM_CAP_BYTES: int = 227 * 1024
    # Worst-case dynamic-SMEM base alignment pad for the 1024-aligned first array.
    SMEM_ALIGN_PAD: int = 1008

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0

    # KV split; 1 = off.  The 2x2 body carries the half-aware fp32-partials arm;
    # the adapter twin keeps split_kv > 1 on the role-split kernel in phase 1.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    # SM100 packs whole groups; the native SM107 body also serves divisors of G.
    PACK_G: int = 1

    # D512 has two 256-column K chunks. Native D576 uses three 192-column
    # chunks so resident Q and both K/V rings fit under the SM100 SMEM cap.
    N_BMM1_CHUNKS: int = 2
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0


_D512_2X2_STAGES_K_SUB = 3
_D512_2X2_STAGES_V_SUB = 3


def _d512_2x2_desc_version_for(cfg: CfgD512X2) -> int:
    """The tcgen05 SMEM-descriptor version the 2x2 layout NEEDS: 1 as soon as the last MMA-operand tile (the P ring,
    the highest slab) reaches the 256 KiB window of a version-0 descriptor, else 0."""
    p_end = d512_2x2_p_ring_start_bytes(cfg) + cfg.XFER_STAGES * cfg.TILE_M * cfg.TILE_N * cfg.BPE
    return 1 if p_end > TCGEN05_V0_ADDR_LIMIT else 0


def _validate_cfg_d512_2x2_sm107(cfg: CfgD512X2, flavor: str) -> None:
    """The arch-neutral 2x2 geometry checks plus the Rubin arch facts."""
    smem = d512_2x2_smem_bytes(cfg)
    want_desc = _d512_2x2_desc_version_for(cfg)
    _check(
        d512_2x2_geometry_checks(cfg)
        + (
            (
                cfg.STAGES_K_SUB == _D512_2X2_STAGES_K_SUB and cfg.STAGES_V_SUB == _D512_2X2_STAGES_V_SUB,
                f"{flavor}: the Rubin arm runs 3-deep K/V sub-chunk rings (got {cfg.STAGES_K_SUB}/{cfg.STAGES_V_SUB})",
            ),
            (
                cfg.DESC_VERSION == want_desc,
                f"{flavor}: DESC_VERSION={cfg.DESC_VERSION} but the layout puts the P ring at {d512_2x2_p_ring_start_bytes(cfg)} B "
                f"(version-0 window ends at {TCGEN05_V0_ADDR_LIMIT}); it needs version {want_desc}",
            ),
            (
                cfg.SMEM_CAP_BYTES == SMEM_USABLE_BYTES,
                f"{flavor}: the SMEM budget is checked against the {SMEM_USABLE_BYTES // 1024} KiB usable carveout, got {cfg.SMEM_CAP_BYTES}",
            ),
            (
                smem["total"] <= SMEM_USABLE_BYTES,
                f"{flavor}: SMEM {smem['total']} B (incl. {cfg.SMEM_ALIGN_PAD} B pad) exceeds the {SMEM_USABLE_BYTES // 1024} KiB usable Rubin "
                f"carveout: {smem}.  Overflowing it does NOT fail the launch -- it clobbers the last buffer allocated",
            ),
            (cfg.TMEM_COLS == 512, f"{flavor}: 512 TMEM columns (388 used); 576 + is_exclusive only with a 3rd/4th S parity"),
            (
                cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE,
                f"{flavor}: the O u V alias gate is PAIR-WIDE -- mb_o_empty init must be ONE_WARP x KV_SHARE = {cfg.ONE_WARP * cfg.KV_SHARE} "
                f"(got {cfg.O_EMPTY_ARRIVERS}); a twin's multicast V(t+1) lands in BOTH twins' sVO",
            ),
        )
    )
    _validate_dtype_k_step(cfg, flavor)


@dataclass(frozen=True)
class CfgD512Mixed(CfgD512X2):
    DTYPE_KV: int = -1


def make_cfg_d512_2x2(params: TemplateParams, *, cga_m: int = 4) -> Tuple[CfgD512X2, TmaIters]:
    """The Rubin 2x2-datapath d512 configuration (``TemplateParams.mma_2x2``).

    ``cga_m`` = 4 (two twin pairs sharing K/V by multicast, the shipped arm) or 2 (the bring-up / bitwise-twin arm:
    one pair, own-bit loads).  Not a knob: tests load the template with the arm they want through the module constant
    the kernel file reads (FROST_D512_2X2_CGA_M)."""
    flavor = "sm107 d512 2x2"
    if not getattr(params, "mma_2x2", False):
        raise ValueError(f"{flavor}: make_cfg_d512_2x2 needs a TemplateParams record with mma_2x2=True (make_cfg_d512 dispatches on it)")
    # The Rubin record guard keys its THD allowlist by FLAVOR NAME (_F16_THD_FLAVORS holds "sm107 d512"): the 2x2 sibling
    # carries the same ported setup-kernel call sites as the role-split d512 body, so it is validated under that name.
    _validate_params("sm107 d512", params, split_wired=True)
    # Share the half geometry guards with SM100. This native twin additionally
    # supports paged 2x2 loads, packed subgroups and the pre-folded scale.
    mixed_kv = params.dtype_kv != -1
    if mixed_kv and cga_m != 2:
        raise ValueError(f"{flavor}: mixed FP8 KV requires one CTA pair (CGA_M=2)")
    # Apply the common half-geometry checks after the native storage contract
    # above. This does not admit mixed storage in the SM100 kernel.
    _validate_params_sm100(
        "d512",
        dataclasses_replace(params, dtype_kv=-1) if mixed_kv else params,
        scale_prefolded_wired=True,
        d512_partial_pack_wired=True,
        d512_paged_2x2_wired=True,
    )
    if params.dtype_qkv not in (_DTYPE_BF16, _DTYPE_FP16):
        raise ValueError(f"{flavor}: BF16/FP16 inputs only")
    if params.cta_mma != 2:
        raise ValueError(f"{flavor}: cta_group::2 pairs only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    n_pairs = cga_m // CfgD512X2.CTA_MMA
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)
    base = (CfgD512Mixed if mixed_kv else CfgD512X2)(
        TILE_K=576 if params.d_flavor == 576 else 512,
        N_BMM1_CHUNKS=3 if params.d_flavor == 576 else 2,
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size) if params.paged_kv else 0,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=cga_m,
        KV_SHARE=n_pairs,
        Q_SUPERS_PER_CLUSTER=cga_m,
        ROWS_PER_CLUSTER=cga_m * CfgD512X2.TILE_M,
        READ_TILE_ARRIVERS=10 * cga_m + n_pairs,
        KV_EMPTY_ARRIVERS=n_pairs,
        # PAIR-WIDE O u V alias gate (fix-lane FATAL-1): every twin's TMA-STG warp arrives on both twins' mb_o_empty.
        O_EMPTY_ARRIVERS=CfgD512X2.ONE_WARP * n_pairs,
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        STAGES_K_SUB=_D512_2X2_STAGES_K_SUB,
        STAGES_V_SUB=_D512_2X2_STAGES_V_SUB,
        STAGES_KV=max(_D512_2X2_STAGES_K_SUB, _D512_2X2_STAGES_V_SUB),
        SMEM_CAP_BYTES=SMEM_USABLE_BYTES,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv or 1),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD512X2.TILE_M, partial=True),
    )
    if mixed_kv:
        base = dataclasses_replace(base, DTYPE_KV=params.dtype_kv)
    # DESC_VERSION is DERIVED from the layout the fields above describe, never set by hand.
    cfg = dataclasses_replace(base, DESC_VERSION=_d512_2x2_desc_version_for(base))
    _validate_cfg_d512_2x2_sm107(cfg, flavor)
    return cfg, _tma_iters(cfg)
