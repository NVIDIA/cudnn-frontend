# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared immutable 2x2 record and geometry checks; architecture factories own tuning."""

from dataclasses import dataclass
from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, MASK_NONE, SCHED_NATURAL


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


def d512_2x2_smem_bytes(cfg: CfgD512X2, n_o_chunks: int = 8) -> dict:
    """The kernel's SMEM allocation from the SAME constants its arrays are declared with.

    data     = sQ (resident) + sK ring + (sV ring u sO staging) + sP ring
    scratch  = sXchgMax [XFER_STAGES parities][2 halves][TILE_M] fp32 + sXchgSum [2 halves][TILE_M] fp32
    barriers = mbarrier words (see the ledger in make_d512_2x2_bars) + scheduler bars + tile ids + tmem_ptr
    total    = data + scratch + barriers + SMEM_ALIGN_PAD (worst-case base pad for the 1024-aligned first array)
    """
    bpe = cfg.BPE
    q = cfg.TILE_M * cfg.TILE_K * bpe
    k_sub = (cfg.TILE_N // cfg.CTA_MMA) * (cfg.TILE_K // cfg.N_BMM1_CHUNKS) * bpe
    v_sub = cfg.TILE_N * (cfg.TILE_O // cfg.CTA_MMA // cfg.N_BMM2_CHUNKS) * bpe  # 128 kv rows x 128 d_v
    o = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O
    p = cfg.TILE_M * cfg.TILE_N * bpe
    v_ring = cfg.STAGES_V_SUB * v_sub
    vo = max(v_ring, o) if cfg.OV_ALIAS else v_ring + o
    data = q + cfg.STAGES_K_SUB * k_sub + vo + cfg.XFER_STAGES * p
    scratch = cfg.XFER_STAGES * 2 * cfg.TILE_M * 4 + 2 * cfg.TILE_M * 4
    n_bars = (
        2  # q_full, q_empty
        + 2 * cfg.STAGES_K_SUB  # k_full, k_empty
        + 2 * cfg.STAGES_V_SUB  # v_full, v_empty
        + 2 * cfg.XFER_STAGES  # bmm1_done, bmm2_done
        + cfg.XFER_STAGES * cfg.N_BMM2_CHUNKS  # bmm2_ready
        + cfg.XFER_STAGES  # p_full
        + 2 * 2  # stat_full, stat_empty (2-stage ring)
        + n_o_chunks  # o_full
        + 1  # o_empty
        + 1  # empty_mainloop
        + 1  # tmem_dealloc
    )
    if getattr(cfg, "DTYPE_KV", -1) in (DTYPE_E4M3, DTYPE_E5M2):
        # Native mixed storage widens one K/V sub-chunk at a time. The raw
        # slab follows the MMA operands, preserving their descriptor offsets.
        data += max(k_sub, v_sub) // bpe
        n_bars += 1
    barriers = n_bars * 8 + 2 * cfg.SCHEDULER_STAGES * 8 + cfg.SCHEDULER_STAGES * 8 * 4 + 16
    return dict(
        q=q,
        k_sub=k_sub,
        v_sub=v_sub,
        o=o,
        p=p,
        data=data,
        scratch=scratch,
        n_bars=n_bars,
        barriers=barriers,
        total=data + scratch + barriers + cfg.SMEM_ALIGN_PAD,
    )


def d512_2x2_p_ring_start_bytes(cfg: CfgD512X2) -> int:
    """SMEM byte offset at which the P ring begins = sQ + the K ring + the (V ring u O staging) slab -- the first
    MMA-operand tile that can cross the 256 KiB version-0 tcgen05 descriptor window (config_sm107.TCGEN05_V0_ADDR_LIMIT):
    192 KiB at 2/2 sub-chunk stages (SM100), exactly 262144 at 3/3 (SM107)."""
    smem = d512_2x2_smem_bytes(cfg)
    vo = max(cfg.STAGES_V_SUB * smem["v_sub"], smem["o"]) if cfg.OV_ALIAS else cfg.STAGES_V_SUB * smem["v_sub"] + smem["o"]
    return smem["q"] + cfg.STAGES_K_SUB * smem["k_sub"] + vo


def d512_2x2_geometry_checks(cfg: CfgD512X2) -> tuple:
    """The ARCH-NEUTRAL consistency checks on the 2x2-datapath d512 geometry, as ``(ok, message)`` pairs: every count
    the kernel's mbarrier inits and setmaxnreg take is re-derived here.  config_sm107 appends the Rubin arch checks
    (descriptor version from the layout, the 320 KiB usable budget) to this list; the SM100 ones follow below."""
    smem = d512_2x2_smem_bytes(cfg)
    n_pairs = cfg.CGA_M // cfg.CTA_MMA
    return (
        (cfg.TILE_M == 64 and cfg.TILE_N == 128, "d512 2x2: TILE_M=64 (cta_group::2 M=128 atom) / TILE_N=128"),
        (
            (cfg.TILE_K == 512 and cfg.N_BMM1_CHUNKS == 2) or (cfg.TILE_K == 576 and cfg.N_BMM1_CHUNKS == 3 and cfg.CGA_M == 2),
            "2x2 QK geometry requires D512 with two K chunks or D576 with three K chunks and one CTA pair",
        ),
        (cfg.TILE_O == 512, "2x2 wide-head geometry requires D_V=512"),
        (cfg.CTA_MMA == 2 and cfg.CGA_N == 1, "d512 2x2: cta_group::2 pairs, CGA_N=1"),
        (cfg.CGA_M in (2, 4) and cfg.CGA_M % cfg.CTA_MMA == 0, f"d512 2x2: CGA_M must be 2 (bring-up arm) or 4; got {cfg.CGA_M}"),
        (cfg.KV_SHARE == n_pairs, f"d512 2x2: KV_SHARE ({cfg.KV_SHARE}) must equal CGA_M // CTA_MMA ({n_pairs})"),
        (cfg.Q_SUPERS_PER_CLUSTER == cfg.CGA_M * cfg.CGA_N, "d512 2x2: one 64-row Q super-tile per CTA of the cluster"),
        (cfg.ROWS_PER_CLUSTER == cfg.TILES_Q * cfg.TILE_M * cfg.CGA_M * cfg.CGA_N, "d512 2x2: ROWS_PER_CLUSTER = TILES_Q * TILE_M * CGA_M * CGA_N"),
        (cfg.TILES_Q == 1 and cfg.SPLIT_PIPELINE == 1, "d512 2x2: TILES_Q=1, blocked NATURAL decode"),
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16) and cfg.DTYPE_O == cfg.DTYPE_QKV, "d512 2x2: half inputs with DTYPE_O == DTYPE_QKV"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d512 2x2: f16 TILE_K_HW must be 16 on SM10x (the 2-chunk form is silently wrong)"),
        (cfg.Q_SWZ_BYTES == cfg.K_SWZ_BYTES == cfg.V_SWZ_BYTES == cfg.O_SWZ_BYTES == 128, "d512 2x2: Q/K/V/O swizzle must all be 128B"),
        (
            cfg.XFER_STAGES >= cfg.BMM1_LOOKAHEAD + 1,
            f"d512 2x2: XFER_STAGES ({cfg.XFER_STAGES}) must be >= BMM1_LOOKAHEAD + 1 ({cfg.BMM1_LOOKAHEAD + 1}): S/P slot reuse rides the MMA issue order",
        ),
        (cfg.STAGES_K_SUB >= 2 and cfg.STAGES_V_SUB >= 2, "d512 2x2: both sub-chunk rings need >= 2 slots (two sub-chunks per iteration)"),
        (cfg.STAGES_KV == max(cfg.STAGES_K_SUB, cfg.STAGES_V_SUB), "d512 2x2: STAGES_KV mirrors the deeper sub-chunk ring"),
        (cfg.N_BMM2_CHUNKS == 2 and cfg.BMM2_N_PER_CALL == 256, "d512 2x2: BMM2 = two collective N=256 blocks (one 32 KiB V sub-chunk each)"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d512 2x2: 12 warps"),
        (cfg.SOFTMAX_WG_WARPS == 4 and cfg.CORRECTION_WARPS == 4 and cfg.SOFTMAX_WARPGROUPS == 1, "d512 2x2: 4 softmax + 4 correction warps"),
        (
            (cfg.SOFTMAX_WG0_BASE, cfg.CORR_WARP_BASE, cfg.MMA_WARP_ID, cfg.TMALDG_WARP_ID, cfg.TMASTG_WARP_ID, cfg.SCHED_WARP_ID) == (0, 4, 8, 9, 10, 11),
            "d512 2x2: warp roles 0-3 softmax, 4-7 correction, 8 MMA, 9 TMA-LDG, 10 TMA-STG, 11 scheduler",
        ),
        (cfg.SOFTMAX_LANES == 128 and cfg.CORR_LANES == 128 and cfg.PAIR_LANES == 256, "d512 2x2: 128 lanes per compute warpgroup, 256 across the pair"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS == cfg.OTHER_REGS, "d512 2x2: single-warp roles share OTHER_REGS"),
        (cfg.OTHER_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_REGS <= 512, "d512 2x2: register budget over 512"),
        (32 * (4 * cfg.SOFTMAX_REGS + 4 * cfg.CORRECTION_REGS + 4 * cfg.OTHER_REGS) <= 65536, "d512 2x2: per-SM register file over 64K"),
        (cfg.SOFTMAX_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.OTHER_REGS % 8 == 0, "d512 2x2: per-role regs must be multiples of 8"),
        (
            cfg.READ_TILE_ARRIVERS == 10 * cfg.CGA_M + n_pairs,
            f"d512 2x2: READ_TILE_ARRIVERS must be 10 * CGA_M + CGA_M // CTA_MMA = {10 * cfg.CGA_M + n_pairs}; got {cfg.READ_TILE_ARRIVERS}",
        ),
        (cfg.KV_EMPTY_ARRIVERS == n_pairs, f"d512 2x2: KV_EMPTY_ARRIVERS must be CGA_M // CTA_MMA = {n_pairs}; got {cfg.KV_EMPTY_ARRIVERS}"),
        (cfg.O_CHUNK_ARRIVERS == cfg.CORR_LANES // 2, "d512 2x2: mb_o_full is arrived by the 64 lanes of ONE column half"),
        (
            cfg.O_EMPTY_ARRIVERS == cfg.ONE_WARP * cfg.KV_SHARE,
            f"d512 2x2: O_EMPTY_ARRIVERS must be ONE_WARP * KV_SHARE = {cfg.ONE_WARP * cfg.KV_SHARE} (own + every twin's TMA-STG warp); got {cfg.O_EMPTY_ARRIVERS}",
        ),
        (
            cfg.O_TMEM_COLS == cfg.TILE_M * cfg.TILE_O // 128 and cfg.S_TMEM_COLS == cfg.TILE_M * cfg.TILE_N // 128,
            "d512 2x2: 2x2 atom TMEM footprints (N/2 cols)",
        ),
        (
            cfg.O_TMEM_COLS + cfg.XFER_STAGES * cfg.S_TMEM_COLS + 2 + 2 * 2 <= cfg.TMEM_COLS,
            f"d512 2x2: TMEM carve O {cfg.O_TMEM_COLS} + S {cfg.XFER_STAGES * cfg.S_TMEM_COLS} + alpha ring 2 + per-slot stats 4 > {cfg.TMEM_COLS}",
        ),
        (
            smem["total"] <= cfg.SMEM_CAP_BYTES,
            f"d512 2x2: SMEM {smem['total']} B (incl. {cfg.SMEM_ALIGN_PAD} B pad) over the {cfg.SMEM_CAP_BYTES} B cap: {smem}",
        ),
        (not cfg.PACK_GQA or cfg.TILE_M % cfg.PACK_G == 0, "d512 2x2: PACK_G must divide TILE_M"),
    )
