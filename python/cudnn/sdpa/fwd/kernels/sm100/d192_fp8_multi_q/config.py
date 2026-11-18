# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Independent configuration for the per-tensor FP8 D192 multi-Q body.

The frozen record retains the legacy field/default layout without inheriting
D128 or half/MXFP8 implementation records. Shared parameter validation and
immutable geometry helpers remain common.
"""

from dataclasses import dataclass
from typing import Tuple
from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, MASK_NONE, MASK_CAUSAL, MASK_SWA, SCHED_NATURAL
from cudnn.sdpa.fwd.config_common import (
    TemplateParams,
    TmaIters,
    _validate_params,
    _mask_flags_from,
    _pack_g,
    _tma_iters,
    bpe,
    q_swz_bytes,
    v_swz_bytes,
    o_swz_bytes,
    rescale_threshold,
    tile_k_hw,
)

# Blackwell per-CTA dynamic shared-memory limit, excluding the reserved KiB.
_SM100_MAX_DYN_SMEM = 227 * 1024


@dataclass(frozen=True)
class CfgD192:
    PADDED_TAIL_FORM: int = 0
    DEAD_SUBTILE_QUIET: int = 0
    BIAS_MODE: int = 0
    BIAS_RESIDENT_TILES: int = 0
    BIAS_BLOCK_ROWS: int = 0
    BIAS_SWZ_BYTES: int = 128
    BIAS_BOXES_PER_TILE: int = 0
    BIAS_EMPTY_ARRIVERS: int = 0
    KV_MASK_FORM: int = 0
    N_CHUNK: int = 0
    STAGES_BIAS: int = 0
    BIAS_MAX_FORM: int = 0
    S_SLOTS: int = 1
    BIAS_MCAST_CTAS: int = 1
    Q_IN_TMEM: int = 0
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 192
    TILE_O: int = 128
    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2
    QO_ALIAS: int = 1
    SPLIT_PIPELINE: int = 0
    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128
    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16
    TILES_Q: int = 2
    Q_TILES_LIVE: int = 2
    DUAL_KV_STREAM: int = 0
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2
    SOFTMAX_WARPGROUPS: int = 2
    CORRECTION_WARPS: int = 4
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40
    RESCALE_THRESHOLD: float = 8.0
    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0
    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL
    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64
    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
    CORR_WARP_BASE: int = 8
    MMA_WARP_ID: int = 12
    TMALDG_WARP_ID: int = 13
    TMASTG_WARP_ID: int = 14
    SCHED_WARP_ID: int = 15
    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256
    READ_TILE_ARRIVERS: int = 15
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1
    PACK_G: int = 1
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    DTYPE_KV: int = -1
    STAGES_K: int = 2
    STAGES_V: int = 2
    K_SUB: int = 1
    L2_PREFETCH_KV: int = 0
    L2_PREFETCH_NEXT_TILE: int = 0
    L2_PREFETCH_MIN_STEPS: int = 4
    LDG_UNROLL: int = 6
    EXP2_TAIL_MIX: int = 1


def _d192_smem_bytes(cfg) -> int:
    """Data-buffer SMEM for the d192 pipeline.

    Same shape as _d128_smem_bytes, but d_qk = 192 makes the Q and K slabs 1.5x
    the d128 ones, which is why this flavor needs a shallower KV pipeline:

        cga2, STAGES_KV=2 : 96(Q u O) + 48(K) + 32(V) = 176 KiB
        cga1, STAGES_KV=2 : 96        + 96    + 64    = 256 KiB  (over cap)
        cga1, STAGES_KV=1 : 96        + 48    + 32    = 176 KiB
        cga1, K=1 / V=2   : 96        + 48    + 64    = 208 KiB  (half input: STAGES_K=1, STAGES_V=2)
    """
    q_slab = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    o_slab = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O // cfg.O_PACK_DIV
    qo = cfg.TILES_Q * (max(q_slab, o_slab) if cfg.QO_ALIAS else q_slab + o_slab)
    k = cfg.STAGES_K * (cfg.TILE_N * cfg.TILE_K * cfg.BPE // cfg.CTA_MMA)
    v = cfg.STAGES_V * (cfg.TILE_O * cfg.TILE_N * cfg.BPE_V // cfg.CTA_MMA)
    return qo + k + v


def _validate_cfg_d192(cfg: CfgD192) -> None:
    """Consistency checks on the native DSv3 d192/d128 geometry."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    checks = (
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16) if fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d192: DTYPE_O must equal DTYPE_QKV for half input; FP8 allows an independent output dtype",
        ),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d192: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d192: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d192: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d192 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            cfg.STAGES_KV == ((3 if cfg.CTA_MMA == 2 else 1) if cfg.PV_BF16 else (2 if fp8 else 1) * cfg.CTA_MMA),
            "d192: STAGES_KV must be 3/1 for BF16 PV at cga2/cga1, otherwise follow the input-dtype pipeline depth",
        ),
        (
            cfg.STAGES_K == cfg.STAGES_KV and cfg.STAGES_V == (2 if (not fp8 and cfg.CTA_MMA == 1) else cfg.STAGES_KV),
            "d192: the K ring is STAGES_KV deep; the V ring is 2 deep on the half-input cga1 tile and STAGES_KV elsewhere",
        ),
        (
            cfg.K_SUB == ((cfg.TILE_K * cfg.BPE) // 128 if (not fp8 and cfg.CTA_MMA == 1) else 1),
            "d192: K is released per TMA D-subtile on the half-input cga1 tile only",
        ),
        (
            cfg.L2_PREFETCH_KV in (0, 1) and (cfg.L2_PREFETCH_KV == 0 or (not fp8 and cfg.CTA_MMA == 1)),
            "d192: the L2-prefetching loader (one kv step ahead) belongs to the half-input cga1 tile only",
        ),
        (cfg.L2_PREFETCH_NEXT_TILE == cfg.L2_PREFETCH_KV, "d192: the next-tile L2 prefetch comes with the K/V L2 prefetch"),
        (cfg.L2_PREFETCH_MIN_STEPS >= 1, "d192: L2_PREFETCH_MIN_STEPS must be at least 1"),
        (
            cfg.LDG_UNROLL == (1 if cfg.L2_PREFETCH_KV else 6),
            "d192: the TMA-LDG kv loop is unrolled 1x with the L2-prefetching loader and 6x otherwise",
        ),
        (
            cfg.L2_PREFETCH_KV == 0 or cfg.EXP2_TAIL_MIX == 0,
            "d192: EXP2_TAIL_MIX (the dense-causal arm's exp2-split spelling) is folded out with the L2-prefetching loader",
        ),
        (
            _d192_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d192: SMEM {_d192_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILE_K == 192 and cfg.TILE_O == 128, "d192: expected D_QK tile 192 and D_V tile 128"),
        (cfg.QO_ALIAS == (0 if fp8 else 1), "d192: Q/O SMEM alias must be disabled for FP8 and enabled for half input"),
        (cfg.TILES_Q == 2, "d192: TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d192: SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d192: CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d192: 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d192: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if fp8 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if fp8 else 16)),
            "d192: BMM1 uses FP8 K=32; BF16 PV uses BMM2 K=16",
        ),
        (
            cfg.Q_SWZ_BYTES == (64 if fp8 else 128) and cfg.K_SWZ_BYTES == (64 if fp8 else 128),
            "d192: Q/K swizzle must be 64B for FP8 and 128B for BF16/FP16",
        ),
        (
            cfg.V_SWZ_BYTES == v_swz_bytes(128, cfg.CTA_MMA, cfg.BPE_V) and cfg.O_SWZ_BYTES in ((64, 128) if fp8 else (128,)),
            "d192: V/O swizzle is inconsistent with the input/output dtype",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    """The SM100 d192x128 configuration for one set of public knobs: the 2-CTA pair, or the single-CTA tile with its
    per-D-subtile K ring and second V stage on half input."""
    _validate_params("d192", params)
    if params.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2):
        raise ValueError("per-tensor FP8 D192 requires E4M3/E5M2 inputs")
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    mask_flags = _mask_flags_from(params)
    thd_swa = fp8 and params.thd_varlen and bool(mask_flags & MASK_SWA)
    e4_thd_swa = thd_swa and params.dtype_qkv == DTYPE_E4M3
    e5_dense_causal_regs = (
        params.dtype_qkv == DTYPE_E5M2
        and not params.thd_varlen
        and params.split_kv == 1
        and not params.bottom_right
        and mask_flags == MASK_CAUSAL
        and (params.window_right or 0) == 0
    )
    wide_role_regs = thd_swa or e5_dense_causal_regs
    b_v = 2 if params.pv_bf16 else b
    stages_kv = (3 if params.cta_mma == 2 else 1) if params.pv_bf16 else (2 if fp8 else 1) * params.cta_mma
    # Half input at cga1: one full K stage (48 KiB) plus TWO V stages (64 KiB) under the 96 KiB Q u O slab -- the V
    # reload then overlaps a whole iteration instead of the PV->PV window a single stage leaves it.
    stages_v = 2 if (not fp8 and params.cta_mma == 1) else stages_kv
    k_sub = (192 * b) // 128 if (not fp8 and params.cta_mma == 1) else 1  # one slot per 128 B TMA D-subtile
    half_input_cga1 = not fp8 and params.cta_mma == 1
    cga1_loader = half_input_cga1 and bool(params.cga1_l2_prefetch)
    # The dense-causal arm's exp2-split spelling (CfgD192.EXP2_TAIL_MIX) comes with the exp2 FMA split claim and the
    # develop loader; a build without the claim, or with the L2-prefetching loader, keeps the plain MUFU exp2 there.
    exp2_tail_mix = bool(params.exp2_fma_split) and not cga1_loader
    cfg = CfgD192(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        SPLIT_KV=int(params.split_kv),
        QO_ALIAS=0 if fp8 else 1,
        Q_SWZ_BYTES=q_swz_bytes(192, b),
        K_SWZ_BYTES=q_swz_bytes(192, b),
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(128, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=32 if fp8 else tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=16 if params.pv_bf16 else (32 if fp8 else tile_k_hw(params.dtype_qkv)),
        STAGES_KV=stages_kv,
        STAGES_K=stages_kv,
        STAGES_V=stages_v,
        K_SUB=k_sub,
        L2_PREFETCH_KV=1 if cga1_loader else 0,
        L2_PREFETCH_NEXT_TILE=1 if cga1_loader else 0,
        LDG_UNROLL=1 if cga1_loader else 6,
        EXP2_TAIL_MIX=1 if exp2_tail_mix else 0,
        SCHEDULER_STAGES=3 if e4_thd_swa and not params.bottom_right else 2,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        L2_SIZE_MIB=params.lpt_l2_size_mib or 60,
        SCHEDULER_POLICY=params.sched_policy,
        SOFTMAX_REGS=192 if wide_role_regs else 184 if fp8 else 216 if mask_flags == MASK_NONE else 192,
        CORRECTION_REGS=88 if wide_role_regs else 104 if fp8 else 40 if mask_flags == MASK_NONE else 88,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD192.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d192(cfg)
    return cfg, _tma_iters(cfg)
