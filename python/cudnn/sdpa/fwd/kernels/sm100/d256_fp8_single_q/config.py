# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Configuration owned by the SM100 per-tensor FP8 D256 single-Q body.

The record preserves the legacy field/default layout. The factory only accepts
FP8 and fills the actual single-CTA geometry; half and MXFP8 own separate records.
"""

from dataclasses import dataclass
from typing import Tuple

from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_E4M3, DTYPE_E5M2, MASK_NONE, MASK_CAUSAL, MASK_PADDED, SCHED_NATURAL
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
)


@dataclass(frozen=True)
class CfgD256:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    SPLIT_PIPELINE: int = 0

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4
    FUSED_CORR_SPLIT_P: int = 0

    SOFTMAX_REGS: int = 240
    # The DSL traces the unreachable WG1 dispatch for one-WG specializations,
    # so its placeholder must still be a legal setmaxnreg operand.
    SOFTMAX_WG1_REGS: int = 40
    CORRECTION_REGS: int = 96
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
    STATS_LOG2: int = 0  # LSE stored in base 2 (stats_use_log2)
    BOTTOM_RIGHT: int = 0  # band diagonal anchored bottom-right

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4
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

    READ_TILE_ARRIVERS: int = ((1 * 4) + 4 + 2) * (2 * 1) + (2 // 2)

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0

    # KV split; 1 = off.  See TemplateParams.split_kv.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0

    # The graph's GQA ratio H_q / H_kv -- the bottom-right diagonal and the
    # LPT_L2 head grouping are derived from it.
    QH_PER_KH: int = 1
    # Heads packed into one Q tile row-group under PACK_GQA (the kernel's
    # HEADS_PER_TILE): QH_PER_KH when the group divides TILE_M, else -- the
    # d128 / d256 f16 kernels only -- its largest divisor that does
    # (pack_gqa_group_size); QH_PER_KH // PACK_G packed heads then share one KV
    # head.  1 when unpacked.  Distinct from QH_PER_KH on purpose: conflating
    # the pack size with the GQA ratio would mis-mask MTP rows.
    PACK_G: int = 1

    # Paged KV cache; see TemplateParams.paged_kv.  PAGE_SIZE tokens per page.
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0

    # Fused epilogue gate (O := O * sigmoid(G), TemplateParams.epilogue_gate), read by the SHARED f16/bf16 d256 body only
    # (sm100/d256_f16_lookahead/kernel.py; every seam folds on cutlass.const_expr(CFG.EPILOGUE_GATE), so 0 traces the pre-gate
    # instruction stream byte for byte -- the render gate of the paged gate).  Appended: every existing record
    # keeps its value.  GATE_BPE: bytes per gate element -- the gate is Q-dtype on the half body (2 B); every gate
    # constant in the kernel (TMA box, swizzle, subtile walk) derives from THIS, never from BPE_O (config_sm107 has the
    # e4m3-O lesson).
    EPILOGUE_GATE: int = 0
    GATE_BPE: int = 2

    # Half causal schedule: the dedicated correction WG also generates P1.
    # Unlike MXFP8 FUSED_CORR_SPLIT_P, this retains one softmax WG and four
    # correction warps. Append-only internal topology, not an autotune knob.
    HALF_FUSED_CORR_SPLIT_P: int = 0


def _validate_cfg_d256(cfg: CfgD256) -> None:
    """Per-tensor FP8 role topology; independent of half and block-scale tuning."""
    split_p = cfg.SOFTMAX_WARPGROUPS == 2
    split_p_supported = cfg.MASK_FLAGS == MASK_NONE or ((cfg.MASK_FLAGS & ~MASK_PADDED) == MASK_CAUSAL and cfg.BOTTOM_RIGHT == 0)
    checks = (
        (cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2), "per-tensor FP8 D256 requires E4M3/E5M2 inputs"),
        (not (cfg.FUSED_CORR_SPLIT_P or cfg.HALF_FUSED_CORR_SPLIT_P or cfg.EPILOGUE_GATE), "per-tensor FP8 D256 has no fused half/MXFP8 correction or gate"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d256: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_REGS + (cfg.SOFTMAX_WG1_REGS if split_p else 0) <= 512, "d256: register budget over 512"),
        (
            all(value % 8 == 0 for value in (cfg.MMA_REGS, cfg.CORRECTION_REGS, cfg.SOFTMAX_REGS, cfg.SOFTMAX_WG1_REGS)),
            "d256: per-role regs must be multiples of 8",
        ),
        (cfg.CGA_M == cfg.CTA_MMA == 1, "per-tensor FP8 D256 requires one CTA per cluster"),
        (cfg.STAGES_KV == 2, "d256 SM100 uses two full/half-width KV stages"),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d256: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d256: V/O swizzle out of range"),
        (cfg.TILES_Q == 1, "d256 pipeline mandates TILES_Q == 1"),
        (not split_p or split_p_supported, "d256: unsupported split-P specialization"),
        (cfg.SOFTMAX_WARPGROUPS == 2 if cfg.MASK_FLAGS == MASK_NONE else True, "d256: dense FP8 must split P generation"),
        (cfg.CORRECTION_WARPS == 4 and cfg.TOTAL_WARPS == (16 if split_p else 12), "d256: role layout and warp count disagree"),
        (cfg.TILE_K_HW_BMM1 == cfg.TILE_K_HW_BMM2 == 32, "per-tensor FP8 D256 requires K32 QMMA"),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    """One FP8 body, including its existing masked and split-P local variants."""
    _validate_params("d256", params)
    if params.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2):
        raise ValueError("per-tensor FP8 D256 requires E4M3/E5M2 inputs")
    if params.cta_mma != 1:
        raise ValueError(f"d256: FP8/MXFP8 requires cta_mma=1; got {params.cta_mma}")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    mask_flags = _mask_flags_from(params)
    plain_top_left_causal = (mask_flags & ~MASK_PADDED) == MASK_CAUSAL and not params.bottom_right and not params.window_left and not params.window_right
    split_p = mask_flags == MASK_NONE or plain_top_left_causal
    padded_split_p = plain_top_left_causal and bool(mask_flags & MASK_PADDED)
    e4_causal_regs = params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_CAUSAL and not params.bottom_right
    e5_causal_regs = params.dtype_qkv == DTYPE_E5M2 and mask_flags == MASK_CAUSAL and not params.bottom_right

    # Register profiles belong to the body's role topology, not public knobs.
    softmax_regs, softmax_wg1_regs, correction_regs = 240, 136 if split_p else 40, 96
    if padded_split_p:
        softmax_wg1_regs, correction_regs = 144, 88
    elif e4_causal_regs:
        correction_regs = 64
        if split_p:
            softmax_regs, softmax_wg1_regs = 216, 168
    elif e5_causal_regs:
        correction_regs = 96 if split_p else 112
    elif mask_flags != MASK_NONE:
        correction_regs = 104
    elif params.dtype_qkv == DTYPE_E4M3:
        correction_regs = 88

    if split_p:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id, read_tile_arrivers = 16, 4, 8, 12, 15
    else:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id, read_tile_arrivers = 12, 64, 4, 8, 11

    cfg = CfgD256(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=1,
        CTA_MMA=1,
        Q_SWZ_BYTES=q_swz_bytes(256, b),
        K_SWZ_BYTES=q_swz_bytes(256, b),
        V_SWZ_BYTES=v_swz_bytes(256, 1, b),
        O_SWZ_BYTES=o_swz_bytes(256, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=32,
        TILE_K_HW_BMM2=32,
        STAGES_KV=2,
        SOFTMAX_WARPGROUPS=2 if split_p else 1,
        CORRECTION_WARPS=4,
        FUSED_CORR_SPLIT_P=0,
        HALF_FUSED_CORR_SPLIT_P=0,
        MMA_REGS=40,
        TMALDG_REGS=40,
        TMASTG_REGS=40,
        SCHEDULER_REGS=40,
        OTHER_REGS=40,
        SOFTMAX_REGS=softmax_regs,
        SOFTMAX_WG1_REGS=softmax_wg1_regs,
        CORRECTION_REGS=correction_regs,
        TOTAL_WARPS=total_warps,
        THREADS_PER_CTA=total_warps * 32,
        SOFTMAX_WG1_BASE=softmax_wg1_base,
        CORR_WARP_BASE=correction_warp_base,
        MMA_WARP_ID=mma_warp_id,
        TMALDG_WARP_ID=mma_warp_id + 1,
        TMASTG_WARP_ID=mma_warp_id + 2,
        SCHED_WARP_ID=mma_warp_id + 3,
        READ_TILE_ARRIVERS=read_tile_arrivers,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        L2_SIZE_MIB=params.lpt_l2_size_mib or 60,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD256.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        EPILOGUE_GATE=0,
        GATE_BPE=2,
    )
    _validate_cfg_d256(cfg)
    return cfg, _tma_iters(cfg)
