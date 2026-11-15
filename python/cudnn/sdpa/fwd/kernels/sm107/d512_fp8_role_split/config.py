# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Configuration owned by the SM107 per-tensor FP8 D512 role-split body."""

from dataclasses import dataclass
from cudnn.sdpa.fwd.config_common import TemplateParams
from cudnn.sdpa.fwd.config_sm107 import (
    _validate_params,
    _validate_dtype_k_step,
    _validate_regs,
    _validate_warp_layout,
    _validate_swizzles,
    _validate_masks,
    _band_fields,
    _check,
    _tma_iters,
    resolve_dtype_o,
    bpe,
    tile_k_hw,
    rescale_threshold,
    q_swz_bytes,
    v_swz_bytes,
    o_swz_bytes,
)
from cudnn.frost.tile_dsl.constants import (
    DTYPE_E4M3 as _DTYPE_E4M3,
    DTYPE_E5M2 as _DTYPE_E5M2,
    DTYPE_FP16 as _DTYPE_FP16,
    MASK_NONE,
    SCHED_NATURAL,
    SCHED_LPT,
)


@dataclass(frozen=True)
class CfgD512:
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
    TILE_K: int = 128
    TILE_O: int = 128
    DTYPE_QKV: int = _DTYPE_FP16
    DTYPE_O: int = _DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2
    O_BLOCK_SCALE: int = 0
    O_PACK_DIV: int = 1
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
    TILES_Q: int = 2
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 4
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
    BOTTOM_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    PACK_GQA: int = 0
    PACK_G: int = 0
    QH_PER_KH: int = 1
    SPLIT_KV: int = 1
    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0
    THD_VARLEN: int = 0
    PAGED_KV: int = 0
    PAGE_SIZE: int = 0
    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL
    N_BMM2_CHUNKS: int = 2
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
    EPILOGUE_GATE: int = 0
    GATE_BPE: int = 2
    XFER_STAGES: int = 2
    BMM2_N_PER_CALL: int = 256
    BMM2_LOOP_N_BLOCKS: int = 2


_D512_TOTAL_WARPS = 8
_D512_CGA_M, _D512_CGA_N, _D512_CTA_MMA = 4, 1, 2
_D512_READ_TILE_ARRIVERS = (
    ((1 * 4) + 1) * (_D512_CGA_M * _D512_CGA_N)
    + (_D512_CGA_M * _D512_CGA_N) // (2 * _D512_CTA_MMA)
    + (_D512_CGA_M * _D512_CGA_N) // 2
    + _D512_CTA_MMA * _D512_CGA_N
)


def _make_cfg_d512_family(params: TemplateParams, *, flavor: str, mxfp8: bool):
    split_wired = not mxfp8 and params.dtype_qkv in (_DTYPE_E4M3, _DTYPE_E5M2)
    _validate_params(flavor, params, split_wired=split_wired)
    if (params.split_kv or 1) > 1 and params.thd_varlen:
        raise ValueError(f"{flavor}: per-tensor FP8 split-KV is dense-only")
    if not mxfp8 and params.dtype_qkv in (_DTYPE_E4M3, _DTYPE_E5M2):
        if params.sched_policy not in (None, SCHED_NATURAL, SCHED_LPT):
            raise ValueError(f"{flavor}: per-tensor FP8 supports NATURAL/LPT schedulers only")
        if params.pack_gqa:
            if params.thd_varlen:
                raise ValueError(f"{flavor}: per-tensor FP8 PackGQA is dense-only")
            if 128 % params.qh_per_kh:
                raise ValueError(f"{flavor}: PackGQA requires a whole head group dividing TILE_M=128; got {params.qh_per_kh}")
    if params.cta_mma != _D512_CTA_MMA:
        raise ValueError(f"{flavor}: the role-split pipeline is cga4x1 / CTA_MMA=2 only (got cta_mma={params.cta_mma})")
    dtype_o = resolve_dtype_o(params)
    b, b_o = bpe(params.dtype_qkv), bpe(dtype_o)
    tile_k = tile_o = 512
    tile_n = 128
    is_fp8 = params.dtype_qkv <= _DTYPE_E5M2
    # FP8 (BPE=1) fits a deeper K/V ring and a 4-deep cross-sg P ring; f16/bf16
    # (BPE=2) are SMEM-cap driven down to 2/2.
    stages_kv = 3 if is_fp8 else 2
    xfer_stages = 4 if is_fp8 else 2
    mask_flags, win_l, win_r, bottom_right, has_sink = _band_fields(params)

    cfg = CfgD512(
        TILE_M=128,
        TILE_N=tile_n,
        TILE_K=tile_k,
        TILE_O=tile_o,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        CGA_M=_D512_CGA_M,
        CGA_N=_D512_CGA_N,
        CTA_MMA=_D512_CTA_MMA,
        SPLIT_PIPELINE=1,
        Q_SWZ_BYTES=q_swz_bytes(tile_k, b),
        K_SWZ_BYTES=q_swz_bytes(tile_k, b),
        V_SWZ_BYTES=v_swz_bytes(tile_o, _D512_CTA_MMA, b),
        O_SWZ_BYTES=o_swz_bytes(tile_o, b_o),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        TILES_Q=1,
        SCHEDULER_STAGES=2,
        STAGES_KV=stages_kv,
        XFER_STAGES=xfer_stages,
        # One 4-warp compute group; correction is fused into it, sub-group
        # conditional, so there are no separate correction warps or registers.
        SOFTMAX_WARPGROUPS=1,
        CORRECTION_WARPS=0,
        SOFTMAX_REGS=240,
        CORRECTION_REGS=0,
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=win_l,
        WINDOW_RIGHT=win_r,
        BOTTOM_RIGHT=bottom_right,
        HAS_SINK=has_sink,
        STATS_LOG2=int(params.stats_log2),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=params.qh_per_kh,
        SPLIT_KV=params.split_kv or 1,
        SEQ_KV_LENS_PRESENT=1 if params.thd_varlen else int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=params.page_size if params.paged_kv else 0,
        SCHEDULER_POLICY=params.sched_policy if params.sched_policy is not None else SCHED_NATURAL,
        BMM2_N_PER_CALL=256,
        BMM2_LOOP_N_BLOCKS=tile_o // 256,
        N_BMM2_CHUNKS=2,
        BMM2_CHUNK_SIZE=256,
        TOTAL_WARPS=_D512_TOTAL_WARPS,
        THREADS_PER_CTA=_D512_TOTAL_WARPS * 32,
        SOFTMAX_WG0_BASE=0,
        SOFTMAX_WG1_BASE=0,  # unused under the role split
        CORR_WARP_BASE=0,  # correction fused with the compute group
        MMA_WARP_ID=4,
        TMALDG_WARP_ID=5,
        TMASTG_WARP_ID=6,
        SCHED_WARP_ID=7,
        READ_TILE_ARRIVERS=_D512_READ_TILE_ARRIVERS,
    )

    _validate_regs(cfg, flavor)
    _validate_warp_layout(cfg, flavor, expect_total=_D512_TOTAL_WARPS, expect_arrivers=_D512_READ_TILE_ARRIVERS)
    _validate_dtype_k_step(cfg, flavor)
    _validate_swizzles(cfg, flavor)
    _validate_masks(cfg, flavor)
    _check([(cfg.SPLIT_PIPELINE == 1, f"{flavor}: the d512 flavor is role-split (SPLIT_PIPELINE=1)")])
    return cfg, _tma_iters(cfg)


def make_cfg_d512(params: TemplateParams):
    if params.dtype_qkv not in (_DTYPE_E4M3, _DTYPE_E5M2):
        raise ValueError("SM107 D512 per-tensor FP8 requires E4M3/E5M2 inputs")
    if params.mma_2x2:
        raise ValueError("SM107 FP8 role-split configuration does not select the 2x2 implementation")
    return _make_cfg_d512_family(params, flavor="sm107 d512", mxfp8=False)
