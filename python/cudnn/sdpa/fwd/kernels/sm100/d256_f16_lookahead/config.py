# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Configuration owned by the SM100 D256 half single-Q lookahead body.

FP16/BF16 share this pipeline. Quantized D256 configuration remains separate;
neither its role defaults nor its register tuning can change this body.
"""

from dataclasses import dataclass
from typing import Tuple

from cudnn.frost.tile_dsl.constants import (
    DTYPE_BF16,
    DTYPE_E4M3,
    DTYPE_E5M2,
    DTYPE_FP16,
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    SCHED_NATURAL,
)
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


def d256_gate_smem_bytes(cfg: CfgD256) -> int:
    """The SMEM the GATED d256 half rendering allocates, from the SAME constants its arrays are declared with:
    sQO (Q u O alias) + the K ring + the V ring + the 64 KiB gate staging tile + the fixed overhead (barriers, the
    scheduler ring, the TMEM pointer, the 1024-B alignment pad of the first array; 2 KiB, config_sm107's figure)."""
    operands = (
        cfg.TILE_M * cfg.TILE_K + cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K // cfg.CTA_MMA) + cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N // cfg.CTA_MMA)
    ) * cfg.BPE
    gate = cfg.TILES_Q * cfg.TILE_M * cfg.TILE_O * cfg.GATE_BPE if cfg.EPILOGUE_GATE else 0
    return operands + gate + 2 * 1024


_D256_GATE_SMEM_USABLE_BYTES = 320 * 1024


def _validate_cfg_d256(cfg: CfgD256) -> None:
    """Consistency checks on the (mostly hardcoded) d256 geometry."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    split_p = cfg.SOFTMAX_WARPGROUPS == 2
    fused_corr_split_p = cfg.FUSED_CORR_SPLIT_P == 1
    split_p_supported = fp8 and (cfg.MASK_FLAGS == MASK_NONE or ((cfg.MASK_FLAGS & ~MASK_PADDED) == MASK_CAUSAL and cfg.BOTTOM_RIGHT == 0))
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d256: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (
            cfg.MMA_REGS + (0 if fused_corr_split_p else cfg.CORRECTION_REGS) + cfg.SOFTMAX_REGS + (cfg.SOFTMAX_WG1_REGS if split_p else 0) <= 512,
            "d256: register budget over 512",
        ),
        (
            cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0 and cfg.SOFTMAX_WG1_REGS % 8 == 0,
            "d256: per-role regs must be multiples of 8",
        ),
        (
            not cfg.HALF_FUSED_CORR_SPLIT_P
            or (
                cfg.DTYPE_QKV == DTYPE_BF16
                and cfg.MASK_FLAGS == MASK_CAUSAL
                and not cfg.BOTTOM_RIGHT
                and not cfg.WINDOW_LEFT
                and not cfg.WINDOW_RIGHT
                and not cfg.THD_VARLEN
                and not cfg.PAGED_KV
                and not cfg.EPILOGUE_GATE
                and cfg.SPLIT_KV == 1
                and not split_p
                and not fused_corr_split_p
                and cfg.CORRECTION_WARPS == 4
            ),
            "d256: half fused correction requires dense unsplit BF16 top-left causal",
        ),
        (cfg.CGA_M == cfg.CTA_MMA, "d256 flavor pairs CGA_M with CTA_MMA"),
        (cfg.CTA_MMA == (1 if fp8 else 2), "d256 SM100: FP8 requires CTA1; BF16/FP16 requires CTA2"),
        (cfg.STAGES_KV == 2, "d256 SM100 uses two full/half-width KV stages"),
        (
            not cfg.EPILOGUE_GATE or (not fp8 and cfg.SPLIT_KV == 1 and cfg.GATE_BPE == cfg.BPE == 2 and cfg.TILES_Q == 1),
            "d256: the epilogue gate is the half body's unsplit arm (GATE_BPE == BPE == 2, TILES_Q == 1: one gate tile per Q tile)",
        ),
        (
            not cfg.EPILOGUE_GATE or d256_gate_smem_bytes(cfg) <= _D256_GATE_SMEM_USABLE_BYTES,
            f"d256: the gated rendering's SMEM ({d256_gate_smem_bytes(cfg) // 1024} KiB: Q/K/V/O + the 64 KiB gate tile + 2 KiB fixed) exceeds the "
            f"{_D256_GATE_SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d256: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d256: V/O swizzle out of range"),
        (cfg.TILES_Q == 1, "d256 pipeline mandates TILES_Q == 1"),
        (
            not fused_corr_split_p or not (cfg.MASK_FLAGS & MASK_SWA),
            "d256/d512: SWA P reuse and split-P correction require different alias phases",
        ),
        (not split_p or split_p_supported, "d256: unsupported split-P specialization"),
        (cfg.SOFTMAX_WARPGROUPS == 2 if cfg.MASK_FLAGS == MASK_NONE and fp8 else True, "d256: dense FP8 must split P generation"),
        (cfg.TOTAL_WARPS == (12 if fused_corr_split_p else 16 if split_p else 12), "d256: role layout and warp count disagree"),
        (not fused_corr_split_p or (split_p and cfg.CORRECTION_WARPS == 0), "d256: fused split-P must replace the correction warp group"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if fp8 else 16) and cfg.TILE_K_HW_BMM2 == (32 if fp8 else 16),
            "d256: TILE_K_HW must be 32 for FP8 and 16 for BF16/FP16",
        ),
        (fp8 or cfg.DTYPE_O == cfg.DTYPE_QKV, "d256: half input requires DTYPE_O == DTYPE_QKV"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    """Build the half lookahead geometry; no dispatch or performance choice."""
    _validate_params("d256", params)
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("d256 lookahead: f16/bf16 inputs only")
    if params.cta_mma != 2:
        raise ValueError(f"d256: BF16/FP16 requires cta_mma=2; got {params.cta_mma}")
    mask_flags = _mask_flags_from(params)
    plain_top_left_causal = (mask_flags & ~MASK_PADDED) == MASK_CAUSAL and not params.bottom_right and not params.window_left and not params.window_right
    fused = (
        params.dtype_qkv == DTYPE_BF16
        and mask_flags == MASK_CAUSAL
        and plain_top_left_causal
        and not params.thd_varlen
        and not params.paged_kv
        and not params.epilogue_gate
        and params.split_kv == 1
    )
    # Preserve the existing padded top-left register profile, including its
    # unused WG1 tracing placeholder. Pipeline retuning is outside this move.
    padded_top_left = plain_top_left_causal and bool(mask_flags & MASK_PADDED)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    producer_regs = 48 if fused else 40
    cfg = CfgD256(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=bpe(params.dtype_qkv),
        BPE_O=bpe(dtype_o),
        Q_SWZ_BYTES=q_swz_bytes(256, bpe(params.dtype_qkv)),
        K_SWZ_BYTES=q_swz_bytes(256, bpe(params.dtype_qkv)),
        V_SWZ_BYTES=v_swz_bytes(256, 2, bpe(params.dtype_qkv)),
        O_SWZ_BYTES=o_swz_bytes(256, bpe(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        HALF_FUSED_CORR_SPLIT_P=int(fused),
        MMA_REGS=producer_regs,
        TMALDG_REGS=producer_regs,
        TMASTG_REGS=producer_regs,
        SCHEDULER_REGS=producer_regs,
        OTHER_REGS=producer_regs,
        SOFTMAX_WG1_REGS=144 if padded_top_left else 40,
        CORRECTION_REGS=216 if fused else 88 if padded_top_left else 96,
        SOFTMAX_WG1_BASE=64,
        MASK_FLAGS=mask_flags,
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        L2_SIZE_MIB=params.lpt_l2_size_mib or 60,
        SEQ_KV_LENS_PRESENT=int(params.thd_varlen or params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD256.TILE_M, partial=True),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
        EPILOGUE_GATE=int(params.epilogue_gate),
    )
    _validate_cfg_d256(cfg)
    return cfg, _tma_iters(cfg)
