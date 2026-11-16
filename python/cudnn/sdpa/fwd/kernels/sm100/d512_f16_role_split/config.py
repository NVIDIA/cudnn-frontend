# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM100 half D512 role-split configuration, independent of FP8 tuning."""

from dataclasses import dataclass
from typing import Tuple
from cudnn.frost.tile_dsl.constants import DTYPE_FP16, DTYPE_BF16, DTYPE_E4M3, DTYPE_E5M2, MASK_NONE, SCHED_NATURAL
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
class CfgD512:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 512
    TILE_O: int = 512

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    CGA_M: int = 4
    CGA_N: int = 1
    CTA_MMA: int = 2

    SPLIT_PIPELINE: int = 1

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 1
    SCHEDULER_STAGES: int = 2
    STAGES_KV: int = 2
    XFER_STAGES: int = 2

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 0

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 0
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

    BMM2_N_PER_CALL: int = 256
    BMM2_LOOP_N_BLOCKS: int = 512 // 256
    N_BMM2_CHUNKS: int = 2
    BMM2_CHUNK_SIZE: int = 256

    TOTAL_WARPS: int = 8
    THREADS_PER_CTA: int = 8 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 0
    CORR_WARP_BASE: int = 0
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TMASTG_WARP_ID: int = 6
    SCHED_WARP_ID: int = 7

    ONE_LANE: int = 1
    ONE_WARP: int = 32
    SOFTMAX_LANES: int = 128
    CORR_LANES: int = 128
    SOFTMAX_PLUS_CORR: int = 256

    READ_TILE_ARRIVERS: int = ((1 * 4) + 1) * (4 * 1) + (4 * 1) // (2 * 2) + (4 * 1) // 2 + 2 * 1

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


def _validate_cfg_d512(cfg: CfgD512) -> None:
    """Consistency checks on the (mostly hardcoded) d512 geometry."""
    _fp8 = cfg.DTYPE_QKV <= DTYPE_E5M2
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d512: MMA/TMALDG/TMASTG/Scheduler regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d512: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d512: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 4 and cfg.CGA_N == 1 and cfg.CTA_MMA == 2, "d512 flavor: cga4x1 / CTA_MMA=2 only"),
        (cfg.CGA_M // cfg.CTA_MMA == 2, "d512 flavor: exactly two sub-groups (CGA_M / CTA_MMA == 2)"),
        (cfg.TILE_K == 512 and cfg.TILE_O == 512, "d512: d_qk = d_v = 512"),
        (cfg.TILES_Q == 1, "d512 (role-split): TILES_Q must be 1"),
        (cfg.SOFTMAX_WARPGROUPS == 1, "d512 (role-split): SOFTMAX_WARPGROUPS must be 1"),
        (cfg.CORRECTION_WARPS == 0, "d512 (role-split): CORRECTION_WARPS must be 0"),
        (cfg.READ_TILE_ARRIVERS == 25, f"d512 cga4x1: expected READ_TILE_ARRIVERS=25, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128, "d512: Q/K/V swizzle must all be 128B"),
        (cfg.O_SWZ_BYTES == 128, "d512: O swizzle must be 128B"),
        # Both dtype families run the SM10x 1-chunk MMA step: f16 at K=16,
        # FP8 at the K=32 QMMA (k_dim=0).  The Rubin K=64 2-chunk fast path is
        # silently WRONG on Blackwell — see rules/mma-tma-matrix.md § 1.
        (
            cfg.TILE_K_HW_BMM1 == (32 if _fp8 else 16) and cfg.TILE_K_HW_BMM2 == (32 if _fp8 else 16),
            "d512: TILE_K_HW must be 32 (fp8 K=32 QMMA) / 16 (f16, 1-chunk on SM10x — 2-chunk silently wrong)",
        ),
        # SMEM-cap driven: the K/V ring costs STAGES_KV * 64 KiB at f16 and
        # STAGES_KV * 32 KiB at FP8, so FP8 buys a third stage under the same
        # 227 KiB cap.
        (cfg.STAGES_KV == (3 if _fp8 else 2), "d512 (SM100): STAGES_KV must be 3 (fp8) / 2 (f16) — SMEM-cap driven"),
        # TMEM-cap driven (512 cols on Blackwell): the sg0 carve is
        # XFER_STAGES * TILE_N (S_acc parities) + TILE_K / (4 // BPE) (Q, moved
        # to TMEM by UTCCP).  f16: 2*128 + 256 = 512.  FP8 packs 4 elems per
        # 4-byte column, so Q costs only 128 cols and a third parity fits:
        # 3*128 + 128 = 512.
        (cfg.XFER_STAGES == (3 if _fp8 else 2), "d512 (SM100): XFER_STAGES must be 3 (fp8) / 2 (f16) — TMEM-cap driven"),
        (
            cfg.XFER_STAGES * cfg.TILE_N + cfg.TILE_K // (4 // cfg.BPE) == 512,
            f"d512 (SM100): sg0 TMEM carve (S_acc {cfg.XFER_STAGES * cfg.TILE_N} + Q {cfg.TILE_K // (4 // cfg.BPE)}) must be exactly 512 cols",
        ),
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16) if _fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d512: DTYPE_O must equal DTYPE_QKV for half input; fp8 allows an independent output dtype",
        ),
        (not cfg.PAGED_KV or cfg.SEQ_KV_LENS_PRESENT == 1, "d512: paged KV requires per-batch KV lengths"),
        (
            not cfg.PAGED_KV or (cfg.PAGE_SIZE >= 8 and cfg.PAGE_SIZE % 8 == 0 and (128 % cfg.PAGE_SIZE == 0 or cfg.PAGE_SIZE % 128 == 0)),
            f"d512: page_size must be a multiple of 8 that divides the 128-row KV tile or is a multiple of it; got {cfg.PAGE_SIZE}",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    if getattr(params, "mma_2x2", False):
        raise ValueError("D512 role-split cannot lower a 2x2 implementation record")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("D512 half role-split requires BF16/FP16 inputs")
    _validate_params("d512", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= DTYPE_E5M2  # E4M3/E5M2 inputs → the fp8 kernel file
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    # FP8 pins the Blackwell K=32 QMMA path.  NOT tile_k_hw(), which returns the
    # Rubin K=64 answer (see mma-tma-matrix.md § 1 "latent trap in the shared
    # helper") — k_dim=0 with TILE_K_HW=64 is the silently-wrong combination.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    cfg = CfgD512(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        Q_SWZ_BYTES=q_swz_bytes(512, b),
        K_SWZ_BYTES=q_swz_bytes(512, b),
        V_SWZ_BYTES=v_swz_bytes(512, 2, b),
        O_SWZ_BYTES=o_swz_bytes(512, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=tile_k_hw_fp8,
        # FP8 halves the K/V ring and the TMEM-resident Q, buying a third KV
        # stage and a third S_acc parity under the same SMEM / TMEM caps.
        STAGES_KV=3 if fp8 else 2,
        XFER_STAGES=3 if fp8 else 2,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PACK_G=_pack_g(params, CfgD512.TILE_M, partial=False),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d512(cfg)
    return cfg, _tma_iters(cfg)
