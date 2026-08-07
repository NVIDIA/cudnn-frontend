# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel configuration for the Frost SM100 DSL SDPA flavors.

The per-flavor tile geometry is fixed here (``d512``, ``d256``);
the per-graph compile-time parameters (dtype, mask, sink, ...) arrive as a
:class:`TemplateParams` instance, built by the adapter in
:mod:`cudnn.sdpa.fwd.api_dsl`. These are graph-derived semantics plus the
chosen tuning-knob values — see the Facts / Capabilities / Knobs section of
python/cudnn/frost/README.md. Nothing in this
module reads environment variables; a kernel template receives its ``TemplateParams``
via the loader (see ``cudnn.frost.template_loader.load_template``) and calls :func:`make_cfg`.

All configuration errors raise :class:`ValueError`. Anything a *user-built
graph* could trip must be rejected earlier, by ``graph_analyzer.probe`` /
``api_dsl.SdpaFwdDslSm100.check_support`` — a ``ValueError`` from here means
those support checks have a gap.
"""

from __future__ import annotations

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
    SCHED_LPT,
    SCHED_NATURAL,
)


@dataclass(frozen=True)
class TemplateParams:
    """Per-graph compile-time parameters threaded into a kernel template.

    Design notes (see also python/cudnn/frost/README.md):

    - Contents: graph-derived semantics (dtype, masks, sink/padded/THD) plus
      the chosen values of true tuning knobs (sched_policy). Shapes are
      deliberately NOT here — they are handled by the template's per-shape
      ``compile()`` cache; this record holds only what changes the *traced
      code* of the kernel.
    - Direction: this is the OUTPUT of a successful eligibility match, never
      an input to it. The probe compares facts and requested knobs against
      an engine's Capabilities; only after that passes does the engine's
      ``lower`` hook assemble this record.
    - Frozen + hashable on purpose: it doubles as the kernel-module cache key
      in frost.template_loader — one distinct TemplateParams == one distinct
      compiled specialization, and identical params reuse the compiled one.
    - Validation here (``_validate_params`` and the ``make_cfg_*`` checks)
      raises ValueError but is a BACKSTOP: every reachable violation must be
      rejected earlier by a Capabilities row; tripping it means that row is
      dishonest, not that a user erred.
    """

    dtype_qkv: int = DTYPE_FP16  # E4M3/E5M2 (0/1, d128 MXFP8 only) or BF16/FP16 (2/3)
    dtype_o: int = -1  # output dtype (0..3); -1 = inherit dtype_qkv. MXFP8 writes BF16/FP16.
    mask_flags: int = MASK_NONE
    swa_window: int = 0  # runtime left-window offset W (keep kv in [q-W, q])
    causal_bottom_right: bool = False
    has_sink: bool = False
    seq_kv_lens_present: bool = False
    # Dense padded-Q trim: per-batch seq_len_q is a SEPARATE (B,)-int32
    # kernel parameter (seq_q_lens_tensor — mirrors cuDNN's distinct SEQLEN_Q
    # pointer / FA's seqused_q; compiled into the signature only under this
    # flag); q rows >= seq_len_q[b] write O := 0 / LSE := -inf (cuDNN >= 9.14
    # convention). Dense-only — THD carries per-sequence Q lengths via
    # cu_seqlens instead.
    seq_q_lens_present: bool = False
    sched_policy: int = SCHED_NATURAL
    thd_varlen: bool = False
    # cc10.3+ fuses the S_acc row-max into the LDTM (tcgen05.ld.red.f32.max); cc10.0
    # lacks it and uses the manual load + software reduction. Auto-set from the device
    # capability at compile time (MXFP8 only; the f16/fp8 kernels do not read it).
    fused_ldtm_stat: bool = False


def _validate_params(flavor: str, k: TemplateParams) -> None:
    if k.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: DTYPE_QKV must be E4M3/E5M2/BF16/FP16 (0..3); got {k.dtype_qkv}")
    fp8 = k.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if fp8 and flavor != "d128":
        raise ValueError(f"{flavor}: FP8/MXFP8 inputs (DTYPE_QKV 0/1) are only supported on d128")
    dtype_o = k.dtype_qkv if k.dtype_o < 0 else k.dtype_o
    if dtype_o not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: DTYPE_O must be 0..3; got {dtype_o}")
    if not fp8 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires DTYPE_O == DTYPE_QKV; got dtype_o={dtype_o}")
    if k.causal_bottom_right:
        if not (k.mask_flags & MASK_CAUSAL):
            raise ValueError(f"{flavor}: CAUSAL_BOTTOM_RIGHT requires MASK_CAUSAL (bit 1)")
        if k.mask_flags & MASK_SWA:
            raise ValueError(f"{flavor}: CAUSAL_BOTTOM_RIGHT + SWA is not supported")
    if k.thd_varlen:
        if not (k.mask_flags & MASK_PADDED):
            raise ValueError(f"{flavor}: THD/varlen implies per-sequence padded masking (MASK_PADDED)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: THD/varlen requires SEQ_KV_LENS_PRESENT")
    if k.seq_q_lens_present:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    if k.sched_policy not in (SCHED_NATURAL, SCHED_LPT):
        raise ValueError(f"{flavor}: only SCHED_NATURAL (0) / SCHED_LPT (1) are wired up; got {k.sched_policy}")


# ---------------------------------------------------------------------------
# Shared derivation helpers
# ---------------------------------------------------------------------------


def bpe(dtype: int) -> int:
    if dtype <= 1:
        return 1
    return 2


def tile_k_hw(dtype_qkv: int) -> int:
    if dtype_qkv <= 1:
        return 64
    return 16


def q_swz_bytes(tile_k: int, bpe_val: int) -> int:
    return 128 if (tile_k * bpe_val) % 128 == 0 else 64


def v_swz_bytes(tile_o: int, cta_mma: int, bpe_val: int) -> int:
    inner = (tile_o // cta_mma) * bpe_val
    if inner % 128 == 0:
        return 128
    if inner % 64 == 0:
        return 64
    if inner % 32 == 0:
        return 32
    raise ValueError(f"V inner bytes {inner} not multiple of 32/64/128")


def o_swz_bytes(tile_o: int, bpe_o: int) -> int:
    return 128 if (tile_o * bpe_o) % 128 == 0 else 64


def rescale_threshold(dtype_qkv: int) -> float:
    return 4.0 if dtype_qkv <= 1 else 8.0


def _tma_iters_for(d_elems: int, bpe_val: int, swz_b: int) -> int:
    inner_bytes = d_elems * bpe_val
    if inner_bytes % swz_b != 0:
        raise ValueError(f"inner {inner_bytes} not multiple of swz {swz_b}")
    return inner_bytes // swz_b


@dataclass(frozen=True)
class TmaIters:
    """TMA iteration granularity derived from a flavor's tile geometry."""

    QK_ITERS: int
    VO_ITERS: int
    QK_GRANU_ELEMS: int
    VO_GRANU_ELEMS: int


def _tma_iters(cfg) -> TmaIters:
    qk = _tma_iters_for(cfg.TILE_K, cfg.BPE, cfg.Q_SWZ_BYTES)
    vo = _tma_iters_for(cfg.TILE_O, cfg.BPE, cfg.V_SWZ_BYTES)
    return TmaIters(
        QK_ITERS=qk,
        VO_ITERS=vo,
        QK_GRANU_ELEMS=cfg.TILE_K // qk,
        VO_GRANU_ELEMS=cfg.TILE_O // vo,
    )


# ---------------------------------------------------------------------------
# d256 flavor — d_qk = d_v = 256, SM100 (Blackwell), cga2 (Qwen-class models)
# ---------------------------------------------------------------------------


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

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 96
    MMA_REGS: int = 40
    TMALDG_REGS: int = 40
    TMASTG_REGS: int = 40
    SCHEDULER_REGS: int = 40
    OTHER_REGS: int = 40

    RESCALE_THRESHOLD: float = 8.0

    MASK_FLAGS: int = MASK_NONE
    SWA_WINDOW: int = 0
    HAS_SINK: int = 0
    CAUSAL_BOTTOM_RIGHT: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    SOFTMAX_WG0_BASE: int = 0
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


def _validate_cfg_d256(cfg: CfgD256) -> None:
    """Consistency checks on the (mostly hardcoded) d256 geometry."""
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d256: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d256: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d256: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA, "d256 flavor pairs CGA_M with CTA_MMA"),
        (cfg.CTA_MMA == 2, "d256 SM100 is cga2-only (CTA_MMA must be 2)"),
        (cfg.STAGES_KV == 2, "d256 SM100 STAGES_KV: f16/bf16 -> 2 (192 KiB SMEM budget)"),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d256: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d256: V/O swizzle out of range"),
        (cfg.TILES_Q == 1, "d256 pipeline mandates TILES_Q == 1"),
        (cfg.SOFTMAX_WARPGROUPS == 1, "d256 pipeline mandates SOFTMAX_WARPGROUPS == 1"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d256: DTYPE_O must equal DTYPE_QKV"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    _validate_params("d256", params)
    b = bpe(params.dtype_qkv)
    cfg = CfgD256(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=params.dtype_qkv,
        BPE=b,
        BPE_O=b,
        Q_SWZ_BYTES=q_swz_bytes(256, b),
        K_SWZ_BYTES=q_swz_bytes(256, b),
        V_SWZ_BYTES=v_swz_bytes(256, 2, b),
        O_SWZ_BYTES=o_swz_bytes(256, b),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=params.mask_flags,
        SWA_WINDOW=params.swa_window,
        HAS_SINK=int(params.has_sink),
        CAUSAL_BOTTOM_RIGHT=int(params.causal_bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
    )
    _validate_cfg_d256(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d512 flavor — d_qk = d_v = 512, SM100 (Blackwell), cga4x1 role-split (DSv4-class models)
# ---------------------------------------------------------------------------


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

    MASK_FLAGS: int = MASK_NONE
    SWA_WINDOW: int = 0
    HAS_SINK: int = 0
    CAUSAL_BOTTOM_RIGHT: int = 0

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


def _validate_cfg_d512(cfg: CfgD512) -> None:
    """Consistency checks on the (mostly hardcoded) d512 geometry."""
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
        (cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128, "d512: Q/K/V/O swizzle must all be 128B"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d512 f16: TILE_K_HW must be 16 (1-chunk only on SM10x — 2-chunk silently wrong)"),
        (cfg.STAGES_KV == 2, "d512 f16 (SM100): STAGES_KV must be 2 (SMEM-cap driven)"),
        (cfg.XFER_STAGES == 2, "d512 f16 (SM100): XFER_STAGES must be 2 (TMEM-cap driven: 2*128 S_acc + 256 Q = 512)"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d512: DTYPE_O must equal DTYPE_QKV for half input"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
    _validate_params("d512", params)
    b = bpe(params.dtype_qkv)
    cfg = CfgD512(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=params.dtype_qkv,
        BPE=b,
        BPE_O=b,
        Q_SWZ_BYTES=q_swz_bytes(512, b),
        K_SWZ_BYTES=q_swz_bytes(512, b),
        V_SWZ_BYTES=v_swz_bytes(512, 2, b),
        O_SWZ_BYTES=o_swz_bytes(512, b),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=params.mask_flags,
        SWA_WINDOW=params.swa_window,
        HAS_SINK=int(params.has_sink),
        CAUSAL_BOTTOM_RIGHT=int(params.causal_bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
    )
    _validate_cfg_d512(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d128 flavor — d_qk = d_v = 128, SM100 (Blackwell), cga2 (Llama-class models)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD128:
    TILE_M: int = 128
    TILE_N: int = 128
    TILE_K: int = 128
    TILE_O: int = 128

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    CGA_M: int = 2
    CGA_N: int = 1
    CTA_MMA: int = 2

    # Q∪O SMEM alias — unused on llama (d_qk=d_v=128 fits without it at cga2).
    QO_ALIAS: int = 0

    SPLIT_PIPELINE: int = 0

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    O_SWZ_BYTES: int = 128

    TILE_K_HW_BMM1: int = 16
    TILE_K_HW_BMM2: int = 16

    TILES_Q: int = 2
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
    SWA_WINDOW: int = 0
    HAS_SINK: int = 0
    CAUSAL_BOTTOM_RIGHT: int = 0

    L2_SIZE_MIB: int = 60
    SCHEDULER_POLICY: int = SCHED_NATURAL

    N_BMM2_CHUNKS: int = 128 // 64
    BMM2_CHUNK_SIZE: int = 64

    TOTAL_WARPS: int = 16
    THREADS_PER_CTA: int = 16 * 32
    SOFTMAX_WG_WARPS: int = 4
    OTHER_WARPS: int = 4

    # Two softmax warpgroups (WG0 @ warp 0, WG1 @ warp 4); corr @ 8; single
    # roles @ 12..15.
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

    # 8 softmax + 4 corr + 1 MMA + 1 TMALDG + 1 TMASTG = 15 arrivers.
    READ_TILE_ARRIVERS: int = 15

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    THD_VARLEN: int = 0


def _validate_cfg_d128(cfg: CfgD128) -> None:
    """Consistency checks on the (mostly hardcoded) d128 (llama) geometry."""
    _fp8 = cfg.DTYPE_QKV <= 1  # E4M3/E5M2 inputs (MXFP8): STAGES_KV=4, TILE_K_HW=32, independent DTYPE_O
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 2 and cfg.CTA_MMA == 2, "d128 SM100 is cga2-only (CGA_M == CTA_MMA == 2)"),
        (cfg.TILE_K == 128 and cfg.TILE_O == 128, "d128: d_qk = d_v = 128"),
        (cfg.TILES_Q == 2, "d128 (llama): TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d128 (llama): SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d128 (llama): CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d128 (llama): 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d128 llama: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.STAGES_KV == (4 if _fp8 else 2), "d128 SM100: STAGES_KV must be 4 (fp8/mxfp8) / 2 (f16, 192 KiB SMEM budget)"),
        (
            cfg.TILE_K_HW_BMM1 == (32 if _fp8 else 16) and cfg.TILE_K_HW_BMM2 == (32 if _fp8 else 16),
            "d128: TILE_K_HW must be 32 (fp8/mxfp8 K=32 QMMA) / 16 (f16, 1-chunk on SM10x)",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d128: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d128: V/O swizzle out of range"),
        (
            cfg.DTYPE_O in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16) if _fp8 else cfg.DTYPE_O == cfg.DTYPE_QKV,
            "d128: DTYPE_O must equal DTYPE_QKV for half input; fp8/mxfp8 allows an independent output dtype",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d128(params: TemplateParams) -> Tuple[CfgD128, TmaIters]:
    _validate_params("d128", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv <= 1  # E4M3/E5M2 inputs → MXFP8 kernel
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    # FP8/MXFP8 pins the Blackwell K=32 QMMA path (TILE_K_HW=32) and STAGES_KV=4
    # (BPE=1 → 8 KiB/stage, fits 4); f16/bf16 keep 16 / 2.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    cfg = CfgD128(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        Q_SWZ_BYTES=q_swz_bytes(128, b),
        K_SWZ_BYTES=q_swz_bytes(128, b),
        V_SWZ_BYTES=v_swz_bytes(128, 2, b),
        O_SWZ_BYTES=o_swz_bytes(128, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=tile_k_hw_fp8,
        STAGES_KV=4 if fp8 else 2,
        MASK_FLAGS=params.mask_flags,
        SWA_WINDOW=params.swa_window,
        HAS_SINK=int(params.has_sink),
        CAUSAL_BOTTOM_RIGHT=int(params.causal_bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
    )
    _validate_cfg_d128(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d192/d128 flavor — DSv3 MLA logical d_qk = 192, d_v = 128, SM100, cga2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD192(CfgD128):
    TILE_K: int = 192
    TILE_O: int = 128
    QO_ALIAS: int = 1
    SOFTMAX_REGS: int = 192
    CORRECTION_REGS: int = 88


def _validate_cfg_d192(cfg: CfgD192) -> None:
    """Consistency checks on the native DSv3 d192/d128 geometry."""
    checks = (
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "d192: only BF16/FP16 inputs are supported"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d192: DTYPE_O must equal DTYPE_QKV"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d192: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d192: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d192: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 2 and cfg.CTA_MMA == 2, "d192 SM100 is cga2-only (CGA_M == CTA_MMA == 2)"),
        (cfg.TILE_K == 192 and cfg.TILE_O == 128, "d192: expected D_QK tile 192 and D_V tile 128"),
        (cfg.QO_ALIAS == 1, "d192: Q/O SMEM alias is required to stay within SM100 SMEM budget"),
        (cfg.TILES_Q == 2, "d192: TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d192: SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d192: CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d192: 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d192: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.STAGES_KV == 2, "d192 SM100: STAGES_KV must be 2 for BF16/FP16"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d192: TILE_K_HW must be 16 for BF16/FP16 on SM10x"),
        (cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128, "d192: Q/K swizzle must be 128B"),
        (cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128, "d192: V/O swizzle must be 128B"),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    _validate_params("d192", params)
    b = bpe(params.dtype_qkv)
    cfg = CfgD192(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=params.dtype_qkv,
        BPE=b,
        BPE_O=b,
        Q_SWZ_BYTES=q_swz_bytes(192, b),
        K_SWZ_BYTES=q_swz_bytes(192, b),
        V_SWZ_BYTES=v_swz_bytes(128, 2, b),
        O_SWZ_BYTES=o_swz_bytes(128, b),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=params.mask_flags,
        SWA_WINDOW=params.swa_window,
        HAS_SINK=int(params.has_sink),
        CAUSAL_BOTTOM_RIGHT=int(params.causal_bottom_right),
        SCHEDULER_POLICY=1,
        SOFTMAX_REGS=216 if params.mask_flags == MASK_NONE else 192,
        CORRECTION_REGS=40 if params.mask_flags == MASK_NONE else 88,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
    )
    _validate_cfg_d192(cfg)
    return cfg, _tma_iters(cfg)


MAKE_CFG = {128: make_cfg_d128, 192: make_cfg_d192, 256: make_cfg_d256, 512: make_cfg_d512}
