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
from typing import Optional, Tuple

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
    # The mask is ONE diagonal band (the model FlashAttention / CUTLASS FMHA /
    # the analyzer facts all share): per-side OFFSETS from the diagonal, None =
    # unbounded on that side. Row q attends kv in [q - window_left, q +
    # window_right] (shifted by S_kv - S_q when bottom_right):
    #   window_right = None -> no upper bound;  0 -> plain causal;  R > 0 ->
    #   causal widened by R future tokens (diagonal_band_right_bound).
    #   window_left  = None -> no lower bound;  W -> sliding window of W past
    #   tokens (cuDNN diagonal_band_left_bound - 1).
    #   bottom_right anchors the band's diagonal at the bottom-right corner.
    # make_cfg_* derives the kernels' MASK_FLAGS bits from these.
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    bottom_right: bool = False
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
    # Compile-time LPT head/batch grouping. Keep 1 unless the selected kernel
    # and concrete graph shape opt into a divisor of B*Hq.
    lpt_head_group: int = 1
    # Dense D192 FP8 may specialize the reverse-row LPT decoder to its exact
    # number of query tiles. Zero keeps the existing runtime derivation.
    lpt_q_tiles: int = 0
    thd_varlen: bool = False
    # PackGQA: pack Q rows from the G query heads sharing one KV head into a
    # single TILE_M tile, token-major (row r ↔ token r // G, head r % G), so
    # tiles stay full for GQA/MQA.
    pack_gqa: bool = False
    qh_per_kh: int = 1
    # KV split: each Q tile's KV loop range is cut into ``split_kv`` contiguous
    # chunks, each run as its own persistent tile writing a partial (O, LSE)
    # that kernels/split_combine_sm100.py reduces.  1 = off (byte-identical
    # codegen to the single-pass kernel).
    split_kv: int = 1
    # MMA cluster width: 2 = cga2 collective tcgen05.mma.cta_group::2 (a CTA
    # pair share one MMA, each holding half of every K/V tile); 1 = cga1, one
    # independent CTA per tile.
    #
    # cga1 has no collective MMA to halve per-CTA K/V, so at a fixed STAGES_KV
    # it doubles that footprint.  d128 buys the 64 KiB back by aliasing Q and O
    # into one slab (make_cfg_d128 turns QO_ALIAS on for cga1; see
    # _validate_cfg_d128's SMEM check); the fp8 family instead scales the stage
    # count with the width so stages x per-CTA-buffer stays constant.
    cta_mma: int = 2
    # cc10.3+ fuses the S_acc row-max into the LDTM (tcgen05.ld.red.f32.max); cc10.0
    # lacks it and uses the manual load + software reduction. Auto-set from the device
    # capability at compile time (MXFP8 only; the f16/fp8 kernels do not read it).
    fused_ldtm_stat: bool = False
    # softmax_precision knob = cudnn.data_type.HALF: exponent + P-cast run as
    # f16x2 pairs (MUFU EX2.F16x2 + cvt.rn.satfinite.*x2.f16x2) instead of
    # scalar f32 ex2. Per-tensor FP8 on the SM107 sibling kernel only — the
    # exp arguments are bounded (<= RESCALE_THRESHOLD + P_CAST_LOG2_SCALE),
    # so f16 range is exact where it matters and P quantizes to FP8 either way.
    softmax_f16: bool = False


# split_kv / cta_mma live on the TemplateParams shared by every SM100 flavor, but
# a flavor only honours them once its make_cfg_* threads them into a Cfg AND its
# kernel reads them.  Accepting them elsewhere would silently ignore them — and
# for split_kv that is not merely surprising but WRONG: the caller sizes an
# (S*B)-batch partial workspace and runs the combine, while the kernel keeps
# writing only slots [0, B).  The untouched slots keep lse_partial = 0 rather
# than -inf, so they carry weight exp(0 - M) != 0 through the log-sum-exp and
# corrupt the result instead of dropping out.  Grow these sets as flavors land.
_SPLIT_KV_FLAVORS = frozenset({"d128", "d192", "d256", "d512"})
_CTA_MMA_FLAVORS = frozenset({"d128", "d192"})


def _validate_params(flavor: str, k: TemplateParams) -> None:
    if k.dtype_qkv not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: DTYPE_QKV must be E4M3/E5M2/BF16/FP16 (0..3); got {k.dtype_qkv}")
    fp8 = k.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if fp8 and flavor not in ("d128", "d192"):
        raise ValueError(f"{flavor}: FP8/MXFP8 inputs (DTYPE_QKV 0/1) are only supported on d128 and d192")
    if k.softmax_f16 and not fp8:
        raise ValueError(f"{flavor}: softmax_f16 is per-tensor-FP8-only (f16/bf16 softmax already runs the f32 pipeline)")
    dtype_o = k.dtype_qkv if k.dtype_o < 0 else k.dtype_o
    if dtype_o not in (DTYPE_E4M3, DTYPE_E5M2, DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"{flavor}: DTYPE_O must be 0..3; got {dtype_o}")
    if not fp8 and dtype_o != k.dtype_qkv:
        raise ValueError(f"{flavor}: half input (BF16/FP16) requires DTYPE_O == DTYPE_QKV; got dtype_o={dtype_o}")
    if k.window_left is not None and k.window_left < 0:
        raise ValueError(f"{flavor}: window_left must be >= 0 (or None for unbounded); got {k.window_left}")
    if k.window_right is not None and k.window_right < 0:
        raise ValueError(f"{flavor}: window_right must be >= 0 (or None for unbounded); got {k.window_right}")
    if k.bottom_right:
        if k.window_right is None:
            raise ValueError(f"{flavor}: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if k.thd_varlen and not k.seq_kv_lens_present:
        raise ValueError(f"{flavor}: THD/varlen requires SEQ_KV_LENS_PRESENT (per-sequence padded masking)")
    if k.seq_q_lens_present:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT is dense-only (THD carries per-sequence Q lengths via cu_seqlens)")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    if k.sched_policy not in (SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2):
        raise ValueError(f"{flavor}: only SCHED_NATURAL (0) / SCHED_LPT (1) / SCHED_LPT_L2 (2) are wired up; got {k.sched_policy}")
    if k.cta_mma not in (1, 2):
        raise ValueError(f"{flavor}: cta_mma must be 1 (cga1) or 2 (cga2); got {k.cta_mma}")
    # split_kv / cta_mma live on the TemplateParams shared by every SM100 flavor,
    # but only make_cfg_d128 threads them into a Cfg and only the d128 kernel
    # reads them.  Accepting them elsewhere would silently ignore them — and for
    # split_kv that is not merely surprising but WRONG: the caller sizes an
    # (S*B)-batch partial workspace and runs the combine, while the kernel keeps
    # writing only slots [0, B).  The untouched slots keep lse_partial = 0 rather
    # than -inf, so they carry weight exp(0 - M) != 0 through the log-sum-exp and
    # corrupt the result instead of dropping out.  Reject at the door.
    if k.split_kv != 1 and flavor not in _SPLIT_KV_FLAVORS:
        raise ValueError(f"{flavor}: split_kv is not implemented on this flavor (got {k.split_kv}); supported: {sorted(_SPLIT_KV_FLAVORS)}")
    if k.cta_mma != 2 and flavor not in _CTA_MMA_FLAVORS:
        raise ValueError(f"{flavor}: cta_mma is not selectable on this flavor (got {k.cta_mma}); supported: {sorted(_CTA_MMA_FLAVORS)}")
    if k.split_kv < 1:
        raise ValueError(f"{flavor}: split_kv must be >= 1 (1 = KV-split off); got {k.split_kv}")
    if k.split_kv > 1:
        # Each of these would need extra machinery in the combine pass, so the
        # backstop rejects them rather than silently producing a wrong answer.
        if k.thd_varlen:
            raise ValueError(f"{flavor}: split_kv > 1 is dense-only (THD packs its own flat grid)")
        if k.has_sink:
            # The sink logit is folded into the softmax denominator in the
            # per-tile epilogue, so every split would add its own copy of it.
            raise ValueError(f"{flavor}: split_kv > 1 with attention sink is not supported (the sink would be counted once per split)")
    if k.lpt_head_group not in (1, 8, 16):
        raise ValueError(f"{flavor}: LPT_HEAD_GROUP must be 1, 8, or 16; got {k.lpt_head_group}")
    if fp8 and flavor == "d192" and k.split_kv != 1:
        raise ValueError("d192: split_kv is not implemented by the per-tensor FP8 kernel")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.pack_gqa:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: pack_gqa is not supported for THD-varlen")


def _mask_flags_from(params: TemplateParams) -> int:
    """Kernel-facing MASK bits, derived from the band + padding: the kernels
    fold trace-time branches on these bits; the band VALUES ride the CFG's
    WINDOW_LEFT / WINDOW_RIGHT fields (0 when the corresponding bit is unset)."""
    flags = MASK_NONE
    if params.window_right is not None:
        flags |= MASK_CAUSAL
    if params.window_left is not None:
        flags |= MASK_SWA
    if params.thd_varlen or params.seq_kv_lens_present:
        flags |= MASK_PADDED
    return flags


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


def pack_gqa_supported(h_q: int, h_kv: int, tile_m: int = 128) -> bool:
    return h_q > 0 and h_kv > 0 and h_q % h_kv == 0 and tile_m % (h_q // h_kv) == 0


def cga_tile_m(d_qk: int) -> int:
    """Q rows one cluster covers for a flavor: TILES_Q * TILE_M * CTA_MMA.

    The tile-count denominator the pack_gqa heuristic needs (d128/d192: 512;
    d256/d512: 256), computed from the flavor Cfg classes, not literals.
    """
    cls = {128: CfgD128, 192: CfgD192, 256: CfgD256, 512: CfgD512}[d_qk]
    return cls.TILES_Q * cls.TILE_M * cls.CTA_MMA


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

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
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

    QH_PER_KH: int = 1


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
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
    )
    _validate_cfg_d256(cfg)
    if cfg.PACK_GQA and cfg.TILE_M % cfg.QH_PER_KH != 0:
        raise ValueError(f"qh_per_kh ({cfg.QH_PER_KH}) must divide TILE_M ({cfg.TILE_M}) when PACK_GQA is enabled")
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

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
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

    QH_PER_KH: int = 1


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
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
    )
    _validate_cfg_d512(cfg)
    if cfg.PACK_GQA and cfg.TILE_M % cfg.QH_PER_KH != 0:
        raise ValueError(f"qh_per_kh ({cfg.QH_PER_KH}) must divide TILE_M ({cfg.TILE_M}) when PACK_GQA is enabled")
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

    MASK_FLAGS: int = MASK_NONE  # derived from the band by make_cfg (see _mask_flags_from)
    WINDOW_LEFT: int = 0  # band left offset W (valid when MASK_SWA is set)
    WINDOW_RIGHT: int = 0  # band right offset R (valid when MASK_CAUSAL is set; 0 = plain causal)
    HAS_SINK: int = 0
    BOTTOM_RIGHT: int = 0  # band diagonal anchored bottom-right

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

    # KV split; 1 = off.  See TemplateParams.split_kv.
    SPLIT_KV: int = 1

    PACK_GQA: int = 0

    QH_PER_KH: int = 1


# Blackwell SM100 per-CTA dynamic SMEM cap (228 KiB physical, 227 KiB usable).
_SM100_MAX_DYN_SMEM = 227 * 1024


def _d128_smem_bytes(cfg) -> int:
    """Data-buffer SMEM for the d128 pipeline (barriers/TMEM ptr are noise).

    Q and O are TILES_Q slabs each; under QO_ALIAS they share one slab sized to
    the larger.  K/V are STAGES_KV buffers each, and their PER-CTA size is
    divided by CTA_MMA because the cga2 collective MMA lets a CTA pair hold half
    of every tile.  That divisor is exactly what cga1 gives up, which is why
    cga1 needs the Q/O alias to break even:

        cga2, no alias : 64(Q) + 64(O) + 32(K) + 32(V) = 192 KiB
        cga1, no alias : 64(Q) + 64(O) + 64(K) + 64(V) = 256 KiB  (over cap)
        cga1, alias    : 64(Q u O)     + 64(K) + 64(V) = 192 KiB
    """
    q_slab = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    o_slab = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O
    qo = cfg.TILES_Q * (max(q_slab, o_slab) if cfg.QO_ALIAS else q_slab + o_slab)
    k = cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K * cfg.BPE // cfg.CTA_MMA)
    v = cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N * cfg.BPE // cfg.CTA_MMA)
    return qo + k + v


def _validate_cfg_d128(cfg: CfgD128) -> None:
    """Consistency checks on the (mostly hardcoded) d128 (llama) geometry."""
    _fp8 = cfg.DTYPE_QKV <= 1  # E4M3/E5M2 inputs (MXFP8): STAGES_KV=4, TILE_K_HW=32, independent DTYPE_O
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128: per-role regs must be multiples of 8"),
        (cfg.CGA_M == cfg.CTA_MMA and cfg.CTA_MMA in (1, 2), "d128 SM100: CGA_M must equal CTA_MMA, and CTA_MMA must be 1 (cga1) or 2 (cga2)"),
        (
            cfg.QO_ALIAS == 1 if cfg.CTA_MMA == 1 else True,
            "d128 cga1: QO_ALIAS is mandatory — cga1 doubles per-CTA K/V (no collective MMA to halve it), "
            "so Q and O must share one slab to stay inside the SMEM cap",
        ),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d128: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.TILE_K == 128 and cfg.TILE_O == 128, "d128: d_qk = d_v = 128"),
        (cfg.TILES_Q == 2, "d128 (llama): TILES_Q must be 2"),
        (cfg.SOFTMAX_WARPGROUPS == 2, "d128 (llama): SOFTMAX_WARPGROUPS must be 2"),
        (cfg.CORRECTION_WARPS == 4, "d128 (llama): CORRECTION_WARPS must be 4"),
        (cfg.TOTAL_WARPS == 16 and cfg.THREADS_PER_CTA == 512, "d128 (llama): 16 warps / 512 threads"),
        (cfg.READ_TILE_ARRIVERS == 15, f"d128 llama: expected READ_TILE_ARRIVERS=15, got {cfg.READ_TILE_ARRIVERS}"),
        (
            cfg.STAGES_KV == ((2 if cfg.CTA_MMA == 1 else 4) if _fp8 else 2),
            "d128 SM100: STAGES_KV must be 2 (f16/bf16) or, for fp8/mxfp8, 4 at cga2 and 2 at cga1 — "
            "the stage depth scales with the cluster width so stages x per-CTA-buffer stays constant",
        ),
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
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        # cga1 has no collective MMA to halve per-CTA K/V, so Q and O must share
        # one slab to stay under the SMEM cap (_validate_cfg_d128 enforces it).
        QO_ALIAS=1 if params.cta_mma == 1 else 0,
        Q_SWZ_BYTES=q_swz_bytes(128, b),
        K_SWZ_BYTES=q_swz_bytes(128, b),
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(128, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=tile_k_hw_fp8,
        # KV stage depth scales with the cluster width, as in cuDNN's own
        # kernels (stages_kv = N * CTA_MMA): cga1 has no collective MMA to halve
        # per-CTA K/V, so the stage count halves instead to keep the product --
        # and hence the SMEM -- constant.  Only the fp8 family needs this here:
        # f16/bf16 already fit at cga1 by aliasing Q and O, and their verified
        # cga1 configuration keeps STAGES_KV=2.  mxfp8 additionally stages E8M0
        # scale factors that the SMEM model cannot see, and at STAGES_KV=4 that
        # pushed a cga1 CTA to 237024 B against the 232448 B cap.
        STAGES_KV=(2 if params.cta_mma == 1 else 4) if fp8 else 2,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
    )
    _validate_cfg_d128(cfg)
    if cfg.PACK_GQA and cfg.TILE_M % cfg.QH_PER_KH != 0:
        raise ValueError(f"qh_per_kh ({cfg.QH_PER_KH}) must divide TILE_M ({cfg.TILE_M}) when PACK_GQA is enabled")
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


def _d192_smem_bytes(cfg) -> int:
    """Data-buffer SMEM for the d192 pipeline.

    Same shape as _d128_smem_bytes, but d_qk = 192 makes the Q and K slabs 1.5x
    the d128 ones, which is why this flavor needs a shallower KV pipeline:

        cga2, STAGES_KV=2 : 96(Q u O) + 48(K) + 32(V) = 176 KiB
        cga1, STAGES_KV=2 : 96        + 96    + 64    = 256 KiB  (over cap)
        cga1, STAGES_KV=1 : 96        + 48    + 32    = 176 KiB
    """
    q_slab = cfg.TILE_M * cfg.TILE_K * cfg.BPE
    o_slab = cfg.TILE_M * cfg.TILE_O * cfg.BPE_O
    qo = cfg.TILES_Q * (max(q_slab, o_slab) if cfg.QO_ALIAS else q_slab + o_slab)
    k = cfg.STAGES_KV * (cfg.TILE_N * cfg.TILE_K * cfg.BPE // cfg.CTA_MMA)
    v = cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N * cfg.BPE // cfg.CTA_MMA)
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
            cfg.STAGES_KV == (2 if fp8 else 1) * cfg.CTA_MMA,
            "d192: STAGES_KV must scale with input dtype and cluster width " "(FP8: 2/4 at cga1/cga2; half: 1/2)",
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
            cfg.TILE_K_HW_BMM1 == (32 if fp8 else 16) and cfg.TILE_K_HW_BMM2 == (32 if fp8 else 16),
            "d192: TILE_K_HW must be 32 for FP8 and 16 for BF16/FP16",
        ),
        (
            cfg.Q_SWZ_BYTES == (64 if fp8 else 128) and cfg.K_SWZ_BYTES == (64 if fp8 else 128),
            "d192: Q/K swizzle must be 64B for FP8 and 128B for BF16/FP16",
        ),
        (
            cfg.V_SWZ_BYTES == v_swz_bytes(128, cfg.CTA_MMA, cfg.BPE) and cfg.O_SWZ_BYTES in ((64, 128) if fp8 else (128,)),
            "d192: V/O swizzle is inconsistent with the input/output dtype",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    _validate_params("d192", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)
    cfg = CfgD192(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        SPLIT_KV=int(params.split_kv),
        QO_ALIAS=0 if fp8 else 1,
        Q_SWZ_BYTES=q_swz_bytes(192, b),
        K_SWZ_BYTES=q_swz_bytes(192, b),
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b),
        O_SWZ_BYTES=o_swz_bytes(128, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=32 if fp8 else tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=32 if fp8 else tile_k_hw(params.dtype_qkv),
        STAGES_KV=(2 if fp8 else 1) * params.cta_mma,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=1,
        SOFTMAX_REGS=184 if fp8 else 216 if _mask_flags_from(params) == MASK_NONE else 192,
        CORRECTION_REGS=104 if fp8 else 40 if _mask_flags_from(params) == MASK_NONE else 88,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
    )
    _validate_cfg_d192(cfg)
    if cfg.PACK_GQA and cfg.TILE_M % cfg.QH_PER_KH != 0:
        raise ValueError(f"qh_per_kh ({cfg.QH_PER_KH}) must divide TILE_M ({cfg.TILE_M}) when PACK_GQA is enabled")
    return cfg, _tma_iters(cfg)


MAKE_CFG = {128: make_cfg_d128, 192: make_cfg_d192, 256: make_cfg_d256, 512: make_cfg_d512}
