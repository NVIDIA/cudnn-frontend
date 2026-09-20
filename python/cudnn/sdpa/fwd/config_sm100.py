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

import math
from dataclasses import dataclass, replace
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
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the
    # epilogue scales the natural-log LSE by log2(e) right before the store.
    stats_log2: bool = False
    seq_kv_lens_present: bool = False
    # Dense padded-Q trim: per-batch seq_len_q is a SEPARATE (B,)-int32
    # kernel parameter (seq_q_lens_addr — mirrors cuDNN's distinct SEQLEN_Q
    # pointer / FA's seqused_q; its reads compile out unless this flag is
    # set); q rows >= seq_len_q[b] write O := 0 / LSE := -inf (cuDNN >= 9.14
    # convention). Dense-only — THD carries per-sequence Q lengths via
    # cu_seqlens instead.
    seq_q_lens_present: bool = False
    sched_policy: int = SCHED_NATURAL
    # Compile-time LPT head/batch grouping. Keep 1 unless the selected kernel
    # and concrete graph shape opt into a divisor of B*Hq.
    lpt_head_group: int = 1
    # Dense FP8 kernels may specialize scheduler selection/decoding to the
    # graph's compile-time number of query tiles. Zero keeps runtime derivation.
    lpt_q_tiles: int = 0
    # Optional L2 working-set budget for SCHED_LPT_L2. Zero keeps the flavor's
    # default budget.
    lpt_l2_size_mib: int = 0
    thd_varlen: bool = False
    # PackGQA: pack Q rows from the G query heads sharing one KV head into a
    # single TILE_M tile, token-major (row r ↔ token r // G, head r % G), so
    # tiles stay full for GQA/MQA.  When G does not divide TILE_M the d128 and
    # d256 f16 kernels pack its largest divisor that does (Cfg.PACK_G, from
    # pack_gqa_group_size); ``qh_per_kh`` is always the graph's GQA ratio.
    pack_gqa: bool = False
    qh_per_kh: int = 1
    # KV split: each Q tile's KV loop range is cut into ``split_kv`` contiguous
    # chunks, each run as its own persistent tile writing a partial (O, LSE)
    # that kernels/sm100/split_combine.py reduces.  1 = off (byte-identical
    # codegen to the single-pass kernel).
    #
    # A split writes its partials as fp32, straight from the epilogue's
    # accumulator registers and bypassing the SMEM O tile and its TMA store, so
    # the combine reduces an unrounded input and performs the only rounding to
    # O's dtype.  Partial-only: the caller-visible O is unchanged.
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
    # sdpa(softmax_precision=cudnn.data_type.HALF) op attribute: exponent + P-cast run as
    # f16x2 pairs (MUFU EX2.F16x2 + cvt.rn.satfinite.*x2.f16x2) instead of
    # scalar f32 ex2. Per-tensor FP8 on the SM107 sibling kernel only — the
    # exp arguments are bounded (<= RESCALE_THRESHOLD + P_CAST_LOG2_SCALE),
    # so f16 range is exact where it matters and P quantizes to FP8 either way.
    softmax_f16: bool = False
    # Paged KV cache (FlashInfer / vLLM decode contract): K/V are page pools
    # indexed through a per-batch ``block_table`` [B, max_pages] int32, and
    # the per-batch KV length is the (B,) ``seq_kv_lens`` device tensor
    # (``seq_kv_lens_present`` is therefore mandatory). ``page_size`` tokens
    # per page. The in-page layout (HND ``[num_pages, H_kv, page_size, D]`` vs
    # NHD ``[num_pages, page_size, H_kv, D]``) is NOT a parameter: it is the
    # pool's strides, which the per-shape compile() already pins. Dense-only.
    paged_kv: bool = False
    page_size: int = 0
    # Experimental D128 MXFP8 specialization: Q/K remain block-scaled FP8,
    # while the complete softmax(P) @ V leg uses BF16 operands. This is a
    # template axis because it changes the TMA maps, shared-memory layout and
    # BMM2 instruction kind. It is intentionally not wired into graph routing.
    pv_bf16: bool = False
    # Output Amax is an independently optional graph output. It is normally
    # enabled for the existing MXFP8 kernels; the direct QK-MXFP8/BF16-PV
    # experiment enables it only when its plan declares an Amax_O buffer.
    # This is plan-time state: it must never be inferred from execute() args.
    emit_amax_o: bool = True
    # Epilogue gate: O := O * sigmoid(GATE). GATE is TMA-staged by the load warp after the KV loop into a
    # 64 KiB SMEM tile and consumed in the correction epilogue. Compile-time: sizes the tile + 2 mbarriers.
    # Served by the SM107 d256 f16/bf16 and per-tensor FP8 kernels only (config_sm107._EPILOGUE_GATE_FLAVORS).
    # Being a TemplateParams field it is part of the module-cache key: gate on/off are two coexisting
    # specializations of one template, and an ungated module traces byte-identically to before the field existed.
    epilogue_gate: bool = False
    # Decode-shaped d256 f16/bf16 graphs (S_q * pack_g <= D256_DECODE_ROUTED_MAX_Q_ROWS)
    # lower onto sm100/decode_d256_f16.py, the swap-AB tile: the KV tokens ride
    # the MMA M axis and the packed Q rows ride N, so the MMA and exp work
    # scale with the LIVE rows instead of a 128-row Q tile.  The value is the
    # N extent the kernel is compiled for (16, or 32 at the template level, from
    # decode_d256_q_tile); 0 = the prefill tile.  Plan-time only (S_q is a
    # declared shape).
    decode_q_tile: int = 0


# Paged KV is wired through the K/V TMA-LDG sites of these flavors only; any
# other flavor must reject it rather than silently reading K/V as dense.
# Flavor tags as make_cfg_* / _validate_params spell them ("d192" is the
# d192x128 kernel, whose K and V pools differ in row width). engines'
# ``paged_d_shapes`` and the adapter's check_support name the same set.
_PAGED_KV_FLAVORS = frozenset({"d128", "d192", "d256"})

# The fused epilogue gate (TemplateParams.epilogue_gate) is a RUBIN feature: no
# SM100 kernel body reads CFG.EPILOGUE_GATE, so a module loaded with the flag on
# this line would silently produce the UN-gated O. Empty on purpose; the Rubin
# flavors that serve it are config_sm107._EPILOGUE_GATE_FLAVORS.
_EPILOGUE_GATE_FLAVORS = frozenset()

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
    if fp8 and flavor not in ("d128", "d192", "d256", "d512"):
        raise ValueError(f"{flavor}: FP8/MXFP8 inputs (DTYPE_QKV 0/1) are only supported on d128, d192, d256, and d512")
    if k.softmax_f16 and not fp8:
        raise ValueError(f"{flavor}: softmax_f16 is per-tensor-FP8-only (f16/bf16 softmax already runs the f32 pipeline)")
    if k.pv_bf16 and (not fp8 or flavor not in ("d128", "d192")):
        raise ValueError(f"{flavor}: pv_bf16 is an experimental MXFP8 D128/D192 specialization")
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
    if k.decode_q_tile:
        # The loader routes a decode_q_tile record to sm100/decode_d256_f16.py
        # (make_cfg_d256_decode); a prefill template must never consume one.
        raise ValueError(f"{flavor}: decode_q_tile={k.decode_q_tile} selects the d256 decode tile; the prefill templates do not consume it")
    # A flavor must explicitly consume split_kv / cta_mma. Accepting either
    # elsewhere would silently ignore it; for split_kv that is also WRONG: the
    # caller sizes an (S*B)-batch partial workspace and runs the combine, while the kernel keeps
    # writing only slots [0, B).  The untouched slots keep lse_partial = 0 rather
    # than -inf, so they carry weight exp(0 - M) != 0 through the log-sum-exp and
    # corrupt the result instead of dropping out.  Reject at the door.
    if k.split_kv != 1 and flavor not in _SPLIT_KV_FLAVORS:
        raise ValueError(f"{flavor}: split_kv is not implemented on this flavor (got {k.split_kv}); supported: {sorted(_SPLIT_KV_FLAVORS)}")
    # D256 selects its topology from the input family rather than exposing a
    # free CTA-MMA knob: half inputs stay CTA2, while FP8/MXFP8 use CTA1.
    d256_quantized_cta1 = flavor == "d256" and fp8 and k.cta_mma == 1
    if k.cta_mma != 2 and flavor not in _CTA_MMA_FLAVORS and not d256_quantized_cta1:
        raise ValueError(f"{flavor}: cta_mma is not selectable on this flavor (got {k.cta_mma}); supported: {sorted(_CTA_MMA_FLAVORS)}")
    if flavor == "d192" and k.split_kv > 1 and k.cta_mma != 2:
        raise ValueError("d192: split_kv > 1 is validated only with cta_mma=2")
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
    lpt_head_groups = (1, 8, 16, 32) if flavor == "d256" else (1, 8, 16)
    if k.lpt_head_group not in lpt_head_groups:
        raise ValueError(f"{flavor}: LPT_HEAD_GROUP must be one of {lpt_head_groups}; got {k.lpt_head_group}")
    if k.qh_per_kh < 1:
        raise ValueError(f"{flavor}: qh_per_kh ({k.qh_per_kh}) must be >= 1")
    if k.pack_gqa:
        if k.thd_varlen:
            raise ValueError(f"{flavor}: pack_gqa is not supported for THD-varlen")
    if k.paged_kv:
        if flavor not in _PAGED_KV_FLAVORS:
            raise ValueError(f"{flavor}: paged_kv is not implemented on this flavor; supported: {sorted(_PAGED_KV_FLAVORS)}")
        if not k.seq_kv_lens_present:
            raise ValueError(f"{flavor}: paged_kv requires seq_kv_lens_present (the per-batch KV length bounds the block-table walk)")
        if fp8:
            raise ValueError(f"{flavor}: paged_kv is wired for the f16/bf16 kernel only")
        # A K/V tile is loaded as a stack of page-sized row boxes (or one box
        # inside a page when the page is taller than the tile). Either way a
        # box must never straddle a page, and the 128 B swizzle atom is 8 rows.
        p = k.page_size
        if p < 8 or p % 8 != 0:
            raise ValueError(f"{flavor}: page_size must be a positive multiple of 8; got {p}")
        if (p < 128 and 128 % p != 0) or (p > 128 and p % 128 != 0):
            raise ValueError(f"{flavor}: page_size must divide the 128-row KV tile or be a multiple of it; got {p}")
    elif k.page_size:
        raise ValueError(f"{flavor}: page_size requires paged_kv=True")
    if k.epilogue_gate and flavor not in _EPILOGUE_GATE_FLAVORS:
        raise ValueError(f"{flavor}: epilogue_gate is not wired on the SM100 line (served by the SM107 d256 kernels only)")


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


def bshd_compact(shape_bhsd: tuple, stride_bhsd: tuple) -> bool:
    """True when a logical BHSD ``(b, h, s, d)`` operand already IS the kernels'
    canonical BSHD-compact layout (batch, then tokens, then heads, then a
    contiguous head dim): nothing to declare, the artifact binds it as a view."""
    b, h, s, d = (int(x) for x in shape_bhsd)
    bs, hs, ss, es = (int(x) for x in stride_bhsd)
    return (bs, ss, hs, es) == (s * h * d, h * d, d, 1)


def bshd_zero_copy_stride(shape_bhsd: tuple, stride_bhsd: tuple, elem_bytes: int) -> Optional[tuple]:
    """The BSHD ``(b, s, h, d)`` stride tuple a dense prefill kernel is COMPILED
    at for a logical BHSD operand, or None.

    None means EITHER "compact, nothing to declare" (``bshd_compact``) OR "a
    layout the kernels cannot bind zero-copy"; a caller that must tell the two
    apart tests ``bshd_compact`` first (the epilogue gate does -- it has no
    copy fallback).  The rules are the kernels' own (``_fake_bshd`` in every
    f16 / per-tensor-FP8 prefill kernel), restated here so the decision is made
    BEFORE a compile rather than as a raise inside one -- and so the engine rows
    (``engines.mismatch`` via ``config_sm107.epilogue_gate_layout_declarable``)
    and the standalone adapter (``api_dsl.SdpaFwdDsl._bshd_zero_copy_stride``)
    judge a layout with ONE function (rule 8b lockstep):

      * the head dim is innermost-contiguous (stride 1);
      * the batch, seq and head strides are 16-byte multiples (TMA global-stride rule;
        the DSL floors a misaligned stride to TMA units, so a violation would mis-address);
      * the declaration is TOKEN-MAJOR and COVERING: head >= d, seq >= h*head,
        batch >= s*seq.  A head-major nest (a torch-contiguous ``[B, H, S, D]``:
        seq stride d < h*head) returns None because the kernels' TMA
        descriptors are built in BSHD order and no other nesting has been
        validated zero-copy; an overlapping declaration returns None because
        it would alias distinct rows onto one address (a write race on O).

    ``graph_analyzer.dense_layout_ok`` is deliberately WIDER (any B/H/S stride
    order, any alignment): that is the contract of the copy-normalising Q/K/V/O
    path, not of a zero-copy binding.
    """
    b, h, s, d = (int(x) for x in shape_bhsd)
    bs, hs, ss, es = (int(x) for x in stride_bhsd)
    if (bs, ss, hs, es) == (s * h * d, h * d, d, 1):
        return None  # compact: nothing to declare
    if es != 1:
        return None
    per16 = 16 // elem_bytes
    if ss % per16 or hs % per16 or (b > 1 and bs % per16):  # every non-innermost TMA global stride, the batch one included
        return None
    if hs < d or ss < h * hs or bs < s * ss:
        return None
    return (bs, ss, hs, es)


def canonical_bhsd_strides(shape_bhsd: tuple, stride_bhsd: tuple) -> tuple:
    """``stride_bhsd`` with every extent-1 axis given the covering canonical stride.

    A stride on an axis of extent 1 is never stepped, so a buffer's own value there is
    unobservable (torch's ``is_contiguous`` wildcards such axes and a one-KV-head K keeps
    whatever stride its allocation had); the kernels' layout rules below are stated for
    stepped axes, so the singleton ones are spelled the way a compact BSHD buffer would
    spell them before the rules are applied."""
    b, h, s, d = (int(x) for x in shape_bhsd)
    bs, hs, ss, es = (int(x) for x in stride_bhsd)
    if h == 1:
        hs = d
    if s == 1:
        ss = h * hs
    if b == 1:
        bs = s * ss
    return (bs, hs, ss, es)


def dense_bind_strides(shape_bhsd: tuple, stride_bhsd: tuple, elem_bytes: int) -> Optional[tuple]:
    """The ``(batch, seq, head)`` element strides a dense prefill kernel binds a logical BHSD
    operand at zero-copy, or None when the layout needs a repack: singleton axes canonicalized,
    then BSHD-compact or ``bshd_zero_copy_stride``. ONE predicate for the lowering's decision to
    attach the prepared launch and for the binder's per-call admission (rule 8b lockstep)."""
    st = canonical_bhsd_strides(shape_bhsd, stride_bhsd)
    if not (bshd_compact(shape_bhsd, st) or bshd_zero_copy_stride(shape_bhsd, st, elem_bytes) is not None):
        return None
    bs, hs, ss, _es = st
    return (bs, ss, hs)


def rescale_threshold(dtype_qkv: int) -> float:
    return 4.0 if dtype_qkv <= 1 else 8.0


def pack_gqa_group_size(qh_per_kh: int, tile_m: int = 128, *, partial: bool = False) -> int:
    """Heads packed into one Q tile row-group for the GQA ratio ``G = qh_per_kh``.

    Full contract (``partial=False``): ``G`` when it divides ``tile_m``, else 0
    (nothing can be packed).  Partial contract (``partial=True``, the d128 and
    d256 f16 kernels): the largest divisor of ``G`` that divides ``tile_m`` --
    ``gcd(G, tile_m)`` -- so a 12-head group packs 4 heads per token row-group
    (three packed heads per KV head), a 6-head group packs 2, and a group with
    no factor in common with the tile (3, 5, ...) yields 1 = unpacked.  MHA
    (``G == 1``) is the identity, 1.
    """
    if qh_per_kh <= 0:
        return 0
    if tile_m % qh_per_kh == 0:
        return qh_per_kh
    return math.gcd(qh_per_kh, tile_m) if partial else 0


def pack_gqa_supported(h_q: int, h_kv: int, tile_m: int = 128, *, partial: bool = False) -> bool:
    """Whether a PACK_GQA=1 plan is HONORABLE for this head count: the group
    divides the tile (full packing) or -- ``partial`` -- shares a factor > 1
    with it.  MHA (G == 1) is the bit-exact unpacked fold, so it is honorable
    too; a group that cannot pack at all (G=3 at tile_m=128) is declined rather
    than silently run unpacked (frost/README.md rule 5)."""
    if h_q <= 0 or h_kv <= 0 or h_q % h_kv != 0:
        return False
    g = h_q // h_kv
    p = pack_gqa_group_size(g, tile_m, partial=partial)
    return p == g or p > 1


def _pack_g(params: TemplateParams, tile_m: int, *, partial: bool) -> int:
    """``Cfg.PACK_G`` for a flavor: the packed group size, validated the way
    ``engines.mismatch`` admits it (full contract, or partial where the kernel
    carries it), so a knob that passed the gate can never reach a kernel that
    would silently run it unpacked."""
    if not params.pack_gqa:
        return 1
    g = int(params.qh_per_kh)
    p = pack_gqa_group_size(g, tile_m, partial=partial)
    if p == 0 or (p == 1 and g != 1):
        how = "share a factor with" if partial else "divide"
        raise ValueError(f"qh_per_kh ({g}) must {how} TILE_M ({tile_m}) when PACK_GQA is enabled")
    return p


def cga_tile_m(d_qk: int, cta_mma: Optional[int] = None) -> int:
    """Q rows one cluster covers for a flavor: TILES_Q * TILE_M * CTA_MMA.

    ``cta_mma`` overrides the flavor default when the selected kernel exposes a
    CGA-width knob (D192). This keeps scheduler and heuristic geometry tied to
    the configuration the launcher actually uses.
    """
    cls = {128: CfgD128, 192: CfgD192, 256: CfgD256, 512: CfgD512}[d_qk]
    if d_qk == 128 and cta_mma == 1:
        # cga1 on the d128 f16/bf16 flavor IS the decode tile (sm100/decode_d128_f16.py,
        # TILES_Q=1): one 128-row Q tile per CTA, not the prefill kernel's two.
        cls = CfgD128Decode
    return cls.TILES_Q * cls.TILE_M * (cls.CTA_MMA if cta_mma is None else cta_mma)


def cga_ctas(d_qk: int, cta_mma: Optional[int] = None) -> int:
    """Physical CTAs per cluster, including the D512 non-MMA role CTAs.

    Public CGA selects the MMA width. It is not a physical launch count for
    the role-split D512 flavor, whose CGA_M/CTA_MMA ratio is two.
    """
    cls = {128: CfgD128, 192: CfgD192, 256: CfgD256, 512: CfgD512}[d_qk]
    return cls.CGA_M // cls.CTA_MMA * (cls.CTA_MMA if cta_mma is None else cta_mma)


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
    vo = _tma_iters_for(cfg.TILE_O, getattr(cfg, "BPE_V", cfg.BPE), cfg.V_SWZ_BYTES)
    return TmaIters(
        QK_ITERS=qk,
        VO_ITERS=vo,
        QK_GRANU_ELEMS=cfg.TILE_K // qk,
        VO_GRANU_ELEMS=cfg.TILE_O // vo,
    )


# ---------------------------------------------------------------------------
# d256 flavor — d_qk = d_v = 256, SM100 (Blackwell), Qwen-class models
# Half inputs use CTA2 collective MMA; FP8/MXFP8 use an independent CTA1 tile.
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


def _validate_cfg_d256(cfg: CfgD256) -> None:
    """Consistency checks on the (mostly hardcoded) d256 geometry."""
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    split_p = cfg.SOFTMAX_WARPGROUPS == 2
    fused_corr_split_p = cfg.FUSED_CORR_SPLIT_P == 1
    split_p_supported = fp8 and (cfg.MASK_FLAGS == MASK_NONE or ((cfg.MASK_FLAGS & ~MASK_PADDED) == MASK_CAUSAL and cfg.BOTTOM_RIGHT == 0))
    checks = (
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d256: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (
            cfg.MMA_REGS + (0 if fused_corr_split_p else cfg.CORRECTION_REGS) + cfg.SOFTMAX_REGS + cfg.SOFTMAX_WG1_REGS <= 512,
            "d256: register budget over 512",
        ),
        (
            cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0 and cfg.SOFTMAX_WG1_REGS % 8 == 0,
            "d256: per-role regs must be multiples of 8",
        ),
        (cfg.CGA_M == cfg.CTA_MMA, "d256 flavor pairs CGA_M with CTA_MMA"),
        (cfg.CTA_MMA == (1 if fp8 else 2), "d256 SM100: FP8 requires CTA1; BF16/FP16 requires CTA2"),
        (cfg.STAGES_KV == 2, "d256 SM100 uses two full/half-width KV stages"),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d256: Q/K swizzle must be 64/128B"),
        (cfg.V_SWZ_BYTES in (32, 64, 128) and cfg.O_SWZ_BYTES in (64, 128), "d256: V/O swizzle out of range"),
        (cfg.TILES_Q == 1, "d256 pipeline mandates TILES_Q == 1"),
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


def d256_square_br_as_tl(params: TemplateParams, *, s_q: int, s_kv: int) -> bool:
    """Whether a D256 bottom-right mask is exactly top-left causal."""

    return (
        params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
        and not params.thd_varlen
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and params.window_left is None
        and params.window_right == 0
        and params.bottom_right
        and s_q == s_kv
    )


def canonicalize_d256_lowering(params: TemplateParams, *, s_q: int, s_kv: int) -> TemplateParams:
    """Apply strictly equivalent D256 lowering canonicalizations."""

    return replace(params, bottom_right=False) if d256_square_br_as_tl(params, s_q=s_q, s_kv=s_kv) else params


def canonicalize_d512_mxfp8_lowering(params: TemplateParams, *, s_q: int, s_kv: int) -> TemplateParams:
    """Canonicalize an exactly square D512 MXFP8 causal diagonal."""

    square_bottom_right = (
        not params.thd_varlen
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and params.window_left is None
        and params.window_right == 0
        and params.bottom_right
        and s_q == s_kv
    )
    return replace(params, bottom_right=False) if square_bottom_right else params


def derive_d256_internal_params(
    params: TemplateParams,
    *,
    pertensor: bool,
    batch_size: int,
    h_q: int,
    s_q: int,
) -> TemplateParams:
    """Derive D256-private codegen fields after public knobs are fixed."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    if not fp8 or params.thd_varlen:
        return params

    pack_gqa_ratio = params.qh_per_kh if params.pack_gqa else 1
    groups = batch_size * h_q // pack_gqa_ratio
    lpt_head_group = 32 if pertensor and groups % 32 == 0 else 8 if not pertensor and groups % 8 == 0 else 1
    q_tiles = (s_q + 255) // 256 if pertensor else 0
    mask_flags = _mask_flags_from(params)
    pt_lpt_l2 = pertensor and params.sched_policy == SCHED_LPT_L2 and mask_flags == MASK_CAUSAL and not params.bottom_right and q_tiles >= 16
    return replace(
        params,
        lpt_head_group=lpt_head_group,
        lpt_l2_size_mib=32 if pt_lpt_l2 else 0,
    )


def _make_cfg_d256(params: TemplateParams, *, mxfp8: bool) -> Tuple[CfgD256, TmaIters]:
    _validate_params("d256", params)
    b = bpe(params.dtype_qkv)
    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    cga = 1 if fp8 else 2
    if params.cta_mma != cga:
        raise ValueError(f"d256: {'FP8/MXFP8' if fp8 else 'BF16/FP16'} requires cta_mma={cga}; got {params.cta_mma}")
    mask_flags = _mask_flags_from(params)
    pt_plain_top_left_causal = (
        not mxfp8 and (mask_flags & ~MASK_PADDED) == MASK_CAUSAL and not params.bottom_right and not params.window_left and not params.window_right
    )
    # The fused correction/split-P schedule is the strict top-left causal fast
    # path. Right-band widening uses the generic masked schedule; forcing the
    # widened specialization through this path makes CUTLASS DSL 4.7 lowering
    # grow pathologically without changing the supported mask semantics.
    strict_top_left_causal = (mask_flags & ~MASK_PADDED) == MASK_CAUSAL and not params.bottom_right and not params.window_right
    split_p = fp8 and (mask_flags == MASK_NONE or pt_plain_top_left_causal)
    pt_thd_split_p = pt_plain_top_left_causal and bool(mask_flags & MASK_PADDED)
    fused_corr_split_p = mxfp8 and strict_top_left_causal
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b_o = bpe(dtype_o)

    # These register profiles are coupled to the kernel's role topologies;
    # they are not independent public knobs.
    pt_e4_causal_regs = not mxfp8 and params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_CAUSAL and not params.bottom_right
    pt_e5_causal_regs = not mxfp8 and params.dtype_qkv == DTYPE_E5M2 and mask_flags == MASK_CAUSAL and not params.bottom_right
    softmax_regs, softmax_wg1_regs, correction_regs = 240, 136 if split_p else 40, 96
    if fused_corr_split_p:
        softmax_regs, softmax_wg1_regs, correction_regs = 248, 216, 64
    elif mxfp8 and split_p:
        softmax_regs, softmax_wg1_regs, correction_regs = 256, 144, 72
    elif pt_thd_split_p:
        softmax_wg1_regs, correction_regs = 144, 88
    elif pt_e4_causal_regs:
        correction_regs = 64
        if split_p:
            softmax_regs = 216
            softmax_wg1_regs = 168
    elif pt_e5_causal_regs:
        correction_regs = 96 if split_p else 112
    elif mxfp8 and mask_flags != MASK_NONE:
        correction_regs = 64
        if params.dtype_qkv == DTYPE_E5M2 or (params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_CAUSAL):
            softmax_regs = 248
    elif not mxfp8 and fp8 and mask_flags != MASK_NONE:
        correction_regs = 104
    elif not mxfp8 and params.dtype_qkv == DTYPE_E4M3 and mask_flags == MASK_NONE:
        correction_regs = 88

    if fused_corr_split_p:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id, read_tile_arrivers = 12, 4, 64, 8, 11
    elif split_p:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id, read_tile_arrivers = 16, 4, 8, 12, 15
    else:
        total_warps, softmax_wg1_base, correction_warp_base, mma_warp_id = 12, 64, 4, 8
        read_tile_arrivers = 11 if fp8 else 21

    cfg = CfgD256(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=b_o,
        # FP8 uses one M128 CTA per work unit. With a full K/V slice per CTA,
        # two KV stages consume the same SMEM payload as CTA2's four half-slices.
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        Q_SWZ_BYTES=q_swz_bytes(256, b),
        K_SWZ_BYTES=q_swz_bytes(256, b),
        V_SWZ_BYTES=v_swz_bytes(256, 1 if fp8 else 2, b),
        O_SWZ_BYTES=o_swz_bytes(256, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=32 if fp8 else tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=32 if fp8 else tile_k_hw(params.dtype_qkv),
        STAGES_KV=2,
        SOFTMAX_WARPGROUPS=2 if split_p or fused_corr_split_p else 1,
        CORRECTION_WARPS=0 if fused_corr_split_p else 4,
        FUSED_CORR_SPLIT_P=1 if fused_corr_split_p else 0,
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
        # Partial PackGQA is wired in the f16/bf16 d256 kernel; the fp8 / mxfp8
        # siblings pack HEADS_PER_TILE = QH_PER_KH and keep the full-ratio contract.
        PACK_G=_pack_g(params, CfgD256.TILE_M, partial=not fp8),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d256(cfg)
    return cfg, _tma_iters(cfg)


def make_cfg_d256(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256(params, mxfp8=False)


def make_cfg_d256_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    return _make_cfg_d256(params, mxfp8=True)


# ---------------------------------------------------------------------------
# d256 DECODE tile — d_qk, d_v <= 256, SM100 (Blackwell), f16/bf16, swap-AB
# ---------------------------------------------------------------------------
#
# The prefill tile pays for 256 collective Q rows per KV tile whatever the
# number of live rows; a decode step has S_q * G of them (16 for Qwen3.5's
# 32/2 heads at S_q = 1).  The decode tile transposes the problem: the KV
# tokens are the MMA M axis (128 per tile, the standard TMEM lane = key
# layout) and the packed Q rows are the N axis (16 or 32), so BMM1 is
# S^T = K Q^T and BMM2 is O^T = V^T P^T, and the softmax reduces over lanes.
# One CTA per (KV-head group, batch, split) unit, cta_group::1, no cluster.

# Largest packed Q-row count the decode tile COMPILES for (N = 32 keeps the
# 3-slot 64 KiB K/V ring, the Q tile and the two P^T buffers inside 227 KiB).
D256_DECODE_MAX_Q_ROWS = 32
_D256_DECODE_Q_TILES = (16, 32)
# Largest packed Q-row count the adapter ROUTES onto the decode tile.  The
# 32-column tile (two softmax column groups over the same 128 TMEM lanes) is
# compiled and tested at the template level but issue-bound per CTA: at b=32
# x 2 KV heads x 4096 keys (B200) it streams a KV tile in 2.8 us against the
# 16-column tile's 1.75 us -- 90 us unsplit where the prefill tile takes 66
# us -- and its split-2 plan (58 us of GPU time) costs an eager caller 96-103
# us per execute for the second launch.  Until its per-CTA issue rate is
# fixed, S_q x G in (16, 32] stays on the prefill tile, which serves those
# shapes at its previous numbers in both regimes (eager and CUDA-graph
# replay).  Raising this to D256_DECODE_MAX_Q_ROWS routes the wide tile.
D256_DECODE_ROUTED_MAX_Q_ROWS = 16


def decode_d256_q_tile(s_q: int, pack_g: int) -> int:
    """The decode tile's N extent for ``s_q`` tokens packed ``pack_g`` heads per
    token (1 = unpacked), or 0 when the prefill tile serves the graph: no rows,
    or more than D256_DECODE_ROUTED_MAX_Q_ROWS of them (the 32-column tile is a
    valid ``make_cfg_d256_decode`` record but is not routed)."""
    rows = int(s_q) * int(pack_g)
    if rows <= 0 or rows > D256_DECODE_ROUTED_MAX_Q_ROWS:
        return 0
    return next(n for n in _D256_DECODE_Q_TILES if rows <= n)


@dataclass(frozen=True)
class CfgD256Decode:
    # KV tokens per MMA (the M axis); TILE_K / TILE_O are the head-dim envelopes.
    TILE_N: int = 128
    TILE_K: int = 256
    TILE_O: int = 256
    # Packed Q rows per unit (the MMA N axis).
    N_Q: int = 16

    DTYPE_QKV: int = DTYPE_FP16
    DTYPE_O: int = DTYPE_FP16
    BPE: int = 2
    BPE_O: int = 2

    Q_SWZ_BYTES: int = 128
    K_SWZ_BYTES: int = 128
    V_SWZ_BYTES: int = 128
    # P^T rows are N_Q half-precision values: 32 B (N_Q = 16) or 64 B (N_Q = 32).
    P_SWZ_BYTES: int = 32
    TILE_K_HW: int = 16

    # K and V tiles share one ring of 64 KiB slots (K(t), V(t), K(t+1), ...).
    STAGES_KV: int = 3

    MASK_FLAGS: int = MASK_NONE
    WINDOW_LEFT: int = 0
    WINDOW_RIGHT: int = 0
    HAS_SINK: int = 0
    STATS_LOG2: int = 0
    BOTTOM_RIGHT: int = 0

    SEQ_KV_LENS_PRESENT: int = 0
    SEQ_Q_LENS_PRESENT: int = 0

    SPLIT_KV: int = 1
    PACK_GQA: int = 0
    QH_PER_KH: int = 1

    PAGED_KV: int = 0
    PAGE_SIZE: int = 0

    # 4 softmax warps per 16 S^T columns (one group at N_Q = 16, two at 32),
    # then the MMA warp and the TMA warp.
    SOFTMAX_WARPS: int = 4
    MMA_WARP_ID: int = 4
    TMALDG_WARP_ID: int = 5
    TOTAL_WARPS: int = 6
    THREADS_PER_CTA: int = 6 * 32


def _validate_cfg_d256_decode(cfg: CfgD256Decode) -> None:
    fp8 = cfg.DTYPE_QKV in (DTYPE_E4M3, DTYPE_E5M2)
    checks = (
        (not fp8, "d256 decode: f16/bf16 inputs only"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d256 decode: half input requires DTYPE_O == DTYPE_QKV"),
        (cfg.N_Q in _D256_DECODE_Q_TILES, f"d256 decode: N_Q must be one of {_D256_DECODE_Q_TILES}"),
        (cfg.SOFTMAX_WARPS == 4 * (cfg.N_Q // 16), "d256 decode: 4 softmax warps per 16 Q columns"),
        (
            cfg.MMA_WARP_ID == cfg.SOFTMAX_WARPS and cfg.TMALDG_WARP_ID == cfg.SOFTMAX_WARPS + 1 and cfg.TOTAL_WARPS == cfg.SOFTMAX_WARPS + 2,
            "d256 decode: role layout is softmax warps, then the MMA warp, then the TMA warp",
        ),
        (cfg.THREADS_PER_CTA == 32 * cfg.TOTAL_WARPS, "d256 decode: THREADS_PER_CTA must be 32 * TOTAL_WARPS"),
        (cfg.TILE_N == 128, "d256 decode: the KV tile is the 128-lane TMEM layout"),
        (cfg.STAGES_KV >= 2, "d256 decode: the K/V ring needs at least K(t) and V(t) resident"),
        (cfg.P_SWZ_BYTES == cfg.N_Q * cfg.BPE and cfg.P_SWZ_BYTES in (32, 64), "d256 decode: P^T row bytes must be the 32 B or 64 B swizzle atom"),
        (not cfg.PACK_GQA or cfg.QH_PER_KH <= cfg.N_Q, "d256 decode: a packed head group must fit the Q tile"),
        (cfg.QH_PER_KH >= 1, "d256 decode: qh_per_kh must be >= 1"),
        (cfg.SPLIT_KV >= 1, "d256 decode: split_kv must be >= 1"),
        (not cfg.SPLIT_KV > 1 or not cfg.HAS_SINK, "d256 decode: split_kv > 1 with a sink is not supported (the sink would be counted once per split)"),
        (not cfg.PAGED_KV or cfg.SEQ_KV_LENS_PRESENT == 1, "d256 decode: paged KV requires per-batch KV lengths"),
        (
            not cfg.PAGED_KV or (cfg.PAGE_SIZE >= 8 and cfg.PAGE_SIZE % 8 == 0 and (128 % cfg.PAGE_SIZE == 0 or cfg.PAGE_SIZE % 128 == 0)),
            "d256 decode: page_size must be a positive multiple of 8 that divides the 128-row tile or is a multiple of it",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d256_decode(params: TemplateParams) -> Tuple[CfgD256Decode, TmaIters]:
    """Config for sm100/decode_d256_f16.py from the adapter's TemplateParams.

    Backstop only (see TemplateParams): the adapter selects this tile for
    decode-shaped f16/bf16 d256 graphs and keeps every other graph on the
    prefill tile, so a ValueError here is an adapter gap, not a user error.
    ``cta_mma`` / ``sched_policy`` are accepted and unused — the decode tile
    is one cta_group::1 CTA per unit with nothing to schedule.
    """
    if params.decode_q_tile not in _D256_DECODE_Q_TILES:
        raise ValueError(f"d256 decode: decode_q_tile must be one of {_D256_DECODE_Q_TILES}; got {params.decode_q_tile}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError("d256 decode: f16/bf16 inputs only")
    if params.thd_varlen:
        raise ValueError("d256 decode: THD/varlen queries ride the prefill tile")
    if params.pv_bf16 or params.softmax_f16:
        raise ValueError("d256 decode: pv_bf16 / softmax_f16 are quantized-kernel specializations")
    if params.window_left is not None and params.window_left < 0:
        raise ValueError(f"d256 decode: window_left must be >= 0 (or None); got {params.window_left}")
    if params.window_right is not None and params.window_right < 0:
        raise ValueError(f"d256 decode: window_right must be >= 0 (or None); got {params.window_right}")
    if params.bottom_right and params.window_right is None:
        raise ValueError("d256 decode: bottom_right anchors the band's diagonal and requires a right bound (window_right)")
    if params.seq_q_lens_present and not params.seq_kv_lens_present:
        raise ValueError("d256 decode: SEQ_Q_LENS_PRESENT requires SEQ_KV_LENS_PRESENT (padding mask)")
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    b = bpe(params.dtype_qkv)
    softmax_warps = 4 * (int(params.decode_q_tile) // 16)
    cfg = CfgD256Decode(
        N_Q=int(params.decode_q_tile),
        SOFTMAX_WARPS=softmax_warps,
        MMA_WARP_ID=softmax_warps,
        TMALDG_WARP_ID=softmax_warps + 1,
        TOTAL_WARPS=softmax_warps + 2,
        THREADS_PER_CTA=(softmax_warps + 2) * 32,
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_O=bpe(dtype_o),
        P_SWZ_BYTES=int(params.decode_q_tile) * b,
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d256_decode(cfg)
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
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d512(params: TemplateParams) -> Tuple[CfgD512, TmaIters]:
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
    )
    _validate_cfg_d512(cfg)
    return cfg, _tma_iters(cfg)


def make_cfg_d512_mxfp8(params: TemplateParams) -> Tuple[CfgD256, TmaIters]:
    """Build the SM100 D512 block-scale CTA1 configuration."""

    cfg, _ = _make_cfg_d256(params, mxfp8=True)
    # A 512-column MXFP8 accumulator leaves no TMEM columns for block scales.
    # Keep the proven CTA1 M128 pipeline and emit two 256-column O slices.
    cfg = replace(cfg, TILE_K=512, STAGES_KV=1)
    if cfg.CTA_MMA != 1 or cfg.CGA_M != 1:
        raise ValueError("d512 MXFP8 requires one-CTA M128 MMA")
    if cfg.TILE_M != 128 or cfg.TILE_N != 128 or cfg.TILE_K != 512 or cfg.TILE_O != 256:
        raise ValueError("d512 MXFP8 requires M128xN128, K512, and a 256-column output slice")
    if cfg.PACK_GQA:
        raise ValueError("d512 MXFP8 does not support PackGQA")
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
    # V may be BF16 in the experimental MXFP8-QK/BF16-PV path while Q/K
    # retain their one-byte MXFP8 storage.
    BPE_V: int = 2
    PV_BF16: int = 0
    EMIT_AMAX_O: int = 1
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
    STATS_LOG2: int = 0  # LSE stored in base 2 (stats_use_log2)
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
    v = cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N * cfg.BPE_V // cfg.CTA_MMA)
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
            cfg.TILE_K_HW_BMM1 == (32 if _fp8 else 16) and cfg.TILE_K_HW_BMM2 == (16 if cfg.PV_BF16 else (32 if _fp8 else 16)),
            "d128: BMM1 uses K=32 for MXFP8; experimental BF16 PV uses K=16",
        ),
        (cfg.Q_SWZ_BYTES in (64, 128) and cfg.K_SWZ_BYTES in (64, 128), "d128: Q/K swizzle must be 64/128B"),
        (cfg.PV_BF16 in (0, 1) and (not cfg.PV_BF16 or _fp8), "d128: pv_bf16 requires MXFP8 Q/K"),
        (cfg.BPE_V == (2 if cfg.PV_BF16 else cfg.BPE), "d128: V bytes/element must match the PV specialization"),
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
    b_v = 2 if params.pv_bf16 else b
    # FP8/MXFP8 pins the Blackwell K=32 QMMA path (TILE_K_HW=32) and STAGES_KV=4
    # (BPE=1 → 8 KiB/stage, fits 4); f16/bf16 keep 16 / 2.
    tile_k_hw_fp8 = 32 if fp8 else tile_k_hw(params.dtype_qkv)
    cfg = CfgD128(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b_v,
        PV_BF16=int(params.pv_bf16),
        EMIT_AMAX_O=int(params.emit_amax_o),
        BPE_O=b_o,
        CGA_M=params.cta_mma,
        CTA_MMA=params.cta_mma,
        # cga1 has no collective MMA to halve per-CTA K/V, so Q and O must share
        # one slab to stay under the SMEM cap (_validate_cfg_d128 enforces it).
        QO_ALIAS=1 if params.cta_mma == 1 else 0,
        Q_SWZ_BYTES=q_swz_bytes(128, b),
        K_SWZ_BYTES=q_swz_bytes(128, b),
        V_SWZ_BYTES=v_swz_bytes(128, params.cta_mma, b_v),
        O_SWZ_BYTES=o_swz_bytes(128, b_o),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw_fp8,
        TILE_K_HW_BMM2=16 if params.pv_bf16 else tile_k_hw_fp8,
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
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=1 if (params.thd_varlen or params.seq_kv_lens_present) else 0,
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        THD_VARLEN=int(params.thd_varlen),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA is wired in the f16/bf16 d128 kernel; the fp8 / mxfp8
        # siblings pack HEADS_PER_TILE = QH_PER_KH and keep the full-ratio contract.
        PACK_G=_pack_g(params, CfgD128.TILE_M, partial=not fp8),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d128(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d128 decode tile — d_qk = d_v = 128, SM100, cga1, TILES_Q=1 (decode / MTP shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CfgD128Decode(CfgD128):
    """The d128 f16/bf16 DECODE tile (``sm100/decode_d128_f16.py``).

    Same head geometry, masks, paged loader and split/epilogue contract as
    :class:`CfgD128` (incl. partial PackGQA: ``PACK_G`` heads per token
    row-group), but shaped for graphs whose Q rows (times the packed group)
    fit ONE 128-row tile per packed head -- S_q = 1 decode and MTP S_q in
    [2, 8]. The prefill pipeline computes 512 Q rows per cga2 cluster
    (TILES_Q=2 x TILE_M=128 x CTA_MMA=2) and, with 1..G of them live, spends
    its time on dead-row MMA and softmax; this tile computes 128 rows per
    independent CTA:

    - ``TILES_Q=1`` / ``CTA_MMA=1``: one Q slab, one S/P slot, one O
      accumulator; BMM1 and BMM2 are M=128 (a quarter of the prefill cluster's
      per-tile MMA work), and there is no cross-CTA barrier traffic.
    - ``SOFTMAX_WARPGROUPS=1``: the second warpgroup owned sub-tile 1, which no
      longer exists -- 12 warps (softmax 0-3, correction 4-7, MMA 8, TMA-LDG 9,
      TMA-STG 10, scheduler 11), the d256 flavor's layout.
    - ``STAGES_KV=3`` with the Q/O slab aliased: 32 (Q u O) + 3 x (32 K + 32 V)
      = 224 KiB, under the SM100 227 KiB cap. At cga1 every CTA streams full
      K/V tiles, so the deeper ring keeps more of the KV cache in flight per SM
      while the tile is memory-bound.

    Selected by the adapter for the (128, 128) f16/bf16 flavor whenever the
    plan's ``TILE_CGA_M`` knob is 1 (``api_dsl._load_sm100_kernel_module``); the
    heuristics propose cga=1 exactly when ``S_q * pack_g <= 128`` (``pack_g`` =
    the candidate's own packing: ``PACK_G`` packed -- the whole group G, or its
    largest divisor of 128 under partial PackGQA -- 1 unpacked).
    """

    CGA_M: int = 1
    CTA_MMA: int = 1
    QO_ALIAS: int = 1

    TILES_Q: int = 1
    STAGES_KV: int = 3

    SOFTMAX_WARPGROUPS: int = 1
    CORRECTION_WARPS: int = 4

    SOFTMAX_REGS: int = 240
    CORRECTION_REGS: int = 96

    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32

    # One softmax warpgroup @ warps 0-3, correction @ 4-7, single roles @ 8..11.
    SOFTMAX_WG0_BASE: int = 0
    SOFTMAX_WG1_BASE: int = 4  # unused: the kernel dispatches no second warpgroup
    CORR_WARP_BASE: int = 4
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11

    # 4 softmax + 4 corr + 1 MMA + 1 TMALDG + 1 TMASTG = 11 arrivers (cga1).
    READ_TILE_ARRIVERS: int = 11


def _validate_cfg_d128_decode(cfg: CfgD128Decode) -> None:
    """Consistency checks on the d128 decode-tile geometry."""
    checks = (
        (cfg.DTYPE_QKV in (DTYPE_BF16, DTYPE_FP16), "d128 decode: f16/bf16 only (the fp8 families keep the prefill tile)"),
        (cfg.DTYPE_O == cfg.DTYPE_QKV, "d128 decode: DTYPE_O must equal DTYPE_QKV"),
        (cfg.MMA_REGS == cfg.TMALDG_REGS == cfg.TMASTG_REGS == cfg.SCHEDULER_REGS, "d128 decode: MMA/TMALDG/TMASTG/SCHEDULER regs must match"),
        (cfg.MMA_REGS + cfg.CORRECTION_REGS + cfg.SOFTMAX_WARPGROUPS * cfg.SOFTMAX_REGS <= 512, "d128 decode: register budget over 512"),
        (cfg.MMA_REGS % 8 == 0 and cfg.CORRECTION_REGS % 8 == 0 and cfg.SOFTMAX_REGS % 8 == 0, "d128 decode: per-role regs must be multiples of 8"),
        (cfg.CGA_M == 1 and cfg.CTA_MMA == 1, "d128 decode: one independent CTA per tile (cga1)"),
        (cfg.QO_ALIAS == 1, "d128 decode: Q and O share one SMEM slab (pays for the third KV stage)"),
        (cfg.TILES_Q == 1, "d128 decode: TILES_Q must be 1 (one 128-row Q tile per CTA)"),
        (cfg.TILE_M == 128 and cfg.TILE_N == 128 and cfg.TILE_K == 128 and cfg.TILE_O == 128, "d128 decode: 128x128 tiles, d_qk = d_v = 128"),
        (cfg.STAGES_KV == 3, "d128 decode: STAGES_KV must be 3"),
        (
            _d128_smem_bytes(cfg) <= _SM100_MAX_DYN_SMEM,
            f"d128 decode: SMEM {_d128_smem_bytes(cfg) // 1024} KiB over the SM100 {_SM100_MAX_DYN_SMEM // 1024} KiB per-CTA cap",
        ),
        (cfg.SOFTMAX_WARPGROUPS == 1 and cfg.CORRECTION_WARPS == 4, "d128 decode: one softmax warpgroup, four correction warps"),
        (cfg.TOTAL_WARPS == 12 and cfg.THREADS_PER_CTA == 384, "d128 decode: 12 warps / 384 threads"),
        (
            cfg.SOFTMAX_WG0_BASE == 0
            and cfg.CORR_WARP_BASE == 4
            and cfg.MMA_WARP_ID == 8
            and cfg.TMALDG_WARP_ID == 9
            and cfg.TMASTG_WARP_ID == 10
            and cfg.SCHED_WARP_ID == 11,
            "d128 decode: role layout and warp count disagree",
        ),
        (cfg.READ_TILE_ARRIVERS == 11, f"d128 decode: expected READ_TILE_ARRIVERS=11, got {cfg.READ_TILE_ARRIVERS}"),
        (cfg.TILE_K_HW_BMM1 == 16 and cfg.TILE_K_HW_BMM2 == 16, "d128 decode: f16 K=16 MMA phases"),
        (cfg.THD_VARLEN == 0, "d128 decode: dense graphs only (THD keeps the prefill tile)"),
        (
            cfg.Q_SWZ_BYTES == 128 and cfg.K_SWZ_BYTES == 128 and cfg.V_SWZ_BYTES == 128 and cfg.O_SWZ_BYTES == 128,
            "d128 decode: 128 B swizzle on every operand",
        ),
    )
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


def make_cfg_d128_decode(params: TemplateParams) -> Tuple[CfgD128Decode, TmaIters]:
    """Config for ``sm100/decode_d128_f16.py`` -- the d128 flavor at ``cta_mma=1``.

    Backstop, like every ``make_cfg_*``: the (128, 128) f16/bf16 row admits
    cga=1 on dense graphs only, and the adapter routes exactly that
    combination here; anything else raising below is a gap in those gates.
    """
    _validate_params("d128", params)
    if params.cta_mma != 1:
        raise ValueError(f"d128 decode: the decode tile is cga1 only (cta_mma=1); got cta_mma={params.cta_mma}")
    if params.dtype_qkv not in (DTYPE_BF16, DTYPE_FP16):
        raise ValueError(f"d128 decode: f16/bf16 inputs only (DTYPE_QKV 2/3); got {params.dtype_qkv}")
    if params.thd_varlen:
        raise ValueError("d128 decode: THD/varlen is not wired on the decode tile (dense graphs only)")
    if params.pv_bf16 or not params.emit_amax_o:
        raise ValueError("d128 decode: pv_bf16 / emit_amax_o are MXFP8-only experiment axes")
    b = bpe(params.dtype_qkv)
    dtype_o = params.dtype_qkv if params.dtype_o < 0 else params.dtype_o
    cfg = CfgD128Decode(
        DTYPE_QKV=params.dtype_qkv,
        DTYPE_O=dtype_o,
        BPE=b,
        BPE_V=b,
        BPE_O=bpe(dtype_o),
        Q_SWZ_BYTES=q_swz_bytes(128, b),
        K_SWZ_BYTES=q_swz_bytes(128, b),
        V_SWZ_BYTES=v_swz_bytes(128, 1, b),
        O_SWZ_BYTES=o_swz_bytes(128, bpe(dtype_o)),
        RESCALE_THRESHOLD=rescale_threshold(params.dtype_qkv),
        TILE_K_HW_BMM1=tile_k_hw(params.dtype_qkv),
        TILE_K_HW_BMM2=tile_k_hw(params.dtype_qkv),
        MASK_FLAGS=_mask_flags_from(params),
        WINDOW_LEFT=params.window_left or 0,
        WINDOW_RIGHT=params.window_right or 0,
        HAS_SINK=int(params.has_sink),
        STATS_LOG2=int(params.stats_log2),
        BOTTOM_RIGHT=int(params.bottom_right),
        SCHEDULER_POLICY=params.sched_policy,
        SEQ_KV_LENS_PRESENT=int(params.seq_kv_lens_present),
        SEQ_Q_LENS_PRESENT=int(params.seq_q_lens_present),
        SPLIT_KV=int(params.split_kv),
        PACK_GQA=int(params.pack_gqa),
        QH_PER_KH=int(params.qh_per_kh),
        # Partial PackGQA, as on the d128 prefill tile: the kernel packs the
        # largest divisor of the group that divides the 128-row tile
        # (HEADS_PER_TILE = PACK_G; a group sharing no factor with it raises here).
        PACK_G=_pack_g(params, CfgD128Decode.TILE_M, partial=True),
        PAGED_KV=int(params.paged_kv),
        PAGE_SIZE=int(params.page_size),
    )
    _validate_cfg_d128_decode(cfg)
    return cfg, _tma_iters(cfg)


# ---------------------------------------------------------------------------
# d192/d128 flavor — DSv3 MLA logical d_qk = 192, d_v = 128, SM100, cga1/cga2
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
    v = cfg.STAGES_KV * (cfg.TILE_O * cfg.TILE_N * cfg.BPE_V // cfg.CTA_MMA)
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


def d192_square_br_as_tl(params: TemplateParams, *, s_q: int, s_kv: int) -> bool:
    """Whether a D192 bottom-right mask is exactly top-left causal."""

    return (
        params.split_kv == 1
        and not params.thd_varlen
        and not params.seq_q_lens_present
        and not params.seq_kv_lens_present
        and params.window_left is None
        and params.window_right == 0
        and params.bottom_right
        and s_q == s_kv
        and 4096 < s_kv <= 8192
    )


def canonicalize_d192_lowering(
    params: TemplateParams,
    *,
    pertensor: bool,
    s_q: int,
    s_kv: int,
) -> TemplateParams:
    """Apply strictly equivalent D192 lowering canonicalizations."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    window_left = params.window_left
    window_right = params.window_right

    template_window_right = window_right
    if fp8 and pertensor and window_left is None and window_right is None and not params.seq_kv_lens_present:
        # CUTLASS DSL 4.7 does not finish lowering the large-shape FP8
        # MASK_NONE x32 path. 1 << 30 exceeds any dense D192 sequence that
        # fits in SM100 memory while leaving signed-int32 headroom for q + R;
        # it preserves the lowering without making the module key depend on S_kv.
        template_window_right = 1 << 30

    template_bottom_right = False if d192_square_br_as_tl(params, s_q=s_q, s_kv=s_kv) else params.bottom_right

    return replace(
        params,
        window_right=template_window_right,
        bottom_right=template_bottom_right,
    )


def derive_d192_internal_params(
    params: TemplateParams,
    *,
    pertensor: bool,
    batch_size: int,
    h_q: int,
    s_q: int,
    s_kv: int,
) -> TemplateParams:
    """Derive D192-private codegen fields after public knobs are fixed."""

    fp8 = params.dtype_qkv in (DTYPE_E4M3, DTYPE_E5M2)
    pack_gqa_ratio = params.qh_per_kh if params.pack_gqa else 1
    groups = batch_size * h_q // pack_gqa_ratio
    lpt_head_group = 8 if fp8 and not params.thd_varlen and groups % 8 == 0 else 1
    q_rows_per_cluster = cga_tile_m(192, params.cta_mma)
    lpt_q_tiles = (s_q * pack_gqa_ratio + q_rows_per_cluster - 1) // q_rows_per_cluster if fp8 and not params.thd_varlen else 0

    lpt_l2_size_mib = 0
    lpt_l2_8k = params.sched_policy == SCHED_LPT_L2 and not params.thd_varlen and params.split_kv == 1 and s_q == 8192 and s_kv == 8192
    if lpt_l2_8k and pertensor and params.dtype_qkv == DTYPE_E4M3 and groups % 24 != 0 and groups % 16 == 0:
        # At 8K, 60 MiB groups 24 one-byte K/V heads; 40 MiB groups 16 and
        # avoids a short final group for these grids.
        lpt_l2_size_mib = 40
    elif lpt_l2_8k and not fp8 and not params.pack_gqa and groups % 16 == 0:
        # At 8K, each half-precision K/V head occupies 5 MiB. Grouping exactly
        # 16 heads avoids a short final LPT-L2 group on the model grids.
        lpt_l2_size_mib = 80

    return replace(
        params,
        lpt_head_group=lpt_head_group,
        lpt_q_tiles=lpt_q_tiles,
        lpt_l2_size_mib=lpt_l2_size_mib,
    )


def make_cfg_d192(params: TemplateParams) -> Tuple[CfgD192, TmaIters]:
    _validate_params("d192", params)
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


MAKE_CFG = {128: make_cfg_d128, 192: make_cfg_d192, 256: make_cfg_d256, 512: make_cfg_d512}
