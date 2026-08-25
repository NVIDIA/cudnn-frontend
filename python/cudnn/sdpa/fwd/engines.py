# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SDPA-forward capability declarations + spec table.

One engine per architecture x phase x geometry, named

    sdpa_fwd_<phase>_sm<arch>[_d<dqk>[x<dv>]]

(dtype is NOT part of the identity: a cell's engine serves every dtype its
kernel handles — fp16 and bf16 today — via ``Capabilities.dtypes``.)

The shared analyzer (``cudnn.sdpa.graph_analyzer.analyze``) parses the graph
once into :class:`SdpaGraphFacts`; each engine's probe is a cheap field-by-field
candidate match against its :class:`Capabilities` row below. Architecture
resource feasibility remains a lowering responsibility. Adding an engine is
one ``Capabilities``/spec row plus (usually) one kernel template.

An engine is a *lowering strategy*, not a kernel: its ``lower`` hook receives
the parsed facts and returns an executor, and is free to compile one kernel,
pick among several, or chain multiple launches (the THD path already launches
an O-descriptor builder kernel before the main one). Conversely several
engines may share one template. Neither direction is 1:1.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import cudnn

from cudnn.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd.config_sm100 import pack_gqa_supported

# The DSL adapters (api_dsl) and cuda.bindings are LOWERING dependencies, not
# support-check ones: importing them here would drag the CuTe DSL (~1.0 s, 357
# modules) into every process that merely asks whether an engine could serve a
# graph. ENGINE_SPECS therefore names its adapter, and _adapter() resolves it
# at build time. Capabilities/mismatch below stay import-free.
_SM100 = "SdpaFwdDslSm100"
_SM120 = "SdpaFwdDslSm120"
_SM80 = "SdpaFwdDslSm80"


def _adapter(name: str):
    from cudnn.sdpa.fwd import api_dsl

    return getattr(api_dsl, name)


def _cuda_driver():
    from cuda.bindings import driver

    return driver


_LOG = logging.getLogger(__name__)

# Arch ranges the spec rows below serve, inclusive, encoded major*10 + minor as
# in engines/manifest.py. RANGES, not the exact device families that exist
# today: an sm100 kernel runs on the whole sm100 line, so enumerating members
# silently declines the parts that ship later -- Rubin (sm107) and Thor (sm110)
# are meant to reuse these kernels and an exact {(10,0), (10,3)} excluded both.
# Within the range, cc10.3+ additionally enables the fused LDTM.STAT row-max for
# MXFP8 — handled in the lowering, from the device capability.
_BLACKWELL = (100, 119)
_BLACKWELL_GEFORCE = (120, 129)


@dataclass(frozen=True)
class SdpaFwdKnobs:
    """Per-plan tuning request for the SDPA-forward engines.

    This is the operation's knob *vocabulary* — typed fields, no global enum.
    ``None`` means "no preference". Travels as ``PlanConfig.knobs``; each
    engine's :class:`Capabilities` row advertises the domain it honors, and the
    probe rejects the engine for any request outside that domain (a knob is
    honored or the engine is ineligible — never silently degraded).
    """

    sched_policy: Optional[int] = None  # tile-scheduler policy (SCHED_NATURAL, ...)
    tile_m: Optional[int] = None  # Q sequence tile width
    tile_n: Optional[int] = None  # KV sequence tile width
    cga: Optional[int] = None  # cluster size (CTAs cooperating per tile)
    pack_gqa: Optional[bool] = None  # Head packing for GQA/MQA
    # KV-split count: each Q tile's KV range cut into this many chunks, each
    # run by its own CTA, recombined by the split_combine pass. 1 = off.
    split_kv: Optional[int] = None
    # Softmax accumulation precision as a cudnn.data_type value. Served rows
    # declare their domain: the per-tensor FP8 d128 rows serve FLOAT, and the
    # sm107 (Rubin) row additionally HALF — its kernel's f16x2 exponent arm.
    # HALF is numerics-changing, so it is honored on explicit request only,
    # never auto-proposed (see heuristics._softmax_points).
    softmax_precision: Optional[int] = None


@dataclass(frozen=True)
class Capabilities:
    """What one ENGINE can serve — the envelope of graphs (and, later, requested
    tuning knobs) its lowering can honor. An engine spanning several kernels
    declares the union its lowering can actually deliver. Compared
    field-by-field against SdpaGraphFacts in the probe."""

    # Arch RANGE, inclusive, encoded major*10 + minor as in engines/manifest.py.
    # A range and not a set of exact device families: an sm100 kernel runs on
    # everything in the sm100 line, so enumerating the parts that exist today
    # silently declines the ones that ship tomorrow. Rubin (sm107) and Thor
    # (sm110) are meant to reuse these kernels and an exact set excluded both.
    sm_lo: int
    sm_hi: int
    phase: str
    # Head-dim DOMAIN. One engine per arch x dtype family: the head dim is a
    # LOWERING concern — the adapter picks the smallest kernel flavor whose
    # native shape covers the graph (api_dsl._pick_flavor) — not an engine
    # identity. ``d_shapes`` is the set of NATIVE flavor shapes (d_qk, d_v)
    # that lowering picks among.
    d_shapes: frozenset
    # Envelope + alignment rule. When > 0: any graph (d_qk, d_v) componentwise
    # <= some native shape AND a multiple of this is served via TMA
    # zero-padding — the kernel's descriptors carry the ACTUAL extents, so
    # padded contraction columns load as exact zeros (S/softmax unchanged) and
    # O stores past d_v are OOB-clipped. 8 = the TMA 16-byte global-stride
    # rule at 2 bytes/elem (f16/bf16), 16 = the same rule at 1 byte/elem
    # (per-tensor FP8), 1 = no constraint (the SM80 lowering pads host-side).
    # 0 = NO envelope: exact native shapes only (MXFP8, whose SF plumbing is
    # not audited for zero-padding).
    d_pad_multiple: int = 8
    # Shapes whose kernels carry the THD leg. None = THD (when the ``thd``
    # capability is set) serves the same head-dim domain as dense — the f16
    # THD compile key carries the head dims, so THD rides the envelope. A set
    # = THD graphs must match one of these NATIVE shapes exactly: the
    # quantized rows' packed THD compile key carries no head-dim entries
    # (native-tile contract), so their envelope is dense-only.
    thd_d_shapes: Optional[frozenset] = None
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})  # cudnn.data_type, see graph_analyzer
    is_mxfp8: bool = False  # block-scale MXFP8 engine (FP8 in + per-32-block E8M0 SF)
    is_fp8: bool = False  # per-tensor FP8 engine (FP8 in + scalar descales)
    # O dtype domain, declared only by the quantized rows: elsewhere O must
    # equal Q, which facts.uniform_dtype already enforces.
    out_dtypes: frozenset = frozenset()

    # optional features a graph may request
    bias: bool = False
    dropout: bool = False
    score_mod: bool = False
    paged_kv: bool = False
    alibi: bool = False
    block_mask: bool = False
    rng_dump: bool = False
    score_max: bool = False  # per-row/tile score-max side output
    score_sum_exp: bool = False  # per-row/tile sum-of-exp side output
    dynamic_scale: bool = False
    unfuse_fma: bool = False
    seq_q_trim: bool = False
    right_band_widening: bool = False

    causal: bool = False
    bottom_right: bool = False
    swa: bool = False
    padded: bool = False
    sink: bool = False
    # Optional dtype subset for sink support. None means every dtype served by
    # the engine; a subset lets one exact-shape flavor decline an unsupported
    # low-precision sink path without affecting its non-sink coverage.
    sink_dtypes: Optional[frozenset] = None
    stats: bool = False
    # The adapter accepts lse_tensor=None, so a stats-less graph needs no
    # dummy-LSE workspace chunk. Every row sets True. The SM100/SM120 kernels
    # None-specialize the LSE store (compiled out entirely); the SM80 kernels
    # still compute LSE into a kernel-internal buffer when unbound — a
    # pre-existing allocation on that path, tracked with its other
    # TemplateParams-conversion follow-ups.
    lse_optional: bool = False
    # THD lowerings assume FULLY-PACKED storage: the packed addressing is
    # re-derived as prefix(lens) x token stride, and the graph's bound
    # ragged-offset values are never read. TE-style padded THD (offsets
    # from cu_seqlens_padded != cu_seqlens, gaps between sequences) is NOT
    # served — and being runtime data, cannot be declined at plan time.
    thd: bool = False
    # cu_seq_len_q / cu_seq_len_kv (B+1,) prefix sums (cuDNN 9.24+). Serving
    # rows consume the form on THD host-side (lens = adjacent differences of
    # the inherent tolist); dense cu graphs stay declined until the kernels
    # grow a CU read mode (len = cu[b+1] - cu[b]) — see mismatch().
    cu_seq_len: bool = False
    # Dense padded + stats needs the per-batch seq_len_q LSE trim (padded
    # q-rows write LSE=-inf / O=0, cuDNN >= 9.14). Plumbed for the half
    # kernels and the SM120 fp8 kernel via SEQ_Q_LENS_PRESENT; the SM100
    # FP8/MXFP8 kernels lack the epilogue trim, so their specs keep this
    # False.
    padded_stats: bool = False
    # Dense padded graphs carrying per-batch seq_len_q: the kernel's epilogue
    # trims padded q rows (O := 0, LSE := -inf). Rows whose kernel lacks the
    # trim keep False: lower_dsl_prefill then drops the buffer instead of
    # binding it, and _execute runtime-rejects lengths shorter than S_q (a
    # kernel property must live here, not in an adapter-class test — a future
    # row reusing an adapter for a trim-less kernel would silently inherit
    # the wrong answer otherwise).
    dense_seq_q_trim: bool = False
    # s_q == 1 (decode-shaped) graphs; the SM80 prefill kernels are gated off.
    decode: bool = True

    # Escape hatch for kernels whose persistent multi-wave rescheduling is
    # numerically broken: eligible only when the whole grid fits in one wave
    # (B*H_q*ceil(S_q/512) <= SM_count / CTA_MMA). NO CURRENT ROW SETS THIS.
    # Historical user: the fp8 d128 row, whose multi-wave wrong-O bug (14-19%
    # mismatched elements, some tiles left unwritten NaN) was root-caused to
    # the classic-pipeline TMEM stats race — the next tile's prologue BMM1
    # overwrote the S_acc-head (total_max, total_sum) before the correction
    # epilogue read them — and fixed by the mb_stats_read barrier (same fix as
    # the f16 kernel's; see prefill_d128_fp8_sm100.py / _common_sm100.Bars).
    single_wave_only: bool = False
    # Serve ragged S_kv on unmasked graphs by synthesizing a full-length
    # seq_len_kv and lowering through the kernel's padded path (masks the KV
    # tail; mathematically identical for full lengths). Costs the padded-path
    # overhead, so only rows that opt in use it; the KV-tail rule is waived.
    skv_tail_via_padding: bool = False

    # Dense layout envelope this engine accepts:
    #   "bshd"       — Q/K/V/O must be BSHD-physical (stride order 3,1,2,0).
    #   "dense_flex" — any B/H/S stride permutation, padded (oversized)
    #                  strides included, as long as the head dim is
    #                  innermost-contiguous (stride 1) and the strides are
    #                  non-broadcast / non-overlapping (facts.dense_layout;
    #                  see graph_analyzer.dense_layout_ok). The DSL executor
    #                  normalizes such tensors to the kernel's canonical
    #                  BSHD-compact buffers (zero-copy when already BSHD).
    # THD (ragged) graphs always require the BSHD-order packing regardless —
    # the ragged lowering rebuilds packed [1,T,H,D] views and only that
    # packing is defined for it.
    layouts: frozenset[str] = frozenset({"bshd"})
    skv_tile: int = 128  # KV tail rule (waived when padded / causal covers the tail)

    # Tuning-knob domains this engine's lowering honors (see SdpaFwdKnobs).
    sched_policies: frozenset[int] = frozenset({SCHED_NATURAL})
    tile_ms: frozenset[int] = frozenset()
    tile_ns: frozenset[int] = frozenset()
    cgas: frozenset[int] = frozenset()
    pack_gqas: frozenset[bool] = frozenset({False})
    # Split-KV domain. {1} = the axis exists but only "off" is served; rows
    # whose kernels wire the split path AND whose adapter launches the combine
    # widen this (the SM100 f16 rows today).
    split_kvs: frozenset[int] = frozenset({1})
    # Shapes whose kernel flavors wire SplitHelpers. None = every flavor in
    # d_shapes does (f16/SM120). A set = split_kv > 1 is honored only when
    # the graph's dims are covered by a member (the quantized families wire
    # the split path in the d128 flavor only).
    split_d_shapes: Optional[frozenset] = None
    # Softmax-precision domain (cudnn.data_type values). Empty = unserved.
    # Arch-dependent membership (the f16x2 exponent arm exists only in the
    # SM107 sibling kernel) is expressed by SPLITTING the row per arch line —
    # each row declares exactly what its own lowering carries — not by a
    # knob x arch notch here.
    softmax_precisions: frozenset[int] = frozenset()


def _band_covers_kv_tail(facts: "ga.SdpaGraphFacts") -> bool:
    """True when the causal band (plain or right-widened) provably masks every
    KV column >= S_kv, so a ragged KV tail cannot leak into the softmax.

    The last unmasked column is (S_q - 1) + R top-left or (S_kv - 1) + R
    bottom-right (R = the right-band bound, 0 for plain causal)."""
    if not (facts.causal or facts.right_band_widening):
        return False
    r = facts.right_bound or 0
    if facts.bottom_right:
        return r == 0
    return facts.s_q + r <= facts.s_kv


def mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> Optional[str]:
    """First reason this engine is not a candidate for these facts and tuning
    knobs, or ``None`` when lowering should perform the final feasibility check.

    Returns a human-readable reason string rather than a bool on purpose:
    with many engine rows, "why was my engine not eligible" is the first
    debugging question, and the strict-select error surfaces this string.
    Knob requests outside the engine's candidate domain are rejected here;
    architecture-specific resource checks remain in the adapter and must never
    silently degrade a requested value.
    """
    if facts.invalid:
        return facts.invalid
    if facts.is_backward:
        return "this engine serves sdpa() forward graphs only"
    if knobs is not None:
        if not isinstance(knobs, SdpaFwdKnobs):
            return f"knob request is a {type(knobs).__name__}, not SdpaFwdKnobs — wrong operation's vocabulary"
        for value, domain, label in (
            (knobs.sched_policy, capabilities.sched_policies, "sched_policy"),
            (knobs.tile_m, capabilities.tile_ms, "tile_m"),
            (knobs.tile_n, capabilities.tile_ns, "tile_n"),
            (knobs.cga, capabilities.cgas, "cga"),
            (knobs.pack_gqa, capabilities.pack_gqas, "pack_gqa"),
            (knobs.split_kv, capabilities.split_kvs, "split_kv"),
            (knobs.softmax_precision, capabilities.softmax_precisions, "softmax_precision"),
        ):
            if value is not None and value not in domain:
                # key=int: knob domains mix plain ints with cudnn.data_type
                # members (softmax_precision), and the pybind enum defines no
                # ordering of its own.
                return f"requested {label}={value} is outside this engine's domain {sorted(domain, key=int)}"
        if knobs.split_kv is not None and knobs.split_kv > 1:
            # Facts x knobs: the split path is structurally dense-only (the
            # per-split LSE is the combine weight; the THD/sink/padded paths
            # do not produce per-split partials). Declined HERE so a split
            # request never reaches a kernel that cannot honor it.
            if facts.thd or facts.has_sink or facts.padded or facts.seq_q_trim:
                return "split_kv > 1 serves dense, unpadded, sink-free graphs only"
            if capabilities.skv_tail_via_padding and facts.s_kv % (capabilities.skv_tile or 128) != 0 and not _band_covers_kv_tail(facts):
                # The lowering would serve this ragged S_kv through the padded
                # kernel path (synthesized per-batch KV lengths) — the same
                # path the split cannot ride. Mirror lower_dsl_prefill's
                # synth_kv_padding predicate so the plan is never listed.
                return "split_kv > 1 cannot ride the synthesized KV-tail padding this S_kv needs"
            if (facts.is_fp8 or facts.is_mxfp8) and facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16):
                # The combine reduces partials in half precision; reducing
                # QUANTIZED partials would lose what the split must be
                # numerically neutral about.
                return "split_kv > 1 on a quantized graph requires a bf16/fp16 O"
            if capabilities.split_d_shapes is not None and not any(facts.d_qk <= sq and facts.d_v <= sv for sq, sv in capabilities.split_d_shapes):
                return (
                    f"split_kv > 1 is wired only in the {sorted(capabilities.split_d_shapes)} kernel " f"flavors; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
                )
        if knobs.pack_gqa:
            if facts.thd:
                return "PackGQA is currently not supported for THD/ragged graphs"
            _pg_tile_m = knobs.tile_m if knobs.tile_m is not None else max(capabilities.tile_ms)
            if not pack_gqa_supported(facts.h_q, facts.h_kv, _pg_tile_m):
                return f"PackGQA requires h_q/h_kv to divide tile_m: h_q/h_kv = {facts.h_q}/{facts.h_kv} does not tile at tile_m={_pg_tile_m}"
    cc = facts.device_cc
    sm = None if cc is None else cc[0] * 10 + cc[1]
    if sm is None or not (capabilities.sm_lo <= sm <= capabilities.sm_hi):
        return f"requires SM{capabilities.sm_lo}-{capabilities.sm_hi}; current device is {cc}"
    installed, version = cutedsl_state()
    if not installed:
        # Said HERE, not when lowering imports the adapter: a decline at build
        # is honest but late -- the plan is already in the ranked list.
        return "requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed"
    if cutedsl_too_old(version):
        want = ".".join(str(v) for v in CUTEDSL_MIN_VERSION)
        return f"requires nvidia-cutlass-dsl >= {want}; found {version[1]}"
    shapes = sorted(capabilities.d_shapes)
    if capabilities.d_pad_multiple:
        # Envelope family: native flavor shapes are upper bounds (TMA
        # zero-padding semantics — see Capabilities.d_shapes/d_pad_multiple);
        # the lowering picks the smallest covering flavor.
        if not any(facts.d_qk <= sq and facts.d_v <= sv for sq, sv in capabilities.d_shapes):
            return f"no kernel-flavor envelope covers (D_QK={facts.d_qk}, D_V={facts.d_v}); native shapes: {shapes}"
        m = capabilities.d_pad_multiple
        if m > 1 and (facts.d_qk % m != 0 or facts.d_v % m != 0):
            return (
                f"envelope zero-padding requires D_QK/D_V multiples of {m} (TMA 16-byte "
                f"global-stride constraint); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            )
    elif (facts.d_qk, facts.d_v) not in capabilities.d_shapes:
        return f"serves exact native shapes {shapes} (no envelope padding); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    if facts.thd and capabilities.thd_d_shapes is not None and (facts.d_qk, facts.d_v) not in capabilities.thd_d_shapes:
        return (
            f"THD (ragged) rides the packed native-tile leg on this engine "
            f"(shapes {sorted(capabilities.thd_d_shapes)}); the head-dim envelope is dense-only"
        )
    if facts.s_q == 1 and not capabilities.decode:
        return "s_q == 1 (decode) is out of scope for the SM80 prefill kernels"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if (facts.is_mxfp8, facts.is_fp8) != (capabilities.is_mxfp8, capabilities.is_fp8):
        quant = "block-scale MXFP8 (sdpa_mxfp8)" if capabilities.is_mxfp8 else "per-tensor FP8 (sdpa_fp8)" if capabilities.is_fp8 else "half (sdpa)"
        return f"this engine serves only {quant} graphs"
    if (capabilities.is_fp8 or capabilities.is_mxfp8) and facts.dtype_o not in capabilities.out_dtypes:
        return f"O dtype {facts.dtype_o} not in {sorted(str(d) for d in capabilities.out_dtypes)}"
    if not facts.uniform_dtype:
        return "K/V dtypes must match Q" if (facts.is_mxfp8 or facts.is_fp8) else "K/V/O dtypes must match Q"
    if facts.thd:
        # Ragged packing is BSHD-order by construction; the relaxation is
        # dense-only (the THD lowering rebuilds packed [1,T,H,D] views).
        if not facts.bshd_layout:
            return "THD (ragged) Q/K/V/O must be BSHD-physical (stride order 3,1,2,0)"
    elif "dense_flex" in capabilities.layouts:
        if not facts.dense_layout:
            return (
                "Q/K/V/O must have the head dim innermost-contiguous (stride 1) and "
                "non-broadcast, non-overlapping strides (any B/H/S order, padded strides allowed)"
            )
    elif "bshd" in capabilities.layouts and not facts.bshd_layout:
        return "Q/K/V/O must be BSHD-physical (stride order 3,1,2,0)"

    for fact, cap, label in (
        (facts.has_bias, capabilities.bias, "bias"),
        (facts.has_dropout, capabilities.dropout, "dropout"),
        (facts.has_score_mod, capabilities.score_mod, "score_mod"),
        (facts.has_paged_kv, capabilities.paged_kv, "paged attention"),
        (facts.has_alibi, capabilities.alibi, "ALiBi"),
        (facts.has_block_mask, capabilities.block_mask, "block_mask"),
        (facts.has_rng_dump, capabilities.rng_dump, "rng_dump"),
        (facts.has_score_max, capabilities.score_max, "score_max output"),
        (facts.has_score_sum_exp, capabilities.score_sum_exp, "score_sum_exp output"),
        (facts.dynamic_scale, capabilities.dynamic_scale, "tensor attn_scale"),
        (facts.has_unfuse_fma, capabilities.unfuse_fma, "unfuse_fma"),
        (facts.seq_q_trim, capabilities.seq_q_trim, "seq_len_q without padding mask"),
        (facts.right_band_widening, capabilities.right_band_widening, "causal right-band widening"),
        (facts.causal, capabilities.causal, "causal mask"),
        (facts.window_left is not None, capabilities.swa, "sliding window"),
        (facts.padded, capabilities.padded, "padding mask"),
        (facts.has_sink, capabilities.sink, "sink token"),
        (facts.wants_stats, capabilities.stats, "stats output"),
        (facts.thd, capabilities.thd, "THD / ragged"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

    if facts.has_sink and capabilities.sink_dtypes is not None and facts.dtype not in capabilities.sink_dtypes:
        return f"sink token with dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.sink_dtypes)}"

    if facts.has_bias and capabilities.bias and facts.bias_t is not None:
        # uniform_dtype covers K/V/O only; the serving adapters compile the
        # bias load as fp32 or the io dtype, so anything else must decline
        # HERE, not ValueError at execute.
        bias_dt = facts.bias_t.get_data_type()
        if bias_dt not in (cudnn.data_type.FLOAT, facts.dtype):
            return f"bias dtype {bias_dt} must be fp32 or match the Q/K/V dtype ({facts.dtype})"

    if facts.right_band_widening and facts.right_bound is not None and facts.right_bound < 0:
        return f"negative diagonal_band_right_bound ({facts.right_bound}) is not supported"

    if facts.has_cu_seq_len:
        # cu_seq_len_* ((B+1,) prefix sums, cuDNN 9.24+). The THD lowering
        # consumes either length form host-side; the dense kernels' CU read
        # mode (len = cu[b+1] - cu[b]) is not plumbed yet, so dense cu graphs
        # stay declined even on serving rows.
        if not capabilities.cu_seq_len:
            return "graph uses cu_seq_len_q / cu_seq_len_kv, which this engine does not support"
        if not facts.thd:
            return "cu_seq_len_* on dense graphs is not supported yet (kernel CU read mode not plumbed)"
        if (facts.seq_q_t is not None and facts.cu_seq_q_t is not None) or (facts.seq_kv_t is not None and facts.cu_seq_kv_t is not None):
            return "seq_len_* and cu_seq_len_* on the same side is ambiguous (backend precedence is not replicated here)"

    if facts.amax_s_t is not None:
        # The FROST FP8 kernels no longer compute Amax_S (dropped: nothing
        # consumed it and the atomicMax serialized the epilogue); a graph that
        # DECLARES the output must go to an engine that writes it.
        return "graph requests the Amax_S output, which the FROST engines do not produce"

    if facts.bottom_right:
        if not (facts.causal or facts.right_band_widening):
            return "bottom-right alignment requires a causal upper bound (plain or right-widened)"
        if not capabilities.bottom_right:
            return "graph uses bottom-right causal, which this kernel does not support"
    if facts.padded and facts.wants_stats and not facts.thd and not capabilities.padded_stats:
        return "padding mask with generate_stats is not supported yet (per-batch seq_len_q LSE trim not plumbed)"

    if facts.stats_t is not None and not facts.thd:
        if facts.stats_t.get_data_type() != cudnn.data_type.FLOAT:
            return f"stats must be fp32; got {facts.stats_t.get_data_type()}"
        stats_dim = tuple(facts.stats_t.get_dim())
        stats_stride = tuple(facts.stats_t.get_stride())
        expected_dim = (facts.b, facts.h_q, facts.s_q, 1)
        if stats_dim != expected_dim:
            return f"stats must be (B, H_q, S_q, 1) = {expected_dim}; got {stats_dim}"
        if not ga.dense_layout_ok(stats_dim, stats_stride):
            return (
                "stats must use a dense-compatible B/H/S permutation or padded layout "
                f"with non-broadcast, non-overlapping-by-span strides; got {stats_stride}"
            )

    if capabilities.single_wave_only:
        # See Capabilities.single_wave_only. 512 = TILES_Q * TILE_M * CTA_MMA
        # rows per cluster; resident clusters = SM count / CTA_MMA (one CTA per
        # SM at this kernel's SMEM footprint). Unknown SM count -> stay gated.
        clusters = facts.b * facts.h_q * ((facts.s_q + 511) // 512)
        resident = (facts.device_sm_count or 0) // 2
        if clusters > resident:
            return (
                f"launch needs {clusters} Q-tile clusters but only {resident} fit in one wave; "
                "the fp8 kernel's persistent multi-wave rescheduling is numerically broken (gated)"
            )

    if capabilities.skv_tile and facts.s_kv % capabilities.skv_tile != 0 and not capabilities.skv_tail_via_padding:
        if not (facts.padded or _band_covers_kv_tail(facts)):
            return f"S_kv ({facts.s_kv}) must be a multiple of {capabilities.skv_tile} unless a padding mask is given or the causal mask covers the KV tail"
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    # Lowering strategy: facts -> executor. A future engine may select between
    # kernels (e.g. decode vs prefill by S_q) or chain several launches under
    # one name.
    lower: "Callable[[EngineSpec, ga.SdpaGraphFacts, Optional[SdpaFwdKnobs]], Any]"


def _sm100_spec() -> EngineSpec:
    """f16/bf16 SM100-family engine: ONE row; the adapter picks the smallest
    kernel flavor (d128 / d192xd128 / d256 / d512) covering the graph's head
    dims (api_dsl._pick_flavor), and every flavor serves its envelope via TMA
    zero-padding. sm_hi=106: no f16 lowering exists on the Rubin line — when
    one lands it gets its own row (the per-arch-line row doctrine)."""
    return EngineSpec(
        name="sdpa_fwd_prefill_sm100",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=106,
            phase="prefill",
            d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            lse_optional=True,
            thd=True,
            cu_seq_len=True,
            padded_stats=True,
            dense_seq_q_trim=True,
            # Ragged S_kv with an uncovered tail is served through the padded
            # path with synthesized full-length per-batch KV lengths (see
            # lower_dsl_prefill's synth_kv_padding) — mathematically identical,
            # costs only the padded-path overhead. Same mechanism the FP8 row
            # has always used.
            skv_tail_via_padding=True,
            # The f16/bf16 lowering serves any dense B/H/S stride permutation
            # (padded strides included) with the head dim innermost; the
            # FP8/MXFP8 rows stay on the strict BSHD gate until their padded /
            # scale-factor paths are validated against relaxed layouts.
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # All four f16 flavor kernels wire SplitHelpers, and the adapter
            # carves the partial slabs + launches split_combine_sm100 when
            # split_kv > 1 (dense f16 only; see mismatch's facts x knobs gate).
            split_kvs=frozenset({1, 2, 4}),
            pack_gqas=frozenset({False, True}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_mxfp8_spec() -> EngineSpec:
    """Block-scale MXFP8 engine (E4M3/E5M2 + per-32-block E8M0 SF).

    THD/varlen (d128/d128 only — the d192/d128 kernel is dense-only) rides the
    shared packed lowering (write_thd_meta envelope design, issue #552; packed
    Q/K/V/O contract only). The SF tensors travel PACKED
    per-sequence-TILE-padded ([1, H, Σ_b ceil(S_b/128), SF_SMEM] tile sequences
    in cu_seqlens order — see api_dsl._reshape_sf_packed); the graph's declared
    SF dims stay the dense capacity, like the ragged Q/K/V storage.
    """

    return EngineSpec(
        name="sdpa_fwd_prefill_sm100_mxfp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=106,  # no Rubin MXFP8 lowering
            phase="prefill",
            # Exact native shapes only (d_pad_multiple=0): the SF plumbing is
            # not audited for envelope zero-padding.
            d_shapes=frozenset({(128, 128), (192, 128)}),
            d_pad_multiple=0,
            # Only the d128 kernel carries the write_thd_meta THD leg and
            # wires SplitHelpers; the d192x128 file is dense-only.
            thd_d_shapes=frozenset({(128, 128)}),
            split_d_shapes=frozenset({(128, 128)}),
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            is_mxfp8=True,
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            lse_optional=True,
            thd=True,
            cu_seq_len=True,
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # The split path also needs a half-precision O (mismatch's
            # facts x knobs gate) and rides the d128 flavor (split_d_shapes).
            split_kvs=frozenset({1, 2, 4}),
            # PackGQA is currently not supported for the MXFP8 SDPA engine:
            # the F8_128x4 sf_q scale-factor atom bundles 128 rows of ONE
            # head, so a packed tile's interleaved (token, head) rows cannot
            # gather their scale factors at token granularity.
            pack_gqas=frozenset({False}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_fp8_spec(*, arch: str = "sm100") -> EngineSpec:
    """Per-tensor FP8 engine with scalar descales.

    ONE row per ARCH LINE (``arch``: "sm100" = pre-Rubin Blackwell 100-106,
    "sm107" = Rubin line 107-119): the two lowerings genuinely diverge and a
    shared row could only describe their union with knob x arch notches.
    Within a row the head dim is a LOWERING concern — the adapter picks the
    kernel flavor (d128 or d192xd128) covering the graph. Each row declares
    exactly what its own kernels carry:

    - d_shapes: the sm100 row picks between the d128 and d192xd128 flavors;
      Rubin has only the d128 sibling, so a Rubin d192 graph is ineligible
      at probe time instead of a late build error.
    - The ENVELOPE (d_pad_multiple=16, the TMA 16-byte global-stride rule at
      1 byte/elem): smaller head dims ride TMA zero-padding — exact in FP8,
      and the descales are scalars so no per-column plumbing is affected.
      THD keeps native dims (thd_d_shapes: the packed THD compile key
      carries no head-dim entries).
    - softmax_precisions: the f16x2 exponent arm lives only in the SM107
      sibling kernel, so only that row admits HALF. FLOAT is the pipeline
      every flavor already runs.
    - split_kvs / split_d_shapes: only the SM100 d128 kernel wires
      SplitHelpers; the SM107 sibling has no split path yet, and the
      d192x128 file forks its own scheduler and has none either.
    - sched_policies: the LPT/LPT_L2 remap is not yet ported to the SM107
      sibling (issue #653); {NATURAL} keeps requests honest AND routes the
      graph path around the un-ported derivation (place() hands the adapter
      an explicit policy from this domain).

    Padding mask (per-batch ``seq_len_kv`` → KV-side masking) is supported: KV-only
    padding leaves every query row real, so each row's total_sum > 0 and the
    per-row softmax normalization stays well-defined — no
    fully-masked row can poison the global amax.  THD/varlen (d128/d128 only —
    the d192/d128 kernel is dense-only) rides the shared packed lowering
    (write_thd_meta envelope design, issue #552; packed Q/K/V/O contract
    only) — on cc10.7 (Rubin) through the SM107 sibling kernel, which carries
    the same THD leg.
    """

    rubin_row = arch == "sm107"
    return EngineSpec(
        name=f"sdpa_fwd_prefill_{arch}_fp8",
        capabilities=Capabilities(
            # Ranges, not the parts that exist today (see sm_lo above): the
            # split point is the Rubin line — 100-106 runs the SM100 modules,
            # 107-119 the SM107 sibling.
            sm_lo=107 if rubin_row else _BLACKWELL[0],
            sm_hi=_BLACKWELL[1] if rubin_row else 106,
            phase="prefill",
            d_shapes=frozenset({(128, 128)}) if rubin_row else frozenset({(128, 128), (192, 128)}),
            d_pad_multiple=16,
            thd_d_shapes=frozenset({(128, 128)}),
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            is_fp8=True,
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            lse_optional=True,
            thd=True,
            cu_seq_len=True,
            # The fp8 kernel lacks the SEQ_Q_LENS_PRESENT epilogue trim, but its
            # only reachable padded+stats population is KV-only padding with
            # full-length seq_len_q (the fp8 suite; test_mhas_v2 fp8 padding
            # comes only via paged/THD, both ineligible here), where the trim
            # is a no-op. Dense fp8 padding with SHORT seq_len_q + stats would
            # report untrimmed LSE — plumb the trim before relying on it.
            padded_stats=True,
            # Multi-wave launches are served: the former single_wave_only gate
            # (wrong O past one wave) was removed after the kernel's TMEM stats
            # race was fixed with the mb_stats_read barrier (verified on the
            # gated 132/192/200-cluster repros, 3x each).
            skv_tail_via_padding=True,
            # LPT/LPT_L2 remap is not yet ported to the SM107 sibling (issue
            # #653) — its row serves NATURAL only until the port lands.
            sched_policies=(frozenset({SCHED_NATURAL}) if rubin_row else frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # f16x2-softmax arm: only the SM107 sibling kernel carries the
            # path (MUFU EX2.F16x2 exists below cc10.7 but no other file wires
            # it). FLOAT is the f32 pipeline every flavor already runs.
            softmax_precisions=(frozenset({cudnn.data_type.FLOAT, cudnn.data_type.HALF}) if rubin_row else frozenset({cudnn.data_type.FLOAT})),
            # Split partials reduce in half precision, so mismatch()'s
            # facts x knobs gate additionally requires a bf16/fp16 O on the
            # quantized rows; split_d_shapes pins it to the d128 flavor.
            split_kvs=frozenset({1}) if rubin_row else frozenset({1, 2, 4}),
            split_d_shapes=frozenset({(128, 128)}),
            pack_gqas=frozenset({False, True}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm80_spec() -> EngineSpec:
    """SM80 (A100) prefill row: lowers through ``lower_dsl_prefill`` onto the
    ``SdpaFwdDslSm80`` adapter (``fwd/api_dsl.py``), which owns kernel-flavor
    selection (gptoss/llama/dsv3/qwen), host-side head-dim padding (hence
    ``d_pad_multiple=1``), BHSD<->BSHD normalization, and per-shape kernel
    caching — the CuTe-DSL JIT happens on the first execute.  THD graphs are
    gated off (the standalone wrapper's varlen path serves THD); knob domains
    are empty (no tunables wired)."""
    return EngineSpec(
        name="sdpa_fwd_prefill_sm80",
        capabilities=Capabilities(
            sm_lo=80,
            sm_hi=80,  # A100 exactly: the kernels assume its 164 KiB opt-in SMEM
            phase="prefill",
            d_shapes=frozenset({(256, 256)}),  # flavor envelopes; host-side zero-padding
            d_pad_multiple=1,
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            bias=True,
            right_band_widening=True,
            causal=True,
            bottom_right=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            padded_stats=True,
            decode=False,
            # The kernels implement the dense padded-Q trim natively
            # (per-batch ``seq_len_q`` forward kwarg): rows >= seq_len_q[b]
            # are written explicitly by the kernel (O := 0, LSE := -inf).
            dense_seq_q_trim=True,
            lse_optional=True,
            layouts=frozenset({"bshd", "dense_flex"}),
            skv_tile=0,  # the kernels' is_even_k path serves ragged S_kv
            # The static-grid remap serves all three policies (the template's
            # sched_policy field); the adapter maps the explicit int to its
            # kernel token and derives only when the knob is None.
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM80),
    )


def _sm120_spec() -> EngineSpec:
    from cudnn.sdpa.fwd.config_sm120 import SUPPORTED_HEAD_TILES

    return EngineSpec(
        name="sdpa_fwd_prefill_sm120",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            phase="prefill",
            # The kernel picks its Q/K and V head tiles independently, so the
            # native shapes are the cross product of the supported tiles.
            d_shapes=frozenset((tq, tv) for tq in SUPPORTED_HEAD_TILES for tv in SUPPORTED_HEAD_TILES),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            swa=True,
            right_band_widening=True,
            padded=True,
            sink=True,
            stats=True,
            lse_optional=True,
            padded_stats=True,
            dense_seq_q_trim=True,
            thd=True,
            # No KV-tail rule: the kernel walks KV tiles right-to-left and its
            # first (masked) step always covers the rightmost — and therefore
            # any partial — tile, comparing columns against seqlen_k regardless
            # of mask flags. Ragged S_kv is served natively with no synthesized
            # padding and no padded-path cost.
            skv_tile=0,
            cu_seq_len=True,
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            # The kernel's inline chunking + the shared split_combine pass
            # (the combine is one block per row — arch-agnostic). The config
            # backstop bars a split under the LPT remaps, so the heuristic's
            # split sets ride SCHED_NATURAL.
            split_kvs=frozenset({1, 2, 4}),
            tile_ms=frozenset({64, 128}),
            tile_ns=frozenset({64, 128}),
            cgas=frozenset({1}),
            pack_gqas=frozenset({False, True}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM120),
    )


def analyze_for(spec: EngineSpec, graph, knobs: Optional[SdpaFwdKnobs] = None):
    """``(facts, reason)``: the parsed graph and the first reason ``spec``
    cannot serve it under ``knobs`` (``None`` when it can).

    The single eligibility entry point, shared by :func:`probe`, :func:`build`
    and ``engine.FrostSdpaFwdEngine.check_support``. ``knobs`` is the plan's
    tuning request (``PlanConfig.knobs``), ``None`` for no preference.
    """
    # The record validate() attached, not a fresh parse: one per graph, shared
    # with whatever ranked these plans before this engine was imported.
    facts = graph._facts_for(ga.analyze)
    if facts is None:
        return None, "graph is not a single sdpa() forward node"
    return facts, mismatch(spec.capabilities, facts, knobs)


def build(spec: EngineSpec, graph, knobs: Optional[SdpaFwdKnobs] = None):
    """Lower ``spec`` for ``graph``, or raise the bare ineligibility reason (the
    caller — the engine — names itself in the message)."""
    facts, reason = analyze_for(spec, graph, knobs)
    if reason is not None:
        raise ValueError(reason)
    return spec.lower(spec, facts, knobs)


def lower_dsl_prefill(
    spec: EngineSpec,
    facts: "ga.SdpaGraphFacts",
    knobs: Optional[SdpaFwdKnobs] = None,
    api_type: str = _SM100,
):
    """Lower one selected SDPA prefill engine through its DSL adapter.

    Every architecture adapter implements the same constructor and execution
    interface — the keyword contract is declared as ``SdpaFwdDsl.execute``.
    ``EngineSpec.lower`` may bind a different implementation through
    ``api_type``; descriptor conversion, adapter lifecycle, variant-pack binding,
    and launch construction remain shared here.
    """
    from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver, ws_align

    # KV-tail via synthesized padding (see Capabilities.skv_tail_via_padding):
    # a ragged S_kv with no mask covering the tail is served through the
    # kernel's padded path with per-batch lengths pinned to the full S_kv.
    skv_tile = spec.capabilities.skv_tile or 128
    synth_kv_padding = (
        spec.capabilities.skv_tail_via_padding and not facts.padded and not facts.thd and facts.s_kv % skv_tile != 0 and not _band_covers_kv_tail(facts)
    )

    seq_q_t = facts.seq_q_t if facts.padded else None
    seq_kv_t = facts.seq_kv_t if facts.padded else None
    # Mirrors the seq_q_lens_present constructor argument below. Execute
    # forwards seq_q only when the compiled specialization consumes it (or THD,
    # which sources cu_seqlens from it) — the adapter rejects mismatches, so a
    # buffer a trim-less kernel can't honor is dropped here rather than
    # erroring at execute (the row declares plumbed-ness; see
    # Capabilities.dense_seq_q_trim).
    seq_q_lens_present = facts.padded and not facts.thd and facts.seq_q_t is not None and spec.capabilities.dense_seq_q_trim
    api = _adapter(api_type)(
        sample_q=ga.tensor_desc_from_ir(facts.q_t, name="q"),
        sample_k=ga.tensor_desc_from_ir(facts.k_t, name="k"),
        sample_v=ga.tensor_desc_from_ir(facts.v_t, name="v"),
        sample_o=ga.tensor_desc_from_ir(facts.o_t, name="o"),
        sample_lse=ga.tensor_desc_from_ir(facts.stats_t, "lse") if facts.stats_t is not None else None,
        # A right-widened band lowers as the causal mask with a BAND_RIGHT
        # diagonal offset (facts.causal is False when right_bound > 0).
        is_causal=facts.causal or facts.right_band_widening,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        window_size_right=(facts.right_bound if facts.right_band_widening else None),
        scale_softmax=facts.scale,
        seq_kv_lens_present=facts.padded or synth_kv_padding,
        # Dense padded-Q trim (q rows >= seq_len_q[b] -> O := 0, LSE := -inf):
        # enabled whenever a dense padded graph carries per-batch Q lengths.
        # THD carries Q lengths via cu_seqlens; the SM100 FP8/MXFP8 kernels
        # are not plumbed (see seq_q_lens_present above).
        seq_q_lens_present=seq_q_lens_present,
        # cu_seq_len form (THD-only; the probe declined dense cu graphs): the
        # adapter's seq-lens execute arguments carry (B+1,) prefix sums.
        cu_seq_q_lens=facts.cu_seq_q_t is not None,
        cu_seq_kv_lens=facts.cu_seq_kv_t is not None,
        has_sink=facts.has_sink,
        thd=facts.thd,
        dtype_o=facts.dtype_o if (facts.is_mxfp8 or facts.is_fp8) else None,
        pertensor_fp8=facts.is_fp8,
        sched_policy=knobs.sched_policy if knobs is not None else None,
        tile_m=knobs.tile_m if knobs is not None else None,
        tile_n=knobs.tile_n if knobs is not None else None,
        cga=knobs.cga if knobs is not None else None,
        pack_gqa=knobs.pack_gqa if knobs is not None else None,
        split_kv=knobs.split_kv if knobs is not None else None,
        softmax_precision=knobs.softmax_precision if knobs is not None else None,
        # SM80-only PLAN-TIME axes (bias presence/dtype are compile-time
        # specializations of that template): forwarded only to adapters whose
        # constructor declares them — every other row's mismatch gated the
        # operands off already.
        **(
            {
                "bias_present": facts.bias_t is not None,
                "bias_fp32": facts.bias_t is not None and facts.bias_t.get_data_type() == cudnn.data_type.FLOAT,
            }
            if "bias_present" in inspect.signature(_adapter(api_type).__init__).parameters
            else {}
        ),
    )
    api.check_support()  # raises ValueError / NotImplementedError if unsupported
    api.compile()

    # Workspace requirement for the compiled geometry: every per-execute scratch
    # buffer is carved from the CALLER's workspace, so its size is fixed here at
    # build time and recorded on the executor as ``workspace_bytes`` — that
    # number is what the plan's CompiledPlan.get_workspace_size() reports.
    #   - synthesized seq_len_kv (skv_tail_via_padding rows): b int32.
    #   - api-level scratch (api.scratch_workspace_bytes()): the dense padded
    #     [seq_kv|seq_q] combine and the THD metadata/LSE buffers.
    # No dummy-LSE chunk: every lower_dsl_prefill row is lse_optional (the
    # kernels None-specialize the LSE argument and compile the store out), so
    # a stats-less graph binds no LSE buffer at any level.
    synth_kv_bytes = ws_align(facts.b * 4) if synth_kv_padding else 0
    api_scratch_bytes = api.scratch_workspace_bytes()
    total_workspace_bytes = synth_kv_bytes + api_scratch_bytes

    # SM80-only feature operand (bias): the row's capability gate admitted it,
    # and the adapter's execute() declares the matching optional keyword —
    # forwarded below only when both hold. ALiBi / block_mask / score-stats
    # graphs never reach a FROST row (every capability row declines them, so
    # the backend serves them).
    _extra_exec_keys = {"bias_tensor"} & set(inspect.signature(type(api).execute).parameters)

    binding = ga.SdpaBinding(
        q=facts.q_t,
        k=facts.k_t,
        v=facts.v_t,
        o=facts.o_t,
        stats=facts.stats_t,
        sink_token=facts.sink_t,
        seq_len_kv=seq_kv_t,
        seq_len_q=seq_q_t,
        cu_seq_len_q=facts.cu_seq_q_t,
        cu_seq_len_kv=facts.cu_seq_kv_t,
        bias=facts.bias_t,
        sf_q=facts.sf_q_t,
        sf_k=facts.sf_k_t,
        sf_v=facts.sf_v_t,
        amax_o=facts.amax_o_t,
        descale_q=facts.descale_q_t,
        descale_k=facts.descale_k_t,
        descale_v=facts.descale_v_t,
        scale_o=facts.scale_o_t,
    )

    def _ir_view(buf, ir_t):
        """Reinterpret a variant-pack buffer through the IR tensor's dim/stride.

        cuDNN's execute contract treats variant-pack entries as raw storage laid
        out per the IR tensor descriptor — callers may hand in a torch tensor
        whose *logical* shape is anything with the right bytes (e.g. a
        (B,S,H,D)-contiguous allocation for a (B,H,S,D) BSHD-strided IR tensor,
        as test_mhas_v2's fp8 harness does). The DSL executor consumes torch
        views, so rebuild the IR-shaped view here instead of trusting the
        caller's metadata. No-op when the caller already passed an IR-shaped
        view. THD buffers are packed (fewer elements than dim x stride
        implies), so the caller's view is kept as-is there.
        """
        dim, stride = tuple(ir_t.get_dim()), tuple(ir_t.get_stride())
        if tuple(buf.shape) == dim and tuple(buf.stride()) == stride:
            return buf
        return buf.as_strided(dim, stride)

    def _execute(variant_pack, workspace=None, stream=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        q_buf = resolved[id(binding.q)]
        k_buf = resolved[id(binding.k)]
        v_buf = resolved[id(binding.v)]
        o_buf = resolved[id(binding.o)]
        if not facts.thd:
            q_buf = _ir_view(q_buf, binding.q)
            k_buf = _ir_view(k_buf, binding.k)
            v_buf = _ir_view(v_buf, binding.v)
            o_buf = _ir_view(o_buf, binding.o)
        # Scratch comes from the CALLER's workspace (never allocated here): the
        # CompiledPlan sized/validated it against workspace_bytes; the carver
        # re-validates so a direct call cannot silently corrupt memory.
        import torch  # execute path: a real tensor is about to be carved

        carver = WorkspaceCarver(workspace, total_workspace_bytes, spec.name) if total_workspace_bytes else None
        # Stats-less graphs bind lse_buf=None: every adapter here is
        # lse_optional (the kernel compiles the LSE store out) — no dummy.
        lse_buf = resolved.get(id(binding.stats)) if binding.stats is not None else None
        # Shared presence-checked resolution (graph_analyzer.resolve_feature_operands);
        # bias flows only to adapters declaring the keyword (the SM80 row);
        # every other feature operand is gated off by mismatch for all rows.
        feature_ops = ga.resolve_feature_operands(facts, resolved)
        sinks_buf = feature_ops.sinks
        seq_kv_buf = feature_ops.seq_kv_lens
        if synth_kv_padding and seq_kv_buf is None:
            # Full-length per-batch KV lengths: mathematically a no-op mask that
            # makes the kernel's padded path cover the ragged KV tail.
            seq_kv_buf = carver.take(facts.b, torch.int32).fill_(facts.s_kv)
        seq_q_buf = feature_ops.seq_len_q
        sf_q_buf = resolved.get(id(binding.sf_q)) if binding.sf_q is not None else None
        sf_k_buf = resolved.get(id(binding.sf_k)) if binding.sf_k is not None else None
        sf_v_buf = resolved.get(id(binding.sf_v)) if binding.sf_v is not None else None
        amax_o_buf = resolved.get(id(binding.amax_o)) if binding.amax_o is not None else None
        dq_buf = resolved.get(id(binding.descale_q)) if binding.descale_q is not None else None
        dk_buf = resolved.get(id(binding.descale_k)) if binding.descale_k is not None else None
        dv_buf = resolved.get(id(binding.descale_v)) if binding.descale_v is not None else None
        so_buf = resolved.get(id(binding.scale_o)) if binding.scale_o is not None else None
        # Rows whose kernel lacks the dense padded-Q trim (dense_seq_q_trim
        # False) drop the per-batch Q lengths, which is harmless only while
        # every seq_len_q equals S_q -- a shorter one writes O and a finite
        # LSE past the valid length. Checked here because the lengths are
        # device values (its own Rule 3 known-violation entry; the descale
        # scalars themselves no longer read back at all).
        # THD is exempt: ragged lengths ARE shorter than S_q by construction,
        # and the packed layout gives each sequence its own extent, so
        # nothing is written past a valid length.
        if not spec.capabilities.dense_seq_q_trim and not facts.thd and seq_q_buf is not None:
            min_seq_q = int(seq_q_buf.min().item())
            if min_seq_q < int(facts.s_q):
                raise NotImplementedError(
                    f"{spec.name}: per-batch seq_len_q shorter than S_q={facts.s_q} is not plumbed "
                    f"(no dense padded-Q trim in this kernel); got min {min_seq_q}"
                )
        execute_kwargs = dict(
            q_tensor=q_buf,
            k_tensor=k_buf,
            v_tensor=v_buf,
            o_tensor=o_buf,
            lse_tensor=lse_buf,
            scale_softmax=facts.scale,
            sinks=sinks_buf,
            seq_kv_lens=seq_kv_buf,
            seq_q_lens=seq_q_buf if (seq_q_lens_present or facts.thd) else None,
            # Stream from the execute-time handle (raw CUstream int, the
            # ExecutionContext's stream); None keeps the default stream.
            current_stream=_cuda_driver().CUstream(stream) if stream is not None else None,
        )
        if facts.is_mxfp8 or facts.is_fp8:
            execute_kwargs.update(
                sf_q=sf_q_buf,
                sf_k=sf_k_buf,
                sf_v=sf_v_buf,
                amax_o=amax_o_buf,
                descale_q=dq_buf,
                descale_k=dk_buf,
                descale_v=dv_buf,
                scale_o=so_buf,
            )
        if _extra_exec_keys:
            # SM80 feature operand (mismatch admitted it for this row).
            if feature_ops.bias is not None and "bias_tensor" in _extra_exec_keys:
                execute_kwargs["bias_tensor"] = feature_ops.bias
        if api_scratch_bytes:
            execute_kwargs["workspace"] = carver.remaining()
        api.execute(**execute_kwargs)
        return None

    # Executor contract (engine._FrostSdpaFwdPlan): a non-zero workspace_bytes
    # means the plan calls _execute(variant_pack, workspace) with the caller's
    # buffer; 0 means _execute(variant_pack) and the buffer is never touched.
    # ``binding`` lets the plan key this executor's operands out of the graph's
    # variant pack (the pack covers every IO tensor of the graph).
    _execute.workspace_bytes = total_workspace_bytes
    _execute.binding = binding
    return _execute


def engine_name(
    phase: str = "prefill",
    arch: str = "sm100",
    mxfp8: bool = False,
    fp8: bool = False,
) -> str:
    """The registered engine name for a coverage cell (test/user convenience).

    One engine per arch x dtype family — head dims are a lowering concern
    (kernel-flavor pick), not part of the engine identity."""

    suffix = "_mxfp8" if mxfp8 else "_fp8" if fp8 else ""
    return f"sdpa_fwd_{phase}_{arch}" + suffix


# ORDER MATTERS: this is the PREFERENCE order — ``engine.FrostSdpaFwdEngines()``
# wraps the specs in it, so the plans they propose reach graph.plans in this
# order and the build walk tries them top-down. One engine per arch x dtype
# family: kernel-FLAVOR choice (which head-dim tile) happens inside the
# lowering (api_dsl._pick_flavor, smallest covering flavor — the tightest
# tile, least padded work), so at most one row of a family is eligible per
# device and the order only breaks ties across families.
# Engine IDS do NOT follow this order — they are pinned per name in
# engines/manifest.py and never move.
def _sm120_fp8_spec() -> EngineSpec:
    """SM120 per-tensor FP8 engine (E4M3/E5M2 in + scalar descales, FP16/BF16/FP8 out).

    P quantization follows the backend's FORT ordering: the softmax denominator
    reads the unscaled result, then Scale_S multiplies P, then the fp8 cast,
    with Descale_S folded into o_scale_fused. (cuDNN's Scale_S/Descale_S scale
    the softmax OUTPUT, not the scores.) The SM100 FP8 row deliberately does
    NOT do this: its lazy-rescale skip leaves P bounded by 2^RESCALE_THRESHOLD
    rather than 1, so e4m3's range above 1.0 is already spent and a caller
    Scale_S would saturate it. It honours a reciprocal pair analytically
    instead and declines anything else.

    Same mma.sync architecture as the f16 SM120 cell with the MMA lowered to
    m16n8k32 e4m3; ``descale_q*descale_k`` folds into the softmax scale and
    ``descale_v*scale_o`` into an epilogue scalar, so beyond those the kernel
    adds only the Amax_O atomic (no Amax_S — the shared mismatch rule declines
    graphs that declare one). E4M3/E5M2 QKV supported; O may
    be FP16, BF16, or FP8 (either flavor — a direct quantizing store applies
    ``scale_o`` before the cast, and Amax_O stays the pre-cast fp32 amax).
    Head TILES any multiple of 32 up to 256 with the QK^T and P@V sides
    independent; actual head dims may be any multiple of 16 up to the tile
    (TMA 16-byte global-stride rule at 1 byte/elem) via TMA zero-padding.
    Attention sinks fold into the softmax denominator: the sink is a virtual
    column with no V row — it rescales O and enters the LSE. THD (ragged) is
    served with token- or head-major Stats.
    """
    from cudnn.sdpa.fwd.config_sm120 import SUPPORTED_HEAD_TILES_FP8

    return EngineSpec(
        name="sdpa_fwd_prefill_sm120_fp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            phase="prefill",
            d_shapes=frozenset((tq, tv) for tq in SUPPORTED_HEAD_TILES_FP8 for tv in SUPPORTED_HEAD_TILES_FP8),
            d_pad_multiple=16,  # TMA 16-byte global-stride rule at 1 byte/elem
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            is_fp8=True,
            causal=True,
            bottom_right=True,
            swa=True,
            right_band_widening=True,
            padded=True,
            sink=True,
            stats=True,
            lse_optional=True,
            thd=True,
            padded_stats=True,
            dense_seq_q_trim=True,
            skv_tile=0,
            # dense_flex: any dense layout with the head dim
            # innermost-contiguous is normalized to the kernel's compact BSHD
            # by the shared adapter (one gather copy in, one scatter copy
            # back for O; zero-copy when already BSHD-physical).
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            tile_ms=frozenset({64, 128}),
            tile_ns=frozenset({64, 128}),
            cgas=frozenset({1}),
            pack_gqas=frozenset({False, True}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM120),
    )


ENGINE_SPECS = (
    _sm100_spec(),
    _sm100_mxfp8_spec(),
    _sm100_fp8_spec(),
    _sm100_fp8_spec(arch="sm107"),
    _sm120_spec(),
    _sm120_fp8_spec(),
    _sm80_spec(),
)

__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "SdpaFwdKnobs", "analyze_for", "build", "engine_name", "mismatch"]
