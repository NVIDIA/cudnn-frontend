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

from cudnn.frost.tile_dsl.constants import SCHED_NATURAL
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old
from cudnn.sdpa import graph_analyzer as ga

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
    d_qk: frozenset[int]
    d_v: frozenset[int]
    # Head-dim ENVELOPE: when True the lowering ALSO serves any graph with
    # d_qk/d_v <= the native caps (both multiples of 8, the TMA 16-byte
    # global-stride rule at 2 bytes/elem) via TMA zero-padding — the kernel's
    # descriptors carry the ACTUAL extents, so padded contraction columns load
    # as exact zeros (S/softmax unchanged) and O stores past d_v are
    # OOB-clipped. False = only the native dims above are eligible (MXFP8,
    # whose SF plumbing is not audited for zero-padding).
    d_envelope: bool = False
    # Alignment rule for envelope rows: graph head dims must be multiples of
    # this. 8 = the TMA 16-byte global-stride rule at 2 bytes/elem (SM100 DSL
    # rows). 1 = no constraint (the SM80 lowering pads host-side).
    d_pad_multiple: int = 8
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
        ):
            if value is not None and value not in domain:
                return f"requested {label}={value} is outside this engine's domain {sorted(domain)}"
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
    if capabilities.d_envelope:
        # Envelope row: native caps are upper bounds (TMA zero-padding semantics
        # — see Capabilities.d_envelope). Alignment: TMA global strides must be
        # 16-byte multiples; compact BSHD H-stride is D * 2 bytes -> D % 8.
        cap_qk, cap_v = max(capabilities.d_qk), max(capabilities.d_v)
        if facts.d_qk > cap_qk or facts.d_v > cap_v:
            return f"serves D_QK<={cap_qk}/D_V<={cap_v} (envelope); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
        m = capabilities.d_pad_multiple
        if m > 1 and (facts.d_qk % m != 0 or facts.d_v % m != 0):
            return (
                f"envelope zero-padding requires D_QK/D_V multiples of {m} (TMA 16-byte "
                f"global-stride constraint); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            )
    elif facts.d_qk not in capabilities.d_qk or facts.d_v not in capabilities.d_v:
        return f"serves D_QK in {sorted(capabilities.d_qk)}/D_V in {sorted(capabilities.d_v)}; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
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


def _sm100_spec(d: int, d_v: Optional[int] = None) -> EngineSpec:
    d_v = d if d_v is None else d_v
    suffix = f"d{d}" if d_v == d else f"d{d}_d{d_v}"
    return EngineSpec(
        name=f"sdpa_fwd_prefill_sm100_{suffix}",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            d_qk=frozenset({d}),
            d_v=frozenset({d_v}),
            d_envelope=True,  # native tile box d; smaller dims via TMA zero-padding
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
            sched_policies=frozenset({SCHED_NATURAL}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_mxfp8_spec(d: int, d_v: Optional[int] = None) -> EngineSpec:
    """Block-scale MXFP8 engine (E4M3/E5M2 + per-32-block E8M0 SF).

    THD/varlen (d128/d128 only — the d192/d128 kernel is dense-only) rides the
    shared packed lowering (write_thd_meta envelope design, issue #552; packed
    Q/K/V/O contract only). The SF tensors travel PACKED
    per-sequence-TILE-padded ([1, H, Σ_b ceil(S_b/128), SF_SMEM] tile sequences
    in cu_seqlens order — see api_dsl._reshape_sf_packed); the graph's declared
    SF dims stay the dense capacity, like the ragged Q/K/V storage.
    """

    d_v = d if d_v is None else d_v
    suffix = f"d{d}" if d_v == d else f"d{d}_d{d_v}"
    thd = (d, d_v) == (128, 128)
    return EngineSpec(
        name=f"sdpa_fwd_prefill_sm100_{suffix}_mxfp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            d_qk=frozenset({d}),
            d_v=frozenset({d_v}),
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
            thd=thd,
            cu_seq_len=thd,
            sched_policies=frozenset({SCHED_NATURAL}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_fp8_spec(
    d: int,
    d_v: Optional[int] = None,
    *,
    dtypes: Optional[frozenset] = None,
    sink_dtypes: Optional[frozenset] = None,
) -> EngineSpec:
    """Exact-shape per-tensor FP8 engine with scalar descales.

    Padding mask (per-batch ``seq_len_kv`` → KV-side masking) is supported: KV-only
    padding leaves every query row real, so each row's total_sum > 0 and the
    per-row softmax normalization stays well-defined — no
    fully-masked row can poison the global amax.  THD/varlen (d128/d128 only —
    the d192/d128 kernel is dense-only) rides the shared packed lowering
    (write_thd_meta envelope design, issue #552; packed Q/K/V/O contract
    only) — on cc10.7 (Rubin) through the SM107 sibling kernel, which carries
    the same THD leg.
    """

    d_v = d if d_v is None else d_v
    suffix = f"d{d}" if d_v == d else f"d{d}_d{d_v}"
    thd = (d, d_v) == (128, 128)
    if dtypes is None:
        dtypes = frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2})
    return EngineSpec(
        name=f"sdpa_fwd_prefill_sm100_{suffix}_fp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            d_qk=frozenset({d}),
            d_v=frozenset({d_v}),
            dtypes=dtypes,
            out_dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            is_fp8=True,
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            sink_dtypes=sink_dtypes,
            stats=True,
            lse_optional=True,
            thd=thd,
            cu_seq_len=thd,
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
            sched_policies=frozenset({SCHED_NATURAL}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
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
            d_qk=frozenset({256}),
            d_v=frozenset({256}),
            d_envelope=True,  # flavor envelopes; host-side zero-padding
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
            sched_policies=frozenset(),
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
            d_qk=frozenset(SUPPORTED_HEAD_TILES),
            d_v=frozenset(SUPPORTED_HEAD_TILES),
            d_envelope=True,
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
            sched_policies=frozenset({SCHED_NATURAL}),
            tile_ms=frozenset({64, 128}),
            tile_ns=frozenset({64, 128}),
            cgas=frozenset({1}),
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
    d: Optional[int] = None,
    phase: str = "prefill",
    arch: str = "sm100",
    mxfp8: bool = False,
    fp8: bool = False,
    d_v: Optional[int] = None,
) -> str:
    """The registered engine name for a coverage cell (test/user convenience)."""

    name = f"sdpa_fwd_{phase}_{arch}"
    if d is not None:
        if d_v is None or d_v == d:
            name += f"_d{d}"
        else:
            name += f"_d{d}_d{d_v}"
    suffix = "_mxfp8" if mxfp8 else "_fp8" if fp8 else ""
    return name + suffix


# ORDER MATTERS: this is the PREFERENCE order — ``engine.FrostSdpaFwdEngines()``
# wraps the specs in it, so the plans they propose reach graph.plans in this
# order and the build walk tries them top-down. With head-dim envelopes every
# small-d graph is eligible for ALL covering f16 flavors, so the f16 rows are
# listed smallest-first to make first-eligible == smallest-covering flavor
# (the tightest tile, least padded work). We deliberately keep mismatch() a
# pure upper-bound check instead of rejecting non-tightest flavors: an
# explicit select_plan() of a larger covering flavor stays legal (useful for
# flavor A/B testing), only the ranking prefers the tightest.
# Engine IDS do NOT follow this order — they are pinned per name in
# engine._ID_OFFSETS and never move.
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
            d_qk=frozenset(SUPPORTED_HEAD_TILES_FP8),
            d_v=frozenset(SUPPORTED_HEAD_TILES_FP8),
            d_envelope=True,  # native tile box d; smaller dims via TMA zero-padding
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
            sched_policies=frozenset({SCHED_NATURAL}),
            tile_ms=frozenset({64, 128}),
            tile_ns=frozenset({64, 128}),
            cgas=frozenset({1}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM120),
    )


ENGINE_SPECS = (
    _sm100_spec(128),
    _sm100_spec(192, d_v=128),
    _sm100_spec(256),
    _sm100_spec(512),
    _sm100_mxfp8_spec(128),
    _sm100_mxfp8_spec(192, d_v=128),
    _sm100_fp8_spec(128),
    _sm100_fp8_spec(
        192,
        d_v=128,
        dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
    ),
    _sm120_spec(),
    _sm120_fp8_spec(),
    _sm80_spec(),
)

__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "SdpaFwdKnobs", "analyze_for", "build", "engine_name", "mismatch"]
