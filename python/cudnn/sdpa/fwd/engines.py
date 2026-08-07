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
    # OOB-clipped. False = only the native dims above are eligible (FP8/MXFP8;
    # SM120, whose lowering has no zero-padding path wired yet).
    d_envelope: bool = False
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})  # cudnn.data_type, see graph_analyzer
    is_mxfp8: bool = False  # block-scale MXFP8 engine (FP8 in + per-32-block E8M0 SF)
    is_fp8: bool = False  # per-tensor FP8 engine (FP8 in + scalar descales)

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
    bottom_right_with_swa: bool = False  # kernel gap: BR diagonal excludes SWA
    # Kernel gap (pre-existing, sibling of thd_bottom_right): the BR diagonal is
    # computed as seq_len_kv[b] - GLOBAL S_q, but cuDNN semantics for a dense
    # padded graph carrying per-batch seq_len_q anchor it at
    # (seq_len_q[b], seq_len_kv[b]) — any batch with seq_len_q[b] < S_q gets a
    # wrongly shifted diagonal (extra zeroed rows at the top). KV-only padding
    # (no seq_len_q tensor) is unaffected: there the actual Q length IS S_q.
    bottom_right_padded_seq_q: bool = False
    swa: bool = False
    padded: bool = False
    sink: bool = False
    stats: bool = False
    # The adapter accepts lse_tensor=None (its kernel None-specializes the LSE
    # store), so a stats-less graph needs no dummy-LSE workspace chunk. Rows
    # that keep False (the SM100 flavors) always write an LSE and get a carved
    # dummy from lower_dsl_prefill when the graph has no Stats output.
    lse_optional: bool = False
    thd: bool = False
    cu_seq_len: bool = False  # cu_seq_len_q / cu_seq_len_kv prefix sums (no row serves these yet)
    # True = the kernel anchors the THD bottom-right diagonal at each sequence's
    # own (seq_len_q[b], seq_len_kv[b]). Rows that keep False (the SM100 flavors)
    # compute it from the GLOBAL S_q — the THD variant of the
    # bottom_right_padded_seq_q gap above.
    thd_bottom_right: bool = False
    thd_stats: bool = False  # packed LSE output plumbing is a follow-up
    # Dense padded + stats needs the per-batch seq_len_q LSE trim (padded
    # q-rows write LSE=-inf / O=0, cuDNN >= 9.14). Plumbed for the half
    # kernels via SEQ_Q_LENS_PRESENT; the FP8/MXFP8 kernels lack the epilogue
    # trim, so their specs keep this False.
    padded_stats: bool = False

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
        if facts.d_qk % 8 != 0 or facts.d_v % 8 != 0:
            return (
                f"envelope zero-padding requires D_QK/D_V multiples of 8 (TMA 16-byte "
                f"global-stride constraint at 2 bytes/elem); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            )
    elif facts.d_qk not in capabilities.d_qk or facts.d_v not in capabilities.d_v:
        return f"serves D_QK in {sorted(capabilities.d_qk)}/D_V in {sorted(capabilities.d_v)}; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if (facts.is_mxfp8, facts.is_fp8) != (capabilities.is_mxfp8, capabilities.is_fp8):
        quant = "block-scale MXFP8 (sdpa_mxfp8)" if capabilities.is_mxfp8 else "per-tensor FP8 (sdpa_fp8)" if capabilities.is_fp8 else "half (sdpa)"
        return f"this engine serves only {quant} graphs"
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
        (facts.has_cu_seq_len, capabilities.cu_seq_len, "cu_seq_len_q / cu_seq_len_kv"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

    if facts.bottom_right:
        if not facts.causal:
            return "bottom-right alignment requires a causal upper bound"
        if not capabilities.bottom_right:
            return "graph uses bottom-right causal, which this kernel does not support"
        if facts.window_left is not None and not capabilities.bottom_right_with_swa:
            return "bottom-right causal combined with a sliding window is not supported"
        if facts.thd and not capabilities.thd_bottom_right:
            return "THD with bottom-right causal is not supported (per-sequence diagonal gap)"
        if facts.padded and not facts.thd and facts.seq_q_t is not None and not capabilities.bottom_right_padded_seq_q:
            return (
                "bottom-right causal with a dense padding mask carrying per-batch seq_len_q is not "
                "supported (kernel anchors the BR diagonal at the global S_q, not seq_len_q[b])"
            )
    if facts.thd and facts.wants_stats and not capabilities.thd_stats:
        return "THD with generate_stats is not supported yet"
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
        causal_covers_tail = facts.causal and (facts.bottom_right or facts.s_q <= facts.s_kv)
        if not (facts.padded or causal_covers_tail):
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
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            thd=True,
            padded_stats=True,
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


def _sm100_mxfp8_spec(d: int) -> EngineSpec:
    """d128 block-scale MXFP8 engine (E4M3/E5M2 in + per-32-block E8M0 SF, half out).

    THD/varlen is deferred (dense execute only for v1), so thd=False here even
    though the kernel itself supports it.
    """

    return EngineSpec(
        name=f"sdpa_fwd_prefill_sm100_d{d}_mxfp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            d_qk=frozenset({d}),
            d_v=frozenset({d}),
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            is_mxfp8=True,
            causal=True,
            bottom_right=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            sched_policies=frozenset({SCHED_NATURAL}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_fp8_spec(d: int) -> EngineSpec:
    """d128 per-tensor FP8 engine (E4M3/E5M2 in + scalar descales, half/FP8 out).

    Padding mask (per-batch ``seq_len_kv`` → KV-side masking) is supported: KV-only
    padding leaves every query row real, so each row's total_sum > 0 and the
    in-kernel amax_s (= max over rows of 1/total_sum) stays well-defined — no
    fully-masked row can poison the global amax.  THD/varlen is still deferred
    (dense execute only for v1), so thd=False.
    """

    return EngineSpec(
        name=f"sdpa_fwd_prefill_sm100_d{d}_fp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            d_qk=frozenset({d}),
            d_v=frozenset({d}),
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            is_fp8=True,
            causal=True,
            bottom_right=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
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


def _sm120_spec() -> EngineSpec:
    return EngineSpec(
        name="sdpa_fwd_prefill_sm120",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            phase="prefill",
            d_qk=frozenset(range(16, 257, 16)),
            d_v=frozenset(range(16, 257, 16)),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            bottom_right_with_swa=True,
            bottom_right_padded_seq_q=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            lse_optional=True,
            padded_stats=True,
            thd=True,
            thd_bottom_right=True,
            thd_stats=True,
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


def probe(spec: EngineSpec, graph, knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    _, reason = analyze_for(spec, graph, knobs)
    if reason is not None:
        _LOG.debug("cudnn.sdpa: %s ineligible: %s", spec.name, reason)
        return False
    return True


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
    causal_covers_tail = facts.causal and (facts.bottom_right or facts.s_q <= facts.s_kv)
    synth_kv_padding = spec.capabilities.skv_tail_via_padding and not facts.padded and not facts.thd and facts.s_kv % skv_tile != 0 and not causal_covers_tail

    seq_q_t = facts.seq_q_t if facts.padded else None
    seq_kv_t = facts.seq_kv_t if facts.padded else None
    # Mirrors the seq_q_lens_present constructor argument below. Execute
    # forwards seq_q only when the compiled specialization consumes it (or THD,
    # which sources cu_seqlens from it) — the adapter rejects mismatches, so a
    # buffer the FP8/MXFP8 kernels can't honor (dense padded-Q trim is not
    # plumbed there — known gap) is dropped here rather than erroring at
    # execute.
    seq_q_lens_present = facts.padded and not facts.thd and facts.seq_q_t is not None and not (facts.is_mxfp8 or facts.is_fp8)
    api = _adapter(api_type)(
        sample_q=ga.tensor_desc_from_ir(facts.q_t, name="q"),
        sample_k=ga.tensor_desc_from_ir(facts.k_t, name="k"),
        sample_v=ga.tensor_desc_from_ir(facts.v_t, name="v"),
        sample_o=ga.tensor_desc_from_ir(facts.o_t, name="o"),
        sample_lse=ga.tensor_desc_from_ir(facts.stats_t, "lse") if facts.stats_t is not None else None,
        is_causal=facts.causal,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        scale_softmax=facts.scale,
        seq_kv_lens_present=facts.padded or synth_kv_padding,
        # Dense padded-Q trim (q rows >= seq_len_q[b] -> O := 0, LSE := -inf):
        # enabled whenever a dense padded graph carries per-batch Q lengths.
        # THD carries Q lengths via cu_seqlens; the FP8/MXFP8 kernels are not
        # plumbed (their specs also keep padded_stats=False).
        seq_q_lens_present=seq_q_lens_present,
        has_sink=facts.has_sink,
        thd=facts.thd,
        dtype_o=facts.dtype_o if (facts.is_mxfp8 or facts.is_fp8) else None,
        pertensor_fp8=facts.is_fp8,
        sched_policy=knobs.sched_policy if knobs is not None else None,
        tile_m=knobs.tile_m if knobs is not None else None,
        tile_n=knobs.tile_n if knobs is not None else None,
        cga=knobs.cga if knobs is not None else None,
    )
    api.check_support()  # raises ValueError / NotImplementedError if unsupported
    api.compile()

    # Workspace requirement for the compiled geometry: every per-execute scratch
    # buffer is carved from the CALLER's workspace, so its size is fixed here at
    # build time and recorded on the executor as ``workspace_bytes`` — that
    # number is what the plan's CompiledPlan.get_workspace_size() reports.
    #   - dummy LSE (dense, stats absent, non-lse_optional adapters): the
    #     SM100 kernels always write an LSE; without a Stats output it lands
    #     in b*h_q*s_q fp32 scratch. lse_optional adapters (SM120) compile the
    #     LSE store out instead and bind no buffer. (THD needs no engine-level
    #     LSE chunk — the packed THD LSE is part of the api-level scratch
    #     below.)
    #   - synthesized seq_len_kv (skv_tail_via_padding rows): b int32.
    #   - api-level scratch (api.scratch_workspace_bytes()): the dense padded
    #     [seq_kv|seq_q] combine and the THD metadata/LSE buffers.
    dummy_lse_bytes = (
        0
        if (not spec.capabilities.stats or spec.capabilities.lse_optional or facts.stats_t is not None or facts.thd)
        else ws_align(facts.b * facts.h_q * facts.s_q * 4)
    )
    synth_kv_bytes = ws_align(facts.b * 4) if synth_kv_padding else 0
    api_scratch_bytes = api.scratch_workspace_bytes()
    total_workspace_bytes = dummy_lse_bytes + synth_kv_bytes + api_scratch_bytes

    binding = ga.SdpaBinding(
        q=facts.q_t,
        k=facts.k_t,
        v=facts.v_t,
        o=facts.o_t,
        stats=facts.stats_t,
        sink_token=facts.sink_t,
        seq_len_kv=seq_kv_t,
        seq_len_q=seq_q_t,
        sf_q=facts.sf_q_t,
        sf_k=facts.sf_k_t,
        sf_v=facts.sf_v_t,
        amax_o=facts.amax_o_t,
        descale_q=facts.descale_q_t,
        descale_k=facts.descale_k_t,
        descale_v=facts.descale_v_t,
        scale_o=facts.scale_o_t,
        amax_s=facts.amax_s_t,
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
        lse_buf = resolved.get(id(binding.stats)) if binding.stats is not None else None
        if lse_buf is None and spec.capabilities.stats and not spec.capabilities.lse_optional and not facts.thd:
            # Dummy LSE for stats-less dense graphs — carved, not allocated
            # (uninitialized is fine: the kernel writes every row). lse_optional
            # adapters take lse_tensor=None instead.
            lse_buf = carver.take(facts.b * facts.h_q * facts.s_q, torch.float32)
        sinks_buf = resolved.get(id(binding.sink_token)) if binding.sink_token is not None else None
        seq_kv_buf = resolved.get(id(binding.seq_len_kv)) if binding.seq_len_kv is not None else None
        if synth_kv_padding and seq_kv_buf is None:
            # Full-length per-batch KV lengths: mathematically a no-op mask that
            # makes the kernel's padded path cover the ragged KV tail.
            seq_kv_buf = carver.take(facts.b, torch.int32).fill_(facts.s_kv)
        seq_q_buf = resolved.get(id(binding.seq_len_q)) if binding.seq_len_q is not None else None
        sf_q_buf = resolved.get(id(binding.sf_q)) if binding.sf_q is not None else None
        sf_k_buf = resolved.get(id(binding.sf_k)) if binding.sf_k is not None else None
        sf_v_buf = resolved.get(id(binding.sf_v)) if binding.sf_v is not None else None
        amax_o_buf = resolved.get(id(binding.amax_o)) if binding.amax_o is not None else None
        dq_buf = resolved.get(id(binding.descale_q)) if binding.descale_q is not None else None
        dk_buf = resolved.get(id(binding.descale_k)) if binding.descale_k is not None else None
        dv_buf = resolved.get(id(binding.descale_v)) if binding.descale_v is not None else None
        so_buf = resolved.get(id(binding.scale_o)) if binding.scale_o is not None else None
        amax_s_buf = resolved.get(id(binding.amax_s)) if binding.amax_s is not None else None
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
                amax_s=amax_s_buf,
            )
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
ENGINE_SPECS = (
    _sm100_spec(128),
    _sm100_spec(192, d_v=128),
    _sm100_spec(256),
    _sm100_spec(512),
    _sm100_mxfp8_spec(128),
    _sm100_fp8_spec(128),
    _sm120_spec(),
)

__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "SdpaFwdKnobs", "analyze_for", "build", "engine_name", "mismatch", "probe"]
