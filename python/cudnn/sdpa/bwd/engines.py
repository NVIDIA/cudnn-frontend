# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SDPA-backward engine registry: capability declarations + spec table.

One registered engine per architecture, named ``sdpa_bwd_sm<arch>`` (dtype is
NOT part of the identity: a cell's engine serves every dtype its kernel
handles — fp16 and bf16 today — via ``Capabilities.dtypes``).

The backward opset keeps its own :class:`Capabilities` record rather than
reusing the forward one: the feature model differs (gradient side outputs,
determinism, no phase axis) and the shared analyzer facts already carry
everything both need. The shared analyzer
(``cudnn.sdpa.graph_analyzer.analyze``) parses the graph once into
:class:`SdpaGraphFacts`; each engine's probe is a cheap field-by-field
candidate match against its row below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import cudnn
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old
from cudnn.sdpa import graph_analyzer as ga

# Lowering dependencies, resolved at build time — see the note in fwd/engines.py:
# importing them here would drag the CuTe DSL into every support check.
_SM120 = "SdpaBwdDslSm120"
_SM80 = "SdpaBwdDslSm80"
_SM100 = "SdpaBwdDslSm100"


def _adapter(name: str):
    from cudnn.sdpa.bwd import api_dsl

    return getattr(api_dsl, name)


def _cuda_driver():
    from cuda.bindings import driver

    return driver


from cudnn.sdpa.bwd.config_sm120 import (
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_DIMS as _SM120_HEAD_DIMS,
)

_LOG = logging.getLogger(__name__)

# Arch range this spec serves, inclusive, major*10 + minor as in
# engines/manifest.py. A range, not the exact device families that exist today:
# an sm120 kernel runs on the sm120 line, and enumerating members silently
# declines whatever ships next.
_BLACKWELL_GEFORCE = (120, 129)


@dataclass(frozen=True)
class SdpaBwdKnobs:
    """Per-plan tuning request for the SDPA-backward engines.

    This is the operation's knob *vocabulary* — typed fields, no global enum.
    ``None`` means "no preference". Travels as ``PlanConfig.knobs``; each
    engine's :class:`Capabilities` row advertises the domain it honors, and the
    probe rejects the engine for any request outside that domain (a knob is
    honored or the engine is ineligible — never silently degraded).
    """

    tile_m: Optional[int] = None  # Q sequence tile width (q_tile)
    tile_n: Optional[int] = None  # KV sequence tile width (kv_tile)


@dataclass(frozen=True)
class Capabilities:
    """What one backward ENGINE can serve — the envelope of graphs its
    lowering can honor. Compared field-by-field against SdpaGraphFacts in the
    probe."""

    # Arch RANGE, inclusive, encoded major*10 + minor as in engines/manifest.py.
    # A range and not a set of exact device families: an sm100 kernel runs on
    # everything in the sm100 line, so enumerating the parts that exist today
    # silently declines the ones that ship tomorrow. Rubin (sm107) and Thor
    # (sm110) are meant to reuse these kernels and an exact set excluded both.
    sm_lo: int
    sm_hi: int
    d: frozenset[int]  # supported head dims (d_qk == d_v unless dqk_ge_dv)
    # When True, ``d`` is an ENVELOPE upper bound: the lowering also serves any
    # graph with head dims <= max(d), computing on the smallest covering
    # kernel size (sm120 in place via TMA zero-fill; sm80 via host-side
    # zero-padding). False = exact-set membership.
    d_envelope: bool = False
    # Alignment rule for envelope rows: head dims must be multiples of this
    # (sm120's TMA 16-byte global-stride rule at 2 B/elem -> 8; sm80's
    # host-side padding has no such constraint -> 1).
    d_pad_multiple: int = 1
    # EXCLUSIVE lower bound on an envelope row: head dims must be > this. An
    # envelope with no floor silently claims every small head dim too, and pads
    # it onto the big kernel -- for the sm100 d512 backward that is the whole
    # 512-wide MMA for a d=128 graph, when a d256 flavor exists. 0 = no floor
    # (the sm120/sm80 rows are continuum kernels and want none).
    d_envelope_floor: int = 0
    # When True the lowering serves rectangular head dims with D_QK >= D_V
    # (e.g. 192/128); False requires D_QK == D_V.
    dqk_ge_dv: bool = False
    # True when stats is not contiguous (B, H_q, S_q, 1) in memory.
    strided_stats: bool = False
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})  # cudnn.data_type, see graph_analyzer

    # optional features a backward graph may request
    gqa: bool = False  # h_q != h_kv
    causal: bool = False
    bottom_right: bool = False
    deterministic: bool = False  # use_deterministic_algorithm=True
    dbias: bool = False  # dBias output
    dsink: bool = False  # dSink_token output
    bias: bool = False
    dropout: bool = False
    score_mod: bool = False
    paged_kv: bool = False
    alibi: bool = False
    block_mask: bool = False
    rng_dump: bool = False
    score_max: bool = False
    score_sum_exp: bool = False
    dynamic_scale: bool = False
    unfuse_fma: bool = False
    seq_q_trim: bool = False
    right_band_widening: bool = False
    swa: bool = False
    padded: bool = False
    sink: bool = False
    thd: bool = False
    # cu_seq_len_q / cu_seq_len_kv prefix sums. No BACKWARD graph can carry
    # them -- the port is forward-only (SDPA_backward_attributes has no
    # CU_SEQ_LEN_* input), so no row claims this and none can be tested.
    cu_seq_len: bool = False
    # THD is a SEPARATE code path (packed views + a blocked S/dS workspace), so
    # a feature the dense path serves is not automatically served under THD.
    # These are the conjunction verdicts; mismatch() knows the shape of each.
    thd_causal: bool = False  # any causal-family bound (causal / SWA / band / bottom-right) under THD
    thd_gqa: bool = False  # h_q != h_kv under THD
    # True when THD REQUIRES sdpa(max_total_seq_len_q=..., max_total_seq_len_kv=...):
    # the row's workspace is sized from the packed token totals at BUILD time,
    # before any buffer exists.
    thd_declared_totals: bool = False
    # s_q == 1 (decode-shaped) graphs; rows whose kernels are prefill-only gate
    # them off.
    decode: bool = True
    # Dense layout envelope this engine accepts (mirrors fwd/engines.py):
    #   "bshd"       — Q/K/V/O must be BSHD-physical (stride order 3,1,2,0).
    #   "dense_flex" — any B/H/S stride permutation, padded (oversized)
    #                  strides included, as long as the head dim is
    #                  innermost-contiguous (stride 1) and the strides are
    #                  non-broadcast / non-overlapping (facts.dense_layout;
    #                  see graph_analyzer.dense_layout_ok).
    layouts: frozenset[str] = frozenset({"bshd"})

    # Tuning-knob domains this engine's lowering honors (see SdpaBwdKnobs).
    tile_ms: frozenset[int] = frozenset()
    tile_ns: frozenset[int] = frozenset()


def mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", requested: Any = None) -> Optional[str]:
    """First reason this engine is not a candidate for these facts and tuning
    knobs, or ``None`` when lowering should perform the final feasibility check.
    """
    if facts.invalid:
        return facts.invalid
    if not facts.is_backward:
        return "this engine serves sdpa_backward() graphs only"
    if requested is not None:
        if not isinstance(requested, SdpaBwdKnobs):
            return f"knob request is a {type(requested).__name__}, not SdpaBwdKnobs — wrong operation's vocabulary"
        for value, domain, label in (
            (requested.tile_m, capabilities.tile_ms, "tile_m"),
            (requested.tile_n, capabilities.tile_ns, "tile_n"),
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
    if facts.is_mxfp8 or facts.is_fp8:
        return "this engine serves only half (fp16/bf16) sdpa_backward graphs"
    if capabilities.dqk_ge_dv:
        if facts.d_qk < facts.d_v:
            return f"D_QK must be >= D_V; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    elif facts.d_qk != facts.d_v:
        return f"D_QK must equal D_V; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    if capabilities.d_envelope:
        if max(facts.d_qk, facts.d_v) > max(capabilities.d):
            return f"head dims (D_QK={facts.d_qk}, D_V={facts.d_v}) exceed the {max(capabilities.d)} envelope"
        m = capabilities.d_pad_multiple
        if m > 1 and (facts.d_qk % m != 0 or facts.d_v % m != 0):
            return f"the envelope serves head-dim multiples of {m} (TMA 16-byte global-stride rule); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
        fl = capabilities.d_envelope_floor
        if fl and min(facts.d_qk, facts.d_v) <= fl:
            return f"the envelope's floor is D > {fl} (a smaller flavor owns these); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    elif facts.d_qk not in capabilities.d:
        return f"serves D in {sorted(capabilities.d)}; graph has D={facts.d_qk}"
    elif facts.d_v not in capabilities.d:
        return f"serves D_V in {sorted(capabilities.d)}; graph has D_V={facts.d_v}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if not facts.uniform_dtype:
        return "K/V/O/dO/dQ/dK/dV dtypes must match Q"
    if facts.h_q != facts.h_kv and not capabilities.gqa:
        return f"GQA / MQA is not supported (H_q={facts.h_q}, H_kv={facts.h_kv})"
    if "dense_flex" in capabilities.layouts:
        if not facts.dense_layout:
            return (
                "Q/K/V/O/dO/dQ/dK/dV must have the head dim innermost-contiguous (stride 1) and "
                "non-broadcast, non-overlapping strides (any B/H/S order, padded strides allowed)"
            )
    elif not facts.bshd_layout:
        return "Q/K/V/O/dO/dQ/dK/dV must be BSHD-physical (stride order 3,1,2,0)"

    for fact, cap, label in (
        (facts.deterministic, capabilities.deterministic, "use_deterministic_algorithm (dQ accumulates through fp32 atomics)"),
        (facts.has_dbias, capabilities.dbias, "dBias output"),
        (facts.has_dsink, capabilities.dsink, "dSink_token output"),
        (facts.has_bias, capabilities.bias, "bias"),
        (facts.has_dropout, capabilities.dropout, "dropout"),
        (facts.has_score_mod, capabilities.score_mod, "score_mod"),
        (facts.has_paged_kv, capabilities.paged_kv, "paged attention"),
        (facts.has_alibi, capabilities.alibi, "ALiBi"),
        (facts.has_block_mask, capabilities.block_mask, "block_mask"),
        (facts.has_rng_dump, capabilities.rng_dump, "rng_dump"),
        (facts.has_score_max, capabilities.score_max, "score_max"),
        (facts.has_score_sum_exp, capabilities.score_sum_exp, "score_sum_exp"),
        (facts.dynamic_scale, capabilities.dynamic_scale, "tensor attn_scale"),
        (facts.has_unfuse_fma, capabilities.unfuse_fma, "unfuse_fma"),
        (facts.seq_q_trim, capabilities.seq_q_trim, "seq_len_q without padding mask"),
        (facts.right_band_widening, capabilities.right_band_widening, "causal right-band widening"),
        (facts.window_left is not None, capabilities.swa, "sliding window"),
        # A ragged graph always sets `padded` (its lengths ARE the mask), but a
        # row may serve the packed path without serving DENSE padding -- the
        # dense mask needs the per-batch length threaded into a kernel that
        # compiles a scalar. Mirrors fwd/engines.py.
        (facts.padded and not facts.thd, capabilities.padded, "padding mask"),
        (facts.has_sink, capabilities.sink, "sink token"),
        (facts.thd, capabilities.thd, "THD / ragged"),
        (facts.has_cu_seq_len, capabilities.cu_seq_len, "cu_seq_len_q / cu_seq_len_kv"),
        (facts.causal, capabilities.causal, "causal mask"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

    if facts.thd and capabilities.thd:
        # Conjunctions: the packed path is its own kernel specialization, and
        # each of these is asserted by a reject test.
        if (facts.causal or facts.right_band_widening or facts.window_left is not None) and not capabilities.thd_causal:
            return "causal-family masks under THD are not supported (the stage-3 K-trim is in absolute workspace rows)"
        # NOTE for a row that sets thd_causal: stage 3 must then be rendered with
        # causal_mode=CAUSAL_K_NONE and the workspace zero-filled, because that
        # trim cannot be expressed in the blocked layout's per-sequence rows.
        # See SdpaBwdDslSm100.compile.
        if facts.h_q != facts.h_kv and not capabilities.thd_gqa:
            return f"GQA / MQA under THD is not supported (H_q={facts.h_q}, H_kv={facts.h_kv})"
        if capabilities.thd_declared_totals and (facts.max_total_seq_len_q is None or facts.max_total_seq_len_kv is None):
            return (
                "THD requires sdpa_backward(max_total_seq_len_q=..., max_total_seq_len_kv=...): "
                "the packed workspace is sized from the declared token totals at build time"
            )
        # The packed path binds the caller's buffers straight to kernels whose
        # operands are compact BSHD; it has no staging copy to fix a layout up.
        if not facts.bshd_layout:
            return "Q/K/V/O/dO/dQ/dK/dV must be BSHD-physical under THD (the packed path has no staging copy)"

    if facts.has_dsink and not facts.has_sink:
        return "dSink_token output requires a sink_token input"
    if facts.s_q == 1 and not capabilities.decode:
        return "s_q == 1 (decode) is out of scope for the prefill kernels"
    if facts.bottom_right and not (facts.causal or facts.right_band_widening):
        return "bottom-right alignment requires a causal upper bound (plain or right-widened)"
    if facts.bottom_right and not capabilities.bottom_right:
        return "graph uses bottom-right causal, which this engine does not support"

    if facts.stats_t is not None:
        if facts.stats_t.get_data_type() != cudnn.data_type.FLOAT:
            return f"stats must be fp32; got {facts.stats_t.get_data_type()}"
        s_dim = tuple(facts.stats_t.get_dim())
        s_stride = tuple(facts.stats_t.get_stride())
        expect_dim = (facts.b, facts.h_q, facts.s_q, 1)
        if s_dim != expect_dim:
            return f"stats must be (B, H_q, S_q, 1) = {expect_dim}; got {s_dim}"
        if facts.thd and capabilities.thd:
            # Ragged Stats is PACKED, so its declared strides describe the
            # packing, not the (B, H, S_max, 1) envelope: token-major
            # (stride_h == 1, stride_s == H_q -- cuDNN's ragged-Stats recipe) or
            # head-major (stride_s == 1, stride_h == the head stride, which the
            # FROST forward emits rounded up to a 64-token capacity). Anything
            # else is a packing this row cannot read.
            if getattr(facts.stats_t, "ragged_offset", None) is None:
                # A DENSE (B, H_q, S_max, 1) stats tensor on a ragged graph is a
                # legal cuDNN graph, and its stride reads as head-major
                # (stride_s == 1, stride_h == S_max) while its storage is
                # per-batch rectangles. Declining it is what keeps the packing
                # inference below from mis-reading that layout.
                return "THD stats must be ragged (packed); a dense per-batch stats tensor is not read by the packed path"
            stride_h, stride_s = s_stride[1], s_stride[2]
            token_major = (stride_h, stride_s) == (1, facts.h_q)
            head_major = not token_major and stride_s == 1 and stride_h >= 1
            if not token_major and not head_major:
                return (
                    f"THD stats must be packed token-major (stride_h == 1, stride_s == {facts.h_q}) "
                    f"or head-major (stride_s == 1, stride_h == head stride); got stride {s_stride}"
                )
            # Head-major: the head stride is the caller's, and the kernel reads
            # [0, h, row] at it for every row below the packed token extent the
            # adapter binds (min(B * S_max, declared) -- see thd_total_q). A
            # shorter stride would put the later heads past the buffer.
            if head_major and facts.max_total_seq_len_q is not None and stride_h < min(facts.b * facts.s_q, facts.max_total_seq_len_q):
                return (
                    f"THD head-major stats head stride {stride_h} must cover the packed token total " f"{min(facts.b * facts.s_q, facts.max_total_seq_len_q)}"
                )
        elif capabilities.strided_stats:
            if any(st == 0 and d > 1 for d, st in zip(s_dim, s_stride)):
                return f"stats must not broadcast (stride 0 on a size > 1 dim); got stride {s_stride}"
        else:
            expect_stride = (facts.h_q * facts.s_q, facts.s_q, 1, 1)
            if s_stride != expect_stride:
                return f"stats must be contiguous {expect_stride}; got stride {s_stride}"

    if facts.has_dbias and not facts.has_bias:
        return "dBias output requires a bias input"
    for bias_like_t, label in ((facts.bias_t if facts.has_bias else None, "bias"), (facts.dbias_t if facts.has_dbias else None, "dBias")):
        if bias_like_t is None:
            continue
        dim = tuple(bias_like_t.get_dim())
        if dim not in ((1, facts.h_q, facts.s_q, facts.s_kv), (facts.b, facts.h_q, facts.s_q, facts.s_kv)):
            return f"{label} must be (1|B, H_q, S_q, S_kv) = (1|{facts.b}, {facts.h_q}, {facts.s_q}, {facts.s_kv}); got {dim}"
        expect_bias_stride = (facts.h_q * facts.s_q * facts.s_kv, facts.s_q * facts.s_kv, facts.s_kv, 1)
        if tuple(bias_like_t.get_stride()) != expect_bias_stride:
            return f"{label} must be contiguous {expect_bias_stride}; got stride {tuple(bias_like_t.get_stride())}"
        if bias_like_t.get_data_type() not in (facts.dtype, cudnn.data_type.FLOAT):
            return f"{label} must be fp32 or match the io dtype {facts.dtype}; got {bias_like_t.get_data_type()}"
    if facts.has_dbias and tuple(facts.dbias_t.get_dim()) != tuple(facts.bias_t.get_dim()):
        return f"dBias dims must match the bias dims {tuple(facts.bias_t.get_dim())}; got {tuple(facts.dbias_t.get_dim())}"
    if facts.has_dbias and facts.deterministic and facts.b > 1 and tuple(facts.bias_t.get_dim())[0] == 1:
        return "deterministic dBias requires a per-batch bias when B > 1 (a broadcast bias reduces over B through unordered atomics)"
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    lower: "Callable[[EngineSpec, ga.SdpaGraphFacts, Any], Any]"


def _sm120_spec() -> EngineSpec:
    return EngineSpec(
        name="sdpa_bwd_sm120",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            # Any head size multipled of 8
            d=frozenset(_SM120_HEAD_DIMS),
            d_envelope=True,  # any multiple of 8 computes on the next native size (TMA zero-fill)
            d_pad_multiple=8,
            dqk_ge_dv=True,
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            gqa=True,
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            dsink=True,
            bias=True,
            dbias=True,
            layouts=frozenset({"bshd", "dense_flex"}),
            strided_stats=True,
            deterministic=True,
            tile_ms=frozenset(_SM120_Q_TILES),
            tile_ns=frozenset(_SM120_KV_TILES),
        ),
        lower=lower_dsl_bwd,
    )


def analyze_for(spec: EngineSpec, graph, knobs: Optional[SdpaBwdKnobs] = None):
    """``(facts, reason)``: the parsed graph and the first reason ``spec``
    cannot serve it under ``knobs`` (``None`` when it can).

    The single eligibility entry point, shared by :func:`probe`, :func:`build`
    and ``engine.FrostSdpaBwdEngine.check_support``. ``knobs`` is the plan's
    tuning request (``PlanConfig.knobs``), ``None`` for no preference.
    """
    # The record validate() attached, not a fresh parse: one per graph, shared
    # with whatever ranked these plans before this engine was imported.
    facts = graph._facts_for(ga.analyze)
    if facts is None:
        return None, "graph is not a single sdpa_backward() node"
    return facts, mismatch(spec.capabilities, facts, knobs)


def build(spec: EngineSpec, graph, knobs: Optional[SdpaBwdKnobs] = None):
    """Lower ``spec`` for ``graph``, or raise the bare ineligibility reason (the
    caller — the engine — names itself in the message)."""
    facts, reason = analyze_for(spec, graph, knobs)
    if reason is not None:
        raise ValueError(reason)
    return spec.lower(spec, facts, knobs)


def lower_dsl_bwd(spec: EngineSpec, facts: "ga.SdpaGraphFacts", requested: Any = None, api_type: str = _SM120):
    """Lower the selected SDPA backward engine through its DSL adapter.

    Descriptor conversion, adapter lifecycle, variant-pack binding, and launch
    construction live here; the adapter owns compilation and the kernel
    execute chain. ``EngineSpec.lower`` binds the implementation through
    ``api_type`` (mirrors ``lower_dsl_prefill``); SM80-only operands (bias →
    dBias, plan-time bias facts) flow only to adapters declaring the matching
    constructor / execute keywords.
    """
    import inspect

    # Per-port geometry from facts.port_layouts, NOT the live IR tensors:
    # build_operation_graph rewrites the backward node's K/V ports to
    # transposed (B, H, D, S) views; the analyzer captured the geometry with
    # that rewrite undone.
    ports = {name: (tuple(dim), tuple(stride)) for name, dim, stride in facts.port_layouts}
    q_geom, k_geom, v_geom, o_geom = ports["q"], ports["k"], ports["v"], ports["o"]
    do_geom, dq_geom, dk_geom, dv_geom = ports["dO"], ports["dQ"], ports["dK"], ports["dV"]
    stats_geom = (tuple(facts.stats_t.get_dim()), tuple(facts.stats_t.get_stride()))

    import torch

    def _desc(geom, dtype, name: str) -> "Any":
        # facts carry cudnn.data_type; torch appears only here, where a real
        # torch tensor is being described.
        dtype = ga.to_torch_dtype(dtype) if dtype in ga._KNOWN_DTYPES else dtype
        from cudnn.api_base import TensorDesc

        dim, stride = geom
        return TensorDesc(
            dtype=dtype,
            shape=dim,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(dim, stride),
            device=torch.device("cuda", torch.cuda.current_device()),
            name=name,
        )

    # Per-side length operand. A ragged graph always carries them (its lengths
    # ARE the mask), and so does a dense padded one.
    #
    # NOT the (B+1,) prefix-sum form: `cu_seq_len_q/kv` exists only on the
    # FORWARD attributes (SDPA_attributes / SDPA_fp8_attributes, and the
    # standalone mask node) -- `SDPA_backward_attributes` has no such port and
    # `pygraph.sdpa_backward()` no such keyword, so `facts.has_cu_seq_len` is
    # always False here. The adapter reads either form (it tells them apart at
    # execute from numel() == B + 1), so when the backward node gains the port
    # this becomes a one-line pick and a `cu_seq_len=True` row -- with an accept
    # test, which cannot be written today. Verified 2026-09-03 against
    # python/pygraph/sdpa.cpp and graph_properties.h.
    seq_kv_t = facts.seq_kv_t if facts.padded else None
    seq_q_t = facts.seq_q_t if facts.padded else None
    # THD carries its lengths through the setup launch's metadata buffer, not
    # as a compiled-in padding mask, so the *_present specialization flags stay
    # off there (the adapter refuses them under THD) while the buffers still
    # flow to execute.
    thd = facts.thd
    # Packed Stats packing, read off the ragged declaration exactly as the
    # forward reads it (fwd/api_dsl.py, `_thd_lse_view`): token-major (T, H)
    # -- cuDNN's ragged-Stats recipe -- or head-major (1, QH, head_stride),
    # which is what the FROST forward emits natively. mismatch() has already
    # rejected anything that is neither.
    stats_stride_h, stats_stride_s = (int(stats_geom[1][1]), int(stats_geom[1][2])) if thd else (0, 0)
    stats_token_major = thd and (stats_stride_h, stats_stride_s) == (1, facts.h_q)
    stats_head_stride = stats_stride_h if (thd and not stats_token_major) else 0
    # Sink ports: geometry straight from the IR tensors (fp32 (1, H_q, 1, 1)).
    sink_geom = (tuple(facts.sink_t.get_dim()), tuple(facts.sink_t.get_stride())) if facts.has_sink else None
    dsink_geom = (tuple(facts.dsink_t.get_dim()), tuple(facts.dsink_t.get_stride())) if facts.has_dsink else None
    bias_geom = (tuple(facts.bias_t.get_dim()), tuple(facts.bias_t.get_stride())) if facts.has_bias else None
    dbias_geom = (tuple(facts.dbias_t.get_dim()), tuple(facts.dbias_t.get_stride())) if facts.has_dbias else None

    adapter_cls = _adapter(api_type)
    # SM80-only plan-time facts, forwarded only to adapters declaring them.
    _extra_ctor = {
        # THD / ragged (base-ctor parameters, so every adapter declares them;
        # they stay off for the rows whose Capabilities decline THD).
        "thd": thd,
        # Caller-declared packed token totals. Under THD they SIZE the blocked
        # S/dS workspace, which scratch_workspace_bytes() has to answer at
        # build time -- which is why mismatch() requires them there.
        "max_total_seq_len_q": facts.max_total_seq_len_q,
        "max_total_seq_len_kv": facts.max_total_seq_len_kv,
        "thd_stats_token_major": stats_token_major,
        "thd_stats_head_stride": stats_head_stride,
        "has_bias": facts.has_bias,
        "bias_is_fp32": (facts.bias_t.get_data_type() == cudnn.data_type.FLOAT) if facts.bias_t is not None else True,
        "bias_batch": int(facts.bias_t.get_dim()[0]) if facts.bias_t is not None else 1,
        "has_rope": False,  # RoPE-fused multi-node graphs never reach the analyzer
    }
    _extra_ctor = {k: v for k, v in _extra_ctor.items() if k in inspect.signature(adapter_cls.__init__).parameters}
    api = adapter_cls(
        sample_q=_desc(q_geom, facts.dtype, "q"),
        sample_k=_desc(k_geom, facts.dtype, "k"),
        sample_v=_desc(v_geom, facts.dtype, "v"),
        sample_o=_desc(o_geom, facts.dtype, "o"),
        sample_do=_desc(do_geom, facts.dtype, "dO"),
        sample_stats=_desc(stats_geom, torch.float32, "stats"),
        sample_dq=_desc(dq_geom, facts.dtype, "dQ"),
        sample_dk=_desc(dk_geom, facts.dtype, "dK"),
        sample_dv=_desc(dv_geom, facts.dtype, "dV"),
        sample_sink=_desc(sink_geom, facts.sink_t.get_data_type(), "sink") if sink_geom is not None else None,
        sample_dsink=_desc(dsink_geom, facts.dsink_t.get_data_type(), "dSink") if dsink_geom is not None else None,
        sample_bias=_desc(bias_geom, facts.bias_t.get_data_type(), "bias") if bias_geom is not None else None,
        sample_dbias=_desc(dbias_geom, facts.dbias_t.get_data_type(), "dBias") if dbias_geom is not None else None,
        is_causal=facts.causal or facts.right_band_widening,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        window_size_right=(facts.right_bound if facts.right_band_widening else None),
        deterministic=facts.deterministic,
        scale_softmax=facts.scale,
        tile_m=requested.tile_m if requested is not None else None,
        tile_n=requested.tile_n if requested is not None else None,
        seq_kv_lens_present=seq_kv_t is not None and not thd,
        seq_q_lens_present=seq_q_t is not None and not thd,
        **_extra_ctor,
    )
    api.check_support()  # raises ValueError / NotImplementedError if unsupported
    api.compile()

    # Workspace requirement for the compiled geometry (the FROST executor
    # contract, see engine._check_workspace): the delta and dq_accum fp32
    # scratch is carved from the CALLER's workspace at execute, so its size is
    # fixed here and recorded on the executor as ``workspace_bytes``.
    total_workspace_bytes = api.scratch_workspace_bytes()

    binding = ga.SdpaBinding(
        q=facts.q_t,
        k=facts.k_t,
        v=facts.v_t,
        o=facts.o_t,
        stats=facts.stats_t,
        do=facts.do_t,
        dq=facts.dq_t,
        dk=facts.dk_t,
        dv=facts.dv_t,
        seq_len_kv=seq_kv_t,
        seq_len_q=seq_q_t,
        sink_token=facts.sink_t if facts.has_sink else None,
        dsink=facts.dsink_t if facts.has_dsink else None,
        bias=facts.bias_t if facts.has_bias else None,
        dbias=facts.dbias_t if facts.has_dbias else None,
    )

    def _canonical_view(buf, geom):
        """Reinterpret a variant-pack buffer through the port's geometry.

        cuDNN's execute contract treats variant-pack entries as raw storage
        laid out per the IR tensor descriptor — callers may hand in a torch
        tensor whose logical shape is anything with the right bytes. The DSL
        executor consumes torch views, so rebuild the port-shaped view here.
        No-op when the caller already passed a matching view.
        """
        dim, stride = geom
        if tuple(buf.shape) == dim and tuple(buf.stride()) == stride:
            return buf
        return buf.as_strided(dim, stride)

    def _thd_view(buf, geom, tokens):
        """The packed ``(1, H, T, D)`` view of a RAGGED port's buffer.

        A ragged graph declares the ENVELOPE ``(B, H, S_max, D)`` plus a device
        ragged offset, so the port geometry describes a rectangle the packed
        buffer does not contain — reinterpreting through it (what
        :func:`_canonical_view` would do) reads past the end. The packed view
        keeps the port's own head / token / element strides and replaces the
        batch+sequence pair with a single T-token axis, which is the
        orientation the adapter consumes (it permutes to ``[1, T, H, D]``).

        ``as_strided`` is the bounds check: it refuses a view that runs past the
        buffer's storage, so a declared total larger than the allocation fails
        here rather than inside a TMA descriptor.
        """
        (_, h, _, d), (_, hs, ts, es) = geom
        return buf.as_strided((1, h, tokens, d), (max(tokens, 1) * ts, hs, ts, es), buf.storage_offset())

    def _thd_stats_view(buf, tokens):
        """The caller's ragged Stats buffer in its declared packing.

        Token-major ``(T, H)`` (cuDNN's ragged-Stats recipe) or head-major
        ``(1, QH, head_stride)`` (what the FROST forward emits). Both are shaped
        here so the adapter's reshape is a no-op — the compiled artifact binds a
        COMPACT fake tensor, so the head stride has to arrive as the third
        EXTENT, not as a stride.
        """
        h = int(stats_geom[0][1])
        if stats_token_major:
            return buf.as_strided((tokens, h), (h, 1), buf.storage_offset())
        hs = stats_head_stride or tokens
        return buf.as_strided((1, h, hs), (h * hs, hs, 1), buf.storage_offset())

    def _execute(variant_pack, workspace=None, stream=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        # Ragged ports bind PACKED views at the adapter's own token extents; the
        # dense ones keep the port-shaped reinterpretation. Q/O/dO/dQ ride the
        # Q-side total, K/V/dK/dV the KV-side one.
        if thd:
            t_q, t_kv = api.thd_total_q, api.thd_total_kv
            _v = lambda ref, geom, tokens: _thd_view(resolved[id(ref)], geom, tokens)
            q_buf, o_buf, do_buf, dq_buf = (_v(r, g, t_q) for r, g in ((binding.q, q_geom), (binding.o, o_geom), (binding.do, do_geom), (binding.dq, dq_geom)))
            k_buf, v_buf, dk_buf, dv_buf = (_v(r, g, t_kv) for r, g in ((binding.k, k_geom), (binding.v, v_geom), (binding.dk, dk_geom), (binding.dv, dv_geom)))
            stats_buf = _thd_stats_view(resolved[id(binding.stats)], t_q)
        else:
            _v = lambda ref, geom: _canonical_view(resolved[id(ref)], geom)
            q_buf, k_buf, v_buf = _v(binding.q, q_geom), _v(binding.k, k_geom), _v(binding.v, v_geom)
            o_buf, do_buf = _v(binding.o, o_geom), _v(binding.do, do_geom)
            dq_buf, dk_buf, dv_buf = _v(binding.dq, dq_geom), _v(binding.dk, dk_geom), _v(binding.dv, dv_geom)
            stats_buf = _v(binding.stats, stats_geom)
        seq_kv_buf = resolved.get(id(binding.seq_len_kv)) if binding.seq_len_kv is not None else None
        seq_q_buf = resolved.get(id(binding.seq_len_q)) if binding.seq_len_q is not None else None
        sink_buf = _canonical_view(resolved[id(binding.sink_token)], sink_geom) if binding.sink_token is not None else None
        dsink_buf = _canonical_view(resolved[id(binding.dsink)], dsink_geom) if binding.dsink is not None else None
        bias_buf = _canonical_view(resolved[id(binding.bias)], bias_geom) if binding.bias is not None else None
        dbias_buf = _canonical_view(resolved[id(binding.dbias)], dbias_geom) if binding.dbias is not None else None
        api.execute(
            q_tensor=q_buf,
            k_tensor=k_buf,
            v_tensor=v_buf,
            o_tensor=o_buf,
            do_tensor=do_buf,
            stats_tensor=stats_buf,
            dq_tensor=dq_buf,
            dk_tensor=dk_buf,
            dv_tensor=dv_buf,
            seq_q_lens=seq_q_buf,
            seq_kv_lens=seq_kv_buf,
            sink_tensor=sink_buf,
            dsink_tensor=dsink_buf,
            bias_tensor=bias_buf,
            dbias_tensor=dbias_buf,
            scale_softmax=facts.scale,
            # Scratch comes from the CALLER's workspace (never allocated
            # here): the dispatch sized/validated it against workspace_bytes;
            # the adapter's carver re-validates so a direct call cannot
            # silently corrupt memory.
            workspace=workspace,
            # Stream from the execute-time context (raw CUstream int,
            # engine plan passes ctx.stream); None keeps the adapter's
            # torch-current-stream fallback.
            current_stream=_cuda_driver().CUstream(stream) if stream is not None else None,
        )
        return None

    # Executor contract (engine._FrostSdpaBwdPlan): a non-zero workspace_bytes
    # means the plan calls _execute(variant_pack, workspace) with the caller's
    # buffer; 0 means _execute(variant_pack) and the buffer is never touched.
    # ``binding`` lets the plan key this executor's operands out of the graph's
    # variant pack (the pack covers every IO tensor of the graph).
    _execute.workspace_bytes = total_workspace_bytes
    _execute.binding = binding
    return _execute


def _sm80_spec() -> EngineSpec:
    """SM80 (A100) backward row: lowers through the shared ``lower_dsl_bwd``
    onto ``SdpaBwdDslSm80`` (``bwd/api_dsl.py``), which owns kernel-flavor selection
    (head-dim envelopes up to (256, 256), incl. rectangular 192/128),
    host-side head-dim zero-padding, BHSD<->BSHD normalization, per-shape
    kernel caching, and the dedicated plain-dense d=64 fast path.  The
    CuTe-DSL JIT happens on the first execute (the kernel modules self-cache
    per shape).  Knob domains are empty (no tunables wired).  sm80 exactly:
    the kernels assume the A100's 164 KiB opt-in SMEM, which the sm86/sm89
    parts do not have."""
    return EngineSpec(
        name="sdpa_bwd_sm80",
        capabilities=Capabilities(
            sm_lo=80,
            sm_hi=80,
            d=frozenset({256}),
            d_envelope=True,  # flavor envelopes; host-side zero-padding
            dqk_ge_dv=True,  # bprop kernel constraint relaxed to D_QK >= D_V
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            gqa=True,
            causal=True,
            bottom_right=True,
            deterministic=True,  # ordered dQ KV-tile reduction
            dbias=True,
            dsink=True,
            bias=True,
            right_band_widening=True,
            seq_q_trim=False,
            swa=True,
            padded=True,
            sink=True,
            decode=False,  # prefill kernels only
            layouts=frozenset({"bshd", "dense_flex"}),
            # The kernels READ the declared stats strides natively (the
            # #712 analogue for the backward's loads), same as sm120 — no
            # gather staging.
            strided_stats=True,
        ),
        lower=partial(lower_dsl_bwd, api_type=_SM80),
    )


def _sm100_spec() -> EngineSpec:
    """SM100 (Blackwell) large-head-dim backward row.

    A three-stage chain rather than one fused kernel, because a fused d=512
    backward does not fit: dV [128 kv, 512] fp32 needs 512 TMEM columns and dK
    another 512, plus the S/dS accumulators -- against 512 per CTA.

        stage 1  do_dot = rowsum(dO * O)          (shared with the sm120 chain)
        stage 2  S and dS workspaces              bprop_d512_f16_sm100.py
        stage 3  dV = S^T.dO, dK = dS^T.Q, dQ = dS.K   bprop_matmul_sm100.py

    ``d`` is an ENVELOPE: the kernel's tiles are fixed at 512 and a smaller head
    dim rides the TMA descriptors' real extent, whose overshoot is HW
    zero-filled, so the padded lanes contribute nothing. That costs the full
    512-wide MMA at any d, which is why the lower bound is 264 -- below that the
    d256 flavors are the right kernel. ``d_pad_multiple=8`` is TMA's 16-byte
    innermost-extent rule at 2 B/elem; the stage-3 epilogue store vector narrows
    from 32 B to 16 B when d is not also a multiple of 16.

    THD / ragged is served on the packed path: Q/K/V/O/dO and the gradients are
    PACKED ``[1, T, H, D]``, the S/dS workspace is ROW-BLOCKED (each sequence
    owns a TILE_M-aligned block, columns uniform at pad(S_kv_max)), and a setup
    launch writes the metadata, the per-sequence block offsets and the clipped
    output descriptors -- all device-side, no host cumsum. Both packed Stats
    layouts the forward can emit are read. Lengths arrive as per-batch
    ``seq_len_q/kv`` (the backward node has no ``cu_seq_len_*`` port). The causal
    family is served: stage 2 masks from the per-sequence metadata lengths (the
    bottom-right diagonal included, as ``S_kv[b] - S_q[b]``), and stage 3 is
    rendered UNTRIMMED because its K-trim is in absolute workspace rows -- the
    adapter zero-fills the blocked workspace instead. Its remaining conjunctions
    are declined and each is asserted by a reject test: GQA (the dK/dV partials
    would have to be packed per Q head), and a graph that does not declare
    ``max_total_seq_len_q/kv`` (``scratch_workspace_bytes()`` is a build-time
    function and the blocked row count comes from the packed totals).

    Everything else is declined for now and each rejection is asserted by a
    test: DENSE padding masks (the kernel compiles a scalar length; THD threads
    the per-sequence machinery, so this is a small follow-up on top of it),
    bias/dbias/sink/dsink, deterministic, and decode.
    """
    return EngineSpec(
        name="sdpa_bwd_sm100",
        capabilities=Capabilities(
            sm_lo=100,
            sm_hi=103,
            d=frozenset({512}),
            d_envelope=True,
            d_pad_multiple=8,
            d_envelope_floor=256,  # <= 256 belongs to the d256 flavors
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            swa=True,
            right_band_widening=True,
            gqa=True,  # per-Q-head dK/dV partials + the shared dkv_reduce group fold
            thd=True,
            thd_causal=True,  # stage 2 masks per sequence; stage 3 drops its trim and leans on the zero-fill
            thd_declared_totals=True,  # the blocked workspace is sized at build time
            # Any dense layout: the adapter uses a BSHD-physical tensor in place
            # (a permuted view, zero copy) and stages a non-conforming one
            # through the workspace. That is not hypothetical -- a caller that
            # builds dO as torch.randn(o.shape) instead of empty_like(o) loses
            # o's memory format and hands over a BHSD-contiguous dO.
            layouts=frozenset({"bshd", "dense_flex"}),
        ),
        lower=partial(lower_dsl_bwd, api_type=_SM100),
    )


def engine_name(arch: str = "sm120") -> str:
    """The shipped engine name for a coverage cell (test/user convenience)."""

    return f"sdpa_bwd_{arch}"


# Preference order: the ranked plan list offers these in this order (see
# cudnn/sdpa/bwd/engine.py, which wraps each spec as a BaseEngine).
ENGINE_SPECS = (_sm120_spec(), _sm80_spec(), _sm100_spec())

__all__ = ["ENGINE_SPECS", "Capabilities", "EngineSpec", "SdpaBwdKnobs", "analyze_for", "engine_name", "mismatch"]
