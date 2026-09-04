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
_SM100_MXFP8 = "SdpaBwdDslSm100Mxfp8"


def _adapter(name: str):
    if name == _SM100_MXFP8:
        # Its own module: the MXFP8 lowering's imports stay out of the
        # half-precision adapters' way.
        from cudnn.sdpa.bwd import api_dsl_mxfp8_sm100

        return api_dsl_mxfp8_sm100.SdpaBwdDslSm100Mxfp8
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
    # Quantization family (mirrors fwd/engines.py): a row serves exactly one of
    # half ``sdpa_backward`` (both False), block-scale MXFP8
    # ``sdpa_mxfp8_backward`` (is_mxfp8), or per-tensor FP8
    # ``sdpa_fp8_backward`` (is_fp8; no row yet). ``out_dtypes`` is the domain
    # of the half-precision side of a quantized graph (o_f16 / dO_f16 / dQ /
    # dK / dV share it); empty on half rows, where the io dtype IS ``dtypes``.
    is_mxfp8: bool = False
    is_fp8: bool = False
    out_dtypes: frozenset = frozenset()
    # MXFP8 backward: amax_dQ / amax_dK / amax_dV outputs. The kernels write
    # half-precision gradients and do not produce these; a graph requesting
    # one (set_output(True)) is declined rather than left with garbage.
    amax_dgrad: bool = False

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
    cu_seq_len: bool = False  # cu_seq_len_q / cu_seq_len_kv prefix sums (no row serves these yet)
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
    if (facts.is_mxfp8, facts.is_fp8) != (capabilities.is_mxfp8, capabilities.is_fp8):
        quant = (
            "block-scale MXFP8 (sdpa_mxfp8_backward)"
            if capabilities.is_mxfp8
            else "per-tensor FP8 (sdpa_fp8_backward)" if capabilities.is_fp8 else "half (fp16/bf16) sdpa_backward"
        )
        return f"this engine serves only {quant} graphs"
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
    if capabilities.is_mxfp8 or capabilities.is_fp8:
        if not facts.uniform_dtype:
            return "the FP8 payloads (K/V/dO and the transposed copies) must share Q's dtype"
        if facts.dtype_o not in capabilities.out_dtypes:
            return f"half-precision O/dO_f16/dQ/dK/dV dtype {facts.dtype_o} not in {sorted(str(d) for d in capabilities.out_dtypes)}"
        if not facts.uniform_out_dtype:
            return "o_f16/dO_f16/dQ/dK/dV dtypes must match"
        if facts.has_amax_dgrad and not capabilities.amax_dgrad:
            return "graph requests amax_dQ/dK/dV outputs, which this engine does not produce"
    elif not facts.uniform_dtype:
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
        ports = "Q/K/V/O/dO/dQ/dK/dV" + ("/q_T/k_T/dO_T/dO_f16" if facts.is_mxfp8 else "")
        return f"{ports} must be BSHD-physical (stride order 3,1,2,0)"

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
        (facts.padded, capabilities.padded, "padding mask"),
        (facts.has_sink, capabilities.sink, "sink token"),
        (facts.thd, capabilities.thd, "THD / ragged"),
        (facts.has_cu_seq_len, capabilities.cu_seq_len, "cu_seq_len_q / cu_seq_len_kv"),
        (facts.causal, capabilities.causal, "causal mask"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

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
        if capabilities.strided_stats:
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

    seq_kv_t = facts.seq_kv_t if facts.padded else None
    seq_q_t = facts.seq_q_t if facts.padded else None
    # Sink ports: geometry straight from the IR tensors (fp32 (1, H_q, 1, 1)).
    sink_geom = (tuple(facts.sink_t.get_dim()), tuple(facts.sink_t.get_stride())) if facts.has_sink else None
    dsink_geom = (tuple(facts.dsink_t.get_dim()), tuple(facts.dsink_t.get_stride())) if facts.has_dsink else None
    bias_geom = (tuple(facts.bias_t.get_dim()), tuple(facts.bias_t.get_stride())) if facts.has_bias else None
    dbias_geom = (tuple(facts.dbias_t.get_dim()), tuple(facts.dbias_t.get_stride())) if facts.has_dbias else None

    adapter_cls = _adapter(api_type)
    # SM80-only plan-time facts, forwarded only to adapters declaring them.
    _extra_ctor = {
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
        seq_kv_lens_present=seq_kv_t is not None,
        seq_q_lens_present=seq_q_t is not None,
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

    def _execute(variant_pack, workspace=None, stream=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        seq_kv_buf = resolved.get(id(binding.seq_len_kv)) if binding.seq_len_kv is not None else None
        seq_q_buf = resolved.get(id(binding.seq_len_q)) if binding.seq_len_q is not None else None
        sink_buf = _canonical_view(resolved[id(binding.sink_token)], sink_geom) if binding.sink_token is not None else None
        dsink_buf = _canonical_view(resolved[id(binding.dsink)], dsink_geom) if binding.dsink is not None else None
        bias_buf = _canonical_view(resolved[id(binding.bias)], bias_geom) if binding.bias is not None else None
        dbias_buf = _canonical_view(resolved[id(binding.dbias)], dbias_geom) if binding.dbias is not None else None
        api.execute(
            q_tensor=_canonical_view(resolved[id(binding.q)], q_geom),
            k_tensor=_canonical_view(resolved[id(binding.k)], k_geom),
            v_tensor=_canonical_view(resolved[id(binding.v)], v_geom),
            o_tensor=_canonical_view(resolved[id(binding.o)], o_geom),
            do_tensor=_canonical_view(resolved[id(binding.do)], do_geom),
            stats_tensor=_canonical_view(resolved[id(binding.stats)], stats_geom),
            dq_tensor=_canonical_view(resolved[id(binding.dq)], dq_geom),
            dk_tensor=_canonical_view(resolved[id(binding.dk)], dk_geom),
            dv_tensor=_canonical_view(resolved[id(binding.dv)], dv_geom),
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

    Everything else is declined for now and each rejection is asserted by a
    test: GQA (needs the dK/dV group reduce wired), the non-causal masks, THD
    (the workspace goes ragged and stage 3 becomes a variable-K grouped GEMM),
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
def lower_dsl_bwd_mxfp8(spec: EngineSpec, facts: "ga.SdpaGraphFacts", requested: Any = None, api_type: str = _SM100_MXFP8):
    """Lower the SM100 MXFP8 backward engine through ``SdpaBwdDslSm100Mxfp8``.

    A sibling of :func:`lower_dsl_bwd` rather than a branch inside it: the
    MXFP8 node binds eleven more operands (the transposed-quantization
    payloads, the half-precision dO, seven block-scale tensors), and threading
    those through the half-precision adapters' keyword filter would have every
    one of them declare slots it never reads.
    """
    import torch

    from cudnn.api_base import TensorDesc

    ports = {name: (tuple(dim), tuple(stride)) for name, dim, stride in facts.port_layouts}

    def _desc(t, dtype_hint, name: str):
        """Descriptor from an IR tensor (dims/strides straight from it; a
        reordered SF tensor is an opaque byte layout and only its byte count is
        ever consulted)."""
        dim, stride = tuple(t.get_dim()), tuple(t.get_stride())
        dtype = ga.to_torch_dtype(dtype_hint) if dtype_hint in ga._KNOWN_DTYPES else dtype_hint
        return TensorDesc(
            dtype=dtype,
            shape=dim,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(dim, stride),
            device=torch.device("cuda", torch.cuda.current_device()),
            name=name,
        )

    def _port_desc(name: str, dtype_hint, ir_t):
        dim, stride = ports[name]
        dtype = ga.to_torch_dtype(dtype_hint) if dtype_hint in ga._KNOWN_DTYPES else dtype_hint
        return TensorDesc(
            dtype=dtype,
            shape=dim,
            stride=stride,
            stride_order=TensorDesc._compute_stride_order(dim, stride),
            device=torch.device("cuda", torch.cuda.current_device()),
            name=name,
        )

    fp8, half = facts.dtype, facts.dtype_o
    api = _adapter(api_type)(
        sample_q=_port_desc("q", fp8, facts.q_t),
        sample_k=_port_desc("k", fp8, facts.k_t),
        sample_v=_port_desc("v", fp8, facts.v_t),
        sample_o=_port_desc("o", half, facts.o_t),
        sample_do=_port_desc("dO", fp8, facts.do_t),
        sample_stats=_desc(facts.stats_t, torch.float32, "stats"),
        sample_dq=_port_desc("dQ", half, facts.dq_t),
        sample_dk=_port_desc("dK", half, facts.dk_t),
        sample_dv=_port_desc("dV", half, facts.dv_t),
        sample_q_T=_port_desc("q_T", fp8, facts.q_T_t),
        sample_k_T=_port_desc("k_T", fp8, facts.k_T_t),
        sample_do_T=_port_desc("dO_T", fp8, facts.dO_T_t),
        sample_do_f16=_port_desc("dO_f16", half, facts.dO_f16_t),
        # E8M0 has no torch dtype on the lowering boundary; the descs carry the
        # byte view the adapter consumes.
        sample_sf_q=_desc(facts.sf_q_t, torch.int8, "sf_q"),
        sample_sf_q_T=_desc(facts.sf_q_T_t, torch.int8, "sf_q_T"),
        sample_sf_k=_desc(facts.sf_k_t, torch.int8, "sf_k"),
        sample_sf_k_T=_desc(facts.sf_k_T_t, torch.int8, "sf_k_T"),
        sample_sf_v=_desc(facts.sf_v_t, torch.int8, "sf_v"),
        sample_sf_do=_desc(facts.sf_dO_t, torch.int8, "sf_do"),
        sample_sf_do_T=_desc(facts.sf_dO_T_t, torch.int8, "sf_do_T"),
        is_causal=facts.causal or facts.right_band_widening,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        window_size_right=(facts.right_bound if facts.right_band_widening else None),
        deterministic=facts.deterministic,
        scale_softmax=facts.scale,
        tile_m=requested.tile_m if requested is not None else None,
        tile_n=requested.tile_n if requested is not None else None,
        seq_kv_lens_present=facts.padded,
        seq_q_lens_present=facts.padded,
    )
    api.check_support()  # raises ValueError / NotImplementedError if unsupported
    api.compile()
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
        q_T=facts.q_T_t,
        k_T=facts.k_T_t,
        dO_T=facts.dO_T_t,
        dO_f16=facts.dO_f16_t,
        sf_q=facts.sf_q_t,
        sf_k=facts.sf_k_t,
        sf_v=facts.sf_v_t,
        sf_q_T=facts.sf_q_T_t,
        sf_k_T=facts.sf_k_T_t,
        sf_dO=facts.sf_dO_t,
        sf_dO_T=facts.sf_dO_T_t,
    )

    def _view(buf, name):
        """Reinterpret a variant-pack buffer through the port's geometry (see
        lower_dsl_bwd._canonical_view); SF buffers are consumed flat."""
        if name not in ports:
            return buf
        dim, stride = ports[name]
        if tuple(buf.shape) == dim and tuple(buf.stride()) == stride:
            return buf
        return buf.as_strided(dim, stride)

    def _execute(variant_pack, workspace=None, stream=None):
        r = ga.resolve_variant_pack(variant_pack, binding)
        stats_dim, stats_stride = tuple(facts.stats_t.get_dim()), tuple(facts.stats_t.get_stride())
        stats_buf = r[id(binding.stats)]
        if tuple(stats_buf.shape) != stats_dim or tuple(stats_buf.stride()) != stats_stride:
            stats_buf = stats_buf.as_strided(stats_dim, stats_stride)
        api.execute(
            q_tensor=_view(r[id(binding.q)], "q"),
            k_tensor=_view(r[id(binding.k)], "k"),
            v_tensor=_view(r[id(binding.v)], "v"),
            o_tensor=_view(r[id(binding.o)], "o"),
            do_tensor=_view(r[id(binding.do)], "dO"),
            stats_tensor=stats_buf,
            dq_tensor=_view(r[id(binding.dq)], "dQ"),
            dk_tensor=_view(r[id(binding.dk)], "dK"),
            dv_tensor=_view(r[id(binding.dv)], "dV"),
            q_T_tensor=_view(r[id(binding.q_T)], "q_T"),
            k_T_tensor=_view(r[id(binding.k_T)], "k_T"),
            do_T_tensor=_view(r[id(binding.dO_T)], "dO_T"),
            do_f16_tensor=_view(r[id(binding.dO_f16)], "dO_f16"),
            sf_q=r[id(binding.sf_q)],
            sf_q_T=r[id(binding.sf_q_T)],
            sf_k=r[id(binding.sf_k)],
            sf_k_T=r[id(binding.sf_k_T)],
            sf_v=r[id(binding.sf_v)],
            sf_do=r[id(binding.sf_dO)],
            sf_do_T=r[id(binding.sf_dO_T)],
            scale_softmax=facts.scale,
            workspace=workspace,
            current_stream=_cuda_driver().CUstream(stream) if stream is not None else None,
        )
        return None

    _execute.workspace_bytes = total_workspace_bytes
    _execute.binding = binding
    return _execute


def _sm100_mxfp8_spec() -> EngineSpec:
    """SM100 (Blackwell) d=256 block-scale MXFP8 backward row.

    Two kernels (dQ; fused dK/dV) ported from the ``fmha_mxfp8_large_head_dim``
    repo, preceded by a
    scale-factor repack into the kernels' 2-CTA slot layout (see
    ``api_dsl_mxfp8_sm100`` for the Rule-2 exception this is). Exact d=256 only
    -- the SF plumbing has no envelope story, as on the forward MXFP8 row.

    Served: E4M3 payloads with fp16/bf16 half side, BSHD-physical layout, MHA /
    GQA / MQA, any fixed S_q / S_kv (tails are masked), dense and top-left
    causal, and ``deterministic`` (both kernels own their output tiles; nothing
    accumulates through atomics). dS is always quantized with an in-kernel
    per-block (online) E8M0 scale -- the upstream "fixed scale 1" mode is not
    exposed. Declined: E5M2, bottom-right / band-widened / sliding-window
    masks, padding, THD, bias / dBias, sink / dSink, amax_dQ/dK/dV outputs,
    non-BSHD strides (the kernels derive their head/batch strides; no staging
    copies).
    """
    return EngineSpec(
        name="sdpa_bwd_sm100_mxfp8",
        capabilities=Capabilities(
            sm_lo=100,
            sm_hi=106,  # no sm107 MXFP8 lowering (matches the forward MXFP8 row)
            d=frozenset({256}),
            d_envelope=False,
            dtypes=frozenset({cudnn.data_type.FP8_E4M3}),
            out_dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            is_mxfp8=True,
            causal=True,
            gqa=True,
            deterministic=True,
            layouts=frozenset({"bshd"}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
        ),
        lower=partial(lower_dsl_bwd_mxfp8, api_type=_SM100_MXFP8),
    )


ENGINE_SPECS = (_sm120_spec(), _sm80_spec(), _sm100_spec(), _sm100_mxfp8_spec())

__all__ = [
    "ENGINE_SPECS",
    "Capabilities",
    "EngineSpec",
    "SdpaBwdKnobs",
    "analyze_for",
    "engine_name",
    "mismatch",
]
