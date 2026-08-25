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
from typing import Any, Callable, Optional

import cudnn
from cudnn.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_state, cutedsl_too_old
from cudnn.sdpa import graph_analyzer as ga


# Lowering dependencies, resolved at build time — see the note in fwd/engines.py:
# importing them here would drag the CuTe DSL into every support check.
def _adapter_sm120():
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120

    return SdpaBwdDslSm120


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


def lower_dsl_bwd(spec: EngineSpec, facts: "ga.SdpaGraphFacts", requested: Any = None):
    """Lower the selected SDPA backward engine through its DSL adapter.

    Descriptor conversion, adapter lifecycle, variant-pack binding, and launch
    construction live here; the adapter owns compilation and the three-kernel
    execute chain.
    """

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

    api = _adapter_sm120()(
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
    """SM80 (A100) backward row: lowers onto the ``cudnn.sdpa`` SM80 APIBase
    adapter (``bwd/api.py``), which owns kernel-flavor selection
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
        ),
        lower=lower_sm80_bwd,
    )


def lower_sm80_bwd(spec: EngineSpec, facts: "ga.SdpaGraphFacts", requested: Any = None):
    """Lower the SM80 backward row through the ``cudnn.sdpa`` SM80 adapter."""
    from .api import sdpa_bwd_wrapper_sm80

    binding = ga.SdpaBinding(
        q=facts.q_t,
        k=facts.k_t,
        v=facts.v_t,
        o=facts.o_t,
        stats=facts.stats_t,
        bias=facts.bias_t,
        sink_token=facts.sink_t,
        seq_len_kv=facts.seq_kv_t,
        seq_len_q=facts.seq_q_t,
        do=facts.do_t,
        dq=facts.dq_t,
        dk=facts.dk_t,
        dv=facts.dv_t,
        dbias=facts.dbias_t,
        dsink=facts.dsink_t,
    )
    mask_args = ga.adapter_mask_args(facts)

    def _execute(variant_pack, stream=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        # mismatch() admits only the contiguous (B, H_q, S_q, 1) stats layout,
        # so this is a pure view (a copying reshape would violate the
        # execute() contract and hide the -inf padded-row trim semantics).
        lse = resolved[id(facts.stats_t)].view(facts.b, facts.h_q, facts.s_q)

        out = sdpa_bwd_wrapper_sm80(
            # dense_flex delivery: normalize to the BSHD-physical order the
            # adapter requires (zero-copy when already BSHD).
            ga.to_bshd_physical(resolved[id(facts.q_t)]),
            ga.to_bshd_physical(resolved[id(facts.k_t)]),
            ga.to_bshd_physical(resolved[id(facts.v_t)]),
            ga.to_bshd_physical(resolved[id(facts.o_t)]),
            ga.to_bshd_physical(resolved[id(facts.do_t)]),
            lse,
            scale_softmax=facts.scale,
            deterministic=facts.deterministic,
            # Stream from the caller's handle (ExecutionContext.stream);
            # None keeps the current stream.
            current_stream=stream,
            **mask_args,
            **ga.adapter_feature_buffers(facts, resolved),
        )

        # copy_ casts in place; no .to() (which would allocate a staging
        # tensor per execute).
        for t_ref, key in ((facts.dq_t, "dq_tensor"), (facts.dk_t, "dk_tensor"), (facts.dv_t, "dv_tensor")):
            resolved[id(t_ref)].copy_(out[key])
        if facts.has_dbias and "dbias_tensor" in out:
            buf = resolved.get(id(facts.dbias_t))
            if buf is not None:
                buf.copy_(out["dbias_tensor"].view(buf.shape))
        if facts.has_dsink and "dsink_tensor" in out:
            buf = resolved.get(id(facts.dsink_t))
            if buf is not None:
                buf.view(-1).copy_(out["dsink_tensor"])
        return None

    # Executor contract (engine._FrostSdpaBwdPlan): torch-native host code,
    # no carved scratch — workspace_bytes 0 means _execute(variant_pack).
    _execute.workspace_bytes = 0
    _execute.binding = binding
    return _execute


def engine_name(arch: str = "sm120") -> str:
    """The shipped engine name for a coverage cell (test/user convenience)."""

    return f"sdpa_bwd_{arch}"


# Preference order: the ranked plan list offers these in this order (see
# cudnn/sdpa/bwd/engine.py, which wraps each spec as a BaseEngine).
ENGINE_SPECS = (_sm120_spec(), _sm80_spec())

__all__ = ["ENGINE_SPECS", "Capabilities", "EngineSpec", "SdpaBwdKnobs", "analyze_for", "engine_name", "mismatch"]
