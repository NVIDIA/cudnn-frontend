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
import torch
from cuda.bindings import driver as _cuda_driver

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm120
from cudnn.sdpa.bwd.config_sm120 import (
    SEQ_KV_TILES as _SM120_KV_TILES,
    SEQ_Q_TILES as _SM120_Q_TILES,
    SUPPORTED_HEAD_DIMS as _SM120_HEAD_DIMS,
)

_LOG = logging.getLogger(__name__)

_BLACKWELL_GEFORCE_ARCHES = frozenset[tuple[int, int]]({(12, 0), (12, 1)})


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

    arches: frozenset[tuple[int, int]]
    d: frozenset[int]  # supported head dims (d_qk == d_v required)
    dtypes: frozenset[torch.dtype] = frozenset({torch.float16, torch.bfloat16})

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
    if facts.device_cc not in capabilities.arches:
        required = " or ".join(f"SM{major}{minor}" for major, minor in sorted(capabilities.arches))
        return f"requires {required}; current device is {facts.device_cc}"
    if facts.is_mxfp8 or facts.is_fp8:
        return "this engine serves only half (fp16/bf16) sdpa_backward graphs"
    if facts.d_qk != facts.d_v:
        return f"D_QK must equal D_V; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    if facts.d_qk not in capabilities.d:
        return f"serves D in {sorted(capabilities.d)}; graph has D={facts.d_qk}"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if not facts.uniform_dtype:
        return "K/V/O/dO/dQ/dK/dV dtypes must match Q"
    if facts.h_q != facts.h_kv and not capabilities.gqa:
        return f"GQA / MQA is not supported (H_q={facts.h_q}, H_kv={facts.h_kv})"
    if not facts.bshd_layout:
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
        (facts.causal, capabilities.causal, "causal mask"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

    if facts.bottom_right and not facts.causal:
        return "bottom-right alignment requires a causal upper bound"
    if facts.causal and facts.bottom_right and not capabilities.bottom_right:
        return "graph uses bottom-right causal, which this engine does not support"

    # The kernel consumes the forward stats as a contiguous natural-log LSE
    # (fp32 (B, H_q, S_q, 1)); a strided stats view has no zero-copy reshape.
    if facts.stats_t is not None:
        if facts.stats_t.get_data_type() != cudnn.data_type.FLOAT:
            return f"stats must be fp32; got {facts.stats_t.get_data_type()}"
        s_dim = tuple(facts.stats_t.get_dim())
        s_stride = tuple(facts.stats_t.get_stride())
        expect_dim = (facts.b, facts.h_q, facts.s_q, 1)
        expect_stride = (facts.h_q * facts.s_q, facts.s_q, 1, 1)
        if s_dim != expect_dim:
            return f"stats must be (B, H_q, S_q, 1) = {expect_dim}; got {s_dim}"
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
            arches=_BLACKWELL_GEFORCE_ARCHES,
            d=frozenset(_SM120_HEAD_DIMS),
            dtypes=frozenset({torch.float16, torch.bfloat16}),
            causal=True,
            bottom_right=True,
            swa=True,
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
    facts = ga.analyze(graph)
    if facts is None:
        return None, "graph is not a single sdpa_backward() node"
    return facts, mismatch(spec.capabilities, facts, knobs)


def probe(spec: EngineSpec, graph, knobs: Optional[SdpaBwdKnobs] = None) -> bool:
    _, reason = analyze_for(spec, graph, knobs)
    if reason is not None:
        _LOG.debug("cudnn.sdpa: %s ineligible: %s", spec.name, reason)
        return False
    return True


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

    # Canonical BSHD-physical geometry, fixed at build time from the facts.
    # Deliberately NOT read back from the IR tensors at execute:
    # ``build_operation_graph`` rewrites the backward node's K/V ports to
    # transposed (B, H, D, S) views, so the live ``get_dim()`` after a native
    # build would describe the transposed view while the underlying buffer
    # keeps the user's canonical layout (which the bshd gate already proved).
    def _bshd_geometry(b: int, h: int, s: int, d: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (b, h, s, d), (s * h * d, d, h * d, 1)

    q_geom = _bshd_geometry(facts.b, facts.h_q, facts.s_q, facts.d_qk)
    kv_geom = _bshd_geometry(facts.b, facts.h_kv, facts.s_kv, facts.d_qk)
    stats_geom = ((facts.b, facts.h_q, facts.s_q, 1), (facts.h_q * facts.s_q, facts.s_q, 1, 1))

    def _desc(geom, dtype: torch.dtype, name: str) -> "Any":
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

    api = SdpaBwdDslSm120(
        sample_q=_desc(q_geom, facts.dtype, "q"),
        sample_k=_desc(kv_geom, facts.dtype, "k"),
        sample_v=_desc(kv_geom, facts.dtype, "v"),
        sample_o=_desc(q_geom, facts.dtype, "o"),
        sample_do=_desc(q_geom, facts.dtype, "dO"),
        sample_stats=_desc(stats_geom, torch.float32, "stats"),
        sample_dq=_desc(q_geom, facts.dtype, "dQ"),
        sample_dk=_desc(kv_geom, facts.dtype, "dK"),
        sample_dv=_desc(kv_geom, facts.dtype, "dV"),
        is_causal=facts.causal,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        scale_softmax=facts.scale,
        tile_m=requested.tile_m if requested is not None else None,
        tile_n=requested.tile_n if requested is not None else None,
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
    )

    def _canonical_view(buf: torch.Tensor, geom) -> torch.Tensor:
        """Reinterpret a variant-pack buffer through the canonical geometry.

        cuDNN's execute contract treats variant-pack entries as raw storage
        laid out per the IR tensor descriptor — callers may hand in a torch
        tensor whose logical shape is anything with the right bytes. The DSL
        executor consumes torch views, so rebuild the canonical view here.
        No-op when the caller already passed a matching view.
        """
        dim, stride = geom
        if tuple(buf.shape) == dim and tuple(buf.stride()) == stride:
            return buf
        return buf.as_strided(dim, stride)

    def _execute(variant_pack, workspace=None, stream=None):
        resolved = ga.resolve_variant_pack(variant_pack, binding)
        api.execute(
            q_tensor=_canonical_view(resolved[id(binding.q)], q_geom),
            k_tensor=_canonical_view(resolved[id(binding.k)], kv_geom),
            v_tensor=_canonical_view(resolved[id(binding.v)], kv_geom),
            o_tensor=_canonical_view(resolved[id(binding.o)], q_geom),
            do_tensor=_canonical_view(resolved[id(binding.do)], q_geom),
            stats_tensor=_canonical_view(resolved[id(binding.stats)], stats_geom),
            dq_tensor=_canonical_view(resolved[id(binding.dq)], q_geom),
            dk_tensor=_canonical_view(resolved[id(binding.dk)], kv_geom),
            dv_tensor=_canonical_view(resolved[id(binding.dv)], kv_geom),
            scale_softmax=facts.scale,
            # Scratch comes from the CALLER's workspace (never allocated
            # here): the dispatch sized/validated it against workspace_bytes;
            # the adapter's carver re-validates so a direct call cannot
            # silently corrupt memory.
            workspace=workspace,
            # Stream from the execute-time context (raw CUstream int,
            # engine plan passes ctx.stream); None keeps the adapter's
            # torch-current-stream fallback.
            current_stream=_cuda_driver.CUstream(stream) if stream is not None else None,
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


def engine_name(arch: str = "sm120") -> str:
    """The shipped engine name for a coverage cell (test/user convenience)."""

    return f"sdpa_bwd_{arch}"


# Preference order: the ranked plan list offers these in this order (see
# cudnn/sdpa/bwd/engine.py, which wraps each spec as a BaseEngine).
ENGINE_SPECS = (_sm120_spec(),)

__all__ = ["ENGINE_SPECS", "Capabilities", "EngineSpec", "SdpaBwdKnobs", "analyze_for", "engine_name", "mismatch"]
