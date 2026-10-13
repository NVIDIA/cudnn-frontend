# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm107_fp8``: the Rubin (SM 10.7) per-tensor FP8 E4M3 d=256 SDPA backward.

The row implements cuDNN's ``sdpa_fp8_backward`` contract: FP8 Q / K / V / O / dO
with scalar descales in, fp32 Stats, FP8 dQ / dK / dV scaled by ``scale_dQ / dK
/ dV`` out, plus the four ``amax_dQ / dK / dV / dP`` outputs (``amax_dP`` is the
amax of the fp32 ``dS = P (dP - D) attn_scale`` right before its ``scale_dP``
cast -- what the C++ node's reduction sits on -- NOT the amax of ``dO V^T``).

ACCEPT tests run the graph with the engine PINNED against the repo's fp8
backward oracle (``sdpa.fp8_ref.compute_ref_backward``, the reference the cuDNN
backend's fp8 suite passes against) on BOTH dS workspace dtypes of the row
(``api_dsl_sm107.FP8_DS_DTYPE``, the ``ds_knob`` fixture): the shipped e4m3 dS
(``dS_q = e4m3(dS * scale_dP)`` into the fp8 K64 GEMM arm with the descale /
quantize epilogue -- the backend's own recipe, ``quantize_ds=True``) and the bf16
twin (bf16 dS into the bf16 GEMMs, ``quantize_ds=False``: the reference composes
the SAME rounding, sdpa-invariants s8 -- an e4m3-dS reference against a bf16-dS
chain misreports the row by its own rounding noise, measured on correct inputs at
0.56 % of dQ / 0.54 % of dK outside atol 0.08 at B1 H2 S512, max |diff| 0.17).
Both run under the backend suite's tolerance recipe and ``assert_close_fp8_grad``'s
midpoint-flip budget (P-side flips on both; dS-side flips on the e4m3 chain,
``flip_unit = 1 / scale_dP``).  dV and amax_dP do not depend on the dS dtype and
are pinned BITWISE across the knob.  REJECT tests
assert the decline through the row's own ``mismatch()`` on REAL graphs with a
faked cc 10.7 device -- including the one shape-conditioned decline this row
carries and the half row does not: bottom-right causal with ``S_q % 128 != 0``
(the fp8 body derives the diagonal from its PADDED S_q; ``api_dsl_sm107`` module
doc), asserted host-side on the row probe AND the adapter backstop, and end to
end on Rubin.  The two graph-admission cases of the bring-up
placeholder are kept (Rubin only).  The host-only static and sm_107a SASS pins
of this row's kernel body live in ``test_sdpa_bwd_dsl_sm107.py`` (both bodies,
one machinery); the kernel-vs-kernel targets are the pre-port kernel's
``fp8_*.pt`` dumps under ``frost_dev/results/bwd_d256_sm107/<ref>/`` (located by
the half suite's ``_ref_dump_dir``).
"""

from __future__ import annotations

import collections
import math
import types

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd.api_dsl import ws_align
from frost_test_utils import _SM, cuda_launch_counts, requires_dsl, requires_rubin, requires_sm80, select_engine

from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3

pytestmark = [pytest.mark.L0, requires_dsl]


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # This module probes Rubin rows on non-Rubin hosts too (the analyzer's cc faked to 10.7).  bwd mismatch() now carries the
    # fwd rows' sm_107a DSL gate (AGENTS.md Rule 7), so match the fake device with a fake compiler target -- exactly as the
    # fwd suites do; real Rubin runs use the real build.
    import torch
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


_ENGINE = "sdpa_bwd_sm107_fp8"
_HALF_ENGINE = "sdpa_bwd_sm107"
_FAMILY_NAME = "frost_sdpa_bwd"
_SLOT = 5  # engines/manifest.py: EngineSlot(5, opt_in=True) -> FROST_SDPA_BWD_ID_BASE + 5
_ENGINE_ID = 20_605
_D = 256
_RUBIN_CC = (10, 7)
_SM_RANGE = (107, 119)
_FP8 = cudnn.data_type.FP8_E4M3
_E5M2 = cudnn.data_type.FP8_E5M2
_BF16 = cudnn.data_type.BFLOAT16
_T_E4M3 = torch.float8_e4m3fn
# cuDNN's sdpa_fp8_backward scalar set, in the op's positional order.
_SCALARS = (
    "descale_q",
    "descale_k",
    "descale_v",
    "descale_o",
    "descale_dO",
    "descale_s",
    "descale_dP",
    "scale_s",
    "scale_dQ",
    "scale_dK",
    "scale_dV",
    "scale_dP",
)
_AMAX = ("amax_dQ", "amax_dK", "amax_dV", "amax_dP")

# ---------------------------------------------------------------------------- tolerances: ONE place
# The cuDNN backend fp8 BACKWARD recipe (test/python/sdpa/fp8.py::exec_sdpa_fp8, e4m3 inputs): dequantized gradients
# within atol 0.08 / rtol 0.2 of the fp8-modelled reference, with assert_close_fp8_grad's budget for the rare e4m3
# midpoint flips (a legal flip is rank-1 along one operand row and is PROVED from the reference's own intermediates,
# never waved through by widening these).  The amax outputs are the maxima of the same fp32 pre-quant tensors, so
# they carry the same bound.  amax_dP is fp32 on both sides (no quantization in between): a tight bound, loose only
# for the exp2 spelling -- and far tighter than the factor ~P * attn_scale that separates max|dS| from max|dO V^T|.
_FP8_GRAD_TOL = dict(atol=0.08, rtol=0.2)
_AMAX_DS_TOL = dict(atol=1e-4, rtol=1e-2)
_BF16_ULP_REL = 2.0**-7


def _spec(name=_ENGINE):
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    spec = next((s for s in ENGINE_SPECS if s.name == name), None)
    assert spec is not None, f"{name} is not in cudnn.sdpa.bwd.engines.ENGINE_SPECS (plan s7: rows sdpa_bwd_sm107 / sdpa_bwd_sm107_fp8)"
    return spec


def _bshd(h, s, d):
    """cuDNN declares logical BHSD; the FROST rows want BSHD-physical storage."""
    return (s * h * d, d, h * d, 1)


def _bhsd(h, s, d):
    return (h * s * d, s * d, d, 1)


def _graph_and_ports(
    b=1,
    h=4,
    s=512,
    d=_D,
    *,
    hkv=None,
    skv=None,
    d_v=None,
    causal=True,
    bottom_right=False,
    left_bound=None,
    right_bound=None,
    fp8=_FP8,
    o_dtype=None,
    grad_dtypes=None,
    request_amax=_AMAX,
    padded=False,
    deterministic=False,
    sink=False,
    stride_fn=_bshd,
    scale="default",
):
    """``sdpa_fp8_backward`` with FP8 Q/K/V/O/dO, fp32 Stats, the twelve scalar descales / scales, dQ/dK/dV in
    ``grad_dtypes`` (default: FP8, the contract) and the amax outputs named in ``request_amax`` declared real.  Every
    output carries its dims / strides, so the analyzer reads it host-side without build_operation_graph.  Returns
    ``(graph, {port: tensor})``."""
    hkv = h if hkv is None else hkv
    skv = s if skv is None else skv
    d_v = d if d_v is None else d_v
    o_dtype = fp8 if o_dtype is None else o_dtype
    grad_dtypes = (fp8, fp8, fp8) if grad_dtypes is None else grad_dtypes
    g = cudnn.pygraph(io_data_type=fp8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q_dims, kv_dims, v_dims, o_dims = (b, h, s, d), (b, hkv, skv, d), (b, hkv, skv, d_v), (b, h, s, d_v)
    ts = dict(
        q=g.tensor(dim=q_dims, stride=stride_fn(h, s, d), data_type=fp8, name="q"),
        k=g.tensor(dim=kv_dims, stride=stride_fn(hkv, skv, d), data_type=fp8, name="k"),
        v=g.tensor(dim=v_dims, stride=stride_fn(hkv, skv, d_v), data_type=fp8, name="v"),
        o=g.tensor(dim=o_dims, stride=stride_fn(h, s, d_v), data_type=o_dtype, name="o"),
        dO=g.tensor(dim=o_dims, stride=stride_fn(h, s, d_v), data_type=fp8, name="dO"),
        stats=g.tensor(dim=(b, h, s, 1), stride=(h * s, s, 1, 1), data_type=cudnn.data_type.FLOAT, name="stats"),
    )
    for name in _SCALARS:
        ts[name] = g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name=name)
    kw = dict(use_causal_mask=causal and not bottom_right, use_causal_mask_bottom_right=bottom_right, use_deterministic_algorithm=deterministic)
    if left_bound is not None:
        kw["left_bound"] = left_bound
    if right_bound is not None:
        kw["right_bound"] = right_bound
    if scale is not None:
        kw["attn_scale"] = 1.0 / math.sqrt(d) if scale == "default" else scale
    if padded:
        ts["seq_len_q"] = g.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_len_q")
        ts["seq_len_kv"] = g.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_len_kv")
        kw.update(use_padding_mask=True, seq_len_q=ts["seq_len_q"], seq_len_kv=ts["seq_len_kv"])
    if sink:
        ts["sink"] = g.tensor(dim=(1, h, 1, 1), stride=(h, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
        ts["dsink"] = g.tensor(dim=(1, h, 1, 1), stride=(h, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="dsink")
        kw.update(sink_token=ts["sink"], dSink_token=ts["dsink"])
    outs = g.sdpa_fp8_backward(name="fb", **{k: ts[k] for k in ("q", "k", "v", "o", "dO", "stats", *_SCALARS)}, **kw)
    ts.update(zip(("dQ", "dK", "dV") + _AMAX, outs))
    for name, dims, strides, dt in (
        ("dQ", q_dims, stride_fn(h, s, d), grad_dtypes[0]),
        ("dK", kv_dims, stride_fn(hkv, skv, d), grad_dtypes[1]),
        ("dV", v_dims, stride_fn(hkv, skv, d_v), grad_dtypes[2]),
    ):
        ts[name].set_output(True).set_dim(dims).set_stride(strides).set_data_type(dt)
    for name in _AMAX:
        ts[name].set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
        if name in request_amax:
            ts[name].set_output(True)
    if sink:
        ts["dsink"].set_output(True)
    return g, ts


def _build_graph(*args, **kw):
    """The graph alone (the placeholder's builder; the half row's suite uses it as a foreign-family probe)."""
    return _graph_and_ports(*args, **kw)[0]


# =========================================================================== graph admission (the bring-up placeholder's cases, Rubin)


@requires_rubin
def test_graph_admits_d256_fp8_backward_on_rubin(monkeypatch):
    """Classic eager C++ validation (FROST off, the real device query -> cc 10.7):
    the node no longer raises the ``hidden_dim`` not-supported at d=256."""
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    g = _build_graph()
    g.validate()
    assert g._lowered_graph is not None, "classic path: the C++ graph validated the node"


@requires_rubin
def test_graph_declines_cleanly_until_a_row_serves_it():
    """FROST on (the suite's autouse opt-in): validate() is python-native and the
    backend's verdict is deferred to planning, where the backend has no plan for
    this shape (``override_heuristics_query`` returns ``{-1, {}}`` so heuristics
    runs and finds no config) and no python row claims it -- the typed
    not-supported is what surfaces, a decline callers can act on.  Retires
    itself once the row is registered: its accept tests take over."""
    from cudnn.sdpa.bwd import engines as bwd_engines

    if any(spec.name == _ENGINE for spec in bwd_engines.ENGINE_SPECS):
        pytest.skip(f"{_ENGINE} is registered; its accept tests supersede this placeholder")
    g = _build_graph()
    g.validate()
    assert g._lowered_graph is None, "with a FROST candidate the backend's verdict is deferred to planning"
    g.build_operation_graph()
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g.create_execution_plans([cudnn.heur_mode.A])


# =========================================================================== registration / capabilities (host)


def test_engine_is_registered_and_opt_in():
    from cudnn.engines.engine_ids import FROST_SDPA_BWD_ID_BASE
    from cudnn.engines.manifest import MANIFEST

    _spec()
    fam = next(f for f in MANIFEST if f.name == _FAMILY_NAME)
    assert _ENGINE in fam.slots, f"{_ENGINE} has no manifest slot (never reuse a retired slot; plan s7 says slot {_SLOT})"
    slot = fam.slots[_ENGINE]
    assert slot.slot == _SLOT and slot.opt_in, "new engines stay opt-in until they earn arch coverage + benchmarks"
    assert FROST_SDPA_BWD_ID_BASE + slot.slot == _ENGINE_ID


def test_engine_name_grammar():
    """``engine_name`` grows an ``fp8=`` arm like the forward's (plan s7); until it does the literal name is the contract."""
    import inspect

    from cudnn.sdpa.bwd.engines import engine_name

    if "fp8" in inspect.signature(engine_name).parameters:
        assert engine_name("sm107", fp8=True) == _ENGINE
        assert engine_name("sm107") == _HALF_ENGINE
    assert _spec().name == _ENGINE


def test_capabilities_match_what_is_implemented():
    """The row must not claim anything the adapter would refuse at build, and it must claim the contract it implements.
    The v1 deferrals (plan Q4 / PR-2c) are pinned False: each flips with its accept test here and a tracker line."""
    c = _spec().capabilities
    assert (c.sm_lo, c.sm_hi) == _SM_RANGE
    assert c.d == frozenset({_D}) and not c.d_envelope and not c.dqk_ge_dv
    assert c.is_fp8 and not c.is_mxfp8, "per-tensor FP8 (sdpa_fp8_backward), not block-scale"
    assert c.dtypes == frozenset({_FP8}), "E4M3 only: the body has no E5M2 arm (config_sm107 rejects it)"
    assert _FP8 in c.out_dtypes, "the contract's FP8 gradients (dtype_o is the GRADIENT dtype on the fp8 backward)"
    assert c.out_dtypes <= frozenset({_FP8, _BF16, cudnn.data_type.HALF}), c.out_dtypes
    assert c.amax_dgrad, "amax_dQ / dK / dV / dP are produced in-kernel (the contract), so a graph may request them"
    assert c.causal and c.bottom_right and c.swa and c.gqa
    assert c.bottom_right_s_q_multiple == 1, "bottom-right is claimed at ANY S_q: the fp8 body threads seqlen_q_real (as the f16 body does)"
    assert not c.right_band_widening
    assert c.thd and c.thd_declared_totals, "THD / ragged is served with declared packed totals (test_sdpa_bwd_thd_fp8_sm107.py)"
    assert not c.cu_seq_len, "no BACKWARD node carries cu_seq_len_* (forward-only ports)"
    assert not c.bias and not c.dbias
    assert not c.decode
    assert c.layouts == frozenset({"bshd"})
    assert not c.tile_ms and not c.tile_ns
    for deferred in ("sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred: claim it together with its accept test here and the tracker line"
    assert not c.padded, (
        "padded stays declined on the graph: a padded backward graph carries seq_len_q (the frontend requires both lengths) and no body "
        "threads per-batch Q lengths; the standalone per-batch kv lengths ARE served (the body reads seq_kv_lens[b] under its padded arm: "
        "test_fp8_adapter_admits_per_batch_kv_lengths), the graph form is not"
    )


# =========================================================================== REJECT -- asserted on REAL graphs (host, fake cc 10.7)


def _decline_reason(monkeypatch, engine=_ENGINE, cc=_RUBIN_CC, **kw):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    spec = _spec(engine)
    try:
        g, _ts = _graph_and_ports(**kw)
    except (cudnn.cudnnGraphNotSupportedError, RuntimeError) as e:
        return f"frontend refused the graph: {e}"
    facts = ga.analyze(g)
    if facts is None:
        return "analyzer did not recognise the graph"
    return mismatch(spec.capabilities, facts)


@pytest.mark.parametrize(
    "kw",
    [
        dict(causal=False),
        dict(causal=True),
        dict(causal=True, bottom_right=True),
        dict(causal=True, bottom_right=True, s=512, skv=1024),
        dict(causal=True, bottom_right=True, s=512, skv=1000),
        dict(causal=True, left_bound=256),
        dict(causal=False, hkv=1),
        dict(causal=False, h=8, hkv=2),
        dict(causal=False, s=500, skv=500),
        dict(causal=True, s=500, skv=500),
        dict(causal=False, scale=None),
        dict(causal=False, request_amax=()),
        dict(causal=True, request_amax=("amax_dP",)),
    ],
    ids=[
        "dense",
        "causal",
        "bottom-right",
        "bottom-right-rect",
        "bottom-right-ragged-kv",
        "swa",
        "mqa",
        "gqa-r4",
        "non-tile-S",
        "causal-non-tile-S",
        "default-scale",
        "no-amax",
        "amax-dP-only",
    ],
)
def test_served_graph_passes_the_row_probe(monkeypatch, kw):
    """Sanity for the rejects and the host half of every accept claim -- the amax outputs are served whether requested
    (all four, or any subset) or left virtual."""
    assert _decline_reason(monkeypatch, **kw) is None


@pytest.mark.parametrize("d", [128, 192, 512])
def test_reject_other_head_dims(monkeypatch, d):
    assert _decline_reason(monkeypatch, d=d) is not None


def test_reject_rectangular_head_dims(monkeypatch):
    assert _decline_reason(monkeypatch, d=256, d_v=128) is not None


def test_reject_gqa_ratio_not_integer(monkeypatch):
    assert _decline_reason(monkeypatch, h=6, hkv=4) is not None


def test_reject_e5m2_payloads(monkeypatch):
    """E4M3 only: the pre-port body has no E5M2 arm and config_sm107 raises on it -- the row must decline first."""
    assert _decline_reason(monkeypatch, fp8=_E5M2) is not None


def test_reject_gradient_dtype_mismatch(monkeypatch):
    """dQ / dK / dV share one dtype (``uniform_out_dtype``)."""
    assert _decline_reason(monkeypatch, grad_dtypes=(_FP8, _BF16, _FP8)) is not None


def test_half_gradients_follow_the_out_dtypes_domain(monkeypatch):
    """bf16 gradients on the fp8 graph (what the backend allows on Blackwell) are served iff the row lists BFLOAT16 in
    ``out_dtypes`` -- the pre-quantization output the bitwise A/B against the pre-port kernel uses (config_sm107
    ``dtype_o``)."""
    reason = _decline_reason(monkeypatch, grad_dtypes=(_BF16, _BF16, _BF16))
    if _BF16 in _spec().capabilities.out_dtypes:
        assert reason is None, reason
    else:
        assert reason is not None


def test_reject_half_o_payload(monkeypatch):
    """O is an FP8 PAYLOAD (it carries descale_o); a half O breaks payload uniformity."""
    assert _decline_reason(monkeypatch, o_dtype=_BF16) is not None


def test_reject_right_band_widening(monkeypatch):
    assert _decline_reason(monkeypatch, causal=False, right_bound=16) is not None


def test_reject_sink(monkeypatch):
    assert _decline_reason(monkeypatch, sink=True) is not None


def test_reject_decode_shaped(monkeypatch):
    assert _decline_reason(monkeypatch, s=1, skv=256, causal=False) is not None


def test_reject_non_bshd_layout(monkeypatch):
    assert _decline_reason(monkeypatch, stride_fn=_bhsd) is not None


_BR_RAGGED_SQ_SHAPES = [(500, 1024), (300, 1000), (129, 256)]
_BR_RAGGED_SQ_IDS = ["500x1024", "300x1000", "129x256"]


@pytest.mark.parametrize("sq,skv", _BR_RAGGED_SQ_SHAPES, ids=_BR_RAGGED_SQ_IDS)
def test_reject_bottom_right_with_ragged_s_q(monkeypatch, sq, skv):
    """Bottom-right causal with ``S_q % 128 != 0`` is DECLINED on this row, not served: the fp8 body derives the diagonal
    ``S_kv - S_q`` and the q-tile trim from its PADDED q extent (its ABI has no ``seqlen_q_real``; the f16 body threads
    it), so S_q = 500 would run the diagonal of S_q = 512 -- finite, wrong near the diagonal, no crash.  The decline is
    typed and at eligibility (``mismatch()``), so the graph falls through to a not-supported instead of a wrong answer.
    The same graph on the HALF row is served (its adapter passes the real length).  Inverts when the fp8 ABI grows
    ``seqlen_q_real`` (then ``bottom_right_s_q_multiple`` goes back to 1 and the accept case below takes over)."""
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph

    reason = _decline_reason(monkeypatch, s=sq, skv=skv, causal=True, bottom_right=True)
    if _spec().capabilities.bottom_right_s_q_multiple == 1:
        assert reason is None, reason
    else:
        assert reason is not None and "S_q % 128 == 0" in reason, reason
    # ... while top-left causal at the same ragged S_q, and bottom-right at a ragged S_KV, stay served.
    assert _decline_reason(monkeypatch, s=sq, skv=skv, causal=True) is None
    assert _decline_reason(monkeypatch, s=512, skv=1000, causal=True, bottom_right=True) is None
    # The half row keeps the claim: its body consumes seqlen_q_real.
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    g, _t, _outs = _half_bwd_graph(b=1, hq=2, sq=sq, skv=skv, use_causal_mask_bottom_right=True)
    facts = ga.analyze(g)
    assert facts is not None and facts.bottom_right and facts.s_q == sq
    assert mismatch(_spec(_HALF_ENGINE).capabilities, facts) is None


def test_bottom_right_alignment_claim_mirrors_the_adapter_pad():
    """``Capabilities.bottom_right_s_q_multiple`` MIRRORS the body's ABI (frost-engine-contract s8b': a row field that mirrors
    adapter / body data is pinned per row): the fp8 body threads ``seqlen_q_real`` now, so the fp8 row claims any S_q like the half
    row; the q pad stays the adapter's tile (128) and no longer leaks into the claim."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_Q_PAD

    assert _SM107_Q_PAD == 128
    assert _spec().capabilities.bottom_right_s_q_multiple == 1, "the fp8 body consumes seqlen_q_real: bottom-right at a ragged S_q is served"
    assert _spec(_HALF_ENGINE).capabilities.bottom_right_s_q_multiple == 1


@pytest.mark.parametrize("sq,skv", _BR_RAGGED_SQ_SHAPES, ids=_BR_RAGGED_SQ_IDS)
def test_fp8_adapter_admits_bottom_right_with_ragged_s_q(sq, skv):
    """The adapter's own ``check_support`` ADMITS bottom-right causal at a ragged S_q (the body derives the diagonal and the
    q-tile trim from ``seqlen_q_real``; the former ``S_q % 128 == 0`` backstop is gone with the body fact it mirrored), and the
    aligned twin and the ragged-S_kv twin as before.  Inverted from the decline it used to pin (``test_reject_bottom_right_with_ragged_s_q``
    self-inverts on the row's claim)."""
    from test_sdpa_bwd_dsl_sm107 import _adapter

    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    kw = dict(b=1, hq=2, dt=_T_E4M3, grad_dt=_T_E4M3, is_causal=True, causal_bottom_right=True)
    assert _adapter(SdpaBwdDslSm107Fp8, sq=sq, skv=skv, **kw).check_support(), "bottom-right at a ragged S_q is served by the fp8 adapter"
    assert _adapter(SdpaBwdDslSm107Fp8, sq=512, skv=1024, **kw).check_support()
    assert _adapter(SdpaBwdDslSm107Fp8, sq=512, skv=1000, **kw).check_support()
    # Top-left causal at the ragged S_q is served (the diagonal is 0; the padded q rows read P = 0 through the +inf LSE).
    assert _adapter(SdpaBwdDslSm107Fp8, sq=sq, skv=skv, b=1, hq=2, dt=_T_E4M3, grad_dt=_T_E4M3, is_causal=True).check_support()


def test_padding_mask_follows_the_padded_claim(monkeypatch):
    """A graph padding mask carries ``seq_len_q`` and ``seq_len_kv`` by construction (the frontend requires both); the fp8
    body takes ONE uniform real kv length (``seqlen_kv_real``) and no per-batch Q length, so the row declines the graph form
    (``padded=False``) -- and, unlike the half row, its adapter declines per-batch kv lengths on the standalone surface too
    (``test_fp8_adapter_declines_per_batch_kv_lengths``).  Inverts, rather than being deleted, once ``padded`` flips."""
    reason = _decline_reason(monkeypatch, padded=True)
    if _spec().capabilities.padded:
        assert reason is None, reason
    else:
        assert reason is not None


def test_fp8_adapter_admits_per_batch_kv_lengths():
    """The fp8 body's padded-mask arm reads ``seq_kv_lens[b]`` now (its appended ``seq_kv_lens_tensor`` operand, the f16 body's
    ``_resolve_seqlen_kv``), so the adapter ADMITS a plan built with ``seq_kv_lens_present=True`` like the half row
    (``test_sdpa_bwd_dsl_sm107.py::test_half_adapter_admits_per_batch_kv_lengths``); THD is admitted with declared totals.  Per-batch
    Q lengths stay declined on every row (no body threads them), and THD without its totals is declined for the totals, not for
    THD.  Inverted from the declines it used to pin."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8
    from test_sdpa_bwd_dsl_sm107 import _adapter

    assert _adapter(SdpaBwdDslSm107Fp8, dt=_T_E4M3, grad_dt=_T_E4M3, seq_kv_lens_present=True).check_support(), "per-batch kv lengths are served on the fp8 row"
    with pytest.raises(ValueError, match="seq_q_lens"):
        _adapter(SdpaBwdDslSm107Fp8, dt=_T_E4M3, grad_dt=_T_E4M3, seq_q_lens_present=True).check_support()
    with pytest.raises(ValueError, match="max_total_seq_len"):
        _adapter(SdpaBwdDslSm107Fp8, dt=_T_E4M3, grad_dt=_T_E4M3, thd=True).check_support()
    assert _adapter(SdpaBwdDslSm107Fp8, dt=_T_E4M3, grad_dt=_T_E4M3, thd=True, max_total_seq_len_q=1024, max_total_seq_len_kv=1024).check_support()


def test_deterministic_follows_the_claim(monkeypatch):
    reason = _decline_reason(monkeypatch, deterministic=True)
    if _spec().capabilities.deterministic:
        assert reason is None, reason
    else:
        assert reason is not None


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (12, 0), (9, 0)], ids=["sm100", "sm103", "sm120", "sm90"])
def test_reject_other_arch_lines(monkeypatch, cc):
    reason = _decline_reason(monkeypatch, cc=cc)
    assert reason is not None and f"SM{_SM_RANGE[0]}-{_SM_RANGE[1]}" in reason, reason


def test_reject_foreign_quantization_families(monkeypatch):
    """The fp8 row declines the half ``sdpa_backward`` and the block-scale ``sdpa_mxfp8_backward`` (the family gate)."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import mismatch
    from test_sdpa_bwd_dsl_sm107 import _half_bwd_graph
    from test_sdpa_bwd_mxfp8_sm100 import _build_graph as _mxfp8_graph

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    caps = _spec().capabilities
    g, _t, _outs = _half_bwd_graph()
    facts = ga.analyze(g)
    assert facts is not None and not facts.is_fp8 and not facts.is_mxfp8
    assert "serves only" in (mismatch(caps, facts) or "")
    g, _t, _outs = _mxfp8_graph(1, 2, 2, 256, 256, scale=1.0 / math.sqrt(_D))
    facts = ga.analyze(g)
    assert facts is not None and facts.is_mxfp8
    assert "serves only" in (mismatch(caps, facts) or "")


# =========================================================================== ACCEPT -- the fp8 backward oracle (Rubin)


def _sc(val):
    return torch.tensor([[[[float(val)]]]], dtype=torch.float32, device="cuda")


def _ds_dtype_code() -> int:
    """The dS workspace dtype the NEXT adapter will read (``api_dsl_sm107.FP8_DS_DTYPE``, monkeypatched by ``ds_knob``)."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    return sm107.FP8_DS_DTYPE


_DS_KNOBS = [pytest.param(DTYPE_E4M3, id="e4m3-ds"), pytest.param(DTYPE_BF16, id="bf16-ds")]
# The oracle of a case depends on (shape, mask, seed, attn_scale) and, for the backward, on the dS rounding (ds_knob); the gradient dtype
# only CASTS the fp32 gradients (compute_ref_backward: ``torch_otype`` casts dQ / dK / dV, nothing else).  _run_fp8 computes each oracle
# once per process and keeps the fp32 results here; every parametrization of the same case reuses the SAME tensors, so the reference
# values are bitwise the unshared form's.  The (b, h, s_q, s_kv) intermediates are NOT kept (``ref_bwd`` re-derives a selection on demand).
# Bounded by bytes, oldest case first: the fp32 gradients of a 32-head S=2K cell are ~200 MB, and the memo must not be what fills a
# 16 GiB part; the parametrizations of one case are adjacent in collection order, so a small budget keeps nearly every hit.
# Under the disk cache's VERIFY mode the memo stands aside (_oracle_memo), so a repeated case is verified like a first one.
_ORACLE_MEMO: "collections.OrderedDict[tuple, tuple]" = collections.OrderedDict()
_ORACLE_MEMO_BUDGET_BYTES = 2 * 2**30


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _oracle_memo(key, compute):
    """``compute()`` once per key per process; a hit is the SAME object the first call produced (bitwise by construction).

    Under the disk cache's VERIFY mode (``CUDNN_TEST_REF_CACHE`` names a directory and ``CUDNN_TEST_REF_CACHE_VERIFY`` is on) the
    memo stands aside: every call runs ``compute`` -- ``cached_reference``, which recomputes the oracle and compares it bitwise with
    the stored entry -- so a repeated case, or a case memoized before the cache was switched on, is verified like a first one."""
    from sdpa import ref_cache

    if ref_cache.cache_dir() is not None and ref_cache.verify():
        return compute()
    hit = _ORACLE_MEMO.get(key)
    if hit is not None:
        _ORACLE_MEMO.move_to_end(key)
        return hit[0]
    value = compute()
    _ORACLE_MEMO[key] = (value, _tensor_bytes(value))
    while len(_ORACLE_MEMO) > 1 and sum(nbytes for _, nbytes in _ORACLE_MEMO.values()) > _ORACLE_MEMO_BUDGET_BYTES:
        _ORACLE_MEMO.popitem(last=False)
    return value


def test_oracle_memo_stands_aside_under_disk_cache_verify(monkeypatch, tmp_path):
    """Under ``CUDNN_TEST_REF_CACHE_VERIFY`` a REPEATED case still reaches the oracle: the process-local memo would otherwise answer
    before ``cached_reference`` could recompute and compare, and a case memoized before the cache was switched on would never be
    verified.  With VERIFY off the memo serves the repeat without a compute, as before."""
    from sdpa import ref_cache

    monkeypatch.setenv("CUDNN_TEST_REF_CACHE", str(tmp_path))
    monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", "1")
    key = ("memo-verify-pin", str(tmp_path))
    calls = []

    def oracle_of(value):
        def oracle():  # ONE source for every value: the disk key hashes the callable's source, so a drift is a drift, not another entry
            calls.append(1)
            return (torch.full((2, 3), value, device="cuda"), 0.25)

        return oracle

    def through_cache(value):
        return lambda: ref_cache.cached_reference("memo_pin", dict(key=str(key)), oracle_of(value))

    try:
        first = _oracle_memo(key, through_cache(1.5))  # a disk miss: computed and stored
        before = ref_cache.stats()["verified"]
        second = _oracle_memo(key, through_cache(1.5))  # the repeat: recomputed and compared bitwise, not answered from the memo
        assert len(calls) == 2 and ref_cache.stats()["verified"] == before + 1
        assert torch.equal(first[0], second[0]) and first[1] == second[1]
        with pytest.raises(AssertionError, match="differs from a fresh compute"):
            _oracle_memo(key, through_cache(1.5 + 2**-20))  # a drifted oracle fails the repeated case too
        monkeypatch.setenv("CUDNN_TEST_REF_CACHE_VERIFY", "0")
        n = len(calls)
        a = _oracle_memo(key, through_cache(1.5))  # VERIFY off: a disk hit (no oracle call), memoized ...
        b = _oracle_memo(key, through_cache(1.5))  # ... and the memo answers the repeat
        assert a is b and len(calls) == n
    finally:
        _ORACLE_MEMO.pop(key, None)


# One accept cell per member of the row's ``out_dtypes`` (engine contract): the fp8 contract's e4m3 gradients and the two
# half dtypes the row also serves -- there the gradients land UNSCALED (scale_dQ / dK / dV = 1, the oracle casts them) while
# dS keeps its e4m3 rounding at scale_dP on the shipped chain.  The dense, causal and GQA-causal cells run all three; every
# other accept cell keeps the e4m3 default.
_GRAD_DTYPES = [pytest.param(_T_E4M3, id="e4m3-grads"), pytest.param(torch.bfloat16, id="bf16-grads"), pytest.param(torch.float16, id="fp16-grads")]


@pytest.fixture(params=_DS_KNOBS)
def ds_knob(request, monkeypatch):
    """Runs an accept case on BOTH dS workspace dtypes: the shipped e4m3 chain (the fp8 GEMM arm quantizes dQ / MHA dK in its
    epilogue; the reference rounds dS to e4m3 with scale_dP like the backend) and the bf16 twin (bf16 GEMMs over exact e4m3
    -> bf16 upcasts, three fold + quantize passes; the reference keeps dS in fp32).  The twin is what proves the GEMM arm and
    not the kernel moved when a case fails on one knob only."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    monkeypatch.setattr(sm107, "FP8_DS_DTYPE", request.param)
    return request.param


def _view_bhsd(x_bshd):
    """A [B, S, H, D] storage tensor as the [B, H, S, D] view the graph's (dims, strides) describe."""
    return x_bshd.permute(0, 2, 1, 3)


class _Fp8Run:
    def __init__(self, outs, amax, refs, ref_amax, ref_bwd, operands, flip_units, keys, descales, grad_dtype):
        self.outs, self.amax, self.refs, self.ref_amax = outs, amax, refs, ref_amax
        self.ref_bwd, self.operands, self.flip_units, self.keys, self.descales, self.grad_dtype = ref_bwd, operands, flip_units, keys, descales, grad_dtype

    def check(self):
        from sdpa.fp8 import assert_close_fp8_grad

        for name in ("dQ", "dK", "dV"):
            got = self.outs[0][name].float() * self.descales[name]  # [B, S, H, D] storage, like the oracle's outputs
            want = self.refs[name].float() * self.descales[name]
            assert torch.isfinite(got).all(), f"{name}: non-finite output"
            assert_close_fp8_grad(
                got,
                want,
                _FP8_GRAD_TOL["atol"],
                _FP8_GRAD_TOL["rtol"],
                tag=name,
                keys=self.keys[name],
                operand=self.operands[name],
                flip_unit=self.flip_units[name],
                intermediates=lambda selection: self.ref_bwd(return_intermediates=selection)[8],
                fp8_dtype=_T_E4M3,
                out_dtype=self.grad_dtype,
            )
        for name in ("dQ", "dK", "dV"):
            if name not in self.amax[0]:
                continue  # not requested on this graph
            a, r = self.amax[0][name].item(), self.ref_amax[name]
            assert math.isfinite(a), f"amax_{name} was not written (NaN poison survived)"
            assert abs(a - r) <= _FP8_GRAD_TOL["atol"] + _FP8_GRAD_TOL["rtol"] * r, f"amax_{name} {a:.5f} vs the oracle's pre-quant max {r:.5f}"
        if "dP" in self.amax[0]:
            a, r = self.amax[0]["dP"].item(), self.ref_amax["dP"]
            assert math.isfinite(a), "amax_dP was not written"
            assert abs(a - r) <= _AMAX_DS_TOL["atol"] + _AMAX_DS_TOL["rtol"] * r, (
                f"amax_dP {a:.6f} vs max|dS| {r:.6f}: the contract reduces the fp32 dS = P (dP - D) attn_scale right before its scale_dP cast "
                f"(sdpa_fp8_bwd.h), not dO V^T (max {self.ref_amax['dP_raw']:.4f})"
            )
        return self


def _run_fp8(
    b=1,
    hq=2,
    hkv=None,
    sq=512,
    skv=512,
    causal=False,
    bottom_right=False,
    left=None,
    seed=0,
    runs=1,
    poison=None,
    grad_dtype=_T_E4M3,
    request_amax=_AMAX,
    ws_poison=None,
    attn_scale=None,
):
    """Quantize unit-normal operands per tensor, run the oracle (forward for O / Stats, then backward with the oracle's
    Stats injected -- the kernel recomputes P from the forward's exact LSE), build the graph with the contract's twelve
    scalars (delayed scaling: scale_dP / dQ / dK / dV from the oracle's amaxes, the recipe the backend suite feeds),
    PIN the engine, execute ``runs`` times and hand back every run's outputs plus what to compare them with.
    ``ws_poison`` (a BYTE, 0xFF = NaN in e4m3 / bf16 / fp32) pre-fills the WORKSPACE before every run, so a stage-3 GEMM
    reading a dS tile the main kernel skipped lands NaN in an output (``check`` asserts finite first).  ``left`` without
    ``causal`` is a sliding window alone.  ``attn_scale`` (None = 1/sqrt(d)) declares an EXPLICIT scale on the graph and the
    oracle alike -- 0.0 included, a valid scale the adapter must preserve."""
    from sdpa.fp8_ref import compute_ref, compute_ref_backward
    from sdpa.helpers import get_fp8_descale_factor, get_fp8_scale_factor
    from sdpa.ref_cache import cached_reference  # a node-local DISK cache of oracle outputs; off unless CUDNN_TEST_REF_CACHE names a dir

    hkv = hq if hkv is None else hkv
    dev = "cuda"
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return torch.randn(bb, s, h, _D, generator=gen).to(dev)  # [B, S, H, D] storage

    def quant(x):
        scale = get_fp8_scale_factor(x.abs().max().item(), _T_E4M3)
        return (x * scale).to(_T_E4M3), 1.0 / scale

    q32, do32, k32, v32 = draw(b, sq, hq), draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    q8, q_ds = quant(q32)
    k8, k_ds = quant(k32)
    v8, v_ds = quant(v32)
    do8, do_ds = quant(do32)
    scale = 1.0 / math.sqrt(_D) if attn_scale is None else float(attn_scale)
    s_scale = get_fp8_scale_factor(1.0, _T_E4M3)  # P <= 1
    s_descale = 1.0 / s_scale
    right = 0 if causal else None
    align = (cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT) if causal else None
    # Everything the oracles depend on that this module's content does not pin: the operands are drawn from `seed` on the CPU
    # generator and quantized per tensor by the two helpers above, `_D` is the row's head dim, `_T_E4M3` the operand dtype.  The
    # RECIPE itself -- this function: the draw order, the quantization call sites, `ref_bwd` -- reaches the disk key as the content
    # of this WHOLE module (sdpa/ref_cache.py hashes the file the oracle callable is defined in and every test-tree file on the call
    # stack at the call), so an edit here is a miss; `quant` scales through sdpa/helpers.py, a listed reference source.
    fwd_key = ("fwd", b, hq, hkv, sq, skv, causal, bottom_right, left, seed, scale)
    _KEY_NAMES = ("b", "hq", "hkv", "sq", "skv", "causal", "bottom_right", "left", "seed", "scale")
    _disk_key_common = dict(d=_D, in_dtype=str(_T_E4M3))

    def _fwd_oracle():
        return compute_ref(q8, k8, v8, scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, _T_E4M3, left_bound=left, right_bound=right, diag_align=align)

    o8, stats, o_amax = _oracle_memo(
        fwd_key,
        lambda: cached_reference("fp8_row_fwd", dict(zip(_KEY_NAMES, fwd_key[1:]), recipe="compute_ref e4m3 P, e4m3 O", **_disk_key_common), _fwd_oracle),
    )
    # compute_ref hands O back as a [B, S, H, D]-SHAPED VIEW over B,H,S,D-contiguous memory (its strides are BHSD).  The
    # graph declares O with BSHD strides, so without this the kernel reads O scrambled -> delta wrong on ~all rows (max
    # 5.1 at B1 H2 S512) -> dS off by attn_scale * P * ddelta -> ~4 dQ elements past atol and amax_dP 5.6 % high, while
    # dV (no delta) stays bitwise.  q8 / k8 / v8 / do8 come from `draw(...).to(...)` and are contiguous already.
    o8 = o8.contiguous()
    o_ds = get_fp8_descale_factor(o_amax, _T_E4M3)

    ds_knob = _ds_dtype_code()

    def ref_bwd(return_intermediates=False, quantize_ds=None, quantize_grads=True):
        # The reference composes the SAME dS rounding as the chain under test (sdpa-invariants s8): the e4m3 chain's dQ / dK
        # consume dS_q = e4m3(dS * scale_dP) -> quantize_ds=True (the backend recipe); the bf16 twin holds dS in bf16 -> the
        # reference keeps it in fp32.  P stays quantized on both (the dV BMM2 consumes e4m3 P on both sides).
        if quantize_ds is None:
            quantize_ds = ds_knob == DTYPE_E4M3
        return compute_ref_backward(
            q8, k8, v8, o8, do8, scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, o_ds, do_ds, grad_dtype,
            left_bound=left, right_bound=right, diag_align=align, stats=stats, return_intermediates=return_intermediates,
            quantize_ds=quantize_ds, quantize_grads=quantize_grads,
        )  # fmt: skip

    bwd_key = ("bwd", b, hq, hkv, sq, skv, causal, bottom_right, left, seed, scale, ds_knob == DTYPE_E4M3)

    def _bwd_oracle():
        dq32, dk32, dv32, _dsink, dp_amax, dq_amax, dk_amax, dv_amax, inter = ref_bwd(return_intermediates=True, quantize_grads=False)
        return (dq32, dk32, dv32, dp_amax, dq_amax, dk_amax, dv_amax, inter["ds_scaled"].abs().max().item())

    dq32, dk32, dv32, dp_amax, dq_amax, dk_amax, dv_amax, ds_scaled_max = _oracle_memo(
        bwd_key,
        lambda: cached_reference(
            "fp8_row_bwd", dict(zip(_KEY_NAMES + ("e4m3_ds",), bwd_key[1:]), recipe="compute_ref_backward fp32 grads", **_disk_key_common), _bwd_oracle
        ),
    )
    dp_scale = get_fp8_scale_factor(dp_amax, _T_E4M3)
    ds_amax = ds_scaled_max / dp_scale  # the fp32 dS the contract's amax_dP reduces
    grad_scale = {n: get_fp8_scale_factor(a, grad_dtype) for n, a in (("dQ", dq_amax), ("dK", dk_amax), ("dV", dv_amax))}
    # the reference's own quantize_grads=True cast, applied here per gradient dtype: (grad * get_fp8_scale_factor(amax, torch_otype)).to(torch_otype)
    dq_ref, dk_ref, dv_ref = ((g * grad_scale[n]).to(grad_dtype) for n, g in (("dQ", dq32), ("dK", dk32), ("dV", dv32)))
    scalars = dict(
        descale_q=q_ds,
        descale_k=k_ds,
        descale_v=v_ds,
        descale_o=o_ds,
        descale_dO=do_ds,
        descale_s=s_descale,
        descale_dP=1.0 / dp_scale,
        scale_s=s_scale,
        scale_dQ=grad_scale["dQ"],
        scale_dK=grad_scale["dK"],
        scale_dV=grad_scale["dV"],
        scale_dP=dp_scale,
    )
    cudnn_grad = {_T_E4M3: _FP8, torch.bfloat16: _BF16, torch.float16: cudnn.data_type.HALF}[grad_dtype]
    g, ts = _graph_and_ports(
        b,
        hq,
        sq,
        _D,
        hkv=hkv,
        skv=skv,
        causal=causal,
        bottom_right=bottom_right,
        left_bound=left,
        grad_dtypes=(cudnn_grad,) * 3,
        scale=scale,
        request_amax=request_amax,
    )
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    outs_t = {n: torch.empty(sh, device=dev, dtype=grad_dtype) for n, sh in (("dQ", (b, sq, hq, _D)), ("dK", (b, skv, hkv, _D)), ("dV", (b, skv, hkv, _D)))}
    amax_t = {n: torch.full((1, 1, 1, 1), float("nan"), device=dev, dtype=torch.float32) for n in ("dQ", "dK", "dV", "dP") if f"amax_{n}" in request_amax}
    pack = {
        ts["q"]: _view_bhsd(q8),
        ts["k"]: _view_bhsd(k8),
        ts["v"]: _view_bhsd(v8),
        ts["o"]: _view_bhsd(o8),
        ts["dO"]: _view_bhsd(do8),
        ts["stats"]: stats.contiguous(),
    }
    pack.update({ts[n]: _sc(v) for n, v in scalars.items()})
    pack.update({ts[n]: _view_bhsd(outs_t[n]) for n in ("dQ", "dK", "dV")})
    pack.update({ts[f"amax_{n}"]: amax_t[n] for n in amax_t})
    outs, amax = [], []
    for _ in range(runs):
        for x in outs_t.values():
            x.fill_(float("nan") if poison is None else poison)
        for x in amax_t.values():
            x.fill_(float("nan"))
        if ws_poison is not None:
            ws.fill_(ws_poison)
        g.execute(pack, ws)
        torch.cuda.synchronize()
        outs.append({n: x.clone() for n, x in outs_t.items()})
        amax.append({n: x.clone() for n, x in amax_t.items()})
    deq = {n: (x8.float() * ds) for n, x8, ds in (("q", q8, q_ds), ("k", k8, k_ds), ("dO", do8, do_ds))}
    run = _Fp8Run(
        outs,
        amax,
        refs=dict(dQ=dq_ref, dK=dk_ref, dV=dv_ref),
        ref_amax=dict(dQ=dq_amax, dK=dk_amax, dV=dv_amax, dP=ds_amax, dP_raw=dp_amax),
        ref_bwd=ref_bwd,
        operands=dict(dQ=deq["k"], dK=deq["q"], dV=deq["dO"]),
        flip_units=dict(dQ=1.0 / dp_scale, dK=1.0 / dp_scale, dV=s_descale),
        keys=dict(dQ=skv, dK=sq, dV=sq),
        descales={n: 1.0 / s for n, s in grad_scale.items()},
        grad_dtype=grad_dtype,
    )
    # The live graph state, for the prepared-launch pins (re-execute, replay, standalone twin).
    run.graph, run.ts, run.pack, run.workspace, run.outs_t, run.amax_t, run.scale = g, ts, pack, ws, outs_t, amax_t, scale
    run.inputs = dict(q=q8, k=k8, v=v8, o=o8, dO=do8, stats=stats.contiguous())
    run.ds_knob, run.dp_scale = ds_knob, dp_scale
    return run


@requires_rubin
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES)
def test_dense(ds_knob, grad_dtype):
    _run_fp8(grad_dtype=grad_dtype).check()


@requires_rubin
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES)
def test_causal(ds_knob, grad_dtype):
    _run_fp8(causal=True, grad_dtype=grad_dtype).check()


@requires_rubin
def test_causal_bottom_right_rectangular(ds_knob):
    """The ALIGNED bottom-right accept (S_q % 128 == 0, S_kv % 256 == 0): the shape the row claims."""
    _run_fp8(sq=512, skv=1024, causal=True, bottom_right=True).check()


@requires_rubin
def test_causal_bottom_right_ragged_kv(ds_knob):
    """Bottom-right with a ragged S_KV is served: the kernel's kv term of the diagonal is the runtime REAL length
    (``seqlen_kv_real``) and the padded kv rows ride the padded-mask arm.  Only a ragged S_Q is declined
    (``test_reject_bottom_right_with_ragged_s_q``)."""
    _run_fp8(sq=512, skv=1000, causal=True, bottom_right=True).check()


@requires_rubin
def test_sliding_window(ds_knob):
    _run_fp8(causal=True, left=256).check()


@requires_rubin
@pytest.mark.parametrize("hq,hkv", [(4, 2), (8, 1), (16, 1)], ids=["r2", "r8-mqa", "r16-mqa"])
def test_gqa(ds_knob, hq, hkv):
    _run_fp8(hq=hq, hkv=hkv, sq=256, skv=256).check()


@requires_rubin
@pytest.mark.parametrize("grad_dtype", _GRAD_DTYPES)
def test_gqa_causal(ds_knob, grad_dtype):
    _run_fp8(hq=8, hkv=2, sq=512, skv=512, causal=True, grad_dtype=grad_dtype).check()


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (257, 129)])
def test_non_tile_multiple_seqlens(ds_knob, sq, skv):
    """The adapter pads S_q to the q tile and S_kv to the kv block (plan Q9); the FP8 forward on Rubin declines these,
    the backward serves them."""
    _run_fp8(sq=sq, skv=skv).check()


@requires_rubin
@pytest.mark.parametrize("n_q_tiles", [1, 2, 3, 4])
def test_q_tiles_per_kv_block(ds_knob, n_q_tiles):
    """q tiles per kv block = 1, 2, STAGES_Q (3 on this body), STAGES_Q + 1: the three 3-deep Q / dO rings and the 2-stage
    P ring wrap at different depths (sdpa-invariants s6)."""
    _run_fp8(hq=1, sq=128 * n_q_tiles, skv=256).check()


@requires_rubin
def test_kv_blocks_with_several_tiles_in_flight(ds_knob):
    _run_fp8(b=2, hq=2, sq=256, skv=768).check()


@requires_rubin
def test_causal_tail_kv_block_writes_zeros_not_residue(ds_knob):
    """Top-left causal with S_kv > S_q: the unattended kv blocks run the forced fully-masked tile and must store exact
    zeros (dK / dV rows >= S_q), never the poison and never NaN residue."""
    sq, skv = 256, 768
    run = _run_fp8(b=2, hq=2, sq=sq, skv=skv, causal=True, poison=float("nan")).check()
    for name in ("dK", "dV"):
        tail = run.outs[0][name][:, sq:].float()
        assert torch.isfinite(tail).all() and (tail == 0).all(), f"{name}: unattended kv rows must be EXACTLY zero"


# The band arms the fp8 row serves, on the shapes where the K-trim's roundings and clamps bite (the bf16 suite's
# ``_MASK_POISON_CASES`` minus the ragged-S_q bottom-right shapes this row declines).  ``left`` = the graph's band bound.
_MASK_POISON_CASES = {
    "causal": dict(sq=512, skv=512, causal=True),
    "causal-tl-rect-q": dict(sq=768, skv=512, causal=True),
    "causal-tl-tail-kv": dict(sq=256, skv=768, causal=True),
    "bottom-right": dict(sq=512, skv=1024, causal=True, bottom_right=True),
    "bottom-right-ragged-kv": dict(sq=512, skv=1000, causal=True, bottom_right=True),
    "swa640": dict(sq=1024, skv=1024, causal=True, left=640),
    "swa200": dict(sq=1024, skv=1024, causal=True, left=200),
    "swa200-bottom-right": dict(sq=512, skv=1024, causal=True, bottom_right=True, left=200),
    "swa200-no-causal": dict(sq=1024, skv=1024, causal=False, left=200),
    "swa200-tl-rect-kv": dict(sq=512, skv=1024, causal=True, left=200),  # top-left window with S_kv > S_q: blocks past S_q run the forced tile
    # The fp8 graph ADMITS a window with S_q > S_kv (the half graph refuses it: "max_s_q <= max_s_kv"), so the fill's one remaining
    # geometry is reachable here: (512, 256, W=127) writes every q pair (no fill), (1024, 256, W=127) leaves q pairs no block writes
    # and `_stage3_needs_zero_fill` turns the fill on -- both ran green on fractal GPU 3, 2026-09-29.
    "swa128-tl-rect-q-within": dict(sq=512, skv=256, causal=True, left=128),
    "swa128-tl-rect-q-past": dict(sq=1024, skv=256, causal=True, left=128),
    "swa200-gqa": dict(hq=8, hkv=2, sq=1024, skv=1024, causal=True, left=200),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_MASK_POISON_CASES), ids=list(_MASK_POISON_CASES))
def test_masked_stage3_reads_only_what_stage2_wrote(ds_knob, case):
    """The two-sided K-trim of the stage-3 GEMMs (the e4m3 K64 arm on the shipped knob, the bf16 renderings on the twin)
    reads ONLY dS tiles the main kernel wrote: the WHOLE workspace is poisoned with 0xFF (NaN in e4m3, bf16 and fp32)
    before the execute, so a GEMM reaching a skipped tile lands NaN in dQ / dK -- and the per-execute zero-fill is gone
    under every served mask but one (``api_dsl_sm107._stage3_needs_zero_fill``: only ``swa128-tl-rect-q-past`` runs with
    it).  The bf16 suite's twin carries the ragged-S_q bottom-right shapes this row declines."""
    _run_fp8(poison=float("nan"), ws_poison=0xFF, **_MASK_POISON_CASES[case]).check()


@requires_rubin
@pytest.mark.parametrize(
    "case",
    [dict(sq=512, skv=512, causal=True), dict(sq=1024, skv=1024, causal=True, left=200), dict(sq=512, skv=1024, causal=True, bottom_right=True, left=200)],
    ids=["causal", "swa200", "swa200-bottom-right"],
)
def test_stage3_k_trim_is_bitwise_the_untrimmed_rendering(monkeypatch, ds_knob, case):
    """The fp8 twin of the bf16 suite's pin: the two-sided K-trim is numerically INERT on the e4m3 K64 arm too --
    rendering both GEMMs untrimmed (``STAGE3_CAUSAL_TRIM = False``: every k tile read over a zero-filled workspace) gives
    the SAME BITS for dQ / dK / dV and the same amax values.  The trimmed run gets the poisoned workspace."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    trimmed = _run_fp8(b=1, hq=4, hkv=2, poison=float("nan"), ws_poison=0xFF, **case).check()
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    untrimmed = _run_fp8(b=1, hq=4, hkv=2, poison=float("nan"), **case).check()
    for name in ("dQ", "dK", "dV"):
        x, y = trimmed.outs[0][name], untrimmed.outs[0][name]
        n_diff = (x.view(torch.int8) != y.view(torch.int8)).sum().item()
        assert n_diff == 0, f"{name}: trimmed vs untrimmed stage 3 differ in {n_diff} elements"
    for name in ("dQ", "dK", "dV", "dP"):
        assert trimmed.amax[0][name].item() == untrimmed.amax[0][name].item(), f"amax_{name}: trimmed vs untrimmed differ"


@requires_rubin
@pytest.mark.parametrize("s", [1024, 2048])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("hq,hkv", [(8, 2), (32, 2)], ids=["gqa8-2", "gqa32-2"])
def test_stage3_single_launch_dq_is_bitwise_the_per_member_launches(monkeypatch, ds_knob, hq, hkv, causal, s):
    """The fp8 twin of the bf16 suite's pin: under GQA the dQ GEMM is ONE launch per head chunk (its rendering indexes B = K by
    ``h // group``, ``MatmulTemplateParams.b_head_group = group``) on the e4m3 K64 arm -- whose QUANT epilogue folds ``amax_dQ``
    with per-TENSOR scalars (descale_dP, descale_k, scale_dQ), so a single launch over every Q head reads the same scalars the
    per-member launches did -- and on the bf16-dS twin's plain rendering alike.  Same head pairing, same k-tile walk per output
    tile: dQ / dK / dV the SAME BITS and the four amax values equal.  ``DQ_SINGLE_LAUNCH = False`` is the per-member twin."""
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    assert sm107.DQ_SINGLE_LAUNCH, "one dQ launch per chunk is what ships; the pin flips it OFF for the twin"
    single = _run_fp8(b=1, hq=hq, hkv=hkv, sq=s, skv=s, causal=causal, poison=float("nan")).check()
    monkeypatch.setattr(sm107, "DQ_SINGLE_LAUNCH", False)
    members = _run_fp8(b=1, hq=hq, hkv=hkv, sq=s, skv=s, causal=causal, poison=float("nan")).check()
    for name in ("dQ", "dK", "dV"):
        x, y = single.outs[0][name], members.outs[0][name]
        n_diff = (x.view(torch.int8) != y.view(torch.int8)).sum().item()
        assert n_diff == 0, f"{name}: the single dQ launch vs the per-member launches differ in {n_diff} of {x.numel()} elements"
    for name in ("dQ", "dK", "dV", "dP"):
        assert single.amax[0][name].item() == members.amax[0][name].item(), f"amax_{name}: single vs per-member launches differ"


@requires_rubin
def test_two_launches_are_bitwise_and_race_free(ds_knob):
    """Launch 2 vs 1 = the two-launch race trick, launch 3 vs 2 = the determinism to show before claiming it; the amax
    outputs must be bitwise stable too (an atomicMax over a fixed set of fp32 values)."""
    run = _run_fp8(b=2, hq=4, hkv=2, sq=512, skv=768, causal=True, runs=3, poison=float("nan")).check()
    for which, i, j in (("launch 2 vs 1 (race)", 1, 0), ("launch 3 vs 2 (determinism)", 2, 1)):
        for name in ("dQ", "dK", "dV"):
            x, y = run.outs[i][name], run.outs[j][name]
            n_diff = (x.view(torch.int8) != y.view(torch.int8)).sum().item()
            assert n_diff == 0, f"{name} {which}: {n_diff} elements differ"
        for name in ("dQ", "dK", "dV", "dP"):
            assert run.amax[i][name].item() == run.amax[j][name].item(), f"amax_{name} {which} differs"


@requires_rubin
def test_explicit_zero_attn_scale_is_preserved(ds_knob):
    """The fp8 row inherits the half row's initializer, so the same ``or == 0.0`` folded an explicit attn_scale = 0.0 into
    1/sqrt(d) (max|dQ| 0.81, max|dK| 0.94, dV off by 0.62, amax_dP 0.25 at this shape with unit scalars -- Codex review on
    #1212).  Preserved, P is uniform: dQ = dK = 0 EXACTLY on both dS knobs (e4m3: ``dS_q = e4m3(0 * scale_dP)`` = 0 into the fp8
    GEMM arm; bf16: a zero workspace), amax_dQ = amax_dK = amax_dP = 0, dV = sum(dO) / S_kv against the oracle at the suite's
    tolerance (``check``, which also asserts every output finite)."""
    run = _run_fp8(b=1, hq=1, sq=128, skv=256, attn_scale=0.0).check()
    for name in ("dQ", "dK"):
        got = run.outs[0][name].float()
        assert (
            got == 0
        ).all(), f"{name}: attn_scale = 0.0 must give an EXACT zero on the dS knob {run.ds_knob} chain, got max |{name}| = {got.abs().max().item():.4f}"
    for name in ("dQ", "dK", "dP"):
        assert run.amax[0][name].item() == 0.0, f"amax_{name} must be exactly 0 under a uniform P (dS == 0)"


def _run_on_knob(monkeypatch, knob, **kw):
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107

    with monkeypatch.context() as patch:
        patch.setattr(sm107, "FP8_DS_DTYPE", knob)
        return _run_fp8(**kw)


@requires_rubin
@pytest.mark.parametrize("case", [dict(), dict(causal=True), dict(hq=8, hkv=2, causal=True)], ids=["dense", "causal", "gqa-causal"])
def test_fp8_ds_workspace_dtypes_agree(monkeypatch, case):
    """The two chains side by side on one input.  dV does not depend on dS and amax_dP is folded BEFORE the scale and the
    cast, so at MHA both are BITWISE across the knob (a difference is a kernel change, not a GEMM-arm change).  Under GQA the
    e4m3 chain folds dV from fp32 per-Q-head partials (one rounding) while the bf16-dS twin keeps bf16 partials (its 96 KiB dS ring
    leaves no SMEM for the fp32 dV staging), so their dV codes differ where the twin's extra partial rounding lands on an e4m3
    midpoint: what is pinned there is that the e4m3 chain's dV is never FARTHER from the once-rounded oracle than the twin's, that
    at most 0.1 % of its codes are off the oracle's (0 measured: the single rounding IS the oracle's), and that amax_dV differs
    by at most ``group`` half-ulps of bf16 relative (the twin's partial roundings).  No per-element bound between the chains is
    claimed: a bf16 partial's rounding is relative to the partial, so under cancellation it exceeds any step of the small sum.  dQ / dK each pass the
    recipe against their OWN oracle (``check()``, the fixture-parametrized accept cases); against EACH OTHER they differ by
    exactly the e4m3 rounding of dS (3 mantissa bits, 2^-4 relative per element, an e4m3 midpoint flip per value) -- a
    per-element flip budget cannot hold between them and is not claimed.  What is pinned: the dequantized dQ / dK of the two
    chains disagree (outside the fp8 recipe, atol 0.08 / rtol 0.2) on NO MORE elements than the two fp64 ORACLES do under the
    same two dS roundings (``refs``: the quantized-dS oracle behind the e4m3 run, the fp32-dS oracle behind the bf16 run --
    sdpa-invariants s8, the reference composes the same conditions), and no single element is farther apart than the oracle
    pair's farthest (or 0.5).  The fraction is a property of the CASE, not of the kernels: MEASURED on c05 2026-09-28 (kernel
    pair / oracle pair) dQ 0.25 / 0.25 % on every case; dK 0.25 / 0.25 % dense, <= 1 % causal (MHA), but 5.519 / 5.521 % on
    gqa-causal (four q-heads' partials summed into one dK, more near-cancelling elements) -- a fixed 1 % pin failed there while
    each chain was 0 % outside its own oracle.  The numbers are printed so a drift shows up in the log before it crosses the pin."""
    e4m3 = _run_on_knob(monkeypatch, DTYPE_E4M3, **case).check()
    bf16 = _run_on_knob(monkeypatch, DTYPE_BF16, **case).check()
    assert e4m3.ds_knob == DTYPE_E4M3 and bf16.ds_knob == DTYPE_BF16
    assert e4m3.amax[0]["dP"].item() == bf16.amax[0]["dP"].item(), "amax_dP is the pre-quant fp32 max on both chains"
    group = case.get("hq", 2) // case.get("hkv", case.get("hq", 2))
    if group == 1:
        assert torch.equal(e4m3.outs[0]["dV"].view(torch.int8), bf16.outs[0]["dV"].view(torch.int8)), "dV must be bitwise across the dS workspace dtype"
        assert e4m3.amax[0]["dV"].item() == bf16.amax[0]["dV"].item()
    else:
        a8, b8, r8 = e4m3.outs[0]["dV"], bf16.outs[0]["dV"], e4m3.refs["dV"]
        assert torch.equal(r8.view(torch.int8), bf16.refs["dV"].view(torch.int8)), "dV does not depend on dS: one oracle dV behind both runs"
        n_ab, n_ar, n_br = (int((x.view(torch.int8) != y.view(torch.int8)).sum()) for x, y in ((a8, b8), (a8, r8), (b8, r8)))
        print(
            f"\ndV e4m3-dS (fp32 partials) vs bf16-dS (bf16 partials) chain ({case}): {n_ab} of {a8.numel()} codes differ; vs the oracle: "
            f"{n_ar} (fp32 partials) / {n_br} (bf16 partials); amax_dV {e4m3.amax[0]['dV'].item():.6g} / {bf16.amax[0]['dV'].item():.6g}"
        )
        assert n_ar <= n_br, "the fp32 per-Q-head partials must leave the e4m3 chain's dV no farther from the once-rounded oracle than the twin's bf16 partials"
        # The fp32-partial chain's dV IS the oracle's rounding up to fp32-vs-fp64 accumulation residue: 0 of 262144 codes off at this cell
        # (MEASURED 2026-10-06; the twin 9905 = 3.8 %).  A per-element bound on the chain-vs-chain difference is NOT claimed: a bf16
        # partial's rounding error is relative to the PARTIAL, so under cancellation it exceeds any step of the small sum.
        assert n_ar <= a8.numel() // 1000, f"{n_ar} of {a8.numel()} dV codes off the once-rounded oracle with fp32 partials (> 0.1 %)"
        am_a, am_b = e4m3.amax[0]["dV"].item(), bf16.amax[0]["dV"].item()
        assert abs(am_a - am_b) <= group * 2.0**-9 * max(
            am_a, am_b
        ), "amax_dV: the twin's bf16 partial roundings move the fold's max by at most group half-ulps of bf16"
    for name in ("dQ", "dK"):
        a = e4m3.outs[0][name].float() * e4m3.descales[name]
        b = bf16.outs[0][name].float() * bf16.descales[name]
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        diff = (a - b).abs()
        outside = diff > _FP8_GRAD_TOL["atol"] + _FP8_GRAD_TOL["rtol"] * b.abs()
        frac = outside.float().mean().item()
        # The same metric on the two fp64 oracles (each run's ``refs`` is the oracle with ITS dS rounding, dequantized by ITS
        # descale): what the e4m3 rounding of dS alone moves on this case.
        ref_q = e4m3.refs[name].float() * e4m3.descales[name]
        ref_f = bf16.refs[name].float() * bf16.descales[name]
        diff_ref = (ref_q - ref_f).abs()
        frac_ref = (diff_ref > _FP8_GRAD_TOL["atol"] + _FP8_GRAD_TOL["rtol"] * ref_f.abs()).float().mean().item()
        print(
            f"\n{name} e4m3-dS vs bf16-dS chain ({case}): {int(outside.sum())} of {a.numel()} outside the recipe ({100 * frac:.3f} %), max |diff| {diff.max().item():.4f}"
            f" -- the oracle pair under the same two dS roundings: {100 * frac_ref:.3f} %, max |diff| {diff_ref.max().item():.4f}"
        )
        assert frac <= frac_ref + 1e-3, (
            f"{name}: the two chains disagree on {100 * frac:.3f} % of elements, the two oracles on {100 * frac_ref:.3f} % -- "
            "more than the e4m3 dS rounding explains (each chain still passed the recipe against its own oracle above)"
        )
        assert diff.max().item() <= max(
            0.5, diff_ref.max().item() + _FP8_GRAD_TOL["atol"]
        ), f"{name}: max |diff| {diff.max().item():.3f} between the chains exceeds the oracle pair's {diff_ref.max().item():.3f} (+ atol) / 0.5"


@requires_rubin
def test_unserved_e5m2_graph_declines_as_not_supported():
    """Error TYPE regression: the C++ node admits d256 fp8 backward on Rubin (PR-3a), this row declines E5M2 and the
    backend has no plan -- ``cudnnGraphNotSupportedError``, never the bare RuntimeError a pinned backend config that
    fails to finalize used to fold into (every SDPA harness skips on the typed error and FAILS on anything else)."""
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g = _build_graph(fp8=_E5M2)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()


@requires_rubin
@pytest.mark.parametrize("sq,skv", _BR_RAGGED_SQ_SHAPES[:1], ids=_BR_RAGGED_SQ_IDS[:1])
def test_unserved_bottom_right_ragged_s_q_graph_declines_as_not_supported(sq, skv):
    """End to end on the device: the row's ``mismatch()`` drops it, no other engine serves a d256 fp8 backward, and the
    typed ``cudnnGraphNotSupportedError`` surfaces (never a silently shifted diagonal, never a bare RuntimeError)."""
    if _spec().capabilities.bottom_right_s_q_multiple == 1:
        pytest.skip("the fp8 row serves a ragged S_q under bottom-right now; test_causal_bottom_right_* cover it")
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        g = _build_graph(b=1, h=2, s=sq, skv=skv, causal=True, bottom_right=True)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        g.build_plans()


@requires_rubin
def test_workspace_is_build_time_honest():
    g, _ts = _graph_and_ports(1, 2, 256, _D, causal=False)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == g.get_workspace_size() > 0


# --------------------------------------------------------------------------- the prepared launch (Rubin): scalars + requested amax are roles of ONE artifact


def _prepared_fp8_case(**kw):
    kw.setdefault("hq", 4)
    kw.setdefault("hkv", 2)
    kw.setdefault("causal", True)
    case = _run_fp8(**kw)
    assert case.graph._compiled_plans[case.graph._plan_index]._prepared is not None, "the fp8 row must lower through the prepared-launch contract"
    return case


def _check_prepared_fp8(case):
    """The live outputs are BITWISE the first run's (the chain is deterministic: two-launch pin) and every requested amax is."""
    for name, t in case.outs_t.items():
        assert torch.equal(t, case.outs[0][name]), f"{name}: a re-execute / replay of the prepared launch changed the bits"
    for name, t in case.amax_t.items():
        assert torch.equal(t, case.amax[0][name]), f"amax_{name}: a re-execute / replay changed the bits"


@requires_rubin
def test_prepared_fp8_execute_has_no_tensor_plumbing(monkeypatch):
    """Warm execute rebuilds no tensor views, allocates nothing, compiles nothing, never synchronizes: the e4m3 -> bf16
    upcast, the amax resets and the fold + quantize passes are launches of the artifact (the pin that proves the torch
    path is gone on the fp8 row).  Bitwise the first run afterwards."""
    import cutlass.cute as cute
    import cutlass.cute.runtime as runtime
    from cudnn.sdpa.fwd.api_dsl import WorkspaceCarver

    case = _prepared_fp8_case(sq=500, skv=500)  # padded: every staging kernel runs
    torch.cuda.synchronize()
    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared fp8 backward rebuilt tensor arguments, allocated or compiled on execute")

    with monkeypatch.context() as patch:
        patch.setattr(cute, "compile", forbidden)
        patch.setattr(runtime, "from_dlpack", forbidden)
        patch.setattr(WorkspaceCarver, "__init__", forbidden)
        for name in ("view", "reshape", "permute", "contiguous", "as_strided", "copy_", "zero_", "fill_"):
            patch.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full"):
            patch.setattr(torch, name, forbidden)
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(3):
                case.graph.execute(case.pack, case.workspace)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    _check_prepared_fp8(case)


@requires_rubin
@pytest.mark.parametrize("request_amax", [(), ("amax_dP",), ("amax_dQ", "amax_dK", "amax_dV")], ids=["none", "dP", "dQ-dK-dV"])
def test_prepared_fp8_binds_only_the_requested_amax(request_amax):
    """An amax the graph left virtual is None-specialized out of the artifact (``Operand`` None, no pointer bound): the
    gradients are BITWISE the all-requested plan's and every requested amax equals its value there."""
    full = _prepared_fp8_case()
    case = _prepared_fp8_case(request_amax=request_amax)
    spec = case.graph._compiled_plans[case.graph._plan_index]._prepared.spec
    bound = {role for role, op in zip(spec.roles, spec.operands) if role.startswith("amax_") and op is not None}
    assert bound == set(request_amax), f"the spec binds {sorted(bound)}; the graph requested {sorted(request_amax)}"
    for name in ("dQ", "dK", "dV"):
        assert torch.equal(case.outs[0][name], full.outs[0][name]), f"{name} depends on which amax outputs are requested"
    for name, t in case.amax[0].items():
        assert torch.equal(t, full.amax[0][name]), f"amax_{name} differs from the all-requested plan's"
    case.check()


@requires_rubin
def test_prepared_fp8_rebind_stream_and_replay():
    """The fp8 plan follows the handle's stream and captures into a CUDA graph; a replay over poisoned outputs / amax /
    workspace reproduces the first run's bits."""
    case = _prepared_fp8_case()
    stream, ambient = torch.cuda.Stream(), torch.cuda.Stream()
    handle = cudnn.create_handle()
    cudnn.set_stream(handle, stream.cuda_stream)
    capture = torch.cuda.CUDAGraph()

    def poison():
        for t in list(case.outs_t.values()) + list(case.amax_t.values()):
            t.fill_(float("nan"))
        case.workspace.fill_(0xAD)

    try:
        poison()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(ambient):
            case.graph.execute(case.pack, case.workspace, handle=handle)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        _check_prepared_fp8(case)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=stream):
            case.graph.execute(case.pack, case.workspace, handle=handle)
        poison()
        capture.replay()
        torch.cuda.synchronize()
        _check_prepared_fp8(case)
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@requires_rubin
def test_prepared_fp8_standalone_twin_matches_the_graph_and_refuses_off_plan_operands():
    """``SdpaBwdDslSm107Fp8.execute`` binds the SAME artifact from torch tensors: bitwise the graph's outputs and amax;
    an amax tensor given for an unrequested output (or missing for a requested one) and an operand off the plan's
    layout are refused before any stage launches."""
    from dataclasses import replace
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    case = _prepared_fp8_case(request_amax=("amax_dQ", "amax_dK", "amax_dV", "amax_dP"))
    inp = case.inputs
    samples = dict(q=_view_bhsd(inp["q"]), k=_view_bhsd(inp["k"]), v=_view_bhsd(inp["v"]), o=_view_bhsd(inp["o"]), do=_view_bhsd(inp["dO"]), stats=inp["stats"])
    grads = {n: torch.empty_like(case.outs_t[n]) for n in ("dQ", "dK", "dV")}
    samples.update(dq=_view_bhsd(grads["dQ"]), dk=_view_bhsd(grads["dK"]), dv=_view_bhsd(grads["dV"]))
    api = SdpaBwdDslSm107Fp8(**{"sample_" + n: t for n, t in samples.items()}, is_causal=True, scale_softmax=case.scale, amax_requested=_AMAX)
    api.check_support()
    api.compile()
    scalars = {n: case.pack[case.ts[n]] for n in _SCALARS}
    amax = {n: torch.full_like(case.amax_t[k], float("nan")) for k, n in (("dQ", "amax_dQ"), ("dK", "amax_dK"), ("dV", "amax_dV"), ("dP", "amax_dP"))}
    ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    args = {n + "_tensor": t for n, t in samples.items()}
    api.execute(**args, workspace=ws, **scalars, **amax)
    torch.cuda.synchronize()
    for n in ("dQ", "dK", "dV"):
        assert torch.equal(grads[n], case.outs[0][n]), f"standalone {n} differs from the graph's"
    for k, n in (("dQ", "amax_dQ"), ("dK", "amax_dK"), ("dV", "amax_dV"), ("dP", "amax_dP")):
        assert torch.equal(amax[n], case.amax[0][k]), f"standalone {n} differs from the graph's"
    launches = []
    api._prepared = replace(api._prepared, fn=lambda *a: launches.append(a))
    with pytest.raises(ValueError, match="amax_dP"):
        api.execute(**args, workspace=ws, **scalars, **{**amax, "amax_dP": None})
    bad = dict(args)
    bad["q_tensor"] = samples["q"].contiguous()
    with pytest.raises(ValueError, match="runtime geometry"):
        api.execute(**bad, workspace=ws, **scalars, **amax)
    assert not launches
    with pytest.raises(ValueError, match="unknown amax"):
        SdpaBwdDslSm107Fp8(**{"sample_" + n: t for n, t in samples.items()}, is_causal=True, scale_softmax=case.scale, amax_requested=("amax_o",))


@requires_rubin
def test_prepared_fp8_artifact_reloads_in_fresh_process(tmp_path):
    from prepared_bwd_cache_utils import check_backward_artifact_reload

    check_backward_artifact_reload("sm107", "fp8", "float8_e4m3fn", tmp_path)


# --------------------------------------------------------------------------- vs the pre-port fp8 kernel (Rubin; dumps under frost_dev/results)

_FP8_REF_STEMS = ["fp8_b1h8s1024_dense", "fp8_b1h8kv2s2048_causal", "fp8_b2h4s768x1280_dense"]
# stem -> (B, H_q, H_kv, S_q, S_kv, causal): the dumps' geometry, so a cell runs the SAME shape when its dump is absent.
_FP8_REF_GEOMETRY = {
    "fp8_b1h8s1024_dense": (1, 8, 8, 1024, 1024, False),
    "fp8_b1h8kv2s2048_causal": (1, 8, 2, 2048, 2048, True),
    "fp8_b2h4s768x1280_dense": (2, 4, 4, 768, 1280, False),
}


def _ref_inputs(stem):
    """The pre-port kernel's dump when present (its ``dv_kern`` / ``dv_red`` / ``dv`` / ``ref_dq`` / ``ref_dk`` carry the
    kernel-vs-kernel pins); otherwise the same INPUT keys synthesized at the stem's geometry -- quantized unit-normal Q / K /
    V / dO (the accept cells' recipe), O payload and LSE from the fp8 forward oracle at the dumps' ``scale_s = descale_s = 1``
    -- so a missing dump drops the kernel-vs-kernel pins and never the fp64-oracle comparison (a cell that skips covers
    nothing)."""
    from sdpa.fp8_ref import compute_ref
    from sdpa.helpers import get_fp8_descale_factor, get_fp8_scale_factor
    from test_sdpa_bwd_dsl_sm107 import _ref_dump_dir

    path = _ref_dump_dir() / f"{stem}.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=False)
    b, hq, hkv, sq, skv, causal = _FP8_REF_GEOMETRY[stem]
    gen = torch.Generator(device="cpu").manual_seed(0)

    def quant(x):
        scale = get_fp8_scale_factor(x.abs().max().item(), _T_E4M3)
        return (x * scale).to(_T_E4M3), 1.0 / scale

    q8, q_ds = quant(torch.randn(b, sq, hq, _D, generator=gen))  # [B, S, H, D] storage, like the dumps
    k8, k_ds = quant(torch.randn(b, skv, hkv, _D, generator=gen))
    v8, v_ds = quant(torch.randn(b, skv, hkv, _D, generator=gen))
    do8, do_ds = quant(torch.randn(b, sq, hq, _D, generator=gen))
    attn = 1.0 / math.sqrt(_D)
    o8, stats, o_amax = compute_ref(
        q8.cuda(), k8.cuda(), v8.cuda(), attn, q_ds, k_ds, v_ds, 1.0, 1.0, _T_E4M3, _T_E4M3,
        right_bound=0 if causal else None, diag_align=cudnn.diagonal_alignment.TOP_LEFT if causal else None,
    )  # fmt: skip
    meta = dict(
        B=b, Hq=hq, Hkv=hkv, Sq=sq, Skv=skv, attn_scale_in=attn, causal=causal, mask_flags=2 if causal else 0,
        storage_dtype="torch.float8_e4m3fn", dscale_Q=q_ds, dscale_K=k_ds, dscale_V=v_ds, dscale_dO=do_ds,
    )  # fmt: skip
    # compute_ref hands O back as a [B, S, H, D]-shaped view over B,H,S,D memory; the dumps store BSHD-contiguous codes.
    return dict(
        meta=meta, q=q8, k=k8, v=v8, do=do8, o_storage=o8.contiguous().cpu(), o_dscale=get_fp8_descale_factor(o_amax, _T_E4M3), lse=stats.squeeze(-1).cpu()
    )


def _fp64_operands(ref, dev):
    """The reference inputs as fp64 on ``dev``: the BSHD codes ``q, k, v, do, o8``, the GQA-expanded ``kx, vx`` and the fp64 P
    the kernel recomputes from the injected LSE (top-left causal when the meta says so -- the dumps are square)."""
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    attn = float(m["attn_scale_in"])
    q, k, v, do, o8 = (ref[n].to(dev).double() for n in ("q", "k", "v", "do", "o_storage"))  # BSHD codes
    grp = hq // hkv
    kx, vx = k.repeat_interleave(grp, dim=2), v.repeat_interleave(grp, dim=2)
    s_mat = torch.einsum("bqhd,bkhd->bhqk", q, kx) * (float(m["dscale_Q"]) * float(m["dscale_K"]) * attn)
    if bool(m["causal"]):
        assert sq == skv, "the top-left recompute below assumes the square causal dumps"
        s_mat = s_mat.masked_fill(torch.ones(sq, skv, device=dev, dtype=torch.bool).triu(1), float("-inf"))
    p_mat = torch.exp(s_mat - ref["lse"].to(dev).double().unsqueeze(-1))
    return m, (b, hq, hkv, sq, skv, grp), q, k, v, do, o8, kx, vx, p_mat


def _fp64_dv_with_e4m3_p(ref, dev="cuda"):
    """The fp64 dV oracle for the same inputs: the e4m3-ROUNDED P at the dumps' ``scale_s = 1`` (the P the kernel's BMM2
    consumes) times ``dO * dscale_dO``, reduced per KV head.  ``[B, H_kv, S_kv, D]`` real units on ``dev``."""
    m, (b, hq, hkv, sq, skv, grp), q, k, v, do, o8, kx, vx, p_mat = _fp64_operands(ref, dev)
    p_q = p_mat.to(torch.float8_e4m3fn).double()
    return torch.einsum("bhqk,bqhd->bhkd", p_q, do * float(m["dscale_dO"])).view(b, hkv, grp, skv, -1).sum(2)


def _fp64_dq_dk_with_payload_delta(ref, dev="cuda", dp_scale=None):
    """The fp64 dQ / dK oracle for the dump's inputs with ``delta = rowsum(dO * O_payload) * o_dscale * dscale_dO`` -- the
    delta the graph's chain computes from the e4m3 O it is handed (cuDNN ``sdpa_fp8_backward``: O is an fp8 payload); with
    ``dp_scale`` the dS is rounded to e4m3 at that scale first (the e4m3-dS chain's recipe).  Returns ``(dQ, dK, max |dS|)``.  The
    dump's own ``ref_dq`` / ``ref_dk`` were computed with the UNQUANTIZED fp64 O's delta (its README says so: "lse / delta
    must be injected, not recomputed"), which a graph cannot do.  At a near-one-hot causal row (q = 0 attends kv = 0 only)
    dP == delta exactly for the fp64 O, so dS is nothing but the O-rounding term and the two oracles differ by
    attn_scale * ddelta * |K|: on the causal dump 26 dQ / 11 dK elements past atol 0.08, max 0.16 / 0.23, all in q or kv rows
    0..5, the dump's element exactly 0 where the payload oracle is 0.15 (sdpa-invariants s8).  Top-left causal only (the dumps
    are square).  Returns ``(dQ [B, H_q, S_q, D], dK [B, H_kv, S_kv, D])`` fp64 real units on ``dev``."""
    m, (b, hq, hkv, sq, skv, grp), q, k, v, do, o8, kx, vx, p_mat = _fp64_operands(ref, dev)
    attn = float(m["attn_scale_in"])
    dp = torch.einsum("bqhd,bkhd->bhqk", do, vx) * (float(m["dscale_dO"]) * float(m["dscale_V"]))
    delta = (o8 * do).sum(-1).permute(0, 2, 1) * (float(ref["o_dscale"]) * float(m["dscale_dO"]))  # [B, H_q, S_q]
    ds = attn * p_mat * (dp - delta.unsqueeze(-1))
    ds_amax = ds.abs().max().item()
    if dp_scale is not None:
        ds = (ds * dp_scale).to(torch.float8_e4m3fn).double() / dp_scale
    dq = torch.einsum("bhqk,bkhd->bhqd", ds, kx * float(m["dscale_K"]))
    dk = torch.einsum("bhqk,bqhd->bhkd", ds, q * float(m["dscale_Q"])).view(b, hkv, grp, skv, -1).sum(2)
    return dq, dk, ds_amax


@requires_rubin
@pytest.mark.parametrize("stem", _FP8_REF_STEMS)
def test_dv_matches_the_reference_kernel_and_dq_dk_the_oracle(ds_knob, stem):
    """The pre-port fp8 kernel cast P UNSCALED to e4m3 and applied host-folded descales; the contract port reads descale
    tensors and scales P by ``scale_s``.  With ``scale_s = descale_s = 1`` and the dump's descales the math coincides,
    so: with bf16 gradients (when the row lists BFLOAT16 in ``out_dtypes`` -- the pre-quantization output config_sm107
    keeps for exactly this A/B) dV is BITWISE the dump's ``dv_kern`` (MHA) / within one bf16 rounding of its folded
    ``dv_red`` (GQA); with FP8 gradients only, the dequantized dV is within the fp8 recipe of the dump's real-unit dV.
    dQ / dK ride the bf16 dS workspace + the ported GEMMs (the pre-port chain lost 22-29 % of dS to an unscaled e4m3
    workspace) and are held to the fp8 recipe against the fp64 oracle recomputed with the delta the graph's chain can
    compute -- from the e4m3 O payload (``_fp64_dq_dk_with_payload_delta``) -- not against the pre-port chain and not
    against the dump's ``ref_dq`` / ``ref_dk`` (unquantized-O delta; the count past the recipe vs those is printed).  On the
    e4m3-dS knob the oracle rounds dS to e4m3 at the delayed ``scale_dP`` the graph is fed (from the oracle's own dS amax), and
    the comparison carries ``assert_close_fp8_grad``'s row budget for the dS midpoint flips the fp32 kernel and the fp64 oracle
    round differently (the dumps' own dQ / dK, produced by an UNSCALED e4m3 dS, are no oracle for dS-dependent outputs).

    WITHOUT a dump (``_ref_inputs``) the cell runs the stem's geometry on synthesized inputs instead of skipping: the
    kernel-vs-kernel dV pins have no target and are left out, dV is held to the fp64 oracle that rounds P as the kernel does
    (``_fp64_dv_with_e4m3_p``, under the recipe's bound and the P-midpoint-flip budget), dQ / dK and amax_dV exactly as above."""
    from sdpa.fp8 import assert_close_fp8_grad
    from sdpa.helpers import get_fp8_scale_factor
    from test_sdpa_bwd_dsl_sm107 import _BF16_ULP_REL as _ULP

    ref = _ref_inputs(stem)
    has_dump = "dv_kern" in ref  # the pre-port kernel's outputs are present: the kernel-vs-kernel pins below run too
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.float8_e4m3fn" and int(m["mask_flags"]) in (0, 2)
    dev = "cuda"
    bf16_grads = _BF16 in _spec().capabilities.out_dtypes
    grad_dt, cudnn_grad = (torch.bfloat16, _BF16) if bf16_grads else (_T_E4M3, _FP8)
    _dq0, _dk0, ds_amax = _fp64_dq_dk_with_payload_delta(ref, dev)
    dv_oracle = _fp64_dv_with_e4m3_p(ref, dev)  # [B, H_kv, S_kv, D] real units
    real_amax = {"dQ": _dq0.abs().max().item(), "dK": _dk0.abs().max().item(), "dV": dv_oracle.abs().max().item()}  # the oracle's pre-quant maxima
    grad_scale = {
        n: (1.0 if bf16_grads else get_fp8_scale_factor(ref[key].abs().max().item() if has_dump else real_amax[n], _T_E4M3))
        for n, key in (("dQ", "dq"), ("dK", "dk"), ("dV", "dv"))
    }
    dp_scale = get_fp8_scale_factor(ds_amax, _T_E4M3) if ds_knob == DTYPE_E4M3 else 1.0  # the e4m3 dS needs the delayed scale; the bf16 twin ignores it
    scalars = dict(
        descale_q=m["dscale_Q"],
        descale_k=m["dscale_K"],
        descale_v=m["dscale_V"],
        descale_o=float(ref["o_dscale"]),
        descale_dO=m["dscale_dO"],
        descale_s=1.0,
        descale_dP=1.0 / dp_scale,
        scale_s=1.0,
        scale_dQ=grad_scale["dQ"],
        scale_dK=grad_scale["dK"],
        scale_dV=grad_scale["dV"],
        scale_dP=dp_scale,
    )
    g, ts = _graph_and_ports(b, hq, sq, _D, hkv=hkv, skv=skv, causal=bool(m["causal"]), grad_dtypes=(cudnn_grad,) * 3, scale=float(m["attn_scale_in"]))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    outs = {
        n: torch.full(sh, float("nan"), device=dev, dtype=grad_dt) for n, sh in (("dQ", (b, sq, hq, _D)), ("dK", (b, skv, hkv, _D)), ("dV", (b, skv, hkv, _D)))
    }
    amax = {n: torch.full((1, 1, 1, 1), float("nan"), device=dev, dtype=torch.float32) for n in ("dQ", "dK", "dV", "dP")}
    pack = {ts[n]: _view_bhsd(ref[k].to(dev)) for n, k in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o_storage"), ("dO", "do"))}
    pack[ts["stats"]] = ref["lse"].to(dev).unsqueeze(-1).contiguous()
    pack.update({ts[n]: _sc(v) for n, v in scalars.items()})
    pack.update({ts[n]: _view_bhsd(outs[n]) for n in outs})
    pack.update({ts[f"amax_{n}"]: amax[n] for n in amax})
    g.execute(pack, ws)
    torch.cuda.synchronize()
    dv = outs["dV"].cpu()  # [B, S_kv, H_kv, D] storage
    if not has_dump:
        # No dump: dV against the fp64 oracle with the kernel's own P rounding (e4m3 at scale_s = 1) under the recipe's bound,
        # plus the P-midpoint-flip budget (fp32 kernel vs fp64 oracle: one e4m3 step of P times |dO| in dV row j; descale_s = 1).
        assert_close_fp8_grad(
            outs["dV"].float() / grad_scale["dV"],  # [B, S_kv, H_kv, D] storage, real units
            dv_oracle.float().permute(0, 2, 1, 3).contiguous(),
            _FP8_GRAD_TOL["atol"],
            _FP8_GRAD_TOL["rtol"],
            tag="dV",
            keys=sq,
            operand=ref["do"].to(dev).float() * float(m["dscale_dO"]),
            flip_unit=1.0,
            fp8_dtype=_T_E4M3,
            out_dtype=grad_dt,
        )
    elif bf16_grads and hq == hkv:
        want = ref["dv_kern"]
        n_diff = (dv.view(torch.int16) != want.view(torch.int16)).sum().item()
        assert (
            n_diff == 0
        ), f"dV is NOT bitwise the pre-port kernel: {n_diff} of {want.numel()} differ, max|diff|={(dv.float() - want.float()).abs().max().item():.3e}"
    elif bf16_grads:
        torch.testing.assert_close(
            dv.float(), ref["dv_red"].float(), rtol=_ULP, atol=1e-6, msg=lambda s: f"folded dV vs the pre-port head-reduce (one bf16 rounding allowed): {s}"
        )
    else:
        got = dv.float().permute(0, 2, 1, 3) / grad_scale["dV"]  # -> [B, H_kv, S_kv, D] real units
        torch.testing.assert_close(got, ref["dv"].float(), **_FP8_GRAD_TOL, msg=lambda s: f"dequantized dV vs the pre-port kernel's real-unit dV: {s}")
    dq_pay, dk_pay, _amax = _fp64_dq_dk_with_payload_delta(ref, dev, dp_scale=dp_scale if ds_knob == DTYPE_E4M3 else None)
    deq = {"dQ": ref["k"].to(dev).float() * float(m["dscale_K"]), "dK": ref["q"].to(dev).float() * float(m["dscale_Q"])}  # BSHD operands
    for name, key, want in (("dQ", "ref_dq", dq_pay), ("dK", "ref_dk", dk_pay)):
        got = outs[name].float().permute(0, 2, 1, 3) / grad_scale[name]  # -> BHSD real units, on the device
        if has_dump:
            n_dump = int(((got - ref[key].to(dev).float()).abs() > _FP8_GRAD_TOL["atol"] + _FP8_GRAD_TOL["rtol"] * ref[key].to(dev).float().abs()).sum())
            print(
                f"\n{stem}: {name} vs the dump's unquantized-O-delta oracle: {n_dump} of {got.numel()} outside the recipe (REPORTED; the payload-delta oracle is asserted)"
            )
        if ds_knob == DTYPE_E4M3:
            # BSHD storage on both sides; the dS midpoint flips between the fp32 kernel and the fp64 oracle ride the row budget.
            assert_close_fp8_grad(
                got.permute(0, 2, 1, 3).contiguous(),
                want.float().permute(0, 2, 1, 3).contiguous(),
                _FP8_GRAD_TOL["atol"],
                _FP8_GRAD_TOL["rtol"],
                tag=name,
                keys=skv if name == "dQ" else sq,
                operand=deq[name],
                flip_unit=1.0 / dp_scale,
                fp8_dtype=_T_E4M3,
                out_dtype=grad_dt,
            )
        else:
            torch.testing.assert_close(got, want.float(), **_FP8_GRAD_TOL, msg=lambda s, n=name: f"{n} vs the fp64 oracle with the e4m3-O-payload delta: {s}")
    a = amax["dV"].item()
    want_amax = ref["dv"].abs().max().item() if has_dump else real_amax["dV"]
    assert (
        math.isfinite(a) and abs(a - want_amax) <= 2 * _ULP * a + 1e-6
    ), "amax_dV is the fp32 pre-quant max (the dump's dV is that value in bf16; the fp64 oracle's without a dump)"


@requires_rubin
@pytest.mark.parametrize("stem", _FP8_REF_STEMS)
def test_kernel_level_dv_and_ds_workspace_vs_the_reference_kernel(stem):
    """KERNEL-level, through the body's documented ``## Launch ABI`` (``compile(b, qh, kh, sq, skv, has_amax=False)`` and
    the positional call with the seven fp32 [1] scale tensors), template-loaded with ``dtype_o=DTYPE_BF16`` -- the
    pre-quantization bf16 dV config_sm107 keeps for exactly this A/B -- and the dump's ``lse`` / ``delta`` INJECTED,
    ``scale_s = descale_s = scale_dv = 1`` (the pre-port body cast P unscaled and applied descale_dO in-kernel).  The
    math coincides up to the ORDER of the fp32 descale multiplications (host-folded then, in-kernel tensors now -- plan
    s4.16), so dV is held to one bf16 rounding and the exact-bitwise count is REPORTED; the dS workspace changed dtype by
    design (bf16 for the bf16 GEMMs vs the pre-port unscaled e4m3, plan Q2(b)) and is held to bf16 storage against the
    dump's fp64 ``ref_ds`` (math orientation, transposed here).  The body is loaded with ``dtype_ds=DTYPE_BF16`` (the twin;
    the shipped e4m3 dS is pinned by ``test_kernel_level_e4m3_ds_workspace_is_the_scaled_quantized_ds``)."""
    import cuda.bindings.driver as cuda_driver

    from test_sdpa_bwd_dsl_sm107 import _BF16_ULP_REL as _ULP, _load_kernel, _load_ref_dump

    ref = _load_ref_dump(stem)
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.float8_e4m3fn" and int(m["mask_flags"]) in (0, 2) and sq % 128 == 0 and skv % 256 == 0
    mod = _load_kernel("fp8", dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_BF16, dtype_ds=DTYPE_BF16, window_right=0 if bool(m["causal"]) else None)
    assert mod.CFG.DTYPE_DS == DTYPE_BF16 and not mod.DS_IS_FP8
    fn = mod.compile(b, hq, hkv, sq, skv, has_amax=False)
    dev = "cuda"
    q, do, k, v = (ref[n].to(dev).contiguous() for n in ("q", "do", "k", "v"))  # e4m3 BSHD storage
    dv = torch.full((b, skv, hq, _D), float("nan"), device=dev, dtype=torch.bfloat16)
    ds = torch.zeros((b, hq, skv, sq), device=dev, dtype=torch.bfloat16)
    lse, delta = ref["lse"].to(dev).contiguous(), ref["delta"].to(dev).contiguous()

    def s1(val):
        return torch.tensor([float(val)], dtype=torch.float32, device=dev)

    scale = float(m["attn_scale_in"])
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q, do, k, v, dv, ds, lse, delta,
        s1(m["dscale_Q"]), s1(m["dscale_K"]), s1(m["dscale_V"]), s1(m["dscale_dO"]), s1(1.0), s1(1.0), s1(1.0), s1(1.0),
        None, None,
        (b, hq, hkv, sq, skv, hq),
        scale, scale * math.log2(math.e), 0, skv,
        stream=stream,
    )  # fmt: skip
    torch.cuda.synchronize()
    dv, ds = dv.cpu(), ds.cpu()
    want = ref["dv_kern"].contiguous()
    assert torch.isfinite(dv.float()).all(), "dv: non-finite output (poison survived or NaN residue)"
    n_diff = (dv.view(torch.int16) != want.view(torch.int16)).sum().item()
    print(f"\n{stem}: dv vs the pre-port kernel: {n_diff} of {want.numel()} elements differ (max|diff|={(dv.float() - want.float()).abs().max().item():.3e})")
    torch.testing.assert_close(
        dv.float(), want.float(), rtol=_ULP, atol=1e-6, msg=lambda s: f"dv vs the pre-port kernel (one bf16 rounding allowed for the descale order): {s}"
    )
    ref_ds_kv_major = ref["ref_ds"].float().transpose(-1, -2).contiguous()  # [B, H_q, S_kv, S_q]
    torch.testing.assert_close(
        ds.float(), ref_ds_kv_major, rtol=2.0**-6, atol=_ULP * ref_ds_kv_major.abs().max().item(), msg=lambda s: f"bf16 dS workspace vs the dump's fp64 dS: {s}"
    )


@requires_rubin
@pytest.mark.parametrize("stem", _FP8_REF_STEMS)
def test_kernel_level_e4m3_ds_workspace_is_the_scaled_quantized_ds(stem):
    """KERNEL-level, the shipped ``dtype_ds = E4M3`` body through its ``## Launch ABI`` with the dump's ``lse`` / ``delta``
    INJECTED: the workspace must hold ``e4m3(dS * scale_dP)`` -- every code within one e4m3 step of the code the dump's fp64
    dS rounds to (the fp32 kernel value and the fp64 oracle straddle a midpoint on a handful of cells), dequantized within
    the e4m3 relative step of the fp64 dS (plus the subnormal step at the scale), the skipped causal tiles exact zeros.  This
    is what pins the half-row ``store_swizzled`` arithmetic of the two warpgroups inside one 128-B row and the s128b TMA
    store box of the e4m3 ring: a wrong column offset or swizzle would scramble whole 16-B chunks, not round them."""
    import cuda.bindings.driver as cuda_driver

    from sdpa.helpers import get_fp8_scale_factor
    from test_sdpa_bwd_dsl_sm107 import _load_kernel, _load_ref_dump

    ref = _load_ref_dump(stem)
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.float8_e4m3fn" and int(m["mask_flags"]) in (0, 2) and sq % 128 == 0 and skv % 256 == 0
    mod = _load_kernel("fp8", dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_BF16, window_right=0 if bool(m["causal"]) else None)
    assert mod.CFG.DTYPE_DS == DTYPE_E4M3 and mod.DS_IS_FP8 and mod.P_D_BLOCK == 128 and mod.P_TMA_ITERS == 1, "the e4m3 dS is the default"
    fn = mod.compile(b, hq, hkv, sq, skv, has_amax=True)
    dev = "cuda"
    q, do, k, v = (ref[n].to(dev).contiguous() for n in ("q", "do", "k", "v"))
    dv = torch.full((b, skv, hq, _D), float("nan"), device=dev, dtype=torch.bfloat16)
    ds = torch.zeros((b, hq, skv, sq), device=dev, dtype=_T_E4M3)
    lse, delta = ref["lse"].to(dev).contiguous(), ref["delta"].to(dev).contiguous()
    ref_ds = ref["ref_ds"].float().transpose(-1, -2).contiguous()  # [B, H_q, S_kv, S_q] fp32 (the dump's fp64 dS, math orientation)
    dp_scale = get_fp8_scale_factor(ref_ds.abs().max().item(), _T_E4M3)

    def s1(val):
        return torch.tensor([float(val)], dtype=torch.float32, device=dev)

    amax_dv, amax_dp = torch.zeros(1, device=dev), torch.zeros(1, device=dev)
    scale = float(m["attn_scale_in"])
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        q, do, k, v, dv, ds, lse, delta,
        s1(m["dscale_Q"]), s1(m["dscale_K"]), s1(m["dscale_V"]), s1(m["dscale_dO"]), s1(1.0), s1(1.0), s1(1.0), s1(dp_scale),
        amax_dv, amax_dp,
        (b, hq, hkv, sq, skv, hq),
        scale, scale * math.log2(math.e), 0, skv,
        stream=stream,
    )  # fmt: skip
    torch.cuda.synchronize()
    got = ds.cpu()
    want_codes = (ref_ds * dp_scale).to(_T_E4M3)
    assert torch.isfinite(got.float()).all(), "e4m3 dS: non-finite code (a NaN residue or a saturated scale)"
    # VALUE-exact, not int8-exact: e4m3 has a signed zero, and the two sides spell a masked cell's zero differently -- the kernel
    # leaves a skipped causal tile at the zero fill's +0 while the fp64 oracle's ``P * (dP - delta)`` is ``-0.0`` wherever
    # ``dP - delta < 0`` (44 % of the skipped cells on the causal dump, 2026-09-28 c05: 21.9 % of ALL codes read "different" on an
    # int8 compare with every unmasked cell exact).  -0 and +0 are one value to the fp8 GEMM; the count is reported, never pinned.
    exact = (got.float() == want_codes.float()).float().mean().item()
    n_signed_zero = int(((got.view(torch.int8) != want_codes.view(torch.int8)) & (got.float() == want_codes.float())).sum())
    step = (got.float() - want_codes.float()).abs()
    ulp = torch.maximum(want_codes.float().abs() * 2.0**-3, torch.full_like(step, 2.0**-9))  # one e4m3 code spacing at |x| (subnormal floor 2^-9)
    n_far = int((step > ulp * 1.0001).sum())
    print(
        f"\n{stem}: e4m3 dS codes: {100 * exact:.3f} % value-exact the fp64 dS's code ({n_signed_zero} differ only in the sign of zero), "
        f"{n_far} of {got.numel()} more than one code apart; scale_dP={dp_scale}"
    )
    assert n_far == 0, "an e4m3 dS code more than one step from the fp64 dS's code: a scrambled chunk, not a midpoint flip"
    assert exact >= 0.99, f"only {100 * exact:.2f} % of the e4m3 dS codes match the fp64 dS's -- midpoint flips are rare, this is not them"
    torch.testing.assert_close(
        got.float() / dp_scale, ref_ds, rtol=2.0**-3 + 2.0**-4, atol=2.0**-9 / dp_scale, msg=lambda s: f"dequantized e4m3 dS vs the dump's fp64 dS: {s}"
    )
    assert amax_dp.item() > 0 and abs(amax_dp.item() - ref_ds.abs().max().item()) <= 1e-2 * ref_ds.abs().max().item(), "amax_dP is the PRE-scale fp32 max |dS|"
    if bool(m["causal"]):
        tri = torch.ones(skv, sq, dtype=torch.bool).tril(-1)  # kv > q: the tiles a kv block skips (stage 2 leaves them to the zero fill)
        assert (got.float()[..., tri] == 0).all(), "skipped causal cells must stay exact zeros"


# =========================================================================== the standalone surface: per-batch kv lengths, external delta, ragged bottom-right
#
# The fp8 row's twins of the half suite's ``_run_adapter`` cells (test_sdpa_bwd_dsl_sm107.py): ``SdpaBwdDslSm107Fp8(seq_kv_lens_present=True)``
# over the twelve scalars and the four amax outputs, the fp8 oracle composing the SAME per-batch lengths and band INSIDE itself
# (``padding=``: slicing the operands per batch entry would be a different e4m3 quantization of nothing but the slice), the
# appended ``external_delta`` plan fact (the dense plans here -- the THD twins are test_sdpa_bwd_thd_fp8_sm107.py's: a caller's fp32
# ``[B, H_q, S_q_pad]`` ``rowsum(dO * O)`` in TRUE units, bound AS IS -- the fp8 kernel reads delta unscaled, nobody applies
# ``descale_o * descale_dO`` to a caller's tensor; zeros past S_q), and
# bottom-right at a ragged S_q through the graph (served now: the body threads ``seqlen_q_real``).  The K / V rows past a batch
# entry's kv length and the delta's pad rows hold FINITE data in every cell (the finite-data contract of the per-batch arm:
# ``chunk_dS = (dP * scale - dot) * P`` and ``0 * NaN = NaN``; the dense rows get finite pads from the zero-filled staging, the
# per-batch arm reads the caller's rows).


def _fp8_per_batch_padding(b, sq, kv_lens):
    """``compute_ref`` / ``compute_ref_backward``'s ``padding`` for per-batch kv lengths with every q row live."""
    return ([sq] * b, [int(n) for n in kv_lens])


def _run_fp8_adapter(
    b=3,
    hq=2,
    hkv=None,
    sq=512,
    skv=512,
    *,
    kv_lens=(512, 300, 0),
    causal=False,
    bottom_right=False,
    left=None,
    dead_lse=None,
    seed=0,
    poison=float("nan"),
    ws_poison=None,
    runs=1,
    grad_dtype=_T_E4M3,
    request_amax=_AMAX,
    external_delta=False,
    delta=None,
):
    """The standalone surface of ``sdpa_bwd_sm107_fp8``: quantize unit-normal operands per tensor, run the fp8 oracle (forward
    for O / Stats, then backward with the oracle's Stats injected) COMPOSING the per-batch kv lengths and the band, build the
    adapter with ``seq_kv_lens_present=True`` (``kv_lens=None`` = dense, no lengths operand) and / or ``external_delta=True``,
    compile, execute ``runs`` times with the twelve scalars (delayed scaling from the oracle's amaxes), the requested amax outputs
    NaN-filled per run and ``seq_kv_lens`` ([B] int32), and hand back an ``_Fp8Run`` (``check()``: the fp8 recipe per gradient +
    the amax contract) with the live plan state attached.  ``dead_lse`` overrides the Stats rows of a zero-length entry (the
    oracle writes the forward's ``-inf``; the suite's 0.0 convention is the other value the kernel must select P = 0 under).
    ``left`` is the graph's band bound (keys per row; the adapter takes ``window_size_left = left - 1``).  ``external_delta``
    with ``delta=None`` hands the kernel the TRUE-unit fp32 ``rowsum(dO * O)`` over the DEQUANTIZED payloads (zeros past S_q) and
    feeds the oracle the SAME delta; a given ``delta`` ([B, H_q, S_q_pad] fp32) is bound as is (the bitwise cells read the chain's
    own pre-pass back)."""
    from sdpa.fp8_ref import compute_ref, compute_ref_backward
    from sdpa.helpers import get_fp8_descale_factor, get_fp8_scale_factor

    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    hkv = hq if hkv is None else hkv
    dense = kv_lens is None
    kv_lens = [skv] * b if dense else [int(n) for n in kv_lens]
    assert len(kv_lens) == b and all(0 <= n <= skv for n in kv_lens), kv_lens
    dev = "cuda"
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(bb, s, h):
        return torch.randn(bb, s, h, _D, generator=gen).to(dev)  # [B, S, H, D] storage

    def quant(x):
        scale = get_fp8_scale_factor(x.abs().max().item(), _T_E4M3)
        return (x * scale).to(_T_E4M3), 1.0 / scale

    q32, do32, k32, v32 = draw(b, sq, hq), draw(b, sq, hq), draw(b, skv, hkv), draw(b, skv, hkv)
    q8, q_ds = quant(q32)
    k8, k_ds = quant(k32)
    v8, v_ds = quant(v32)
    do8, do_ds = quant(do32)
    scale = 1.0 / math.sqrt(_D)
    s_scale = get_fp8_scale_factor(1.0, _T_E4M3)
    s_descale = 1.0 / s_scale
    right = 0 if causal else None
    align = (cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT) if causal else None
    padding = None if dense else _fp8_per_batch_padding(b, sq, kv_lens)
    o8, stats, o_amax = compute_ref(
        q8, k8, v8, scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, _T_E4M3, padding=padding, left_bound=left, right_bound=right, diag_align=align
    )
    # compute_ref's O is a BSHD-shaped view over BHSD memory (see _run_fp8); a CLONE in contiguous format, not ``.contiguous()``: at
    # H == 1 the view already counts as contiguous (a size-1 dim's stride is ignored) and would keep the head stride S * D, which the
    # adapter's exact BSHD-physical stride check refuses.
    o8 = o8.clone(memory_format=torch.contiguous_format)
    o_ds = get_fp8_descale_factor(o_amax, _T_E4M3)
    stats = stats.contiguous()  # [B, H, S_q, 1]; -inf on a row with no key (the forward's contract)
    if dead_lse is not None:
        for bi, n in enumerate(kv_lens):
            if n == 0:
                stats[bi] = float(dead_lse)
    s_pad = -(-sq // 128) * 128
    delta_t = None
    if external_delta:
        if delta is None:
            # the TRUE-unit row-sum a producer hands the kernel: fp32 over the DEQUANTIZED payloads, zeros past S_q
            delta_t = torch.zeros(b, hq, s_pad, device=dev, dtype=torch.float32)
            delta_t[:, :, :sq] = (o8.float() * o_ds * do8.float() * do_ds).sum(-1).permute(0, 2, 1)
        else:
            delta_t = delta
    ds_knob = _ds_dtype_code()

    def ref_bwd(return_intermediates=False, quantize_ds=None):
        if quantize_ds is None:
            quantize_ds = ds_knob == DTYPE_E4M3
        return compute_ref_backward(
            q8, k8, v8, o8, do8, scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, o_ds, do_ds, grad_dtype,
            padding=padding, left_bound=left, right_bound=right, diag_align=align, stats=stats, return_intermediates=return_intermediates,
            quantize_ds=quantize_ds, delta=None if delta_t is None else delta_t[:, :, :sq],
        )  # fmt: skip

    dq_ref, dk_ref, dv_ref, _dsink, dp_amax, dq_amax, dk_amax, dv_amax, inter = ref_bwd(return_intermediates=True)
    dp_scale = get_fp8_scale_factor(dp_amax, _T_E4M3)
    ds_amax = inter["ds_scaled"].abs().max().item() / dp_scale
    grad_scale = {n: get_fp8_scale_factor(a, grad_dtype) for n, a in (("dQ", dq_amax), ("dK", dk_amax), ("dV", dv_amax))}
    scalars = dict(
        descale_q=q_ds,
        descale_k=k_ds,
        descale_v=v_ds,
        descale_o=o_ds,
        descale_dO=do_ds,
        descale_s=s_descale,
        descale_dP=1.0 / dp_scale,
        scale_s=s_scale,
        scale_dQ=grad_scale["dQ"],
        scale_dK=grad_scale["dK"],
        scale_dV=grad_scale["dV"],
        scale_dP=dp_scale,
    )
    samples = dict(q=_view_bhsd(q8), k=_view_bhsd(k8), v=_view_bhsd(v8), o=_view_bhsd(o8), do=_view_bhsd(do8), stats=stats)
    outs_t = {n: torch.empty(sh, device=dev, dtype=grad_dtype) for n, sh in (("dQ", (b, sq, hq, _D)), ("dK", (b, skv, hkv, _D)), ("dV", (b, skv, hkv, _D)))}
    samples.update(dq=_view_bhsd(outs_t["dQ"]), dk=_view_bhsd(outs_t["dK"]), dv=_view_bhsd(outs_t["dV"]))
    api = SdpaBwdDslSm107Fp8(
        **{"sample_" + n: t for n, t in samples.items()},
        is_causal=bool(causal),
        causal_bottom_right=bool(bottom_right),
        window_size_left=None if left is None else int(left) - 1,
        scale_softmax=scale,
        amax_requested=tuple(request_amax),
        seq_kv_lens_present=not dense,
        external_delta=bool(external_delta),
    )
    api.check_support()
    api.compile()
    ws = torch.empty(max(api.scratch_workspace_bytes(), 1), device=dev, dtype=torch.uint8)
    lens = None if dense else torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    sc_t = {n: torch.tensor([float(v)], dtype=torch.float32, device=dev) for n, v in scalars.items()}
    amax_t = {n: torch.full((1,), float("nan"), device=dev, dtype=torch.float32) for n in request_amax}
    args = {n + "_tensor": t for n, t in samples.items()}
    base_kwargs = dict(workspace=ws, seq_kv_lens=lens, delta_tensor=delta_t, **sc_t, **{n: amax_t.get(n) for n in _AMAX})

    def rerun(**over):
        kw = dict(base_kwargs)
        kw.update(over)
        api.execute(**args, **kw)

    outs, amax = [], []
    for _ in range(runs):
        for x in outs_t.values():
            x.fill_(poison)
        for x in amax_t.values():
            x.fill_(float("nan"))
        if ws_poison is not None:
            ws.fill_(ws_poison)
        rerun()
        torch.cuda.synchronize()
        outs.append({n: x.clone() for n, x in outs_t.items()})
        amax.append({n[len("amax_") :]: x.clone() for n, x in amax_t.items()})
    deq = {n: (x8.float() * ds) for n, x8, ds in (("q", q8, q_ds), ("k", k8, k_ds), ("dO", do8, do_ds))}
    run = _Fp8Run(
        outs,
        amax,
        refs=dict(dQ=dq_ref, dK=dk_ref, dV=dv_ref),
        ref_amax=dict(dQ=dq_amax, dK=dk_amax, dV=dv_amax, dP=ds_amax, dP_raw=dp_amax),
        ref_bwd=ref_bwd,
        operands=dict(dQ=deq["k"], dK=deq["q"], dV=deq["dO"]),
        flip_units=dict(dQ=1.0 / dp_scale, dK=1.0 / dp_scale, dV=s_descale),
        keys=dict(dQ=skv, dK=sq, dV=sq),
        descales={n: 1.0 / s for n, s in grad_scale.items()},
        grad_dtype=grad_dtype,
    )
    run.api, run.args, run.ws, run.lens, run.scalars, run.amax_t, run.outs_t, run.delta, run.kv_lens, run.rerun = (
        api,
        args,
        ws,
        lens,
        sc_t,
        amax_t,
        outs_t,
        delta_t,
        kv_lens,
        rerun,
    )
    run.ds_knob, run.dp_scale, run.s_pad = ds_knob, dp_scale, s_pad
    return run


def _assert_dead_kv_rows_exactly_zero_fp8(run, kv_lens):
    """Every kv row at or past its batch entry's length: dK / dV EXACTLY zero and finite (a select, never residue * 0; the e4m3
    code 0, not a quantized tiny value); a zero-length entry additionally owes an exactly-zero dQ (every dS row of it is zero)."""
    outs = run.outs[0]
    for bi, n in enumerate(kv_lens):
        for name in ("dK", "dV"):
            tail = outs[name][bi, n:].float()
            assert torch.isfinite(tail).all(), f"{name}[{bi}]: non-finite past kv length {n}"
            assert (tail == 0).all(), f"{name}[{bi}]: kv rows past the length {n} must be EXACTLY zero, got max|.|={tail.abs().max().item():.3e}"
        if n == 0:
            dead = outs["dQ"][bi].float()
            assert torch.isfinite(dead).all() and (dead == 0).all(), f"dQ[{bi}]: the seq_kv_len == 0 entry must be EXACTLY zero"


def _chain_delta(run):
    """The chain's own delta: region ``R_DELTA`` of the plan's carve (``[B, H_q, S_q_pad]`` fp32, zeros on the pad rows), read back
    out of the run's workspace after its last execute."""
    from cudnn.sdpa.bwd import prepared_sm107
    from cudnn.sdpa.bwd.kernels.sm107.prepared_host import R_DELTA

    offset, shape, _strides = prepared_sm107._regions(run.api, prepared_sm107._REGION_SLOTS_FP8)[0][R_DELTA]
    assert shape == run.api.external_delta_shape == (run.api.batch_size, run.api.h_q, run.s_pad)
    delta = run.ws[offset : offset + 4 * math.prod(shape)].view(torch.float32).view(*shape).clone()
    sq = run.api.s_q_max
    assert torch.isfinite(delta).all() and torch.equal(
        delta[:, :, sq:], torch.zeros_like(delta[:, :, sq:])
    ), "the chain's pad rows are the zeros the contract asks a caller for"
    return delta


def _assert_runs_bitwise(a, b, what):
    for name in ("dQ", "dK", "dV"):
        x, y = a.outs[0][name], b.outs[0][name]
        n_diff = (x.contiguous().view(torch.uint8) != y.contiguous().view(torch.uint8)).sum().item()
        assert n_diff == 0, f"{name}: {what} differ in {n_diff} bytes (max|diff|={(x.float() - y.float()).abs().max().item():.3e})"
    for name in a.amax[0]:
        assert a.amax[0][name].item() == b.amax[0][name].item(), f"amax_{name}: {what} differ"


@requires_rubin
@pytest.mark.parametrize("mask", ["dense", "causal"])
def test_fp8_adapter_per_batch_kv_lengths(ds_knob, mask):
    """The accept matrix of the standalone per-batch kv lengths on the fp8 row (dS knob x band): lengths [512, 300, 0] on
    S_kv = 512 -- a full entry, one ending at 300 (no tile multiple), one DEAD (length 0) -- against the fp8 oracle composing the
    same lengths; the rows past each length exactly zero, the dead entry's dQ exactly zero (poisoned outputs); the four amax
    outputs the LIVE rows' maxima (the dead entry folds nothing: sdpa-invariants s5)."""
    lens = [512, 300, 0]
    run = _run_fp8_adapter(b=3, hq=2, sq=512, skv=512, kv_lens=lens, causal=(mask == "causal")).check()
    _assert_dead_kv_rows_exactly_zero_fp8(run, lens)


@requires_rubin
@pytest.mark.parametrize("dead_lse", [0.0, float("-inf")], ids=["lse-0", "lse-neg-inf"])
def test_fp8_adapter_dead_kv_entry_is_exactly_zero(dead_lse):
    """One batch entry with seq_kv_len == 0, its Stats rows holding 0 or the forward's ``-inf`` (``exp2(S - (-inf)) = inf`` before
    the mask: the kernel's P must be a SELECT to zero, never ``inf * 0``, and its amax folds must see nothing of it)."""
    lens = [512, 0]
    run = _run_fp8_adapter(b=2, hq=2, sq=256, skv=512, kv_lens=lens, dead_lse=dead_lse).check()
    _assert_dead_kv_rows_exactly_zero_fp8(run, lens)


@requires_rubin
def test_fp8_adapter_per_batch_kv_lengths_with_a_ragged_s_kv_gqa_and_window(ds_knob):
    """S_kv = 640 is not a 256-multiple (padded to 768): the zero-filled K / V staging and the caller's lengths share the one
    padded-mask arm; GQA folds the per-Q-head bf16 partials through the staging; a top-left causal window (band bound 200)
    bounds the band from both sides -- the arm every top-left mask takes without the zero-fill (0xFF workspace)."""
    lens = [640, 300, 0]
    run = _run_fp8_adapter(b=3, hq=4, hkv=2, sq=384, skv=640, kv_lens=lens, causal=True, left=200, ws_poison=0xFF).check()
    assert run.api._zero_ws is False, "a top-left band does not move with the length: no zero-fill"
    _assert_dead_kv_rows_exactly_zero_fp8(run, lens)


@requires_rubin
@pytest.mark.parametrize("left", [None, 200], ids=["no-window", "window-200"])
def test_fp8_adapter_per_batch_kv_lengths_bottom_right_fills_the_workspace(ds_knob, left):
    """Bottom-right causal with per-batch lengths: the kernel's diagonal is ``len_b - S_q`` per batch entry (negative for the
    300-length entry: its first 212 query rows have no key and read as dead rows -- the forward's ``-inf`` Stats, P = 0, dQ = 0)
    while the stage-3 K-trim is computed from the uniform ``S_kv - S_q``.  The three rules the half row carries hold here too:
    the zero-fill KEPT (``_stage3_needs_zero_fill(per_batch_kv=True)``; 0xFF-poisoned workspace), the window DROPPED from the
    trim (``_stage3_trim_window``), and -- this row has no batch chunking -- ONE fill per execute ahead of the head loop."""
    lens = [1024, 300, 700, 0]
    run = _run_fp8_adapter(b=4, hq=2, sq=512, skv=1024, kv_lens=lens, causal=True, bottom_right=True, left=left, ws_poison=0xFF).check()
    assert run.api._zero_ws is True, "bottom-right under per-batch lengths must zero-fill the dS workspace"
    assert (run.api._b_chunk, run.api._qh_chunk) == (4, 2), "no batch chunking on the fp8 row: the whole batch is in-grid"
    _assert_dead_kv_rows_exactly_zero_fp8(run, lens)


@requires_rubin
def test_fp8_adapter_per_batch_kv_lengths_past_the_fill_block():
    """B > 256 with the caller's lengths: the kernel reads ``seq_kv_lens[b]`` for EVERY batch entry straight from the caller's
    buffer, zero-length entries among the entries past 256 included."""
    b, sq, skv = 300, 128, 256
    lens = [(skv, skv // 2, 0)[i % 3] for i in range(b)]
    run = _run_fp8_adapter(b=b, hq=1, sq=sq, skv=skv, kv_lens=lens).check()
    _assert_dead_kv_rows_exactly_zero_fp8(run, lens)


@requires_rubin
def test_fp8_adapter_per_batch_kv_lengths_are_a_plan_fact():
    """The lengths operand is bound by the plan: a plan built with ``seq_kv_lens_present`` refuses an execute without the buffer,
    one built without refuses a buffer, ``seq_q_lens`` is refused on both, and a buffer of the wrong element count (B + 1, the
    prefix-sum form) is refused by the prepared ``bind`` -- every one a ValueError before any stage launches."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    lens = [512, 256]
    run = _run_fp8_adapter(b=2, hq=2, sq=256, skv=512, kv_lens=lens).check()
    with pytest.raises(ValueError, match="exactly when"):
        run.rerun(seq_kv_lens=None)
    with pytest.raises(ValueError, match="seq_q_lens"):
        run.rerun(seq_q_lens=run.lens)
    with pytest.raises(ValueError, match="contiguous with"):
        run.rerun(seq_kv_lens=torch.zeros(3, dtype=torch.int32, device="cuda"))
    samples = {"sample_" + name[: -len("_tensor")]: value for name, value in run.args.items()}
    plain = SdpaBwdDslSm107Fp8(**samples, scale_softmax=1.0 / math.sqrt(_D), amax_requested=_AMAX)
    plain.check_support()
    plain.compile()
    ws = torch.empty(max(plain.scratch_workspace_bytes(), 1), device="cuda", dtype=torch.uint8)
    with pytest.raises(ValueError, match="exactly when"):
        plain.execute(**run.args, workspace=ws, seq_kv_lens=run.lens, **run.scalars, **{n: run.amax_t[n] for n in _AMAX})


@requires_sm80
def test_fp8_adapter_external_delta_contract_fires_before_compile():
    """At execute, BEFORE ``compile()`` (no artifact, no launch -- any CUDA host): both directions of the plan fact, then the exact
    ``dot_do_o`` layout -- fp32, contiguous ``(B, H_q, S_q_pad)`` (the PADDED extent, zeros past S_q), the plan's device, a 16-byte
    base -- each a ValueError naming ``delta_tensor``, ahead of the scalar checks.  The Rubin half is
    ``test_fp8_adapter_external_delta_is_bitwise_the_chains_own_pre_pass``."""
    from test_sdpa_bwd_dsl_sm107 import _adapter

    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8

    kw = dict(b=1, hq=4, hkv=2, sq=500, skv=512, dt=_T_E4M3, grad_dt=_T_E4M3, is_causal=True)
    own, ext = _adapter(SdpaBwdDslSm107Fp8, **kw), _adapter(SdpaBwdDslSm107Fp8, external_delta=True, **kw)
    assert own.check_support() and ext.check_support()
    dummy = torch.empty(1, device="cuda")  # never bound: every reject below fires before the bind
    args = {name + "_tensor": dummy for name in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv")}
    args.update(workspace=dummy, **{name: dummy for name in _SCALARS}, **{name: dummy for name in _AMAX})
    good = torch.zeros(1, 4, 512, device="cuda")
    with pytest.raises(ValueError, match="external_delta=False"):
        own.execute(**args, delta_tensor=good)
    with pytest.raises(ValueError, match="delta_tensor is required"):
        ext.execute(**args)
    with pytest.raises(ValueError, match="must be fp32"):
        ext.execute(**args, delta_tensor=good.to(torch.bfloat16))
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        ext.execute(**args, delta_tensor=torch.zeros(1, 4, 1024, device="cuda")[:, :, ::2])
    with pytest.raises(ValueError, match=r"CONTIGUOUS \[B, H_q, S_q_pad\] = \(1, 4, 512\)"):
        ext.execute(**args, delta_tensor=torch.zeros(1, 4, 500, device="cuda"))
    with pytest.raises(ValueError, match="plan's device"):
        ext.execute(**args, delta_tensor=torch.zeros(1, 4, 512))
    with pytest.raises(ValueError, match="16-byte aligned"):
        ext.execute(**args, delta_tensor=torch.zeros(1 * 4 * 512 + 1, device="cuda")[1:].view(1, 4, 512))
    assert own._compiled is None and ext._compiled is None, "a reject must fire before compile()"


@requires_rubin
@pytest.mark.parametrize("sq", [512, 500], ids=["aligned", "padded"])
def test_fp8_adapter_external_delta_is_bitwise_the_chains_own_pre_pass(ds_knob, sq):
    """A plan built with ``external_delta=True`` and fed the delta the chain's OWN scaled pre-pass wrote (read back out of a sibling
    plan's ``R_DELTA`` region: ``rowsum(dO8 * O8) * descale_o * descale_dO`` in fp32) returns dQ / dK / dV and the four amax
    ``torch.equal`` the sibling's -- the same artifact minus the ``dot`` launch, reading the caller's tensor where the sibling reads
    its region.  Also pinned: the external plan launches exactly one kernel fewer (CUPTI, when available); a wrong-shape delta on
    the compiled plan is refused with no launch."""
    from dataclasses import replace

    own = _run_fp8_adapter(b=2, hq=4, hkv=2, sq=sq, skv=512, kv_lens=None, causal=True).check()
    delta = _chain_delta(own)
    ext = _run_fp8_adapter(b=2, hq=4, hkv=2, sq=sq, skv=512, kv_lens=None, causal=True, external_delta=True, delta=delta).check()
    assert own.api._prepared.operands[-1] is None and ext.api._prepared.operands[-1] is not None, "the delta slot binds on the external plan only"
    assert "delta" not in [n for n, _n, _d in ext.api._scratch_plan()] and "delta" in [n for n, _n, _d in own.api._scratch_plan()]
    _assert_runs_bitwise(ext, own, "the external-delta plan vs the chain's own pre-pass")
    # one launch fewer (the `dot` kernel), counted with CUPTI: only the profiler's own start may fail (-> None); a failure from
    # rerun() propagates and the count assertion sits outside any handler, so a restored dot launch FAILS the test
    counts = cuda_launch_counts(own.rerun, ext.rerun)
    if counts is None:
        print("\nlaunch count unverified here (no CUDA profiler activity: CUPTI unavailable)")
    else:
        assert counts[1] == counts[0] - 1, counts
        print(f"\nlaunches: own {counts[0]}, external delta {counts[1]}")
    launches = []
    ext.api._prepared = replace(ext.api._prepared, fn=lambda *args: launches.append(args))
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        ext.rerun(delta_tensor=delta[:, :, :sq].contiguous() if sq % 128 else delta.transpose(1, 2))
    with pytest.raises(ValueError, match="delta_tensor is required"):
        ext.rerun(delta_tensor=None)
    assert not launches


@requires_rubin
def test_fp8_adapter_external_delta_binds_true_units_as_is(ds_knob):
    """The TRUE-unit contract of the fp8 row's external delta: a producer's fp32 ``rowsum(dO * O)`` over the DEQUANTIZED payloads
    (what a bf16-derived delta is) is bound AS IS -- the kernel reads delta unscaled, nobody applies ``descale_o * descale_dO`` --
    and the gradients hold under the recipe against the fp8 oracle FED THE SAME DELTA (a different fp32 summation order than the
    chain's own pre-pass, so this cell is a recipe compare, never bitwise: ``test_fp8_adapter_external_delta_is_bitwise_the_chains_own_pre_pass``
    is the bitwise one)."""
    _run_fp8_adapter(b=2, hq=4, hkv=2, sq=500, skv=512, kv_lens=None, causal=True, external_delta=True).check()


@requires_rubin
def test_fp8_adapter_per_batch_kv_lengths_compose_with_the_external_delta(ds_knob):
    """A plan built with BOTH appended plan facts -- the per-batch kv lengths (slot 25) and the caller's delta (slot 26) -- fed the
    delta the lengths-only plan's own pre-pass wrote returns dQ / dK / dV and the amax ``torch.equal`` that plan's over the same
    lengths [512, 300, 0] (S_q = 500 ragged too: the pad rows of the delta are the zeros the contract asks for).  The slots: both
    bound on the combined plan, exactly one on each single-fact sibling; the compiled plans' refusals each a ValueError before any
    launch."""
    from dataclasses import replace

    from cudnn.sdpa.bwd.prepared_sm107 import EXTERNAL_DELTA_ROLE

    lens = [512, 300, 0]
    lengths_only = _run_fp8_adapter(b=3, hq=4, hkv=2, sq=500, skv=512, kv_lens=lens, causal=True).check()
    _assert_dead_kv_rows_exactly_zero_fp8(lengths_only, lens)
    delta = _chain_delta(lengths_only)
    both = _run_fp8_adapter(b=3, hq=4, hkv=2, sq=500, skv=512, kv_lens=lens, causal=True, external_delta=True, delta=delta).check()
    delta_only = _run_fp8_adapter(b=3, hq=4, hkv=2, sq=500, skv=512, kv_lens=None, causal=True, external_delta=True)
    for run, bound in ((lengths_only, (True, False)), (both, (True, True)), (delta_only, (False, True))):
        spec = run.api._prepared
        assert spec.roles[-2:] == ("seq_kv", EXTERNAL_DELTA_ROLE) and len(spec.operands) == len(spec.roles) == 27
        assert (spec.operands[-2] is not None, spec.operands[-1] is not None) == bound
    assert lengths_only.api.scratch_workspace_bytes() - both.api.scratch_workspace_bytes() == ws_align(
        3 * 4 * 512 * 4
    ), "the carve lost exactly the delta region"
    _assert_runs_bitwise(both, lengths_only, "the combined plan vs the unfused per-batch-lengths plan")
    launches = []
    for run in (lengths_only, both, delta_only):
        run.api._prepared = replace(run.api._prepared, fn=lambda *args: launches.append(args))
    with pytest.raises(ValueError, match="exactly when"):
        both.rerun(seq_kv_lens=None)  # the lengths missing
    with pytest.raises(ValueError, match="delta_tensor is required"):
        both.rerun(delta_tensor=None)  # the delta missing
    with pytest.raises(ValueError, match="external_delta=False"):
        lengths_only.rerun(delta_tensor=delta)  # a delta the plan did not ask for
    with pytest.raises(ValueError, match="exactly when"):
        delta_only.rerun(seq_kv_lens=lengths_only.lens)  # lengths the plan did not ask for
    assert not launches


@requires_rubin
@pytest.mark.parametrize("sq,skv", _BR_RAGGED_SQ_SHAPES, ids=_BR_RAGGED_SQ_IDS)
def test_causal_bottom_right_ragged_s_q_is_served(ds_knob, sq, skv):
    """Bottom-right causal at a RAGGED S_q through the GRAPH -- the shape the row used to decline (``bottom_right_s_q_multiple``
    128 -> 1): the body derives the diagonal ``S_kv - S_q`` and its q-tile trim from ``seqlen_q_real``, so S_q = 500 runs its own
    diagonal, not S_q = 512's; the staging's zero-filled dO and ``+inf`` LSE rows keep the pad rows at P = 0."""
    _run_fp8(b=1, hq=2, sq=sq, skv=skv, causal=True, bottom_right=True).check()


# --------------------------------------------------------------------------- the GQA fold rounds ONCE: fp32 per-Q-head partials (host) + one fold / one setup launch


def test_fp8_gqa_partials_are_fp32_on_the_e4m3_chain_and_bf16_otherwise(monkeypatch, ds_knob):
    """Under GQA the fp8 row's per-Q-head dK / dV partials are fp32 on the e4m3-dS chain -- the main kernel is loaded at
    ``dtype_o = DTYPE_FP32`` (dV_true stored UNROUNDED) and the dK record's ``EPI_DESCALE`` output is fp32 -- so the fold's
    fixed-order fp32 sum of the group is rounded ONCE (a bf16 partial was rounded a second time by the fold: relative RMS 2.7e-3 on
    dK / dV vs the reference under GQA, 0 under MHA).  At MHA nothing changes (``dtype_o = BF16``, a bf16 ``dv_part``, no ``dk_part``:
    the fold is a copy + amax, dK is quantized in the GEMM -- the pre-fp32 kernels and bits), and the bf16-dS twin keeps bf16
    partials on every path (no SMEM for the fp32 dV staging beside its 96 KiB dS ring; its bf16 renderings store the io dtype)."""
    from cudnn.frost.tile_dsl.constants import DTYPE_FP32
    from cudnn.sdpa.bwd import api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8
    from cudnn.sdpa.bwd.config_sm100 import EPI_DESCALE, EPI_QUANT, matmul_out_dtype
    from test_sdpa_bwd_dsl_sm107 import _adapter

    monkeypatch.setattr(sm107, "FP8_DS_DTYPE", ds_knob)
    e4m3_chain = ds_knob == DTYPE_E4M3
    gqa = _adapter(SdpaBwdDslSm107Fp8, b=1, hq=4, hkv=2, dt=_T_E4M3, grad_dt=torch.bfloat16, amax_requested=_AMAX)
    mha = _adapter(SdpaBwdDslSm107Fp8, b=1, hq=2, hkv=2, dt=_T_E4M3, grad_dt=torch.bfloat16, amax_requested=_AMAX)
    assert gqa.check_support() and mha.check_support()
    assert gqa._fp32_partials is e4m3_chain and mha._fp32_partials is False
    assert gqa._template_params().dtype_o == (DTYPE_FP32 if e4m3_chain else DTYPE_BF16), "the main kernel stores dV_true unrounded under GQA"
    assert mha._template_params().dtype_o == DTYPE_BF16, "MHA keeps the bf16 dV_true the copy-fold reads: the same kernel as before"
    plan_gqa = {name: (tuple(int(x) for x in shape), dt) for name, shape, dt in gqa._scratch_shapes()}
    plan_mha = {name: (tuple(int(x) for x in shape), dt) for name, shape, dt in mha._scratch_shapes()}
    part = torch.float32 if e4m3_chain else torch.bfloat16
    assert plan_gqa["dv_part"] == ((1, 512, 4, _D), part) and plan_gqa["dk_part"] == ((1, 512, 4, _D), part)
    assert plan_mha["dv_part"] == ((1, 512, 2, _D), torch.bfloat16), "MHA: the bf16 dV_true the fold copies"
    assert ("dk_part" in plan_mha) is (not e4m3_chain), "MHA on the e4m3 chain quantizes dK in the GEMM: no dK partial"
    # the records: the DESCALE output IS the fp32 partial (the template pins FP32 to DESCALE), QUANT the gradient dtype
    dk, dq = gqa._stage3_records(types.SimpleNamespace(CFG=types.SimpleNamespace(TILE_M=128, CTA_MMA=2)), (256, 256))
    if e4m3_chain:
        assert (dk.epi_mode, matmul_out_dtype(dk)) == (EPI_DESCALE, DTYPE_FP32) and (dq.epi_mode, matmul_out_dtype(dq)) == (EPI_QUANT, DTYPE_BF16)
    else:
        assert matmul_out_dtype(dk) == matmul_out_dtype(dq) == DTYPE_BF16, "the twin's bf16 renderings store the io dtype"
    # the carve pays for it: the two partials double under GQA on the e4m3 chain (64 KiB per kv token at the 397B geometry)
    bytes_part = 2 * (1 * 512 * 4 * _D) * part.itemsize
    assert sum(ws_align(math.prod(s) * dt.itemsize) for n, (s, dt) in plan_gqa.items() if n in ("dv_part", "dk_part")) == ws_align(bytes_part // 2) * 2


def test_fp32_dv_partial_config_fits_the_rubin_cap_only_with_the_e4m3_ds_ring():
    """``dtype_o = DTYPE_FP32`` on the fp8 body: the dV staging aliasing K + V grows to 128 KiB (``BPE_O = 4``, 32-element 128-B
    rows, 8 TMA subtiles), 322 KiB of slabs -- 1 KiB under the 325 KiB usable -- with every tcgen05 descriptor root where it
    was (K / V sit at the alias slab's start, nothing descriptor-reads the slabs after it); with the bf16-dS twin's 96 KiB ring
    the same staging is 370 KiB and the config REFUSES it (which is why the twin keeps bf16 partials), and the MXFP8 body
    does not take the code at all (half-precision gradients, no fold-quant pass)."""
    from cudnn.frost.tile_dsl.constants import DTYPE_FP32
    from cudnn.sdpa.bwd import config_sm107 as cfg

    fp32 = cfg.make_cfg_d256_bwd(cfg.TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_FP32), cfg.FAMILY_FP8)
    bf16 = cfg.make_cfg_d256_bwd(cfg.TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_BF16), cfg.FAMILY_FP8)
    assert (fp32.DTYPE_O, fp32.BPE_O) == (DTYPE_FP32, 4) and cfg.bpe(DTYPE_FP32) == 4
    b = cfg.buffer_elems(fp32)
    assert (b.DV_D_BLOCK, b.TMA_DV_ITERS, b.DV_BLOCK_SLAB) == (32, 8, 128 * 32), "32 fp32 per 128-B store row, eight subtiles per 256-wide dV row"
    assert cfg.smem_bytes(bf16) == 258 * 1024 and cfg.smem_bytes(fp32) == 322 * 1024
    assert cfg.kernel_smem_bytes(fp32) <= cfg.SMEM_CAP_BYTES, "the fp32 staging must fit the 327 KiB oversized cap with the scaffold"
    assert cfg.desc_roots(fp32) == cfg.desc_roots(bf16), "no descriptor root moves: the alias slab only grows at its end"
    assert max(off for _lbl, off in cfg.desc_roots(fp32)) < 256 * 1024
    with pytest.raises(ValueError, match="exceed the 327 KiB"):
        cfg.make_cfg_d256_bwd(cfg.TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_FP32, dtype_ds=DTYPE_BF16), cfg.FAMILY_FP8)
    with pytest.raises(ValueError, match="dtype_o must be"):
        cfg.make_cfg_d256_bwd(cfg.TemplateParams(dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_FP32), cfg.FAMILY_MXFP8)
    with pytest.raises(ValueError, match="dtype_o must be"):
        cfg.make_cfg_d256_bwd(cfg.TemplateParams(dtype_qkv=DTYPE_BF16, dtype_o=DTYPE_FP32), cfg.FAMILY_F16)


def test_stage3_fp32_output_is_the_descale_or_block_scale_partial_only():
    """The stage-3 template's fp32 D is a per-Q-head true-unit partial and nothing else: ``validate_matmul_params`` admits
    ``dtype_out = DTYPE_FP32`` with ``EPI_DESCALE`` on the per-tensor fp8 arm and with ``EPI_NONE`` on the BLOCK-SCALE arm (whose MMA
    already dequantized: the accumulator IS the true-unit value), and refuses it with ``EPI_QUANT`` (a quantized gradient has a
    gradient dtype), with ``EPI_NONE`` on the per-tensor fp8 arm (an unscaled accumulator has no fp32 consumer) and on the bf16 rows
    (``EPI_NONE`` stores the io dtype); ``_stage3_params`` writes it on the DESCALE record by itself (no caller passes it) and on the
    block-scale dK record under ``dk_fp32_out`` only (the dQ record keeps the inherited bf16; the flag without ``block_scale`` is
    refused), leaving the QUANT record's gradient dtype and the half row's inherited -1 alone."""
    from cudnn.frost.tile_dsl.constants import DTYPE_FP32
    from cudnn.sdpa.bwd.api_dsl_sm107 import _stage3_params
    from cudnn.sdpa.bwd.config_sm100 import EPI_DESCALE, EPI_NONE, EPI_QUANT, MatmulTemplateParams, matmul_out_dtype, validate_matmul_params

    fp8 = dict(dtype_qkv=DTYPE_E4M3, cgrp_tile_mn=(256, 256))
    validate_matmul_params(MatmulTemplateParams(**fp8, epi_mode=EPI_DESCALE, dtype_out=DTYPE_FP32))
    with pytest.raises(ValueError, match="FP32 output is the DESCALE"):
        validate_matmul_params(MatmulTemplateParams(**fp8, epi_mode=EPI_QUANT, dtype_out=DTYPE_FP32))
    with pytest.raises(ValueError):
        validate_matmul_params(MatmulTemplateParams(dtype_qkv=DTYPE_BF16, cgrp_tile_mn=(256, 256), epi_mode=EPI_NONE, dtype_out=DTYPE_FP32))
    with pytest.raises(ValueError):  # the per-tensor fp8 arm: EPI_NONE is not even an epilogue it renders, fp32 or not
        validate_matmul_params(MatmulTemplateParams(**fp8, epi_mode=EPI_NONE, dtype_out=DTYPE_FP32))
    # the block-scale arm: its EPI_NONE partial may be fp32 (the MXFP8 row's GQA dK), the inherited default stays bf16
    validate_matmul_params(MatmulTemplateParams(**fp8, block_scale=True, epi_mode=EPI_NONE, dtype_out=DTYPE_FP32))
    assert matmul_out_dtype(MatmulTemplateParams(**fp8, block_scale=True, epi_mode=EPI_NONE)) == DTYPE_BF16
    with pytest.raises(ValueError, match="block_scale dequantizes IN the MMA"):  # a block-scale record has no epilogue to quantize with, fp32 or not
        validate_matmul_params(MatmulTemplateParams(**fp8, block_scale=True, epi_mode=EPI_QUANT, dtype_out=DTYPE_FP32))
    dk, dq = _stage3_params(DTYPE_E4M3, causal=True, shift=0, gran=256, cgrp_tile_mn=(256, 256), epi_modes=(EPI_DESCALE, EPI_QUANT), dtype_out=DTYPE_BF16)
    assert (dk.dtype_out, matmul_out_dtype(dk)) == (DTYPE_FP32, DTYPE_FP32) and (dq.dtype_out, matmul_out_dtype(dq)) == (DTYPE_BF16, DTYPE_BF16)
    dk, dq = _stage3_params(DTYPE_E4M3, causal=False, shift=0, gran=256, cgrp_tile_mn=(256, 256), epi_modes=(EPI_QUANT, EPI_QUANT), dtype_out=DTYPE_E4M3)
    assert dk.dtype_out == dq.dtype_out == DTYPE_E4M3
    dk, dq = _stage3_params(DTYPE_BF16, causal=False, shift=0, gran=256, cgrp_tile_mn=(256, 256))
    assert dk.dtype_out == dq.dtype_out == -1 and matmul_out_dtype(dk) == DTYPE_BF16
    # the block-scale records: fp32 dK partial on request, bf16 dQ always; today's bytes without the flag; the flag needs the arm
    dk, dq = _stage3_params(DTYPE_E4M3, causal=True, shift=0, gran=256, cgrp_tile_mn=(256, 256), block_scale=True, gqa_group=4, dq_single_launch=False)
    assert dk.dtype_out == dq.dtype_out == -1 and matmul_out_dtype(dk) == matmul_out_dtype(dq) == DTYPE_BF16 and dk.block_scale and dq.block_scale
    dk, dq = _stage3_params(
        DTYPE_E4M3, causal=True, shift=0, gran=256, cgrp_tile_mn=(256, 256), block_scale=True, gqa_group=4, dq_single_launch=False, dk_fp32_out=True
    )
    assert (
        (dk.dtype_out, matmul_out_dtype(dk)) == (DTYPE_FP32, DTYPE_FP32)
        and dk.epi_mode == EPI_NONE
        and (dq.dtype_out, matmul_out_dtype(dq)) == (-1, DTYPE_BF16)
    )
    validate_matmul_params(dk)
    validate_matmul_params(dq)
    with pytest.raises(ValueError, match="dk_fp32_out"):
        _stage3_params(
            DTYPE_E4M3, causal=True, shift=0, gran=256, cgrp_tile_mn=(256, 256), epi_modes=(EPI_DESCALE, EPI_QUANT), dtype_out=DTYPE_BF16, dk_fp32_out=True
        )
    with pytest.raises(ValueError, match="dk_fp32_out"):
        _stage3_params(DTYPE_BF16, causal=False, shift=0, gran=256, cgrp_tile_mn=(256, 256), dk_fp32_out=True)


@requires_rubin
@pytest.mark.parametrize("hkv", [2, 4], ids=["gqa", "mha"])
def test_fp8_row_launches_one_setup_and_one_fold_whatever_the_group(hkv):
    """The fp8 row's dense chain is ``setup + dot + c x (main + dK + q x dQ) + fold`` = 6 launches at one chunk with one dQ launch,
    at GQA and MHA alike: the uniform kv-length fill and the amax resets are ONE setup launch (``_fp8_setup``), and under GQA
    the dK fold rides the dV fold's launch (``fold_quant_pair``) -- the two launches the chain used to spend on them are gone.
    Counted with CUPTI (``cuda_launch_counts``; a profiler that records nothing here prints 'unverified' and asserts nothing)."""
    gqa = _run_fp8_adapter(b=1, hq=4, hkv=hkv, sq=512, skv=512, kv_lens=None, causal=True)
    gqa.check()
    counts = cuda_launch_counts(gqa.rerun)
    if counts is None:
        print("\nlaunch count unverified here (no CUDA profiler activity: CUPTI unavailable)")
    else:
        print(f"\nlaunches at hq=4 hkv={hkv}: {counts[0]}")
        assert counts == [6], f"setup + dot + main + dK + dQ + fold = 6 launches expected at hq=4 hkv={hkv}; CUPTI saw {counts[0]}"


# ---------------------------------------------------------------------------- the fold + quantize pass: geometry and sm_107a SASS pins
# The pass streams the GQA partials (2 x 256 MiB of fp32 for the dV + dK pair at B=1, H_q=32, S_kv=8K), so its form is a bandwidth
# decision -- and one no numerics test can see: the previous form read every partial through 32-bit scalar loads (its vector load
# had lost the tensor's alignment in the pointer arithmetic) and ran at 41-56 % of the HBM pin, bitwise correct.  Two tripwires:
# the geometry helpers (one 16-byte load per partial per thread, a persistent grid capped at 8 CTAs of 256 threads per SM) and the
# SASS of the pair kernel compiled for sm_107a (128-bit loads, no scalar partial reads, no spill, ONE atomicMax site per operand
# set).  A third pin keeps the fp8 host-compile entries from growing a placeholder default for the device's multiprocessor count.
_FOLD_SASS_PROBE = """
import glob, os, re, subprocess, sys
dump, group, cands = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
os.environ["CUTE_DSL_DUMP_DIR"] = dump
os.environ["CUTE_DSL_KEEP"] = "cubin"
os.environ["CUTE_DSL_ARCH"] = "sm_107a"
os.environ["CUDNN_FRONTEND_DISABLE_COMPILED_CACHE"] = "1"
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cudnn.sdpa.bwd.kernels.bprop_chain_common import fold_quant_pair_host
B, S, Hk, D = 1, 1024, 2, 256
ws = make_fake_compact_tensor(cutlass.Float32, (B, S, Hk * group, D), stride_order=(3, 2, 1, 0), assumed_align=16)
out = make_fake_compact_tensor(cutlass.Float8E4M3FN, (B, S, Hk, D), stride_order=(3, 2, 1, 0), assumed_align=16)
sc = make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=16)
cute.compile(fold_quant_pair_host, ws, out, None, sc, sc, ws, out, None, sc, sc, D, group, cutlass.Float8E4M3FN, 204,
             make_fake_stream(use_tvm_ffi_env_stream=False), None, options="--enable-tvm-ffi --gpu-arch sm_107a")
cubins = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)
if not cubins:
    print("FAIL no cubin dumped into", dump, os.listdir(dump)); sys.exit(3)
nvd = None
for c in cands:
    try:
        proc = subprocess.run([c, "-c", cubins[-1]], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        print("REJECT", c, "->", repr(exc)); continue
    if proc.returncode == 0 and proc.stdout.strip():
        nvd = c; break
    print("REJECT", c, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
if nvd is None:
    print("SKIP no nvdisasm candidate decodes the cubin"); sys.exit(0)
sass = subprocess.run([nvd, "-c", cubins[-1]], capture_output=True, text=True, check=True).stdout.splitlines()
def cnt(pat):
    rx = re.compile(pat)
    return sum(1 for ln in sass if rx.search(ln))
for key, pat in (("LDG128", r"LDG\\.E\\.128"), ("LDG64", r"LDG\\.E\\.64"), ("LDG32", r"LDG\\.E\\b(?!\\.)"), ("STL", r"\\bSTL\\b"), ("LDL", r"\\bLDL\\b"),
                 ("REDMAX", r"REDG\\.E\\.MAX"), ("FMNMX", r"\\bFMNMX"), ("BAR", r"\\bBAR\\.SYNC")):
    print("SASS", key, cnt(pat))
"""


def test_fp8_fold_pass_geometry_is_one_aligned_load_per_partial_on_a_persistent_grid():
    """The helpers the fold hosts size their launch with: one 16-byte load per partial per thread (4 fp32 / 8 bf16 elements), work
    items of ``threads x vec`` elements, and a persistent CTA count capped at ``FOLD_QUANT_CTAS_PER_SM`` CTAs of 256 threads per SM
    in the GRID (2048 threads per SM requested -- two resident sets, the fp32 group-16 pair kernel being 64 registers per thread;
    the measured optimum on Rubin: a cap of 4 CTAs per SM lost 5-8 %, a one-item-per-CTA grid 7-41 % at S=32K) and never above
    the items; the pair split hands each set at least one CTA and the cap in proportion to its items."""
    import cutlass

    from cudnn.sdpa.bwd.kernels import bprop_chain_common as C

    assert C.FOLD_QUANT_LOAD_BYTES == 16 and C.FOLD_QUANT_THREADS * C.FOLD_QUANT_CTAS_PER_SM == 2048
    assert C.fold_quant_vec(cutlass.Float32) == 4 and C.fold_quant_vec(cutlass.BFloat16) == 8
    items = C.fold_quant_items((1, 8192, 2, 256), 256, 4)  # the 397B geometry's dV set at S=8K: 4M elements / (256 x 4)
    assert items == 8192 * 2 * 256 // (256 * 4) == 4096
    assert C.fold_quant_items((2, 1000, 2, 256), 256, 4) == -(-2 * 1000 * 2 * 256 // 1024)  # a tail item, never a dropped one
    assert C.fold_quant_ctas(items, 204) == 204 * 8 == 1632 and C.fold_quant_ctas(100, 204) == 100 and C.fold_quant_ctas(0, 204) == 1
    assert C.fold_quant_pair_ctas(items, items, 204) == (816, 816)
    for items_a, items_b in ((3, 3000), (3000, 3), (1, 1), (5000, 5000), (1632, 1)):
        a, b = C.fold_quant_pair_ctas(items_a, items_b, 204)
        assert 1 <= a <= items_a and 1 <= b <= items_b and a + b <= 1632, (items_a, items_b, a, b)  # at least one CTA per set, never more than items or the cap
    assert C.fold_quant_pair_ctas(3, 3000, 204) == (1, 1631)  # proportional (floor): the small set keeps one CTA, the cap goes to the large one
    assert C.fold_quant_pair_ctas(1, 1, 204) == (1, 1)


def test_fp8_prepared_host_compile_takes_the_device_sm_count_without_a_default():
    """The two fp8 host-compile entries size the fold passes' persistent grid on ``sm_count`` -- the device's multiprocessor count,
    a plan fact the caller reads from the device and folds into the cache key.  A placeholder default would compile a CORRECT
    artifact whose fold pass streams half a gigabyte of partials through a handful of CTAs, so the argument is keyword-only with
    no default, and a value that is not a positive int is rejected by name before anything is traced."""
    import inspect

    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host as P

    for entry in (P.compile_host_fp8, P.compile_host_fp8_thd):
        param = inspect.signature(entry).parameters["sm_count"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty, entry.__name__
        with pytest.raises(TypeError, match="sm_count"):
            entry(None, None, None, None, None, None, None, (False, False, False, False), 107, "key")
        for bad in (0, -1, 1.5, None, 204.0):
            with pytest.raises(ValueError, match="multiprocessor count"):
                entry(None, None, None, None, None, None, None, (False, False, False, False), 107, "key", sm_count=bad)


def test_fp8_fold_pass_sass_reads_128_bit_vectors_and_folds_one_atomic_per_cta(tmp_path):
    """sm_107a SASS of ``fold_quant_pair_kernel`` (fp32 partials, group 16, e4m3 out): every partial read is a 128-bit load (one
    per partial per set body -- the previous form read 258 32-bit scalars per thread and no LDG.128), the only 32-bit loads are
    the two scale words, no stack spill, the amax tree on FMNMX, and exactly ONE ``REDG.E.MAX`` site per operand set behind the
    CTA barrier (the per-warp atomic of the previous form was 32768 same-address arrivals per launch at S=8K)."""
    import subprocess
    import sys

    from frost_test_utils import arch_known_to_the_dsl, nvdisasm_candidates

    if not arch_known_to_the_dsl("sm_107a"):
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0)")
    cands = nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    group = 16
    dump = tmp_path / "fold_quant_pair_sm107a"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _FOLD_SASS_PROBE, str(dump), str(group), *cands], capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"sm_107a trace-compile of fold_quant_pair_host failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    out = proc.stdout.splitlines()
    if any(ln.startswith("SKIP") for ln in out):
        pytest.skip(str([ln for ln in out if ln.startswith(("SKIP", "REJECT"))]))
    stats = {ln.split()[1]: int(ln.split()[2]) for ln in out if ln.startswith("SASS ")}
    print(f"\nfold_quant_pair sm_107a SASS: {stats}")
    assert stats["LDG128"] >= 2 * group, f"fewer 128-bit loads than partials: the partial reads are not vectorized ({stats})"
    assert stats["LDG64"] == 0 and stats["LDG32"] <= 4, f"scalar / 64-bit global loads beyond the scale words: a partial read lost its alignment ({stats})"
    assert stats["STL"] == 0 and stats["LDL"] == 0, f"the fold pass spills ({stats})"
    assert stats["FMNMX"] > 0, f"the amax tree is compare + select, not FMNMX ({stats})"
    assert stats["REDMAX"] == 2, f"expected ONE amax atomic site per operand set (2 for the pair), got {stats['REDMAX']}"
    assert stats["BAR"] >= 2, f"no CTA barrier: the warp maxima are not combined before the atomic ({stats})"
