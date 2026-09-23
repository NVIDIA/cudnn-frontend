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
backend's fp8 suite passes against, which models the e4m3 quantization of P
and dS exactly as the contract does), under the backend suite's tolerance
recipe and ``assert_close_fp8_grad``'s midpoint-flip budget; REJECT tests
assert the decline through the row's own ``mismatch()`` on REAL graphs with a
faked cc 10.7 device.  The two graph-admission cases of the bring-up
placeholder are kept (Rubin only).  The host-only static and sm_107a SASS pins
of this row's kernel body live in ``test_sdpa_bwd_dsl_sm107.py`` (both bodies,
one machinery); the kernel-vs-kernel bitwise targets are the
``frost_dev/results/bwd_d256_sm107/vibetile_ref/fp8_*.pt`` dumps.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from frost_test_utils import _SM, requires_dsl, requires_rubin, select_engine

pytestmark = [pytest.mark.L0, requires_dsl]

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
    assert not c.right_band_widening
    assert not c.thd and not c.thd_declared_totals and not c.cu_seq_len
    assert not c.bias and not c.dbias
    assert not c.decode
    assert c.layouts == frozenset({"bshd"})
    assert not c.tile_ms and not c.tile_ns
    for deferred in ("padded", "sink", "dsink", "deterministic"):
        assert not getattr(c, deferred), f"{deferred} is deferred to PR-2c (plan Q4): claim it together with its accept test here and the tracker line"


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
        dict(causal=True, left_bound=256),
        dict(causal=False, hkv=1),
        dict(causal=False, h=8, hkv=2),
        dict(causal=False, s=500, skv=500),
        dict(causal=False, scale=None),
        dict(causal=False, request_amax=()),
        dict(causal=True, request_amax=("amax_dP",)),
    ],
    ids=["dense", "causal", "bottom-right", "swa", "mqa", "gqa-r4", "non-tile-S", "default-scale", "no-amax", "amax-dP-only"],
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


def test_padding_mask_follows_the_padded_claim(monkeypatch):
    """Deferred in v1 (and the pre-port fp8 d256 FORWARD hangs on a seq_kv_len == 0 entry, so the claim is gated on the
    poisoned degenerate case below); inverts, rather than being deleted, once ``padded`` flips."""
    reason = _decline_reason(monkeypatch, padded=True)
    if _spec().capabilities.padded:
        assert reason is None, reason
    else:
        assert reason is not None


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
            a, r = self.amax[0][name].item(), self.ref_amax[name]
            assert math.isfinite(a), f"amax_{name} was not written (NaN poison survived)"
            assert abs(a - r) <= _FP8_GRAD_TOL["atol"] + _FP8_GRAD_TOL["rtol"] * r, f"amax_{name} {a:.5f} vs the oracle's pre-quant max {r:.5f}"
        a, r = self.amax[0]["dP"].item(), self.ref_amax["dP"]
        assert math.isfinite(a), "amax_dP was not written"
        assert abs(a - r) <= _AMAX_DS_TOL["atol"] + _AMAX_DS_TOL["rtol"] * r, (
            f"amax_dP {a:.6f} vs max|dS| {r:.6f}: the contract reduces the fp32 dS = P (dP - D) attn_scale right before its scale_dP cast "
            f"(sdpa_fp8_bwd.h), not dO V^T (max {self.ref_amax['dP_raw']:.4f})"
        )
        return self


def _run_fp8(b=1, hq=2, hkv=None, sq=512, skv=512, causal=False, bottom_right=False, left=None, seed=0, runs=1, poison=None, grad_dtype=_T_E4M3):
    """Quantize unit-normal operands per tensor, run the oracle (forward for O / Stats, then backward with the oracle's
    Stats injected -- the kernel recomputes P from the forward's exact LSE), build the graph with the contract's twelve
    scalars (delayed scaling: scale_dP / dQ / dK / dV from the oracle's amaxes, the recipe the backend suite feeds),
    PIN the engine, execute ``runs`` times and hand back every run's outputs plus what to compare them with."""
    from sdpa.fp8_ref import compute_ref, compute_ref_backward
    from sdpa.helpers import get_fp8_descale_factor, get_fp8_scale_factor

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
    scale = 1.0 / math.sqrt(_D)
    s_scale = get_fp8_scale_factor(1.0, _T_E4M3)  # P <= 1
    s_descale = 1.0 / s_scale
    right = 0 if causal else None
    align = (cudnn.diagonal_alignment.BOTTOM_RIGHT if bottom_right else cudnn.diagonal_alignment.TOP_LEFT) if causal else None
    o8, stats, o_amax = compute_ref(
        q8, k8, v8, scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, _T_E4M3, left_bound=left, right_bound=right, diag_align=align
    )
    o_ds = get_fp8_descale_factor(o_amax, _T_E4M3)

    def ref_bwd(return_intermediates=False):
        return compute_ref_backward(
            q8, k8, v8, o8, do8, scale, q_ds, k_ds, v_ds, s_scale, s_descale, _T_E4M3, o_ds, do_ds, grad_dtype,
            left_bound=left, right_bound=right, diag_align=align, stats=stats, return_intermediates=return_intermediates
        )  # fmt: skip

    dq_ref, dk_ref, dv_ref, _dsink, dp_amax, dq_amax, dk_amax, dv_amax, inter = ref_bwd(return_intermediates=True)
    dp_scale = get_fp8_scale_factor(dp_amax, _T_E4M3)
    ds_amax = inter["ds_scaled"].abs().max().item() / dp_scale  # the fp32 dS the contract's amax_dP reduces
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
    cudnn_grad = {_T_E4M3: _FP8, torch.bfloat16: _BF16, torch.float16: cudnn.data_type.HALF}[grad_dtype]
    g, ts = _graph_and_ports(
        b, hq, sq, _D, hkv=hkv, skv=skv, causal=causal, bottom_right=bottom_right, left_bound=left, grad_dtypes=(cudnn_grad,) * 3, scale=scale
    )
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    outs_t = {n: torch.empty(sh, device=dev, dtype=grad_dtype) for n, sh in (("dQ", (b, sq, hq, _D)), ("dK", (b, skv, hkv, _D)), ("dV", (b, skv, hkv, _D)))}
    amax_t = {n: torch.full((1, 1, 1, 1), float("nan"), device=dev, dtype=torch.float32) for n in ("dQ", "dK", "dV", "dP")}
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
    pack.update({ts[f"amax_{n}"]: amax_t[n] for n in ("dQ", "dK", "dV", "dP")})
    outs, amax = [], []
    for _ in range(runs):
        for x in outs_t.values():
            x.fill_(float("nan") if poison is None else poison)
        for x in amax_t.values():
            x.fill_(float("nan"))
        g.execute(pack, ws)
        torch.cuda.synchronize()
        outs.append({n: x.clone() for n, x in outs_t.items()})
        amax.append({n: x.clone() for n, x in amax_t.items()})
    deq = {n: (x8.float() * ds) for n, x8, ds in (("q", q8, q_ds), ("k", k8, k_ds), ("dO", do8, do_ds))}
    return _Fp8Run(
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


@requires_rubin
def test_dense():
    _run_fp8().check()


@requires_rubin
def test_causal():
    _run_fp8(causal=True).check()


@requires_rubin
def test_causal_bottom_right_rectangular():
    _run_fp8(sq=512, skv=1024, causal=True, bottom_right=True).check()


@requires_rubin
def test_sliding_window():
    _run_fp8(causal=True, left=256).check()


@requires_rubin
@pytest.mark.parametrize("hq,hkv", [(4, 2), (8, 1), (16, 1)], ids=["r2", "r8-mqa", "r16-mqa"])
def test_gqa(hq, hkv):
    _run_fp8(hq=hq, hkv=hkv, sq=256, skv=256).check()


@requires_rubin
def test_gqa_causal():
    _run_fp8(hq=8, hkv=2, sq=512, skv=512, causal=True).check()


@requires_rubin
@pytest.mark.parametrize("sq,skv", [(768, 1280), (500, 500), (257, 129)])
def test_non_tile_multiple_seqlens(sq, skv):
    """The adapter pads S_q to the q tile and S_kv to the kv block (plan Q9); the FP8 forward on Rubin declines these,
    the backward serves them."""
    _run_fp8(sq=sq, skv=skv).check()


@requires_rubin
@pytest.mark.parametrize("n_q_tiles", [1, 2, 3, 4])
def test_q_tiles_per_kv_block(n_q_tiles):
    """q tiles per kv block = 1, 2, STAGES_Q (3 on this body), STAGES_Q + 1: the three 3-deep Q / dO rings and the 2-stage
    P ring wrap at different depths (sdpa-invariants s6)."""
    _run_fp8(hq=1, sq=128 * n_q_tiles, skv=256).check()


@requires_rubin
def test_kv_blocks_with_several_tiles_in_flight():
    _run_fp8(b=2, hq=2, sq=256, skv=768).check()


@requires_rubin
def test_causal_tail_kv_block_writes_zeros_not_residue():
    """Top-left causal with S_kv > S_q: the unattended kv blocks run the forced fully-masked tile and must store exact
    zeros (dK / dV rows >= S_q), never the poison and never NaN residue."""
    sq, skv = 256, 768
    run = _run_fp8(b=2, hq=2, sq=sq, skv=skv, causal=True, poison=float("nan")).check()
    for name in ("dK", "dV"):
        tail = run.outs[0][name][:, sq:].float()
        assert torch.isfinite(tail).all() and (tail == 0).all(), f"{name}: unattended kv rows must be EXACTLY zero"


@requires_rubin
def test_two_launches_are_bitwise_and_race_free():
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
def test_workspace_is_build_time_honest():
    g, _ts = _graph_and_ports(1, 2, 256, _D, causal=False)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _ENGINE)
    g.check_support()
    g.build_plans()
    assert g.get_workspace_size() == g.get_workspace_size() > 0


# --------------------------------------------------------------------------- vs the pre-port fp8 kernel (Rubin; dumps under frost_dev/results)

_FP8_REF_STEMS = ["fp8_b1h8s1024_dense", "fp8_b1h8kv2s2048_causal", "fp8_b2h4s768x1280_dense"]


@requires_rubin
@pytest.mark.parametrize("stem", _FP8_REF_STEMS)
def test_dv_matches_the_reference_kernel_and_dq_dk_the_oracle(stem):
    """The pre-port fp8 kernel cast P UNSCALED to e4m3 and applied host-folded descales; the contract port reads descale
    tensors and scales P by ``scale_s``.  With ``scale_s = descale_s = 1`` and the dump's descales the math coincides,
    so: with bf16 gradients (when the row lists BFLOAT16 in ``out_dtypes`` -- the pre-quantization output config_sm107
    keeps for exactly this A/B) dV is BITWISE the dump's ``dv_kern`` (MHA) / within one bf16 rounding of its folded
    ``dv_red`` (GQA); with FP8 gradients only, the dequantized dV is within the fp8 recipe of the dump's real-unit dV.
    dQ / dK ride the bf16 dS workspace + the ported GEMMs (the pre-port chain lost 22-29 % of dS to an unscaled e4m3
    workspace) and are held to the fp8 recipe against the dump's fp64 oracle, not against the pre-port chain."""
    from sdpa.helpers import get_fp8_scale_factor
    from test_sdpa_bwd_dsl_sm107 import _BF16_ULP_REL as _ULP, _load_ref_dump

    ref = _load_ref_dump(stem)
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.float8_e4m3fn" and int(m["mask_flags"]) in (0, 2)
    dev = "cuda"
    bf16_grads = _BF16 in _spec().capabilities.out_dtypes
    grad_dt, cudnn_grad = (torch.bfloat16, _BF16) if bf16_grads else (_T_E4M3, _FP8)
    grad_scale = {
        n: (1.0 if bf16_grads else get_fp8_scale_factor(ref[key].abs().max().item(), _T_E4M3)) for n, key in (("dQ", "dq"), ("dK", "dk"), ("dV", "dv"))
    }
    scalars = dict(
        descale_q=m["dscale_Q"],
        descale_k=m["dscale_K"],
        descale_v=m["dscale_V"],
        descale_o=float(ref["o_dscale"]),
        descale_dO=m["dscale_dO"],
        descale_s=1.0,
        descale_dP=1.0,
        scale_s=1.0,
        scale_dQ=grad_scale["dQ"],
        scale_dK=grad_scale["dK"],
        scale_dV=grad_scale["dV"],
        scale_dP=1.0,
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
    if bf16_grads and hq == hkv:
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
    for name, key in (("dQ", "ref_dq"), ("dK", "ref_dk")):
        got = outs[name].cpu().float().permute(0, 2, 1, 3) / grad_scale[name]  # -> BHSD real units
        torch.testing.assert_close(got, ref[key].float(), **_FP8_GRAD_TOL, msg=lambda s, n=name: f"{n} vs the dump's fp64 oracle: {s}")
    a = amax["dV"].item()
    assert (
        math.isfinite(a) and abs(a - ref["dv"].abs().max().item()) <= 2 * _ULP * a + 1e-6
    ), "amax_dV is the fp32 pre-quant max (the dump's dV is that value in bf16)"


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
    dump's fp64 ``ref_ds`` (math orientation, transposed here)."""
    import cuda.bindings.driver as cuda_driver

    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3
    from test_sdpa_bwd_dsl_sm107 import _BF16_ULP_REL as _ULP, _load_kernel, _load_ref_dump

    ref = _load_ref_dump(stem)
    m = ref["meta"]
    b, hq, hkv, sq, skv = (int(m[k]) for k in ("B", "Hq", "Hkv", "Sq", "Skv"))
    assert m["storage_dtype"] == "torch.float8_e4m3fn" and int(m["mask_flags"]) in (0, 2) and sq % 128 == 0 and skv % 256 == 0
    mod = _load_kernel("fp8", dtype_qkv=DTYPE_E4M3, dtype_o=DTYPE_BF16, window_right=0 if bool(m["causal"]) else None)
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
        s1(m["dscale_Q"]), s1(m["dscale_K"]), s1(m["dscale_V"]), s1(m["dscale_dO"]), s1(1.0), s1(1.0), s1(1.0),
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
