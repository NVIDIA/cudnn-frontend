# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM89 (Ada / L20) d64 dense SDPA forward: ``sdpa_fwd_prefill_sm89``.

The row is the SAME ``sm80/prefill_f16.py`` template and the same frozen gptoss
geometry as the A100 row, served on a 99 KiB-opt-in-SMEM part.  What this file
pins, at the level of the engine row rather than any one kernel run:

  * the row is registered (opt-in) and its ``Capabilities`` box is exactly the
    validated one — d64, f16/bf16, prefill, dense, no decode, no THD, no bias,
    no sink, no padded;
  * the row's advertised ``sm_lo``/``sm_hi`` agrees with the device families
    the adapter will actually admit, so widening one without the other fails
    here rather than at a user's graph;
  * the plan-time SMEM gate declines a geometry that cannot fit the device,
    instead of failing at launch;
  * the graph-level d64 dense forward matches a torch reference on an L20, and
    the selected plan really IS the sm89 engine (a passing number from another
    engine is not evidence for this row);
  * the exclusion list is rejected by the row: d > 64, fp8, THD, decode.

Cases that need an Ada device skip elsewhere; the registration, gate-agreement
and rejection cases run on any GPU.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest
import torch

import cudnn
import cudnn.sdpa  # noqa: F401 — the capability tables live here
from cudnn.engines import MANIFEST
from frost_test_utils import requires_dsl, select_engine

from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import api_dsl as api_dsl_mod
from cudnn.sdpa.fwd import config_sm80
from cudnn.sdpa.fwd import engines as engines_fwd

_FWD_SM89 = "sdpa_fwd_prefill_sm89"
_FWD_SM80 = "sdpa_fwd_prefill_sm80"


def _spec(name=_FWD_SM89):
    return next(sp for sp in engines_fwd.ENGINE_SPECS if sp.name == name)


def _is_ada() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(torch.cuda.current_device()) == (8, 9)


requires_ada = pytest.mark.skipif(not _is_ada(), reason="needs an SM89 (Ada) GPU")


# ---------------------------------------------------------------------------
# Registration + capability box
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_sm89_row_is_registered_and_opt_in():
    row = next(r for r in MANIFEST if r.factory == "FrostSdpaFwdEngines")
    assert _FWD_SM89 in row.offered_ids()
    assert row.slots[_FWD_SM89].opt_in is True
    # A distinct slot: reusing one would alias two engines' identities.
    slots = {name: slot.slot for name, slot in row.slots.items()}
    assert slots[_FWD_SM89] != slots[_FWD_SM80]


@pytest.mark.L0
def test_sm89_capability_box_is_the_validated_one():
    caps = _spec().capabilities
    # The architecture gate is Ada EXACTLY: the row must not claim the whole
    # >= 8.9 range, because nothing on sm86/sm90 was measured.
    assert (caps.sm_lo, caps.sm_hi) == (89, 89)
    assert caps.phase == "prefill"
    # d64 exactly, with the SM80-style host-side padding rule.
    assert caps.d_shapes == frozenset({(64, 64)})
    assert caps.d_pad_multiple == 1
    assert caps.dtypes == frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})
    assert caps.out_dtypes == frozenset()
    assert caps.layouts == frozenset({"bshd", "dense_flex"})
    # Features the L20 run did NOT qualify must stay declined — inheriting the
    # SM80 row's True values wholesale is the failure mode this asserts against.
    for declined in (
        "bias",
        "padded",
        "sink",
        "thd",
        "cu_seq_len",
        "padded_stats",
        "decode",
        "single_wave_only",
        "is_fp8",
        "is_mxfp8",
        "dropout",
        "alibi",
        "block_mask",
    ):
        assert getattr(caps, declined) is False, f"{declined} must not be claimed by the SM89 row"
    # The tile geometry is the row's validated box, not a user knob.
    assert caps.tile_ms == frozenset() and caps.tile_ns == frozenset()
    assert not caps.split_kv_supported


@pytest.mark.L0
def test_sm89_row_and_adapter_device_gate_agree():
    """The row's sm range and the adapter's admitted device set are two halves of
    one fact; a patch that widens either alone must fail here."""
    caps = _spec().capabilities
    family = next(name for name, entry in api_dsl_mod._SM80_DEVICE_FAMILIES.items() if any(cc[0] * 10 + cc[1] == caps.sm_lo for cc in entry["cc"]))
    ccs = api_dsl_mod._SM80_DEVICE_FAMILIES[family]["cc"]
    assert {maj * 10 + min_ for maj, min_ in ccs} == {caps.sm_lo}, "row range and adapter gate disagree"
    assert caps.sm_hi == caps.sm_lo, "the row must not claim a range the adapter cannot enumerate"


@pytest.mark.L0
def test_sm89_flavor_box_is_d64_and_fits_ada():
    """The row lowers onto a d64 box whose compile-time SMEM fits the 99 KiB Ada
    limit — with margin, so a future tile change is a deliberate act."""
    smem = config_sm80.smem_bytes(64, 64, 128, 64)
    assert smem == (8192 + 8192) * 2, f"d64 gptoss SMEM arithmetic changed: {smem}"
    assert smem <= 99 * 1024
    # The SM80 d256 point is what makes cc 8.0 exactly the honest gate there.
    assert config_sm80.smem_bytes(256, 256, 128, 64) > 99 * 1024
    # And the SM89 view is narrower than SM80's: only the flavor Ada can hold.
    assert api_dsl_mod._SM89_FLAVOR_DIMS == {"gptoss": (64, 64)}


@pytest.mark.L0
def test_sm80_row_is_unchanged_by_the_ada_addition():
    """The A100 row keeps cc 8.0 exactly and its full feature set: this change
    adds a sibling row, it does not widen an existing gate."""
    caps = _spec(_FWD_SM80).capabilities
    assert (caps.sm_lo, caps.sm_hi) == (80, 80)
    assert caps.d_shapes == frozenset({(256, 256)})
    assert caps.bias and caps.padded and caps.sink


# ---------------------------------------------------------------------------
# Probe-level rejection (no device needed: normalize the capability)
# ---------------------------------------------------------------------------


def _facts(b=2, h=4, s=512, d=64, *, h_kv=None, thd=False, fp8=False, dtype=None, decode=False):
    """A legal SM89 facts sample, built through the analyzer so the row is
    compared against real facts rather than a hand-rolled stand-in.

    ``decode`` moves S_q to 1 and ``h_kv`` gives K/V their own head count; the
    THD and FP8 axes are patched onto the ANALYZED sample instead, because both
    need graph operands of their own (ragged offsets, amax) whose construction
    belongs to the analyzer's tests -- what is under test here is the row's
    decision on the resulting facts.
    """
    h_kv = h if h_kv is None else h_kv
    s_q = 1 if decode else s
    it = cudnn.data_type.HALF if dtype is None else dtype
    g = cudnn.pygraph(io_data_type=it, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    qs, ks = (h * s_q * d, d, h * d, 1), (h_kv * s * d, d, h_kv * d, 1)
    q = g.tensor(name="q", dim=(b, h, s_q, d), stride=qs, data_type=it)
    k = g.tensor(name="k", dim=(b, h_kv, s, d), stride=ks, data_type=it)
    v = g.tensor(name="v", dim=(b, h_kv, s, d), stride=ks, data_type=it)
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(d), generate_stats=True)
    o.set_output(True).set_dim((b, h, s_q, d)).set_stride(qs).set_data_type(it)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    facts = ga.analyze(g)
    overrides = {}
    if thd:
        # Ragged graphs carry the packed (H, S, D)-innermost layout the THD
        # lowering reads; the ragged offsets themselves are the analyzer's test.
        overrides.update(thd=True, packed_layout=True)
    if fp8:
        overrides["is_fp8"] = True
    return dataclasses.replace(facts, **overrides) if overrides else facts


@pytest.mark.L0
@pytest.mark.parametrize(
    ("label", "kwargs", "expected"),
    [
        ("d64_dense_f16", {}, None),
        ("d128_dense_f16", {"d": 128}, "no kernel-flavor envelope covers"),
        ("d192_dense_f16", {"d": 192}, "no kernel-flavor envelope covers"),
        ("gqa_h4_hkv2", {"h_kv": 2}, "GQA / MQA"),
        ("decode_sq1", {"decode": True}, "decode"),
        ("thd", {"thd": True}, "THD"),
        ("fp8", {"fp8": True}, "serves only half"),
        ("fp32_io", {"dtype": cudnn.data_type.FLOAT}, "FLOAT"),
    ],
)
def test_probe_declines_everything_the_row_does_not_claim(label, kwargs, expected):
    """The exclusion list is ENFORCED by the probe, not only documented.

    An off-box head dim must not silently land on the 128-wide flavor (the
    adapter's host-side padding covers d < 64, and d > 64 has no Ada flavor),
    and the same goes for every other axis the L20 run did not qualify: GQA/MQA,
    decode-shaped graphs, THD, FP8/MXFP8 and a non-half io dtype.
    """
    if _spec().capabilities.d_shapes != frozenset({(64, 64)}):
        pytest.skip("row changed")
    caps = _spec().capabilities
    # The probe compares device cc first; pin it to Ada so the row's own gates
    # are what is under test.
    import unittest.mock as mock

    with mock.patch.object(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)):
        reason = engines_fwd.mismatch(caps, _facts(**kwargs))
    if expected is None:
        assert reason is None, f"{label}: mismatch() said {reason!r}"
    else:
        assert reason is not None and expected in reason, f"{label}: mismatch() said {reason!r}"


# ---------------------------------------------------------------------------
# Plan-time resource gate
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_smem_gate_declines_a_geometry_that_does_not_fit(monkeypatch):
    """The gate is arithmetic on the row's box, so it can be exercised without
    an Ada part: a tile geometry whose SMEM exceeds the device limit must raise
    a diagnosable ValueError, and the validated geometry must not."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

    if not torch.cuda.is_available():
        pytest.skip("needs any CUDA device for descriptor construction")

    b, h, s, d = 1, 2, 128, 64
    dtype = torch.float16
    q = torch.empty((b, h, s, d), dtype=dtype, device="cuda")
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    o = torch.empty_like(q)
    lse = torch.empty((b, h, s), dtype=torch.float32, device="cuda")

    oversized = config_sm80.smem_bytes(256, 256, 128, 64)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: type("P", (), {"shared_memory_per_block_optin": oversized - 1})())

    api = SdpaFwdDslSm80(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o,
        sample_lse=lse,
        device_cc=((8, 9),),
        flavor_params={"flavor": "qwen", "d_qk": 256, "d_v": 256, "tile_m": 128, "tile_n": 64, "num_warps": 8},
    )
    with pytest.raises(ValueError, match="shared memory"):
        api.check_support()


@pytest.mark.L0
def test_adapter_rejects_a_device_family_the_row_did_not_declare(monkeypatch):
    """A plan built for Ada must reject an A100 (and vice versa): the declared
    set is the gate, not a hint."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

    if not torch.cuda.is_available():
        pytest.skip("needs any CUDA device")
    q = torch.empty((1, 2, 128, 64), dtype=torch.float16, device="cuda")
    api = SdpaFwdDslSm80(
        sample_q=q,
        sample_k=q.clone(),
        sample_v=q.clone(),
        sample_o=q.clone(),
        sample_lse=torch.empty((1, 2, 128), dtype=torch.float32, device="cuda"),
        device_cc=((8, 0),),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    with pytest.raises(ValueError, match="was built for cc"):
        api.check_support()


@pytest.mark.L0
def test_declared_device_set_refuses_a_flat_pair_by_name():
    """``device_cc`` is the SET of (major, minor) pairs the row was validated on.
    A flat pair is the mistake that used to reject EVERY device with "was built
    for cc [0, 8]", so it is refused where it is written."""
    from cudnn.sdpa.fwd.api_dsl import _device_cc_set

    assert _device_cc_set(None) is None
    assert _device_cc_set(((8, 9),)) == frozenset({(8, 9)})
    assert _device_cc_set([[8, 9], [8, 0]]) == frozenset({(8, 9), (8, 0)})
    with pytest.raises(ValueError, match="iterable of"):
        _device_cc_set((8, 9))
    with pytest.raises(ValueError, match="iterable of"):
        _device_cc_set(())


@pytest.mark.L0
def test_unknown_ctor_extra_is_rejected_at_lowering():
    """The row's construction extras are checked against the adapter signature,
    so a typo in a row fails loudly at build time rather than silently doing
    nothing."""
    sig = set(inspect.signature(api_dsl_mod.SdpaFwdDslSm80.__init__).parameters)
    declared = _spec().lower.keywords["api_ctor_extra"]
    assert set(declared) <= sig, f"row declares extras the adapter lacks: {set(declared) - sig}"
    assert declared["device_cc"] == ((8, 9),)
    assert declared["flavor_params"]["flavor"] == "gptoss"


# ---------------------------------------------------------------------------
# Graph-level end-to-end (needs Ada)
# ---------------------------------------------------------------------------


def _bshd(b, h, s, d):
    return (s * h * d, d, h * d, 1)


def _ref(q, k, v, causal, scale):
    qf, kf, vf = q.double(), k.double(), v.double()
    scores = (qf @ kf.transpose(-1, -2)) * scale
    if causal:
        m = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], dtype=torch.bool, device=scores.device), 1)
        scores = scores.masked_fill(m, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return (p @ vf), torch.logsumexp(scores, dim=-1)


def _run_graph(b, h, s, d, dtype, causal, pin_sm89=True):
    it = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=it, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st = _bshd(b, h, s, d)
    q = g.tensor(name="q", dim=(b, h, s, d), stride=st, data_type=it)
    k = g.tensor(name="k", dim=(b, h, s, d), stride=st, data_type=it)
    v = g.tensor(name="v", dim=(b, h, s, d), stride=st, data_type=it)
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(d), use_causal_mask=causal, generate_stats=True)
    o.set_output(True).set_dim((b, h, s, d)).set_stride(st).set_data_type(it)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((b, h, s, 1)).set_stride((h * s, s, 1, 1))

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    if pin_sm89:
        select_engine(g, _FWD_SM89)
    g.check_support()
    g.build_plans()
    return g, q, k, v, o, stats


@requires_dsl
@requires_ada
@pytest.mark.L1
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("s", [128, 256, 512, 1024], ids=lambda v: f"s{v}")
def test_sm89_d64_dense_forward_matches_fp64_reference(dtype, causal, s):
    torch.manual_seed(0)
    b, h, d = 2, 4, 64
    tt = 2.0 ** (-11) if dtype == torch.float16 else 2.0 ** (-8)
    tol = 8 * tt * math.sqrt(s)
    q = torch.randn(b, s, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    k = torch.randn(b, s, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    v = torch.randn(b, s, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)

    g, qt, kt, vt, ot, stt = _run_graph(b, h, s, d, dtype, causal)
    stt_out = torch.empty(b, h, s, 1, dtype=torch.float32, device="cuda")
    ws = g.get_workspace_size()
    workspace = torch.empty(ws, dtype=torch.uint8, device="cuda") if ws else None
    # The variant pack is keyed by the IR tensors the graph returned.
    # Bind the BSHD-physical buffers the graph declared, like every existing
    # SDPA case: a compact (B,H,S,D) buffer has different strides than the
    # declared view and would silently mis-address the epilogue store.
    o = torch.empty(b, s, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    g.execute({qt: q, kt: k, vt: v, ot: o, stt: stt_out}, workspace)
    torch.cuda.synchronize()

    o_ref, lse_ref = _ref(q, k, v, causal, 1.0 / math.sqrt(d))
    err = (o.double() - o_ref).abs().max().item()
    scale_ref = max(o_ref.abs().max().item(), 1.0)
    assert err <= tol * scale_ref, f"O max|err|={err:.3e} > {tol * scale_ref:.3e}"
    lse_err = (stt_out.squeeze(-1).double() - lse_ref).abs().max().item()
    assert lse_err <= 4e-3, f"Stats max|err|={lse_err:.3e}"
    assert torch.isfinite(o).all() and torch.isfinite(stt_out).all()


@requires_dsl
@requires_ada
@pytest.mark.L1
def test_sm89_selected_plan_is_the_sm89_engine():
    """The numbers above are only evidence for this row if the plan really is it."""
    b, h, s, d = 1, 2, 256, 64
    q = torch.randn(b, h, s, d, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    it = cudnn.data_type.HALF
    g = cudnn.pygraph(io_data_type=it, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    st = _bshd(b, h, s, d)
    qt = g.tensor(name="q", dim=(b, h, s, d), stride=st, data_type=it)
    kt = g.tensor(name="k", dim=(b, h, s, d), stride=st, data_type=it)
    vt = g.tensor(name="v", dim=(b, h, s, d), stride=st, data_type=it)
    o, stats = g.sdpa(q=qt, k=kt, v=vt, attn_scale=1.0 / math.sqrt(d), generate_stats=True)
    o.set_output(True).set_dim((b, h, s, d)).set_stride(st).set_data_type(it)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    assert any(_FWD_SM89 in n for n in names), f"the sm89 row proposed no plan; plans={names}"
    # And the SM80 row must NOT be eligible on Ada even though the box overlaps.
    assert not any(n.startswith(_FWD_SM80) for n in names), f"the A100 row proposed a plan on Ada: {names}"


# ---------------------------------------------------------------------------
# Mask families (Ada)
#
# The reference for every case is an explicit boolean mask built here, so a case
# passes only when the kernel applies the mask this test says it applies.  The
# sliding-window wording is the MEASURED one: `sliding_window_length=W` keeps the
# W keys ending at self (fitted from the kernel; forbidding `i - j >= W` scored
# 4.3e-4 while the "keys left of self" reading scored 1.04).
# ---------------------------------------------------------------------------


def _mask_ref(q, k, v, mask, scale):
    scores = (q.double() @ k.double().transpose(-1, -2)) * scale
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v.double(), torch.logsumexp(scores, dim=-1)


def _run_masked(b, h, sq, skv, d, dtype, kwargs, mask, tol_scale):
    """Build/execute one masked graph and return (max|dO|, max|dLSE|) vs the mask."""
    it = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=it, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    qs = _bshd(b, h, sq, d)
    ks = _bshd(b, h, skv, d)
    qt = g.tensor(name="q", dim=(b, h, sq, d), stride=qs, data_type=it)
    kt = g.tensor(name="k", dim=(b, h, skv, d), stride=ks, data_type=it)
    vt = g.tensor(name="v", dim=(b, h, skv, d), stride=ks, data_type=it)
    o, stats = g.sdpa(q=qt, k=kt, v=vt, attn_scale=1.0 / math.sqrt(d), generate_stats=True, **kwargs)
    o.set_output(True).set_dim((b, h, sq, d)).set_stride(qs).set_data_type(it)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((b, h, sq, 1)).set_stride((h * sq, sq, 1, 1))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    select_engine(g, _FWD_SM89)
    g.check_support()
    g.build_plans()

    torch.manual_seed(0)
    qb = torch.randn(b, sq, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    kb = torch.randn(b, skv, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    vb = torch.randn(b, skv, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    ob = torch.empty(b, sq, h, d, dtype=dtype, device="cuda").permute(0, 2, 1, 3)
    st = torch.empty(b, h, sq, 1, dtype=torch.float32, device="cuda")
    ws = g.get_workspace_size()
    workspace = torch.empty(ws, dtype=torch.uint8, device="cuda") if ws else None
    g.execute({qt: qb, kt: kb, vt: vb, o: ob, stats: st}, workspace)
    torch.cuda.synchronize()
    o_ref, lse_ref = _mask_ref(qb, kb, vb, mask, 1.0 / math.sqrt(d))
    return (ob.double() - o_ref).abs().max().item(), (st.squeeze(-1).double() - lse_ref).abs().max().item()


def _idx(sq, skv):
    i = torch.arange(sq, device="cuda").view(-1, 1)
    j = torch.arange(skv, device="cuda").view(1, -1)
    return i, j


@requires_dsl
@requires_ada
@pytest.mark.L1
@pytest.mark.parametrize(
    ("label", "kwargs", "mask_fn"),
    [
        ("unmasked", {}, lambda sq, skv: torch.zeros(sq, skv, dtype=torch.bool, device="cuda")),
        ("top_left_causal", {"use_causal_mask": True}, lambda sq, skv: _idx(sq, skv)[1] > _idx(sq, skv)[0]),
        ("sliding_window_w64", {"sliding_window_length": 64}, lambda sq, skv: (_idx(sq, skv)[0] - _idx(sq, skv)[1]) >= 64),
        (
            "sliding_window_w64_causal",
            {"use_causal_mask": True, "sliding_window_length": 64},
            lambda sq, skv: ((_idx(sq, skv)[0] - _idx(sq, skv)[1]) >= 64) | (_idx(sq, skv)[1] > _idx(sq, skv)[0]),
        ),
    ],
)
def test_sm89_mask_families_match_explicit_boolean_masks(label, kwargs, mask_fn):
    """One case per mask family, each against its own hand-built mask.

    Bottom-right causal is exercised with `S_q < S_kv` on purpose: on a square
    graph the two anchors are the same mask, so a square case would prove
    nothing about the anchor.
    """
    b, h, d = 2, 4, 64
    # Square here on purpose: the bottom-right anchor is a REJECTION case for
    # this row (test_sm89_declines_rectangular_bottom_right_causal), because a
    # rectangular graph is exactly where the two anchors differ.
    sq, skv = 512, 512
    dtype = torch.float16
    err_o, err_lse = _run_masked(b, h, sq, skv, d, dtype, kwargs, mask_fn(sq, skv), 1.0)
    budget = 8 * 2.0 ** (-11) * math.sqrt(skv)
    assert err_o <= budget, f"{label}: max|dO|={err_o:.3e} > {budget:.3e}"
    assert err_lse <= 4e-3, f"{label}: max|dLSE|={err_lse:.3e}"


@requires_dsl
@requires_ada
@pytest.mark.L1
def test_sm89_row_claims_exactly_its_measured_band():
    """`diagonal_band_right_bound` is accepted and IGNORED by this kernel, so the
    row must not advertise right-band widening.  Pinning the negative here keeps
    the row and the support matrix from drifting apart in the permissive
    direction."""
    caps = _spec().capabilities
    # Post-#601 the canonical claim is what the probe decides with; the legacy
    # flags are the spelling that derived it, and the two must state one claim.
    from cudnn.sdpa import band as _band

    assert caps.right_band_widening is False
    assert caps.swa is True and caps.causal is True
    assert caps.band == _band.BandSupport(
        right=frozenset({_band.RIGHT_UNBOUNDED, _band.RIGHT_CAUSAL}),
        left=frozenset({_band.LEFT_NONE, _band.LEFT_WINDOW}),
        anchors=frozenset({_band.ANCHOR_TOP_LEFT}),
    )
    # The anchor restriction is the whole point of the explicit band spelling:
    # the flags could not express it.
    assert _band.ANCHOR_BOTTOM_RIGHT not in caps.band.anchors
    assert caps.bottom_right is False
    # ... and the head-count claim is enforced for the same reason: the tracker
    # marks GQA/MQA unsupported, so the probe has to decline it (the L20 run
    # covered the equal-head layout only).
    import unittest.mock as mock

    with mock.patch.object(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9)):
        assert "GQA / MQA" in (engines_fwd.mismatch(caps, _facts(h_kv=2)) or "")


@requires_dsl
@requires_ada
@pytest.mark.L0
def test_sm89_declines_rectangular_bottom_right_causal():
    """A rectangular bottom-right-causal graph must NOT land on this row.

    This is the case where the anchor actually differs from top-left, so it is
    the one that evidences the restriction.  The graph is still served — by the
    BACKEND — so this asserts on the ROW's absence from the plan list, not on an
    exception.
    """
    b, h, sq, skv, d = 2, 4, 256, 512, 64
    it = cudnn.data_type.HALF
    g = cudnn.pygraph(io_data_type=it, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    qs, ks = _bshd(b, h, sq, d), _bshd(b, h, skv, d)
    q = g.tensor(name="q", dim=(b, h, sq, d), stride=qs, data_type=it)
    k = g.tensor(name="k", dim=(b, h, skv, d), stride=ks, data_type=it)
    v = g.tensor(name="v", dim=(b, h, skv, d), stride=ks, data_type=it)
    # generate_stats is mandatory: without it the BACKEND refuses the graph
    # ("Exactly one of {generate_stats, is_inference} must be set"), and then
    # "the row declined" would be untestable because nobody would propose a plan.
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=1.0 / math.sqrt(d), generate_stats=True, use_causal_mask_bottom_right=True)
    o.set_output(True).set_dim((b, h, sq, d)).set_stride(qs).set_data_type(it)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim((b, h, sq, 1)).set_stride((h * sq, sq, 1, 1))
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    assert not any(n.startswith(_FWD_SM89) for n in names), f"the row must decline this graph; plans={names}"
    assert names, "the backend must still serve it (a decline is not a failure)"
