# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm107`` THD / varlen: the packed d = 256 bf16 / fp16 backward on the Rubin line, end to end.

Two surfaces, deliberately both (the SM100 THD suite's shape):

* ``_run`` drives ``SdpaBwdDslSm107`` DIRECTLY -- the numerics of the two-kernel chain over PACKED input and the kv-BLOCKED
  dS workspace, one layer below the plan machinery, so a failure localises to the kernels.
* ``_run_graph`` goes through the ragged GRAPH and pins the engine -- the lowering: packed views over the caller's
  buffers, the length binding, the Stats packing inference.  A pass there with a decline underneath would be a cuDNN
  plan, which is why the engine is pinned by name.

The reference is per-sequence dense attention in fp64 on the storage-rounded operands: unpack, run each sequence on its
own, compare gradient by gradient under the sm107 suite's bounds (``_TOL_COS`` / ``_TOL[dt]`` -- the dense row's, never a
new looser one).  A THD bug that leaks across a sequence boundary shows up as one sequence's gradient contaminated by its
neighbour's, which a whole-tensor cosine would average away -- so every assertion is per sequence.  The GPU cases carry
``requires_rubin``; the rejects run off the Rubin line (the analyzer's cc faked to 10.7, as the dense suite does): the direct
adapter rejects over ``TensorDesc`` samples with no device at all, the graph probes (``_thd_mismatch``) on any CUDA device the
backend serves -- the frontend lowers the graph on the ambient device -- behind the suite's SM80 floor, a clean skip without one.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin, requires_sm80
from test_sdpa_bwd_dsl_sm107 import _TOL, _TOL_COS, _adapter

import cudnn

pytestmark = [pytest.mark.L0, requires_dsl]

_ENGINE = "sdpa_bwd_sm107"
_D = 256
_RUBIN_CC = (10, 7)


@pytest.fixture(autouse=True)
def _mock_target_for_cross_arch_contracts(monkeypatch):
    # The reject tests probe the Rubin row off the Rubin line (the analyzer's cc faked to 10.7); mismatch() carries the sm_107a
    # DSL gate, so the fake device gets a fake compiler target too -- exactly as the dense sm107 suite does.
    from cudnn.frost import buffers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != _RUBIN_CC:
        monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)


# --------------------------------------------------------------------------- the per-sequence fp64 oracle


def _causal_bias(s_q, s_kv, device, bottom_right=False, window_left=None):
    """Additive -inf mask for ONE sequence, in that sequence's OWN geometry: under THD the diagonal is per sequence, so a
    mask built from the envelope ``S_max`` or the packed total would be a different mask for every sequence but the
    longest.  ``bottom_right`` aligns the diagonal to the last row (offset ``S_kv[b] - S_q[b]``); ``window_left`` is the
    graph's ``sliding_window_length`` = the number of keys a row keeps: ``kv > q + diag - window`` (the analyzer hands the
    rows ``window_left = length - 1`` and the kernel keeps ``kv >= q + diag - window_left`` -- the same set)."""
    q_i = torch.arange(s_q, device=device).view(-1, 1)
    kv_i = torch.arange(s_kv, device=device).view(1, -1)
    diag = (s_kv - s_q) if bottom_right else 0
    keep = kv_i <= q_i + diag
    if window_left is not None:
        keep = keep & (kv_i > q_i + diag - window_left)
    return torch.where(keep, 0.0, float("-inf")).double()


def _ref_bwd(q, k, v, do, scale, causal=False, bottom_right=False, window_left=None):
    """fp64 reference backward for ONE sequence ([H, S, D] operands, K / V already broadcast to the Q heads)."""
    q, k, v, do = (t.detach().double().requires_grad_(t is not do) for t in (q, k, v, do))
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        s = s + _causal_bias(q.shape[-2], k.shape[-2], q.device, bottom_right, window_left)
    p = torch.softmax(s, dim=-1)
    o = p @ v
    o.backward(do)
    return q.grad, k.grad, v.grad


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _grad_failure(name, got, ref, dt):
    """The dense sm107 suite's ``_check`` as a verdict string (None = pass): finite, cosine above ``_TOL_COS``, and
    ``torch.testing.assert_close`` under ``_TOL[dt]`` -- the same bounds the dense row holds, applied per sequence."""
    if not torch.isfinite(got.float()).all():
        return f"{name}: non-finite output"
    if not ref.bool().any():
        return None if (got.float() == 0).all() else f"{name}: the reference is identically zero, got max|{got.float().abs().max().item():.3e}|"
    cos = _cos(got, ref)
    if not (cos > _TOL_COS):  # `not (x > tol)`: a NaN cosine must FAIL, not slip through a `<=`
        diff = (got.double() - ref.double()).abs()
        return f"{name}: cos={cos:.6f} (max|diff|={diff.max().item():.3e} at |ref|max={ref.abs().max().item():.3e})"
    try:
        torch.testing.assert_close(got.float(), ref.float(), **_TOL[dt])
    except AssertionError as exc:
        return f"{name} vs the fp64 oracle: {str(exc).splitlines()[0]}"
    return None


# --------------------------------------------------------------------------- the packed case


def _thd_case(
    lens_q, lens_kv, h, d, dtype, cap_q=None, cap_kv=None, poison=False, seed=7, causal=False, bottom_right=False, window_left=None, hkv=None, device="cuda"
):
    """Packed Q/K/V/dO plus the forward's O and packed LSE, per sequence in fp64 on the storage-rounded operands.

    ``cap_*`` over-allocates the packed buffers past the real totals; with ``poison`` the slack is NaN -- the declared
    totals only bound the buffers, so the rows between the current ``cu_*[B]`` and the capacity have to be kept out of
    reach by the kernels' OWN device-side descriptor clamps (``0 * NaN`` would otherwise poison whole sequences).
    Unit-normal inputs on a CPU generator (the dataset does not depend on the GPU's SM count).  ``device`` is where the
    buffers live: ``"cuda"`` for the kernels, ``"cpu"`` for a host-side graph probe that needs only the geometry.
    """
    dev, b = device, len(lens_q)
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cap_q, cap_kv = cap_q or t_q, cap_kv or t_kv
    cu_q, cu_k = [0], [0]
    for a, c in zip(lens_q, lens_kv):
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + c)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hkv = h if hkv is None else hkv
    grp = h // hkv

    def _pk(cap, live, nh):
        x = torch.randn(1, cap, nh, d, generator=g).to(device=dev, dtype=dtype)
        if poison and cap > live:
            x[0, live:] = float("nan")
        return x

    q_p, do_p = _pk(cap_q, t_q, h), _pk(cap_q, t_q, h)
    k_p, v_p = _pk(cap_kv, t_kv, hkv), _pk(cap_kv, t_kv, hkv)
    o_p = torch.full_like(q_p, float("nan") if poison else 0.0)
    lse_p = torch.full((1, h, cap_q), float("nan") if poison else 0.0, device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    for i in range(b):
        if lens_q[i] == 0 or lens_kv[i] == 0:
            continue
        qs, ks, vs = (
            x[0, cu[i] : cu[i] + L].transpose(0, 1).double() for x, cu, L in ((q_p, cu_q, lens_q[i]), (k_p, cu_k, lens_kv[i]), (v_p, cu_k, lens_kv[i]))
        )
        # GQA: one KV head feeds `grp` CONSECUTIVE Q heads (kv_head = q_head // grp) -- repeat_interleave, not repeat.
        if grp > 1:
            ks, vs = ks.repeat_interleave(grp, dim=0), vs.repeat_interleave(grp, dim=0)
        sc = (qs @ ks.transpose(-1, -2)) * scale
        if causal:
            sc = sc + _causal_bias(lens_q[i], lens_kv[i], dev, bottom_right, window_left)
        lse_p[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = torch.logsumexp(sc, dim=-1).float()
        o_p[0, cu_q[i] : cu_q[i] + lens_q[i]] = (torch.softmax(sc, dim=-1) @ vs).transpose(0, 1).to(dtype)
    return SimpleNamespace(
        b=b, h=h, hkv=hkv, d=d, dtype=dtype, scale=scale, causal=causal, bottom_right=bottom_right, window_left=window_left,
        lens_q=list(lens_q), lens_kv=list(lens_kv), cu_q=cu_q, cu_k=cu_k, t_q=t_q, t_kv=t_kv, cap_q=cap_q, cap_kv=cap_kv,
        q=q_p, k=k_p, v=v_p, do=do_p, o=o_p, lse=lse_p,
    )  # fmt: skip


def _check(case, dq, dk, dv):
    """Per-sequence comparison against the fp64 reference, COLLECTED and asserted at the end: which of the three a
    sequence gets wrong is the attribution (dK and dQ come from the two stage-3 renderings, dV from the main kernel; all
    three read what stage 2 wrote).  A sequence empty on either side has no reference and is checked by its caller."""
    bad, verdicts = [], []
    for i in range(case.b):
        if case.lens_q[i] == 0 or case.lens_kv[i] == 0:
            continue
        sl_q = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
        sl_k = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
        grp = case.h // case.hkv
        rep = (lambda x: x.repeat_interleave(grp, dim=0)) if grp > 1 else (lambda x: x)
        rq, rk, rv = _ref_bwd(
            case.q[0, sl_q].transpose(0, 1),
            rep(case.k[0, sl_k].transpose(0, 1)),
            rep(case.v[0, sl_k].transpose(0, 1)),
            case.do[0, sl_q].transpose(0, 1),
            case.scale,
            causal=case.causal,
            bottom_right=case.bottom_right,
            window_left=case.window_left,
        )
        if grp > 1:
            # The reference ran with K / V broadcast to every Q head; the kernel returns dK / dV folded onto the KV heads.
            rk = rk.reshape(case.hkv, grp, *rk.shape[1:]).sum(1)
            rv = rv.reshape(case.hkv, grp, *rv.shape[1:]).sum(1)
        for name, got, want in (("dQ", dq[0, sl_q].transpose(0, 1), rq), ("dK", dk[0, sl_k].transpose(0, 1), rk), ("dV", dv[0, sl_k].transpose(0, 1), rv)):
            verdict = _grad_failure(f"seq {i} {name} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})", got, want, case.dtype)
            verdicts.append(verdict or f"seq {i} {name}: ok")
            if verdict:
                bad.append(verdict)
    assert not bad, "\n".join(verdicts)


def _assert_empty_sequence_exactly_zero(case, dq, dk, dv, i):
    sl_q = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
    sl_k = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
    for name, got in (("dQ", dq[0, sl_q]), ("dK", dk[0, sl_k]), ("dV", dv[0, sl_k])):
        got = got.float()  # the fp8 twin hands e4m3 gradients (no .any() / .abs() on that dtype); exact zero is dtype-independent
        assert got.numel() == 0 or not got.any(), f"{name} of a one-sided-empty sequence must be exactly zero, got max |{got.abs().max().item()}|"


# A FINITE sentinel for the gradient rows past the packed totals, exactly representable in bf16 / fp16 and never a value the
# chain produces over a whole tail.  Not NaN: the one way the tail used to be written -- the GQA fold copying the never-written
# (0xFF-poisoned) partials past cu_k[B] -- writes NaN, which a NaN fill could not tell from "untouched".
_TAIL_SENTINEL = -7.0


def _sentinel_tails(case, dq, dk, dv):
    """Fill the rows PAST the packed totals (the declared capacity the chain may not touch) of all three gradients."""
    for x, live in ((dq, case.t_q), (dk, case.t_kv), (dv, case.t_kv)):
        x[0, live:] = _TAIL_SENTINEL


def _assert_tails_untouched(case, dq, dk, dv):
    """Nothing past the packed total is written into the caller's gradients: dQ / dK stop at the per-sequence clipped output
    descriptors, dV at the kernel's per-sequence descriptors, and the GQA fold at the live kv total ``cu_k[B]`` on device."""
    for name, x, live in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
        tail = x[0, live:]
        written = int((tail != _TAIL_SENTINEL).sum()) if tail.numel() else 0
        assert written == 0, f"{name}: {written} of {tail.numel()} elements past the packed total ({live} rows) were written"


# --------------------------------------------------------------------------- the direct adapter surface


def _envelope_samples(case):
    """The adapter's SAMPLES declare the ENVELOPE (B, H, S_max, D) the way a ragged graph does; the packed buffers only show
    up at execute.  Declaring the packed shape instead makes batch_size 1 and the chain runs one sequence."""
    dev, b, h, hkv, d = "cuda", case.b, case.h, case.hkv, case.d
    s_max_q, s_max_kv = max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
    env = lambda n, s, nh: torch.empty(1, n, s, nh, d, device=dev, dtype=case.dtype)[0].permute(0, 2, 1, 3)  # noqa: E731
    eq, ekv = env(b, s_max_q, h), env(b, s_max_kv, hkv)
    e_stats = torch.empty(b, h, s_max_q, 1, device=dev, dtype=torch.float32)
    return eq, ekv, e_stats


def _run(
    lens_q,
    lens_kv,
    h=2,
    hkv=None,
    d=_D,
    dtype=torch.bfloat16,
    token_major_stats=False,
    causal=False,
    bottom_right=False,
    window_left=None,
    poison_outputs=False,
):
    """Build the case, drive ``SdpaBwdDslSm107(thd=True)`` directly on PACKED views, compare per sequence.  ``window_left`` is
    the graph's ``sliding_window_length`` (keys per row); the adapter takes the analyzer's ``window_size_left = length - 1``.
    Returns the case and the gradients for the caller's extra assertions."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    case = _thd_case(lens_q, lens_kv, h, d, dtype, causal=causal, bottom_right=bottom_right, window_left=window_left, hkv=hkv)
    dev = "cuda"
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731  [1,T,H,D] -> logical [1,H,T,D], the dense path's orientation
    eq, ekv, e_stats = _envelope_samples(case)
    fill = float("nan") if poison_outputs else 0.0
    dq, dk, dv = (torch.full_like(case.q, fill), torch.full_like(case.k, fill), torch.full_like(case.v, fill))
    # TRANSPOSED, not reshaped: lse is head-major [1, H, T]; a reshape to (T, H) would reinterpret the memory.
    stats = case.lse[0].transpose(0, 1).contiguous() if token_major_stats else case.lse
    api = SdpaBwdDslSm107(
        sample_q=eq,
        sample_k=ekv,
        sample_v=ekv,
        sample_o=eq,
        sample_do=eq,
        sample_stats=e_stats,
        sample_dq=eq,
        sample_dk=ekv,
        sample_dv=ekv,
        scale_softmax=case.scale,
        is_causal=causal,
        causal_bottom_right=bottom_right,
        window_size_left=None if window_left is None else window_left - 1,
        thd=True,
        max_total_seq_len_q=case.t_q,
        max_total_seq_len_kv=case.t_kv,
        thd_stats_token_major=token_major_stats,
    )
    assert api.check_support()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    ws.fill_(0xFF)  # NaN in every dtype the chain stores: a stage reading a scratch region before writing it surfaces as NaN
    api.execute(
        view(case.q),
        view(case.k),
        view(case.v),
        view(case.o),
        view(case.do),
        stats,
        view(dq),
        view(dk),
        view(dv),
        workspace=ws,
        seq_q_lens=torch.tensor(lens_q, dtype=torch.int32, device=dev),
        seq_kv_lens=torch.tensor(lens_kv, dtype=torch.int32, device=dev),
    )
    torch.cuda.synchronize()
    _check(case, dq, dk, dv)
    return case, dq, dk, dv


@requires_rubin
@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_thd_self_attention(dt):
    """Three sequences of unequal length, none a tile multiple."""
    _run((300, 128, 200), (300, 128, 200), dtype=dt)


@requires_rubin
def test_thd_cross_attention():
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run((256, 100), (180, 300))


@requires_rubin
def test_thd_single_sequence_matches_dense_shape():
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run((512,), (512,))


@requires_rubin
def test_thd_stats_token_major():
    """The other packed Stats layout the forward can emit."""
    _run((300, 128), (300, 128), token_major_stats=True)


@requires_rubin
def test_thd_dead_units_run_one_masked_tile():
    """FEWER live units than clusters (one 256-row kv block of one head on an occupancy-sized grid): every other cluster's
    first unit is past the device live total and runs ONE forced fully-masked tile.  Poisoned outputs and workspace: a dead
    unit that stored anything into a live row, or a live unit that skipped a store, surfaces as NaN or a wrong sequence."""
    _run((256,), (256,), h=1, poison_outputs=True)


@requires_rubin
def test_thd_causal_direct():
    """Top-left causal on the direct surface: the masked tiles the kernel never writes meet stage 3's per-sequence K-trim over a
    workspace this driver NaN-poisons and the chain never zero-fills -- a read past the band would surface as NaN."""
    _run((300, 128, 200), (300, 128, 200), causal=True)


def _thd_adapter(cls=None, b=2, h=2, s_max=256, dt=torch.bfloat16, **kw):
    """A THD adapter over the ENVELOPE (B, H, S_max, D) as ``TensorDesc``s -- the dense suite's ``_adapter``, so no buffer and no
    device: the host rejects below check shapes, dtypes, strides and plan facts, and nothing executes.  ``thd=True`` is set; the
    declared totals and the other plan facts ride in ``kw``."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    return _adapter(SdpaBwdDslSm107 if cls is None else cls, b=b, hq=h, sq=s_max, skv=s_max, dt=dt, thd=True, **kw)


def test_thd_requires_declared_totals():
    """THD without ``max_total_seq_len_*`` is DECLINED, not silently mis-sized: the kv-blocked workspace's row count, delta's
    row stride and the GQA partials are fixed at build time from the packed token capacity; undeclared, that capacity
    falls back to B * S_max -- more tokens than a packed buffer holds."""
    with pytest.raises(ValueError, match="max_total_seq_len"):
        _thd_adapter().check_support()
    assert _thd_adapter(max_total_seq_len_q=400, max_total_seq_len_kv=400).check_support()


def test_thd_refuses_the_dense_length_flags_and_the_quantized_rows():
    """THD carries its lengths in the metadata buffer: ``seq_kv_lens_present`` / ``seq_q_lens_present`` with THD are refused
    (two sources of truth drift apart) -- a ValueError naming the reason, before anything compiles.  The fp8 adapter ACCEPTS THD
    with declared totals now (``test_sdpa_bwd_thd_fp8_sm107.py`` is its suite; inverted from the outright refusal it used to pin);
    the MXFP8 adapter still refuses it outright (its per-sequence scale-factor pads are the next wave)."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107Fp8
    from test_sdpa_bwd_mxfp8_sm107 import _mxfp8_adapter

    totals = dict(max_total_seq_len_q=400, max_total_seq_len_kv=400)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _thd_adapter(seq_kv_lens_present=True, **totals).check_support()
    assert _thd_adapter(SdpaBwdDslSm107Fp8, dt=torch.float8_e4m3fn, **totals).check_support(), "the fp8 row serves THD"
    with pytest.raises(ValueError, match="mutually exclusive"):
        _thd_adapter(SdpaBwdDslSm107Fp8, dt=torch.float8_e4m3fn, seq_kv_lens_present=True, **totals).check_support()
    with pytest.raises(ValueError, match="THD / ragged is not implemented on this row"):
        _mxfp8_adapter(thd=True, **totals).check_support()


def test_thd_plan_facts_are_in_the_adapter_constructors_own_signature():
    """The engines' lowering forwards a plan-time fact to the adapter only when it appears in the constructor's OWN signature
    (``lower_dsl_bwd`` filters by ``inspect.signature(adapter_cls.__init__)``; a bare ``**kwargs`` hides the base constructor's
    parameters).  The half row's constructor re-declares the THD facts -- ``thd``, the declared packed totals, the packed Stats
    packing -- next to ``external_delta``, so a ragged graph reaches ``check_support`` as a THD plan.  Without this pin the symptom
    is a ragged graph refused as ``stats must be contiguous (B, H_q, S_q, 1)``: the dense Stats check of a plan built without
    ``thd=True``.  The fp8 row serves THD, so its constructor carries the facts too (inverted from the exemption it used to pin);
    the MXFP8 row declines THD at eligibility until its wave lands, so its constructor need not."""
    import inspect

    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDsl
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107, SdpaBwdDslSm107Fp8

    base = inspect.signature(SdpaBwdDsl.__init__).parameters
    for cls in (SdpaBwdDslSm107, SdpaBwdDslSm107Fp8):
        own = inspect.signature(cls.__init__).parameters
        for name in ("thd", "max_total_seq_len_q", "max_total_seq_len_kv", "thd_stats_token_major", "thd_stats_head_stride"):
            assert name in own and name in base, f"{name} must be in {cls.__name__}'s own constructor signature (the lowering forwards only those)"
            assert own[name].default == base[name].default, name
        assert "external_delta" in own


def test_thd_declines_the_external_delta():
    """The externally computed delta (``SdpaBwdDslSm107(external_delta=True)``, the dense standalone surface: a contiguous
    ``[B, H_q, S_q_pad]`` fp32 ``rowsum(dO * O)``) is DECLINED under THD, typed and before any plan is built: the THD chain's delta
    is its own ``dot_do_o`` over the packed O / dO in the head-major ``[1, H_q, ceil128(T_q)]`` layout the THD main kernel reads,
    and no producer emits that packed layout.  The THD plan's roles carry no delta slot and no standalone-only role, the THD carve
    keeps its own delta region, and a ``delta_tensor`` handed to a THD plan's execute is refused by the plan-fact check
    (``external_delta=False``) -- never silently ignored."""
    from cudnn.sdpa.bwd.prepared_sm107 import ATTRIBUTES_F16_THD, EXTERNAL_DELTA_ROLE, ROLES_F16, ROLES_F16_THD

    totals = dict(max_total_seq_len_q=400, max_total_seq_len_kv=400)
    with pytest.raises(ValueError, match="external_delta is not served on the packed chain"):
        _thd_adapter(external_delta=True, **totals).check_support()
    api = _thd_adapter(**totals)
    assert api.check_support() and api.external_delta is False
    assert "delta" in [name for name, _n, _d in api._scratch_plan()], "the THD carve keeps the chain's own delta region"
    assert EXTERNAL_DELTA_ROLE in ROLES_F16 and EXTERNAL_DELTA_ROLE not in ROLES_F16_THD and EXTERNAL_DELTA_ROLE not in ATTRIBUTES_F16_THD
    with pytest.raises(ValueError, match="external_delta=False"):
        api._check_external_delta(torch.zeros(1, 2, 128))  # refused for being given at all, before any device read


def test_thd_execute_requires_both_lengths(monkeypatch):
    """A THD plan's execute needs ``seq_q_lens`` AND ``seq_kv_lens`` (the setup launch builds the metadata from them) -- refused
    typed before compile."""
    api = _thd_adapter(max_total_seq_len_q=400, max_total_seq_len_kv=400)
    monkeypatch.setattr(api, "compile", lambda: pytest.fail("refused before compile"))
    lens = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(ValueError, match="seq_q_lens AND seq_kv_lens"):
        api.execute(*([None] * 9), seq_kv_lens=lens)
    with pytest.raises(ValueError, match="seq_q_lens AND seq_kv_lens"):
        api.execute(*([None] * 9), seq_q_lens=lens)


# --------------------------------------------------------------------------- the graph path: ragged tensors -> lower_dsl_bwd -> packed views


def _plan_index(g, name=_ENGINE):
    for i in range(g.get_execution_plan_count()):
        if name in g.get_plan_name_at_index(i):
            return i
    return None


def _build_thd_bwd_graph(case, *, stats_layout="head_major", declare_totals=True, hkv=None, envelope_q=None, **sdpa_kwargs):
    """A ragged backward graph over ``case``'s packed buffers: everything declared as the ENVELOPE (B, H, S_max, D) plus a
    per-tensor ragged offset -- how cuDNN spells a packed tensor (the packed totals never appear as a dim, which is why the
    lowering cannot reinterpret a variant-pack buffer through the port geometry).  The envelope is the longest sequence
    unless ``envelope_q`` names it: a batch with no query anywhere would otherwise declare S_q = 1, a decode shape no
    prefill row serves, while a real caller declares the batch's capacity."""
    b, h, d, dev = case.b, case.h, case.d, case.q.device
    hkv = h if hkv is None else hkv
    io = cudnn.data_type.HALF if case.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    s_max_q, s_max_kv = envelope_q or max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
    st_q = [s_max_q * h * d, d, h * d, 1]
    st_kv = [s_max_kv * hkv * d, d, hkv * d, 1]
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    ro_q_t = (torch.tensor(case.cu_q, dtype=torch.int64, device=dev) * (h * d)).view(b + 1, 1, 1, 1)
    ro_k_t = (torch.tensor(case.cu_k, dtype=torch.int64, device=dev) * (hkv * d)).view(b + 1, 1, 1, 1)
    vp, t = {}, {}

    def _ragged(name, s_max, stride, nh, ro_t):
        x = g.tensor(name=name, dim=[b, nh, s_max, d], stride=stride, data_type=io)
        ro = g.tensor(name=f"{name}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        x.set_ragged_offset(ro)
        vp[ro] = ro_t
        return x

    for n in ("q", "o", "do"):
        t[n] = _ragged(n, s_max_q, st_q, h, ro_q_t)
    for n in ("k", "v"):
        t[n] = _ragged(n, s_max_kv, st_kv, hkv, ro_k_t)
    # Packed Stats in one of the two layouts the forward emits.  head_major is (1, QH, head_stride) with a token capacity
    # rounded up to 64 -- WIDER than the packed total (sized on the DECLARED capacity: the adapter binds the Stats view at
    # min(B * S_max, declared), so the head stride has to cover that).
    t_cap = max(64, -(-case.cap_q // 64) * 64)
    if stats_layout == "head_major":
        stats_stride = [h * t_cap, t_cap, 1, 1]
        stats_stor = torch.zeros(h * t_cap, dtype=torch.float32, device=dev)
        stats_stor.as_strided((1, h, case.cap_q), (h * t_cap, t_cap, 1)).copy_(case.lse[:, :, : case.cap_q])
        stats_ro_t = torch.tensor(case.cu_q, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1)
    else:
        stats_stride = [s_max_q * h, 1, h, 1]
        stats_stor = case.lse[0].transpose(0, 1).contiguous().reshape(-1)
        stats_ro_t = (torch.tensor(case.cu_q, dtype=torch.int64, device=dev) * h).view(b + 1, 1, 1, 1)
    st = g.tensor(name="stats", dim=[b, h, s_max_q, 1], stride=stats_stride, data_type=cudnn.data_type.FLOAT)
    st_ro = g.tensor(name="stats_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
    st.set_ragged_offset(st_ro)
    vp[st_ro] = stats_ro_t
    vp[st] = stats_stor
    slq = torch.tensor(case.lens_q, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    slk = torch.tensor(case.lens_kv, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    tq_len = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    tk_len = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    vp[tq_len], vp[tk_len] = slq, slk
    kw = dict(
        name="bwd",
        q=t["q"],
        k=t["k"],
        v=t["v"],
        o=t["o"],
        dO=t["do"],
        stats=st,
        attn_scale=case.scale,
        use_padding_mask=True,
        seq_len_q=tq_len,
        seq_len_kv=tk_len,
    )
    if declare_totals:
        kw.update(max_total_seq_len_q=case.cap_q, max_total_seq_len_kv=case.cap_kv)
    kw.update(sdpa_kwargs)
    dq_t, dk_t, dv_t = g.sdpa_backward(**kw)
    for out, s_max, stride, nh, ro_t in ((dq_t, s_max_q, st_q, h, ro_q_t), (dk_t, s_max_kv, st_kv, hkv, ro_k_t), (dv_t, s_max_kv, st_kv, hkv, ro_k_t)):
        out.set_output(True).set_data_type(io).set_dim([b, nh, s_max, d]).set_stride(stride)
        ro = g.tensor(name=f"{out.get_name()}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        out.set_ragged_offset(ro)
        vp[ro] = ro_t
    vp.update({t["q"]: case.q, t["k"]: case.k, t["v"]: case.v, t["o"]: case.o, t["do"]: case.do})
    return g, vp, (dq_t, dk_t, dv_t)


def _run_graph(
    lens_q,
    lens_kv,
    *,
    h=2,
    hkv=None,
    d=_D,
    dtype=torch.bfloat16,
    stats_layout="head_major",
    poison=False,
    pad_cap=0,
    poison_outputs=False,
    runs=1,
    envelope_q=None,
    check=None,
    case_fixup=None,
    **kw,
):
    """Build the ragged graph, PIN the engine, execute, compare per sequence.  ``use_causal_mask`` / ``use_causal_mask_bottom_right``
    / ``sliding_window_length`` thread through ``kw`` to the graph AND into the case, so the fp64 reference masks with the same
    per-sequence geometry the kernel does.  ``poison_outputs`` NaN-fills dQ / dK / dV before every run; the workspace is
    byte-poisoned (0xFF = NaN in every dtype the chain stores) before every run.  ``check`` replaces the per-sequence oracle
    (``_check``) and ``case_fixup`` edits the case before the graph is built -- for geometries whose fp64 softmax is NaN on a row
    with no key (``_check_rows_with_keys``)."""
    case = _thd_case(
        lens_q, lens_kv, h, d, dtype, cap_q=sum(lens_q) + pad_cap, cap_kv=sum(lens_kv) + pad_cap, poison=poison,
        causal=bool(kw.get("use_causal_mask") or kw.get("use_causal_mask_bottom_right")), bottom_right=bool(kw.get("use_causal_mask_bottom_right")),
        window_left=kw.get("sliding_window_length"), hkv=hkv,
    )  # fmt: skip
    if case_fixup is not None:
        case_fixup(case)
    g, vp, (dq_t, dk_t, dv_t) = _build_thd_bwd_graph(case, stats_layout=stats_layout, hkv=hkv, envelope_q=envelope_q, **kw)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    _kvh = h if hkv is None else hkv
    fill = float("nan") if poison_outputs else 0.0
    dq = torch.full_like(case.q, fill)
    dk, dv = (torch.full((1, case.cap_kv, _kvh, d), fill, device="cuda", dtype=dtype) for _ in range(2))
    vp.update({dq_t: dq, dk_t: dk, dv_t: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    outs = []
    for _ in range(runs):
        for x in (dq, dk, dv):
            x.fill_(fill)
        _sentinel_tails(case, dq, dk, dv)
        ws.fill_(0xFF)
        g.execute(vp, ws)
        torch.cuda.synchronize()
        outs.append(tuple(x.clone() for x in (dq, dk, dv)))
    for name, x in (("dQ", dq), ("dK", dk), ("dV", dv)):
        live = x[0, : case.t_q] if name == "dQ" else x[0, : case.t_kv]
        assert torch.isfinite(live).all(), f"{name} has non-finite values in the packed region"
    _assert_tails_untouched(case, dq, dk, dv)
    (check or _check)(case, dq, dk, dv)
    return case, dq, dk, dv, outs


@requires_rubin
@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_self_attention(dt):
    """The whole point: a ragged graph reaches the kernels through the engine."""
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt)


@requires_rubin
def test_graph_thd_cross_attention():
    _run_graph((256, 100), (180, 300))


@requires_rubin
@pytest.mark.parametrize("layout", ("head_major", "token_major"))
def test_graph_thd_stats_packings(layout):
    """Both packed Stats layouts the forward can emit, each its own test (a frozenset-style claim); the head-major one
    carries a head stride WIDER than the packed total."""
    _run_graph((300, 128), (300, 128), stats_layout=layout)


@requires_rubin
def test_graph_thd_zero_length_sequence():
    """A sequence with no tokens on either side must not corrupt its neighbours, and its own gradients are exact zeros."""
    case, dq, dk, dv, _ = _run_graph((256, 0, 128), (256, 0, 128), poison_outputs=True)
    _assert_empty_sequence_exactly_zero(case, dq, dk, dv, 1)


@requires_rubin
@pytest.mark.parametrize("lens_q,lens_kv", (((0, 128, 256), (192, 128, 256)), ((192, 128, 256), (0, 128, 256))), ids=("empty_q_side", "empty_kv_side"))
def test_graph_thd_one_sided_empty_sequence(lens_q, lens_kv):
    """A sequence empty on ONE side only.  Empty Q: its kv blocks still get units (one forced fully-masked q tile each -> dS = 0
    into their own rows, dV = 0 stored to the sequence's rows), the dK GEMM's reduction is empty and must store zeros by
    SELECT (not residue), dQ has no rows.  Empty KV: no unit at all, the dQ GEMM's reduction is empty -> zeros by select, dK /
    dV have no rows.  Zero is the answer, not a convention; the poisoned outputs make an unwritten live row NaN."""
    case, dq, dk, dv, _ = _run_graph(lens_q, lens_kv, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(case, dq, dk, dv, 0)


@requires_rubin
def test_graph_thd_nan_capacity_tail():
    """Declared totals larger than the live packing, with a NaN tail: the kernels' device-side descriptor clamps (the five
    input maps of the main kernel, stage 3's B slot) keep the rows between ``cu_*[B]`` and the capacity out of every MMA."""
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384)


@requires_rubin
def test_graph_thd_nan_capacity_tail_unaligned_last_sequence():
    """The capacity tail reached by a K-TILE OVERSHOOT (a 100-token last sequence: ceil(100/64) tiles read 28 rows past the
    packed total) on both GEMM orientations, and by the main kernel's last kv block / q tile."""
    _run_graph((256, 100), (256, 100), poison=True, pad_cap=384)


@requires_rubin
def test_graph_thd_b1_matches_dense_shape():
    _run_graph((512,), (512,))


@requires_rubin
@pytest.mark.parametrize("hkv", (2, 1), ids=("gqa_group2", "mqa"))
def test_graph_thd_gqa(hkv):
    """Packed GQA / MQA: dK / dV partials ONE PER Q HEAD over the packed kv axis, folded onto the KV heads; the reference
    SUMS a group's contributions (comparing against one member passes for MQA and fails for GQA, so both sizes run)."""
    _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=hkv)


@requires_rubin
def test_graph_thd_gqa_causal():
    _run_graph((256, 100), (256, 100), h=4, hkv=2, use_causal_mask=True)


@requires_rubin
def test_graph_thd_gqa_cross_attention_and_zero_length():
    """GQA over unequal Q / KV totals with an empty sequence in the middle: the dK / dV partials are sized on the packed KV
    capacity while dQ rides the Q one."""
    _run_graph((256, 0, 100), (180, 0, 300), h=4, hkv=2)


@requires_rubin
def test_graph_thd_gqa_capacity_tail_untouched():
    """GQA with declared totals past the live packing and a NaN tail on BOTH sides.  The per-Q-head partials past ``cu_k[B]``
    are never written (dV and dK store through per-sequence clipped descriptors), so the fold must stop at the live kv total
    on device: an unbounded fold reads the 0xFF-poisoned workspace there and writes NaN into the caller's dK / dV capacity
    tail -- which only the FINITE tail sentinel of ``_run_graph`` can see (a NaN-filled output could not)."""
    _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, poison=True, pad_cap=384, poison_outputs=True)


@requires_rubin
def test_graph_thd_every_sequence_empty_q_with_nan_in_dO_row_0():
    """No query anywhere (``cu_q[B] = 0``), keys in every sequence, the Q / dO capacity all NaN (no live row to keep it finite).
    The clamped Q / dO descriptors keep an extent of 1 (0 is invalid), so a forced tile that addressed ``cu_q[b] = 0`` would
    load the NaN row and ``dV = P^T . dO`` would be ``0 * NaN`` on every live kv row.  The kernel routes every load of a unit
    without query rows past the clamped extent instead (zero-filled), so dK and dV come out as exact zeros and nothing past
    the packed totals is written.  The graph declares the capacity envelope (128), as a caller does: the longest sequence
    would be S_q = 1, a decode shape."""
    case, dq, dk, dv, _ = _run_graph((0, 0), (128, 256), poison=True, pad_cap=128, poison_outputs=True, envelope_q=128)
    for i in range(case.b):
        _assert_empty_sequence_exactly_zero(case, dq, dk, dv, i)


# --- causal family: the per-sequence diagonal is the whole risk, and stage 3 trims its K range per sequence over a workspace
# whose masked tiles the kernel never wrote (poisoned here, never zero-filled: a read past the band lands NaN).  Unequal
# lengths, none a tile multiple: with equal lengths a diagonal from the envelope would agree with the per-sequence one and the
# bug would hide.


@requires_rubin
@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_causal(dt):
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt, use_causal_mask=True)


@requires_rubin
def test_graph_thd_causal_zero_length_sequence():
    """Causal plus an empty sequence: the masked tiles the kernel never wrote and the zero-length sequence's missing unit /
    extent-1 descriptor are independent mechanisms, both live at once."""
    case, dq, dk, dv, _ = _run_graph((256, 0, 128), (256, 0, 128), use_causal_mask=True, poison_outputs=True)
    _assert_empty_sequence_exactly_zero(case, dq, dk, dv, 1)


@requires_rubin
def test_graph_thd_causal_cross_attention():
    """Top-left aligned, so the diagonal does not move -- but S_kv[b] != S_q[b] makes the live kv blocks per q row differ per
    sequence, which a trim keyed on absolute workspace rows would get wrong."""
    _run_graph((256, 100), (180, 300), use_causal_mask=True)


@requires_rubin
def test_graph_thd_causal_bottom_right():
    """Bottom-right alignment: the diagonal offset IS per sequence (56 and 200 here) -- the quantity stage 3's host-scalar
    shift cannot express, so the THD trim reads it from the metadata per sequence (``thd_causal_bottom_right``).  Both
    sequences keep S_kv >= S_q (shorter the other way, the leading rows have no key and the fp64 softmax of a fully masked
    row is NaN -- a degenerate mask, not a kernel property)."""
    _run_graph((200, 100), (256, 300), use_causal_mask_bottom_right=True)


@requires_rubin
def test_graph_thd_causal_bottom_right_ragged_s_q():
    """Bottom-right at a ragged S_q (the half row keeps ``bottom_right_s_q_multiple = 1``): the per-sequence diagonal from REAL
    lengths, the last q tile's pad columns select-masked."""
    _run_graph((200, 300), (456, 300), use_causal_mask_bottom_right=True)


@requires_rubin
def test_graph_thd_causal_swa():
    """Sliding window: a LEFT bound on top of the causal right one -- the band's second per-sequence edge, through the
    kernel's q-tile trim from above and stage 3's per-sequence window edge, so the window-skipped tiles are never read."""
    _run_graph((300, 128, 200), (300, 128, 200), use_causal_mask=True, sliding_window_length=64)


@requires_rubin
def test_graph_thd_causal_nan_capacity_tail():
    """Causal with a NaN tail past the declared totals: a different set of workspace rows than the dense tail case, read
    through the per-sequence trim with no zero-fill in between -- neither the kernel nor the GEMMs may reach the poisoned
    capacity rows."""
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384, use_causal_mask=True)


@requires_rubin
def test_graph_thd_two_launches_are_bitwise():
    """Two launches of a THD causal GQA graph are bitwise identical: the chain has no atomics, so any difference is a
    first-launch race (a missing fence at an SMEM -> async boundary, a descriptor read before its publish)."""
    _, _, _, _, outs = _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, use_causal_mask=True, runs=2)
    for name, a, b in zip(("dQ", "dK", "dV"), outs[0], outs[1]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name} differs between two launches"


# --------------------------------------------------------------------------- rejects: the THD conjunctions the row declines (host, asserted)


# The graph probes below lower a REAL ragged graph through the frontend (``pygraph.validate`` takes the ambient CUDA device for
# its backend handle), so they need a device -- any arch the backend serves, not a Rubin one: the analyzer's cc is faked to 10.7
# off the Rubin line.  The suite's SM80 floor is that gate; without a device they skip instead of failing ahead of the assertion.
# The direct adapter rejects above run over ``TensorDesc`` samples and need no device at all.
_requires_cuda_device = requires_sm80


def _thd_mismatch(lens_q=(256, 128), lens_kv=(256, 128), *, h=2, hkv=None, stats_layout="head_major", engine=_ENGINE, monkeypatch=None, **kw):
    """``mismatch()`` for a ragged backward graph on ``engine`` (the analyzer's cc faked to 10.7 off the Rubin line), or None if served.
    The case lives on the CPU: the graph needs its geometry, never its data."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    if monkeypatch is not None:
        monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    case = _thd_case(lens_q, lens_kv, h, _D, torch.bfloat16, hkv=hkv, device="cpu")
    g, _, _ = _build_thd_bwd_graph(case, stats_layout=stats_layout, hkv=hkv, **kw)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as exc:
        return f"refused by the node: {exc}"
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == engine)
    assert facts is not None
    return mismatch(spec.capabilities, facts)


@_requires_cuda_device
def test_graph_thd_accepts_the_plain_case(monkeypatch):
    """The counterweight to the rejects: the same builder, no extras, IS served -- a reject passing for the wrong reason fails here."""
    assert _thd_mismatch(monkeypatch=monkeypatch) is None


@_requires_cuda_device
def test_accept_thd_causal_family(monkeypatch):
    """Top-left, bottom-right and a sliding window are served under THD (the kernel masks from the per-sequence metadata
    lengths; stage 3 renders the dense two-sided K-trim per sequence); right-band widening stays declined on this row, THD or not."""
    assert _thd_mismatch(monkeypatch=monkeypatch, use_causal_mask=True) is None
    assert _thd_mismatch(monkeypatch=monkeypatch, use_causal_mask_bottom_right=True) is None
    assert _thd_mismatch(monkeypatch=monkeypatch, use_causal_mask=True, sliding_window_length=64) is None
    reason = _thd_mismatch(monkeypatch=monkeypatch, diagonal_band_right_bound=64)
    assert reason is not None and "right-band" in reason


@_requires_cuda_device
def test_accept_thd_gqa(monkeypatch):
    assert _thd_mismatch(monkeypatch=monkeypatch, h=4, hkv=2) is None
    assert _thd_mismatch(monkeypatch=monkeypatch, h=4, hkv=1) is None
    assert _thd_mismatch(monkeypatch=monkeypatch, h=4, hkv=2, use_causal_mask=True) is None


@_requires_cuda_device
def test_reject_thd_without_declared_totals(monkeypatch):
    reason = _thd_mismatch(monkeypatch=monkeypatch, declare_totals=False)
    assert reason is not None and "max_total_seq_len" in reason


def test_reject_thd_on_the_quantized_rows(monkeypatch):
    """The fp8 row CLAIMS THD now (``test_sdpa_bwd_thd_fp8_sm107.py``; inverted from the decline it used to pin here), with declared
    totals; the MXFP8 row still does not (its body has no THD arm until its wave lands) -- the rows must say exactly that."""
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS

    fp8 = next(s for s in ENGINE_SPECS if s.name == "sdpa_bwd_sm107_fp8")
    assert fp8.capabilities.thd and fp8.capabilities.thd_declared_totals, "the fp8 row serves THD with declared packed totals"
    mx = next(s for s in ENGINE_SPECS if s.name == "sdpa_bwd_sm107_mxfp8")
    assert not mx.capabilities.thd, "sdpa_bwd_sm107_mxfp8 must not claim THD (its body has no THD arm yet)"


@_requires_cuda_device
def test_reject_thd_dense_stats(monkeypatch):
    """A ragged graph with a DENSE per-batch Stats tensor: legal cuDNN, and its stride reads as head-major while its storage
    is per-batch rectangles -- declined on the absent ragged offset."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    monkeypatch.setattr(ga, "_device_cc", lambda: _RUBIN_CC)
    case = _thd_case((256, 128), (256, 128), 2, _D, torch.bfloat16, device="cpu")
    g, _, _ = _build_thd_bwd_graph(case)
    g.nodes[0].inputs["stats"].set_ragged_offset(None)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    reason = mismatch(spec.capabilities, facts)
    assert reason is not None and "ragged" in reason


# --------------------------------------------------------------------------- the prepared plan: rebind, replay, length forms


@requires_rubin
@pytest.mark.parametrize("stats_layout", ["head_major", "token_major"])
@pytest.mark.parametrize("causal", [False, True])
def test_prepared_thd_rebind_lengths_and_replay(stats_layout, causal, monkeypatch):
    """The prepared THD plan rebinds new buffers AND new lengths without rebuilding tensor operands, allocating, synchronizing
    or compiling, and replays under CUDA-graph capture with changed lengths (every length fact is a device value)."""
    from unittest.mock import patch

    import cutlass.cute as cute

    from cudnn.sdpa.bwd.api_dsl import WorkspaceCarver

    calls = []
    original = cudnn.pygraph.execute

    def record(graph, *args, **kwargs):
        calls.append((graph, args))
        return original(graph, *args, **kwargs)

    with patch.object(cudnn.pygraph, "execute", record):
        old, _, _, _, _ = _run_graph((129, 97, 63), (143, 83, 79), h=4, hkv=2, stats_layout=stats_layout, poison=True, pad_cap=64, use_causal_mask=causal)
    graph, (pack, old_workspace) = calls[-1]
    names = [ref.get_name() for ref in pack]
    assert len(set(names)) == len(names)

    def next_buffers(lens_q, lens_kv):
        case = _thd_case(lens_q, lens_kv, 4, _D, torch.bfloat16, cap_q=old.cap_q, cap_kv=old.cap_kv, poison=True, causal=causal, hkv=2)
        _, fresh, outputs = _build_thd_bwd_graph(case, stats_layout=stats_layout, hkv=2, use_causal_mask=causal)
        gradients = [
            torch.full_like(case.q, float("nan")),
            *(torch.full((1, case.cap_kv, 2, _D), float("nan"), device="cuda", dtype=torch.bfloat16) for _ in range(2)),
        ]
        _sentinel_tails(case, *gradients)
        fresh.update(zip(outputs, gradients))
        by_name = {ref.get_name(): tensor for ref, tensor in fresh.items()}
        assert set(by_name) == set(names)
        return case, {ref: by_name[ref.get_name()] for ref in pack}, [ref.get_name() for ref in outputs]

    case, rebound, output_names = next_buffers((63, 0, 121), (37, 81, 0))
    by_name = {ref.get_name(): tensor for ref, tensor in rebound.items()}
    gradients = [by_name[name] for name in output_names]
    workspace = torch.empty_like(old_workspace).fill_(0xBD)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared THD rebuilt tensor operands, allocated, synchronized or compiled")

    def execute_guarded():
        with monkeypatch.context() as patcher:
            for name in ("view", "reshape", "as_strided", "permute", "transpose", "copy_", "zero_"):
                patcher.setattr(torch.Tensor, name, forbidden)
            for name in ("empty", "empty_like", "zeros", "zeros_like"):
                patcher.setattr(torch, name, forbidden)
            patcher.setattr(WorkspaceCarver, "__init__", forbidden)
            patcher.setattr(cute, "compile", forbidden)
            graph.execute(rebound, workspace)

    torch.cuda.set_sync_debug_mode("error")
    try:
        execute_guarded()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _check(case, *gradients)
    _assert_tails_untouched(case, *gradients)  # GQA + a padded capacity: the fold must stop at the rebound live total
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            execute_guarded()
        case, changed, _ = next_buffers((31, 123, 41), (113, 67, 109))
        for ref, target in rebound.items():
            target.copy_(changed[ref])
        workspace.fill_(0xBD)
        capture.replay()
        _check(case, *gradients)
        _assert_tails_untouched(case, *gradients)
    finally:
        capture.reset()


@requires_rubin
@pytest.mark.parametrize("token_major", [False, True])
def test_prepared_thd_standalone_switches_length_and_prefix_form(token_major, monkeypatch):
    """The standalone surface takes ``(B,)`` lengths or ``(B+1,)`` prefixes per side without recompiling (the form is host
    metadata, derived from numel), and replays under capture."""
    import cutlass.cute as cute

    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    original = SdpaBwdDslSm107.execute

    def exercise(api, *args, **kwargs):
        original(api, *args, **kwargs)
        artifact = api._prepared.artifact
        for name in ("seq_q_lens", "seq_kv_lens"):
            lens = kwargs[name]
            kwargs[name] = torch.cat((torch.zeros(1, device=lens.device, dtype=torch.int32), lens.cumsum(0, dtype=torch.int32)))
        for tensor in args[6:9]:
            tensor.fill_(float("nan"))
        kwargs["workspace"].fill_(0xBD)
        capture = torch.cuda.CUDAGraph()
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(cute, "compile", lambda *a, **k: pytest.fail("length/prefix form must reuse the compiled host"))
                original(api, *args, **kwargs)
                with torch.cuda.graph(capture):
                    original(api, *args, **kwargs)
            for tensor in args[6:9]:
                tensor.fill_(float("nan"))
            kwargs["workspace"].fill_(0xBD)
            capture.replay()
            assert api._prepared.artifact is artifact
        finally:
            capture.reset()

    monkeypatch.setattr(SdpaBwdDslSm107, "execute", exercise)
    _run((129, 63, 97), (113, 75, 141), token_major_stats=token_major)


@requires_rubin
@pytest.mark.parametrize("token_major", [False, True])
def test_prepared_thd_standalone_accepts_flat_stats(token_major, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl_sm107 import SdpaBwdDslSm107

    original = SdpaBwdDslSm107.execute

    def flat_stats(api, *args, **kwargs):
        args = list(args)
        args[5] = args[5].view(-1)
        return original(api, *args, **kwargs)

    monkeypatch.setattr(SdpaBwdDslSm107, "execute", flat_stats)
    _run((129, 63, 97), (113, 75, 141), token_major_stats=token_major)


# --------------------------------------------------------------------------- stage 3 under THD: the per-sequence K-trim and the grouped-B dQ launch
#
# The stage-3 GEMMs render the dense path's two-sided K-trim PER SEQUENCE (``bprop_matmul_blackwell._thd_causal_k_range``: every
# bound sequence-local, the sequence's real lengths and its own diagonal from the metadata, an EMPTY range -- a select-zero
# store -- where the tile has no kept cell), so the kv-blocked dS workspace is no longer zero-filled per execute; and the dQ
# GEMM runs ONCE per head chunk under GQA (``b_head_group = group``, its packed B = K indexed by ``h // group``) instead of once
# per group member.  Every ``_run_graph`` case above already runs over a 0xFF-poisoned workspace, so the trim's read set is
# proven by every causal-family case in this module; the tests below pin the two levers' own contracts.


def _thd_q_range_twin(k0, s_q, s_kv, window, causal, bottom_right, tile_q=128, kv_blk=256, pair=2):
    """Pure-Python twin of the f16 body's ``_q_loop_bounds`` under THD: ``n_q = max(1, ceil(s_q / 128))`` from the SEQUENCE's q
    length, ``compute_q_loop_bounds`` with the per-sequence bottom-right anchor, the pair rounding (always on under THD -- the
    padded mask flag is set by construction) and the never-empty clamps.  ``[q_lo, q_hi)`` in q tiles for the kv block at ``k0``."""
    n_q = max(1, -(-s_q // tile_q))
    diag = (s_kv - s_q) if bottom_right else 0
    lo = max((k0 - diag) // tile_q, 0) if causal else 0
    hi = n_q
    if window is not None:
        hi = min(-(-max(k0 + kv_blk + window - diag, 0) // tile_q), n_q)
    lo = (lo // pair) * pair
    hi = min(-(-hi // pair) * pair, n_q)
    q_lo = min(lo, n_q - 1)
    return q_lo, max(hi, q_lo + 1)


def _thd_k_range_twin(mode, m0, nkt, shift, window, diag, tk=64, gran=256, cgrp_m=256):
    """Pure-Python twin of ``bprop_matmul_blackwell._thd_causal_k_range``, statement by statement (every dividend clamped at 0
    before its ``//`` so floor and trunc agree; every bound clamped to ``nkt``; the range MAY be empty)."""
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_LO, CAUSAL_K_NONE

    if mode == CAUSAL_K_NONE:
        return 0, nkt
    if mode == CAUSAL_K_LO:
        k_lo = 0
        if diag:
            k_lo = min(((max(m0 - shift, 0) // gran) * gran) // tk, nkt)
        k_hi = nkt
        if window > 0:
            k_hi = min((max(m0 + cgrp_m - shift + window, 0) + tk - 1) // tk, nkt)
        return k_lo, max(k_hi, k_lo)
    k_hi = nkt
    if diag:
        hi_raw = m0 + cgrp_m - 1 + shift
        hi = ((max(hi_raw, 0) // gran) + 1) * gran
        k_hi = min((hi + tk - 1) // tk, nkt)
        if hi_raw < 0:
            k_hi = 0
    k_lo = 0
    if window > 0:
        k_lo = min(max(m0 + shift - window, 0) // tk, k_hi)
    return k_lo, k_hi


# (s_q, s_kv, left, causal, bottom_right) PER SEQUENCE: every mask arm the row serves under THD, on lengths where the roundings,
# the clamps and the degenerate sides bite -- tile multiples and not, an odd q-tile count (the last q pair half written), a
# one-block sequence, both empty sides, bottom-right with s_q > s_kv (a NEGATIVE diagonal offset: leading rows without a key)
# and a top-left window with s_q > s_kv + W (trailing rows without a key: the q pairs no kv block writes).
_THD_BAND_GEOMETRIES = [
    (300, 300, None, True, False),
    (128, 300, None, True, False),
    (300, 128, None, True, False),
    (1, 1, None, True, False),
    (129, 65, None, True, False),
    (0, 256, None, True, False),
    (256, 0, None, True, False),
    (200, 256, None, True, True),
    (100, 300, None, True, True),
    (300, 456, None, True, True),
    (300, 200, None, True, True),
    (1000, 100, None, True, True),
    (513, 513, None, True, True),
    (300, 300, 64, True, False),
    (300, 300, 2, True, False),
    (1000, 1000, 200, True, False),
    (1024, 256, 64, True, False),
    (2048, 500, 300, True, False),
    (700, 100, 64, True, False),
    (200, 256, 64, True, True),
    (300, 456, 200, True, True),
    (300, 200, 64, True, True),
    (1000, 100, 64, True, True),
    (300, 300, 64, False, False),
    (1000, 1280, 200, False, False),
    (700, 100, 200, False, False),
    (300, 300, None, False, False),
    (300, 128, None, False, False),
]


@pytest.mark.parametrize("tk", (64,), ids=("bf16-k64",))
def test_stage3_thd_band_arithmetic(tk):
    """The THD K-trim's contract on ~28 per-sequence geometries, no GPU: for EVERY cluster M tile of both GEMMs whose output rows
    are live, (a) the K range COVERS every cell the band keeps and (b) every dS tile it reads was WRITTEN by the sequence's own kv
    blocks (twin of the f16 body's THD ``_q_loop_bounds``, pair rounding included) -- the property the poisoned-workspace cases
    measure on the device; (c) a tile the kernel wrote nothing for -- a q pair past every key -- gets an EMPTY range (the
    select-zero store), never a clamp onto an unwritten tile; and (d) the adapter's records spell exactly these modes and no
    constant shift.  Spare M tiles past the sequence's extent (clipped by its C descriptor) read what they like and are not
    checked."""
    from cudnn.sdpa.bwd.api_dsl_sm107 import _SM107_KV_PAD, _SM107_Q_PAD, _stage3_params, _stage3_thd_needs_zero_fill
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, validate_matmul_params
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from dataclasses import replace

    gran, cgrp_m, tile_q = _SM107_KV_PAD, 256, _SM107_Q_PAD
    for sq, skv, left, causal, bottom_right in _THD_BAND_GEOMETRIES:
        window = None if left is None else left - 1
        dk, dq = _stage3_params(DTYPE_BF16, causal, 0, gran, trim=True, cgrp_tile_mn=(cgrp_m, cgrp_m), window=window, gqa_group=2)
        thd_br = bool(causal and bottom_right)
        dk, dq = (replace(r, thd_varlen=True, thd_rows_kv=True, thd_causal_bottom_right=thd_br) for r in (dk, dq))
        validate_matmul_params(dk)
        validate_matmul_params(dq)
        band = causal or window is not None
        assert (dk.causal_mode, dq.causal_mode) == ((CAUSAL_K_LO, CAUSAL_K_HI) if band else (CAUSAL_K_NONE, CAUSAL_K_NONE))
        assert dk.causal_shift == dq.causal_shift == 0 and dq.b_head_group == 2 and dk.b_head_group == 1
        assert not _stage3_thd_needs_zero_fill(causal, window, gran, trim=True, cgrp_tile_m=cgrp_m)
        shift = (skv - sq) if thd_br else 0
        common = dict(shift=shift, window=dk.causal_window, diag=dk.causal_diag, tk=tk, gran=gran, cgrp_m=cgrp_m)
        tag = f"s_q={sq} s_kv={skv} left={left} causal={causal} bottom_right={bottom_right}"
        qi = torch.arange(sq).view(-1, 1)
        ki = torch.arange(skv).view(1, -1)
        kept = (ki <= qi + shift) if causal else torch.ones(sq, skv, dtype=torch.bool)
        if window is not None:
            kept &= ki >= qi + shift - window
        # the tiles the sequence's kv blocks write: block j (exists iff s_kv > 0) writes q tiles [q_lo, q_hi) of its 256 rows
        n_blk, n_q = -(-skv // gran), max(1, -(-sq // tile_q))
        written = torch.zeros(max(n_blk, 1), n_q + 1, dtype=torch.bool)  # one spare q tile: a GEMM bound may not reach it
        for blk in range(n_blk):
            lo, hi = _thd_q_range_twin(blk * gran, sq, skv, window, causal, bottom_right)
            written[blk, lo:hi] = True
        # dK: M = kv (one block of the sequence per cluster tile), K = q tokens of the sequence
        nkt = -(-sq // tk)
        for blk in range(n_blk):
            m0 = blk * cgrp_m
            k_lo, k_hi = _thd_k_range_twin(dk.causal_mode, m0, nkt, **common)
            assert 0 <= k_lo <= k_hi <= nkt, (tag, m0, k_lo, k_hi)
            rows = slice(m0, min(m0 + cgrp_m, skv))
            kept_q = kept[:, rows].any(dim=1) if sq else torch.zeros(0, dtype=torch.bool)
            outside = torch.ones(sq, dtype=torch.bool)
            outside[k_lo * tk : min(k_hi * tk, sq)] = False
            assert not (kept_q & outside).any(), f"dK {tag}: block {blk} K range [{k_lo}, {k_hi}) drops kept q rows"
            for kt in range(k_lo, k_hi):
                assert written[blk, (kt * tk) // tile_q], f"dK {tag}: block {blk} reads q tile {(kt * tk) // tile_q} it never wrote"
        # dQ: M = q (a 256-row pair of the sequence per cluster tile), K = kv tokens of the sequence
        nkt = -(-skv // tk)
        for m0 in range(0, sq, cgrp_m):
            k_lo, k_hi = _thd_k_range_twin(dq.causal_mode, m0, nkt, **common)
            assert 0 <= k_lo <= k_hi <= nkt, (tag, m0, k_lo, k_hi)
            rows = slice(m0, min(m0 + cgrp_m, sq))
            kept_kv = kept[rows, :].any(dim=0) if skv else torch.zeros(0, dtype=torch.bool)
            outside = torch.ones(skv, dtype=torch.bool)
            outside[k_lo * tk : min(k_hi * tk, skv)] = False
            assert not (kept_kv & outside).any(), f"dQ {tag}: pair at {m0} K range [{k_lo}, {k_hi}) drops kept kv rows"
            live_tiles = [t for t in (m0 // tile_q, m0 // tile_q + 1) if t * tile_q < sq]  # the pair's q tiles with live rows
            for kt in range(k_lo, k_hi):
                blk = (kt * tk) // gran
                for t in live_tiles:
                    assert written[blk, t], f"dQ {tag}: pair at {m0} reads kv block {blk} x q tile {t}, which that block never wrote"
            if band and not kept[rows, :].any():
                assert k_lo == k_hi, f"dQ {tag}: pair at {m0} has no kept cell but a non-empty K range [{k_lo}, {k_hi})"
        # the dense row's never-empty clamps would have read one tile here; under THD the untrimmed twin keeps the fill instead
        assert _stage3_thd_needs_zero_fill(causal, window, gran, trim=False) == bool(causal or window is not None)
        assert _stage3_thd_needs_zero_fill(causal, window, gran, trim=True, cgrp_tile_m=512) == bool(causal or window is not None)


def test_stage3_thd_arm_is_gated_and_the_records_carry_no_constant_shift():
    """Source + rendering pins for the THD trim: the template keys every line of it on ``_THD_TRIM`` (THD with a trimmed mode), the
    dense and the untrimmed-THD renderings keep their statements, the epilogue's store select follows the per-tile K range, and
    the host launches the THD dQ through ``_dq_launches`` like the dense one.  ``validate_matmul_params`` refuses a constant shift
    on a THD record and ``thd_causal_bottom_right`` off its leg."""
    import re
    from pathlib import Path

    from cudnn.frost.template_loader import load_template
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16
    from cudnn.sdpa.bwd.api_dsl import _SM100_MATMUL_FILE, _sm100_kernel_path
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE, MatmulTemplateParams, validate_matmul_params
    from cudnn.sdpa.bwd.kernels.sm107 import prepared_host

    src = Path(_sm100_kernel_path(_SM100_MATMUL_FILE)).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "_THD_TRIM = _THD_MM and causal_mode != CAUSAL_K_NONE" in code
    assert code.count("return _thd_causal_k_range(coord_m_cgrp, nkt, thd_shift)") == 1
    assert "if cutlass.const_expr(_THD_TRIM):\n        return _thd_causal_k_range(" in code, "the THD arm sits after the CAUSAL_K_NONE early return"
    assert code.count("_thd_shift(meta_t, tile_b, n_batch)") == 3, "the definition, the TMA warp's and the epilogue's call"
    assert code.count("_thd_shift(meta_t, tile_b_mma, n_batch)") == 1, "the MMA warp's call"
    assert "_thd_store_live = _ke_epi > _kb_epi" in code and "if _thd_store_live:" in code
    assert "elif cutlass.const_expr(_THD_MM):\n                        if _thd_k_len > cutlass.Int32(0):" in code, "the untrimmed THD store select is untouched"
    assert re.search(r"k_hi = cutlass\.Int32\(arith\.select\(\(hi_raw < zero\)\.ir_value\(\), zero\.ir_value\(\), k_hi\.ir_value\(\)\)\)", code)
    host = Path(prepared_host.__file__).read_text(encoding="utf-8")
    body = host[host.index("def _stage3_thd(") : host.index("def _dq_launches(")]
    assert "n_launch = _dq_launches(group, dq_b_head_group)" in body and "for member in range(n_launch):" in body
    assert "_window(ds, 1, member, heads, n_launch)" in body and "_window(dq_out, 2, hb + member, heads, n_launch)" in body
    # renderings: the gate folds per record
    common = dict(causal_gran=256, vec_bytes_epi=32, dtype_qkv=DTYPE_BF16, cgrp_tile_mn=(256, 256), thd_varlen=True, thd_rows_kv=True)
    for mode, br, trim in ((CAUSAL_K_NONE, False, False), (CAUSAL_K_LO, False, True), (CAUSAL_K_HI, True, True)):
        mod = load_template(
            _sm100_kernel_path(_SM100_MATMUL_FILE),
            MatmulTemplateParams(a_is_m_major=mode == CAUSAL_K_HI, causal_mode=mode, thd_causal_bottom_right=br, b_head_group=2, **common),
            tag="test_thd_stage3_gate",
        )
        assert mod._THD_TRIM is trim and mod.thd_causal_bottom_right is br and mod.b_head_group == 2
    with pytest.raises(ValueError, match="per sequence"):
        validate_matmul_params(MatmulTemplateParams(a_is_m_major=False, causal_mode=CAUSAL_K_LO, causal_shift=256, **common))
    with pytest.raises(ValueError, match="requires thd_varlen"):
        validate_matmul_params(
            MatmulTemplateParams(a_is_m_major=False, causal_mode=CAUSAL_K_LO, causal_gran=256, cgrp_tile_mn=(256, 256), thd_causal_bottom_right=True)
        )
    with pytest.raises(ValueError, match="trimmed causal_mode"):
        validate_matmul_params(MatmulTemplateParams(a_is_m_major=False, causal_mode=CAUSAL_K_NONE, thd_causal_bottom_right=True, **common))


def test_thd_zero_fill_rule_follows_the_trim(monkeypatch):
    """The THD workspace is zero-filled per execute ONLY for the untrimmed twin (``STAGE3_CAUSAL_TRIM = False``) and the wide-tile
    twin; every served mask arm runs trimmed over whatever the workspace held.  The adapter's THD records flip with the constant
    (read at CALL time, so the bitwise pin can flip it)."""
    import types

    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_NONE

    for causal, window in ((True, None), (True, 63), (False, 199), (False, None)):
        masked = causal or window is not None
        assert sm107._stage3_thd_needs_zero_fill(causal, window, 256) is False
        assert sm107._stage3_thd_needs_zero_fill(causal, window, 256, trim=False) is masked
        assert sm107._stage3_thd_needs_zero_fill(causal, window, 256, cgrp_tile_m=512) is masked
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    assert sm107._stage3_thd_needs_zero_fill(True, None, 256) is True and sm107._stage3_thd_needs_zero_fill(False, None, 256) is False
    # the envelope of two sequences of 256 / 128 tokens as TensorDescs: the records need the plan facts, never a buffer
    api = _thd_adapter(is_causal=True, causal_bottom_right=True, max_total_seq_len_q=384, max_total_seq_len_kv=384)
    mod = types.SimpleNamespace(CFG=types.SimpleNamespace(TILE_M=128, CTA_MMA=2))
    dk, dq = api._stage3_records(mod, (256, 256))
    assert dk.causal_mode == dq.causal_mode == CAUSAL_K_NONE and not dk.thd_causal_bottom_right, "the untrimmed twin: no band, no per-sequence diagonal"
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", True)
    dk, dq = api._stage3_records(mod, (256, 256))
    assert dk.causal_mode != CAUSAL_K_NONE and dk.thd_causal_bottom_right and dq.thd_causal_bottom_right and dq.b_head_group == 1


def _rows_with_keys(case, i):
    """``[r0, r1)``: the q rows of sequence ``i`` with at least one key under the case's mask (bottom-right with ``s_q > s_kv``
    strips LEADING rows, a top-left window with ``s_q > s_kv + W`` strips TRAILING rows); every other row is fully masked.
    Row ``r`` keeps ``kv`` in ``[r + diag - (w - 1), r + diag]`` (``w`` = the graph's keys per row, every key when None)."""
    assert case.causal, "the case machinery applies a window only together with a causal diagonal"
    s_q, s_kv = case.lens_q[i], case.lens_kv[i]
    diag = (s_kv - s_q) if case.bottom_right else 0
    w = case.window_left
    rows = [r for r in range(s_q) if (0 if w is None else max(r + diag - (w - 1), 0)) <= min(r + diag, s_kv - 1)]
    if not rows:
        return 0, 0
    assert rows == list(range(rows[0], rows[-1] + 1)), "the rows with keys are one contiguous span"
    return rows[0], rows[-1] + 1


def _forward_convention_for_rows_without_keys(case):
    """The forward's contract on a fully masked row (sdpa-invariants: ``O = 0``, ``LSE = -inf``): the fp64 softmax of an all
    ``-inf`` row is NaN, so the case's O carries NaN there -- which would reach ``delta = rowsum(dO * O)`` as NaN and poison
    the written dS tiles of those rows; ``-inf`` the logsumexp already is."""
    for i in range(case.b):
        r0, r1 = _rows_with_keys(case, i)
        q0 = case.cu_q[i]
        for lo, hi in ((0, r0), (r1, case.lens_q[i])):
            if hi > lo:
                case.o[0, q0 + lo : q0 + hi] = 0.0
                case.lse[0, :, q0 + lo : q0 + hi] = float("-inf")
                assert torch.isinf(case.lse[0, :, q0 + lo : q0 + hi]).all()


def _check_rows_with_keys(case, dq, dk, dv):
    """Per-sequence oracle for a case with fully masked rows: the rows WITHOUT a key get ``dQ = 0`` exactly (asserted on the output
    directly -- a diff against the NaN reference proves nothing) and contribute nothing to dK / dV; the rows with keys are checked
    against ``_ref_bwd`` over that row span alone, whose mask is the same band (a bottom-right diagonal moves with the span, a
    top-left one does not)."""
    bad = []
    for i in range(case.b):
        if case.lens_q[i] == 0 or case.lens_kv[i] == 0:
            continue
        r0, r1 = _rows_with_keys(case, i)
        q0, k0 = case.cu_q[i], case.cu_k[i]
        grp = case.h // case.hkv
        rep = (lambda x: x.repeat_interleave(grp, dim=0)) if grp > 1 else (lambda x: x)
        for lo, hi in ((0, r0), (r1, case.lens_q[i])):
            dead = dq[0, q0 + lo : q0 + hi]
            if dead.numel():
                assert torch.isfinite(dead).all() and not dead.any(), f"seq {i}: dQ rows [{lo}, {hi}) have no key and must be exactly zero"
        if r1 <= r0:
            for name, got in (("dK", dk[0, k0 : k0 + case.lens_kv[i]]), ("dV", dv[0, k0 : k0 + case.lens_kv[i]])):
                assert torch.isfinite(got).all() and not got.any(), f"seq {i}: no row has a key, {name} must be exactly zero"
            continue
        rq, rk, rv = _ref_bwd(
            case.q[0, q0 + r0 : q0 + r1].transpose(0, 1),
            rep(case.k[0, k0 : k0 + case.lens_kv[i]].transpose(0, 1)),
            rep(case.v[0, k0 : k0 + case.lens_kv[i]].transpose(0, 1)),
            case.do[0, q0 + r0 : q0 + r1].transpose(0, 1),
            case.scale,
            causal=case.causal,
            bottom_right=case.bottom_right,
            window_left=case.window_left,
        )
        if grp > 1:
            rk = rk.reshape(case.hkv, grp, *rk.shape[1:]).sum(1)
            rv = rv.reshape(case.hkv, grp, *rv.shape[1:]).sum(1)
        for name, got, want in (
            ("dQ", dq[0, q0 + r0 : q0 + r1].transpose(0, 1), rq),
            ("dK", dk[0, k0 : k0 + case.lens_kv[i]].transpose(0, 1), rk),
            ("dV", dv[0, k0 : k0 + case.lens_kv[i]].transpose(0, 1), rv),
        ):
            verdict = _grad_failure(f"seq {i} {name} rows [{r0}, {r1}) (lens q={case.lens_q[i]} kv={case.lens_kv[i]})", got, want, case.dtype)
            if verdict:
                bad.append(verdict)
    assert not bad, "\n".join(bad)


def _bitwise(name, a, b):
    n_diff = (a.view(torch.int16) != b.view(torch.int16)).sum().item()
    assert n_diff == 0, f"{name}: {n_diff} of {a.numel()} elements differ (max|diff|={(a.float() - b.float()).abs().max().item():.3e})"


_GQA_TWIN_CASES = {
    "gqa32-2-dense": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), h=32, hkv=2),
    "gqa32-2-causal": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), h=32, hkv=2, use_causal_mask=True),
    "gqa64-8-dense": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), h=64, hkv=8),
    "gqa64-8-causal": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), h=64, hkv=8, use_causal_mask=True),
    "gqa32-2-bottom-right": dict(lens_q=(200, 100), lens_kv=(256, 300), h=32, hkv=2, use_causal_mask_bottom_right=True),
    "gqa32-2-causal-swa64": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), h=32, hkv=2, use_causal_mask=True, sliding_window_length=64),
    "gqa32-2-causal-empty-q-side": dict(lens_q=(0, 128, 256), lens_kv=(192, 128, 256), h=32, hkv=2, use_causal_mask=True, poison_outputs=True),
    "gqa32-2-causal-empty-kv-side": dict(lens_q=(192, 128, 256), lens_kv=(0, 128, 256), h=32, hkv=2, use_causal_mask=True, poison_outputs=True),
    "gqa8-2-cross-causal-ragged": dict(lens_q=(256, 100, 700), lens_kv=(180, 300, 700), h=8, hkv=2, use_causal_mask=True),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_GQA_TWIN_CASES), ids=list(_GQA_TWIN_CASES))
def test_graph_thd_single_launch_dq_is_bitwise_the_per_member_launches(case, monkeypatch):
    """Under GQA the THD dQ GEMM is ONE launch per head chunk -- its rendering indexes the packed B = K by ``h // group``
    (``b_head_group = group``) over the whole dS and dQ chunk, the packed B descriptor ``kv_n`` heads deep with its per-sequence
    token clamp -- where it used to be one launch per group MEMBER.  Both pair every Q head with the same K head and walk the same
    k tiles per output tile into an fp32 accumulator, so dQ must be the SAME BITS (and dK / dV, which the change never touches).
    ``DQ_SINGLE_LAUNCH = False`` is the twin; both runs are held to the per-sequence fp64 oracle too.  Covers GQA 32/2 and 64/8,
    tails that are no multiple of 256, dense / causal / bottom-right / a window, and a sequence empty on either side."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_GQA_TWIN_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    assert sm107.DQ_SINGLE_LAUNCH, "one dQ launch per chunk is what ships; the pin flips it OFF for the twin"
    _, *single, _ = _run_graph(lens_q, lens_kv, **kw)
    monkeypatch.setattr(sm107, "DQ_SINGLE_LAUNCH", False)
    _, *members, _ = _run_graph(lens_q, lens_kv, **kw)
    for name, a, b in zip(("dQ", "dK", "dV"), single, members):
        _bitwise(f"{name} (single dQ launch vs per-member launches)", a, b)


_TRIM_TWIN_CASES = {
    "causal": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), use_causal_mask=True),
    "causal-cross": dict(lens_q=(256, 100), lens_kv=(180, 300), use_causal_mask=True),
    "causal-gqa": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), h=4, hkv=2, use_causal_mask=True),
    "bottom-right": dict(lens_q=(200, 100), lens_kv=(256, 300), use_causal_mask_bottom_right=True),
    "bottom-right-ragged-sq": dict(lens_q=(200, 300), lens_kv=(456, 300), use_causal_mask_bottom_right=True),
    "causal-swa64": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), use_causal_mask=True, sliding_window_length=64),
    "causal-swa200-long": dict(lens_q=(1000, 700), lens_kv=(1000, 700), use_causal_mask=True, sliding_window_length=200),
    "bottom-right-swa64": dict(lens_q=(200, 100), lens_kv=(256, 300), use_causal_mask_bottom_right=True, sliding_window_length=64),
    "causal-empty-q-side": dict(lens_q=(0, 128, 256), lens_kv=(192, 128, 256), use_causal_mask=True, poison_outputs=True),
    "bottom-right-empty-kv-side": dict(lens_q=(192, 128, 256), lens_kv=(0, 128, 256), use_causal_mask_bottom_right=True, poison_outputs=True),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_TRIM_TWIN_CASES), ids=list(_TRIM_TWIN_CASES))
def test_graph_thd_trimmed_stage3_is_bitwise_the_untrimmed_rendering_over_a_poisoned_workspace(case, monkeypatch):
    """The per-sequence K-trim reads ONLY dS tiles the main kernel wrote: ``_run_graph`` poisons the workspace with 0xFF (NaN in
    bf16) before the execute and the shipped chain runs WITHOUT the zero-fill, so a GEMM reaching a tile the kernel skipped lands
    NaN in dQ / dK (``_run_graph`` asserts finite first; the per-sequence fp64 oracle then holds the values).  The untrimmed twin
    (``STAGE3_CAUSAL_TRIM = False``: ``CAUSAL_K_NONE`` on both GEMMs, the fill back on) walks every k tile of the zero-filled
    workspace -- the skipped tiles are exact zeros, so the two accumulate the same values in the same order: the gradients must be
    the SAME BITS.  Every mask arm the THD row serves, on tails that are no tile multiple, with an empty side on either axis."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_TRIM_TWIN_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    assert sm107.STAGE3_CAUSAL_TRIM, "the per-sequence trim is what ships; the pin flips it OFF for the twin"
    _, *trimmed, _ = _run_graph(lens_q, lens_kv, **kw)
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    _, *untrimmed, _ = _run_graph(lens_q, lens_kv, **kw)
    for name, a, b in zip(("dQ", "dK", "dV"), trimmed, untrimmed):
        _bitwise(f"{name} (trimmed stage 3, no fill vs untrimmed, zero-filled)", a, b)


_NO_KEY_ROW_CASES = {
    # a top-left window with s_q > s_kv + W in one sequence: its trailing q pairs are written by NO kv block -- the one geometry
    # the dense adapter still zero-fills for; the envelope keeps max_s_q <= max_s_kv, which is what the frontend requires of a window
    "tl-window-past-the-keys": dict(lens_q=(1024, 300), lens_kv=(256, 1024), use_causal_mask=True, sliding_window_length=64),
    "tl-window-past-the-keys-gqa": dict(lens_q=(700, 300), lens_kv=(100, 700), h=4, hkv=2, use_causal_mask=True, sliding_window_length=64),
    # bottom-right with s_q > s_kv in one sequence: a NEGATIVE diagonal offset, the leading rows have no key
    "bottom-right-s_q-past-s_kv": dict(lens_q=(300, 100), lens_kv=(200, 300), use_causal_mask_bottom_right=True),
    "bottom-right-s_q-past-s_kv-swa": dict(lens_q=(300, 100), lens_kv=(200, 300), use_causal_mask_bottom_right=True, sliding_window_length=64),
}


@requires_rubin
@pytest.mark.parametrize("case", list(_NO_KEY_ROW_CASES), ids=list(_NO_KEY_ROW_CASES))
def test_graph_thd_rows_without_a_key_are_exactly_zero_and_read_no_unwritten_tile(case, monkeypatch):
    """Per-sequence geometries with FULLY MASKED q rows -- the trailing rows of a top-left window past ``s_kv + W`` (q pairs no
    kv block writes) and the leading rows of a bottom-right sequence with ``s_q > s_kv`` (a negative diagonal offset).  Their dQ
    is exactly zero: the trim hands those M tiles an EMPTY K range (the select-zero store) instead of clamping onto an unwritten
    tile of the poisoned workspace, and the rows with keys match the fp64 oracle over their own span.  The forward's convention
    on a fully masked row (``O = 0``, ``LSE = -inf``) is applied to the case first; the untrimmed, zero-filled twin must agree
    bitwise."""
    import cudnn.sdpa.bwd.api_dsl_sm107 as sm107

    kw = dict(_NO_KEY_ROW_CASES[case])
    lens_q, lens_kv = kw.pop("lens_q"), kw.pop("lens_kv")
    run = dict(poison_outputs=True, check=_check_rows_with_keys, case_fixup=_forward_convention_for_rows_without_keys, **kw)
    _, *trimmed, _ = _run_graph(lens_q, lens_kv, **run)
    monkeypatch.setattr(sm107, "STAGE3_CAUSAL_TRIM", False)
    _, *untrimmed, _ = _run_graph(lens_q, lens_kv, **run)
    for name, a, b in zip(("dQ", "dK", "dV"), trimmed, untrimmed):
        _bitwise(f"{name} (trimmed vs untrimmed, rows without a key)", a, b)


@requires_rubin
@pytest.mark.parametrize(
    "lens_q,lens_kv,kw",
    [
        ((0, 128, 256), (192, 128, 256), dict(use_causal_mask_bottom_right=True)),
        ((192, 128, 256), (0, 128, 256), dict(use_causal_mask=True, sliding_window_length=64)),
        ((256, 0, 128), (256, 0, 128), dict(use_causal_mask_bottom_right=True, sliding_window_length=64)),
        ((300, 5, 200), (300, 5, 200), dict(use_causal_mask=True)),
        ((300, 65, 200), (300, 129, 200), dict(use_causal_mask=True, sliding_window_length=2)),  # s_q <= s_kv: every row keeps a key
    ],
    ids=["br-empty-q-side", "swa-empty-kv-side", "br-swa-zero-length", "five-row-sequence", "window-of-two-keys-odd-tiles"],
)
def test_graph_thd_trimmed_degenerate_sequences(lens_q, lens_kv, kw):
    """The degenerate-input matrix under the trimmed stage 3 (sdpa-invariants): an empty q side (dK / dV rows exist, the
    reduction is empty -> exact zeros through the select), an empty kv side (no unit, no rows), a zero-length sequence inside
    the batch, a five-row sequence (one q tile, one kv block), the narrowest served window (two keys: ``window_left = 1``) on odd
    q-tile counts (the last pair half written) -- all over a poisoned
    workspace and poisoned outputs, every live gradient held to the oracle."""
    case, dq, dk, dv, _ = _run_graph(lens_q, lens_kv, poison_outputs=True, **kw)
    for i in range(case.b):
        if case.lens_q[i] == 0 or case.lens_kv[i] == 0:
            _assert_empty_sequence_exactly_zero(case, dq, dk, dv, i)


@requires_rubin
@pytest.mark.L1
@pytest.mark.parametrize("batch", [33, 129])
def test_graph_thd_batched_descriptors(batch):
    """More sequences than setup warps (the per-sequence dV descriptors and stage 3's are shared round-robin), empty
    sequences interleaved, a poisoned capacity tail."""
    lens_q = [[0, 17, 65, 129][i % 4] for i in range(batch)]
    lens_kv = [[33, 0, 127, 257][i % 4] for i in range(batch)]
    _run_graph(lens_q, lens_kv, poison=True, pad_cap=256)
