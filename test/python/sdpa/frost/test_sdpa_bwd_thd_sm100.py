# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm100`` THD / varlen: the packed backward, end to end.

Two surfaces, deliberately both:

* ``_run`` drives ``SdpaBwdDslSm100`` DIRECTLY -- the numerics of the
  three-stage chain over PACKED input and a BLOCKED S/dS workspace, one layer
  below the plan machinery, so a failure localises to the kernels.
* ``_run_graph`` goes through the ragged GRAPH and pins the engine -- the
  lowering: packed views over the caller's buffers, the length binding, and the
  Stats packing inference. A pass there with a decline underneath would be a
  cuDNN plan, which is why the engine is pinned by name.

The reference is per-sequence dense attention: unpack, run each sequence on its
own, and compare gradient by gradient.  A THD bug that leaks across a sequence
boundary shows up as one sequence's gradient contaminated by its neighbour's,
which a whole-tensor cosine would happily average away -- so every assertion is
per sequence.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

import cudnn

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

_ENGINE = "sdpa_bwd_sm100"
_D = 512
_TOL_COS = 0.999


def _causal_bias(s_q, s_kv, device, bottom_right=False, window_left=None, window_right=None):
    """Additive -inf mask for ONE sequence, in that sequence's OWN geometry.

    This is the whole reason the causal tests are per sequence: under THD the
    diagonal is per sequence, so a mask built from the envelope ``S_max`` -- or
    from the packed total -- is a different mask for every sequence but the
    longest.  ``bottom_right`` aligns the diagonal to the last row, which under
    THD makes the offset ``S_kv[b] - S_q[b]``, a per-sequence quantity.
    ``window_left`` adds the sliding-window lower bound and ``window_right``
    widens the upper one; both follow ``test_sdpa_bwd_dsl_sm100._causal_keep``,
    i.e. ``left_bound`` is an inclusive column count
    (``kv >= q + diag - (left_bound - 1)``) and ``right_bound`` an offset
    (``kv <= q + diag + right_bound``).
    """
    q_i = torch.arange(s_q, device=device).view(-1, 1)
    kv_i = torch.arange(s_kv, device=device).view(1, -1)
    diag = (s_kv - s_q) if bottom_right else 0
    keep = kv_i <= q_i + diag + (window_right or 0)
    if window_left is not None:
        keep = keep & (kv_i > q_i + diag - window_left)
    return torch.where(keep, 0.0, float("-inf")).double()


def _ref_bwd(q, k, v, do, scale, causal=False, bottom_right=False, window_left=None, window_right=None):
    """fp64 reference backward for ONE sequence."""
    q, k, v, do = (t.detach().double().requires_grad_(t is not do) for t in (q, k, v, do))
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        s = s + _causal_bias(q.shape[-2], k.shape[-2], q.device, bottom_right, window_left, window_right)
    p = torch.softmax(s, dim=-1)
    o = p @ v
    o.backward(do)
    return q.grad, k.grad, v.grad


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _run(lens_q, lens_kv, h=2, d=_D, dtype=torch.bfloat16, token_major_stats=False):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    dev, b = "cuda", len(lens_q)
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cu_q, cu_k = [0], [0]
    for a, c in zip(lens_q, lens_kv):
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + c)
    scale = 1.0 / math.sqrt(d)
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda t: torch.randn(1, t, h, d, generator=g, device=dev, dtype=dtype) * 0.3

    # Packed [1, T, H, D] storage, handed over as logical [1, H, T, D] views --
    # the same orientation the dense path takes.
    q_p, k_p, v_p, do_p = rnd(t_q), rnd(t_kv), rnd(t_kv), rnd(t_q)
    o_p = torch.empty_like(q_p)
    lse_p = torch.empty(1, h, t_q, device=dev, dtype=torch.float32)

    # Forward reference, per sequence, filling O and the packed LSE.
    for i in range(b):
        qs, ks, vs = (
            x[0, cu[i] : cu[i] + L].transpose(0, 1).double() for x, cu, L in ((q_p, cu_q, lens_q[i]), (k_p, cu_k, lens_kv[i]), (v_p, cu_k, lens_kv[i]))
        )
        s = (qs @ ks.transpose(-1, -2)) * scale
        lse_p[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = torch.logsumexp(s, dim=-1).float()
        o_p[0, cu_q[i] : cu_q[i] + lens_q[i]] = (torch.softmax(s, dim=-1) @ vs).transpose(0, 1).to(dtype)

    view = lambda t: t.permute(0, 2, 1, 3)  # [1,T,H,D] -> logical [1,H,T,D]
    # The SAMPLES declare the ENVELOPE (B, H, S_max, D), the way a ragged graph
    # does; the packed buffers only show up at execute.  Declaring the packed
    # shape instead makes batch_size 1, and then the whole chain runs as if
    # there were a single sequence -- seq 0 exact, every other sequence never
    # visited.
    s_max_q, s_max_kv = max(lens_q), max(lens_kv)
    env = lambda n, s: torch.empty(1, n, s, h, d, device=dev, dtype=dtype)[0].permute(0, 2, 1, 3)
    eq, ekv = env(b, s_max_q), env(b, s_max_kv)
    e_stats = torch.empty(b, h, s_max_q, 1, device=dev, dtype=torch.float32)
    dq, dk, dv = torch.zeros_like(q_p), torch.zeros_like(k_p), torch.zeros_like(v_p)
    # TRANSPOSED, not reshaped: lse_p is head-major [1, H, T], so a reshape to
    # (T, H) reinterprets that memory instead of re-laying it and hands the
    # kernel garbage that looks correctly shaped.
    stats = lse_p[0].transpose(0, 1).contiguous() if token_major_stats else lse_p

    api = SdpaBwdDslSm100(
        sample_q=eq,
        sample_k=ekv,
        sample_v=ekv,
        sample_o=eq,
        sample_do=eq,
        sample_stats=e_stats,
        sample_dq=eq,
        sample_dk=ekv,
        sample_dv=ekv,
        scale_softmax=scale,
        thd=True,
        max_total_seq_len_q=t_q,
        max_total_seq_len_kv=t_kv,
        thd_stats_token_major=token_major_stats,
    )
    assert api.check_support()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    api.execute(
        view(q_p),
        view(k_p),
        view(v_p),
        view(o_p),
        view(do_p),
        stats,
        view(dq),
        view(dk),
        view(dv),
        workspace=ws,
        seq_q_lens=torch.tensor(lens_q, dtype=torch.int32, device=dev),
        seq_kv_lens=torch.tensor(lens_kv, dtype=torch.int32, device=dev),
    )
    torch.cuda.synchronize()

    bad = []
    for i in range(b):
        sl_q, sl_k = slice(cu_q[i], cu_q[i] + lens_q[i]), slice(cu_k[i], cu_k[i] + lens_kv[i])
        rq, rk, rv = _ref_bwd(
            q_p[0, sl_q].transpose(0, 1),
            k_p[0, sl_k].transpose(0, 1),
            v_p[0, sl_k].transpose(0, 1),
            do_p[0, sl_q].transpose(0, 1),
            scale,
        )
        for name, got, want in (
            ("dQ", dq[0, sl_q].transpose(0, 1), rq),
            ("dK", dk[0, sl_k].transpose(0, 1), rk),
            ("dV", dv[0, sl_k].transpose(0, 1), rv),
        ):
            # Collected, not asserted per gradient: which of the three a
            # sequence gets wrong is the attribution.  dK/dV come from the
            # m-major GEMM and dQ from the k-major one, and all three read a
            # workspace stage 2 wrote -- so "dQ alone" and "all three" point at
            # different kernels, and stopping at the first failure throws that
            # away.
            bad.append(f"seq {i} {name}: cos {_cos(got, want):.6f} (lens q={lens_q[i]} kv={lens_kv[i]})")
    failures = [m for m in bad if float(m.split("cos ")[1].split(" ")[0]) <= _TOL_COS]
    assert not failures, "\n".join(bad)


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_thd_self_attention(dt):
    """Three sequences of unequal length, none a tile multiple."""
    _run((300, 128, 200), (300, 128, 200), dtype=dt)


def test_thd_cross_attention():
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run((256, 100), (180, 300))


def test_thd_single_sequence_matches_dense_shape():
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run((512,), (512,))


def test_thd_stats_token_major():
    """The other packed Stats layout the forward can emit."""
    _run((300, 128), (300, 128), token_major_stats=True)


def test_thd_requires_declared_totals():
    """THD without max_total_seq_len_* is DECLINED, not silently mis-sized.

    Asserted rather than skipped, per the engine contract: the blocked
    workspace's row count and delta's row stride are both fixed at build time
    from the packed token capacity, and undeclared that capacity falls back to
    B * S_max -- more tokens than a packed buffer holds.
    """
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    dev, b, h, s_max = "cuda", 2, 2, 256
    env = torch.empty(b, s_max, h, _D, device=dev, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    e_stats = torch.empty(b, h, s_max, 1, device=dev, dtype=torch.float32)
    kw = dict(
        sample_q=env,
        sample_k=env,
        sample_v=env,
        sample_o=env,
        sample_do=env,
        sample_stats=e_stats,
        sample_dq=env,
        sample_dk=env,
        sample_dv=env,
        scale_softmax=1.0 / math.sqrt(_D),
        thd=True,
    )
    with pytest.raises(ValueError, match="max_total_seq_len"):
        SdpaBwdDslSm100(**kw).check_support()
    # ... and declared, the same graph is accepted.
    assert SdpaBwdDslSm100(**kw, max_total_seq_len_q=400, max_total_seq_len_kv=400).check_support()


# ---------------------------------------------------------------------------
# The graph path: ragged tensors -> lower_dsl_bwd -> packed views
# ---------------------------------------------------------------------------


def _plan_index(g, name=_ENGINE):
    for i in range(g.get_execution_plan_count()):
        if name in g.get_plan_name_at_index(i):
            return i
    return None


def _thd_case(
    lens_q, lens_kv, h, d, dtype, cap_q=None, cap_kv=None, poison=False, seed=7, causal=False, bottom_right=False, window_left=None, window_right=None
):
    """Packed Q/K/V/dO plus the forward's O and packed LSE, per sequence in fp64.

    ``cap_*`` over-allocates the packed buffers past the real totals; with
    ``poison`` the slack is NaN.  That is the #624 analogue for the backward:
    the declared totals only bound the buffers, so the rows between the current
    ``cu_*[B]`` and the capacity have to be kept out of reach by the kernels'
    OWN device-side descriptor clamps -- a NaN there would otherwise reach a
    ``0 * NaN`` and poison whole sequences.
    """
    dev, b = "cuda", len(lens_q)
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cap_q, cap_kv = cap_q or t_q, cap_kv or t_kv
    cu_q, cu_k = [0], [0]
    for a, c in zip(lens_q, lens_kv):
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + c)
    g = torch.Generator(device=dev).manual_seed(seed)

    def _pk(cap, live):
        x = torch.randn(1, cap, h, d, generator=g, device=dev, dtype=dtype) * 0.3
        if poison and cap > live:
            x[0, live:] = float("nan")
        return x

    q_p, do_p = _pk(cap_q, t_q), _pk(cap_q, t_q)
    k_p, v_p = _pk(cap_kv, t_kv), _pk(cap_kv, t_kv)
    o_p = torch.full_like(q_p, float("nan") if poison else 0.0)
    lse_p = torch.full((1, h, cap_q), float("nan") if poison else 0.0, device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    for i in range(b):
        if lens_q[i] == 0 or lens_kv[i] == 0:
            continue
        qs, ks, vs = (
            x[0, cu[i] : cu[i] + L].transpose(0, 1).double() for x, cu, L in ((q_p, cu_q, lens_q[i]), (k_p, cu_k, lens_kv[i]), (v_p, cu_k, lens_kv[i]))
        )
        sc = (qs @ ks.transpose(-1, -2)) * scale
        if causal:
            sc = sc + _causal_bias(lens_q[i], lens_kv[i], dev, bottom_right, window_left, window_right)
        lse_p[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = torch.logsumexp(sc, dim=-1).float()
        o_p[0, cu_q[i] : cu_q[i] + lens_q[i]] = (torch.softmax(sc, dim=-1) @ vs).transpose(0, 1).to(dtype)
    return SimpleNamespace(
        b=b, h=h, d=d, dtype=dtype, scale=scale, causal=causal, bottom_right=bottom_right, window_left=window_left, window_right=window_right,
        lens_q=list(lens_q), lens_kv=list(lens_kv), cu_q=cu_q, cu_k=cu_k,
        t_q=t_q, t_kv=t_kv, cap_q=cap_q, cap_kv=cap_kv,
        q=q_p, k=k_p, v=v_p, do=do_p, o=o_p, lse=lse_p,
    )  # fmt: skip


def _check(case, dq, dk, dv):
    """Per-sequence comparison against the fp64 reference.

    Collected, not asserted per gradient: which of the three a sequence gets
    wrong is the attribution.  dK/dV come from the m-major GEMM and dQ from the
    k-major one, and all three read a workspace stage 2 wrote -- so "dQ alone"
    and "all three" point at different kernels, and stopping at the first
    failure throws that away.
    """
    bad = []
    for i in range(case.b):
        if case.lens_q[i] == 0 or case.lens_kv[i] == 0:
            continue
        sl_q = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
        sl_k = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
        rq, rk, rv = _ref_bwd(
            case.q[0, sl_q].transpose(0, 1),
            case.k[0, sl_k].transpose(0, 1),
            case.v[0, sl_k].transpose(0, 1),
            case.do[0, sl_q].transpose(0, 1),
            case.scale,
            causal=case.causal,
            bottom_right=case.bottom_right,
            window_left=case.window_left,
            window_right=case.window_right,
        )
        for name, got, want in (
            ("dQ", dq[0, sl_q].transpose(0, 1), rq),
            ("dK", dk[0, sl_k].transpose(0, 1), rk),
            ("dV", dv[0, sl_k].transpose(0, 1), rv),
        ):
            bad.append(f"seq {i} {name}: cos {_cos(got, want):.6f} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})")
    failures = [m for m in bad if float(m.split("cos ")[1].split(" ")[0]) <= _TOL_COS]
    assert not failures, "\n".join(bad)


def _build_thd_bwd_graph(case, *, stats_layout="head_major", declare_totals=True, hkv=None, **sdpa_kwargs):
    """A ragged backward graph over ``case``'s packed buffers.

    Everything is declared as the ENVELOPE (B, H, S_max, D) plus a per-tensor
    ragged offset, which is how cuDNN spells a packed tensor -- the packed
    totals never appear as a dim, which is exactly why the lowering cannot
    reinterpret a variant-pack buffer through the port geometry.
    """
    b, h, d, dev = case.b, case.h, case.d, "cuda"
    hkv = h if hkv is None else hkv
    io = cudnn.data_type.HALF if case.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    s_max_q, s_max_kv = max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
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

    # Packed Stats, in one of the two layouts the forward emits. head_major is
    # (1, QH, head_stride) with a token capacity rounded up to 64 -- WIDER than
    # the packed total, which is the case that forced the head stride to reach
    # `compile()` as the LSE fake tensor's third extent.
    # Sized on the DECLARED capacity, not the live total: the adapter binds the
    # Stats view at min(B * S_max, declared), so the head stride has to cover
    # that -- which is what test_graph_thd_nan_capacity_tail exercises.
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
        q=t["q"], k=t["k"], v=t["v"], o=t["o"], dO=t["do"], stats=st,
        attn_scale=case.scale,
        use_padding_mask=True,
        seq_len_q=tq_len,
        seq_len_kv=tk_len,
    )  # fmt: skip
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


def _run_graph(lens_q, lens_kv, *, h=2, d=_D, dtype=torch.bfloat16, stats_layout="head_major", poison=False, pad_cap=0, **kw):
    """Build the ragged graph, PIN the engine, execute, compare per sequence.

    ``use_causal_mask`` / ``use_causal_mask_bottom_right`` thread through ``kw``
    to the graph AND back into the case, so the fp64 reference masks with the
    same per-sequence geometry the kernel does.  Passing one without the other
    would compare a masked kernel against an unmasked reference.
    """
    case = _thd_case(
        lens_q,
        lens_kv,
        h,
        d,
        dtype,
        cap_q=sum(lens_q) + pad_cap,
        cap_kv=sum(lens_kv) + pad_cap,
        poison=poison,
        causal=bool(kw.get("use_causal_mask") or kw.get("use_causal_mask_bottom_right") or kw.get("diagonal_band_right_bound") is not None),
        bottom_right=bool(kw.get("use_causal_mask_bottom_right")),
        window_left=kw.get("sliding_window_length"),
        window_right=kw.get("diagonal_band_right_bound"),
    )
    g, vp, (dq_t, dk_t, dv_t) = _build_thd_bwd_graph(case, stats_layout=stats_layout, **kw)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    dq, dk, dv = (torch.zeros_like(x) for x in (case.q, case.k, case.v))
    vp.update({dq_t: dq, dk_t: dk, dv_t: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    for name, x in (("dQ", dq), ("dK", dk), ("dV", dv)):
        live = x[0, : case.t_q] if name == "dQ" else x[0, : case.t_kv]
        assert torch.isfinite(live).all(), f"{name} has non-finite values in the packed region"
    _check(case, dq, dk, dv)
    return case, dq, dk, dv


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_self_attention(dt):
    """The whole point: a ragged graph reaches the kernels through the engine."""
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt)


def test_graph_thd_cross_attention():
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run_graph((256, 100), (180, 300))


@pytest.mark.parametrize("layout", ("head_major", "token_major"))
def test_graph_thd_stats_packings(layout):
    """Both packed Stats layouts the forward can emit.

    A frozenset-style claim: the row reads either packing, so each is its own
    test rather than one test over the pair (engine contract section 9). The
    head-major case additionally carries a head stride WIDER than the packed
    total -- the forward rounds its token capacity up to 64.
    """
    _run_graph((300, 128), (300, 128), stats_layout=layout)


def test_graph_thd_zero_length_sequence():
    """A sequence with no tokens must not corrupt its neighbours."""
    _run_graph((256, 0, 128), (256, 0, 128))


@pytest.mark.parametrize(
    "lens_q,lens_kv",
    (
        ((0, 128, 256), (192, 128, 256)),
        ((192, 128, 256), (0, 128, 256)),
    ),
    ids=("empty_q_side", "empty_kv_side"),
)
def test_graph_thd_one_sided_empty_sequence(lens_q, lens_kv):
    """A sequence empty on ONE side only -- the other side still has rows.

    ``test_graph_thd_zero_length_sequence`` empties BOTH sides, which hides this:
    there the output length is 0 too, so the epilogue's store-skip covers it.
    Empty on one side only, the group's REDUCTION axis is empty (``nkt == 0``,
    zero mainloop iterations, ``scale_d`` still False) while its OUTPUT rows
    exist and must be written -- so the accumulator is never initialised and the
    epilogue has to store zeros explicitly.

    Zero is not a convention here, it is the answer: with no queries a sequence
    contributes nothing to its keys' and values' gradients, and with no keys
    there is no gradient to receive.

    Verified RED without the epilogue's ``_thd_k_len`` guard: ``empty_kv_side``
    returns ``max |dQ| = 2.9e+20``.  Note ``empty_q_side`` PASSED unguarded on
    that run -- the residue it read happened to be zeros.  Both cases are kept
    for exactly that reason: which one shows the bug depends on what the
    previous tile left in TMEM, so a single case is not a reliable detector, and
    the residue being FINITE (2.9e+20 is) means ``_run_graph``'s ``isfinite``
    assertion does not catch it either.  The explicit zero check below is the
    detector.
    """
    case, dq, dk, dv = _run_graph(lens_q, lens_kv)
    # _check skips a sequence that is empty on either side (there is no reference
    # for it), so assert the empty one's own gradients are exactly zero here.
    i = 0
    sl_q = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
    sl_k = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
    for name, got in (("dQ", dq[0, sl_q]), ("dK", dk[0, sl_k]), ("dV", dv[0, sl_k])):
        assert got.numel() == 0 or not got.any(), f"{name} of a one-sided-empty sequence must be exactly zero, got max |{got.abs().max().item()}|"


def test_graph_thd_nan_capacity_tail():
    """Declared totals larger than the live packing, with a NaN tail.

    The #624 analogue. ``max_total_seq_len_*`` is a MAXIMUM, so the rows between
    the current ``cu_*[B]`` and it are inside every packed view -- masked, but
    still multiplied. Only the kernels' device-side descriptor clamps keep them
    out, and ``0 * NaN`` is NaN, so this is the test that proves the clamps.
    """
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384)


def test_graph_thd_b1_matches_dense_shape():
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run_graph((512,), (512,))


# --- causal family ----------------------------------------------------------
#
# The per-sequence diagonal is the whole risk here.  Stage 2 masks from the
# metadata lengths, but stage 3 reads a workspace whose masked tiles stage 2
# never wrote -- so these also assert the THD zero-fill, and they are the tests
# that would catch it being dropped as "the trim covers it".  Every case uses
# unequal lengths, none a tile multiple: with equal lengths a diagonal built
# from the envelope S_max would agree with the per-sequence one and the bug
# would hide.


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_causal(dt):
    """Top-left causal over three unequal sequences."""
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt, use_causal_mask=True)


def test_graph_thd_causal_zero_length_sequence():
    """Causal plus an empty sequence: the two skip paths meeting.

    The masked tiles stage 2 never wrote and the zero-extent descriptor an empty
    sequence needs are independent mechanisms, and this is the only case where
    both are live at once.
    """
    _run_graph((256, 0, 128), (256, 0, 128), use_causal_mask=True)


def test_graph_thd_causal_cross_attention():
    """Unequal Q and KV lengths under causal.

    Top-left aligned, so the diagonal does not move -- but S_kv[b] != S_q[b]
    makes the number of live kv tiles per q row differ per sequence, which a
    trim keyed on absolute workspace rows gets wrong.
    """
    _run_graph((256, 100), (180, 300), use_causal_mask=True)


def test_graph_thd_causal_bottom_right():
    """Bottom-right alignment: the diagonal offset IS per sequence.

    ``S_kv[b] - S_q[b]`` is 56 for the first sequence and 200 for the second --
    exactly the quantity stage 3's host-scalar ``causal_shift`` cannot express,
    and the reason THD drops that trim rather than trying to.

    Both sequences keep ``S_kv >= S_q`` on purpose.  Shorter the other way, the
    leading ``S_q - S_kv`` rows have no unmasked column at all, and a fully
    masked softmax row is NaN in the fp64 reference too -- a degenerate mask,
    not a kernel property, and it would make this test assert nothing.
    """
    _run_graph((200, 100), (256, 300), use_causal_mask_bottom_right=True)


def test_graph_thd_causal_swa():
    """Sliding window: a LEFT bound on top of the causal right one.

    A second per-sequence band edge, and the one stage 2 reaches through
    ``kv_left`` rather than ``kv_right`` -- so it exercises the other end of
    ``_kv_tile_bounds`` under packed lengths.
    """
    _run_graph((300, 128, 200), (300, 128, 200), use_causal_mask=True, sliding_window_length=64)


def test_graph_thd_causal_right_band():
    """Right-band widening: the causal upper bound pushed out by an offset.

    The fourth member of the single ``thd_causal`` flag -- it covers causal, SWA,
    bottom-right AND band widening, so each is its own accept test (engine
    contract section 9).  ``diagonal_band_right_bound`` is exclusive with
    ``use_causal_mask``; it IS the upper bound.
    """
    _run_graph((300, 128, 200), (300, 128, 200), diagonal_band_right_bound=64)


def test_graph_thd_causal_nan_capacity_tail():
    """Causal with a NaN tail past the declared totals.

    Causal reads FEWER kv tiles, so it visits a different set of workspace rows
    than the no-mask case the original tail test covers -- and the zero-fill now
    writes the whole blocked buffer, which is a second thing that must not reach
    the poisoned capacity rows.
    """
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384, use_causal_mask=True)


# --- rejects: every THD conjunction the row declines ------------------------


def _thd_mismatch(lens_q=(256, 128), lens_kv=(256, 128), *, h=2, hkv=None, stats_layout="head_major", **kw):
    """``mismatch()`` for a ragged backward graph, or None if it is served."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    case = _thd_case(lens_q, lens_kv, h, _D, torch.bfloat16)
    g, _, _ = _build_thd_bwd_graph(case, stats_layout=stats_layout, hkv=hkv, **kw)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as exc:
        return f"refused by the node: {exc}"  # a decline before engine selection is also a decline
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    assert facts is not None
    return mismatch(spec.capabilities, facts)


def test_graph_thd_accepts_the_plain_case():
    """The counterweight to the rejects below: the same builder, no extras, IS
    served -- so a reject test that passes for the wrong reason (a broken
    graph, a builder typo) fails here first."""
    assert _thd_mismatch() is None


def test_accept_thd_causal():
    """The inverse of the reject this used to be (test/AGENTS.md: invert, do not
    delete).  Stage 2 masks from the per-sequence metadata lengths and stage 3
    drops its absolute-row K-trim under THD, so the causal family is served."""
    assert _thd_mismatch(use_causal_mask=True) is None
    assert _thd_mismatch(use_causal_mask_bottom_right=True) is None
    assert _thd_mismatch(use_causal_mask=True, sliding_window_length=64) is None
    assert _thd_mismatch(diagonal_band_right_bound=64) is None


def test_reject_thd_gqa():
    """The dK/dV partials would have to be packed per Q head."""
    reason = _thd_mismatch(h=4, hkv=2)
    assert reason is not None and "GQA" in reason


def test_reject_thd_without_declared_totals():
    """``scratch_workspace_bytes()`` is a BUILD-time function and the blocked
    row count comes from the packed totals, so the declaration is required --
    and declined as a typed mismatch, not raised at execute."""
    reason = _thd_mismatch(declare_totals=False)
    assert reason is not None and "max_total_seq_len" in reason


def test_reject_thd_dense_stats():
    """A ragged graph with a DENSE per-batch Stats tensor.

    Legal cuDNN, and a trap: its stride (H*S_max, S_max, 1, 1) READS as
    head-major with head stride S_max, while its storage is per-batch
    rectangles. Declining on the absent ragged offset is what stops the packing
    inference from mis-reading it.
    """
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    case = _thd_case((256, 128), (256, 128), 2, _D, torch.bfloat16)
    g, _, _ = _build_thd_bwd_graph(case)
    # Strip the ragged offset from Stats only; everything else stays packed.
    node = g.nodes[0]
    node.inputs["stats"].set_ragged_offset(None)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    reason = mismatch(spec.capabilities, facts)
    assert reason is not None and "ragged" in reason
