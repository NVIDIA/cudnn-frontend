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


def _below_tol(msg):
    """True when a collected ``cos ...`` line is a FAILURE.

    Written as ``not (x > tol)`` and not ``x <= tol`` on purpose: ``_cos``
    returns NaN whenever a gradient holds NaN or Inf, and every comparison
    against NaN is False -- so ``<=`` would let a fully poisoned gradient
    through as a pass.  ``_run_graph`` has a separate ``isfinite`` guard, but
    ``_run`` (the direct-adapter path) does not, and this is its only net.
    """
    return not (float(msg.split("cos ")[1].split(" ")[0]) > _TOL_COS)


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
    failures = [m for m in bad if _below_tol(m)]
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
    lens_q,
    lens_kv,
    h,
    d,
    dtype,
    cap_q=None,
    cap_kv=None,
    poison=False,
    seed=7,
    causal=False,
    bottom_right=False,
    window_left=None,
    window_right=None,
    hkv=None,
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

    # K/V carry H_kv heads under GQA; Q/dO/O always carry H_q.
    hkv = h if hkv is None else hkv
    grp = h // hkv

    def _pk(cap, live, nh):
        x = torch.randn(1, cap, nh, d, generator=g, device=dev, dtype=dtype) * 0.3
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
        # GQA: one KV head feeds `grp` consecutive Q heads. repeat_interleave,
        # not repeat -- the kernel's mapping is `kv_head = q_head // grp`, so
        # heads 0..grp-1 share KV head 0. Getting this backwards still produces
        # a plausible-looking O and would only show as a cosine miss.
        if grp > 1:
            ks, vs = ks.repeat_interleave(grp, dim=0), vs.repeat_interleave(grp, dim=0)
        sc = (qs @ ks.transpose(-1, -2)) * scale
        if causal:
            sc = sc + _causal_bias(lens_q[i], lens_kv[i], dev, bottom_right, window_left, window_right)
        lse_p[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = torch.logsumexp(sc, dim=-1).float()
        o_p[0, cu_q[i] : cu_q[i] + lens_q[i]] = (torch.softmax(sc, dim=-1) @ vs).transpose(0, 1).to(dtype)
    return SimpleNamespace(
        b=b, h=h, hkv=hkv, d=d, dtype=dtype, scale=scale, causal=causal, bottom_right=bottom_right, window_left=window_left, window_right=window_right,
        lens_q=list(lens_q), lens_kv=list(lens_kv), cu_q=cu_q, cu_k=cu_k,
        t_q=t_q, t_kv=t_kv, cap_q=cap_q, cap_kv=cap_kv,
        q=q_p, k=k_p, v=v_p, do=do_p, o=o_p, lse=lse_p,
    )  # fmt: skip


def _check(case, dq, dk, dv, hkv=None):
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
        _grp = case.h // case.hkv
        _rep = (lambda x: x.repeat_interleave(_grp, dim=0)) if _grp > 1 else (lambda x: x)
        rq, rk, rv = _ref_bwd(
            case.q[0, sl_q].transpose(0, 1),
            _rep(case.k[0, sl_k].transpose(0, 1)),
            _rep(case.v[0, sl_k].transpose(0, 1)),
            case.do[0, sl_q].transpose(0, 1),
            case.scale,
            causal=case.causal,
            bottom_right=case.bottom_right,
            window_left=case.window_left,
            window_right=case.window_right,
        )
        if case.hkv != case.h:
            # GQA: the reference ran with K/V BROADCAST to every Q head, so its
            # dK/dV are per Q head. The kernel returns them already folded onto
            # the KV heads, so sum each group before comparing -- comparing
            # against one group member would pass for MQA and fail for GQA, and
            # comparing unfolded would fail for both.
            rk = rk.reshape(case.hkv, _grp, *rk.shape[1:]).sum(1)
            rv = rv.reshape(case.hkv, _grp, *rv.shape[1:]).sum(1)
        for name, got, want in (
            ("dQ", dq[0, sl_q].transpose(0, 1), rq),
            ("dK", dk[0, sl_k].transpose(0, 1), rk),
            ("dV", dv[0, sl_k].transpose(0, 1), rv),
        ):
            bad.append(f"seq {i} {name}: cos {_cos(got, want):.6f} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})")
    failures = [m for m in bad if _below_tol(m)]
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


def _run_graph(lens_q, lens_kv, *, h=2, hkv=None, d=_D, dtype=torch.bfloat16, stats_layout="head_major", poison=False, pad_cap=0, check=True, **kw):
    """Build the ragged graph, PIN the engine, execute, compare per sequence.

    ``use_causal_mask`` / ``use_causal_mask_bottom_right`` thread through ``kw``
    to the graph AND back into the case, so the fp64 reference masks with the
    same per-sequence geometry the kernel does.  Passing one without the other
    would compare a masked kernel against an unmasked reference.

    ``check=False`` skips the per-sequence comparison (a caller that inspects the
    workspace words first and wants the gradient verdict separately); the
    returned ``case`` carries the executed ``workspace`` either way.
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
        hkv=hkv,
    )
    g, vp, (dq_t, dk_t, dv_t) = _build_thd_bwd_graph(case, stats_layout=stats_layout, hkv=hkv, **kw)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    # dK / dV carry H_kv heads under GQA, so they are shaped from the graph's
    # KV head count, not from the Q tensors.
    _kvh = h if hkv is None else hkv
    dq = torch.zeros_like(case.q)
    dk, dv = (torch.zeros(1, case.cap_kv, _kvh, d, device="cuda", dtype=dtype) for _ in range(2))
    vp.update({dq_t: dq, dk_t: dk, dv_t: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    case.workspace, case.graph, case.vp = ws, g, vp
    if not check:
        return case, dq, dk, dv
    for name, x in (("dQ", dq), ("dK", dk), ("dV", dv)):
        live = x[0, : case.t_q] if name == "dQ" else x[0, : case.t_kv]
        assert torch.isfinite(live).all(), f"{name} has non-finite values in the packed region"
    _check(case, dq, dk, dv, hkv=_kvh)
    return case, dq, dk, dv


_TWIN_WATCHDOG_S = 900.0  # see test_sdpa_bwd_dsl_sm100._TWIN_WATCHDOG_S


@pytest.fixture(params=[False, True], ids=["4x1", "2x2"])
def stage2_datapath(request, monkeypatch):
    """Both stage-2 datapaths of the chain on the THD leg (see ``test_sdpa_bwd_dsl_sm100.stage2_datapath``): the twin's
    THD arm changes the q-block unit (``q_tile_idx * 4 + cta_id_x`` over 64-row blocks) and the 64-row store box against
    the 128-row blocked workspace, and its four-CTA K / V ring must collapse to zero trips on a dead unit.  The 2x2 arm
    runs under the process watchdog, as in the dense file (no longer ``gpu_exclusive``: 12 x 100 launches beside a 4x1
    load process, 0 hangs, with the shipped poll form)."""
    import contextlib

    from frost_test_utils import process_watchdog

    from cudnn.sdpa.bwd import api_dsl

    monkeypatch.setattr(api_dsl, "STAGE2_2X2", request.param)
    served = []
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag == "sdpa_bwd_sm100_stage2":
            served.append(path.rsplit("/", 1)[-1])
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    guard = process_watchdog(_TWIN_WATCHDOG_S, f"the 2x2 stage-2 arm of {request.node.nodeid}") if request.param else contextlib.nullcontext()
    with guard:
        yield request.param
    want = api_dsl._SM100_STAGE2_FILE_2X2.rsplit("/", 1)[-1] if request.param else api_dsl._SM100_STAGE2_FILE.rsplit("/", 1)[-1]
    assert served and all(s == want for s in served), f"stage 2 served by {served}, expected {want} (STAGE2_2X2={request.param})"


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_self_attention(dt, stage2_datapath):
    """The whole point: a ragged graph reaches the kernels through the engine."""
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt)


def test_graph_thd_cross_attention(stage2_datapath):
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run_graph((256, 100), (180, 300))


@pytest.mark.parametrize("layout", ("head_major", "token_major"))
def test_graph_thd_stats_packings(layout, stage2_datapath):
    """Both packed Stats layouts the forward can emit.

    A frozenset-style claim: the row reads either packing, so each is its own
    test rather than one test over the pair (engine contract section 9). The
    head-major case additionally carries a head stride WIDER than the packed
    total -- the forward rounds its token capacity up to 64.
    """
    _run_graph((300, 128), (300, 128), stats_layout=layout)


def test_graph_thd_zero_length_sequence(stage2_datapath):
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
def test_graph_thd_one_sided_empty_sequence(lens_q, lens_kv, stage2_datapath):
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


def test_graph_thd_kv_len_64_masks_the_upper_column_half(stage2_datapath):
    """The 2x2 lane-map detector: a GENUINELY masked case whose band edge is column 64 of a tile.

    THD compiles the per-cell mask (``thd_varlen`` -> ``MASK_PADDED``), and a kv length of 64 mod 128 masks exactly
    columns [64, 128) of the sequence's last kv tile for EVERY q row -- sequence 0 (kv 64) in tile 0, sequence 1
    (kv 192) in tile 1.  Under the fused 2x2 kernel the two column halves of a tile live in different warps
    (``col_half = tid // 64``: warps 0-1 hold columns [0, 64), warps 2-3 hold [64, 128), the same rows); the mask's
    ``kv_col_base`` picks the half, so a lane map that swaps the halves at the mask site masks the wrong 64 columns of
    every row -- correct on dense (no mask code is traced) and wrong here.  The per-sequence fp64 reference sees the
    difference in all three gradients.  Proven RED once by swapping ``col_half`` at the ``apply_mask_chunk`` call
    (``1 - col_half``): the 2x2 arm fails both sequences on dQ / dK / dV while the 4x1 arm stays green
    (``lane_d512_bprop/fix/RED_lane_map_swap.log``).  The dense ``S_kv = 96`` case in ``test_sdpa_bwd_dsl_sm100.py``
    is NOT a detector: dense padding is not served, so it compiles ``MASK_NONE`` and its 32 tail columns are TMA
    zero-filled, not masked.
    """
    _run_graph((128, 64), (64, 192))


def test_graph_thd_nan_capacity_tail(stage2_datapath):
    """Declared totals larger than the live packing, with a NaN tail.

    The #624 analogue. ``max_total_seq_len_*`` is a MAXIMUM, so the rows between
    the current ``cu_*[B]`` and it are inside every packed view -- masked, but
    still multiplied. Only the kernels' device-side descriptor clamps keep them
    out, and ``0 * NaN`` is NaN, so this is the test that proves the clamps.
    """
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384)


def test_graph_thd_nan_capacity_tail_unaligned_last_sequence():
    """The capacity tail reached by a K-TILE OVERSHOOT, not by an M tile.

    ``test_graph_thd_nan_capacity_tail`` uses lengths that are multiples of the
    64-wide k tile, so every k tile stops exactly at ``cu_*[B]`` and the packed
    B operand is never read past it.  Here the LAST sequence is 100 tokens: its
    reduction runs ceil(100/64) = 2 tiles = 128 rows, so the final tile reads 28
    rows PAST the packed total, into the caller's declared-but-unwritten
    capacity.

    Those rows meet A's block padding, which stage 2 zeroed -- and ``0 * NaN``
    is NaN, so the whole gradient goes.  Only the packed-total-clamped B
    descriptor keeps them out (the GridConstant one is built at the buffer's
    CAPACITY, and a declared ``max_total_seq_len`` cannot help: it is a maximum,
    while the row that must read zero is ``cu_*[B]``, which moves every step).

    Both GEMM orientations are covered by one shape: 100 is a non-multiple of 64
    on the q side (the m-major dV/dK reduction) and on the kv side (the k-major
    dQ one).
    """
    _run_graph((256, 100), (256, 100), poison=True, pad_cap=384)


def test_graph_thd_b1_matches_dense_shape():
    """B == 1 is the degenerate packing: it must agree with the dense answer."""
    _run_graph((512,), (512,))


@pytest.mark.parametrize("hkv", (2, 1), ids=("gqa_group2", "mqa"))
def test_graph_thd_gqa(hkv, stage2_datapath):
    """Packed GQA / MQA end to end.

    The dK/dV partials are ONE PER Q HEAD over the packed kv axis, then folded
    onto the KV heads. Two things this pins that a single group size would not:
    MQA (group 4) collapses every Q head onto one KV head, and the reference
    must SUM a group's contributions -- comparing against one member passes for
    MQA and fails for GQA, so both sizes run.
    """
    _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=hkv)


def test_graph_thd_gqa_causal():
    """GQA together with a per-sequence causal mask -- the two THD conjunctions
    that were declined separately, now exercised together."""
    _run_graph((256, 100), (256, 100), h=4, hkv=2, use_causal_mask=True)


def test_graph_thd_gqa_cross_attention_and_zero_length():
    """GQA over unequal Q/KV totals with an empty sequence in the middle.

    The dK/dV partial buffer is sized on the packed KV capacity while dQ rides
    the Q one, so unequal totals are what would catch the two being conflated.
    """
    _run_graph((256, 0, 100), (180, 0, 300), h=4, hkv=2)


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
def test_graph_thd_causal(dt, stage2_datapath):
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


def test_graph_thd_causal_bottom_right(stage2_datapath):
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


def test_graph_thd_causal_swa(stage2_datapath):
    """Sliding window: a LEFT bound on top of the causal right one.

    A second per-sequence band edge, and the one stage 2 reaches through
    ``kv_left`` rather than ``kv_right`` -- so it exercises the other end of
    ``_kv_tile_bounds`` under packed lengths.
    """
    _run_graph((300, 128, 200), (300, 128, 200), use_causal_mask=True, sliding_window_length=64)


def test_graph_thd_causal_right_band(stage2_datapath):
    """Right-band widening: the causal upper bound pushed out by an offset.

    The fourth causal-family band, and each gets its own accept test rather than
    one test for "causal" (engine contract section 9).  That granularity is what
    justified deleting the ``thd_causal`` conjunction flag outright: all four
    bands are served, so nothing is left for the flag to decline.
    ``diagonal_band_right_bound`` is exclusive with ``use_causal_mask``; it IS
    the upper bound.
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


# --- head chunking: more than one stage-2 launch over ONE metadata buffer ----
#
# Every other THD case in this file fits its heads in one chunk, so the chain
# issues ONE stage-2 launch and the persistent scheduler's two words -- the live
# unit total and the claim counter the setup launch wrote -- serve exactly one
# launch.  A d512 plan whose S+dS slab exceeds the budget (B=1 H=128 S>=4096 is
# enough) loops `H / chunk` stage-2 launches over the SAME words, and each
# launch decodes units with the CHUNK's head count.  Two things must then hold
# per launch, and both failed silently until this section existed:
#
# * `live` must count the heads ONE LAUNCH decodes (`sum_b ceil(s_b/256) *
#   chunk`), not every head of the plan -- otherwise the scheduler hands out
#   dead units that cost the whole per-tile protocol for nothing;
# * the claim counter must be re-seeded before EVERY launch -- otherwise launch
#   k >= 2 finds it already past `live`, processes only the units its clusters
#   were pre-assigned by blockIdx, and stage 3 reads the PREVIOUS chunk's S/dS
#   for the rest: wrong, finite gradients for every head past the first
#   `clusters` units of each later chunk.
#
# The second bug only shows when a launch has MORE live units than clusters
# (SM count / 4: 37 on a 148-SM B200), so the lengths below carry 12 q-units of
# 256 rows and the forced chunk is >= 4 -- a precondition each test asserts
# rather than assumes.

_CHUNK_LENS = (300, 128, 200) * 3  # 9 sequences; ceil(s/256) = 2 + 1 + 1 per triple = 12 q-units
_CHUNK_Q_UNITS = sum(-(-s // 256) for s in _CHUNK_LENS)


def _force_thd_head_chunk(chunk):
    """Patch the THD head-chunk rule to answer ``chunk`` and record what it was asked (the blocked row count, kv columns
    and bytes per element -- the test rebuilds the workspace carve from them).  Returns ``(patch, seen)``."""
    from unittest.mock import patch

    from cudnn.sdpa.bwd import api_dsl

    seen = {}

    def rule(h_q, ws_rows, s_kv, bpe, budget=api_dsl._SM100_WS_BUDGET_BYTES, group=1, **kw):
        assert h_q % chunk == 0 and chunk % group == 0, (h_q, chunk, group)
        seen.update(h_q=h_q, ws_rows=ws_rows, s_kv=s_kv, bpe=bpe)
        return chunk

    return patch.object(api_dsl, "_sm100_head_chunk_thd", side_effect=rule), seen


def _thd_meta_words(case, h, chunk, seen):
    """The ``(5B+5,)`` int32 metadata words out of the executed workspace.

    Re-derives the carve ``scratch_workspace_bytes`` documents -- delta, then S
    and dS, then the metadata -- rather than reaching into the plan: the carve
    is a contract the test should be able to spell."""
    from cudnn.sdpa.fwd.api_dsl import ws_align

    b = case.b
    delta = ws_align(h * (-(-case.cap_q // 128) * 128) * 4)
    slab = ws_align(chunk * seen["ws_rows"] * seen["s_kv"] * seen["bpe"])
    off = delta + 2 * slab
    return case.workspace[off : off + (5 * b + 5) * 4].view(torch.int32).cpu()


def _clusters():
    from cudnn.sdpa.bwd.api_dsl import _sm100_device_clusters

    return _sm100_device_clusters(torch.device("cuda"), 4)


def _assert_chunking_is_observable(chunk):
    live = _CHUNK_Q_UNITS * chunk
    assert (
        live > _clusters()
    ), f"precondition: a launch needs more live units ({live}) than clusters ({_clusters()}) for a missed claim reset to show; grow _CHUNK_LENS"


_CHUNK_CELLS = [
    pytest.param(16, 16, 8, {}, id="mha_h16_chunk8"),
    pytest.param(16, 4, 4, {}, id="gqa_h16_hkv4_chunk4"),
    pytest.param(16, 16, 8, dict(use_causal_mask=True), id="mha_h16_chunk8_causal"),
]


@pytest.mark.parametrize("h, hkv, chunk, kw", _CHUNK_CELLS)
def test_graph_thd_forced_head_chunks_match_reference_and_unchunked(h, hkv, chunk, kw, stage2_datapath):
    """Heads split over ``h // chunk`` stage-2 launches: every sequence's dQ/dK/dV against the fp64 reference, and -- for
    the MHA cell -- BITWISE the one-chunk plan over the same inputs (same per-tile MMA order, same GEMM k-walk per head;
    the chunk only moves `head_base`), on int16 views (the twin-bitwise precedent, sdpa/AGENTS.md 2x2 lessons)."""
    _assert_chunking_is_observable(chunk)
    guard, seen = _force_thd_head_chunk(chunk)
    with guard:
        _, dq, dk, dv = _run_graph(_CHUNK_LENS, _CHUNK_LENS, h=h, hkv=hkv, **kw)
    assert seen["h_q"] == h, seen
    if hkv == h and not kw:
        _, dq0, dk0, dv0 = _run_graph(_CHUNK_LENS, _CHUNK_LENS, h=h, hkv=hkv, **kw)
        for name, a, b in (("dQ", dq, dq0), ("dK", dk, dk0), ("dV", dv, dv0)):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name}: the {h // chunk}-launch plan is not bitwise the one-launch plan"


def test_graph_thd_forced_head_chunks_publish_live_and_reseed_the_claim_counter(stage2_datapath):
    """The two scheduler words after an execute of ``h // chunk`` stage-2 launches:

    * ``live`` (``meta[4B+2]``) == ``sum_b ceil(s_b/256) * chunk`` -- the unit count ONE launch decodes;
    * the claim counter (``meta[4B+3]``) == ``live + clusters`` -- the LAST launch started from a fresh seed of
      ``clusters``, claimed the ``live - clusters`` units past the blockIdx-assigned ones, and every cluster then drew
      exactly one invalid claim.  A counter never re-seeded reads ``clusters + live_published + clusters`` per launch
      instead, and a ``live`` published for every head of the plan is ``h / chunk`` times too large.
    """
    h, chunk = 16, 8
    _assert_chunking_is_observable(chunk)
    guard, seen = _force_thd_head_chunk(chunk)
    with guard:
        case, *_ = _run_graph(_CHUNK_LENS, _CHUNK_LENS, h=h, check=False)
    words = _thd_meta_words(case, h, chunk, seen)
    b, live = case.b, _CHUNK_Q_UNITS * chunk
    assert int(words[2 * b]) == case.t_q and int(words[3 * b + 1]) == case.t_kv, "the carve re-derivation is off: cu_q[B] / cu_k[B] do not read back"
    assert int(words[4 * b + 2]) == live, f"live word {int(words[4 * b + 2])} != {_CHUNK_Q_UNITS} q-units x chunk {chunk} = {live}"
    assert int(words[4 * b + 3]) == live + _clusters(), f"claim counter {int(words[4 * b + 3])} after the last launch != live {live} + clusters {_clusters()}"


def _cuda_kernel_names(fn):
    """Names of the CUDA kernels ONE call of ``fn`` launches (torch.profiler over CUPTI), in launch order."""
    from torch.autograd import DeviceType

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    cuda = [ev for ev in prof.events() if ev.device_type == DeviceType.CUDA]
    mem = [ev.name for ev in cuda if ev.name.startswith(("Memcpy", "Memset"))]
    assert not mem, f"execute() must launch kernels only (Rules 1-2: no hidden memset / memcpy): {mem}"
    evs = [ev for ev in cuda if not ev.name.startswith(("Memcpy", "Memset"))]
    evs.sort(key=lambda ev: ev.time_range.start)
    return [ev.name for ev in evs]


@pytest.mark.parametrize("h, hkv, chunk, kw", _CHUNK_CELLS)
def test_graph_thd_launch_count_is_the_promised_chain(h, hkv, chunk, kw, stage2_datapath):
    """Launches per execute, from a CUPTI trace, == exactly the chain the host promises (Rules 1-2: no hidden launches):
    ``setup + dot [+ zero-fill] + chunks * (clamp + stage 2 + (2 + group) * (patch + GEMM)) [+ dkv_reduce]`` -- for a dense-mask
    MHA plan ``2 + 8 * chunks``.  The per-chunk THD helpers are counted by name too, so a chunk that silently skipped (or
    doubled) a launch is attributed."""
    guard, _ = _force_thd_head_chunk(chunk)
    with guard:
        case, *_ = _run_graph(_CHUNK_LENS, _CHUNK_LENS, h=h, hkv=hkv, check=False, **kw)
    names = _cuda_kernel_names(lambda: case.graph.execute(case.vp, case.workspace))
    group, chunks, causal = h // hkv, h // chunk, bool(kw.get("use_causal_mask"))
    gemms = chunks * (2 + group)
    want = 2 + int(causal) + chunks * 2 + 2 * gemms + int(group > 1)
    count = lambda sub: sum(sub in n for n in names)  # noqa: E731
    assert len(names) == want, f"{len(names)} launches != {want} (chunks {chunks}, group {group}, causal {causal}); trace:\n" + "\n".join(names)
    assert count("thd_bwd_setup") == 1 and count("clamp_thd_input_descs") == chunks and count("thd_patch_descs") == gemms, names
    assert count("bprop_matmul") == gemms and count("dot_do_o") == 1 and count("zero_workspace") == int(causal) and count("dkv_reduce") == int(group > 1), names


def test_thd_head_chunk_matches_dense_at_equal_tile_multiple_lengths():
    """The THD head-chunk rule gives an equal-length packed plan the DENSE plan's chunk (so the same launch count), and the
    blocked padding may carry the slab past the budget by at most ``budget / _SM100_WS_THD_PAD_SLACK``.  Without the token
    rows the ORIGINAL rule halves the chunk at the budget edge (B=1 / B=4 S=8192: 16 -> 8 / 4 -> 2; B=4 S=2048: 64 -> 32;
    the GQA cases too, since develop's divisor rule (#956) no longer floors the chunk at the group) --
    the regression this pins.  Many short sequences against a long kv (B=64 x 100 tokens, S_kv 8192) stay charged in full:
    the slack caps the overshoot, it does not hand out the token chunk unconditionally.  Host-only arithmetic."""
    from cudnn.sdpa.bwd.api_dsl import _SM100_WS_BUDGET_BYTES, _SM100_WS_THD_PAD_SLACK, _sm100_head_chunk, _sm100_head_chunk_thd

    bpe, budget = 2, _SM100_WS_BUDGET_BYTES
    rows_of = lambda tokens, b: -(-(tokens + b * 128) // 128) * 128  # noqa: E731  the adapter's blocked row count
    for b, s, h, group, old in (
        (1, 8192, 128, 1, 8),
        (4, 8192, 128, 1, 2),
        (4, 2048, 128, 1, 32),
        (8, 1024, 128, 1, 64),
        (1, 8192, 128, 16, 8),
        (2, 4096, 64, 8, 16),
    ):
        dense = _sm100_head_chunk(b, h, s, s, bpe, group=group)
        rows = rows_of(b * s, b)
        thd = _sm100_head_chunk_thd(h, rows, s, bpe, group=group, t_rows=b * s)
        assert thd == dense, f"B={b} S={s} H={h} group={group}: THD chunk {thd} != dense chunk {dense}"
        assert _sm100_head_chunk_thd(h, rows, s, bpe, group=group) == old, f"B={b} S={s}: the original rule's answer moved from {old}"
        assert 2 * rows * s * bpe * thd <= budget + budget // _SM100_WS_THD_PAD_SLACK, f"B={b} S={s}: chunk {thd} overshoots the slack"
    rows = rows_of(64 * 100, 64)
    assert _sm100_head_chunk_thd(128, rows, 8192, bpe, t_rows=64 * 100) == 8 == _sm100_head_chunk_thd(128, rows, 8192, bpe)


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
    served -- so a reject test that passes for the wrong reason (an invalid
    graph, a builder typo) fails here first."""
    assert _thd_mismatch() is None


def test_accept_thd_causal():
    """The inverse of the reject this used to be (test/AGENTS.md: invert, do not
    delete).  Stage 2 masks from the per-sequence metadata lengths and stage 3
    drops its absolute-row K-trim under THD, so the causal family is served.

    Kept after the ``thd_causal`` flag was deleted, and now MORE load-bearing
    than before: with no flag left to decline on, this is the only thing that
    would catch a regression re-introducing a causal-under-THD refusal."""
    assert _thd_mismatch(use_causal_mask=True) is None
    assert _thd_mismatch(use_causal_mask_bottom_right=True) is None
    assert _thd_mismatch(use_causal_mask=True, sliding_window_length=64) is None
    assert _thd_mismatch(diagonal_band_right_bound=64) is None


def test_accept_thd_gqa():
    """The inverse of the reject this used to be (test/AGENTS.md: invert, do not
    delete).  The dK/dV partials are now packed per Q head over the kv axis and
    folded by the shared reduce, so GQA and MQA are both served."""
    assert _thd_mismatch(h=4, hkv=2) is None  # GQA, group 2
    assert _thd_mismatch(h=4, hkv=1) is None  # MQA, group 4
    assert _thd_mismatch(h=4, hkv=2, use_causal_mask=True) is None  # with a mask


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


@pytest.mark.parametrize("stats_layout", ["head_major", "token_major"])
@pytest.mark.parametrize("causal", [False, True])
def test_prepared_thd_rebind_lengths_and_replay(stats_layout, causal, monkeypatch):
    from unittest.mock import patch
    import cutlass.cute as cute
    from cudnn.sdpa.bwd.api_dsl import WorkspaceCarver

    calls = []
    original = cudnn.pygraph.execute

    def record(graph, *args, **kwargs):
        calls.append((graph, args))
        return original(graph, *args, **kwargs)

    with patch.object(cudnn.pygraph, "execute", record):
        old, _, _, _ = _run_graph((129, 97, 63), (143, 83, 79), h=4, hkv=2, stats_layout=stats_layout, poison=True, pad_cap=64, use_causal_mask=causal)
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
    _check(case, *gradients, hkv=2)
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            execute_guarded()
        case, changed, _ = next_buffers((31, 123, 41), (113, 67, 109))
        for ref, target in rebound.items():
            target.copy_(changed[ref])
        workspace.fill_(0xBD)
        capture.replay()
        _check(case, *gradients, hkv=2)
    finally:
        capture.reset()


@pytest.mark.parametrize("token_major", [False, True])
def test_prepared_thd_standalone_switches_length_and_prefix_form(token_major, monkeypatch):
    import cutlass.cute as cute
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    original = SdpaBwdDslSm100.execute

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

    monkeypatch.setattr(SdpaBwdDslSm100, "execute", exercise)
    _run((129, 63, 97), (113, 75, 141), token_major_stats=token_major)


@pytest.mark.parametrize("token_major", [False, True])
def test_prepared_thd_standalone_accepts_flat_stats(token_major, monkeypatch):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm100

    original = SdpaBwdDslSm100.execute

    def flat_stats(api, *args, **kwargs):
        args = list(args)
        args[5] = args[5].view(-1)
        return original(api, *args, **kwargs)

    monkeypatch.setattr(SdpaBwdDslSm100, "execute", flat_stats)
    _run((129, 63, 97), (113, 75, 141), token_major_stats=token_major)


@pytest.mark.L1
@pytest.mark.parametrize("batch", [33, 129])
def test_graph_thd_batched_descriptors(batch):
    """Warp leaders cover all dQ/dK/dV descriptors, including empty sequences."""
    lens_q = [[0, 17, 65, 129][i % 4] for i in range(batch)]
    lens_kv = [[33, 0, 127, 257][i % 4] for i in range(batch)]
    _run_graph(lens_q, lens_kv, poison=True, pad_cap=256)


# --------------------------------------------------------------------------- #
# Stage 3 under THD: the causal K-trim is PER SEQUENCE (api_dsl.THD_STAGE3_TRIM) #
# --------------------------------------------------------------------------- #


def _stage3_record_spy(monkeypatch):
    """Record the stage-3 ``MatmulTemplateParams`` the adapter renders (off ``api_dsl.load_template``): ``{tag: params}``."""
    from cudnn.sdpa.bwd import api_dsl

    records = {}
    original = api_dsl.load_template

    def spy(path, params, tag="template"):
        if tag in ("sdpa_bwd_sm100_mm_lo", "sdpa_bwd_sm100_mm_hi"):
            records[tag] = params
        return original(path, params, tag)

    monkeypatch.setattr(api_dsl, "load_template", spy)
    return records


# Every causal-family arm the THD leg serves, with lengths that are NOT tile multiples (the per-sequence k count is
# `ceil(len / 64)`, the stage-2 write granularity 256): unequal lengths, cross attention (S_kv[b] != S_q[b] under
# top-left, where the live kv tiles per q row differ per sequence), an empty sequence, each ONE-SIDED empty case (the
# reduction axis empty: `nkt == 0` must keep an EMPTY range, dV/dK when S_q[b] == 0 and dQ when S_kv[b] == 0),
# bottom-right (the diagonal offset IS per sequence: 56 and 200 here), the right band (the constant part of the shift),
# GQA, and sequences long enough for several 512-row cluster M tiles so the trim actually skips tiles.
_THD_TRIM_CASES = {
    "causal_unequal": dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), kw=dict(use_causal_mask=True)),
    "causal_cross_attention": dict(lens_q=(256, 100), lens_kv=(180, 300), kw=dict(use_causal_mask=True)),
    "causal_zero_length": dict(lens_q=(256, 0, 128), lens_kv=(256, 0, 128), kw=dict(use_causal_mask=True)),
    "causal_empty_q_side": dict(lens_q=(0, 128, 256), lens_kv=(192, 128, 256), kw=dict(use_causal_mask=True)),
    "causal_empty_kv_side": dict(lens_q=(192, 128, 256), lens_kv=(0, 128, 256), kw=dict(use_causal_mask=True)),
    "bottom_right": dict(lens_q=(200, 100), lens_kv=(256, 300), kw=dict(use_causal_mask_bottom_right=True)),
    "causal_gqa": dict(lens_q=(256, 100), lens_kv=(256, 100), h=4, hkv=2, kw=dict(use_causal_mask=True)),
    "causal_long": dict(lens_q=(1100, 700), lens_kv=(1100, 700), kw=dict(use_causal_mask=True)),
    "bottom_right_long": dict(lens_q=(1000, 600), lens_kv=(1100, 900), kw=dict(use_causal_mask_bottom_right=True)),
}
# Stay UNTRIMMED on this chain: the window edge is not taken under THD (its per-sequence bound was not validated on the
# Q-major rows), and the template's THD arm takes no CONSTANT shift, so the right-band widening is not trimmed either.
_THD_SWA_CASE = dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), kw=dict(use_causal_mask=True, sliding_window_length=64))
_THD_RIGHT_BAND_CASE = dict(lens_q=(300, 128, 200), lens_kv=(300, 128, 200), kw=dict(diagonal_band_right_bound=64))


def _thd_trim_run(case, **extra):
    case = dict(case)
    kw = dict(case.pop("kw"))
    kw.update(extra)
    return _run_graph(case.pop("lens_q"), case.pop("lens_kv"), **case, **kw)


@pytest.mark.parametrize(
    "case",
    list(_THD_TRIM_CASES) + ["swa_untrimmed", "right_band_untrimmed", "dense"],
    ids=list(_THD_TRIM_CASES) + ["swa_untrimmed", "right_band_untrimmed", "dense"],
)
def test_graph_thd_causal_stage3_is_trimmed_per_sequence(monkeypatch, case):
    """What the adapter RENDERS for a packed graph: under every causal-family arm but the sliding window and the right-band
    widening, stage 3 takes the same K-trim as the dense path -- ``CAUSAL_K_LO`` on the dV / dK record, ``CAUSAL_K_HI`` on
    the dQ one, ``causal_shift`` 0 (the THD arm takes no constant shift), ``thd_causal_bottom_right`` exactly when the graph
    is bottom-right (the kernel then reads each sequence's ``S_kv[b] - S_q[b]`` from the metadata), no window edge
    (``causal_window`` 0).  A sliding window, a right-band and a dense graph render ``CAUSAL_K_NONE`` on both.  RED on the
    tree before the per-sequence trim: every causal record rendered ``CAUSAL_K_NONE`` there."""
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO, CAUSAL_K_NONE

    records = _stage3_record_spy(monkeypatch)
    if case == "dense":
        _run_graph((300, 128, 200), (300, 128, 200), check=False)
        want = dict(lo=CAUSAL_K_NONE, hi=CAUSAL_K_NONE, shift=0, per_seq=False)
    elif case == "swa_untrimmed":
        _thd_trim_run(_THD_SWA_CASE, check=False)
        want = dict(lo=CAUSAL_K_NONE, hi=CAUSAL_K_NONE, shift=0, per_seq=False)
    elif case == "right_band_untrimmed":
        _thd_trim_run(_THD_RIGHT_BAND_CASE, check=False)
        want = dict(lo=CAUSAL_K_NONE, hi=CAUSAL_K_NONE, shift=0, per_seq=False)
    else:
        spec = _THD_TRIM_CASES[case]
        _thd_trim_run(spec, check=False)
        want = dict(lo=CAUSAL_K_LO, hi=CAUSAL_K_HI, shift=0, per_seq=bool(spec["kw"].get("use_causal_mask_bottom_right")))
    assert records.keys() == {"sdpa_bwd_sm100_mm_lo", "sdpa_bwd_sm100_mm_hi"}, sorted(records)
    lo, hi = records["sdpa_bwd_sm100_mm_lo"], records["sdpa_bwd_sm100_mm_hi"]
    got = dict(lo=lo.causal_mode, hi=hi.causal_mode, shift=hi.causal_shift, per_seq=bool(getattr(hi, "thd_causal_bottom_right", False)))
    assert got == want, f"{case}: the packed stage-3 records render {got}, expected {want}"
    for name, rec in (("mm_lo", lo), ("mm_hi", hi)):
        assert rec.thd_varlen and rec.causal_window == 0 and rec.causal_diag, f"{name}: {rec}"
        assert rec.causal_shift == hi.causal_shift and bool(getattr(rec, "thd_causal_bottom_right", False)) == want["per_seq"], f"{name}: {rec}"


@pytest.mark.parametrize("case", list(_THD_TRIM_CASES), ids=list(_THD_TRIM_CASES))
def test_graph_thd_causal_trim_is_bitwise_the_untrimmed_rendering(monkeypatch, case):
    """The THD causal K-trim is numerically INERT: rendering stage 3 untrimmed (``api_dsl.THD_STAGE3_TRIM = False``, every
    k tile of every group read over the zero-filled workspace -- the rendering this chain shipped before) must give the
    SAME BITS for dQ / dK / dV over the packed region, on int16 views.  A bound keyed on an absolute instead of a
    sequence-relative row, a bottom-right diagonal taken from the wrong sequence (or from the envelope), a right band not
    added, or a never-empty floor applied to an empty reduction axis (the one-sided cases) drops or adds real tiles and
    shows up as a non-zero diff; both runs are also held to the per-sequence fp64 reference.  The twin-bitwise precedent
    (sdpa/AGENTS.md 2x2 lessons); the zero-fill stays on both arms (the 512-row cluster M tile straddles two 256-row
    stage-2 blocks), so this is the trim's correctness proof, not the fill's."""
    from cudnn.sdpa.bwd import api_dsl

    assert api_dsl.THD_STAGE3_TRIM, "the per-sequence trim is what ships; the pin flips it OFF for the twin"
    spec = _THD_TRIM_CASES[case]
    trimmed, dq, dk, dv = _thd_trim_run(spec)
    monkeypatch.setattr(api_dsl, "THD_STAGE3_TRIM", False)
    untrimmed, dq0, dk0, dv0 = _thd_trim_run(spec)
    assert trimmed.lens_q == untrimmed.lens_q and trimmed.cu_q == untrimmed.cu_q
    for name, a, b, live in (("dQ", dq, dq0, trimmed.t_q), ("dK", dk, dk0, trimmed.t_kv), ("dV", dv, dv0, trimmed.t_kv)):
        x, y = a[0, :live].contiguous().view(torch.int16), b[0, :live].contiguous().view(torch.int16)
        n_diff = (x != y).sum().item()
        assert n_diff == 0, (
            f"{case} {name}: trimmed vs untrimmed stage 3 differ in {n_diff} of {x.numel()} int16 words "
            f"(max|diff|={(a[0, :live].float() - b[0, :live].float()).abs().max().item():.3e}); first at {(x != y).nonzero()[0].tolist()}"
        )


@pytest.mark.parametrize("mask", ["dense", "causal", "bottom_right"])
@pytest.mark.parametrize("hkv", [2, 1], ids=["mha", "gqa2"])
def test_graph_thd_causal_matches_dense_bits_at_equal_tile_multiple_lengths(mask, hkv):
    """Equal-length packed buffers ARE the BSHD buffers (``[1, B*S, H, D]`` is ``[B, S, H, D]`` memory), stage 2's
    per-tile work and the GEMMs' k walk are identical by construction, and -- since the per-sequence trim -- so is the
    stage-3 K range per (sequence, M tile) under causal: ``torch.equal`` on int16 views of dQ / dK / dV, THD vs the dense
    BSHD graph of the same lengths (B=2, S=512 = two 256-row stage-2 blocks per sequence, one 512-row cluster M tile).
    Dense is the control (it held before the trim); causal and bottom-right (diagonal 0 at equal lengths, the per-seq
    path taken) are the pins.  The dense graph is the sm100 suite's own builder over the SAME tensors."""
    import test_sdpa_bwd_dsl_sm100 as dense

    b, s, h, d, dt = 2, 512, 2, _D, torch.bfloat16
    kw = dict(use_causal_mask=True) if mask == "causal" else (dict(use_causal_mask_bottom_right=True) if mask == "bottom_right" else {})
    case, dq_t, dk_t, dv_t = _run_graph((s,) * b, (s,) * b, h=h, hkv=hkv, **kw)
    # the dense graph over the same memory: [B, H, S, D] BSHD views of the packed [1, B*S, H, D] buffers
    view_q = lambda x, nh: x.view(b, s, nh, d).permute(0, 2, 1, 3)  # noqa: E731
    q, k, v, o, do = view_q(case.q, h), view_q(case.k, case.hkv), view_q(case.v, case.hkv), view_q(case.o, h), view_q(case.do, h)
    stats = case.lse[0, :, : b * s].view(h, b, s).permute(1, 0, 2).contiguous().unsqueeze(-1)  # head-major packed (1, H, T) -> [B, H, S, 1]
    g, t, (ddq, ddk, ddv) = dense._build_graph(b, h, case.hkv, s, s, d, case.scale, dt=dt, **kw)
    idx = dense._plan_index(g)
    assert idx is not None
    g.select_plan(idx)
    g.check_support()
    g.build_plans()
    dq, dk, dv = (
        dense._bshd(b, s, h, d, dt=dt, fill=False),
        dense._bshd(b, s, case.hkv, d, dt=dt, fill=False),
        dense._bshd(b, s, case.hkv, d, dt=dt, fill=False),
    )
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute({t["q"]: q, t["k"]: k, t["v"]: v, t["o"]: o, t["do"]: do, t["stats"]: stats, ddq: dq, ddk: dk, ddv: dv}, ws)
    torch.cuda.synchronize()
    for name, packed, bshd in (("dQ", dq_t, dq), ("dK", dk_t, dk), ("dV", dv_t, dv)):
        x = packed[0, : b * s].contiguous().view(torch.int16)
        y = bshd.permute(0, 2, 1, 3).contiguous().view(-1, bshd.shape[1], d).view(torch.int16)
        n_diff = (x != y).sum().item()
        assert (
            n_diff == 0
        ), f"{mask} {name}: THD vs dense BSHD differ in {n_diff} of {x.numel()} int16 words (max|diff|={(x.view(dt).float() - y.view(dt).float()).abs().max().item():.3e})"


def _thd_k_range_twin(mode, m0, nkt, *, shift, gran=256, tk=64, cgrp_m=512):
    """Pure-Python twin of ``bprop_matmul_blackwell._thd_causal_k_range`` as the SM100 chain renders it (diagonal edge
    only, ``shift`` = constant + per-sequence part, possibly negative).  Every dividend is clamped at 0 before its ``//``
    and every bound at ``nkt``; the range MAY BE EMPTY -- at ``nkt == 0`` (an empty reduction side) and for a tile none of
    whose rows keeps a cell (the kernel then stores zeros, never the dense arm's never-empty one-tile floor)."""
    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_LO

    if mode == CAUSAL_K_LO:
        lo = max(m0 - shift, 0)
        k_lo = min(((lo // gran) * gran) // tk, nkt)
        k_hi = max(nkt, k_lo)
    else:
        hi_raw = m0 + cgrp_m - 1 + shift
        hi = ((max(hi_raw, 0) // gran) + 1) * gran
        k_hi = 0 if hi_raw < 0 else min((hi + tk - 1) // tk, nkt)
        k_lo = 0
    return k_lo, k_hi


@pytest.mark.parametrize("seed", range(4))
def test_thd_trim_covers_every_cell_stage2_writes_per_sequence(seed):
    """Host-only: for random packed sequences (lengths 0..1500, top-left and bottom-right, right band 0 / 64), every
    512-row cluster M tile's K range from the twin of ``_thd_causal_k_range`` covers every structurally non-zero cell of
    that sequence's band (``kv <= q + shift``) -- the dQ tile's HI bound reaches the last kept kv of its live rows, the
    dV / dK tile's LO bound reaches the first kept q of its live kv rows -- and an empty reduction axis gets an EMPTY
    range (one-sided empty sequences), never the one-tile floor.  A dV / dK tile none of whose kv rows any query attends
    may ALSO come back empty (``k_lo == k_hi == nkt``): its gradient rows are exact zeros, which the kernel stores."""
    import random

    from cudnn.sdpa.bwd.config_sm100 import CAUSAL_K_HI, CAUSAL_K_LO

    rng = random.Random(seed)
    for _ in range(40):
        s_q = rng.choice([0, 1, 63, 64, 100, 128, 255, 256, 300, 511, 512, 700, 1024, 1100, 1500])
        s_kv = rng.choice([0, 1, 64, 100, 192, 256, 300, 512, 900, 1100, 1500])
        bottom_right, wr = rng.choice([False, True]), rng.choice([0, 64])
        shift = wr + ((s_kv - s_q) if bottom_right else 0)
        nkt_q, nkt_kv = -(-s_q // 64), -(-s_kv // 64)
        # dQ: M = q, K = kv; the tile's live rows keep kv <= q + shift (a sequence with no q rows has no live dQ tile:
        # the grid's spare tiles are clipped by the per-sequence output descriptor, whatever range they compute)
        for m0 in range(0, s_q, 512):
            k_lo, k_hi = _thd_k_range_twin(CAUSAL_K_HI, m0, nkt_kv, shift=shift)
            if nkt_kv == 0:
                assert (k_lo, k_hi) == (0, 0), (s_q, s_kv, bottom_right, wr, m0)
                continue
            last_kept = min(min(m0 + 511, s_q - 1) + shift, s_kv - 1)
            assert 0 <= k_lo <= k_hi <= nkt_kv and (last_kept < 0 or last_kept < k_hi * 64), (s_q, s_kv, bottom_right, wr, m0, k_lo, k_hi, last_kept)
        # dV / dK: M = kv, K = q; the tile's live kv rows keep q >= kv - shift (no kv rows, no live tile -- as above)
        for m0 in range(0, s_kv, 512):
            k_lo, k_hi = _thd_k_range_twin(CAUSAL_K_LO, m0, nkt_q, shift=shift)
            if nkt_q == 0:
                assert (k_lo, k_hi) == (0, 0), (s_q, s_kv, bottom_right, wr, m0)
                continue
            first_kept = max(m0 - shift, 0)
            assert 0 <= k_lo <= k_hi == nkt_q and (first_kept >= s_q or first_kept >= k_lo * 64), (s_q, s_kv, bottom_right, wr, m0, k_lo, k_hi, first_kept)
