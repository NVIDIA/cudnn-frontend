# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_bwd_sm80`` THD / varlen through the graph API: the packed backward, end to end.

Two surfaces, deliberately both (the shape of ``test_sdpa_bwd_thd_sm100.py``):

* ``_run`` drives ``SdpaBwdDslSm80`` DIRECTLY with ``thd=True`` -- the adapter's
  packed chain (setup launch -> do_dot -> main -> cast -> reduce) over packed
  buffers, one layer below the plan machinery.
* ``_run_graph`` goes through a ragged GRAPH and pins the engine -- the lowering:
  packed views over the caller's buffers at the declared capacities, the
  per-batch length binding, and the Stats packing inference.

The reference is per-sequence dense attention in fp64, compared gradient by
gradient and sequence by sequence: a THD bug that leaks across a sequence
boundary shows up as one sequence contaminated by its neighbour, which a
whole-tensor cosine would average away.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from frost_test_utils import _SM, requires_dsl

import cudnn

_SM80 = pytest.mark.skipif(_SM != 80, reason="needs an SM80 (A100) GPU, have " + ("none" if _SM is None else f"sm_{_SM}"))
pytestmark = [pytest.mark.L0, _SM80, requires_dsl]

_ENGINE = "sdpa_bwd_sm80"
_D = 128
_TOL_COS = 0.999


def _causal_bias(s_q, s_kv, device, bottom_right=False, window_left=None, window_right=None):
    """Additive -inf mask for ONE sequence, in that sequence's OWN geometry.

    Under THD the diagonal is per sequence: ``bottom_right`` anchors it at
    ``S_kv[b] - S_q[b]``, a per-sequence offset.  ``window_left`` is cuDNN's
    inclusive column count (``kv > q + diag - window_left``), ``window_right``
    an offset that widens the upper bound.
    """
    q_i = torch.arange(s_q, device=device).view(-1, 1)
    kv_i = torch.arange(s_kv, device=device).view(1, -1)
    diag = (s_kv - s_q) if bottom_right else 0
    keep = kv_i <= q_i + diag + (window_right or 0)
    if window_left is not None:
        keep = keep & (kv_i > q_i + diag - window_left)
    return torch.where(keep, 0.0, float("-inf")).double()


def _ref_bwd(q, k, v, do, scale, causal=False, bottom_right=False, window_left=None, window_right=None):
    """fp64 reference backward for ONE sequence (K/V already broadcast to the Q heads)."""
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


def _below_tol(msg):
    # ``not (x > tol)``: a NaN cosine must FAIL, and every comparison against NaN is False.
    return not (float(msg.split("cos ")[1].split(" ")[0]) > _TOL_COS)


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
    d_v=None,
):
    """Packed Q/K/V/dO plus the forward's O and packed LSE, per sequence in fp64.

    ``cap_*`` over-allocates the packed buffers past the live totals; with
    ``poison`` the slack is NaN.  The declared totals only bound the buffers, so
    the rows between ``cu_*[B]`` and the capacity have to stay out of every
    MMA through the kernel's own per-sequence row gating (issue #624).
    """
    dev, b = "cuda", len(lens_q)
    d_v = d if d_v is None else d_v  # Q/K/dQ/dK carry d; V/O/dO/dV carry d_v
    t_q, t_kv = sum(lens_q), sum(lens_kv)
    cap_q, cap_kv = cap_q or t_q, cap_kv or t_kv
    cu_q, cu_k = [0], [0]
    for a, c in zip(lens_q, lens_kv, strict=True):
        cu_q.append(cu_q[-1] + a)
        cu_k.append(cu_k[-1] + c)
    g = torch.Generator(device=dev).manual_seed(seed)
    hkv = h if hkv is None else hkv
    grp = h // hkv

    def _pk(cap, live, nh, dd):
        x = torch.randn(1, cap, nh, dd, generator=g, device=dev, dtype=dtype) * 0.3
        if poison and cap > live:
            x[0, live:] = float("nan")
        return x

    q_p, do_p = _pk(cap_q, t_q, h, d), _pk(cap_q, t_q, h, d_v)
    k_p, v_p = _pk(cap_kv, t_kv, hkv, d), _pk(cap_kv, t_kv, hkv, d_v)
    o_p = torch.full_like(do_p, float("nan") if poison else 0.0)
    lse_p = torch.full((1, h, cap_q), float("nan") if poison else 0.0, device=dev, dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    for i in range(b):
        if lens_q[i] == 0 or lens_kv[i] == 0:
            continue
        qs, ks, vs = (
            x[0, cu[i] : cu[i] + L].transpose(0, 1).double() for x, cu, L in ((q_p, cu_q, lens_q[i]), (k_p, cu_k, lens_kv[i]), (v_p, cu_k, lens_kv[i]))
        )
        if grp > 1:  # kv_head = q_head // grp: heads 0..grp-1 share KV head 0
            ks, vs = ks.repeat_interleave(grp, dim=0), vs.repeat_interleave(grp, dim=0)
        sc = (qs @ ks.transpose(-1, -2)) * scale
        if causal:
            sc = sc + _causal_bias(lens_q[i], lens_kv[i], dev, bottom_right, window_left, window_right)
        lse_p[0, :, cu_q[i] : cu_q[i] + lens_q[i]] = torch.logsumexp(sc, dim=-1).float()
        o_p[0, cu_q[i] : cu_q[i] + lens_q[i]] = (torch.softmax(sc, dim=-1) @ vs).transpose(0, 1).to(dtype)
    return SimpleNamespace(
        b=b, h=h, hkv=hkv, d=d, d_v=d_v, dtype=dtype, scale=scale, causal=causal, bottom_right=bottom_right, window_left=window_left, window_right=window_right,
        lens_q=list(lens_q), lens_kv=list(lens_kv), cu_q=cu_q, cu_k=cu_k, t_q=t_q, t_kv=t_kv, cap_q=cap_q, cap_kv=cap_kv,
        q=q_p, k=k_p, v=v_p, do=do_p, o=o_p, lse=lse_p,
    )  # fmt: skip


def _check(case, dq, dk, dv):
    """Per-sequence comparison against the fp64 reference; collected, then asserted."""
    bad = []
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
            window_right=case.window_right,
        )
        if grp > 1:  # the kernel folds the per-Q-head partials onto the KV heads
            rk = rk.reshape(case.hkv, grp, *rk.shape[1:]).sum(1)
            rv = rv.reshape(case.hkv, grp, *rv.shape[1:]).sum(1)
        for name, got, want in (("dQ", dq[0, sl_q].transpose(0, 1), rq), ("dK", dk[0, sl_k].transpose(0, 1), rk), ("dV", dv[0, sl_k].transpose(0, 1), rv)):
            bad.append(f"seq {i} {name}: cos {_cos(got, want):.6f} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})")
    failures = [m for m in bad if _below_tol(m)]
    assert not failures, "\n".join(bad)


# ---------------------------------------------------------------------------
# The adapter, directly
# ---------------------------------------------------------------------------


def _run(lens_q, lens_kv, h=2, hkv=None, d=_D, dtype=torch.bfloat16, token_major_stats=False, head_stride=0, deterministic=False):
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    case = _thd_case(lens_q, lens_kv, h, d, dtype, hkv=hkv)
    dev, b = "cuda", case.b
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731  [1,T,H,D] -> logical [1,H,T,D]
    # The SAMPLES declare the ENVELOPE (B, H, S_max, D), the way a ragged graph
    # does; the packed buffers only show up at execute.
    s_max_q, s_max_kv = max(lens_q), max(lens_kv)
    env = lambda n, s, nh: torch.empty(1, n, s, nh, d, device=dev, dtype=dtype)[0].permute(0, 2, 1, 3)  # noqa: E731
    eq, ekv = env(b, s_max_q, h), env(b, s_max_kv, case.hkv)
    e_stats = torch.empty(b, h, s_max_q, 1, device=dev, dtype=torch.float32)
    dq, dk, dv = torch.zeros_like(case.q), torch.zeros_like(case.k), torch.zeros_like(case.v)
    if token_major_stats:
        stats = case.lse[0].transpose(0, 1).contiguous()  # (T, H)
    elif head_stride:
        stats = torch.zeros(1, h, head_stride, device=dev, dtype=torch.float32)
        stats[..., : case.t_q] = case.lse
    else:
        stats = case.lse  # compact head-major [1, H, T]
    api = SdpaBwdDslSm80(
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
        deterministic=deterministic,
        thd=True,
        max_total_seq_len_q=case.t_q,
        max_total_seq_len_kv=case.t_kv,
        thd_stats_token_major=token_major_stats,
        thd_stats_head_stride=head_stride,
    )
    assert api.check_support()
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    api.execute(
        view(case.q), view(case.k), view(case.v), view(case.o), view(case.do), stats, view(dq), view(dk), view(dv),
        workspace=ws,
        seq_q_lens=torch.tensor(lens_q, dtype=torch.int32, device=dev),
        seq_kv_lens=torch.tensor(lens_kv, dtype=torch.int32, device=dev),
    )  # fmt: skip
    torch.cuda.synchronize()
    _check(case, dq, dk, dv)


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_thd_self_attention(dt):
    """Three sequences of unequal length, none a tile multiple."""
    _run((300, 128, 200), (300, 128, 200), dtype=dt)


def test_thd_cross_attention():
    """Unequal Q and KV lengths, and unequal packed totals with them."""
    _run((256, 100), (180, 300))


def test_thd_single_sequence_matches_dense_shape():
    _run((512,), (512,))


@pytest.mark.parametrize("layout", ("token_major", "head_major_wide"))
def test_thd_stats_packings(layout):
    """Both packed Stats layouts Rule S1 admits, beyond the wrapper's compact
    head-major one: cuDNN's ``(T, H)`` recipe, and head-major at a head stride
    WIDER than the packed total (the FROST forwards' 64-token-rounded capacity)."""
    if layout == "token_major":
        _run((300, 128), (300, 128), token_major_stats=True)
    else:
        _run((300, 128), (300, 128), head_stride=512)


def test_thd_gqa_direct():
    _run((300, 128, 200), (300, 128, 200), h=4, hkv=2)


def test_thd_deterministic_direct():
    """The kv-ordered dQ relay under THD, its counter carved from the workspace."""
    _run((300, 128, 200), (300, 128, 200), deterministic=True)


def test_thd_requires_declared_totals():
    """THD without max_total_seq_len_* is DECLINED, not silently mis-sized:
    the packed fp32 dQ accumulator, the dK/dV partials and do_dot are sized
    from the token totals at build time."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

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
        SdpaBwdDslSm80(**kw).check_support()
    assert SdpaBwdDslSm80(**kw, max_total_seq_len_q=400, max_total_seq_len_kv=400).check_support()


def test_thd_rejects_bias_and_padding_mask():
    """Dense-only features under THD are typed declines at check_support."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    dev, b, h, s_max = "cuda", 2, 2, 256
    env = torch.empty(b, s_max, h, _D, device=dev, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    e_stats = torch.empty(b, h, s_max, 1, device=dev, dtype=torch.float32)
    kw = dict(
        sample_q=env, sample_k=env, sample_v=env, sample_o=env, sample_do=env, sample_stats=e_stats, sample_dq=env, sample_dk=env, sample_dv=env,
        scale_softmax=1.0 / math.sqrt(_D), thd=True, max_total_seq_len_q=400, max_total_seq_len_kv=400,
    )  # fmt: skip
    with pytest.raises(ValueError, match="dense-only"):
        SdpaBwdDslSm80(**kw, has_bias=True).check_support()
    with pytest.raises(ValueError, match="dense-only"):
        SdpaBwdDslSm80(**kw, seq_kv_lens_present=True).check_support()


# ---------------------------------------------------------------------------
# The graph path: ragged tensors -> lower_dsl_bwd -> packed views
# ---------------------------------------------------------------------------


def _plan_index(g, name=_ENGINE):
    for i in range(g.get_execution_plan_count()):
        if name in g.get_plan_name_at_index(i):
            return i
    return None


_GAP_ROLES = ("q", "k", "v", "o", "do", "dq", "dk", "dv")


def _gapped(x, gap, fill=float("nan")):
    """``x`` ([1, cap, nh, dd] compact) re-homed in a per-token record ``gap``
    elements wider: the returned view has token stride ``nh*dd + gap`` and the
    gap columns hold ``fill`` (NaN by default -- a read of a gap column would
    poison the result, a write would show up in the storage).  ``gap <= 0``
    returns ``x`` itself (a negative gap declares OVERLAPPING rows, a
    probe-only decline that never binds data)."""
    if gap <= 0:
        return x
    _, cap, nh, dd = x.shape
    ts = nh * dd + gap
    stor = torch.full((cap, ts), fill, device=x.device, dtype=x.dtype)
    view = stor.as_strided((1, cap, nh, dd), (cap * ts, ts, dd, 1))
    view.copy_(x)
    return view


def _gap_columns(view):
    """The gap columns of a ``_gapped`` view's per-token records (empty for a
    compact tensor)."""
    _, cap, nh, dd = view.shape
    ts = view.stride(1)
    if ts == nh * dd:
        return view.new_empty(0)
    return view.as_strided((cap, ts - nh * dd), (ts, 1), view.storage_offset() + nh * dd)


def _build_thd_bwd_graph(case, *, stats_layout="head_major", declare_totals=True, gaps=None, sink=None, **sdpa_kwargs):
    """A ragged backward graph over ``case``'s packed buffers.

    Everything is declared as the ENVELOPE (B, H, S_max, D) plus a per-tensor
    ragged offset -- cuDNN's spelling of a packed tensor.  ``gaps`` maps a port
    role (``q k v o do dq dk dv``) to extra elements on its token stride: the
    port is then a view into a wider per-token record (``k``/``v`` at
    ``H_kv * D`` is the fused-KV interleaved layout), which the SM80 packed
    path serves at that stride.  The bound input buffers are re-homed in such
    records with NaN in the gap columns.  ``sink`` adds the ``sink_token`` /
    ``dSink_token`` ports ((1, H, 1, 1) fp32); the graph's
    ``_thd_test_ports["dsink"]`` handle finds the dSink buffer in the pack.
    """
    b, h, hkv, d, d_v, dev = case.b, case.h, case.hkv, case.d, case.d_v, "cuda"
    gaps = dict(gaps or {})
    assert set(gaps) <= set(_GAP_ROLES), gaps
    io = cudnn.data_type.HALF if case.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    s_max_q, s_max_kv = max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    vp, t, geom = {}, {}, {}

    def _port(name, s_max, nh, dd, cu, gap=0):
        """Envelope (B, nh, s_max, dd) with packed BSHD strides (token stride
        ``nh*dd`` widened by ``gap``) and an element ragged offset of cu * token stride."""
        ts = nh * dd + gap
        stride = [s_max * ts, dd, ts, 1]
        ro_t = (torch.tensor(cu, dtype=torch.int64, device=dev) * ts).view(b + 1, 1, 1, 1)
        geom[name] = (s_max, stride, nh, dd, ro_t)
        x = g.tensor(name=name, dim=[b, nh, s_max, dd], stride=stride, data_type=io)
        ro = g.tensor(name=f"{name}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        x.set_ragged_offset(ro)
        vp[ro] = ro_t
        return x

    t["q"] = _port("q", s_max_q, h, d, case.cu_q, gaps.get("q", 0))
    t["o"] = _port("o", s_max_q, h, d_v, case.cu_q, gaps.get("o", 0))
    t["do"] = _port("do", s_max_q, h, d_v, case.cu_q, gaps.get("do", 0))
    t["k"] = _port("k", s_max_kv, hkv, d, case.cu_k, gaps.get("k", 0))
    t["v"] = _port("v", s_max_kv, hkv, d_v, case.cu_k, gaps.get("v", 0))
    # The gradients' own records (declared below, once the node exists).
    geom["dq"] = (
        s_max_q,
        [s_max_q * (h * d + gaps.get("dq", 0)), d, h * d + gaps.get("dq", 0), 1],
        h,
        d,
        torch.tensor(case.cu_q, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1) * (h * d + gaps.get("dq", 0)),
    )
    geom["dk"] = (
        s_max_kv,
        [s_max_kv * (hkv * d + gaps.get("dk", 0)), d, hkv * d + gaps.get("dk", 0), 1],
        hkv,
        d,
        torch.tensor(case.cu_k, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1) * (hkv * d + gaps.get("dk", 0)),
    )
    geom["dv"] = (
        s_max_kv,
        [s_max_kv * (hkv * d_v + gaps.get("dv", 0)), d_v, hkv * d_v + gaps.get("dv", 0), 1],
        hkv,
        d_v,
        torch.tensor(case.cu_k, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1) * (hkv * d_v + gaps.get("dv", 0)),
    )

    # Packed Stats in one of the layouts a forward emits.  head_major is
    # (1, H, head_stride) with a 64-rounded token capacity (WIDER than the
    # packed total -- the FROST forwards); head_major_compact has head stride
    # == the declared capacity exactly (the SM80 forward wrapper); token_major
    # is (T, H) (cuDNN's ragged recipe).  Sized on the DECLARED capacity: the
    # adapter binds the Stats view at min(B * S_max, declared).
    if stats_layout in ("head_major", "head_major_compact"):
        t_cap = max(64, -(-case.cap_q // 64) * 64) if stats_layout == "head_major" else case.cap_q
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
    if sink is not None:
        sink_t = g.tensor(name="sink", dim=[1, h, 1, 1], stride=[h, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        dsink_t = g.tensor(name="dSink", dim=[1, h, 1, 1], stride=[h, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
        vp[sink_t] = sink.view(1, h, 1, 1).contiguous()
        vp[dsink_t] = torch.zeros(1, h, 1, 1, device=dev, dtype=torch.float32)
        t["dsink"] = dsink_t
        kw.update(sink_token=sink_t, dSink_token=dsink_t)
    if declare_totals:
        kw.update(max_total_seq_len_q=case.cap_q, max_total_seq_len_kv=case.cap_kv)
    kw.update(sdpa_kwargs)
    dq_t, dk_t, dv_t = g.sdpa_backward(**kw)
    for out, role in ((dq_t, "dq"), (dk_t, "dk"), (dv_t, "dv")):
        s_max, stride, nh, dd, ro_t = geom[role]
        out.set_output(True).set_data_type(io).set_dim([b, nh, s_max, dd]).set_stride(stride)
        ro = g.tensor(name=f"{out.get_name()}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        out.set_ragged_offset(ro)
        vp[ro] = ro_t
    # Inputs re-homed in their declared records (NaN gap columns: never read).
    vp.update({t[r]: _gapped(getattr(case, r), gaps.get(r, 0)) for r in ("q", "k", "v", "o", "do")})
    g._thd_test_ports = t  # the sink/dSink handles, for the sinks test
    g._thd_test_gaps = gaps  # the gradients' records, for _run_graph
    return g, vp, (dq_t, dk_t, dv_t)


def _plan_graph(g):
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    idx = _plan_index(g)
    assert idx is not None, f"{_ENGINE} not offered; plans = {[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
    g.select_plan(idx)
    g.check_support()
    g.build_plans()


def _run_graph(lens_q, lens_kv, *, h=2, hkv=None, d=_D, d_v=None, dtype=torch.bfloat16, stats_layout="head_major", poison=False, pad_cap=0, **kw):
    """Build the ragged graph, PIN the engine, execute, compare per sequence.

    The gradient buffers are pre-filled with NaN, not zeros: a live row the
    kernels fail to write then shows up in the ``isfinite`` check (and a
    skipped zero-store in the one-sided-empty test), instead of hiding behind
    a zero the caller happened to put there.
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
        d_v=d_v,
    )
    g, vp, (dq_t, dk_t, dv_t) = _build_thd_bwd_graph(case, stats_layout=stats_layout, **kw)
    _plan_graph(g)
    gaps = g._thd_test_gaps
    # NaN-filled gradient records at the declared token strides: a gap column
    # that stops being NaN was written, a live row that stays NaN was skipped.
    dq = _gapped(torch.full_like(case.q, float("nan")), gaps.get("dq", 0))
    dk = _gapped(torch.full((1, case.cap_kv, case.hkv, case.d), float("nan"), device="cuda", dtype=dtype), gaps.get("dk", 0))
    dv = _gapped(torch.full((1, case.cap_kv, case.hkv, case.d_v), float("nan"), device="cuda", dtype=dtype), gaps.get("dv", 0))
    vp.update({dq_t: dq, dk_t: dk, dv_t: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    for name, x in (("dQ", dq), ("dK", dk), ("dV", dv)):
        live = x[0, : case.t_q] if name == "dQ" else x[0, : case.t_kv]
        assert torch.isfinite(live).all(), f"{name} has non-finite values in the packed region"
        assert torch.isnan(_gap_columns(x)).all(), f"{name}: a gap column of the per-token record was written"
    _check(case, dq, dk, dv)
    return case, g, vp, ws, (dq, dk, dv)


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_self_attention(dt):
    """The whole point: a ragged graph reaches the SM80 kernel through the engine."""
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt)


def test_graph_thd_cross_attention():
    _run_graph((256, 100), (180, 300))


@pytest.mark.parametrize("layout", ("head_major", "head_major_compact", "token_major"))
def test_graph_thd_stats_packings(layout):
    """Every packed Stats layout the row reads, each its own accept test: the
    FROST forwards' 64-rounded head-major, the SM80 forward wrapper's compact
    head-major, and cuDNN's token-major recipe."""
    _run_graph((300, 128), (300, 128), stats_layout=layout)


def test_graph_thd_zero_length_sequence():
    """A sequence with no tokens must not corrupt its neighbours."""
    _run_graph((256, 0, 128), (256, 0, 128))


@pytest.mark.parametrize("lens_q,lens_kv", (((0, 128, 256), (192, 128, 256)), ((192, 128, 256), (0, 128, 256))), ids=("empty_q_side", "empty_kv_side"))
def test_graph_thd_one_sided_empty_sequence(lens_q, lens_kv):
    """Empty on ONE side only: the other side's rows exist and must read zero.

    ``empty_q_side``: the sequence's kv tiles run zero Q-iterations, so the
    epilogue must store zero dK/dV from untouched accumulators.  ``empty_kv_side``:
    no kv tile touches the sequence's dQ rows, which stay at the accumulator's
    zero fill.  The explicit zero check is the detector (residue is finite).
    """
    case, _, _, _, (dq, dk, dv) = _run_graph(lens_q, lens_kv)
    i = 0
    sl_q = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
    sl_k = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
    for name, got in (("dQ", dq[0, sl_q]), ("dK", dk[0, sl_k]), ("dV", dv[0, sl_k])):
        assert got.numel() == 0 or not got.any(), f"{name} of a one-sided-empty sequence must be exactly zero, got max |{got.abs().max().item()}|"


def test_graph_thd_nan_capacity_tail():
    """Declared totals larger than the live packing, with a NaN tail (#624):
    the rows between ``cu_*[B]`` and the capacity are inside every packed view,
    and only the kernel's per-sequence row gating keeps them out of the MMAs."""
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384)


@pytest.mark.parametrize("variant", ("mha_native", "gqa", "padded_d"), ids=("mha_native", "gqa", "padded_d"))
def test_graph_thd_capacity_tail_left_untouched(variant):
    """cuDNN's contract, which test_mhas_v2's ragged sweeps assert with a
    NaN-filled tail: nothing past the packed total is written into the caller's
    dQ/dK/dV, on every output path -- the direct-bound MHA epilogue, the GQA
    fold and the padded-head-dim fold are all bounded by ``cu_*[B]`` on device.
    The workspace is poisoned too, so a fold reading past the live rows would
    show up as well."""
    kw = {"gqa": dict(h=4, hkv=2), "padded_d": dict(d=96)}.get(variant, {})
    lens = (256, 128)
    case, g, vp, ws, (dq, dk, dv) = _run_graph(lens, lens, poison=True, pad_cap=384, **kw)
    ws.fill_(0xFF)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    # _run_graph pre-fills the gradient buffers with NaN: every row past the
    # packed total must still be NaN, every live row finite.
    for name, x, t in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
        assert torch.isfinite(x[0, :t]).all(), f"{name}: live rows must be finite"
        assert torch.isnan(x[0, t:]).all(), f"{name}: rows past the packed total were written"


_GAP_CASES = {
    # K/V (and dK/dV) as slices of an interleaved [T, 2, H_kv, D] record: the
    # fused-KV layout torch.nn.attention.varlen users produce.
    "kv_interleaved": dict(gaps={"k": 2 * _D, "v": 2 * _D, "dk": 2 * _D, "dv": 2 * _D}),
    # The Q side only: Q/O/dO/dQ each in a different record.
    "q_side": dict(gaps={"q": 8, "o": 16, "do": 24, "dq": 32}),
    # Every port in a record of its own width.
    "all_ports": dict(gaps={r: 8 * (i + 1) for i, r in enumerate(_GAP_ROLES)}),
    # GQA: the dK/dV fold writes the caller's record at its stride.
    "gqa": dict(h=4, hkv=2, gaps={r: 8 * (i + 1) for i, r in enumerate(_GAP_ROLES)}),
    # MQA: the size-1 KV head axis wildcards its stride.
    "mqa": dict(h=4, hkv=1, gaps={"k": 2 * _D, "v": 2 * _D, "dk": 16, "dv": 8}),
    # A head dim inside the envelope: the staging copy reads the record, the
    # cast and fold write it.
    "padded_d": dict(d=96, gaps={r: 8 * (i + 1) for i, r in enumerate(_GAP_ROLES)}),
    # The causal + deterministic relay with interleaved K/V.
    "causal_det": dict(use_causal_mask=True, use_deterministic_algorithm=True, gaps={"k": 2 * _D, "v": 2 * _D}),
}


@pytest.mark.parametrize("variant", sorted(_GAP_CASES), ids=sorted(_GAP_CASES))
def test_graph_thd_gapped_token_strides(variant):
    """Ports declared as views into wider per-token records are read and
    written at their own token stride: the reference matches per sequence, the
    NaN gap columns of the inputs were never read (they would poison the
    result) and those of the gradients never written."""
    _run_graph((300, 128, 200), (300, 128, 200), **_GAP_CASES[variant])


def test_graph_thd_gapped_capacity_tail_left_untouched():
    """The capacity-tail contract on gapped records: rows past the packed total
    and the gap columns both stay NaN, on the direct-bound MHA epilogue."""
    gaps = {r: 8 * (i + 1) for i, r in enumerate(_GAP_ROLES)}
    case, g, vp, ws, (dq, dk, dv) = _run_graph((256, 128), (256, 128), poison=True, pad_cap=384, gaps=gaps)
    ws.fill_(0xFF)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    for name, x, t in (("dQ", dq, case.t_q), ("dK", dk, case.t_kv), ("dV", dv, case.t_kv)):
        assert torch.isfinite(x[0, :t]).all(), f"{name}: live rows must be finite"
        assert torch.isnan(x[0, t:]).all(), f"{name}: rows past the packed total were written"
        assert torch.isnan(_gap_columns(x)).all(), f"{name}: a gap column was written"


def test_graph_thd_nan_capacity_tail_unaligned_last_sequence():
    """The capacity tail reached by a TILE OVERSHOOT of the last sequence: 100
    tokens is not a tile multiple on either side, so the last q- and kv-tiles
    of the last sequence straddle the packed total."""
    _run_graph((256, 100), (256, 100), poison=True, pad_cap=384)


def test_graph_thd_b1_matches_dense_shape():
    _run_graph((512,), (512,))


@pytest.mark.parametrize("hkv", (2, 1), ids=("gqa_group2", "mqa"))
def test_graph_thd_gqa(hkv):
    """Packed GQA / MQA: per-Q-head dK/dV partials over the packed kv axis,
    folded onto the KV heads by the reduce."""
    _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=hkv)


def test_graph_thd_gqa_causal():
    _run_graph((256, 100), (256, 100), h=4, hkv=2, use_causal_mask=True)


def test_graph_thd_gqa_cross_attention_and_zero_length():
    _run_graph((256, 0, 100), (180, 0, 300), h=4, hkv=2)


@pytest.mark.parametrize("dt", (torch.bfloat16, torch.float16), ids=("bf16", "fp16"))
def test_graph_thd_causal(dt):
    _run_graph((300, 128, 200), (300, 128, 200), dtype=dt, use_causal_mask=True)


def test_graph_thd_causal_zero_length_sequence():
    _run_graph((256, 0, 128), (256, 0, 128), use_causal_mask=True)


def test_graph_thd_causal_cross_attention():
    _run_graph((256, 100), (180, 300), use_causal_mask=True)


def test_graph_thd_causal_bottom_right():
    """Bottom-right alignment: the diagonal offset ``S_kv[b] - S_q[b]`` IS per
    sequence (56 and 200 here); S_kv >= S_q on purpose so no reference row is
    fully masked."""
    _run_graph((200, 100), (256, 300), use_causal_mask_bottom_right=True)


def test_graph_thd_causal_swa():
    _run_graph((300, 128, 200), (300, 128, 200), use_causal_mask=True, sliding_window_length=64)


def test_graph_thd_causal_right_band():
    _run_graph((300, 128, 200), (300, 128, 200), diagonal_band_right_bound=64)


def test_graph_thd_causal_nan_capacity_tail():
    _run_graph((256, 128), (256, 128), poison=True, pad_cap=384, use_causal_mask=True)


def test_graph_thd_deterministic():
    """The kv-ordered dQ relay through the graph API, its counter carved and
    zeroed per execute; bitwise identical across executes."""
    _, g, vp, ws, (dq, dk, dv) = _run_graph((300, 128, 200), (300, 128, 200), use_causal_mask=True, use_deterministic_algorithm=True)
    first = tuple(x.clone() for x in (dq, dk, dv))
    g.execute(vp, ws)
    torch.cuda.synchronize()
    for a, b_ in zip(first, (dq, dk, dv), strict=True):
        torch.testing.assert_close(a, b_, rtol=0, atol=0, equal_nan=True)  # NaN only in the untouched prefill past the packed total


def _ref_bwd_sink(q, k, v, do, sink_h, scale, causal, s_q, s_kv):
    """fp64 reference for ONE sequence with an attention sink: a virtual column
    holding the per-head sink logit joins the softmax and receives no value."""
    q, k, v = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    sk = sink_h.detach().double().clone().requires_grad_(True)
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        s = s + _causal_bias(s_q, s_kv, q.device)
    s_ext = torch.cat([s, sk.view(-1, 1, 1).expand(s.shape[0], s.shape[1], 1)], dim=-1)
    p = torch.softmax(s_ext, dim=-1)[..., :-1]
    (p @ v).backward(do.double())
    return q.grad, k.grad, v.grad, sk.grad


@pytest.mark.parametrize("stats_layout", ("head_major", "token_major"))
@pytest.mark.parametrize("causal", (False, True), ids=("plain", "causal"))
def test_graph_thd_sinks(causal, stats_layout):
    """Attention sinks under THD through the graph: the dSink kernel walks each
    (sequence, head) row over ``cu_q`` and reads the packed Stats in either
    packing; dSink is the sum over every sequence's rows and must be re-zeroed
    per execute (the carved accumulator)."""
    lens_q, lens_kv, h, d, dtype, dev = (300, 128, 200), (300, 128, 200), 2, _D, torch.bfloat16, "cuda"
    case = _thd_case(lens_q, lens_kv, h, d, dtype, causal=causal)
    sink = torch.randn(h, generator=torch.Generator(device=dev).manual_seed(11), device=dev, dtype=torch.float32) * 0.5
    # Re-derive O and the LSE with the sink column in the softmax, per sequence.
    for i in range(case.b):
        sq_, sk_ = slice(case.cu_q[i], case.cu_q[i] + lens_q[i]), slice(case.cu_k[i], case.cu_k[i] + lens_kv[i])
        qs, ks, vs = (x[0, sl].transpose(0, 1).double() for x, sl in ((case.q, sq_), (case.k, sk_), (case.v, sk_)))
        s = (qs @ ks.transpose(-1, -2)) * case.scale
        if causal:
            s = s + _causal_bias(lens_q[i], lens_kv[i], dev)
        s_ext = torch.cat([s, sink.double().view(h, 1, 1).expand(h, lens_q[i], 1)], dim=-1)
        case.lse[0, :, sq_] = torch.logsumexp(s_ext, dim=-1).float()
        case.o[0, sq_] = (torch.softmax(s_ext, dim=-1)[..., :-1] @ vs).transpose(0, 1).to(dtype)
    g, vp, (dq_t, dk_t, dv_t) = _build_thd_bwd_graph(case, sink=sink, stats_layout=stats_layout, **({"use_causal_mask": True} if causal else {}))
    _plan_graph(g)
    dq, dk, dv = torch.zeros_like(case.q), torch.zeros_like(case.k), torch.zeros_like(case.v)
    vp.update({dq_t: dq, dk_t: dk, dv_t: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    dsink = vp[g._thd_test_ports["dsink"]].view(-1)
    first = dsink.clone()
    g.execute(vp, ws)  # the carved dSink accumulator is re-zeroed, so a second run agrees (atomic order aside)
    torch.cuda.synchronize()
    torch.testing.assert_close(dsink, first, rtol=1e-5, atol=1e-6)
    bad, dsink_ref = [], torch.zeros(h, dtype=torch.float64, device=dev)
    for i in range(case.b):
        sq_, sk_ = slice(case.cu_q[i], case.cu_q[i] + lens_q[i]), slice(case.cu_k[i], case.cu_k[i] + lens_kv[i])
        rq, rk, rv, rs = _ref_bwd_sink(
            case.q[0, sq_].transpose(0, 1),
            case.k[0, sk_].transpose(0, 1),
            case.v[0, sk_].transpose(0, 1),
            case.do[0, sq_].transpose(0, 1),
            sink,
            case.scale,
            causal,
            lens_q[i],
            lens_kv[i],
        )
        dsink_ref += rs
        for name, got, want in (("dQ", dq[0, sq_].transpose(0, 1), rq), ("dK", dk[0, sk_].transpose(0, 1), rk), ("dV", dv[0, sk_].transpose(0, 1), rv)):
            bad.append(f"seq {i} {name}: cos {_cos(got, want):.6f}")
    failures = [m for m in bad if _below_tol(m)]
    assert not failures, "\n".join(bad)
    # dSink is a per-head scalar pair: compare values, not a scale-blind cosine.
    torch.testing.assert_close(dsink.double(), dsink_ref, rtol=2e-2, atol=1e-4)


def test_thd_rejects_prefix_sum_lengths():
    """The backward node carries per-batch lengths only; a (B+1,) prefix sum at
    execute is a contract violation the adapter names, not a silent mis-read."""
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80

    lens_q = lens_kv = (256, 128)
    case = _thd_case(lens_q, lens_kv, 2, _D, torch.bfloat16)
    dev, b, h = "cuda", case.b, case.h
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731
    env = torch.empty(b, 256, h, _D, device=dev, dtype=torch.bfloat16).permute(0, 2, 1, 3)
    e_stats = torch.empty(b, h, 256, 1, device=dev, dtype=torch.float32)
    api = SdpaBwdDslSm80(
        sample_q=env, sample_k=env, sample_v=env, sample_o=env, sample_do=env, sample_stats=e_stats, sample_dq=env, sample_dk=env, sample_dv=env,
        scale_softmax=case.scale, thd=True, max_total_seq_len_q=case.t_q, max_total_seq_len_kv=case.t_kv,
    )  # fmt: skip
    assert api.check_support()
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    dq, dk, dv = torch.zeros_like(case.q), torch.zeros_like(case.k), torch.zeros_like(case.v)
    cu = torch.tensor(case.cu_q, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="per-batch lengths"):
        api.execute(
            view(case.q),
            view(case.k),
            view(case.v),
            view(case.o),
            view(case.do),
            case.lse,
            view(dq),
            view(dk),
            view(dv),
            workspace=ws,
            seq_q_lens=cu,
            seq_kv_lens=cu,
        )


@pytest.mark.parametrize(
    "d,d_v", ((64, 64), (96, 96), (192, 128), (160, 128), (256, 256)), ids=("gptoss_native", "llama_padded", "dsv3_native", "dsv3_padded", "qwen_native")
)
def test_graph_thd_head_dim_envelope(d, d_v):
    """Every flavor the row serves, under THD: native (64, 128 via the other
    tests, 192x128, 256) and padded onto the next envelope (96 -> llama,
    160x128 -> dsv3) through carved staging at the packed token capacities
    (the wrapper pads with allocations; the engine must not).  The rectangular
    flavors carve distinct d_qk / d_v buffers, in a fixed order."""
    _run_graph((300, 128), (300, 128), d=d, d_v=d_v)


def test_graph_thd_execute_does_not_allocate():
    """Issue #514 on the packed path: after the warm run, re-executing must not
    touch the CUDA caching allocator (the cu_seqlens metadata, the relay
    counter and every scratch buffer are carved from the caller's workspace)."""
    _, g, vp, ws, (dq, dk, dv) = _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, d=96, use_causal_mask=True, use_deterministic_algorithm=True)
    ref = tuple(x.clone() for x in (dq, dk, dv))
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        g.execute(vp, ws)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert after == before, f"execute allocated {after - before} times; the packed path must carve from the workspace only"
    for a, b_ in zip(ref, (dq, dk, dv), strict=True):
        torch.testing.assert_close(a, b_, rtol=0, atol=0, equal_nan=True)


def test_graph_thd_execute_does_not_sync():
    """Rule 3 on the packed path: after the warm run, execute performs no
    device->host read (the lengths become cu_seqlens on device; the grid and
    relay bounds are plan-time).  torch's sync debug mode turns any
    synchronizing call into an error; the ``.item()`` probe first proves the
    mode is armed (the check is RED against a deliberate read)."""
    _, g, vp, ws, _ = _run_graph((300, 128, 200), (300, 128, 200), h=4, hkv=2, use_causal_mask=True, use_deterministic_algorithm=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            torch.zeros(1, device="cuda").item()
        g.execute(vp, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_graph_thd_compile_key_is_plan_time_only():
    """Issue #604 on the graph path: two ragged graphs with the same envelope,
    sequence count and Stats packing but different packed totals share ONE
    compiled artifact (the bprop template's per-shape lru sees a hit, no miss).

    Token-major Stats on purpose: the head-major packing's head stride is
    plan-time tensor geometry that legitimately keys the artifact (it is the
    fake's extent), and the 64-rounded capacity differs between the two runs.
    """
    from cudnn.frost import template_loader

    def cache_totals():
        modules = [m for (path, _params), m in template_loader._MODULES.items() if "bprop" in str(path)]
        infos = [m.compile.cache_info() for m in modules if hasattr(m.compile, "cache_info")]
        return sum(i.misses for i in infos), sum(i.hits for i in infos)

    _run_graph((300, 128), (300, 128), stats_layout="token_major")
    n_modules = len(template_loader._MODULES)
    misses_0, hits_0 = cache_totals()
    # Same S_max envelope (300) and B (2), different totals and lengths.
    _run_graph((300, 64), (300, 64), stats_layout="token_major")
    misses_1, hits_1 = cache_totals()
    assert misses_1 == misses_0, "different packed totals minted a compile (runtime data leaked into the key)"
    assert hits_1 > hits_0, "expected the same-envelope re-plan to cache-hit"
    assert len(template_loader._MODULES) == n_modules


# --- probe: accepts and rejects -----------------------------------------------


def _thd_mismatch(lens_q=(256, 128), lens_kv=(256, 128), *, h=2, hkv=None, stats_layout="head_major", d=_D, **kw):
    """``mismatch()`` for a ragged backward graph on the SM80 row, or None if served."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    case = _thd_case(lens_q, lens_kv, h, d, torch.bfloat16, hkv=hkv)
    g, _, _ = _build_thd_bwd_graph(case, stats_layout=stats_layout, **kw)
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError as exc:
        return f"refused by the node: {exc}"
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    assert facts is not None
    return mismatch(spec.capabilities, facts)


def test_graph_thd_accepts_the_plain_case():
    assert _thd_mismatch() is None


def test_accept_thd_features():
    """Every THD conjunction the row serves, each on its own line so a
    regression names the feature."""
    assert _thd_mismatch(use_causal_mask=True) is None
    assert _thd_mismatch(use_causal_mask_bottom_right=True) is None
    assert _thd_mismatch(use_causal_mask=True, sliding_window_length=64) is None
    assert _thd_mismatch(diagonal_band_right_bound=64) is None
    assert _thd_mismatch(h=4, hkv=2) is None
    assert _thd_mismatch(h=4, hkv=1) is None
    assert _thd_mismatch(use_deterministic_algorithm=True) is None
    assert _thd_mismatch(stats_layout="token_major") is None
    assert _thd_mismatch(d=96) is None


def test_reject_thd_without_declared_totals():
    reason = _thd_mismatch(declare_totals=False)
    assert reason is not None and "max_total_seq_len" in reason


def test_accept_thd_gapped_token_strides():
    """Ports declared as views into a wider per-token record are served at
    their own token stride: K/V at 2*H_kv*D (the fused-KV slicing layout, the
    one the ragged sweeps draw most), and every port at once."""
    assert _thd_mismatch(gaps={"k": 2 * _D, "v": 2 * _D}) is None
    assert _thd_mismatch(gaps={r: 8 * (i + 1) for i, r in enumerate(_GAP_ROLES)}) is None
    assert _thd_mismatch(h=4, hkv=2, gaps={"k": 2 * _D, "v": 2 * _D, "dk": 8, "dv": 16}) is None


def test_reject_thd_misaligned_token_stride():
    """A token stride off 16-byte alignment (4 fp16 elements past the row) is a
    typed plan-time decline: the cp.async loads move 16-byte chunks."""
    reason = _thd_mismatch(gaps={"k": 4})
    assert reason is not None and "multiple of 8" in reason, reason
    reason = _thd_mismatch(gaps={"dq": 12})
    assert reason is not None and "multiple of 8" in reason, reason


def test_reject_thd_token_stride_below_the_row():
    """A token stride shorter than H*D (overlapping rows) never reaches the
    packed-row rule: the layout envelope's non-overlap check (or the node)
    declines it first."""
    reason = _thd_mismatch(gaps={"q": -8})
    assert reason is not None and ("refused by the node" in reason or "overlapping" in reason or ">= H*D" in reason), reason


def test_reject_thd_bias():
    """Bias / dBias under THD is a path property every THD row declines at plan
    time (a packed graph has no [B, H, S_q, S_kv] bias)."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    case = _thd_case((256, 128), (256, 128), 2, _D, torch.bfloat16)
    g, _, _ = _build_thd_bwd_graph(case)
    node = g.nodes[0]
    s_q, s_kv = max(case.lens_q), max(case.lens_kv)
    bias_t = g.tensor(name="bias", dim=[1, case.h, s_q, s_kv], stride=[case.h * s_q * s_kv, s_q * s_kv, s_kv, 1], data_type=cudnn.data_type.FLOAT)
    node.inputs["bias"] = bias_t
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return  # refused by the node itself: also a decline before engine selection
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    reason = mismatch(spec.capabilities, facts)
    assert reason is not None and "dense-only" in reason, reason


def test_reject_thd_dense_stats():
    """A ragged graph with a DENSE per-batch Stats tensor (no ragged offset,
    per-sequence stride ``[H*S_max, S_max, 1, 1]``) reads as head-major with
    head stride S_max while its storage is per-batch rectangles: the row must
    decline it rather than infer a packing from that stride."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    case = _thd_case((256, 128), (256, 128), 2, _D, torch.bfloat16)
    g, _, _ = _build_thd_bwd_graph(case)
    node = g.nodes[0]
    s_max_q = max(case.lens_q)
    node.inputs["stats"].set_ragged_offset(None)
    node.inputs["stats"].set_stride([case.h * s_max_q, s_max_q, 1, 1])
    try:
        g.validate()
        g.build_operation_graph()
    except cudnn.cudnnGraphNotSupportedError:
        return
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    reason = mismatch(spec.capabilities, facts)
    assert reason is not None and "dense per-batch stats" in reason, reason
