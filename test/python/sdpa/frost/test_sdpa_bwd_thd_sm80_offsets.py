# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""``sdpa_bwd_sm80`` reads the bound ragged offsets on device (issue #737).

A ragged graph binds one ragged-offset tensor per port, and a padded THD
layout -- TE's ``cu_seqlens_padded`` -- puts a sequence's rows somewhere other
than ``prefix(lengths)``.  The setup launch derives every port's per-sequence
TOKEN origin from its own offsets (``ro[b] * multiplier / token_stride``), so
Q/K/V/O/dO and the Stats are read, and dQ/dK/dV written, where the caller put
them; the rows in the gaps between sequences are never touched.  Every port may
be padded differently (TE pads the Q side and the KV side independently) and
the offsets may be element counts or token counts with a multiplier.  Whole-
token offsets are a supported-input precondition, not checked on device.
Origins are refreshed from the bound values on every execute and replay.

Reuses the packed-case builder of test_sdpa_bwd_thd_sm80.py: the reference is
per sequence and does not care where the rows live.
"""

from __future__ import annotations

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_sdpa_bwd_thd_sm80 import _D, _ENGINE, _below_tol, _cos, _plan_graph, _ref_bwd, _thd_case

import cudnn

_SM80 = pytest.mark.skipif(_SM != 80, reason="needs an SM80 (A100) GPU, have " + ("none" if _SM is None else f"sm_{_SM}"))
pytestmark = [pytest.mark.L0, _SM80, requires_dsl]

# Every port a ragged backward graph binds an offset for, in the builder's
# vocabulary; "q side" ports are placed by the Q padding, "kv side" by the KV one.
_Q_SIDE = ("q", "o", "do", "dq", "stats")
_KV_SIDE = ("k", "v", "dk", "dv")


def _starts(lens, pads):
    """Padded per-sequence start rows and the padded total: sequence ``b``
    occupies ``[start[b], start[b] + lens[b])`` and ``pads[b]`` empty rows follow."""
    starts, t = [], 0
    for n, p in zip(lens, pads, strict=True):
        starts.append(t)
        t += n + p
    return starts, t


def _scatter(packed, cu, lens, starts, total):
    """``packed`` ([1, T, H, D], sequence b at cu[b]) re-laid with sequence b at
    ``starts[b]`` in a NaN-filled buffer of ``total`` rows (the gaps are NaN: a
    read of a gap row would poison the result)."""
    out = torch.full((1, total, *packed.shape[2:]), float("nan"), device=packed.device, dtype=packed.dtype)
    for b, n in enumerate(lens):
        out[0, starts[b] : starts[b] + n] = packed[0, cu[b] : cu[b] + n]
    return out


def _gather(padded, lens, starts):
    """Inverse of :func:`_scatter` onto a packed [1, sum(lens), ...] tensor."""
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    out = torch.empty((1, cu[-1], *padded.shape[2:]), device=padded.device, dtype=padded.dtype)
    for b, n in enumerate(lens):
        out[0, cu[b] : cu[b] + n] = padded[0, starts[b] : starts[b] + n]
    return out


def _gap_rows(padded, lens, starts):
    """The rows of ``padded`` that belong to no sequence."""
    live = torch.zeros(padded.shape[1], dtype=torch.bool, device=padded.device)
    for b, n in enumerate(lens):
        live[starts[b] : starts[b] + n] = True
    return padded[0, ~live]


def _build_padded_graph(case, *, pads, units="elements", stats_layout="head_major", **sdpa_kwargs):
    """A ragged backward graph whose ports are PADDED: ``pads[role]`` is the
    per-sequence pad row count after each sequence on that port (a role
    missing from ``pads`` is packed).  ``units="tokens"`` binds token-count
    offsets with ``ragged_offset_multiplier = token stride``.  Returns the
    graph, its variant pack with the inputs bound in their padded records, the
    gradient handles, and the per-role ``(starts, total)`` geometry."""
    b, h, hkv, d, d_v, dev = case.b, case.h, case.hkv, case.d, case.d_v, "cuda"
    io = cudnn.data_type.HALF if case.dtype == torch.float16 else cudnn.data_type.BFLOAT16
    s_max_q, s_max_kv = max(max(case.lens_q), 1), max(max(case.lens_kv), 1)
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    vp, t, geom = {}, {}, {}

    # Every port's starts come from its own pads; every buffer on a side is
    # allocated at the SIDE's declared total (the widest port's padded span):
    # the declared max_total_seq_len bounds the views the lowering binds, so a
    # port's buffer has to cover it whatever its own padding adds up to.
    for role in _Q_SIDE + _KV_SIDE:
        lens = case.lens_q if role in _Q_SIDE else case.lens_kv
        geom[role] = _starts(lens, pads.get(role, [0] * b))
    tot_q = max(geom[r][1] for r in _Q_SIDE)
    tot_kv = max(geom[r][1] for r in _KV_SIDE)
    geom = {r: (st, tot_q if r in _Q_SIDE else tot_kv) for r, (st, _) in geom.items()}

    def _geom(role):
        return geom[role]

    def _offsets(role, starts, total, ts):
        vals = starts + [total]
        if units == "tokens":
            return torch.tensor(vals, dtype=torch.int64, device=dev).view(b + 1, 1, 1, 1), ts
        return (torch.tensor(vals, dtype=torch.int64, device=dev) * ts).view(b + 1, 1, 1, 1), 1

    def _port(role, s_max, nh, dd):
        ts = nh * dd
        starts, total = _geom(role)
        ro_t, mult = _offsets(role, starts, total, ts)
        x = g.tensor(name=role, dim=[b, nh, s_max, dd], stride=[s_max * ts, dd, ts, 1], data_type=io)
        ro = g.tensor(name=f"{role}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        x.set_ragged_offset(ro)
        if mult != 1:
            x.set_ragged_offset_multiplier(mult)
        vp[ro] = ro_t
        ro_vals[role] = (ro_t, ts)
        return x

    ro_vals = {}
    t["q"] = _port("q", s_max_q, h, d)
    t["o"] = _port("o", s_max_q, h, d_v)
    t["do"] = _port("do", s_max_q, h, d_v)
    t["k"] = _port("k", s_max_kv, hkv, d)
    t["v"] = _port("v", s_max_kv, hkv, d_v)

    # Stats: token-major (T, H) rows at the Stats port's starts (token stride
    # H), or head-major (1, H, head_stride) with the token index as the offset.
    st_starts, st_total = _geom("stats")
    if stats_layout == "token_major":
        st_stride = [st_total * h, 1, h, 1]
        st_ro_t, st_mult = _offsets("stats", st_starts, st_total, h)
        stats_stor = torch.full((st_total, h), float("nan"), device=dev, dtype=torch.float32)
        for i, n in enumerate(case.lens_q):
            stats_stor[st_starts[i] : st_starts[i] + n] = case.lse[0, :, case.cu_q[i] : case.cu_q[i] + n].transpose(0, 1)
    else:
        hs = st_total
        st_stride = [h * hs, hs, 1, 1]
        st_ro_t, st_mult = _offsets("stats", st_starts, st_total, 1)
        stats_stor = torch.full((h, hs), float("nan"), device=dev, dtype=torch.float32)
        for i, n in enumerate(case.lens_q):
            stats_stor[:, st_starts[i] : st_starts[i] + n] = case.lse[0, :, case.cu_q[i] : case.cu_q[i] + n]
    st = g.tensor(name="stats", dim=[b, h, s_max_q, 1], stride=st_stride, data_type=cudnn.data_type.FLOAT)
    st_ro = g.tensor(name="stats_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
    st.set_ragged_offset(st_ro)
    if st_mult != 1:
        st.set_ragged_offset_multiplier(st_mult)
    vp[st_ro] = st_ro_t
    vp[st] = stats_stor
    ro_vals["stats"] = (st_ro_t, h if stats_layout == "token_major" else 1)

    slq = torch.tensor(case.lens_q, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    slk = torch.tensor(case.lens_kv, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    tq_len = g.tensor(name="seq_len_q", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    tk_len = g.tensor(name="seq_len_kv", dim=[b, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT32)
    vp[tq_len], vp[tk_len] = slq, slk

    # Declared totals cover the PADDED span of each side (what a padded-THD
    # producer declares); the views are bound at min(B * S_max, declared).
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
        max_total_seq_len_q=tot_q,
        max_total_seq_len_kv=tot_kv,
    )
    kw.update(sdpa_kwargs)
    dq_t, dk_t, dv_t = g.sdpa_backward(**kw)
    for out, role, nh, dd, s_max in ((dq_t, "dq", h, d, s_max_q), (dk_t, "dk", hkv, d, s_max_kv), (dv_t, "dv", hkv, d_v, s_max_kv)):
        ts = nh * dd
        starts, total = geom[role]
        ro_t, mult = _offsets(role, starts, total, ts)
        out.set_output(True).set_data_type(io).set_dim([b, nh, s_max, dd]).set_stride([s_max * ts, dd, ts, 1])
        ro = g.tensor(name=f"{role}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.INT64)
        out.set_ragged_offset(ro)
        if mult != 1:
            out.set_ragged_offset_multiplier(mult)
        vp[ro] = ro_t
        ro_vals[role] = (ro_t, ts)

    # Inputs in their padded records.
    for role, src, lens, cu in (
        ("q", case.q, case.lens_q, case.cu_q),
        ("o", case.o, case.lens_q, case.cu_q),
        ("do", case.do, case.lens_q, case.cu_q),
        ("k", case.k, case.lens_kv, case.cu_k),
        ("v", case.v, case.lens_kv, case.cu_k),
    ):
        starts, total = geom[role]
        vp[t[role]] = _scatter(src, cu, lens, starts, total)
    # For in-place rebinding (changed offsets on the same plan and buffers).
    g._test_rebind = dict(ro=ro_vals, ports=t, stats=(stats_stor, stats_layout), units=units)
    return g, vp, (dq_t, dk_t, dv_t), geom


def _check_padded(case, geom, dq, dk, dv):
    """Per-sequence comparison against the fp64 reference, reading each
    gradient at its own padded starts; the gap rows must still be NaN."""
    for name, x, role, lens in (("dQ", dq, "dq", case.lens_q), ("dK", dk, "dk", case.lens_kv), ("dV", dv, "dv", case.lens_kv)):
        starts, _ = geom[role]
        gaps = _gap_rows(x, lens, starts)
        assert gaps.numel() == 0 or torch.isnan(gaps).all(), f"{name}: a gap row between sequences was written"
    dq_p, dk_p, dv_p = (_gather(x, lens, geom[r][0]) for x, r, lens in ((dq, "dq", case.lens_q), (dk, "dk", case.lens_kv), (dv, "dv", case.lens_kv)))
    bad = []
    grp = case.h // case.hkv
    rep = (lambda x: x.repeat_interleave(grp, dim=0)) if grp > 1 else (lambda x: x)
    for i in range(case.b):
        if case.lens_q[i] == 0 or case.lens_kv[i] == 0:
            continue
        sl_q = slice(case.cu_q[i], case.cu_q[i] + case.lens_q[i])
        sl_k = slice(case.cu_k[i], case.cu_k[i] + case.lens_kv[i])
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
        if grp > 1:
            rk = rk.reshape(case.hkv, grp, *rk.shape[1:]).sum(1)
            rv = rv.reshape(case.hkv, grp, *rv.shape[1:]).sum(1)
        for name, got, want in (
            ("dQ", dq_p[0, sl_q].transpose(0, 1), rq),
            ("dK", dk_p[0, sl_k].transpose(0, 1), rk),
            ("dV", dv_p[0, sl_k].transpose(0, 1), rv),
        ):
            assert torch.isfinite(got).all(), f"seq {i} {name}: a live row was left unwritten"
            bad.append(f"seq {i} {name}: cos {_cos(got, want):.6f} (lens q={case.lens_q[i]} kv={case.lens_kv[i]})")
    failures = [m for m in bad if _below_tol(m)]
    assert not failures, "\n".join(bad)


def _run_padded(lens_q, lens_kv, pads, *, units="elements", h=2, hkv=None, d=_D, stats_layout="head_major", **kw):
    case = _thd_case(
        lens_q,
        lens_kv,
        h,
        d,
        torch.bfloat16,
        hkv=hkv,
        causal=bool(kw.get("use_causal_mask")),
    )
    g, vp, (dq_t, dk_t, dv_t), geom = _build_padded_graph(case, pads=pads, units=units, stats_layout=stats_layout, **kw)
    _plan_graph(g)
    dev = "cuda"
    dq = torch.full((1, geom["dq"][1], case.h, case.d), float("nan"), device=dev, dtype=torch.bfloat16)
    dk = torch.full((1, geom["dk"][1], case.hkv, case.d), float("nan"), device=dev, dtype=torch.bfloat16)
    dv = torch.full((1, geom["dv"][1], case.hkv, case.d_v), float("nan"), device=dev, dtype=torch.bfloat16)
    vp.update({dq_t: dq, dk_t: dk, dv_t: dv})
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    _check_padded(case, geom, dq, dk, dv)
    return case, geom, (dq, dk, dv)


# --- graph tests -------------------------------------------------------------


@pytest.mark.parametrize("units", ("elements", "tokens"))
def test_padded_thd_te_style(units):
    """TE's padded THD: every port on a side shares one ``cu_seqlens_padded``
    (Q side and KV side padded differently), offsets in elements or in tokens
    with the token-stride multiplier."""
    lens_q, lens_kv = (300, 128, 200), (256, 100, 224)
    pad_q, pad_kv = [4, 64, 8], [64, 28, 0]
    pads = {r: pad_q for r in _Q_SIDE} | {r: pad_kv for r in _KV_SIDE}
    _run_padded(lens_q, lens_kv, pads, units=units)


def test_padded_thd_every_port_differently():
    """Each of the nine ports padded on its own schedule: the origins are
    per port, not per side."""
    lens_q, lens_kv = (200, 96, 130), (150, 200, 64)
    pads = {
        "q": [8, 0, 16],
        "o": [0, 32, 0],
        "do": [16, 16, 16],
        "dq": [0, 0, 64],
        "stats": [32, 0, 8],
        "k": [64, 0, 0],
        "v": [0, 64, 0],
        "dk": [8, 8, 8],
        "dv": [0, 0, 128],
    }
    _run_padded(lens_q, lens_kv, pads)


@pytest.mark.parametrize("stats_layout", ("head_major", "token_major"))
def test_padded_thd_stats_packings(stats_layout):
    """The Stats port's own offsets, in both packings Rule S1 admits."""
    pads = {r: [16, 0, 32] for r in _Q_SIDE} | {r: [0, 48, 0] for r in _KV_SIDE}
    _run_padded((128, 200, 64), (256, 128, 100), pads, stats_layout=stats_layout)


def test_padded_thd_gqa_causal():
    """The GQA fold and the causal family ride the origins too."""
    pads = {r: [64, 0, 8] for r in _Q_SIDE} | {r: [0, 64, 24] for r in _KV_SIDE}
    _run_padded((300, 128, 200), (300, 128, 200), pads, h=4, hkv=2, use_causal_mask=True)


def test_padded_thd_capacity_past_the_pads():
    """Declared totals wider than the padded span: the rows past the last
    sequence's pad stay untouched like any gap row."""
    pads = {r: [16, 16, 200] for r in _Q_SIDE} | {r: [0, 0, 128] for r in _KV_SIDE}
    _run_padded((100, 300, 50), (256, 128, 64), pads)


# --- the whole-token contract ---------------------------------------------------


def test_probe_offsets_do_not_change_eligibility():
    """Ragged offsets are execute-time data: a padded graph is served by the
    same plan-time facts as a packed one."""
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd.engines import ENGINE_SPECS, mismatch

    case = _thd_case((256, 128), (256, 128), 2, _D, torch.bfloat16)
    pads = {r: [64, 0] for r in _Q_SIDE} | {r: [0, 64] for r in _KV_SIDE}
    g, _, _, _ = _build_padded_graph(case, pads=pads)
    g.validate()
    g.build_operation_graph()
    facts = ga.analyze(g)
    spec = next(s for s in ENGINE_SPECS if s.name == _ENGINE)
    assert mismatch(spec.capabilities, facts) is None


# --- changed offsets: re-execute and CUDA-graph replay ---------------------


def _rebind(g, vp, case, geom):
    """Move every port's sequences to ``geom``'s starts IN PLACE: rewrite the
    bound offset tensors' values and re-lay the inputs (and Stats) in their
    existing NaN-filled buffers.  Plan, pack and allocations are unchanged."""
    rb = g._test_rebind
    for role, (ro_t, ts) in rb["ro"].items():
        starts, total = geom[role]
        vals = torch.tensor(starts + [total], dtype=torch.int64, device="cuda")
        ro_t.copy_((vals if rb["units"] == "tokens" else vals * ts).view_as(ro_t))
    for role, src, lens, cu in (
        ("q", case.q, case.lens_q, case.cu_q),
        ("o", case.o, case.lens_q, case.cu_q),
        ("do", case.do, case.lens_q, case.cu_q),
        ("k", case.k, case.lens_kv, case.cu_k),
        ("v", case.v, case.lens_kv, case.cu_k),
    ):
        starts, total = geom[role]
        vp[rb["ports"][role]].copy_(_scatter(src, cu, lens, starts, total))
    stor, layout = rb["stats"]
    stor.fill_(float("nan"))
    st_starts, _ = geom["stats"]
    for i, n in enumerate(case.lens_q):
        rows = case.lse[0, :, case.cu_q[i] : case.cu_q[i] + n]
        if layout == "token_major":
            stor[st_starts[i] : st_starts[i] + n] = rows.transpose(0, 1)
        else:
            stor[:, st_starts[i] : st_starts[i] + n] = rows


def _moved_geom(case, pads):
    """The geometry of ``pads`` at the SAME per-side totals as the build."""
    geom = {r: _starts(case.lens_q if r in _Q_SIDE else case.lens_kv, pads[r]) for r in _Q_SIDE + _KV_SIDE}
    return geom


def _padded_plan(pads, *, lens_q=(200, 96, 160), lens_kv=(176, 128, 64), units="elements", **kw):
    case = _thd_case(lens_q, lens_kv, 2, _D, torch.bfloat16, causal=bool(kw.get("use_causal_mask")))
    g, vp, outs, geom = _build_padded_graph(case, pads=pads, units=units, **kw)
    _plan_graph(g)
    grads = {
        "dq": torch.full((1, geom["dq"][1], case.h, case.d), float("nan"), device="cuda", dtype=torch.bfloat16),
        "dk": torch.full((1, geom["dk"][1], case.hkv, case.d), float("nan"), device="cuda", dtype=torch.bfloat16),
        "dv": torch.full((1, geom["dv"][1], case.hkv, case.d_v), float("nan"), device="cuda", dtype=torch.bfloat16),
    }
    vp.update(dict(zip(outs, grads.values())))
    ws = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return case, g, vp, geom, grads, ws


# Two layouts with the same per-side totals: the padding moves between sequences.
_PADS_A = {r: [32, 0, 16] for r in _Q_SIDE} | {r: [0, 48, 0] for r in _KV_SIDE}
_PADS_B = {r: [0, 16, 32] for r in _Q_SIDE} | {r: [24, 0, 24] for r in _KV_SIDE}


@pytest.mark.parametrize("units", ("elements", "tokens"))
def test_changed_offsets_on_re_execute(units):
    """Changing the offset VALUES between executes of one plan moves every
    port's rows: origins are recomputed on device each execute, never cached."""
    case, g, vp, geom, grads, ws = _padded_plan(_PADS_A, units=units)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    _check_padded(case, geom, grads["dq"], grads["dk"], grads["dv"])

    moved = _moved_geom(case, _PADS_B)
    assert all(moved[r][1] == geom[r][1] for r in moved), "test layouts must share the declared totals"
    _rebind(g, vp, case, moved)
    for x in grads.values():
        x.fill_(float("nan"))
    g.execute(vp, ws)
    torch.cuda.synchronize()
    _check_padded(case, moved, grads["dq"], grads["dk"], grads["dv"])


def test_changed_offsets_on_cuda_graph_replay():
    """A captured execute replays with the offsets bound at replay time: the
    setup launch is in the graph, so new offset values place every port anew."""
    case, g, vp, geom, grads, ws = _padded_plan(_PADS_A)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g.execute(vp, ws)  # warm-up outside capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    cg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cg):
        g.execute(vp, ws)

    for pads in (_PADS_B, _PADS_A):
        moved = _moved_geom(case, pads)
        _rebind(g, vp, case, moved)
        for x in grads.values():
            x.fill_(float("nan"))
        cg.replay()
        torch.cuda.synchronize()
        _check_padded(case, moved, grads["dq"], grads["dk"], grads["dv"])


def test_padded_thd_staged_head_dim():
    """A head dim inside the flavor envelope runs the staged path (inputs copied
    row-for-row into widened scratch); origins stay in the caller's tokens."""
    pads = {r: [48, 0] for r in _Q_SIDE} | {r: [0, 32] for r in _KV_SIDE}
    _run_padded((256, 128), (192, 160), pads, d=96)


def test_padded_thd_staged_gaps_wider_than_the_packed_capacity():
    """Staged head dim with gaps wider than ``B * S_max``: the staging copies
    must cover the declared physical span, not the compact packed capacity
    (here sequence 1 starts at token 128 while ``B * S_max`` is 34)."""
    pads = {r: [111, 113] for r in _Q_SIDE + _KV_SIDE}
    _run_padded((17, 15), (17, 15), pads, d=96)


def test_staged_offsets_inside_the_workspace_are_rejected():
    """An offset tensor aliasing the caller workspace's staging prefix would be
    overwritten by the input copies before the setup launch reads it, so the
    binding is rejected before any copy runs."""
    case = _thd_case((64, 32), (64, 32), 2, 96, torch.bfloat16)
    g, vp, (dq_t, dk_t, dv_t), geom = _build_padded_graph(case, pads={r: [16, 0] for r in _Q_SIDE + _KV_SIDE})
    _plan_graph(g)
    vp[dq_t] = torch.empty((1, geom["dq"][1], case.h, case.d), device="cuda", dtype=torch.bfloat16)
    vp[dk_t] = torch.empty((1, geom["dk"][1], case.hkv, case.d), device="cuda", dtype=torch.bfloat16)
    vp[dv_t] = torch.empty((1, geom["dv"][1], case.hkv, case.d_v), device="cuda", dtype=torch.bfloat16)
    ws = torch.zeros(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    ro_q = g._test_rebind["ro"]["q"][0]
    key = next(k for k, v in vp.items() if v is ro_q)
    vp[key] = ws[: 8 * ro_q.numel()].view(torch.int64).view_as(ro_q)
    with pytest.raises(ValueError, match="overlaps"):
        g.execute(vp, ws)
