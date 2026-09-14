# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""``sdpa_bwd_sm80`` reads the bound ragged offsets on device (issue #737).

A ragged graph binds one ragged-offset tensor per port, and a padded THD
layout -- TE's ``cu_seqlens_padded`` -- puts a sequence's rows somewhere other
than ``prefix(lengths)``.  The setup launch derives every port's per-sequence
TOKEN origin from its own offsets (``ro[b] * multiplier / token_stride``), so
Q/K/V/O/dO and the Stats are read, and dQ/dK/dV written, where the caller put
them; the rows in the gaps between sequences are never touched.  Every port may
be padded differently (TE pads the Q side and the KV side independently), the
offsets may be element counts or token counts with a multiplier, and an offset
that is not a whole number of tokens makes the sequence dead: nothing read or
written, and the metadata's flag word set.

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
        return x

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


def test_offset_not_a_whole_token_marks_the_sequence_dead():
    """An offset that is not a multiple of the port's token stride cannot be
    addressed: the setup launch marks that sequence dead on every port, the
    kernels leave its rows untouched and compute the others, and the
    metadata's flag word is set (read here after a synchronize; nothing on the
    execute path reads it)."""
    from cudnn.frost.tile_dsl.thd import THD_ORG_FLAG_OFF
    from cudnn.sdpa.bwd.api_dsl import SdpaBwdDslSm80
    from cudnn.sdpa.bwd.kernels.thd_helpers import SM80_BWD_N_ORG_PORTS

    lens_q, lens_kv = (128, 96, 200), (128, 96, 200)
    case = _thd_case(lens_q, lens_kv, 2, _D, torch.bfloat16)
    dev, b, h, d = "cuda", case.b, case.h, case.d
    view = lambda t: t.permute(0, 2, 1, 3)  # noqa: E731
    s_max = max(lens_q)
    env = lambda n, s, nh: torch.empty(1, n, s, nh, d, device=dev, dtype=torch.bfloat16)[0].permute(0, 2, 1, 3)  # noqa: E731
    e = env(b, s_max, h)
    e_stats = torch.empty(b, h, s_max, 1, device=dev, dtype=torch.float32)
    ts = h * d
    # Packed offsets everywhere, except K's offset for sequence 1 lands one
    # ELEMENT into a token.
    cu = torch.tensor(case.cu_q, dtype=torch.int64, device=dev)
    ro_ok = cu * ts
    ro_k = ro_ok.clone()
    ro_k[1] += 1
    api = SdpaBwdDslSm80(
        sample_q=e,
        sample_k=e,
        sample_v=e,
        sample_o=e,
        sample_do=e,
        sample_stats=e_stats,
        sample_dq=e,
        sample_dk=e,
        sample_dv=e,
        scale_softmax=case.scale,
        thd=True,
        max_total_seq_len_q=case.t_q,
        max_total_seq_len_kv=case.t_kv,
        thd_ragged_offsets={r: (torch.int64, 1) for r in ("q", "k", "v", "o", "do", "dq", "dk", "dv")},
    )
    assert api.check_support()
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device=dev)
    dq, dk, dv = (torch.full_like(x, float("nan")) for x in (case.q, case.k, case.v))
    ros = {r: ro_ok for r in ("q", "v", "o", "do", "dq", "dk", "dv")} | {"k": ro_k}
    api.execute(
        view(case.q), view(case.k), view(case.v), view(case.o), view(case.do), case.lse, view(dq), view(dk), view(dv),
        workspace=ws,
        seq_q_lens=torch.tensor(lens_q, dtype=torch.int32, device=dev),
        seq_kv_lens=torch.tensor(lens_kv, dtype=torch.int32, device=dev),
        thd_ragged_offsets=ros,
    )  # fmt: skip
    torch.cuda.synchronize()
    flag = int(api._thd_meta_view[THD_ORG_FLAG_OFF(b, SM80_BWD_N_ORG_PORTS)].item())
    assert flag == 1, "the non-whole-token offset must set the metadata flag"
    dead = slice(case.cu_q[1], case.cu_q[2])
    for name, x in (("dQ", dq), ("dK", dk), ("dV", dv)):
        assert torch.isnan(x[0, dead]).all(), f"{name}: the dead sequence's rows were written"
    # The live sequences are correct.
    for i in (0, 2):
        sl = slice(case.cu_q[i], case.cu_q[i] + lens_q[i])
        rq, rk, rv = _ref_bwd(
            case.q[0, sl].transpose(0, 1), case.k[0, sl].transpose(0, 1), case.v[0, sl].transpose(0, 1), case.do[0, sl].transpose(0, 1), case.scale
        )
        for name, got, want in (("dQ", dq[0, sl].transpose(0, 1), rq), ("dK", dk[0, sl].transpose(0, 1), rk), ("dV", dv[0, sl].transpose(0, 1), rv)):
            assert torch.isfinite(got).all(), f"seq {i} {name} unwritten"
            assert _cos(got, want) > 0.999, f"seq {i} {name}: cos {_cos(got, want):.6f}"


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
