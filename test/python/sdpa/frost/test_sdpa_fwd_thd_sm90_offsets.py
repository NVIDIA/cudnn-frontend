# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdpa_fwd_prefill_sm90`` reads the bound ragged offsets of a THD graph.

A ragged graph binds one ragged-offset tensor per port, and a padded THD layout -- TE's
``cu_seqlens_padded`` -- puts a sequence's rows somewhere other than ``prefix(lengths)``.
Q/K/V must be read, and O written, where the caller put them, and the O gaps are never written
(they hold a sentinel). Every port may be padded differently (TE pads the Q side and the KV side
independently), and the offsets may be element counts or token counts with a multiplier. The
addressing cases fill the gaps with finite values; a separate case fills them with NaN, which a
K/V tile that runs past its sequence end into the gap would turn into a NaN output (a masked
probability of exactly 0 times NaN is NaN), so the gaps must not reach the PV product either.
The packed case (offsets == prefix(lengths) * token stride) is the existing suite's. Stats rows
follow their own offsets too, in both packed Stats layouts, and an offset past its buffer stays
inside it (the setup launch clamps every origin and clips every map).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import _SM, requires_dsl
from frost_test_utils import select_engine as _select_engine

from test_sdpa_fwd_dsl_sm100 import _mask_graph_kwargs, _mask_ref_kwargs, _ref_sdpa_full

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(_SM != 90, reason="needs an SM90 Hopper GPU, have " + ("none" if _SM is None else f"sm_{_SM}")),
    requires_dsl,
]

_D = 512
_SENTINEL = 2048.0


def _starts(lens, pads, lead=0):
    """Padded start rows: ``lead`` empty rows, then sequence ``b`` occupies ``[start[b], start[b] + lens[b])``,
    then ``pads[b]`` empty rows."""
    starts, t = [], lead
    for n, p in zip(lens, pads, strict=True):
        starts.append(t)
        t += n + p
    return starts, t


def _run_padded_full(
    lens_q,
    lens_kv,
    pads,
    *,
    h_q=2,
    h_kv=2,
    mask="none",
    units="elements",
    dtype=torch.float16,
    seed=0,
    gap="nan",
    offset_dtype=torch.int64,
    stats=None,
    edit_offsets=None,
    lead=None,
    slack=0,
    handles=False,
):
    """Build and execute a padded THD forward graph on the SM90 row; returns the packed Q/K/V,
    the O (and Stats) storage and the per-port geometry. ``stats`` is None, ``"token_major"``
    (packed (T, H)) or ``"head_major"`` (packed (H, T)); ``edit_offsets(port, values)`` may
    rewrite a port's offset values after the geometry is laid out; ``lead`` gives a port empty
    rows before its first sequence; ``slack`` adds spare rows to every buffer. The fourth value
    is None, or with ``handles`` the graph, its variant pack, workspace, offset and length tensors
    and token strides, for a caller that re-executes or captures the plan."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    b = len(lens_q)
    d = _D
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    lead = lead or {}
    geom = {
        port: _starts(lens_q if port in ("q", "o", "stats") else lens_kv, pads.get(port, [0] * b), lead.get(port, 0)) for port in ("q", "k", "v", "o", "stats")
    }
    # One declared extent per side: the widest port's padded total bounds the views.
    tot = {port: max(geom[p][1] for p in (("q", "o") if port in ("q", "o", "stats") else ("k", "v"))) for port in geom}
    heads = {"q": h_q, "o": h_q, "k": h_kv, "v": h_kv}
    s_max_q, s_max_kv = max(lens_q), max(lens_kv)

    packed, stor = {}, {}
    for port in ("q", "k", "v"):
        lens = lens_q if port == "q" else lens_kv
        packed[port] = torch.randn(sum(lens), heads[port], d, generator=gen, device=dev, dtype=dtype)
        rows = tot[port] + max(s_max_q, s_max_kv) + slack
        if gap == "nan":
            buf = torch.full((rows, heads[port], d), float("nan"), device=dev, dtype=dtype)
        else:
            buf = torch.randn(rows, heads[port], d, generator=gen, device=dev, dtype=dtype) * 4
        cu = 0
        for i, n in enumerate(lens):
            buf[geom[port][0][i] : geom[port][0][i] + n] = packed[port][cu : cu + n]
            cu += n
        stor[port] = buf
    stor["o"] = torch.full((tot["o"] + s_max_q + slack, h_q, d), _SENTINEL, device=dev, dtype=dtype)

    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    vp, t = {}, {}

    ro_type = {torch.int32: cudnn.data_type.INT32, torch.int64: cudnn.data_type.INT64}[offset_dtype]
    ro_by_port, ts_by_port = {}, {}

    def ragged(port, x, ts):
        vals = geom[port][0] + [geom[port][1]]
        if edit_offsets is not None:
            vals = edit_offsets(port, vals)
        vals = torch.tensor(vals, dtype=torch.int64, device=dev)
        ro_t = (vals if units == "tokens" else vals * ts).to(offset_dtype).view(b + 1, 1, 1, 1)
        ro = g.tensor(name=f"{port}_ro", dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=ro_type)
        x.set_ragged_offset(ro)
        if units == "tokens":
            x.set_ragged_offset_multiplier(ts)
        vp[ro] = ro_t
        ro_by_port[port], ts_by_port[port] = ro_t, ts

    for port, s_max in (("q", s_max_q), ("k", s_max_kv), ("v", s_max_kv)):
        ts = heads[port] * d
        t[port] = g.tensor(name=port, dim=[b, heads[port], s_max, d], stride=[s_max * ts, d, ts, 1], data_type=io)
        ragged(port, t[port], ts)
        vp[t[port]] = stor[port]
    slq = torch.tensor(lens_q, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    slk = torch.tensor(lens_kv, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
    sq, skv = g.tensor_like(slq), g.tensor_like(slk)
    vp[sq], vp[skv] = slq, slk
    o, st = g.sdpa(
        name="sdpa",
        q=t["q"],
        k=t["k"],
        v=t["v"],
        generate_stats=stats is not None,
        attn_scale=1.0 / math.sqrt(d),
        use_padding_mask=True,
        seq_len_q=sq,
        seq_len_kv=skv,
        **_mask_graph_kwargs(mask),
    )
    ts_o = h_q * d
    o.set_output(True).set_dim([b, h_q, s_max_q, d]).set_stride([s_max_q * ts_o, d, ts_o, 1])
    ragged("o", o, ts_o)
    vp[o] = stor["o"]
    if stats is not None:
        rows = tot["stats"] + s_max_q + slack
        if stats == "token_major":
            stor["stats"] = torch.full((rows, h_q), _SENTINEL, device=dev, dtype=torch.float32)
            st.set_output(True).set_dim([b, h_q, s_max_q, 1]).set_stride([s_max_q * h_q, 1, h_q, 1]).set_data_type(cudnn.data_type.FLOAT)
            ragged("stats", st, h_q)
        else:
            stor["stats"] = torch.full((h_q, rows), _SENTINEL, device=dev, dtype=torch.float32)
            st.set_output(True).set_dim([b, h_q, s_max_q, 1]).set_stride([h_q * rows, rows, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
            ragged("stats", st, 1)
        vp[st] = stor["stats"]

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    _select_engine(g, engine_name(arch="sm90"))
    g.check_support()
    g.build_plans()
    workspace = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    g.execute(vp, workspace)
    torch.cuda.synchronize()
    if handles:
        return packed, stor, geom, SimpleNamespace(g=g, vp=vp, workspace=workspace, ro=ro_by_port, ts=ts_by_port, slq=slq, slk=slk)
    return packed, stor, geom, None


def _run_padded(*args, **kwargs):
    """``_run_padded_full`` without the handles: the packed Q/K/V, the O (and Stats) storage and the geometry."""
    return _run_padded_full(*args, **kwargs)[:3]


def _check(lens_q, lens_kv, packed, stor, geom, mask, stats=None, skip=()):
    """Every sequence's O (and Stats) rows at its origin; nothing written outside them.
    Sequences in ``skip`` (deliberately bad offsets) are not compared, but their rows still
    count as live."""
    o_starts = geom["o"][0]
    cu_q, cu_k = 0, 0
    live = torch.zeros(stor["o"].shape[0], dtype=torch.bool, device=stor["o"].device)
    st_rows = 0 if stats is None else stor["stats"].shape[0 if stats == "token_major" else 1]
    st_live = torch.zeros(st_rows, dtype=torch.bool, device=stor["o"].device)
    for i, (nq, nkv) in enumerate(zip(lens_q, lens_kv)):
        live[o_starts[i] : o_starts[i] + nq] = True
        st0 = geom["stats"][0][i]
        st_live[st0 : st0 + nq] = True
        if i not in skip and nq:
            qb = packed["q"][cu_q : cu_q + nq].permute(1, 0, 2).unsqueeze(0)
            kb = packed["k"][cu_k : cu_k + nkv].permute(1, 0, 2).unsqueeze(0)
            vb = packed["v"][cu_k : cu_k + nkv].permute(1, 0, 2).unsqueeze(0)
            ref, lse_ref = _ref_sdpa_full(qb, kb, vb, scale=1.0 / math.sqrt(_D), return_stats=True, **_mask_ref_kwargs(mask))
            got = stor["o"][o_starts[i] : o_starts[i] + nq].permute(1, 0, 2).unsqueeze(0)
            torch.testing.assert_close(got, ref, atol=5e-2, rtol=3e-2, msg=f"sequence {i}")
            if stats is not None:
                rows = stor["stats"][st0 : st0 + nq].transpose(0, 1) if stats == "token_major" else stor["stats"][:, st0 : st0 + nq]
                torch.testing.assert_close(rows, lse_ref[0], atol=2e-2, rtol=2e-2, msg=f"Stats of sequence {i}")
        cu_q += nq
        cu_k += nkv
    assert (stor["o"][~live] == _SENTINEL).all(), "a gap or tail row of O was written"
    if stats is not None:
        dead = stor["stats"][~st_live] if stats == "token_major" else stor["stats"][:, ~st_live]
        assert (dead == _SENTINEL).all(), "a gap or tail row of Stats was written"


_LENS_Q = [130, 64, 7]
_LENS_KV = [130, 200, 7]


@pytest.mark.parametrize(
    "pads",
    [
        pytest.param({"q": [5, 0, 9], "o": [5, 0, 9], "k": [3, 11, 0], "v": [3, 11, 0]}, id="te-padded"),
        pytest.param({"q": [0, 17, 1], "o": [4, 4, 4], "k": [0, 0, 0], "v": [8, 0, 2]}, id="per-port"),
        pytest.param({"k": [64, 1, 64], "v": [64, 1, 64]}, id="kv-only"),
    ],
)
@pytest.mark.parametrize("units", ["elements", "tokens"])
def test_sm90_thd_reads_ragged_offsets(pads, units):
    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, pads, units=units, gap="finite")
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none")


@pytest.mark.parametrize("mask", ["causal", "causal_br"])
def test_sm90_thd_ragged_offsets_masked(mask):
    pads = {"q": [5, 0, 9], "o": [5, 0, 9], "k": [3, 11, 0], "v": [3, 11, 0]}
    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, pads, h_q=4, h_kv=2, mask=mask, dtype=torch.bfloat16, seed=1, gap="finite")
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, mask)


def test_sm90_thd_gap_rows_never_reach_the_output():
    """NaN in the gaps: a K/V tile that reads past a sequence end into its gap poisons that sequence."""
    pads = {"q": [5, 0, 9], "o": [5, 0, 9], "k": [3, 11, 0], "v": [3, 11, 0]}
    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, pads, gap="nan")
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none")


def test_sm90_thd_packed_offsets_unchanged():
    """Offsets equal to prefix(lengths): the packed layout every existing caller binds."""
    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, {})
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none")


@pytest.mark.parametrize("stats", ["token_major", "head_major"])
@pytest.mark.parametrize("units", ["elements", "tokens"])
def test_sm90_thd_stats_read_their_own_offsets(stats, units):
    """Stats rows land at the Stats port's own origins, which may differ from O's."""
    pads = {"q": [5, 0, 9], "o": [5, 0, 9], "k": [3, 11, 0], "v": [3, 11, 0], "stats": [2, 7, 0]}
    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, pads, units=units, gap="nan", stats=stats)
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none", stats=stats)


@pytest.mark.parametrize("units", ["elements", "tokens"])
def test_sm90_thd_int32_offsets(units):
    pads = {"q": [0, 17, 1], "o": [4, 4, 4], "k": [0, 0, 0], "v": [8, 0, 2]}
    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, pads, units=units, offset_dtype=torch.int32, gap="nan")
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none")


def test_sm90_thd_single_padded_sequence():
    """B = 1 takes the setup's single-writer path; a leading gap still moves every port."""
    lead = {"q": 3, "o": 6, "k": 9, "v": 1, "stats": 4}
    packed, stor, geom = _run_padded([77], [150], {}, gap="nan", stats="token_major", lead=lead)
    _check([77], [150], packed, stor, geom, "none", stats="token_major")


def test_sm90_thd_many_padded_sequences():
    """B = 40 (more sequences than the setup's map-writing warps and a prefix warp's lanes),
    random gaps on every port, empty Q sequences, GQA, causal, NaN gaps."""
    gen = torch.Generator().manual_seed(7)
    b = 40
    lens_q = torch.randint(0, 70, (b,), generator=gen).tolist()
    lens_kv = [max(n, 1) + int(x) for n, x in zip(lens_q, torch.randint(0, 30, (b,), generator=gen))]
    pads = {port: torch.randint(0, 10, (b,), generator=gen).tolist() for port in ("q", "k", "v", "o")}
    packed, stor, geom = _run_padded(lens_q, lens_kv, pads, h_q=4, h_kv=2, mask="causal_br", dtype=torch.bfloat16, gap="nan")
    _check(lens_q, lens_kv, packed, stor, geom, "causal_br")


def test_sm90_thd_offset_past_the_buffer_stays_inside_it():
    """A K/V offset past its buffer is clamped into it: the launch completes, the other
    sequences are exact, and nothing outside O's sequences is written."""
    pads = {"q": [5, 0, 9], "o": [5, 0, 9], "k": [3, 11, 0], "v": [3, 11, 0]}

    def far(port, vals):
        return vals[:2] + [10**7] + vals[3:] if port in ("k", "v") else vals

    packed, stor, geom = _run_padded(_LENS_Q, _LENS_KV, pads, gap="nan", edit_offsets=far)
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none", skip=(2,))
    torch.cuda.synchronize()


def test_sm90_thd_cuda_graph_replay_reads_new_offsets():
    """TE replays one captured graph while ``cu_seqlens_padded`` changes every step: a replay
    reads the offsets and lengths bound at replay time, not the capture's."""
    pads_a = {"q": [5, 0, 9], "o": [5, 0, 9], "k": [3, 11, 0], "v": [3, 11, 0], "stats": [2, 7, 0]}
    packed, stor, geom, h = _run_padded_full(_LENS_Q, _LENS_KV, pads_a, gap="nan", stats="token_major", slack=96, handles=True)
    _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none", stats="token_major")
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            h.g.execute(h.vp, h.workspace)

        def replay():
            stor["o"].fill_(_SENTINEL)
            stor["stats"].fill_(_SENTINEL)
            graph.replay()
            torch.cuda.synchronize()

        replay()
        _check(_LENS_Q, _LENS_KV, packed, stor, geom, "none", stats="token_major")

        # A new step: other lengths, other gaps, the same buffers rewritten in place.
        lens_q, lens_kv = [64, 130, 30], [100, 7, 200]
        pads_b = {"q": [0, 33, 2], "o": [12, 0, 1], "k": [40, 0, 5], "v": [0, 9, 9], "stats": [0, 3, 30]}
        geom_b = {port: _starts(lens_q if port in ("q", "o", "stats") else lens_kv, pads_b[port]) for port in pads_b}
        gen = torch.Generator(device="cuda").manual_seed(3)
        packed_b = {}
        for port in ("q", "k", "v"):
            lens = lens_q if port == "q" else lens_kv
            packed_b[port] = torch.randn(sum(lens), stor[port].shape[1], _D, generator=gen, device="cuda", dtype=stor[port].dtype)
            assert geom_b[port][1] <= stor[port].shape[0]
            stor[port].fill_(float("nan"))
            cu = 0
            for start, n in zip(geom_b[port][0], lens):
                stor[port][start : start + n] = packed_b[port][cu : cu + n]
                cu += n
        for port, ro in h.ro.items():
            ro.copy_((torch.tensor(geom_b[port][0] + [geom_b[port][1]], device="cuda") * h.ts[port]).view_as(ro))
        h.slq.copy_(torch.tensor(lens_q, dtype=torch.int32, device="cuda").view_as(h.slq))
        h.slk.copy_(torch.tensor(lens_kv, dtype=torch.int32, device="cuda").view_as(h.slk))
        replay()
        _check(lens_q, lens_kv, packed_b, stor, geom_b, "none", stats="token_major")
    finally:
        graph.reset()
