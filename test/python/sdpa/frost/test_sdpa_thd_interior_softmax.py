# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Causal THD interior softmax across changed inputs, masks and graph replay."""

import itertools

import pytest
import torch

from frost_test_utils import requires_dsl

B, S, H, D = 3, 2048, 4, 128
CAP = B * S


def _make_case(kv_heads, scale, seed, qlens, klens, d_qk=D, d_v=D):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cpu = [
        (torch.randn(CAP, heads, d_qk if i < 2 else d_v, generator=gen) * (scale if i < 2 else 0.5)).bfloat16()
        for i, heads in enumerate((H, kv_heads, kv_heads))
    ]
    cuq, cuk = [0], [0]
    for nq, nk in zip(qlens, klens):
        cuq.append(cuq[-1] + nq)
        cuk.append(cuk[-1] + nk)
    values = {n: t.cuda() for n, t in zip(("q", "k", "v"), cpu)}
    values.update(
        sq=torch.tensor(qlens, device="cuda", dtype=torch.int32).view(B, 1, 1, 1),
        sk=torch.tensor(klens, device="cuda", dtype=torch.int32).view(B, 1, 1, 1),
        rq=(torch.tensor(cuq, device="cuda", dtype=torch.int64) * H * d_qk).view(B + 1, 1, 1, 1),
        rk=(torch.tensor(cuk, device="cuda", dtype=torch.int64) * kv_heads * d_qk).view(B + 1, 1, 1, 1),
    )
    values["rv"] = values["rk"] // d_qk * d_v
    values["ro"] = values["rq"] // d_qk * d_v
    return values, (cpu, cuq, cuk)


def _reference(host, bottom_right):
    (q, k, v), cuq, cuk = host
    outputs, stats = [], []
    for i in range(B):
        nq, nk = cuq[i + 1] - cuq[i], cuk[i + 1] - cuk[i]
        if not nq:
            continue
        qr, kr, vr = [x.double().transpose(0, 1) for x in (q[cuq[i] : cuq[i + 1]], k[cuk[i] : cuk[i + 1]], v[cuk[i] : cuk[i + 1]])]
        repeat = H // k.shape[1]
        scores = qr @ kr.repeat_interleave(repeat, 0).transpose(-1, -2) * q.shape[-1] ** -0.5
        keep = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + (nk - nq if bottom_right else 0)
        scores.masked_fill_(~keep, -torch.inf)
        probs = scores.softmax(-1)
        probs = torch.where(keep.any(-1)[None, :, None], probs, torch.zeros_like(probs))
        outputs.append((probs @ vr.repeat_interleave(repeat, 0)).transpose(0, 1))
        stats.append(scores.logsumexp(-1).transpose(0, 1))
    return torch.cat(outputs), torch.cat(stats)


def _build_graph(values, kv_heads, bottom_right, layout, cta_mma=2):
    import cudnn

    d_qk, d_v = values["q"].shape[-1], values["v"].shape[-1]
    io = cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nodes = {}
    for name, heads in [("q", H), ("k", kv_heads), ("v", kv_heads)]:
        dim = values[name].shape[-1]
        nodes[name] = g.tensor(dim=[B, heads, S, dim], stride=[S * heads * dim, dim, heads * dim, 1], data_type=io, name=name)
    for name in ("sq", "sk", "rq", "rk", "rv", "ro"):
        nodes[name] = g.tensor_like(values[name], name=name)
    nodes["q"].set_ragged_offset(nodes["rq"])
    nodes["k"].set_ragged_offset(nodes["rk"])
    nodes["v"].set_ragged_offset(nodes["rv"])
    mask = dict(use_causal_mask_bottom_right=True) if bottom_right else dict(use_causal_mask=True)
    o, stats = g.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        generate_stats=True,
        attn_scale=d_qk**-0.5,
        use_padding_mask=True,
        seq_len_q=nodes["sq"],
        seq_len_kv=nodes["sk"],
        **mask,
    )
    o.set_output(True).set_dim([B, H, S, d_v]).set_stride([S * H * d_v, d_v, H * d_v, 1]).set_ragged_offset(nodes["ro"])
    factor = 1 if layout == "head_major" else H
    values["rs"] = values["rq"] // d_qk // H * factor
    nodes["rs"] = g.tensor_like(values["rs"], name="stats_offsets")
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([B, H, S, 1])
    stats.set_stride([H * CAP, CAP, 1, 1] if layout == "head_major" else [S * H, 1, H, 1]).set_ragged_offset(nodes["rs"])
    values["o"] = torch.full((CAP, H, d_v), 2048.0, device="cuda", dtype=torch.bfloat16)
    values["stats"] = torch.full((H, CAP) if layout == "head_major" else (CAP, H), 2048.0, device="cuda", dtype=torch.float32)
    nodes.update(o=o, stats=stats)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    index = next((i for i, name in enumerate(names) if name.split("[")[0] == "sdpa_fwd_prefill_sm100"), None)
    assert index is not None, names
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    knobs.update(
        {
            cudnn.knob_type.TILE_M: 128,
            cudnn.knob_type.TILE_N: 128,
            cudnn.knob_type.TILE_CGA_M: cta_mma,
            cudnn.knob_type.SPLIT_KV: 1,
            cudnn.knob_type.PACK_GQA: int(cta_mma == 2 and (d_qk, d_v) == (128, 128)),
        }
    )
    g.create_execution_plan(engine, knobs)
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    workspace = torch.empty(max(1, g.get_workspace_size()), device="cuda", dtype=torch.uint8)
    vp = {nodes[name]: tensor for name, tensor in values.items()}
    for name, heads in [("q", H), ("k", kv_heads), ("v", kv_heads), ("o", H)]:
        dim = values[name].shape[-1]
        vp[nodes[name]] = values[name].as_strided((B, heads, S, dim), (S * heads * dim, dim, heads * dim, 1))
    vp[stats] = values["stats"].view(-1)
    return g, vp, workspace


def _check(values, expected, layout):
    want_o, want_s = expected
    live = len(want_o)
    got_s = values["stats"].T if layout == "head_major" else values["stats"]
    for role, got, want, tol in [("o", values["o"][:live], want_o, 0.015), ("stats", got_s[:live], want_s, 1e-4)]:
        actual = got.double().cpu()
        torch.testing.assert_close(actual, want, atol=tol, rtol=tol)
        assert torch.equal(torch.isneginf(actual), torch.isneginf(want))
        finite = torch.isfinite(want)
        rms = ((actual[finite] - want[finite]).square().sum() / want[finite].square().sum().clamp_min(1e-20)).sqrt().item()
        assert rms < (0.02 if role == "o" else 1e-4)

    assert torch.equal(values["o"][live:], torch.full_like(values["o"][live:], 2048.0))
    assert torch.equal(got_s[live:], torch.full_like(got_s[live:], 2048.0))


@requires_dsl
@pytest.mark.parametrize(
    "bottom_right,kv_heads,layout,scale",
    [
        pytest.param(br, heads, layout, scale, marks=pytest.mark.L0 if (br, heads, layout, scale) == (False, 1, "token_major", 2.0) else pytest.mark.L1)
        for br, heads, layout, scale in itertools.product((False, True), (1, 4), ("token_major", "head_major"), (0.5, 2.0))
    ],
)
def test_causal_thd_interior_replay(bottom_right, kv_heads, layout, scale):
    _run_causal_thd_interior_replay(bottom_right, kv_heads, layout, scale)


def _run_causal_thd_interior_replay(bottom_right, kv_heads, layout, scale, d_qk=D, d_v=D):
    major, minor = torch.cuda.get_device_capability()
    if major != 10 or minor >= 7:
        pytest.skip("SM100-family D128 forward")
    values, host = _make_case(kv_heads, scale, 320, (1537, 513, 0), (1025, 1793, 0), d_qk, d_v)
    g, vp, workspace = _build_graph(values, kv_heads, bottom_right, layout)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.cuda.stream(stream):
            g.execute(vp, workspace)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.cuda.current_stream().wait_stream(stream)
    _check(values, _reference(host, bottom_right), layout)
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(captured, stream=stream):
            g.execute(vp, workspace)
        for name in ("sq", "sk", "rq", "rk", "rv", "ro", "rs"):
            values[name].zero_()
        for name in ("o", "stats"):
            values[name].fill_(2048.0)
        captured.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(values[n], torch.full_like(values[n], 2048.0)) for n in ("o", "stats"))
        fresh, host = _make_case(kv_heads, scale, 321, (769, 0, 1537), (1537, 0, 1025), d_qk, d_v)
        for name, tensor in fresh.items():
            values[name].copy_(tensor)
        values["rs"].copy_(values["rq"] // d_qk // H * (1 if layout == "head_major" else H))
        captured.replay()
        torch.cuda.synchronize()
        _check(values, _reference(host, bottom_right), layout)
    finally:
        captured.reset()


@requires_dsl
@pytest.mark.L1
def test_unsplit_nonpaged_causal_thd_declines_cga1():
    """This optimization preserves the explicit unsplit nonpaged THD contract."""
    major, minor = torch.cuda.get_device_capability()
    if major != 10 or minor >= 7:
        pytest.skip("SM100-family D128 forward")
    values, _ = _make_case(1, 0.5, 350, (1537, 513, 0), (1025, 1793, 0))
    with pytest.raises(NotImplementedError, match="other THD graphs run the cga2 prefill tile"):
        _build_graph(values, 1, False, "token_major", cta_mma=1)
    for name in ("o", "stats"):
        assert torch.equal(values[name], torch.full_like(values[name], 2048.0))


@requires_dsl
@pytest.mark.L1
@pytest.mark.parametrize(
    "d_qk,d_v,bottom_right,layout",
    [
        (72, 72, False, "token_major"),
        (120, 120, True, "head_major"),
        (96, 80, False, "head_major"),
        (80, 96, True, "token_major"),
        (128, 64, True, "head_major"),
        (64, 128, False, "token_major"),
    ],
)
def test_causal_thd_interior_head_envelope(d_qk, d_v, bottom_right, layout):
    """Real widths use the D128 tile with distinct Q/K versus V/O offsets.

    These public THD envelope plans use unpacked GQA; packed GQA is admitted
    only at the engine's explicit native shapes. No capability is expanded.
    """
    _run_causal_thd_interior_replay(bottom_right, 1, layout, 2.0, d_qk, d_v)
