# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM80 packed-THD forward through the graph API (issue #377)."""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from cudnn.sdpa import graph_analyzer as ga
from cudnn.sdpa.fwd import engines as fwd_engines
from frost_test_utils import select_engine

_SM80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="needs an SM80 (A100) GPU",
)
pytestmark = [pytest.mark.L0, _SM80]
_ENGINE = "sdpa_fwd_prefill_sm80"


def _make_graph(
    lens_q,
    lens_kv,
    *,
    h=2,
    hkv=None,
    d=128,
    dv=None,
    dtype=torch.float16,
    causal=False,
    bottom_right=False,
    cu_lens=False,
    stats_layout="head_major",
    cap_extra=0,
    stats_output=True,
    stats_log2=False,
    sdpa_kwargs=None,
    declare_total=True,
    skv_max=None,
    sq_max=None,
):
    b = len(lens_q)
    assert b == len(lens_kv)
    hkv = h if hkv is None else hkv
    dv = d if dv is None else dv
    dev = "cuda"
    sq, skv = max(lens_q), max(lens_kv)
    sq = sq if sq_max is None else sq_max
    skv = skv if skv_max is None else skv_max
    tq, tkv = sum(lens_q), sum(lens_kv)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    vp = {}
    cu_q = [0]
    cu_kv = [0]
    for nq, nkv in zip(lens_q, lens_kv, strict=True):
        cu_q.append(cu_q[-1] + nq)
        cu_kv.append(cu_kv[-1] + nkv)

    def port(name, smax, total, cu, nh, dd):
        stride = (smax * nh * dd, dd, nh * dd, 1)
        node = graph.tensor(name=name, dim=(b, nh, smax, dd), stride=stride, data_type=io)
        offset = graph.tensor(name=f"{name}_ro", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        node.set_ragged_offset(offset)
        vp[offset] = (torch.tensor(cu, dtype=torch.int64, device=dev) * nh * dd).view(b + 1, 1, 1, 1)
        vp[node] = torch.randn(1, total + cap_extra, nh, dd, dtype=dtype, device=dev)
        if cap_extra:
            vp[node][0, total:] = float("nan")
        return node

    q = port("q", sq, tq, cu_q, h, d)
    q_ro_buf = vp[next(node for node in vp if node.get_name() == "q_ro")]
    k = port("k", skv, tkv, cu_kv, hkv, d)
    v = port("v", skv, tkv, cu_kv, hkv, dv)
    q_is_cu, kv_is_cu = (cu_lens, cu_lens) if isinstance(cu_lens, bool) else cu_lens
    nq_lens, nkv_lens = b + int(q_is_cu), b + int(kv_is_cu)
    q_len = graph.tensor(name="cu_seq_len_q" if q_is_cu else "seq_len_q", dim=(nq_lens, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    kv_len = graph.tensor(name="cu_seq_len_kv" if kv_is_cu else "seq_len_kv", dim=(nkv_lens, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    vp[q_len] = torch.tensor(cu_q if q_is_cu else lens_q, dtype=torch.int32, device=dev).view(nq_lens, 1, 1, 1)
    vp[kv_len] = torch.tensor(cu_kv if kv_is_cu else lens_kv, dtype=torch.int32, device=dev).view(nkv_lens, 1, 1, 1)
    kw = dict(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        attn_scale=1 / math.sqrt(d),
        generate_stats=stats_output,
        use_padding_mask=True,
        use_causal_mask=causal,
        use_causal_mask_bottom_right=bottom_right,
        stats_use_log2=stats_log2,
    )
    if declare_total:
        kw.update(max_total_seq_len_q=tq, max_total_seq_len_kv=tkv)
    kw.update({("cu_seq_len_q" if q_is_cu else "seq_len_q"): q_len, ("cu_seq_len_kv" if kv_is_cu else "seq_len_kv"): kv_len})
    kw.update(sdpa_kwargs or {})
    o, stats = graph.sdpa(**kw)
    o.set_output(True).set_data_type(io).set_dim((b, h, sq, dv)).set_stride((sq * h * dv, dv, h * dv, 1))
    o_ro = graph.tensor(name="o_ro", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
    o.set_ragged_offset(o_ro)
    vp[o_ro] = q_ro_buf
    vp[o] = torch.full((1, tq + cap_extra, h, dv), float("nan"), dtype=dtype, device=dev)

    if stats_output:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        if stats_layout in ("head_major", "head_major_padded"):
            head_stride = tq if stats_layout == "head_major" else ((tq + 63) // 64) * 64
            stats.set_dim((b, h, sq, 1)).set_stride((h * head_stride, head_stride, 1, 1))
        else:
            assert stats_layout == "token_major"
            stats.set_dim((b, h, sq, 1)).set_stride((sq * h, 1, h, 1))
        stats_ro = graph.tensor(name="stats_ro", dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        stats.set_ragged_offset(stats_ro)
        vp[stats_ro] = (torch.tensor(cu_q, dtype=torch.int64, device=dev) * (h if stats_layout == "token_major" else 1)).view(b + 1, 1, 1, 1)
        vp[stats] = torch.full((tq, h) if stats_layout == "token_major" else (1, h, head_stride), float("nan"), dtype=torch.float32, device=dev)
    return graph, vp, (q, k, v, o, stats), (cu_q, cu_kv)


def _plan(graph):
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, _ENGINE)
    graph.check_support()
    graph.build_plans()
    return torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize(
    "dtype,hkv,d,dv,causal,bottom_right,cu_lens,stats_layout,stats_output,stats_log2",
    [
        (torch.float16, 2, 128, 128, False, False, False, "head_major", True, False),
        (torch.bfloat16, 1, 128, 128, True, False, False, "head_major", True, False),
        (torch.float16, 2, 64, 64, False, False, False, "head_major", True, False),
        (torch.bfloat16, 1, 192, 128, False, False, False, "head_major", True, False),
        (torch.float16, 2, 256, 256, True, False, False, "head_major", True, False),
        (torch.float16, 2, 128, 128, False, False, True, "head_major", True, False),
        (torch.float16, 2, 128, 128, False, False, (True, False), "head_major", True, False),
        (torch.float16, 2, 128, 128, False, False, (False, True), "head_major", True, False),
        (torch.float16, 2, 64, 64, False, False, True, "head_major", True, False),
        (torch.bfloat16, 1, 192, 128, False, False, True, "head_major", True, False),
        (torch.float16, 2, 256, 256, False, False, True, "head_major", True, False),
        (torch.float16, 2, 128, 128, False, False, False, "token_major", True, False),
        (torch.float16, 2, 128, 128, False, False, False, "head_major_padded", True, False),
        (torch.bfloat16, 2, 128, 128, False, True, False, "head_major", True, True),
        (torch.float16, 2, 128, 128, False, False, False, "head_major", False, False),
    ],
    ids=(
        "d128_fp16",
        "d128_bf16_gqa_causal",
        "d64_fp16",
        "d192x128_bf16_gqa",
        "d256_fp16_causal",
        "d128_cu_lens",
        "d128_cu_q_only",
        "d128_cu_kv_only",
        "d64_cu_lens",
        "d192x128_cu_lens",
        "d256_cu_lens",
        "d128_token_major_stats",
        "d128_padded_head_stride",
        "d128_bottom_right_log2",
        "d128_no_stats",
    ),
)
def test_graph_thd_forward_unequal_lengths(dtype, hkv, d, dv, causal, bottom_right, cu_lens, stats_layout, stats_output, stats_log2):
    torch.manual_seed(7)
    graph, vp, nodes, (cu_q, cu_kv) = _make_graph(
        (144, 96),
        (160, 128),
        hkv=hkv,
        d=d,
        dv=dv,
        dtype=dtype,
        causal=causal,
        bottom_right=bottom_right,
        cu_lens=cu_lens,
        stats_layout=stats_layout,
        stats_output=stats_output,
        stats_log2=stats_log2,
    )
    q, k, v, o, stats = nodes
    facts = ga.analyze(graph)
    assert facts is not None and facts.thd and facts.packed_layout
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    assert fwd_engines.mismatch(caps, facts) is None

    workspace = _plan(graph)
    graph.execute(vp, workspace)
    torch.cuda.synchronize()

    scale = 1 / math.sqrt(d)
    for batch in range(2):
        q_slice = vp[q][0, cu_q[batch] : cu_q[batch + 1]].transpose(0, 1).double()
        k_slice = vp[k][0, cu_kv[batch] : cu_kv[batch + 1]].transpose(0, 1).double()
        v_slice = vp[v][0, cu_kv[batch] : cu_kv[batch + 1]].transpose(0, 1).double()
        if hkv != 2:
            k_slice = k_slice.repeat_interleave(2 // hkv, dim=0)
            v_slice = v_slice.repeat_interleave(2 // hkv, dim=0)
        scores = (q_slice @ k_slice.transpose(-1, -2)) * scale
        if causal:
            masked = torch.ones(scores.shape[-2:], dtype=torch.bool, device="cuda").triu(1)
            scores = scores.masked_fill(masked, float("-inf"))
        if bottom_right:
            nq, nkv = scores.shape[-2:]
            qi = torch.arange(nq, device="cuda").view(nq, 1)
            ki = torch.arange(nkv, device="cuda").view(1, nkv)
            scores = scores.masked_fill(ki > qi + (nkv - nq), float("-inf"))
        ref_o = (scores.softmax(-1) @ v_slice).transpose(0, 1)
        ref_lse = scores.logsumexp(-1) * (math.log2(math.e) if stats_log2 else 1)
        got_o = vp[o][0, cu_q[batch] : cu_q[batch + 1]]
        torch.testing.assert_close(got_o.float(), ref_o.float(), rtol=1e-2, atol=4e-3)
        if stats_output:
            got_lse = vp[stats][cu_q[batch] : cu_q[batch + 1]].T if stats_layout == "token_major" else vp[stats][0, :, cu_q[batch] : cu_q[batch + 1]]
            torch.testing.assert_close(got_lse, ref_lse.float(), rtol=3e-2, atol=5e-2)


def test_graph_thd_zero_kv_and_poisoned_tail():
    """A keyless sequence is zero/-inf; unused capacity must not contaminate its neighbor."""
    torch.manual_seed(11)
    graph, vp, nodes, (cu_q, cu_kv) = _make_graph((96, 144), (0, 160), cap_extra=32)
    q, k, v, o, stats = nodes
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()

    assert torch.count_nonzero(vp[o][0, : cu_q[1]]) == 0
    assert torch.isneginf(vp[stats][0, :, : cu_q[1]]).all()
    assert torch.isnan(vp[o][0, cu_q[-1] :]).all()
    q1 = vp[q][0, cu_q[1] : cu_q[2]].transpose(0, 1).double()
    k1 = vp[k][0, cu_kv[1] : cu_kv[2]].transpose(0, 1).double()
    v1 = vp[v][0, cu_kv[1] : cu_kv[2]].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(vp[o][0, cu_q[1] : cu_q[2]].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)
    torch.testing.assert_close(vp[stats][0, :, cu_q[1] : cu_q[2]], scores.logsumexp(-1).float(), rtol=3e-2, atol=5e-2)


def test_graph_thd_all_kv_empty():
    graph, vp, nodes, _ = _make_graph((96, 144), (0, 0), skv_max=128)
    facts = ga.analyze(graph)
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    assert fwd_engines.mismatch(caps, facts) is None
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()
    assert torch.count_nonzero(vp[nodes[3]]) == 0
    assert torch.isneginf(vp[nodes[4]]).all()


def test_graph_thd_all_q_empty():
    graph, vp, nodes, _ = _make_graph((0, 0), (96, 144), sq_max=128)
    facts = ga.analyze(graph)
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    assert fwd_engines.mismatch(caps, facts) is None
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()
    assert vp[nodes[3]].numel() == 0
    assert vp[nodes[4]].numel() == 0


def test_graph_thd_execute_does_not_allocate_or_sync():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128))
    workspace = _plan(graph)
    graph.execute(vp, workspace)
    torch.cuda.synchronize()
    ref = vp[nodes[3]].clone()

    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(3):
        graph.execute(vp, workspace)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert after == before, f"THD graph execute allocated {after - before} CUDA blocks"
    torch.testing.assert_close(vp[nodes[3]], ref, rtol=0, atol=0)

    prior_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            torch.zeros(1, device="cuda").item()
        graph.execute(vp, workspace)
    finally:
        torch.cuda.set_sync_debug_mode(prior_mode)
    torch.cuda.synchronize()


def test_graph_thd_compile_key_ignores_packed_totals():
    from cudnn.frost import template_loader

    def cache_totals():
        modules = [mod for (path, params), mod in template_loader._MODULES.items() if "sm80/prefill" in str(path) and getattr(params, "thd_varlen", False)]
        infos = [mod.compile.cache_info() for mod in modules]
        return sum(info.misses for info in infos), sum(info.hits for info in infos)

    first, vp1, _, _ = _make_graph((144, 96), (160, 128), stats_layout="token_major")
    first.execute(vp1, _plan(first))
    torch.cuda.synchronize()
    misses_0, hits_0 = cache_totals()
    second, vp2, _, _ = _make_graph((144, 64), (160, 64), stats_layout="token_major")
    second.execute(vp2, _plan(second))
    torch.cuda.synchronize()
    misses_1, hits_1 = cache_totals()
    assert misses_0 > 0
    assert misses_1 == misses_0, "a different live token total minted a new compiled SM80 artifact"
    assert hits_1 > hits_0, "same plan-time geometry should hit the SM80 compile cache"


def test_graph_thd_rebinds_smaller_live_total_on_one_plan():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), stats_layout="head_major_padded")
    workspace = _plan(graph)
    graph.execute(vp, workspace)
    torch.cuda.synchronize()

    def buffer(name):
        return vp[next(node for node in vp if node.get_name() == name)]

    buffer("seq_len_q").view(-1)[1] = 64
    buffer("seq_len_kv").view(-1)[1] = 64
    for name in ("q_ro", "o_ro"):
        buffer(name).view(-1)[-1] = 208 * 2 * 128
    for name in ("k_ro", "v_ro"):
        buffer(name).view(-1)[-1] = 224 * 2 * 128
    buffer("stats_ro").view(-1)[-1] = 208
    out, stats = vp[nodes[3]], vp[nodes[4]]
    out.fill_(float("nan"))
    stats.fill_(float("nan"))
    graph.execute(vp, workspace)
    torch.cuda.synchronize()

    q1 = vp[nodes[0]][0, 144:208].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:224].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:224].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(out[0, 144:208].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)
    torch.testing.assert_close(stats[0, :, 144:208], scores.logsumexp(-1).float(), rtol=3e-2, atol=5e-2)
    assert torch.isnan(out[0, 208:]).all()
    assert torch.isnan(stats[0, :, 208:240]).all()


def test_graph_thd_head_major_stats_without_declared_total():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), declare_total=False)
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()
    q1 = vp[nodes[0]][0, 144:240].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:288].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:288].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(vp[nodes[3]][0, 144:240].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)


def test_graph_thd_head_major_stats_bounds_overallocated_q_without_declared_total():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), declare_total=False)
    q, _, _, o, _ = nodes
    vp[q] = torch.cat((vp[q], torch.full((1, 32, 2, 128), float("nan"), dtype=vp[q].dtype, device="cuda")), dim=1)
    vp[o] = torch.full((1, 272, 2, 128), float("nan"), dtype=vp[o].dtype, device="cuda")
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()
    q1 = vp[nodes[0]][0, 144:240].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:288].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:288].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(vp[o][0, 144:240].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)
    assert torch.isnan(vp[o][0, 240:]).all()


def test_graph_thd_head_major_stats_accepts_strided_view():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), stats_layout="head_major_padded")
    vp[nodes[4]] = vp[nodes[4]][:, :, :240]
    assert vp[nodes[4]].stride()[1] > vp[nodes[4]].shape[2]
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()
    q1 = vp[nodes[0]][0, 144:240].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:288].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:288].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(vp[nodes[4]][0, :, 144:240], scores.logsumexp(-1).float(), rtol=3e-2, atol=5e-2)


def test_graph_thd_token_major_stats_rebinds_shorter_buffers():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), stats_layout="token_major")
    workspace = _plan(graph)
    graph.execute(vp, workspace)
    torch.cuda.synchronize()
    for node in nodes[:4]:
        new_total = 208 if node is nodes[0] or node is nodes[3] else 224
        vp[node] = vp[node][:, :new_total]
    for node, tensor in vp.items():
        name = node.get_name()
        if name in ("q_ro", "o_ro"):
            tensor.view(-1)[-1] = 208 * 2 * 128
        elif name in ("k_ro", "v_ro"):
            tensor.view(-1)[-1] = 224 * 2 * 128
        elif name == "stats_ro":
            tensor.view(-1)[-1] = 208 * 2
        elif name in ("seq_len_q", "seq_len_kv"):
            tensor.view(-1)[1] = 64
    vp[nodes[3]].fill_(float("nan"))
    vp[nodes[4]].fill_(float("nan"))
    graph.execute(vp, workspace)
    torch.cuda.synchronize()
    q1 = vp[nodes[0]][0, 144:208].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:224].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:224].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(vp[nodes[3]][0, 144:208].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)
    torch.testing.assert_close(vp[nodes[4]][144:208].T, scores.logsumexp(-1).float(), rtol=3e-2, atol=5e-2)


def test_graph_thd_cumulative_lengths_normalize_nonzero_base():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), cu_lens=True)
    workspace = _plan(graph)
    for node, tensor in vp.items():
        if node.get_name() == "cu_seq_len_q":
            tensor.add_(11)
        elif node.get_name() == "cu_seq_len_kv":
            tensor.add_(23)
    graph.execute(vp, workspace)
    torch.cuda.synchronize()

    q1 = vp[nodes[0]][0, 144:240].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:288].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:288].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    torch.testing.assert_close(vp[nodes[3]][0, 144:240].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)
    torch.testing.assert_close(vp[nodes[4]][0, :, 144:240], scores.logsumexp(-1).float(), rtol=3e-2, atol=5e-2)


@pytest.mark.parametrize(
    "sdpa_kwargs",
    ({"use_causal_mask": True, "sliding_window_length": 64}, {"diagonal_band_right_bound": 16}),
    ids=("causal_left_window", "right_band"),
)
def test_graph_thd_mask_family(sdpa_kwargs):
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128), sdpa_kwargs=sdpa_kwargs)
    graph.execute(vp, _plan(graph))
    torch.cuda.synchronize()

    q1 = vp[nodes[0]][0, 144:240].transpose(0, 1).double()
    k1 = vp[nodes[1]][0, 160:288].transpose(0, 1).double()
    v1 = vp[nodes[2]][0, 160:288].transpose(0, 1).double()
    scores = (q1 @ k1.transpose(-1, -2)) / math.sqrt(128)
    qi = torch.arange(96, device="cuda").view(96, 1)
    ki = torch.arange(128, device="cuda").view(1, 128)
    right = sdpa_kwargs.get("diagonal_band_right_bound", 0)
    keep = ki <= qi + right
    if "sliding_window_length" in sdpa_kwargs:
        keep = keep & (ki > qi - sdpa_kwargs["sliding_window_length"])
    scores = scores.masked_fill(~keep, float("-inf"))
    torch.testing.assert_close(vp[nodes[3]][0, 144:240].float(), (scores.softmax(-1) @ v1).transpose(0, 1).float(), rtol=1e-2, atol=4e-3)
    torch.testing.assert_close(vp[nodes[4]][0, :, 144:240], scores.logsumexp(-1).float(), rtol=3e-2, atol=5e-2)


def test_graph_thd_respects_handle_stream_and_captures():
    graph, vp, nodes, _ = _make_graph((144, 96), (160, 128))
    workspace = _plan(graph)
    out = vp[nodes[3]]
    handle = cudnn.create_handle()
    graph.execute(vp, workspace, handle=handle)
    torch.cuda.synchronize()
    reference = out.clone()
    assert torch.isfinite(reference).all()

    side = torch.cuda.Stream()
    cudnn.set_stream(handle=handle, stream=side.cuda_stream)
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.execute(vp, workspace, handle=handle)
    side.synchronize()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)

    with torch.cuda.stream(side):
        for _ in range(3):
            graph.execute(vp, workspace, handle=handle)
    side.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured, stream=side):
        graph.execute(vp, workspace, handle=handle)
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    captured.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_graph_thd_declines_off_flavor_head_dim():
    graph, _, _, _ = _make_graph((144, 96), (160, 128), d=96)
    facts = ga.analyze(graph)
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    reason = fwd_engines.mismatch(caps, facts)
    assert reason is not None and "native-tile" in reason


@pytest.mark.parametrize("sched_policy", (fwd_engines.SCHED_LPT, fwd_engines.SCHED_LPT_L2))
def test_graph_thd_declines_linear_grid_schedulers(sched_policy):
    graph, _, _, _ = _make_graph((144, 96), (160, 128), causal=True)
    facts = ga.analyze(graph)
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    reason = fwd_engines.mismatch(caps, facts, fwd_engines.SdpaFwdKnobs(sched_policy=sched_policy))
    assert reason is not None and "natural scheduler" in reason


def test_graph_thd_declines_dense_padded_stats():
    graph, _, nodes, _ = _make_graph((144, 96), (160, 128))
    nodes[4].set_ragged_offset(None)
    facts = ga.analyze(graph)
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    reason = fwd_engines.mismatch(caps, facts)
    assert reason is not None and "ragged stats offsets" in reason


def test_graph_thd_declines_gapped_token_stride():
    graph, _, nodes, _ = _make_graph((144, 96), (160, 128))
    q = nodes[0]
    q.set_stride((144 * (2 * 128 + 8), 128, 2 * 128 + 8, 1))
    facts = ga.analyze(graph)
    caps = next(spec.capabilities for spec in fwd_engines.ENGINE_SPECS if spec.name == _ENGINE)
    reason = fwd_engines.mismatch(caps, facts)
    assert reason is not None and "compact packed BSHD strides" in reason


def test_graph_thd_rejects_undersized_workspace():
    graph, vp, _, _ = _make_graph((144, 96), (160, 128))
    workspace = _plan(graph)
    assert workspace.numel() > 1
    with pytest.raises(ValueError, match="workspace"):
        graph.execute(vp, torch.empty(1, dtype=torch.uint8, device="cuda"))
