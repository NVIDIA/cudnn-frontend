# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD work lookup across changing ragged metadata on a retained CUDA graph."""

import itertools

import pytest
import torch

from test_utils import torch_fork_set_rng
from frost_test_utils import requires_dsl


def _build_graph(q, k, v, cu, max_seqlen, stats_layout, pack, policy, monkeypatch, *, page_tables=None, kv_lengths=None):
    """Pin the bounded decoder with packed Q/O and either packed or paged K/V."""
    import cudnn
    import cudnn.sdpa.fwd.api_dsl as api_dsl

    batch, hq, hkv, dim = cu.numel() - 1, q.shape[1], k.shape[1], q.shape[2]
    capacity = q.shape[0]
    io = cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bindings, inputs, origins = {}, {}, []

    def ragged_tensor(name, buf, heads):
        tensor = g.tensor(name=name, dim=[batch, heads, max_seqlen, dim], stride=[max_seqlen * heads * dim, dim, heads * dim, 1], data_type=io)
        offset = (cu.to(torch.int64) * heads * dim).view(batch + 1, 1, 1, 1)
        ref = g.tensor_like(offset, name=name + "_origin")
        tensor.set_ragged_offset(ref)
        bindings.update({tensor: buf, ref: offset})
        origins.append((offset, heads * dim))
        return tensor

    inputs["q"] = ragged_tensor("q", q, hq)
    for name, buf in (("k", k), ("v", v)):
        if page_tables is None:
            inputs[name] = ragged_tensor(name, buf, hkv)
        else:
            inputs[name] = g.tensor_like(buf, name=name)
            bindings[inputs[name]] = buf
    cu4 = cu.view(batch + 1, 1, 1, 1)
    sq = g.tensor_like(cu4, name="cu_q")
    bindings[sq] = cu4
    if page_tables is None:
        sk = g.tensor_like(cu4, name="cu_kv")
        bindings[sk] = cu4
        kv_args = dict(cu_seq_len_kv=sk)
    else:
        sk = g.tensor_like(kv_lengths, name="seq_kv")
        bindings[sk] = kv_lengths
        tables = [g.tensor_like(buf, name=name) for name, buf in zip(("k_table", "v_table"), page_tables)]
        bindings.update(zip(tables, page_tables))
        kv_args = dict(
            seq_len_kv=sk,
            paged_attention_k_table=tables[0],
            paged_attention_v_table=tables[1],
            paged_attention_max_seq_len_kv=page_tables[0].shape[2] * k.shape[2],
        )
    o, stats = g.sdpa(
        **inputs,
        generate_stats=True,
        attn_scale=dim**-0.5,
        use_padding_mask=True,
        use_causal_mask=True,
        cu_seq_len_q=sq,
        **kv_args,
    )
    ob = torch.empty_like(q)
    token_major = stats_layout == "token"
    sb = torch.empty((capacity, hq) if token_major else (hq, capacity), device=q.device, dtype=torch.float32)
    for tensor, name, factor in ((o, "o_origin", hq * dim), (stats, "stats_origin", hq if token_major else 1)):
        offset = (cu.to(torch.int64) * factor).view(batch + 1, 1, 1, 1)
        ref = g.tensor_like(offset, name=name)
        tensor.set_ragged_offset(ref)
        origins.append((offset, factor))
        bindings[ref] = offset
    o.set_output(True).set_dim([batch, hq, max_seqlen, dim]).set_stride([max_seqlen * hq * dim, dim, hq * dim, 1])
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([batch, hq, max_seqlen, 1])
    stats.set_stride([max_seqlen * hq, 1, hq, 1] if token_major else [max_seqlen * hq, capacity, 1, 1])
    bindings.update({o: ob, stats: sb})
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    arch = "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"
    index = next((i for i, name in enumerate(names) if name.split("[")[0] == "sdpa_fwd_prefill_" + arch), None)
    assert index is not None, names  # A missing FROST route must not silently skip this test.
    engine, knobs = g.get_engine_and_knobs_at_index(index)
    knobs.update(
        {
            cudnn.knob_type.PACK_GQA: pack,
            cudnn.knob_type.SCHED_POLICY: policy,
            cudnn.knob_type.TILE_M: 128,
            cudnn.knob_type.TILE_N: 128,
            cudnn.knob_type.TILE_CGA_M: 2,
            cudnn.knob_type.SPLIT_KV: 1,
        }
    )
    g.create_execution_plan(engine, knobs)
    g.select_plan(len(g.plans) - 1)
    loaded = []
    load_module = api_dsl._load_sm100_kernel_module

    def record_module(*args, **kwargs):
        module = load_module(*args, **kwargs)
        loaded.append(module)
        return module

    with monkeypatch.context() as patch:
        patch.setattr(api_dsl, "_load_sm100_kernel_module", record_module)
        g.build_plans()
    assert loaded and all(m._BOUNDED_THD_DECODE for m in loaded)
    if page_tables is not None:
        assert all(m.CFG.PAGED_KV for m in loaded), "The paged bounded decoder must be exercised"
    workspace = torch.empty(max(1, g.get_workspace_size()), device=q.device, dtype=torch.uint8)

    def execute():
        g.execute(bindings, workspace)

    def set_lengths(lengths):
        cu.copy_(torch.tensor([0, *itertools.accumulate(lengths)], device=q.device, dtype=torch.int32))
        for offset, factor in origins:
            offset.copy_((cu.to(torch.int64) * factor).view(batch + 1, 1, 1, 1))

    return execute, set_lengths, (ob, sb if token_major else sb.transpose(0, 1))


def _reference(q, k, v, lengths):
    """Independent CPU FP64 attention, including grouped heads and natural-log LSE."""
    outputs, stats, start = [], [], 0
    for length in lengths:
        if length == 0:
            continue
        qq = q[start : start + length].cpu().double().transpose(0, 1)
        kk, vv = (x[start : start + length].cpu().double().transpose(0, 1).repeat_interleave(q.shape[1] // k.shape[1], 0) for x in (k, v))
        score = (qq @ kk.transpose(-1, -2)) * q.shape[2] ** -0.5
        score.masked_fill_(torch.arange(length)[None, :] > torch.arange(length)[:, None], -torch.inf)
        outputs.append((score.softmax(-1) @ vv).transpose(0, 1))
        stats.append(score.logsumexp(-1).transpose(0, 1))
        start += length
    return torch.cat(outputs), torch.cat(stats)


@requires_dsl
@pytest.mark.parametrize(
    "heads,pack,policy,stats_layout",
    [
        pytest.param(heads, pack, policy, layout, marks=pytest.mark.L0 if (heads, pack, policy, layout) == ((4, 1), 1, 1, "token") else pytest.mark.L1)
        for heads, pack, policy, layout in itertools.product(((4, 1), (8, 8), (32, 8)), (0, 1), (0, 1, 2), ("token", "head"))
    ],
)
@torch_fork_set_rng(seed=13)
def test_thd_work_lookup_replay(heads, pack, policy, stats_layout, monkeypatch):
    major, minor = torch.cuda.get_device_capability()
    if major != 10 or minor >= 7:
        pytest.skip("SM100-family forward kernel")
    monkeypatch.setenv("FROST_THD_CLUSTERS", "4")
    hq, hkv = heads
    batch, max_seqlen, dim = 5, 2049, 128
    q, k, v = (torch.randn(batch * max_seqlen, h, dim, device="cuda", dtype=torch.bfloat16) * 0.5 for h in (hq, hkv, hkv))
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * max_seqlen
    execute, set_lengths, actual = _build_graph(q, k, v, cu, max_seqlen, stats_layout, pack, policy, monkeypatch)
    # Force several persistent claims per cluster, including dead overlaunch.
    for _ in range(3):
        execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            execute()
        length_sets = [
            [0, 1, 511, 1024, 513],
            [768, 0, 3, 1025, 253],
            [513, 560, 575, 520, 544],  # Unequal lengths, equal rounded tile counts.
            [572, 515, 550, 565, 519],  # Different longest-first permutation.
            [0, 0, 0, 0, 0],
            [513, 560, 575, 520, 544],  # Live again without recapture.
        ]
        for iteration, lengths in enumerate(length_sets):
            set_lengths(lengths)
            v.mul_(-1)  # Replayed outputs cannot be inherited from an earlier iteration.
            for buf in actual:
                buf.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            total = sum(lengths)
            for buf in actual:
                assert torch.isnan(buf[total:]).all(), (iteration, "dead rows were written")
            if total:
                reference = _reference(q, k, v, lengths)
                for role, got, want in zip(("o", "stats"), actual, reference):
                    tolerance = 0.01 if role == "o" else 1e-4
                    torch.testing.assert_close(got[:total].cpu().double(), want, atol=tolerance, rtol=tolerance)
        previous_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            execute()
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
        torch.cuda.synchronize()
    finally:
        graph.reset()


def _paged_reference(q, k, v, page_tables, q_lengths, kv_lengths):
    """CPU FP64 top-left causal attention over independently gathered K/V pages."""
    q, k, v = (buf.cpu().double() for buf in (q, k, v))
    kt, vt = (table.cpu().view(len(q_lengths), -1).long() for table in page_tables)
    hq, hkv, dim = q.shape[1], k.shape[1], q.shape[2]
    out = torch.zeros(sum(q_lengths), hq, dim, dtype=torch.float64)
    stats = torch.full((sum(q_lengths), hq), -torch.inf, dtype=torch.float64)
    start = 0
    for i, (nq, nk) in enumerate(zip(q_lengths, kv_lengths)):
        if nq and nk:
            kk, vv = (pool[table[i]].transpose(1, 2).reshape(-1, hkv, dim)[:nk].repeat_interleave(hq // hkv, 1) for pool, table in ((k, kt), (v, vt)))
            qq = q[start : start + nq].transpose(0, 1)
            score = (qq @ kk.transpose(0, 1).transpose(-1, -2)) * dim**-0.5
            score.masked_fill_(torch.arange(nk)[None, :] > torch.arange(nq)[:, None], -torch.inf)
            out[start : start + nq] = (score.softmax(-1) @ vv.transpose(0, 1)).transpose(0, 1)
            stats[start : start + nq] = score.logsumexp(-1).transpose(0, 1)
        start += nq
    return out, stats


@requires_dsl
@pytest.mark.parametrize("hnd", [False, True], ids=["NHD", "HND"])
@pytest.mark.parametrize(
    "policy,stats_layout",
    [
        pytest.param(0, "token", marks=pytest.mark.L0),
        pytest.param(1, "head", marks=pytest.mark.L1),
        pytest.param(2, "token", marks=pytest.mark.L1),
    ],
)
@torch_fork_set_rng(seed=17)
def test_thd_work_lookup_paged_replay(hnd, policy, stats_layout, monkeypatch):
    """BF16 causal paged THD with unequal Q/KV lengths and partial PackGQA."""
    from test_sdpa_fwd_paged_sm100 import _pools

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Shared paged D128 kernel on SM100, SM103 and SM107")
    if torch.cuda.get_device_capability() == (10, 7) and policy == 2:
        pytest.skip("The SM107 graph row does not offer LPT_L2 for BF16")
    monkeypatch.setenv("FROST_THD_CLUSTERS", "4")
    batch, max_seqlen, hq, hkv, dim = 3, 513, 12, 2, 128
    # Six query heads per KV head leave a partial group in the packing tile.
    rng = torch.Generator(device="cuda").manual_seed(17)
    _, _, k, v, table = _pools(batch, hkv, dim, 16, 40, hnd, torch.bfloat16, generator=rng)
    k.mul_(0.5)
    v.mul_(0.5)
    page_tables = (table, table.flip(2))
    q = torch.randn(batch * max_seqlen, hq, dim, device="cuda", dtype=torch.bfloat16, generator=rng) * 0.5
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * max_seqlen
    kv_lengths = torch.full((batch, 1, 1, 1), 640, device="cuda", dtype=torch.int32)
    execute, set_lengths, actual = _build_graph(q, k, v, cu, max_seqlen, stats_layout, 1, policy, monkeypatch, page_tables=page_tables, kv_lengths=kv_lengths)
    for _ in range(3):
        execute()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            execute()
        for ql, kl in (
            ([0, 129, 513], [640, 63, 257]),
            ([257, 0, 1], [513, 0, 0]),
            ([129, 140, 151], [211, 99, 640]),  # Equal rounded Q tile counts.
            ([150, 130, 141], [79, 512, 199]),  # New longest-first remap.
            ([0, 0, 0], [0, 640, 1]),
            ([129, 140, 151], [211, 99, 640]),
        ):
            set_lengths(ql)
            kv_lengths.copy_(torch.tensor(kl, device="cuda", dtype=torch.int32).view_as(kv_lengths))
            q.mul_(-1)
            v.mul_(-1)
            for table in page_tables:
                table.copy_(table.roll(1, dims=2))
            for buf in actual:
                buf.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            total = sum(ql)
            for buf in actual:
                assert torch.isnan(buf[total:]).all(), "Dead packed rows were written"
            if total:
                for role, got, want in zip(("o", "stats"), actual, _paged_reference(q, k, v, page_tables, ql, kl)):
                    tolerance = 0.01 if role == "o" else 1e-4
                    torch.testing.assert_close(got[:total].cpu().double(), want, atol=tolerance, rtol=tolerance)
        previous_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            execute()
        finally:
            torch.cuda.set_sync_debug_mode(previous_mode)
        torch.cuda.synchronize()
    finally:
        graph.reset()
