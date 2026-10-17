# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD work lookup across changing ragged metadata on a retained CUDA graph."""

import itertools

import pytest
import torch

from test_utils import torch_fork_set_rng
from frost_test_utils import requires_dsl


def _build_graph(q, k, v, cu, max_seqlen, stats_layout, pack, policy, monkeypatch):
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

    for name, buf, heads in (("q", q, hq), ("k", k, hkv), ("v", v, hkv)):
        inputs[name] = ragged_tensor(name, buf, heads)
    cu4 = cu.view(batch + 1, 1, 1, 1)
    sq, sk = g.tensor_like(cu4, name="cu_q"), g.tensor_like(cu4, name="cu_kv")
    bindings.update({sq: cu4, sk: cu4})
    o, stats = g.sdpa(
        **inputs,
        generate_stats=True,
        attn_scale=dim**-0.5,
        use_padding_mask=True,
        use_causal_mask=True,
        cu_seq_len_q=sq,
        cu_seq_len_kv=sk,
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
    index = next((i for i, name in enumerate(names) if name.split("[")[0] == "sdpa_fwd_prefill_sm100"), None)
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
