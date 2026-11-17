# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent dense cumulative-length graph and numerical oracle helpers."""

from types import SimpleNamespace

import cudnn
import torch


def make_case(dims, dtype, forms, *, splits=1, pack_gqa=False, batch=3):
    d, dv = dims
    b, h, hk, sq, sk = batch, 4, 2, 17, 65
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    bufs, refs = {}, {}
    generator = torch.Generator().manual_seed(525)
    for role, heads, rows, width in (("q", h, sq, d), ("k", hk, sk, d), ("v", hk, sk, dv), ("o", h, sq, dv)):
        data = torch.randn((b, rows, heads, width), generator=generator, dtype=dtype) * 0.2
        bufs[role] = data.cuda().transpose(1, 2)
        if role != "o":
            refs[role] = graph.tensor_like(bufs[role], name=role)
    bufs["stats"] = torch.empty((b, h, sq, 1), device="cuda", dtype=torch.float32)
    optional = {}
    for side, cumulative in zip(("q", "kv"), forms, strict=True):
        name = ("cu_seq_len_" if cumulative else "seq_len_") + side
        buf = torch.zeros(b + int(cumulative), dtype=torch.int32, device="cuda")
        bufs[side + "_len"] = buf
        ref = graph.tensor(name=name, dim=(buf.numel(), 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        refs[side + "_len"] = ref
        optional[name] = ref
    if splits > 1:
        bufs["sink"] = torch.tensor([-2.0, -1.0, 1.0, 2.0], device="cuda")
        refs["sink"] = graph.tensor(name="sink", dim=(1, h, 1, 1), stride=(h, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
        optional["sink_token"] = refs["sink"]
    o, stats = graph.sdpa(q=refs["q"], k=refs["k"], v=refs["v"], attn_scale=d**-0.5, generate_stats=True, use_padding_mask=True, **optional)
    for role, ref, dt in (("o", o, io), ("stats", stats, cudnn.data_type.FLOAT)):
        ref.set_output(True).set_dim(bufs[role].shape).set_stride(bufs[role].stride()).set_data_type(dt)
        refs[role] = ref
    graph.validate()
    graph.build_operation_graph()
    engine = 20546 if d <= 192 and dv <= 128 else 20547 if d <= 256 and dv <= 256 else 20548
    knobs = {cudnn.knob_type.SCHED_POLICY: 0, cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.PACK_GQA: int(pack_gqa)}
    graph.create_execution_plan(engine, knobs)
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(0)[0] == engine
    assert graph._compiled_plans[graph._plan_index]._prepared is not None
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    return SimpleNamespace(graph=graph, bufs=bufs, refs=refs, forms=forms, workspace=workspace, pack={refs[n]: t for n, t in bufs.items()})


def set_lengths(case, qlens, klens):
    for side, values, cumulative in zip(("q", "kv"), (qlens, klens), case.forms, strict=True):
        tensor = torch.tensor(values, dtype=torch.int32)
        if cumulative:
            tensor = torch.cat((torch.zeros(1, dtype=torch.int32), tensor.cumsum(0, dtype=torch.int32)))
        case.bufs[side + "_len"].copy_(tensor)


def check_reference(case, qlens, klens):
    q, k, v, o = (case.bufs[n].cpu().double() for n in ("q", "k", "v", "o"))
    stats = case.bufs["stats"].cpu().double().squeeze(-1)
    for b, (nq, nk) in enumerate(zip(qlens, klens, strict=True)):
        expected = torch.zeros_like(o[b])
        expected_stats = torch.full_like(stats[b], -float("inf"))
        if nq:
            kk, vv = (x[b, :, :nk].repeat_interleave(q.shape[1] // k.shape[1], 0) for x in (k, v))
            scores = q[b, :, :nq] @ kk.transpose(-1, -2) / q.shape[-1] ** 0.5
            logsum = scores.logsumexp(-1)
            if "sink" in case.bufs:
                logsum = torch.logaddexp(logsum, case.bufs["sink"].cpu().double()[:, None])
            probs = torch.where(torch.isfinite(logsum[..., None]), torch.exp(scores - logsum[..., None]), 0.0)
            expected[:, :nq] = probs @ vv
            expected_stats[:, :nq] = logsum
        torch.testing.assert_close(o[b], expected, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(stats[b], expected_stats, atol=8e-4, rtol=8e-4)


def execute(case, handle=None):
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        case.graph.execute(case.pack, case.workspace, handle)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def snapshot(case):
    return tuple(case.bufs[n].detach().clone() for n in ("o", "stats"))


def same_bytes(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
