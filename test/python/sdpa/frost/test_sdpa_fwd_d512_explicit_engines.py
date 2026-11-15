# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wide half explicit records retain the body and execute through the shared host."""

import math

import pytest
import torch

import cudnn
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "engine_id,dq,tile,split,template,cluster",
    [
        (20521, 512, 128, 1, "prefill_d512_f16", 4),
        (20521, 512, 128, 4, "prefill_d512_f16", 4),
        (20522, 512, 64, 1, "prefill_d512_f16_2x2", 4),
        (20522, 576, 64, 1, "prefill_d512_f16_2x2", 2),
    ],
    ids=["d512-role", "d512-role-split4", "d512-2x2", "d576-2x2"],
)
def test_public_record_rebuild_and_retained_capture(monkeypatch, dtype, engine_id, dq, tile, split, template, cluster):
    from cudnn.sdpa.fwd import api_dsl
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    # A legacy standalone A/B switch cannot change an explicit graph record.
    monkeypatch.setattr(api_dsl, "D512_2X2", engine_id != 20522)
    torch.manual_seed(47)
    b, h, sq, sk = 1, 4, 33, 256
    q = torch.randn(b, sq, h, dq, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(b, sk, h, dq, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(b, sk, h, 512, device="cuda", dtype=dtype).transpose(1, 2)
    o = torch.empty(b, sq, h, 512, device="cuda", dtype=dtype).transpose(1, 2)
    stats = torch.empty(b, h, sq, 1, device="cuda", dtype=torch.float32)
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    public = SdpaFwdKnobs(tile_m=tile, tile_n=128, cga=cluster, sched_policy=0, pack_gqa=False, split_kv=split).to_public()

    def build(record):
        g = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        tq, tk, tv = (g.tensor_like(t) for t in (q, k, v))
        to, ts = g.sdpa(q=tq, k=tk, v=tv, generate_stats=True, attn_scale=1.0 / math.sqrt(dq))
        to.set_output(True).set_dim(o.shape).set_stride(o.stride())
        ts.set_output(True).set_dim(stats.shape).set_stride(stats.stride()).set_data_type(cudnn.data_type.FLOAT)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plan(*record)
        g.select_plan(0)
        g.check_support()
        g.build_plans()
        compiled = g._compiled_plans[0]._compiled
        assert compiled.kernel_template == template
        metadata = compiled.implementation_config
        assert (metadata["query_tile"], metadata["cga"], metadata["cluster_m"], metadata["split_kv"]) == (tile, cluster, cluster, split)
        assert metadata["cta_group"] == 2
        return g, {tq: q, tk: k, tv: v, to: o, ts: stats}, torch.empty(max(1, g.get_workspace_size()), device="cuda", dtype=torch.uint8)

    g, vp, workspace = build((engine_id, public))
    g.execute(vp, workspace)
    torch.cuda.synchronize()
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(dq)
    torch.testing.assert_close(o.float(), scores.softmax(-1) @ v.float(), atol=3e-2 if dtype == torch.bfloat16 else 1e-2, rtol=2e-2)
    torch.testing.assert_close(stats.squeeze(-1), scores.logsumexp(-1), atol=5e-3, rtol=0)
    expected_o, expected_stats = o.clone(), stats.clone()
    record = g.get_engine_and_knobs_at_index(0)
    assert record == (engine_id, public)

    handle = cudnn.create_handle()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle, stream.cuda_stream)
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            g.execute(vp, workspace, handle=handle)
        stream.synchronize()
        with torch.cuda.graph(captured, stream=stream):
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                g.execute(vp, workspace, handle=handle)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
        # Build a second graph solely from the public record while the first
        # graph's capture remains alive; the old capture keeps its own metadata.
        rebuilt, bindings, scratch = build(record)
        rebuilt.execute(bindings, scratch)
        torch.cuda.synchronize()
        torch.testing.assert_close(o, expected_o, atol=0, rtol=0)
        v.mul_(0.5)
        o.fill_(float("nan"))
        stats.fill_(float("nan"))
        captured.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(o, expected_o * 0.5, atol=2e-3, rtol=5e-3)
        torch.testing.assert_close(stats, expected_stats, atol=0, rtol=0)
    finally:
        torch.cuda.synchronize()
        cudnn.destroy_handle(handle)
