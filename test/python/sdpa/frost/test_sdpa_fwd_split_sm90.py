# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM90 split-KV partial initialization and prepared replay."""

import pytest
import torch
import cudnn
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native SM90")]


@pytest.mark.parametrize("dims", [(512, 512), (384, 320)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("stats_log2", [False, True])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_dense_split_partial_initialization(dims, dtype, stats_log2, pack_gqa, graph_mode, cudnn_handle):
    """Every scratch row is written before reduction, under replay and empty KV."""
    import math
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    d, dv = dims
    b, h, hk, sq, sk = 3, 8, 2, 5, 129
    gen = torch.Generator().manual_seed(321)
    qc = torch.randn((b, sq, h, d + 8), generator=gen, dtype=dtype)
    kc = torch.randn((b, hk, sk, d + 8), generator=gen, dtype=dtype)
    vc = torch.randn((b, sk, hk, dv + 8), generator=gen, dtype=dtype)
    q = qc.cuda()[..., :d].transpose(1, 2)
    k = kc.cuda()[..., :d]
    v = vc.cuda()[..., :dv].transpose(1, 2)
    ob = torch.full((b, sq, h, dv + 8), 77, dtype=dtype, device="cuda")
    o = ob[..., :dv].transpose(1, 2)
    lb = torch.full((b, h, sq * 2), 77.0, device="cuda")
    lse = lb[..., ::2] if stats_log2 else None
    sink = torch.tensor([-float("inf"), float("inf"), -3, 2, -float("inf"), float("inf"), -2, 3], device="cuda")
    qlens = torch.tensor([sq, 3, 0], dtype=torch.int32, device="cuda")
    klens = torch.tensor([sk, 0, 17], dtype=torch.int32, device="cuda")
    api = SdpaFwdDslSm90(
        q,
        k,
        v,
        o,
        sample_lse=lse,
        split_kv=17,
        pack_gqa=pack_gqa,
        sched_policy=1,
        has_sink=True,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
        is_causal=True,
        causal_bottom_right=True,
        window_size_left=31,
        stats_log2=stats_log2,
    )
    assert api.check_support()
    if graph_mode:
        from frost_test_utils import select_engine

        io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
        fe = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
        qi, ki, vi = (fe.tensor_like(t) for t in (q, k, v))
        qli, kli = (fe.tensor_like(t.view(b, 1, 1, 1)) for t in (qlens, klens))
        si = fe.tensor_like(sink.view(1, h, 1, 1))
        oi, li = fe.sdpa(
            q=qi,
            k=ki,
            v=vi,
            generate_stats=lse is not None,
            attn_scale=1 / math.sqrt(d),
            use_padding_mask=True,
            seq_len_q=qli,
            seq_len_kv=kli,
            sink_token=si,
            use_causal_mask_bottom_right=True,
            sliding_window_length=32,
            stats_use_log2=stats_log2,
        )
        oi.set_output(True).set_dim(o.shape).set_stride(o.stride())
        if li is not None:
            li.set_output(True).set_dim((*lse.shape, 1)).set_stride((*lse.stride(), 1)).set_data_type(cudnn.data_type.FLOAT)
        fe.validate()
        fe.build_operation_graph()
        fe.create_execution_plans([cudnn.heur_mode.A])
        selected = select_engine(fe, "sdpa_fwd_prefill_sm90")
        knobs = {cudnn.knob_type.SPLIT_KV: 17, cudnn.knob_type.SCHED_POLICY: 1, cudnn.knob_type.PACK_GQA: int(pack_gqa)}
        fe.create_execution_plan(selected.engine_id, knobs)
        fe.select_plan(len(fe.plans) - 1)
        fe.check_support()
        fe.build_plans()
        assert fe.get_engine_and_knobs_at_index(fe._plan_index)[1] == knobs
        vp = dict(zip((qi, ki, vi, oi, qli, kli, si), (q, k, v, o, qlens, klens, sink)))
        if li is not None:
            vp[li] = lse
        ws = torch.empty(fe.get_workspace_size(), dtype=torch.uint8, device="cuda")

        def execute(workspace=ws):
            saved = cudnn.get_stream(cudnn_handle)
            try:
                cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
                fe.execute(vp, workspace, handle=cudnn_handle)
            finally:
                cudnn.set_stream(cudnn_handle, saved)

    else:
        api.compile()
        ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

        def execute(workspace=ws):
            api.execute(q, k, v, o, lse, sinks=sink, seq_q_lens=qlens, seq_kv_lens=klens, workspace=workspace)

    # Rejections precede all scratch or output writes.
    if not graph_mode and dims == (512, 512) and dtype == torch.bfloat16 and stats_log2:
        for bad in (None, ws[:-1], torch.empty(ws.numel() + 16, dtype=torch.uint8, device="cuda")[1 : ws.numel() + 1], torch.empty_like(ws, device="cpu")):
            with pytest.raises(ValueError, match="workspace"):
                execute(bad)
            assert torch.all(ob == 77) and torch.all(lb == 77)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            execute()
            with torch.cuda.graph(graph, stream=stream):
                execute()
        torch.cuda.current_stream().wait_stream(stream)
        baseline = None
        for poison in (0, 255, 127):
            ws.fill_(poison)
            ob.fill_(77)
            lb.fill_(77)
            graph.replay()
            torch.cuda.synchronize()
            actual = (o.cpu(), None if lse is None else lse.cpu())
            if baseline is not None:
                assert torch.equal(actual[0].view(torch.uint8), baseline[0].view(torch.uint8))
                if lse is not None:
                    assert torch.equal(actual[1].view(torch.uint8), baseline[1].view(torch.uint8))
            baseline = actual
            for batch, (nq, nk) in enumerate(zip((sq, 3, 0), (sk, 0, 17))):
                qq = qc[batch, :nq, :, :d].transpose(0, 1).double()
                kk = kc[batch, :, :nk, :d].double().repeat_interleave(h // hk, 0)
                vv = vc[batch, :nk, :, :dv].transpose(0, 1).double().repeat_interleave(h // hk, 0)
                scores = qq @ kk.transpose(-1, -2) / math.sqrt(d)
                qp = torch.arange(nq)[:, None] + nk - nq
                kp = torch.arange(nk)[None, :]
                scores.masked_fill_((kp > qp) | (kp < qp - 31), -float("inf"))
                logsum = torch.logaddexp(scores.logsumexp(-1), sink.cpu().double()[:, None])
                weights = torch.where(torch.isfinite(logsum[..., None]), torch.exp(scores - logsum[..., None]), 0.0)
                ref = weights @ vv
                torch.testing.assert_close(actual[0][batch, :, :nq].double(), ref, atol=4e-3, rtol=1e-2)
                assert torch.all(actual[0][batch, :, nq:] == 0)
                if lse is not None:
                    torch.testing.assert_close(actual[1][batch, :, :nq].double(), logsum * math.log2(math.e), atol=2e-3, rtol=1e-3)
                    assert torch.all(actual[1][batch, :, nq:] == -float("inf"))
            assert torch.all(ob[..., dv:] == 77)
            assert torch.all(lb[..., 1::2] == 77)
    finally:
        graph.reset()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sched,pack,mask,scale,splits",
    [
        (0, False, "none", 0.125, 3),
        (1, True, "none", -0.125, 7),
        (2, False, "br_window", 0.125, 3),
        (2, True, "causal", -0.125, 3),
        (0, True, "br_widened", 0.0, 3),
        (1, False, "none", 0.0, 17),
    ],
)
def test_split_multiple_kv_stages(dtype, sched, pack, mask, scale, splits):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    b, h, hk, sq, sk, d = 2, 8, 2, 70, 1025, 512
    rng = torch.Generator().manual_seed(923)
    qc, kc, vc = (torch.randn(shape, generator=rng).to(dtype) * 0.25 for shape in ((b, sq, h, d), (b, sk, hk, d), (b, sk, hk, d)))
    q, k, v = (t.cuda().transpose(1, 2) for t in (qc, kc, vc))
    o, lse = torch.empty_like(q), torch.empty(b, h, sq, device="cuda")
    ql, kl = (torch.tensor(x, dtype=torch.int32, device="cuda") for x in ([sq, 33], [sk, 193]))
    api = SdpaFwdDslSm90(
        q,
        k,
        v,
        o,
        sample_lse=lse,
        split_kv=splits,
        sched_policy=sched,
        pack_gqa=pack,
        scale_softmax=scale,
        is_causal=mask != "none",
        causal_bottom_right=mask.startswith("br"),
        window_size_left=287 if mask == "br_window" else None,
        window_size_right=7 if mask == "br_widened" else None,
        seq_q_lens_present=True,
        seq_kv_lens_present=True,
    )
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def run():
        api.execute(q, k, v, o, lse, seq_q_lens=ql, seq_kv_lens=kl, workspace=ws)

    run()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run()
        for nq, nk in (([sq, 33], [sk, 193]), ([1, 65], [129, 769])):
            ql.copy_(torch.tensor(nq, dtype=torch.int32, device="cuda"))
            kl.copy_(torch.tensor(nk, dtype=torch.int32, device="cuda"))
            ws.fill_(255)
            v.mul_(-0.5)
            graph.replay()
            # CPU FP64 reference uses current payloads and device-length values.
            actual, stats = o.cpu().double(), lse.cpu().double()
            vv = v.cpu().double()
            for bi in range(b):
                qq = qc[bi, : nq[bi]].transpose(0, 1).double()
                kk = kc[bi, : nk[bi]].transpose(0, 1).double().repeat_interleave(h // hk, 0)
                values = vv[bi, :, : nk[bi]].repeat_interleave(h // hk, 0)
                scores = (qq @ kk.transpose(-1, -2)) * scale
                qp = torch.arange(nq[bi])[:, None] + (nk[bi] - nq[bi] if mask.startswith("br") else 0)
                kp = torch.arange(nk[bi])[None, :]
                if mask != "none":
                    scores.masked_fill_(kp > qp + (7 if mask == "br_widened" else 0), -torch.inf)
                if mask == "br_window":
                    scores.masked_fill_(kp < qp - 287, -torch.inf)
                lr = scores.logsumexp(-1)
                ref = torch.exp(scores - lr[..., None]) @ values
                torch.testing.assert_close(actual[bi, :, : nq[bi]], ref, atol=4e-3, rtol=1e-2)
                torch.testing.assert_close(stats[bi, :, : nq[bi]], lr, atol=2e-3, rtol=1e-3)
                assert torch.all(actual[bi, :, nq[bi] :] == 0)
                assert torch.all(stats[bi, :, nq[bi] :] == -torch.inf)
    finally:
        graph.reset()


@pytest.mark.parametrize("pack_gqa", [False, True])
def test_split_artifact_reload(pack_gqa, tmp_path):
    from test_sdpa_prepared_sm90 import test_prepared_artifact_reloads_in_fresh_process

    test_prepared_artifact_reloads_in_fresh_process(False, tmp_path, splits=3, pack_gqa=pack_gqa)


def test_split_launch_domain():
    from dataclasses import replace
    from types import SimpleNamespace
    from cudnn.sdpa.fwd import config_sm90, engines

    params = config_sm90.TemplateParams(split_kv=3)
    config_sm90.validate_params(params)
    with pytest.raises(ValueError, match="dense Q"):
        config_sm90.validate_params(replace(params, thd_varlen=True))
    for splits in (0, 65536, True):
        with pytest.raises(ValueError, match="split_kv"):
            config_sm90.validate_params(replace(params, split_kv=splits))
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm90")
    knobs = engines.SdpaFwdKnobs(split_kv=3, sched_policy=0)
    assert engines.launch_grid_mismatch(caps, SimpleNamespace(b=65535, h_q=1), knobs) is None
    for batch, heads, splits in ((65536, 1, 3), (1, 65536, 3), (1, 1, 65536)):
        assert "65535" in engines.launch_grid_mismatch(caps, SimpleNamespace(b=batch, h_q=heads), replace(knobs, split_kv=splits))
