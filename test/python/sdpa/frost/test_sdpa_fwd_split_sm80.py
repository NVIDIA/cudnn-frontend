# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native SM80 split-KV qualification over independently bound page pools."""

import pytest
import torch
import cudnn
from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_paged_sm80 import test_native_paged_adapter_rebind_replay as _paged_probe

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


def test_split_launch_domain():
    from dataclasses import replace
    from types import SimpleNamespace
    from cudnn.sdpa.fwd import config_sm80, engines

    params = config_sm80.TemplateParams(split_kv=3)
    config_sm80.validate_params(params)
    with pytest.raises(ValueError, match="dense Q"):
        config_sm80.validate_params(replace(params, thd_varlen=True))
    for splits in (0, 65536):
        with pytest.raises(ValueError, match="split_kv"):
            config_sm80.validate_params(replace(params, split_kv=splits))
    caps = next(s.capabilities for s in engines.ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm80")
    # Batch times splits may exceed gridDim.z: the producer flattens batch
    # into x, while the combine places only the unsplit batch on z.
    knobs = engines.SdpaFwdKnobs(split_kv=3, sched_policy=0)
    assert engines.launch_grid_mismatch(caps, SimpleNamespace(b=65535, h_q=1), knobs) is None
    for batch, heads, splits in ((65536, 1, 3), (1, 65536, 3), (1, 1, 65536)):
        assert "65535" in engines.launch_grid_mismatch(caps, SimpleNamespace(b=batch, h_q=heads), replace(knobs, split_kv=splits))


@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512), (576, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_paged_split_rebind(dims, dtype, graph_mode, cudnn_handle, monkeypatch):
    single_q = graph_mode
    splits = 17 if single_q else 3
    knobs = {cudnn.knob_type.SPLIT_KV: splits, cudnn.knob_type.SCHED_POLICY: 0}
    _paged_probe(
        (*dims, 16, "HND", "NHD", "br_swa"),
        dtype,
        False,
        graph_mode,
        cudnn_handle,
        monkeypatch,
        case_index=31,
        split_kv=splits,
        graph_knobs=knobs,
        single_q=single_q,
        has_sink=True,
        sink_endpoints=True,
        sink_stride=1,
    )


@pytest.mark.parametrize("dims", [(128, 128), (512, 512), (576, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("stats_log2", [False, True])
def test_dense_split_partial_initialization(dims, dtype, stats_log2):
    """Every scratch row is written before reduction, under replay and empty KV."""
    import math
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm80

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
    api = SdpaFwdDslSm80(
        q,
        k,
        v,
        o,
        sample_lse=lse,
        split_kv=17,
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
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def execute(workspace=ws):
        api.execute(q, k, v, o, lse, sinks=sink, seq_q_lens=qlens, seq_kv_lens=klens, workspace=workspace)

    # Rejections precede all scratch or output writes.
    if dims == (128, 128) and dtype == torch.bfloat16 and stats_log2:
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
