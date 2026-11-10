# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bytewise repeatability and batch independence of the SM90 split forward path."""

import math

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native SM90")]


@pytest.mark.parametrize("splits,pack_gqa", [(3, False), (17, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_forward_determinism(splits, pack_gqa, dtype):
    d = 512
    h, hk, sq, sk = 8, 2, 17, 257
    rng = torch.Generator().manual_seed(941)
    records = [tuple(torch.randn(shape, generator=rng).to(dtype) * 0.25 for shape in ((h, sq, d), (hk, sk, d), (hk, sk, d))) for _ in range(3)]
    sink_cpu = torch.linspace(-3, 3, h)
    sink = sink_cpu.cuda()

    def prepare(batch):
        q = torch.empty(batch, h, sq, d, dtype=dtype, device="cuda")
        k = torch.empty(batch, hk, sk, d, dtype=dtype, device="cuda")
        v = torch.empty_like(k)
        o = torch.empty_like(q)
        lse = torch.empty(batch, h, sq, device="cuda")
        ql = torch.empty(batch, dtype=torch.int32, device="cuda")
        kl = torch.empty_like(ql)
        api = SdpaFwdDslSm90(
            q,
            k,
            v,
            o,
            sample_lse=lse,
            split_kv=splits,
            pack_gqa=pack_gqa,
            sched_policy=0,
            has_sink=True,
            seq_q_lens_present=True,
            seq_kv_lens_present=True,
            is_causal=True,
            causal_bottom_right=True,
            window_size_left=63,
        )
        assert api.check_support()
        api.compile()
        ws = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

        def bind(order):
            for i, index in enumerate(order):
                if index >= 0:
                    for dst, src in zip((q, k, v), records[index]):
                        dst[i].copy_(src)
                else:
                    q[i].fill_(float("nan"))
                    k[i].fill_(float("nan"))
                    v[i].fill_(float("nan"))
            ql.copy_(torch.tensor([sq if i >= 0 else 0 for i in order], dtype=torch.int32, device="cuda"))
            kl.copy_(torch.tensor([sk if i >= 0 else 0 for i in order], dtype=torch.int32, device="cuda"))

        def launch(output=o, stats=lse, scratch=ws):
            api.execute(q, k, v, output, stats, sinks=sink, seq_q_lens=ql, seq_kv_lens=kl, workspace=scratch)

        return q, k, v, o, lse, ws, bind, launch

    q, k, v, o, lse, ws, bind, launch = prepare(1)
    bind((0,))

    def fresh():
        oo, ll = torch.empty_like(o), torch.empty_like(lse)
        ww = torch.full_like(ws, 255)
        launch(oo, ll, ww)
        return {"o": oo[0], "lse": ll[0]}

    expected = assert_replays(fresh, repeats=4)
    qq, kk, vv = records[0]
    scores = qq.double() @ kk.double().repeat_interleave(h // hk, 0).transpose(-1, -2) / math.sqrt(d)
    qp, kp = torch.arange(sq)[:, None] + sk - sq, torch.arange(sk)[None, :]
    scores.masked_fill_((kp > qp) | (kp < qp - 63), -torch.inf)
    lr = torch.logaddexp(scores.logsumexp(-1), sink_cpu.double()[:, None])
    ref = torch.exp(scores - lr[..., None]) @ vv.double().repeat_interleave(h // hk, 0)
    torch.testing.assert_close(expected["o"].cpu().double(), ref, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(expected["lse"].cpu().double(), lr, atol=2e-3, rtol=1e-3)
    for got, want in zip((q[0], k[0], v[0]), records[0]):
        assert torch.equal(got.cpu(), want)

    q, k, v, o, lse, ws, bind, launch = prepare(129)
    bind([0] + [-1] * 128)
    launch()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            launch()
        torch.cuda.current_stream().wait_stream(stream)
        for order in ([0] + [-1] * 128, [1, 0, 0] + [-1] * 126, [1, 2] * 64 + [0]):
            bind(order)
            ws.fill_(127)
            graph.replay()
            for batch, index in enumerate(order):
                if index == 0:
                    assert_bitwise(snapshot({"o": o[batch], "lse": lse[batch]}), expected, f"batch slot {batch}")
            empty = [i for i, x in enumerate(order) if x < 0]
            if empty:
                assert torch.all(o[empty] == 0) and torch.all(lse[empty] == -torch.inf)
        k[0].mul_(-0.5)
        v[0].add_(0.75)
        graph.replay()
        assert_bitwise(snapshot({"o": o[128], "lse": lse[128]}), expected, "changed neighbor")
        assert_golden(
            f"sdpa-sm90-dense-d{d}-split{splits}-pack{int(pack_gqa)}-{str(dtype).split('.')[-1]}", {"q": qq, "k": kk, "v": vv, "sink": sink_cpu}, expected
        )
    finally:
        graph.reset()
