# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime padding is unread data, including NaNs in physically valid storage."""

import math

import pytest
import torch
import cudnn

from frost_test_utils import _SM, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 80, reason="requires native SM80")]


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sq,sk", [(33, 65), (128, 128)])
@pytest.mark.parametrize("causal", [False, True])
def test_runtime_padding_nan_rebind_replay(d, dv, dtype, sq, sk, causal, cudnn_handle):
    torch.manual_seed(324 + d + sq)
    b, h, hk = 3, 4, 2

    def allocate():
        data = {
            name: torch.empty(b, seq, heads, width + 8, device="cuda", dtype=dtype)[..., :width].transpose(1, 2)
            for name, heads, seq, width in (("q", h, sq, d), ("k", hk, sk, d), ("v", hk, sk, dv))
        }
        ob = torch.full((b, sq, h, dv + 8), 77, dtype=dtype, device="cuda")
        lb = torch.full((b, h, 2 * sq), 77.0, device="cuda")
        data.update(o=ob[..., :dv].transpose(1, 2), stats=lb[..., ::2].unsqueeze(-1))
        data["ql"] = torch.empty(b, dtype=torch.int32, device="cuda")
        data["kl"] = torch.empty_like(data["ql"])
        if causal:
            data["bias"] = torch.randn(1, h, sq, sk, device="cuda") * 0.1
            data["sink"] = torch.randn(1, h, 1, 1, device="cuda")
        return data, ob, lb

    values, ob, lb = allocate()
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nodes = {name: graph.tensor_like(values[name], name=name) for name in ("q", "k", "v")}
    for name in ("ql", "kl"):
        nodes[name] = graph.tensor(name=name, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
    optional = {}
    if causal:
        for name in ("bias", "sink"):
            nodes[name] = graph.tensor_like(values[name], name=name)
        optional.update(bias=nodes["bias"], sink_token=nodes["sink"])
    out, stats = graph.sdpa(
        q=nodes["q"],
        k=nodes["k"],
        v=nodes["v"],
        attn_scale=1 / math.sqrt(d),
        generate_stats=True,
        use_padding_mask=True,
        seq_len_q=nodes["ql"],
        seq_len_kv=nodes["kl"],
        use_causal_mask=causal,
        **optional,
    )
    for name, node, dt in (("o", out, io), ("stats", stats, cudnn.data_type.FLOAT)):
        node.set_output(True).set_dim(values[name].shape).set_stride(values[name].stride()).set_data_type(dt)
        nodes[name] = node
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_fwd_prefill_sm80")
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    for binding in range(2):
        if binding:
            values, ob, lb = allocate()
        pack = {nodes[name]: tensor for name, tensor in values.items()}
        capture = torch.cuda.CUDAGraph()

        def launch():
            previous = cudnn.get_stream(cudnn_handle)
            try:
                cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
                graph.execute(pack, workspace, handle=cudnn_handle)
            finally:
                cudnn.set_stream(cudnn_handle, previous)

        try:
            for phase, (ql, kl) in enumerate((((sq, sq // 2, 0), (sk - 3, 0, sk)), ((0, sq, 1), (sk, sk - 7, 0)))):
                values["ql"].copy_(torch.tensor(ql, dtype=torch.int32, device="cuda"))
                values["kl"].copy_(torch.tensor(kl, dtype=torch.int32, device="cuda"))
                for name in ("q", "k", "v"):
                    values[name].fill_(float("nan"))
                    for bi, count in enumerate(ql if name == "q" else kl):
                        values[name][bi, :, :count].normal_()
                ob.fill_(77)
                lb.fill_(77)
                if phase == 0:
                    launch()
                else:
                    capture.replay()
                for bi, (nq, nk) in enumerate(zip(ql, kl)):
                    q = values["q"][bi, :, :nq].double()
                    k = values["k"][bi, :, :nk].double().repeat_interleave(h // hk, 0)
                    v = values["v"][bi, :, :nk].double().repeat_interleave(h // hk, 0)
                    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
                    if causal:
                        scores += values["bias"][0, :, :nq, :nk].double()
                        scores.masked_fill_(torch.arange(nk, device="cuda")[None, :] > torch.arange(nq, device="cuda")[:, None], -float("inf"))
                        all_scores = torch.cat((scores, values["sink"][0, :, 0, 0].double()[:, None, None].expand(h, nq, 1)), -1)
                    else:
                        all_scores = scores
                    expected_stats = all_scores.logsumexp(-1)
                    expected_o = torch.nan_to_num((scores - expected_stats[..., None]).exp(), nan=0) @ v
                    torch.testing.assert_close(values["o"][bi, :, :nq].double(), expected_o, atol=4e-3, rtol=1e-2)
                    torch.testing.assert_close(values["stats"][bi, :, :nq, 0].double(), expected_stats, atol=2e-3, rtol=1e-3)
                    assert torch.all(values["o"][bi, :, nq:] == 0)
                    assert torch.all(values["stats"][bi, :, nq:] == -float("inf"))
                assert torch.all(ob[..., dv:] == 77) and torch.all(lb[..., 1::2] == 77)
                if phase == 0:
                    with torch.cuda.graph(capture):
                        launch()
        finally:
            capture.reset()
