# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 native prefill: runtime KV padding is unread data, including NaNs in physically valid storage."""

import math

import pytest
import torch
import cudnn

from frost_test_utils import requires_blackwell_geforce, requires_dsl, select_engine

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sq,sk", [(33, 65), (128, 200)])
@pytest.mark.parametrize("causal", [False, True])
def test_runtime_kv_padding_nan_rebind_replay(d, dv, dtype, sq, sk, causal, cudnn_handle):
    torch.manual_seed(120 + d + sq)
    b, h, hk = 3, 4, 2

    def allocate():
        data = {
            name: torch.zeros(b, seq, heads, width + 8, device="cuda", dtype=dtype)[..., :width].transpose(1, 2)
            for name, heads, seq, width in (("q", h, sq, d), ("k", hk, sk, d), ("v", hk, sk, dv))
        }
        data["o"] = torch.full((b, sq, h, dv), 77, dtype=dtype, device="cuda").transpose(1, 2)
        data["stats"] = torch.full((b, h, sq, 1), 77.0, device="cuda")
        data["ql"] = torch.empty(b, dtype=torch.int32, device="cuda")
        data["kl"] = torch.empty_like(data["ql"])
        return data

    values = allocate()
    io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16
    graph = cudnn.pygraph(io_data_type=io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    nodes = {name: graph.tensor_like(values[name], name=name) for name in ("q", "k", "v")}
    for name in ("ql", "kl"):
        nodes[name] = graph.tensor(name=name, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
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
    )
    for name, node, dt in (("o", out, io), ("stats", stats, cudnn.data_type.FLOAT)):
        node.set_output(True).set_dim(values[name].shape).set_stride(values[name].stride()).set_data_type(dt)
        nodes[name] = node
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    select_engine(graph, "sdpa_fwd_prefill_sm120")
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    for binding in range(2):
        if binding:
            values = allocate()
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
            # Phase 1 replays the captured graph with different lengths, so the pad rows move.
            for phase, (ql, kl) in enumerate((((sq, sq // 2, 1), (sk - 3, sk, sk // 3)), ((1, sq, sq // 3), (sk - 7, sk // 2 + 1, sk)))):
                values["ql"].copy_(torch.tensor(ql, dtype=torch.int32, device="cuda"))
                values["kl"].copy_(torch.tensor(kl, dtype=torch.int32, device="cuda"))
                for name in ("q", "k", "v"):
                    values[name].fill_(float("nan"))
                    for bi, count in enumerate(ql if name == "q" else kl):
                        values[name][bi, :, :count].normal_()
                values["o"].fill_(77)
                if phase == 0:
                    launch()
                else:
                    capture.replay()
                torch.cuda.synchronize()
                for bi, (nq, nk) in enumerate(zip(ql, kl)):
                    q = values["q"][bi, :, :nq].double()
                    k = values["k"][bi, :, :nk].double().repeat_interleave(h // hk, 0)
                    v = values["v"][bi, :, :nk].double().repeat_interleave(h // hk, 0)
                    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
                    if causal:
                        scores.masked_fill_(torch.arange(nk, device="cuda")[None, :] > torch.arange(nq, device="cuda")[:, None], -float("inf"))
                    expected_stats = scores.logsumexp(-1)
                    expected_o = (scores - expected_stats[..., None]).exp() @ v
                    torch.testing.assert_close(values["o"][bi, :, :nq].double(), expected_o, atol=4e-3, rtol=1e-2)
                    torch.testing.assert_close(values["stats"][bi, :, :nq, 0].double(), expected_stats, atol=2e-3, rtol=1e-3)
                if phase == 0:
                    with torch.cuda.graph(capture):
                        launch()
        finally:
            capture.reset()


@pytest.mark.parametrize("sink_value", [-float("inf"), float("inf")])
def test_d512_dense_sink_limits(sink_value):
    from sdpa.frost.native_forward_sm120_cases import _case, _execute

    api, tensors, storage = _case(d=512, features=True)
    tensors["sinks"].fill_(sink_value)
    tensors["seq_kv_lens"][1] = 0
    api.compile()
    assert api._k_mod.__file__.endswith("sm120/prefill_d512_f16.py")
    assert api._k_mod.PARAMS.page_size == 0
    workspace = torch.empty(max(1, api.scratch_workspace_bytes()), dtype=torch.uint8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    try:
        _execute(api, tensors, workspace)
        with torch.cuda.graph(graph):
            _execute(api, tensors, workspace)
        for _ in range(2):
            tensors["o"].fill_(float("nan"))
            graph.replay()
            for batch, nq in enumerate((17, 14)):
                out, lse = tensors["o"][batch, :, :nq], tensors["lse"][batch, :, :nq]
                if sink_value > 0 or batch == 1:
                    assert torch.count_nonzero(out) == 0
                    assert torch.all(lse == sink_value)
                else:
                    q = tensors["q"][batch, :, :nq].double()
                    k = tensors["k"][batch].double().repeat_interleave(2, 0)
                    v = tensors["v"][batch].double().repeat_interleave(2, 0)
                    scores = q @ k.transpose(-1, -2) / math.sqrt(512)
                    qi = torch.arange(nq, device="cuda")[:, None]
                    ki = torch.arange(k.shape[1], device="cuda")[None, :]
                    scores.masked_fill_(ki > qi, -torch.inf)
                    torch.testing.assert_close(out.double(), scores.softmax(-1) @ v, atol=0.003, rtol=0.02)
                    torch.testing.assert_close(lse.double(), scores.logsumexp(-1), atol=0.0006, rtol=0)
            assert (storage["o"][..., -1] == 11).all()
            assert torch.isnan(storage["lse"][..., 1::2]).all()
    finally:
        graph.reset()
