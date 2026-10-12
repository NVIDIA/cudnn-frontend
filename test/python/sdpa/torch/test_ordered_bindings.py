# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Torch wrapper caches UID order, never execution buffers or streams."""

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import torch_op
from sdpa.torch.test_torch_ops import TestSdpaBwdDense as DenseReference, TestSdpaVarlen as VarlenReference, ref_attention

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("provider", ["backend", "frost"])
@pytest.mark.parametrize("layout", ["dense", "thd"])
def test_torch_ordered_bindings_rebind_and_replay(provider, layout, monkeypatch):
    if provider == "frost":
        from sdpa.torch.test_varlen_metadata import _require_prepared

        _require_prepared()
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("pinned dense/THD FROST forward smoke requires SM100/SM103")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    # Isolate provider-specific plans without modifying the shared graph cache.
    monkeypatch.setattr(torch_op, "_graph_cache", {})
    original_plans = cudnn.pygraph.create_execution_plans
    original_execute = cudnn.pygraph.execute
    orders = {}
    bindings = []

    def create_plans(graph, *args, **kwargs):
        original_plans(graph, *args, **kwargs)
        candidates = [i for i, cfg in enumerate(graph.plans) if (graph._engine_for(cfg) is None) == (provider == "backend")]
        assert candidates, [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
        graph.select_plan(candidates[0])

    def execute(graph, values, workspace, *, handle, tensor_uids=None):
        assert isinstance(values, (tuple, list)), "warm Torch wrapper rebuilt a UID mapping"
        assert isinstance(tensor_uids, tuple) and len(tensor_uids) == len(values)
        assert len(set(tensor_uids)) == len(tensor_uids)
        if graph in orders:
            assert tensor_uids is orders[graph], "UID order was rebuilt on a cached graph"
        orders[graph] = tensor_uids
        # Keep only integer addresses: the cache itself must not keep old tensors alive.
        bindings.append(dict(zip(tensor_uids, (value.data_ptr() for value in values))))
        return original_execute(graph, values, workspace, handle=handle, tensor_uids=tensor_uids)

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plans", create_plans)
    monkeypatch.setattr(cudnn.pygraph, "execute", execute)
    torch.manual_seed(827)
    thd = layout == "thd"
    # Packed backward currently needs padded Stats, which FROST correctly declines.
    backward = not thd or provider == "backend"
    cu = torch.tensor([0, 64, 160], dtype=torch.int32, device="cuda")
    kwargs = dict(cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=96, max_seqlen_kv=96) if thd else {}
    # Pin each provider within its current backward domain on SM100: the
    # half FROST row serves D > 256, while this backend serves D128.
    d = 512 if provider == "frost" and not thd else 128
    shape = (160, 4, d) if thd else (2, 96, 4, d)

    def make_inputs():
        values = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2 for _ in range(4)]
        return values if thd else [value.transpose(1, 2) for value in values]

    def run(values):
        q, k, v, grad = values
        output, stats = torch.ops.cudnn.sdpa_fwd(q, k, v, d**-0.5, is_causal=True, **kwargs)
        if backward:
            lse = torch_op.thd_lse_to_padded(stats[:, :, 0], cu, 96) if thd else stats
            grads = torch.ops.cudnn.sdpa_bwd(grad, q, k, v, output, lse, d**-0.5, is_causal=True, **kwargs)
        else:
            grads = ()
        return output, stats, grads

    def check(values, result):
        q, k, v, grad = values
        if thd:
            output, lse, *grads = VarlenReference()._ref(q, k, v, cu, True, grad=grad if backward else None)
            lse = lse.unsqueeze(-1)
        else:
            qr, kr, vr = (value.detach().float().requires_grad_(True) for value in (q, k, v))
            output, lse = ref_attention(qr, kr, vr, d**-0.5, is_causal=True, return_lse=True)
            output.backward(grad.float())
            grads = (qr.grad, kr.grad, vr.grad)
        for name, actual, expected in [("O", result[0], output), ("Stats", result[1], lse)]:
            DenseReference._assert_close(name, actual, expected)
        if backward:
            for name, actual, expected in zip(("dQ", "dK", "dV"), result[2], grads):
                DenseReference._assert_close(name, actual, expected)

    first = make_inputs()
    check(first, run(first))
    second = make_inputs()
    result = run(second)
    check(second, result)
    assert len(orders) == (2 if backward else 1)
    first_bindings = bindings[0]
    second_bindings = bindings[2 if backward else 1]
    for role, value in zip((torch_op._UIDs.Q, torch_op._UIDs.K, torch_op._UIDs.V), second):
        assert second_bindings[role] == value.data_ptr() != first_bindings[role]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            captured = run(second)
        second[0].mul_(0.5)
        second[3].mul_(0.75)
        if thd:
            cu[1] = 96
        for value in (captured[0], captured[1], *captured[2]):
            value.fill_(float("nan"))
        graph.replay()
        check(second, captured)
    finally:
        graph.reset()
