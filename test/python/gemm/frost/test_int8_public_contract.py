# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact large-accumulator INT8 contracts through public FROST graph execution."""

import json

import cudnn
import pytest
import torch

from gemm_test_utils import requires_int8_mma


def _operands(batch, m, n, k, pattern):
    assert k * 128 * 127 < 2**31
    av = torch.full((batch, k), 127, dtype=torch.int8)
    bv = torch.full((batch, k), 127, dtype=torch.int8)
    av[:, 0] = 126
    if pattern == "negative":
        bv.fill_(-127)
    elif pattern == "minimum":
        av.fill_(-128)
        av[:, 0] = -127
    elif pattern == "cancellation":
        bv[:, k // 2 :] = -127
    elif pattern == "zero":
        av.zero_()
    # Distinct batch values detect accidental reuse of batch zero.
    if batch > 1:
        av[1, 0] -= 1
    exact = (av.to(torch.int64) * bv.to(torch.int64)).sum(-1)
    if pattern in ("positive", "negative", "minimum"):
        assert exact[0].to(torch.float32).to(torch.int64) != exact[0]
    a = av[:, None, :].repeat(1, m, 1).cuda()
    b = bv[:, None, :].repeat(1, n, 1).cuda().transpose(-1, -2)
    reference = exact[:, None, None].expand(batch, m, n)
    return a, b, reference


def _check_public(cudnn_handle, record_property, batch, m, n, k, pattern, replay=False):
    a, b, reference = _operands(batch, m, n, k, pattern)
    assert a.dtype == b.dtype == torch.int8
    assert a.stride() == (m * k, k, 1)
    assert b.stride() == (n * k, 1, k)
    graph = cudnn.pygraph(io_data_type=cudnn.data_type.INT8, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.INT32)
    ga = graph.tensor(name="A", dim=list(a.shape), stride=list(a.stride()))
    gb = graph.tensor(name="B", dim=list(b.shape), stride=list(b.stride()))
    gc = graph.matmul(A=ga, B=gb, name="int8_mm")
    gc.set_output(True).set_data_type(cudnn.data_type.INT32)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    names = [graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())]
    record_property("plan_names", json.dumps(names))
    indices = [i for i, name in enumerate(names) if name.startswith("frost_gemm")]
    assert indices, "No public FROST INT8 plan on a supported MMA architecture"
    graph.select_plan(indices[0])
    graph.check_support()
    graph.build_plans()
    assert graph.selected_engine.name == "frost_gemm"
    record_property("selected_plan", names[indices[0]])
    record_property("engine_and_knobs", repr(graph.get_engine_and_knobs_at_index(indices[0])))
    output = torch.empty((batch, m, n), dtype=torch.int32, device="cuda")
    workspace = torch.empty(max(1, graph.get_workspace_size()), dtype=torch.uint8, device="cuda")
    pack = {ga: a, gb: b, gc: output}
    record_property("workspace_bytes", graph.get_workspace_size())
    stream = torch.cuda.Stream()
    previous = cudnn.get_stream(cudnn_handle)
    capture = None
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output.fill_(-777)
            graph.execute(pack, workspace, handle=cudnn_handle)
        stream.synchronize()
        assert output.dtype == torch.int32
        assert torch.equal(output.cpu().to(torch.int64), reference)
        if replay:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                graph.execute(pack, workspace, handle=cudnn_handle)
            for _ in range(3):
                with torch.cuda.stream(stream):
                    output.fill_(-777)
                    capture.replay()
                stream.synchronize()
                assert torch.equal(output.cpu().to(torch.int64), reference)
    finally:
        stream.synchronize()
        if capture is not None:
            capture.reset()
        cudnn.set_stream(cudnn_handle, previous)


@requires_int8_mma
@pytest.mark.L0
def test_int8_public_large_accumulator(cudnn_handle, record_property):
    _check_public(cudnn_handle, record_property, 1, 128, 128, 2048, "positive", replay=True)


@requires_int8_mma
@pytest.mark.L1
@pytest.mark.parametrize("pattern", ["positive", "negative", "minimum", "cancellation", "zero"])
@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("batch,m,n", [(1, 128, 128), (2, 128, 128), (1, 144, 208)])
def test_int8_public_integer_matrix(cudnn_handle, record_property, batch, m, n, k, pattern):
    _check_public(cudnn_handle, record_property, batch, m, n, k, pattern)
