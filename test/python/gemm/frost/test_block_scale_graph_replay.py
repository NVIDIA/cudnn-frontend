# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public FROST block-scale execution must read live inputs on graph replay."""

import pytest
import torch

import cudnn
from cudnn.gemm.frost.graph_analyzer import analyze_with_binding
from gemm_test_utils import requires_sm100, to_blocked
from test_block_scale_matmul import _build_nvfp4_graph, _make_block_scale_inputs

pytestmark = [pytest.mark.L1, requires_sm100]


@pytest.mark.parametrize("shape", [(128, 128, 128), (144, 144, 256)])
@pytest.mark.parametrize(
    "combo,edge", [("nvfp4", "normal"), ("mxfp4", "normal"), ("mxfp8", "normal"), ("nvfp4", "zero"), ("nvfp4", "mixed_zero"), ("nvfp4", "overflow")]
)
def test_public_block_scale_graph_reads_live_inputs(cudnn_handle, monkeypatch, shape, combo, edge):
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    m, n, k = shape
    torch.manual_seed(919)
    a, b, sfa, sfb, reference, block_size, sf_dtype, a_dtype = _make_block_scale_inputs(combo, m, n, k)
    if edge != "normal":
        # Packed E2M1 0x77 contains two +6 values. Exact powers / integer
        # products make the zero and positive-overflow expectations independent
        # of a low-precision GEMM reference or its TF32 configuration.
        a.view(torch.uint8).fill_(0x77)
        b.view(torch.uint8).fill_(0x77)
        sf_a = torch.ones((m, k // block_size), device="cuda")
        sf_b = torch.ones((n, k // block_size), device="cuda")
        if edge == "zero":
            sf_a.zero_()
        elif edge == "mixed_zero":
            sf_a[:, ::2] = 0
        else:
            sf_a.fill_(448)
            sf_b.fill_(448)
        sfa, sfb = sf_a.to(torch.float8_e4m3fn), sf_b.to(torch.float8_e4m3fn)
        reference = (6 * sf_a.double().repeat_interleave(block_size, -1)) @ (6 * sf_b.double().repeat_interleave(block_size, -1)).t()
    reference = reference.to(torch.float16)
    sfa = to_blocked(sfa).view(1, ((m + 127) // 128) * 128, -1)
    sfb = to_blocked(sfb).view(1, ((n + 127) // 128) * 128, -1)
    graph = _build_nvfp4_graph(m, n, k, block_size=block_size, sf_dt=sf_dtype, a_dt=a_dtype)
    _, binding = analyze_with_binding(graph)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    plans = [i for i in range(graph.get_execution_plan_count()) if graph.get_plan_name_at_index(i).startswith("frost_gemm")]
    assert plans, "expected a FROST block-scale plan on a supported architecture"
    graph.select_plan(plans[0])
    graph.check_support()
    graph.build_plans()
    assert graph.selected_engine.name == "frost_gemm"
    output = torch.empty((1, m, n), device="cuda", dtype=torch.float16)
    workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
    pack = {binding.a_operands[0]: a, binding.b_operands[0]: b, binding.sfa_operands[0]: sfa, binding.sfb_operands[0]: sfb, binding.outputs[0]: output}
    saved = a.view(torch.uint8).clone()
    previous_stream = cudnn.get_stream(cudnn_handle)
    capture = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    try:
        cudnn.set_stream(cudnn_handle, stream.cuda_stream)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                graph.execute(pack, workspace, handle=cudnn_handle)
        stream.synchronize()
        torch.testing.assert_close(output[0], reference, atol=2e-1, rtol=2e-2)
        if edge == "overflow":
            assert torch.isposinf(output).all()
        with torch.cuda.graph(capture, stream=stream):
            graph.execute(pack, workspace, handle=cudnn_handle)
        # Keep every captured address alive. Poison output before every replay,
        # change input bytes in-place and restore them, including a repeat zero.
        for zero in (False, True, True, False):
            if zero:
                a.view(torch.uint8).zero_()
            else:
                a.view(torch.uint8).copy_(saved)
            output.fill_(float("nan"))
            capture.replay()
            torch.cuda.synchronize()
            if zero:
                assert torch.count_nonzero(output) == 0
            else:
                torch.testing.assert_close(output[0], reference, atol=2e-1, rtol=2e-2)
    finally:
        torch.cuda.synchronize()
        capture.reset()
        cudnn.set_stream(cudnn_handle, previous_stream)
