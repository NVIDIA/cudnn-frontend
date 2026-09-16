# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tanh epilogue must retain small arguments before downstream scaling."""

import cudnn
import pytest
import torch

from cudnn.gemm.frost.knobs import GemmKnobs
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]


@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
def test_tanh_retains_small_arguments_and_capture(config_name):
    m, n, k = 129, 256, 256
    a = torch.ones((1, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.eye(k, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
    scale = (
        torch.tensor(
            [0.0, 1e-9, -1e-9, 1e-7, -1e-7, 1e-4, -1e-4, 1.0, -1.0, 10.0, -10.0, 80.0, -80.0, 0.125, -0.125, 0.5],
            device="cuda",
            dtype=torch.float32,
        )
        .repeat(n // 16)
        .view(1, 1, n)
    )
    output = torch.empty((1, m, n), device="cuda", dtype=torch.float32)
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
    )
    aa = graph.tensor(name="a", dim=[1, m, k], stride=[m * k, k, 1])
    bb = graph.tensor(name="b", dim=[1, k, n], stride=[n * k, 1, k])
    ss = graph.tensor(name="scale", dim=[1, 1, n], stride=[n, n, 1], data_type=cudnn.data_type.FLOAT)
    yy = graph.tanh(graph.mul(graph.matmul(aa, bb), ss)).set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plan(20400, GemmKnobs.from_config(by_name(config_name)).to_public())
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(0)[0] == 20400
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    pack = {aa: a, bb: b, ss: scale, yy: output}

    def run():
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
        graph.execute(pack, workspace, handle=handle)

    def check(factor):
        expected = torch.tanh(scale * factor).expand_as(output)
        # A zero result for a 1e-9 argument must fail, even before amplification.
        torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-12)

    run()
    check(1.0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=side):
        run()
    torch.cuda.current_stream().wait_stream(side)
    a.fill_(-2)
    scale.mul_(0.5)
    output.fill_(float("nan"))
    capture.replay()
    check(-2.0)
