# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dirty MoE scheduler counter must be reset on every handle-stream launch."""

import cudnn
import pytest
import torch

from cudnn.gemm.frost.knobs import GemmKnobs
from cudnn.gemm.frost.tile_config import CATALOG
from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]


@pytest.mark.parametrize("sched_policy", [0, 1])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma",
        "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma",
        "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma",
    ],
)
def test_moe_counter_reset_dirty_workspace_live_capture(fused, config_name, sched_policy):
    # More tiles than the persistent grid: a stale counter cannot accidentally
    # pass by computing only the first resident wave. Include empty groups.
    tokens, experts, width = 32769, 8, 256
    torch.manual_seed(1209)
    x = torch.randint(-2, 3, (1, tokens, width), device="cuda").to(torch.bfloat16) * 0.125
    weights = [torch.randint(-2, 3, (experts, width, width), device="cuda").to(torch.bfloat16) * 0.125 for _ in range(2 if fused else 1)]
    bounds = [0, 0, 1, 1, 4097, 8192, 16387, tokens, tokens]
    offsets = torch.tensor(bounds[:-1], device="cuda", dtype=torch.int32)
    output = torch.empty_like(x)
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        handle=handle,
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = graph.tensor(name="tokens", dim=[1, tokens, width], stride=[tokens * width, width, 1])
    fto = graph.tensor(name="offsets", dim=[experts, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    pack = {a: x, fto: offsets}
    products = []
    for index, weight in enumerate(weights):
        b = graph.tensor(name=f"weight{index}", dim=[experts, width, width], stride=[width * width, 1, width])
        pack[b] = weight
        products.append(graph.moe_grouped_matmul(a, b, fto, mode=cudnn.moe_grouped_matmul_mode.NONE))
    out = graph.mul(a=products[0], b=graph.swish(input=products[1])) if fused else products[0]
    out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    pack[out] = output
    config = next(c for c in CATALOG if c.name == config_name)
    graph.validate()
    graph.build_operation_graph()
    from dataclasses import replace

    knobs = replace(GemmKnobs.from_config(config), moe_sched_policy=sched_policy).to_public()
    graph.create_execution_plan(20400, knobs)
    graph.check_support()
    graph.build_plans()
    assert graph.get_engine_and_knobs_at_index(0) == (20400, knobs)
    assert GemmKnobs.from_public(knobs).moe_sched_policy == sched_policy
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def reference():
        result = torch.empty_like(output)
        for expert, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            if begin == end:
                continue
            up = x[0, begin:end].float() @ weights[0][expert].float().T
            if fused:
                gate = x[0, begin:end].float() @ weights[1][expert].float().T
                up = up * torch.nn.functional.silu(gate)
            result[0, begin:end] = up
        return result

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    cudnn.set_stream(handle, side.cuda_stream)
    with torch.cuda.stream(side):
        # Positive large scheduler count: a disabled reset terminates without
        # doing all tiles, giving a deterministic numerical failure, not a hang.
        workspace.fill_(0x7F)
        output.fill_(float("nan"))
        graph.execute(pack, workspace, handle=handle)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(output, reference(), rtol=0.02, atol=0.02)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=side):
        graph.execute(pack, workspace, handle=handle)
    torch.cuda.current_stream().wait_stream(side)
    previous = output.clone()
    for repeat in range(3):
        x.neg_()
        bounds = [0, 2, 2, 31, 127, 4099, tokens - 1, tokens, tokens]
        offsets.copy_(torch.tensor(bounds[:-1], device="cuda", dtype=torch.int32))
        workspace.fill_(0x7F)
        output.fill_(float("nan"))
        capture.replay()
        torch.testing.assert_close(output, reference(), rtol=0.02, atol=0.02)
        if repeat == 0:
            assert not torch.equal(previous, output), "Changed-input negative control"
