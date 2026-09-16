# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-A MoE pairs preserve ragged routing, expert pitch and graph replay."""

from pathlib import Path
import re

import cudnn
from cuda.bindings import driver as cuda_driver
import pytest
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm120

pytestmark = [pytest.mark.L1, requires_sm120]


def _graph(rows, experts, width, inner, expert_pitch, dtype):
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = graph.tensor(name="tokens", dim=[1, rows, inner], stride=[rows * inner, inner, 1])
    offsets = graph.tensor(name="offsets", dim=[experts, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    weights, products = [], []
    for index in range(2):
        weight = graph.tensor(name=f"weight{index}", dim=[experts, inner, width], stride=[expert_pitch, 1, inner])
        weights.append(weight)
        product = graph.moe_grouped_matmul(a, weight, offsets, mode=cudnn.moe_grouped_matmul_mode.NONE)
        products.append(product)
    result = graph.mul(products[0], graph.swish(products[1]))
    result.set_output(True).set_data_type(dtype)
    # A final activation can hide GEMM errors. Check both exact FP32 products.
    taps = [graph.identity(input=product, name=f"product_tap{index}") for index, product in enumerate(products)]
    for tap in taps:
        tap.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return graph, a, offsets, weights, taps, result


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sched_policy", [0, 1])
@pytest.mark.parametrize("config_name", ["CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4", "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2"])
def test_shared_a_live_pitched_capture(dtype, sched_policy, config_name):
    cfg = by_name(config_name)
    rows, experts, width, inner = 65537, 40, 144, 272
    # Both views alias one allocation; padding makes the expert pitch16B-aligned
    # but not32B-aligned, independent of the FP32 output tap vector width.
    expert_pitch = 2 * width * inner + 8
    io = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF
    graph, a, off, weight_nodes, product_nodes, result_node = _graph(rows, experts, width, inner, expert_pitch, io)
    plan = jit_from_cudnn_graph(graph, config=cfg, moe_sched_policy=sched_policy)
    # Compilation alone does not prove every requested graph output is bound.
    assert {id(t) for t in plan.binding.outputs} == {id(result_node), *(id(t) for t in product_nodes)}
    assert len(plan.chain.output_specs) == 3
    generated = Path(plan.generated_path).read_text()
    assert "num_gemms = 2" in generated and "num_a_operands = 1" in generated and "num_b_operands = 2" in generated
    grid = int(re.search(r"^grid_num_clusters = (\d+)$", generated, re.M).group(1))
    assert plan.workspace_bytes == 128
    base_bounds = [0, 0, 1, 1, 4097, 8192, 16387, rows, rows]
    changed_bounds = [0, 2, 2, 31, 127, 4099, rows - 1, rows, rows]
    bounds_cases = [b[:4] + [b[3]] * (experts - 8) + b[4:] for b in (base_bounds, changed_bounds)]
    for bounds in bounds_cases:
        tiles = sum((end - begin + cfg.cta_tile_m - 1) // cfg.cta_tile_m for begin, end in zip(bounds, bounds[1:]))
        tiles *= (width + cfg.cta_tile_n - 1) // cfg.cta_tile_n
        assert tiles > 4 * grid, (tiles, grid)

    generator = torch.Generator(device="cuda").manual_seed(1012)
    x = (torch.randint(-2, 3, (1, rows, inner), generator=generator, device="cuda").float() * 0.125).to(dtype)
    storage = torch.full((experts * expert_pitch,), float("nan"), dtype=dtype, device="cuda")
    up = torch.as_strided(storage, (experts, width, inner), (expert_pitch, inner, 1))
    gate = torch.as_strided(storage, (experts, width, inner), (expert_pitch, inner, 1), storage_offset=width * inner)
    for weight in (up, gate):
        weight.copy_(torch.randint(-2, 3, weight.shape, generator=generator, device="cuda").float() * 0.125)
    offset_variants = [torch.tensor(bounds[:-1], dtype=torch.int32, device="cuda") for bounds in bounds_cases]
    offsets = offset_variants[0].clone()
    products = [torch.empty((1, rows, width), dtype=torch.float32, device="cuda") for _ in range(2)]
    output = torch.empty((1, rows, width), dtype=dtype, device="cuda")
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    pack = {a: x, off: offsets, result_node: output, **dict(zip(weight_nodes, (up, gate))), **dict(zip(product_nodes, products))}
    refs = []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for changed, bounds in enumerate(bounds_cases):
            values = [torch.empty_like(p) for p in products]
            for expert, (begin, end) in enumerate(zip(bounds, bounds[1:])):
                if begin != end:
                    # The changed variant negates both x and the up view.
                    values[0][0, begin:end] = x[0, begin:end].float() @ up[expert].float().T
                    values[1][0, begin:end] = ((-1) ** changed) * (x[0, begin:end].float() @ gate[expert].float().T)
            expected = (values[0] * torch.nn.functional.silu(values[1])).to(dtype)
            refs.append((values, expected))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    assert not torch.allclose(refs[0][1], refs[1][1], rtol=0.02, atol=0.02)

    def check(index):
        expected_products, expected = refs[index]
        for actual, target in zip(products, expected_products):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        relative = (output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)
        assert float(relative) <= 0.01

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        workspace.fill_(0x7F)
        for value in (*products, output):
            value.fill_(float("nan"))
        plan(pack, workspace=workspace, stream=side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)
    check(0)
    capture = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(capture, stream=side):
        plan(pack, workspace=workspace, stream=side.cuda_stream)

    def checked(result):
        error, *values = result
        assert int(error) == 0, result
        return values

    handle = cuda_driver.CUgraph(capture.raw_cuda_graph())
    _, count = checked(cuda_driver.cuGraphGetNodes(handle, 0))
    nodes, actual_count = checked(cuda_driver.cuGraphGetNodes(handle, count))
    assert count == actual_count == (2 if sched_policy == 0 else 1)
    names = []
    for node in nodes:
        (kind,) = checked(cuda_driver.cuGraphNodeGetType(node))
        assert kind == cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        (params,) = checked(cuda_driver.cuGraphKernelNodeGetParams(node))
        (name,) = checked(cuda_driver.cuFuncGetName(params.func))
        names.append(name.decode())
    assert sum("frost_sm120_moe_grouped_matmul" in name for name in names) == 1
    assert sum("reset_moe_sched_counter" in name for name in names) == int(sched_policy == 0)
    for replay_index, index in enumerate((0, 1, 0)):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            if replay_index > 0:
                x.neg_()
                up.neg_()
            offsets.copy_(offset_variants[index])
            workspace.fill_(0x7F)
            for value in (*products, output):
                value.fill_(float("nan"))
            capture.replay()
        torch.cuda.current_stream().wait_stream(side)
        check(index)
        assert not torch.allclose(output, refs[1 - index][1], rtol=0.02, atol=0.02), "Stale-output negative control"
    capture.reset()
