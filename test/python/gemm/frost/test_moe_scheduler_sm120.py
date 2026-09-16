# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise scheduler-slot reuse under racecheck, including live graph replay."""

import re
from pathlib import Path

import cudnn
import pytest
import torch
from cuda.bindings import driver as cuda_driver

from cudnn.frost.buffers import cutedsl_requirement_error

if (dsl_error := cutedsl_requirement_error("Frost MoE scheduler tests")) is not None:
    pytest.skip(dsl_error, allow_module_level=True)

from cudnn.gemm.frost.tile_config import by_name
from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from gemm_test_utils import requires_sm120, to_blocked, with_static_segmented_capacity
from test_moe_grouped_matmul_fwd import _build_graph as _plain_graph, _vp_moe
from test_moe_grouped_block_scale_matmul_fwd import _build_graph as _scaled_graph, _vp_bs

pytestmark = [pytest.mark.L1, requires_sm120]


@pytest.mark.parametrize("sched_policy", [0, 1])
@pytest.mark.parametrize(
    "config_name,kind,experts",
    [
        (name, kind, 8)
        for kind in ("bf16", "fp16", "nvfp4", "mxfp8")
        for name in (
            "CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4",
            "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
            # Block-scale TMA boxes require 128-byte K rows. Keep the large
            # accumulator geometry while respecting that layout contract.
            "CONFIG_sm120_256x256x128_16x16x32_cluster1x1_warps2x4" if kind in ("nvfp4", "mxfp8") else "CONFIG_sm120_256x256x64_16x16x32_cluster1x1_warps2x4",
        )
    ]
    + [("CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4", kind, 40) for kind in ("bf16", "nvfp4", "mxfp8")],
)
def test_scheduler_ring_reuse_live_capture(kind, config_name, sched_policy, experts):
    # Small tiles expose the original WAR hazard; the second geometry varies
    # the compute-warp arrangement. SM120 does not support CTA clusters.
    # The large accumulator tile stresses the last shared-memory reads before
    # a consumer releases its A/B stage for the next async TMA load.
    width = 256
    cfg = by_name(config_name)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    n_tiles = (width + cfg.cta_tile_n - 1) // cfg.cta_tile_n
    tokens = max(65537, (5 * sm_count * cfg.cta_tile_m + n_tiles - 1) // n_tiles + 1)
    bounds_cases = [
        [0, 0, 1, 1, 4097, 8192, 16387, tokens, tokens],
        [0, 2, 2, 31, 127, 4099, tokens - 1, tokens, tokens],
    ]
    # Empty groups span the 32-lane prefix-scan boundary, with live work after it.
    bounds_cases = [b[:4] + [b[3]] * (experts - 8) + b[4:] for b in bounds_cases]
    generator = torch.Generator().manual_seed(1719)
    x_values = torch.randint(-2, 3, (1, tokens, width), generator=generator).float().cuda()
    w_values = torch.randint(-2, 3, (experts, width, width), generator=generator).float().cuda()

    def pack_fp4(value):
        codes = (value.abs() * 2 + (value < 0) * 8).to(torch.uint8)
        return (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)

    def encode(value):
        if kind == "nvfp4":
            return pack_fp4(value)
        return value.to({"bf16": torch.bfloat16, "fp16": torch.float16, "mxfp8": torch.float8_e4m3fn}[kind])

    x, weight = encode(x_values), encode(w_values)
    if kind in ("bf16", "fp16"):
        input_dt = cudnn.data_type.BFLOAT16 if kind == "bf16" else cudnn.data_type.HALF
        graph = _plain_graph(experts, tokens, width, width, output_dt=cudnn.data_type.FLOAT, input_dt=input_dt)
    else:
        graph = _scaled_graph(experts, tokens, width, width, experts, combo=kind, output_dt=cudnn.data_type.FLOAT)
    plan = jit_from_cudnn_graph(graph, config=cfg, moe_sched_policy=sched_policy)
    assert plan.workspace_bytes == 128
    generated = Path(plan.generated_path).read_text()
    grid = int(re.search(r"^grid_num_clusters = (\d+)$", generated, re.M).group(1))
    # Prove the case has enough work to recycle both ring slots repeatedly,
    # using the actual compiled grid rather than assuming an SM count.
    for bounds in bounds_cases:
        tiles = sum((end - begin + cfg.cta_tile_m - 1) // cfg.cta_tile_m for begin, end in zip(bounds, bounds[1:]))
        tiles *= (width + cfg.cta_tile_n - 1) // cfg.cta_tile_n
        assert tiles > 4 * grid, (tiles, grid, config_name)

    offsets = torch.tensor(bounds_cases[0][:-1], dtype=torch.int32, device="cuda")
    output = torch.empty((1, tokens, width), dtype=torch.float32, device="cuda")
    workspace = torch.empty(plan.workspace_bytes, dtype=torch.uint8, device="cuda")
    if kind in ("bf16", "fp16"):
        pack = _vp_moe(plan, x, weight, offsets, output)
        scaled_x, scaled_w = x_values[0], w_values
        a_scales = None
    else:
        block = 16 if kind == "nvfp4" else 32
        cols = width // block
        scale_dtype = torch.float8_e4m3fn if kind == "nvfp4" else torch.float8_e8m0fnu
        a_log = (2.0 ** torch.randint(-1, 2, (tokens, cols), generator=generator)).cuda().to(scale_dtype)
        b_log = (2.0 ** torch.randint(-1, 2, (experts, width, cols), generator=generator)).cuda().to(scale_dtype)

        def segmented_scales(bounds):
            parts = [to_blocked(a_log[begin:end]).view(torch.uint8) for begin, end in zip(bounds, bounds[1:])]
            live = torch.cat(parts).view(scale_dtype)
            return with_static_segmented_capacity(live, tokens, experts, cols)

        a_scales = [segmented_scales(bounds) for bounds in bounds_cases]
        sfa = a_scales[0].clone()
        sfb = torch.cat([to_blocked(b_log[e]).view(torch.uint8) for e in range(experts)]).view(scale_dtype).view(experts, cols, width)
        pack = _vp_bs(plan, x, weight, output, sfa, sfb, fto=offsets)
        scaled_x = x_values[0] * a_log.float().repeat_interleave(block, 1)
        scaled_w = w_values * b_log.float().repeat_interleave(block, 2)

    # All values and products are exactly representable binary fractions.
    # Compute references before capture and require exact FP32 accumulation.
    references = []
    for sign, bounds in zip([1, -1], bounds_cases):
        ref = torch.empty_like(output)
        for expert, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            if begin != end:
                ref[0, begin:end] = sign * (scaled_x[begin:end] @ scaled_w[expert].T)
        references.append(ref)
    assert not torch.equal(references[0], references[1])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        workspace.fill_(0x7F)
        output.fill_(float("nan"))
        plan(pack, workspace=workspace, stream=side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(output, references[0], rtol=0, atol=0)
    capture = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(capture, stream=side):
        plan(pack, workspace=workspace, stream=side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)

    # Direct JIT plans are framework-neutral: the torch context alone does
    # not select their launch stream. Prove that this graph contains Frost.
    def checked(result):
        error, *values = result
        assert int(error) == 0, result
        return values

    graph_handle = cuda_driver.CUgraph(capture.raw_cuda_graph())
    _, count = checked(cuda_driver.cuGraphGetNodes(graph_handle, 0))
    assert count > 0, "Frost must execute inside the captured graph"
    nodes, actual_count = checked(cuda_driver.cuGraphGetNodes(graph_handle, count))
    assert actual_count == count
    kernel_names = []
    for node in nodes:
        (node_type,) = checked(cuda_driver.cuGraphNodeGetType(node))
        assert node_type != cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET, "MoE reset must remain in the compiled compute launch"
        if node_type == cuda_driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            (params,) = checked(cuda_driver.cuGraphKernelNodeGetParams(node))
            (name,) = checked(cuda_driver.cuFuncGetName(params.func))
            kernel_names.append(name.decode())
    assert sum("cudnn_kernel_frost_sm120_moe" in name for name in kernel_names) == 1, kernel_names
    assert sum("reset_moe_sched_counter" in name for name in kernel_names) == (1 - sched_policy), kernel_names
    assert len(kernel_names) == (2 - sched_policy), kernel_names
    for index in [1, 0, 1]:
        x.copy_(encode(x_values if index == 0 else -x_values))
        offsets.copy_(torch.tensor(bounds_cases[index][:-1], dtype=torch.int32, device="cuda"))
        if a_scales is not None:
            sfa.copy_(a_scales[index])
        workspace.fill_(0x7F)
        output.fill_(float("nan"))
        capture.replay()
        torch.testing.assert_close(output, references[index], rtol=0, atol=0)
        assert not torch.equal(output, references[1 - index]), "Stale-output negative control"
