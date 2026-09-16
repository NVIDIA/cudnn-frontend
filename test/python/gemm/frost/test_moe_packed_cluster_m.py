# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-B M multicast through the public graph API, including ragged persistent waves."""

from dataclasses import replace
from itertools import product

import cudnn
import pytest
import torch

pytestmark = pytest.mark.L1
CLUSTERS = ((2, 1), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (16, 1))
CASES = tuple(product((17, 32769), (1, 2), (0, 1)))


class _PackedStage:
    def __init__(self, handle, x, weights, offsets, scale, output, multiplier, config, policy):
        from cudnn.gemm.frost.knobs import GemmKnobs

        self.handle = handle
        self.graph = g = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, handle=handle
        )
        self.pack = {}

        def operand(name, value, dtype):
            node = g.tensor(name=name, dim=list(value.shape), stride=list(value.stride()), data_type=dtype)
            self.pack[node] = value
            return node

        a = operand("tokens", x.unsqueeze(0), cudnn.data_type.FP8_E4M3)
        first = operand("first_token_offsets", offsets[:-1].view(-1, 1, 1), cudnn.data_type.INT32)
        scales = operand("expert_scale", scale, cudnn.data_type.FLOAT)
        products = []
        for index, value in enumerate(weights):
            b = operand(f"weight{index}", value, cudnn.data_type.FP8_E4M3)
            products.append(g.mul(g.moe_grouped_matmul(a, b, first, mode=cudnn.moe_grouped_matmul_mode.NONE, weight_layout="blocked_128x128_v1"), scales))
        if len(weights) == 2:
            factor = operand("multiplier", multiplier, cudnn.data_type.FLOAT)
            lo = operand("minimum", torch.full((1, 1, 1), -448.0, device=x.device), cudnn.data_type.FLOAT)
            hi = operand("maximum", torch.full((1, 1, 1), 448.0, device=x.device), cudnn.data_type.FLOAT)
            y = g.min(g.max(g.mul(g.mul(products[0], g.swish(products[1])), factor), lo), hi)
            dtype = cudnn.data_type.FP8_E4M3
        else:
            y, dtype = products[0], cudnn.data_type.BFLOAT16
        y.set_dim([1, *output.shape]).set_stride([output.numel(), *output.stride()]).set_data_type(dtype).set_output(True)
        self.pack[y] = output.unsqueeze(0)
        g.validate()
        g.build_operation_graph()
        public = replace(GemmKnobs.from_config(config), moe_sched_policy=policy).to_public()
        g.create_execution_plan(20400, public)
        g.check_support()
        assert g.get_execution_plan_count() == 1
        engine, replayed = g.get_engine_and_knobs_at_index(0)
        assert int(engine) == 20400
        assert {int(k): int(v) for k, v in replayed.items()} == {int(k): int(v) for k, v in public.items()}
        g.build_plan_at_index(0)
        self.workspace = torch.empty(g.get_workspace_size_plan_at_index(0), device=x.device, dtype=torch.uint8)

    def run(self):
        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        self.graph.execute_plan_at_index(self.pack, self.workspace, 0, handle=self.handle)


def _compare(actual, expected):
    diff = actual.float() - expected.float()
    relative = float(diff.norm() / expected.float().norm().clamp_min(1e-12))
    assert torch.isfinite(actual.float()).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
    assert relative <= 0.01, relative


@pytest.mark.parametrize("cluster", CLUSTERS, ids=[f"m{m}n{n}" for m, n in CLUSTERS])
@pytest.mark.parametrize("case", CASES, ids=[f"rows{m}_gemms{g}_policy{p}" for m, g, p in CASES])
def test_packed_cluster_m_live_capture(cudnn_handle, monkeypatch, cluster, case):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The packed-B contract currently requires SM100")
    from cudnn.frost.buffers import cutedsl_requirement_error, cutedsl_state
    from cudnn.gemm.frost.tile_config import by_name

    if not cutedsl_state()[0]:
        pytest.skip("CuTe DSL is required")
    if reason := cutedsl_requirement_error("Packed Frost MoE"):
        pytest.skip(reason)
    rows, count, policy = case
    cm, cn = cluster
    torch.manual_seed(371 + rows)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    e, n, k = 8, 256, 256
    x = (torch.randn(rows, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    weights = [(torch.randn(e, n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn) for _ in range(count)]
    blocked = [w.reshape(e, n // 128, 128, k // 128, 128).permute(0, 1, 3, 2, 4).contiguous() for w in weights]
    initial = [0, 0, 1, 1, rows // 4, rows // 2, rows // 2, rows - 1, rows]
    changed = [0, rows // 8, rows // 8, rows // 3, rows // 3, rows // 2, 3 * rows // 4, rows, rows]
    starts = initial
    offsets = torch.tensor(initial, device="cuda", dtype=torch.int32)
    original_offsets, changed_offsets = offsets.clone(), torch.tensor(changed, device="cuda", dtype=torch.int32)
    scale = torch.ones(e, 1, 1, device="cuda")
    multiplier = torch.ones(1, 1, 1, device="cuda")
    dtype = torch.bfloat16 if count == 1 else torch.float8_e4m3fn
    output = torch.empty(rows, n, device="cuda", dtype=dtype)

    def reference():
        result = torch.empty(rows, n, device="cuda", dtype=torch.float32)
        for expert in range(e):
            a, b = starts[expert : expert + 2]
            values = [(x[a:b].float() @ w[expert].float().T) * scale[expert] for w in weights]
            result[a:b] = values[0] if count == 1 else values[0] * torch.nn.functional.silu(values[1]) * multiplier.view(())
        return result.clamp(-448, 448).to(dtype) if count == 2 else result.to(dtype)

    tile_n = 128 if count == 1 else 64
    config = by_name(f"CONFIG_sm100_64x{tile_n}x128_64x{tile_n}x32_cluster{cm}x{cn}_1ctamma")
    stage = _PackedStage(cudnn_handle, x, blocked, offsets, scale, output, multiplier, config, policy)
    stage.run()
    _compare(output, reference())

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared packed MoE must not allocate or convert during capture")

    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(torch, "empty_like", forbidden)
        patch.setattr(torch.Tensor, "contiguous", forbidden)
        with torch.cuda.graph(graph):
            stage.run()
    for which in ["input", "offsets", "scale"] + (["multiplier"] if count == 2 else []) + list(range(count)):
        previous = output.clone()
        if which == "input":
            x.view(torch.uint8).bitwise_xor_(128)
        elif which == "offsets":
            offsets.copy_(changed_offsets)
            starts = changed
        elif which == "scale":
            scale.mul_(0.5)
        elif which == "multiplier":
            multiplier.mul_(0.5)
        else:
            weights[which].view(torch.uint8).bitwise_xor_(128)
            blocked[which].view(torch.uint8).bitwise_xor_(128)
        expected = reference()
        assert float((previous.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)) > 0.01
        output.view(torch.uint8).fill_(127)
        torch.cuda.set_sync_debug_mode("error")
        try:
            graph.replay()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        _compare(output, expected)
        if which == "input":
            x.view(torch.uint8).bitwise_xor_(128)
        elif which == "offsets":
            offsets.copy_(original_offsets)
            starts = initial
        elif which == "scale":
            scale.mul_(2)
        elif which == "multiplier":
            multiplier.mul_(2)
        else:
            weights[which].view(torch.uint8).bitwise_xor_(128)
            blocked[which].view(torch.uint8).bitwise_xor_(128)
        output.view(torch.uint8).fill_(127)
        graph.replay()
        _compare(output, reference())
    torch.cuda.set_sync_debug_mode("error")
    try:
        stage.run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
    names = [event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    kernels = [name for name in names if "cudnn_kernel_frost_sm100_moe" in name]
    assert len(kernels) == 1 and f"cluster{cm}x{cn}_" in kernels[0] and "packed_b_128x128_v1" in kernels[0], names
