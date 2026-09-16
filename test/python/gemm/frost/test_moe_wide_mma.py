# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-A MMA packing must retain both expert outputs and live ragged offsets."""

import cudnn
import cutlass.cute as cute
import pytest
import torch

from cudnn.gemm.frost import compiler
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100

pytestmark = [pytest.mark.L0, requires_sm100]


def _graph(dtype=cudnn.data_type.BFLOAT16, b_major="k", shared_a=True, gemms=2, out_dtype=None):
    rows, experts, width = 513, 4, 256
    graph = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = graph.tensor(name="tokens", dim=[1, rows, width], stride=[rows * width, width, 1])
    offsets = graph.tensor(name="offsets", dim=[experts, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.INT32)
    weights, products = [], []
    for index in range(gemms):
        operand = a if shared_a or index == 0 else graph.tensor(name=f"tokens{index}", dim=[1, rows, width], stride=[rows * width, width, 1])
        weight = graph.tensor(
            name=f"weight{index}", dim=[experts, width, width], stride=[width * width, 1, width] if b_major == "k" else [width * width, width, 1]
        )
        weights.append(weight)
        products.append(graph.moe_grouped_matmul(operand, weight, offsets, mode=cudnn.moe_grouped_matmul_mode.NONE))
    out = products[0]
    for product in products[1:]:
        out = graph.mul(a=out, b=graph.swish(input=product))
    out.set_output(True).set_data_type(dtype if out_dtype is None else out_dtype)
    return graph, a, offsets, weights, out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tile", [64, 128])
@pytest.mark.parametrize("sched_policy", [0, 1])
def test_wide_mma_ragged_live_capture(monkeypatch, dtype, tile, sched_policy):
    io = {torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float16: cudnn.data_type.HALF, torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3}[dtype]
    fp8 = dtype == torch.float8_e4m3fn
    out_dtype = torch.bfloat16 if fp8 else dtype
    graph, a, offsets, weights, out = _graph(io, out_dtype=cudnn.data_type.BFLOAT16 if fp8 else io)
    config = by_name(f"CONFIG_sm100_{tile}x{tile}x128_{tile}x{tile}x32_cluster1x4_1ctamma")
    imported, issued = [], []
    original_import = compiler._import_kernel

    def check_wide_module(source):
        module = original_import(source)
        assert getattr(module, "moe_wide_mma", False), "Wide packing was not rendered"
        assert module.mma_inst_shape_mnk[1] == 2 * tile
        assert module.mma_issue_gemms == 1
        assert module.b_stage_multiplier == 2
        original_mma = module._tcgen05_mma

        def checked_mma(*args, **kwargs):
            # The old pair uses A collector FILL/LASTUSE. Require a real
            # compile through this detector, not a cached launch or a tag.
            assert kwargs.get("collector_op") is None, "Legacy paired MMA still issued"
            issued.append(True)
            return original_mma(*args, **kwargs)

        def uncached(fn, *args, cache_key=None, symbol=None, **kwargs):
            return cute.compile(fn, *args, **kwargs)

        module._tcgen05_mma = checked_mma
        module._compile_cached = uncached
        module.compile.cache_clear()
        imported.append(module)
        return module

    monkeypatch.setattr(compiler, "_import_kernel", check_wide_module)
    compiled = compiler.jit_from_cudnn_graph(graph, config, moe_sched_policy=sched_policy)
    assert imported and issued
    torch.manual_seed(5512)
    generation_dtype = torch.bfloat16 if fp8 else dtype
    x = (torch.randn(1, 513, 256, device="cuda", dtype=generation_dtype) * 0.125).to(dtype)
    ws = [(torch.randn(4, 256, 256, device="cuda", dtype=generation_dtype) * 0.125).to(dtype) for _ in weights]
    bounds = [0, 1, 1, 512, 513]
    off = torch.tensor(bounds[:-1], device="cuda", dtype=torch.int32)
    output = torch.full_like(x, float("nan"), dtype=out_dtype)
    pack = {a: x, offsets: off, out: output, **dict(zip(weights, ws))}

    def reference():
        result = torch.empty_like(output)
        for expert, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            up = x[:, begin:end].float() @ ws[0][expert].float().T
            gate = x[:, begin:end].float() @ ws[1][expert].float().T
            result[:, begin:end] = up * torch.nn.functional.silu(gate)
        return result

    compiled(pack, stream=torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(output, reference(), rtol=0.02, atol=0.02)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=side):
        compiled(pack, stream=side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)
    previous = output.clone()
    for repeat in range(3):
        if fp8:
            x.copy_((-x.float()).to(dtype))
            ws[1].copy_((ws[1].float() * 0.75).to(dtype))
        else:
            x.neg_()
            ws[1].mul_(0.75)
        bounds = [0, 0, 127, 128, 513]
        off.copy_(torch.tensor(bounds[:-1], device="cuda", dtype=torch.int32))
        output.fill_(float("nan"))
        capture.replay()
        expected = reference()
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        assert (output.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12) <= 0.01
        if repeat == 0:
            assert not torch.equal(previous, output)


@pytest.mark.parametrize("case", ["eligible", "n_major", "distinct_a", "three_gemms", "single_gemm", "two_cta", "split_m", "wide_n", "fp8_e4m3", "fp8_e5m2"])
def test_wide_mma_eligibility_preserves_operand_layouts(case):
    options = {}
    config_name = "CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma"
    if case == "n_major":
        options["b_major"] = "n"
    elif case == "distinct_a":
        options["shared_a"] = False
    elif case in ("three_gemms", "single_gemm"):
        options["gemms"] = 3 if case == "three_gemms" else 1
    elif case == "two_cta":
        config_name = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
    elif case == "split_m":
        config_name = "CONFIG_sm100_128x64x128_64x64x32_cluster1x4_1ctamma"
    elif case == "wide_n":
        config_name = "CONFIG_sm100_128x256x128_128x256x32_cluster1x4_1ctamma"
    elif case in ("fp8_e4m3", "fp8_e5m2"):
        options["dtype"] = cudnn.data_type.FP8_E4M3 if case == "fp8_e4m3" else cudnn.data_type.FP8_E5M2
    chain = analyze(_graph(**options)[0])
    assert compiler._moe_can_use_wide_mma(chain, by_name(config_name)) == (case in ("eligible", "fp8_e4m3"))
