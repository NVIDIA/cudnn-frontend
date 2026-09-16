# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Absolute A addressing removes descriptor updates without losing ragged clipping."""

import cudnn
import cutlass.cute as cute
import pytest
import torch

from cudnn.gemm.frost import compiler
from cudnn.gemm.frost.tile_config import by_name
from gemm_test_utils import requires_sm100
from test_moe_grouped_matmul_fwd import _build_graph, _vp_moe

pytestmark = [pytest.mark.L0, requires_sm100]


@pytest.mark.parametrize("sched_policy", [0, 1])
@pytest.mark.parametrize("cta_group", [1, 2])
def test_absolute_a_never_replaces_base_address(monkeypatch, sched_policy, cta_group):
    # Almost every tile crosses a group boundary. Strongly different expert
    # weights expose a missing output clip; include empty and one-row groups.
    sizes = [1, 0, 127, 1]
    rows, width, experts = sum(sizes), 256, len(sizes)
    graph = _build_graph(experts, rows, width, width)
    config = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma" if cta_group == 1 else "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma")
    original_import = compiler._import_kernel
    imported = []

    def import_with_forbidden_update(source):
        module = original_import(source)

        def forbidden(*args, **kwargs):
            raise AssertionError("absolute A still replaces a per-group base address")

        def uncached(fn, *args, cache_key=None, symbol=None, **kwargs):
            return cute.compile(fn, *args, **kwargs)

        module._replace_tensormap_global_address = forbidden
        module._compile_cached = uncached
        module.compile.cache_clear()
        imported.append(module)
        return module

    monkeypatch.setattr(compiler, "_import_kernel", import_with_forbidden_update)
    compiled = compiler.jit_from_cudnn_graph(graph, config, moe_sched_policy=sched_policy)
    assert imported, "The forbidden-update detector must participate in compilation"
    token = torch.ones(1, rows, width, device="cuda", dtype=torch.bfloat16)
    token[:, 1:128].mul_(4)
    weight = torch.empty(experts, width, width, device="cuda", dtype=torch.bfloat16)
    for expert, value in enumerate([0.125, 0.25, -0.5, 2.0]):
        weight[expert].fill_(value)
    offsets = torch.tensor([0, 1, 1, 128], device="cuda", dtype=torch.int32)
    output = torch.full_like(token, float("nan"))
    compiled(_vp_moe(compiled, token, weight, offsets, output))
    reference = torch.empty_like(output)
    for expert, (begin, end) in enumerate([(0, 1), (1, 1), (1, 128), (128, 129)]):
        reference[:, begin:end] = token[:, begin:end].float() @ weight[expert].float().T
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


@pytest.mark.parametrize("epilogue", ["pointwise", "reduction", "quantization"])
def test_absolute_a_eligibility_preserves_cross_row_fusions(epilogue):
    from cudnn.gemm.frost.graph_analyzer import analyze

    kwargs = {}
    if epilogue == "reduction":
        kwargs.update(reduction_mode=cudnn.reduction_mode.AMAX, reduction_dims=(1, 1, 1))
    elif epilogue == "quantization":
        kwargs["quant"] = True
    chain = analyze(_build_graph(4, 129, 256, 256, **kwargs))
    assert compiler._moe_can_use_absolute_a(chain) == (epilogue == "pointwise")
