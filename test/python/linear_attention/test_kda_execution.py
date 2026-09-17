# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native launch-cache and output-layout regressions for the JAX integration."""

import importlib

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("KDA executor tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

import cutlass.cute as cute

import cudnn
from cudnn.frost.workspace import Workspace
from cudnn.linear_attention.frost.kda_engine import build_kda

from .test_la import CUDNN_DTYPE, LEAF_NAMES, make_case, thd_tensors

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost KDA")


def make_plan(total, heads, *, backward, invariant):
    hq, hk, hv = heads
    case = make_case("kda", torch.bfloat16, T=total, H=hq, HK=hk, HV=hv, K=64, V=64)
    values = dict(zip(LEAF_NAMES["kda"], thd_tensors(case)))
    values["cu_seqlens"] = case.cu
    if backward:
        values["dO"] = torch.randn(total, max(hq, hv), 64, dtype=torch.bfloat16, device="cuda")
    graph = cudnn.pygraph()
    tensors = {name: graph.tensor(list(t.shape), data_type=CUDNN_DTYPE[t.dtype], name=name) for name, t in values.items()}
    if backward:
        graph.kda_bwd(**tensors, batch_invariant=invariant)
        sources = dict(dQ="q", dK="k", dV="v", dG="g", dBeta="beta")
    else:
        graph.kda(**tensors, batch_invariant=invariant)
        sources = dict(O="q")
    node = graph.nodes[0]
    outputs = {}
    for name, tensor in node.outputs.items():
        dtype = values[sources[name]].dtype
        tensor.set_output(True).set_data_type(CUDNN_DTYPE[dtype])
        outputs[name] = torch.empty(tuple(tensor.dim), dtype=dtype, device="cuda")
    graph.validate()
    plan = build_kda(graph)
    plan.bind(tuple(values) + tuple(outputs))
    scratch = torch.empty(plan.workspace_bytes(), dtype=torch.uint8, device="cuda")
    workspace = Workspace(scratch, plan.workspace_bytes(), "KDA test")
    return plan, values, outputs, scratch, workspace


def execute(plan, values, outputs, workspace):
    plan.run(tuple(values.values()) + tuple(outputs.values()), workspace, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert all(torch.isfinite(t).all() for t in outputs.values())


@pytest.mark.parametrize("backward", [False, True], ids=["forward", "backward"])
@pytest.mark.parametrize("schedule", ["uncut", "warmup", "chain"])
def test_native_cache_reuses_kda_across_shapes(monkeypatch, backward, schedule):
    family = "chain" if schedule == "chain" else "warmup"
    direction = "backward" if backward else "forward"
    module = importlib.import_module(f"cudnn.linear_attention.frost.kernel.kda_{family}_{direction}_f16")
    monkeypatch.setattr(module, f"{family}_{direction}_cache", {})
    compile_kernel = cute.compile
    compilations = []

    def record_compile(*args, **kwargs):
        compilations.append(args[0])
        return compile_kernel(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", record_compile)
    shapes = [(256, 1), (384, 1), (384, 2), (512, 4)] if schedule == "chain" else [(32, 1), (64, 1), (64, 2), (96, 4)]
    for i, (total, heads) in enumerate(shapes):
        plan, values, outputs, scratch, workspace = make_plan(total, (heads,) * 3, backward=backward, invariant=schedule == "uncut")
        assert plan.chain == (schedule == "chain")
        assert plan.split == (schedule == "warmup")
        execute(plan, values, outputs, workspace)
        if i == 0:
            assert compilations
            first = len(compilations)
        else:
            assert len(compilations) == first, "changing token/head counts must reuse the compiled native host"


@pytest.mark.parametrize("heads", [(1, 1, 2), (2, 2, 1)], ids=["fold_qk", "fold_v"])
@pytest.mark.parametrize("total", [64, 512], ids=["uncut", "chain"])
def test_native_kda_backward_respects_output_strides(total, heads):
    plan, values, outputs, scratch, workspace = make_plan(total, heads, backward=True, invariant=total == 64)
    execute(plan, values, outputs, workspace)
    padded, strided = {}, {}
    for name, tensor in outputs.items():
        padded[name] = torch.full((*tensor.shape[:-1], 2 * tensor.shape[-1]), float("nan"), dtype=tensor.dtype, device="cuda")
        strided[name] = padded[name][..., : tensor.shape[-1]]
    execute(plan, values, strided, workspace)
    for name, reference in outputs.items():
        torch.testing.assert_close(strided[name], reference, rtol=0, atol=0)
        assert torch.isnan(padded[name][..., reference.shape[-1] :]).all(), f"{name} wrote into output padding"
