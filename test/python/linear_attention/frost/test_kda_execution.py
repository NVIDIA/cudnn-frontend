# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native KDA execution, launch-cache, and output-layout regressions."""

import importlib
import math

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
from cudnn.linear_attention.ops import kimi_delta_attention

from linear_attention.test_la import CUDNN_DTYPE, LEAF_NAMES, make_case, thd_tensors

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture(autouse=True)
def blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost KDA")


def make_plan(total, heads, *, backward, invariant):
    """Build and bind a native KDA forward or backward execution plan."""
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
@pytest.mark.parametrize("schedule", ["uncut", "chain"])
def test_frost_native_cache_reuses_kda_across_shapes(monkeypatch, backward, schedule):
    """One compiled native host is reused across dynamic token/head counts."""
    family = schedule
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
        execute(plan, values, outputs, workspace)
        if i == 0:
            assert compilations
            first = len(compilations)
        else:
            assert len(compilations) == first, "changing token/head counts must reuse the compiled native host"


def test_kda_retains_history_under_tiny_channel_gates():
    """KDA keeps state that a per-channel gate product would call decayed.

    With two equal live key channels, choosing alpha so that
    ``alpha * (1 - ||k||^2) == -1`` preserves the state component parallel to
    k even though every individual channel has ``alpha**16 < exp(-10)``: the
    rank-one update routes the state around the decay.  An impulse 17 tokens
    before the checked window must remain visible there.
    """
    total, heads, dim = 3840, 96, 64
    cut, impulse = 1280, 1263
    live = 1.2265625  # exactly representable in bfloat16
    alpha = 1.0 / (2.0 * live * live - 1.0)
    log_alpha = float(torch.tensor(math.log(alpha), dtype=torch.float32))

    q = torch.zeros(total, heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    q[..., :2] = live
    k[..., :2] = live
    v = torch.zeros_like(q)
    v[impulse, :, 0] = 1.0
    g = torch.full((total, heads, dim), log_alpha, dtype=torch.float32, device="cuda")
    beta = torch.ones(total, heads, dtype=torch.float32, device="cuda")
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")

    actual, _ = kimi_delta_attention(q, k, v, g, beta, cu, plan_name="kda_frost")

    # Independent recurrence over the only two live key channels.  Use the
    # rounded float32 gate value consumed by the kernel, not ideal alpha.
    rounded_alpha = math.exp(log_alpha)
    state = [0.0, 0.0]
    expected = []
    scale = 1.0 / math.sqrt(dim)
    for token in range(total):
        state[0] *= rounded_alpha
        state[1] *= rounded_alpha
        residual = (1.0 if token == impulse else 0.0) - live * (state[0] + state[1])
        state[0] += live * residual
        state[1] += live * residual
        if cut <= token < cut + 16:
            expected.append(scale * live * (state[0] + state[1]))

    expected = torch.tensor(expected, dtype=torch.float64, device="cuda")[:, None]
    observed = actual[cut : cut + 16, :, 0].double()
    relative_l2 = ((observed - expected).square().mean() / expected.square().mean()).sqrt().item()
    assert relative_l2 < 0.03, f"history under tiny channel gates was lost: relative L2 {relative_l2:.5f}"
    assert alpha**16 < math.exp(-10), "every fixed channel's gate product over a chunk must be negligible"


@pytest.mark.parametrize("heads", [(1, 1, 2), (2, 2, 1)], ids=["fold_qk", "fold_v"])
@pytest.mark.parametrize("total", [64, 512], ids=["uncut", "chain"])
def test_frost_native_kda_backward_respects_output_strides(total, heads):
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
