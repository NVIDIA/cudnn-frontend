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

from linear_attention.test_la import Backend, CUDNN_DTYPE, LEAF_NAMES, assert_bwd_parity, make_case, thd_tensors

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


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inputs", ["ordinary", "small_norm", "cancellation"])
def test_kda_bwd_l2norm_reference(dim, dtype, inputs):
    """Compare output and every input gradient to the suite's FP64 recurrence."""
    case = make_case("kda", dtype, T=65, H=2, K=dim, V=dim, seed=418)
    if inputs == "small_norm":
        case.q[:, ::3] *= 1e-6
        case.k[:, 1::3] *= 1e-6
    elif inputs == "cancellation":
        # Correlated keys and opposite values stress the L2 projection's subtraction.
        case.k.copy_(case.k[:, :1].clone().expand_as(case.k))
        case.v[:, 1::2].copy_(-case.v[:, :-1:2])
    assert_bwd_parity(Backend("frost"), case, l2norm=True)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("checkpoint", [0, 16])
def test_uncut_bwd_raw_ring_wraps_and_changed_input_replay(monkeypatch, dim, dtype, checkpoint):
    """Five chunks cover raw-ring phase wraps, a tail and fresh replay operands."""
    from cudnn.linear_attention.ops import kda as ops

    monkeypatch.setattr(ops, "bprop_cache", {})
    generator = torch.Generator().manual_seed(418)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        shape = (65, 1, dim)
        q, k, v = [torch.randn(shape, dtype=dtype, generator=generator).cuda().requires_grad_(True) for _ in range(3)]
        g = (torch.randn(shape, generator=generator) - 4).cuda().requires_grad_(True)
        beta = torch.randn(shape[:2], generator=generator).cuda().requires_grad_(True)
        cu = torch.tensor([0, shape[0]], device="cuda", dtype=torch.int32)
        do = torch.randn(shape, dtype=dtype, generator=generator).cuda()
        leaves = [q, k, v, g, beta]
        options = dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True, batch_invariant=True, plan_name="kda_frost")
        if checkpoint == 0:
            state = torch.randn((1, 1, dim, dim), generator=generator).cuda().requires_grad_(True)
            a_log = torch.zeros(1, device="cuda", requires_grad=True)
            dt_bias = torch.zeros((1, dim), device="cuda", requires_grad=True)
            options.update(initial_state=state, a_log=a_log, dt_bias=dt_bias)
            leaves.extend((state, a_log, dt_bias))

        def call():
            """Use the public autograd path, including checkpoint/recompute work."""
            output = kimi_delta_attention(q, k, v, g, beta, cu, checkpoint_every_n_tokens=checkpoint, **options)[0]
            return torch.autograd.grad(output, leaves, do)

        call()
        baseline = call()
        assert len(ops.bprop_cache) == 1
        host_graph, _ = next(iter(ops.bprop_cache.values()))
        assert not host_graph._compiled_plans[host_graph._plan_index].compiled.chain
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                actual = call()
            for _ in range(2):
                graph.replay()
                for got, want in zip(actual, baseline, strict=True):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            with torch.no_grad():
                v.add_(0.25)
            baseline = call()
            graph.replay()
            for got, want in zip(actual, baseline, strict=True):
                assert torch.isfinite(got).all()
                torch.testing.assert_close(got, want, rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("schedule", ["uncut", "chain"])
def test_kda_backward_only_changed_input_replay(monkeypatch, dim, dtype, optional, schedule):
    """Exercise both backward schedules without forward checkpoint generation.

    Chain still builds its M summary through recompute. Sanitizer reports from
    that auxiliary kernel must remain visible, not counted as a clean backward.
    """
    from cudnn.linear_attention.ops import kda as ops

    monkeypatch.setattr(ops, "bprop_cache", {})
    generator = torch.Generator().manual_seed(419)
    shape = (65 if schedule == "uncut" else 513, 2, dim)
    q, k, v, do = [(torch.randn(shape, generator=generator) * 0.1).to(dtype).cuda() for _ in range(4)]
    g = torch.full(shape, -4.0, device="cuda")
    beta = torch.zeros(shape[:2], device="cuda")
    cu = torch.tensor([0, shape[0]], device="cuda", dtype=torch.int32)
    checkpoints = (torch.randn(shape[0] // 16 + 1, 2, dim, dim, generator=generator) * 0.01).to(dtype).cuda()
    options = dict(
        state_checkpoints=checkpoints,
        checkpoint_every_n_tokens=16,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        batch_invariant=schedule == "uncut",
        plan_name="kda_frost",
    )
    if optional:
        state = torch.zeros((1, 2, dim, dim), device="cuda")
        options.update(
            initial_state=state, d_final_state=torch.full_like(state, 0.125), a_log=torch.zeros(2, device="cuda"), dt_bias=torch.zeros(2, dim, device="cuda")
        )

    def call():
        """Execute backward with caller checkpoints and the current live input contents."""
        return ops.kda_bwd(do, q, k, v, g, beta, cu, dim**-0.5, **options)

    call()
    assert len(ops.bprop_cache) == 1
    host_graph, _ = next(iter(ops.bprop_cache.values()))
    assert host_graph._compiled_plans[host_graph._plan_index].compiled.chain == (schedule == "chain")
    expected = tuple(t.clone() for t in call())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            actual = call()
        graph.replay()
        for got, want in zip(actual, expected, strict=True):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        do.add_(0.125)
        v.add_(0.25)
        expected = tuple(t.clone() for t in call())
        graph.replay()
        for got, want in zip(actual, expected, strict=True):
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("optional", [False, True])
@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("checkpoint", [0, 32])
def test_kda_warm_chain_binds_current_operands(optional, safe_gate, checkpoint):
    """Prepared indices preserve the cold launch ABI and never retain buffers."""
    from types import SimpleNamespace
    from cuda.bindings.driver import CUstream
    from cudnn.linear_attention.frost.kda_engine import CompiledKda
    from cudnn.linear_attention.frost.kernel.kda_chain_forward_f16 import CHAIN_FORWARD_BUFFERS, run_chain_forward

    plan = CompiledKda.__new__(CompiledKda)
    plan.chain = True
    plan.safe_gate = safe_gate
    plan.has_state_checkpoints = optional
    plan.checkpoint = checkpoint
    plan.pieces, plan.n_heads_out, plan.num_seqs, plan.scale = 4, 2, 1, 0.125
    plan.stream_type = CUstream
    plan.chain_buffer_names = CHAIN_FORWARD_BUFFERS
    plan.carve_names = (
        "scheduler_all",
        "scheduler_h",
        "scheduler_prefill",
        "work_items",
        "work_items_summary",
        "main_rows",
        "summary_rows",
        "cu_pieces",
        "main_count",
        "summary_count",
        "state_h",
        "state_m",
        "state_x",
        "fused_tensormaps",
        "tensormaps",
    )
    names = ("q", "k", "v", "g", "beta", "cu_seqlens", "O")
    if optional:
        names += ("initial_state", "final_state", "state_indices", "state_checkpoints", "a_log", "dt_bias")
    plan.bind(names)
    calls = []
    plan.chain_launch = lambda *args: calls.append(args)
    plan.run_chain_forward = run_chain_forward
    plan.carve = object()
    for stream in (0, 17):
        inputs = dict.fromkeys(names)
        inputs.update({name: object() for name in names})
        region = {name: object() for name in plan.carve_names}
        workspace = SimpleNamespace(carve=lambda _: tuple(region.values()))
        plan.run_chain(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["cu_seqlens"],
            inputs.get("initial_state"),
            inputs["O"],
            inputs.get("final_state"),
            inputs.get("state_checkpoints"),
            inputs.get("a_log"),
            inputs.get("dt_bias"),
            inputs.get("state_indices"),
            region,
            stream,
        )
        plan.run(tuple(inputs.values()), workspace, stream)
        cold, warm = calls[-2:]
        assert cold[:-1] == warm[:-1]
        assert int(cold[-1]) == int(warm[-1]) == stream


def test_kda_warm_chain_new_workspace_stream_and_capture():
    """The warm launch uses new storage and captures on the caller's stream."""
    plan, values, outputs, scratch, workspace = make_plan(512, (2, 2, 2), backward=False, invariant=False)
    assert plan.chain
    execute(plan, values, outputs, workspace)
    values = {name: value.clone() for name, value in values.items()}
    values["v"].add_(0.25)
    actual = {name: torch.empty_like(value) for name, value in outputs.items()}
    new_scratch = torch.empty_like(scratch)
    new_workspace = Workspace(new_scratch, plan.workspace_bytes(), "KDA warm test")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    captured = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            execute(plan, values, actual, new_workspace)
            with torch.cuda.graph(captured, stream=stream):
                plan.run(tuple(values.values()) + tuple(actual.values()), new_workspace, stream.cuda_stream)
            captured.replay()
        stream.synchronize()
        plan.chain_launch = None
        execute(plan, values, outputs, workspace)
        for name, expected in outputs.items():
            torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
    finally:
        stream.synchronize()
        captured.reset()
