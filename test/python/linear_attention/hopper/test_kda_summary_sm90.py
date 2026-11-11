# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 span summaries must not be inferred by subtracting BF16-MMA forwards."""

import pytest
import torch

import cudnn
from cudnn.linear_attention import kimi_delta_attention_summary, kimi_delta_attention_summary_bwd
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot

ENGINE = "kda_summary_hopper_cuda"
pytestmark = [pytest.mark.L0, pytest.mark.skipif(torch.cuda.get_device_capability() != (9, 0), reason="SM90 summary")]


def make_case(lengths=(17, 0, 65), heads=2, seed=713, gate=-0.001):
    rng = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    q, k = [torch.nn.functional.normalize(torch.randn(total, heads, 128, generator=rng), dim=-1).bfloat16().cuda() for _ in range(2)]
    v, do = [torch.randn(total, heads, 128, generator=rng).bfloat16().cuda() for _ in range(2)]
    g = torch.full((total, heads, 128), gate, device="cuda")
    beta = torch.full((total, heads), 0.5, device="cuda")
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    state = (torch.randn(len(lengths), heads, 128, 128, generator=rng) * 0.1).cuda()
    return dict(q=q, k=k, v=v, g=g, beta=beta, dO=do, cu=cu, state=state)


def run(case, backward=False, seeded=False, **kwargs):
    x = case
    if backward:
        return kimi_delta_attention_summary_bwd(
            x["q"],
            x["k"],
            x["g"],
            x["beta"],
            x["dO"],
            x["cu"],
            d_final_state=x["state"] if seeded else None,
            output_transition=True,
            plan_name=ENGINE,
            **kwargs,
        )
    return kimi_delta_attention_summary(
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        x["cu"],
        initial_state=x["state"] if seeded else None,
        plan_name=ENGINE,
        **kwargs,
    )


def reference(case, backward=False, seeded=False):
    """Independent FP64 recurrence; CPU input construction also fixes golden inputs."""
    x = {name: value.double() for name, value in case.items() if name != "cu"}
    bounds = case["cu"].tolist()
    states, transitions = [], []
    h = x["k"].shape[1]
    for n, (start, end) in enumerate(zip(bounds, bounds[1:])):
        state = x["state"][n].clone() if seeded else torch.zeros(h, 128, 128, device="cuda", dtype=torch.float64)
        transition = torch.eye(128, device="cuda", dtype=torch.float64).expand(h, -1, -1).clone()
        for t in range(end - 1, start - 1, -1) if backward else range(start, end):
            k, decay, beta = x["k"][t], x["g"][t].exp(), x["beta"][t]
            if backward:
                state = state + x["dO"][t, :, :, None] * x["q"][t, :, None, :] * 128**-0.5
                state = (state - beta[:, None, None] * (state * k[:, None, :]).sum(-1, keepdim=True) * k[:, None, :]) * decay[:, None, :]
                transition = (transition - beta[:, None, None] * (transition * k[:, None, :]).sum(-1, keepdim=True) * k[:, None, :]) * decay[:, None, :]
            else:
                state = state * decay[:, None, :]
                residual = x["v"][t] - (state * k[:, None, :]).sum(-1)
                state = state + beta[:, None, None] * residual[:, :, None] * k[:, None, :]
                transition = transition * decay[:, None, :]
                transition = transition - beta[:, None, None] * (transition * k[:, None, :]).sum(-1, keepdim=True) * k[:, None, :]
        states.append(state)
        transitions.append(transition)
    return torch.stack(states), torch.stack(transitions)


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize("seeded", [False, True], ids=["zero", "seed"])
@pytest.mark.parametrize("gate", [-0.001, -5.0], ids=["weak-decay", "strong-decay"])
def test_summary_matches_fp64(backward, seeded, gate):
    x = make_case(gate=gate)
    actual = run(x, backward, seeded)
    expected = reference(x, backward, seeded)
    for got, ref in zip(actual, expected):
        assert got.dtype == torch.float32
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.double(), ref, atol=2e-6, rtol=3e-5)


def test_transition_is_independent_of_values_and_seed():
    x = make_case(lengths=(1024,), heads=1)
    _, transition = run(x)
    other = dict(x, v=x["v"] * 128)
    _, changed = run(other, seeded=True)
    assert torch.equal(transition.view(torch.uint8), changed.view(torch.uint8))
    _, ref = reference(x)
    torch.testing.assert_close(transition.double(), ref, atol=2e-6, rtol=5e-5)


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_cp64_composition(backward):
    # 64 logical spans, not a claim of a 64-GPU communication experiment.
    x = make_case(lengths=(17,) * 64, heads=1)
    state, transition = run(x, backward)
    merged = x["state"][0].double()
    for n in range(63, -1, -1) if backward else range(64):
        merged = merged @ transition[n].double() + state[n].double()
    whole = dict(x, cu=torch.tensor([0, 17 * 64], device="cuda", dtype=torch.int32), state=x["state"][:1])
    expected, _ = reference(whole, backward, seeded=True)
    torch.testing.assert_close(merged, expected[0], atol=3e-6, rtol=8e-5)


def outputs(values):
    return dict(state=values[0], transition=values[1])


@pytest.mark.parametrize("backward", [False, True], ids=["fwd-bf16-K128", "bwd-bf16-K128"])
def test_summary_determinism(backward):
    target = make_case(lengths=(33,), heads=2)
    before = {name: value.clone() for name, value in target.items()}
    expected = assert_replays(
        lambda: outputs(run({name: value.clone() for name, value in target.items()}, backward, seeded=True, batch_invariant=True)), repeats=4
    )
    reference_values = reference(target, backward, seeded=True)
    for got, ref in zip(expected.values(), reference_values):
        torch.testing.assert_close(got.double(), ref, atol=2e-6, rtol=3e-5)
    # Move and duplicate the same sequence, including an empty neighbor and
    # enough rows/heads to schedule many independent CTAs. Only neighbors vary.
    for order in ((0, 1, 2, 0), (2, 0, 1, 0), tuple([1] * 20 + [0, 2, 0])):
        cases = [target, make_case((17,), seed=121), make_case((0,), seed=717)]
        packed = {key: torch.cat([cases[i][key] for i in order]) for key in target if key != "cu"}
        lengths = [cases[i]["k"].shape[0] for i in order]
        packed["cu"] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        got = snapshot(outputs(run(packed, backward, seeded=True, batch_invariant=True)))
        for n, i in enumerate(order):
            if i == 0:
                assert_bitwise({name: value[n : n + 1] for name, value in got.items()}, expected, "batch packing")
    assert_bitwise(target, before, "input mutation")
    assert_golden(f"kda_summary_sm90_{'bwd' if backward else 'fwd'}_bf16", target, expected)


def make_graph(x, backward=False, transition=True):
    graph = cudnn.pygraph()
    names = ("q", "k", "g", "beta", "dO") if backward else ("k", "v", "g", "beta")
    names += ("cu", "state")
    tensors = {name: graph.tensor_like(x[name]) for name in names}
    args = {name: tensors[name] for name in names if name not in ("cu", "state")}
    args["cu_seqlens"] = tensors["cu"]
    args["d_final_state" if backward else "initial_state"] = tensors["state"]
    if backward:
        state_t, transition_t = graph.kda_summary_bwd(**args, output_transition=transition, scale=128**-0.5)
    else:
        state_t, transition_t = graph.kda_summary(**args, output_transition=transition)
    state_t.set_data_type(cudnn.data_type.FLOAT).set_output(True)
    if transition_t is not None:
        transition_t.set_data_type(cudnn.data_type.FLOAT).set_output(True)
    from cudnn.linear_attention.ops.common import select_plan

    select_plan(graph, ENGINE)
    graph.build_plans()
    state = torch.empty_like(x["state"])
    trans = torch.empty_like(state) if transition else None
    bindings = {tensors[name]: x[name] for name in names}
    bindings[state_t] = state
    if transition:
        bindings[transition_t] = trans
    assert graph.get_workspace_size() == 0
    return graph, tensors, state_t, transition_t, bindings, state, trans


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64], ids=["cu32", "cu64"])
def test_strided_rebind_capture(backward, cu_dtype, cudnn_handle, monkeypatch):
    x = make_case()
    # Padded outer dimensions, including state and token-major fused slices.
    for name, value in list(x.items()):
        if name == "cu":
            x[name] = value.to(cu_dtype)
        else:
            shape = list(value.shape)
            shape[1] *= 2
            wide = torch.empty(shape, device="cuda", dtype=value.dtype)
            view = wide[:, : value.shape[1]]
            view.copy_(value)
            x[name] = view
    graph, tensors, state_t, transition_t, bindings, state, trans = make_graph(x, backward)
    padding = []
    for tensor, value in ((state_t, state), (transition_t, trans)):
        shape = list(value.shape)
        shape[1] *= 2
        storage = torch.full(shape, float("nan"), device="cuda")
        bindings[tensor] = storage[:, : value.shape[1]]
        padding.append(storage[:, value.shape[1] :])
    state, trans = bindings[state_t], bindings[transition_t]
    from cudnn.linear_attention.cake import compiler

    def no_compile(*args, **kwargs):
        raise AssertionError("execute must not compile")

    monkeypatch.setattr(compiler, "compile_cubin", no_compile)
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
    saved_stream = cudnn.get_stream(cudnn_handle)
    side = torch.cuda.Stream()
    capture = torch.cuda.CUDAGraph()
    try:
        cudnn.set_stream(cudnn_handle, side.cuda_stream)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            graph.execute(bindings, workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(side)
        for got, ref in zip((state, trans), reference(x, backward, seeded=True)):
            torch.testing.assert_close(got.double(), ref, atol=2e-6, rtol=3e-5)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(capture, stream=side):
            graph.execute(bindings, workspace, handle=cudnn_handle)
        for bounds in ([0, 0, 19, 82], [0, 33, 82, 82]):
            x["cu"].copy_(torch.tensor(bounds, device="cuda", dtype=cu_dtype))
            x["v"].mul_(1.25)
            x["dO"].mul_(0.75)
            x["g"].mul_(2)
            state.fill_(float("nan"))
            trans.fill_(float("nan"))
            capture.replay()
            for got, ref in zip((state, trans), reference(x, backward, seeded=True)):
                torch.testing.assert_close(got.double(), ref, atol=2e-6, rtol=3e-5)
        # A fresh set of addresses uses the same compiled kernel/plan.
        rebound = {tensor: value.clone() for tensor, value in bindings.items()}
        side.wait_stream(torch.cuda.current_stream())
        graph.execute(rebound, workspace, handle=cudnn_handle)
        torch.cuda.current_stream().wait_stream(side)
        assert_bitwise(dict(state=rebound[state_t], transition=rebound[transition_t]), dict(state=state, transition=trans))
        assert all(torch.isnan(guard).all() for guard in padding)
    finally:
        capture.reset()
        cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("backward", [False, True], ids=["fwd", "bwd"])
def test_state_only_empty_and_optional_transition(backward, cudnn_handle):
    for lengths in ((0, 0), (1, 0, 16)):
        x = make_case(lengths)
        graph, _, _, _, bindings, state, _ = make_graph(x, backward, transition=False)
        workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
        saved_stream = cudnn.get_stream(cudnn_handle)
        try:
            cudnn.set_stream(cudnn_handle, torch.cuda.current_stream().cuda_stream)
            graph.execute(bindings, workspace, handle=cudnn_handle)
            ref, _ = reference(x, backward, seeded=True)
            torch.testing.assert_close(state.double(), ref, atol=2e-6, rtol=3e-5)
        finally:
            cudnn.set_stream(cudnn_handle, saved_stream)


@pytest.mark.parametrize("option", ["use_qk_l2norm", "safe_gate", "use_beta_sigmoid", "allow_neg_eigval", "beta_guard", "output_context", "has_state_indices"])
def test_unsupported_features_decline(option):
    from dataclasses import replace
    from cudnn.linear_attention.graph_analyzer import analyze
    from cudnn.linear_attention.hopper.summary_engine import check_contract

    graph, *_ = make_graph(make_case())
    facts = replace(analyze(graph), **{option: True})
    with pytest.raises(NotImplementedError, match="kda_summary_hopper_cuda"):
        check_contract(facts, next(iter(graph.nodes)))


@pytest.mark.parametrize("bad_binding", ["short", "alias", "overlap", "inner_stride", "wrong_device"])
def test_invalid_binding_rejects_before_launch(bad_binding, cudnn_handle, monkeypatch):
    x = make_case()
    graph, tensors, state_t, _, bindings, state, trans = make_graph(x)
    state.fill_(123)
    trans.fill_(456)
    if bad_binding == "short":
        bindings[tensors["k"]] = x["k"][:-1]
    elif bad_binding == "alias":
        bindings[state_t] = x["state"]
    elif bad_binding == "overlap":
        bindings[state_t] = state[:1].expand_as(state)
    elif bad_binding == "inner_stride":
        bindings[tensors["k"]] = torch.empty((*x["k"].shape[:-1], 256), device="cuda", dtype=torch.bfloat16)[..., ::2]
    else:
        if torch.cuda.device_count() < 2:
            pytest.skip("requires a second device for the negative binding test")
        bindings[tensors["k"]] = x["k"].to("cuda:1")
    from cudnn.linear_attention.cake import compiler

    def no_launch(*args, **kwargs):
        raise AssertionError("invalid binding reached CUDA")

    monkeypatch.setattr(compiler, "launch", no_launch)
    with pytest.raises((ValueError, RuntimeError), match="shape|overlap|stride|cuda:|device|storage"):
        graph.execute(bindings, torch.empty(0, dtype=torch.uint8, device="cuda"), handle=cudnn_handle)
    assert (state == 123).all() and (trans == 456).all()


def test_compilation_is_independent_of_lengths_and_direction(monkeypatch):
    from cudnn.linear_attention.cake import compiler
    from cudnn.linear_attention.hopper import summary_engine

    libraries = []
    original = compiler.KernelLibrary

    def counted(*args, **kwargs):
        library = original(*args, **kwargs)
        libraries.append(library)
        return library

    monkeypatch.setattr(summary_engine, "_LIBRARIES", {})
    monkeypatch.setattr(compiler, "KernelLibrary", counted)
    for lengths, backward, cu_dtype in (((7,), False, torch.int32), ((3, 8), True, torch.int64), ((1,) * 65, False, torch.int32)):
        x = make_case(lengths, heads=3)
        x["cu"] = x["cu"].to(cu_dtype)
        make_graph(x, backward)
    assert len(libraries) == 1


def test_default_summary_uses_native_sm90_plan():
    x = make_case()
    actual = kimi_delta_attention_summary(x["k"], x["v"], x["g"], x["beta"], x["cu"])
    assert_bitwise(outputs(actual), outputs(run(x)))


def test_nvrtc_precision_policy_has_separate_cache_entries(monkeypatch, tmp_path):
    from pathlib import Path
    from cudnn.linear_attention.cake import compiler
    from cudnn.linear_attention.hopper import summary_engine

    calls = []
    original = compiler.nvrtc.nvrtcCompileProgram

    def compile_program(program, count, options):
        calls.append(tuple(options))
        return original(program, count, options)

    monkeypatch.setattr(compiler, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(compiler.nvrtc, "nvrtcCompileProgram", compile_program)
    source = Path(summary_engine.__file__).with_name("cuda_kernels") / "kda_summary_sm90.cu"
    accurate = compiler.compile_cubin(source, "sm_90", fast_math=False)
    compiler.compile_cubin(source, "sm_90")
    assert compiler.compile_cubin(source, "sm_90", fast_math=False) == accurate
    assert len(calls) == 2
    assert b"--use_fast_math" not in calls[0]
    assert b"--use_fast_math" in calls[1]


def test_backward_summary_matches_fp64_autograd():
    x = make_case((17,), heads=1)
    seed = x["state"].double().requires_grad_()
    s = seed[0]
    loss = seed.sum() * 0
    for t in range(17):
        k = x["k"][t].double()
        s = s * x["g"][t].double().exp()[:, None, :]
        residual = x["v"][t].double() - (s * k[:, None, :]).sum(-1)
        s = s + x["beta"][t, :, None, None].double() * residual[:, :, None] * k[:, None, :]
        out = (s * x["q"][t, :, None, :].double()).sum(-1) * 128**-0.5
        loss = loss + (out * x["dO"][t].double()).sum()
    loss = loss + (s * x["state"][0].double()).sum()
    (expected,) = torch.autograd.grad(loss, seed)
    actual, _ = run(x, backward=True, seeded=True)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=3e-5)
