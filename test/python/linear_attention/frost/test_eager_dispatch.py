# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The eager paths of kimi_delta_attention (EagerKda) and gated_delta_net (EagerGdn) against the custom ops they bypass:
the same outputs, gradients and autograd metadata; a torch function mode keeps the call on the op."""

import contextlib

import pytest
import torch
from torch.overrides import TorchFunctionMode

from cudnn.linear_attention.ops import gated_delta_net, gdn, kda, kimi_delta_attention

from linear_attention.test_la import frost_la_unavailable

requirement_error = frost_la_unavailable()
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

D = 128
# family: (public op, its custom op's name, its eager Function, whether its gate is per key channel)
FAMILIES = {
    "kda": (kimi_delta_attention, "kimi_delta_attention_fwd", "EagerKda", True),
    "gdn": (gated_delta_net, "gated_delta_net_fwd", "EagerGdn", False),
}


class PassThrough(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def inputs(H, preceding, channel_gate):
    gen = torch.Generator().manual_seed(7)
    cu = [0, 700, 2048]
    T, N = cu[-1], len(cu) - 1

    def draw(shape, dtype=torch.float32, scale=1.0, shift=0.0):
        return (torch.randn(shape, generator=gen) * scale + shift).to(dtype).cuda()

    leaves = dict(
        q=draw((T, H, D), torch.bfloat16),
        k=draw((T, H, D), torch.bfloat16),
        v=draw((T, H, D), torch.bfloat16),
        g=draw((T, H, D) if channel_gate else (T, H), shift=-4.0),
        beta=draw((T, H)),
        initial_state=draw((N, H, D, D), scale=0.05),
        a_log=draw((H,), scale=0.1),
        dt_bias=draw((H, D) if channel_gate else (H,), scale=0.1),
    )
    extra = dict(preceding_state_update=draw((2, H, D, D), scale=0.05), preceding_transition=draw((2, H, D, D), scale=0.01)) if preceding else {}
    grads = (draw((T, H, D), torch.bfloat16), draw((N, H, D, D), scale=0.05))
    return leaves, torch.tensor(cu, dtype=torch.int32, device="cuda"), extra, grads


def run(public, leaves, cu, extra, grads, requires_grad):
    leaves = {k: v.detach().requires_grad_(requires_grad) for k, v in leaves.items()}
    res = public(
        *(leaves[k] for k in ("q", "k", "v", "g", "beta")),
        cu,
        initial_state=leaves["initial_state"],
        a_log=leaves["a_log"],
        dt_bias=leaves["dt_bias"],
        output_final_state=True,
        checkpoint_every_n_tokens=64,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        **extra,
    )
    meta = [(t.requires_grad, type(t.grad_fn).__name__ if t.grad_fn is not None else None) for t in res]
    if not requires_grad:
        return res, meta, ()
    gs = torch.autograd.grad(res[:2], list(leaves.values()), grads, allow_unused=True)
    return res, meta, gs


def assert_same(got, want):
    for a, b in zip(got, want, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.shape == b.shape and a.dtype == b.dtype
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("requires_grad", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize("preceding", [False, True], ids=["local", "context_parallel"])
@pytest.mark.parametrize("family", FAMILIES)
def test_eager_path_matches_custom_op(family, preceding, requires_grad):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    public, op_name, eager_name, channel_gate = FAMILIES[family]
    case = inputs(4, preceding, channel_gate)
    with PassThrough():
        want, want_meta, want_grads = run(public, *case, requires_grad)
    got, got_meta, got_grads = run(public, *case, requires_grad)
    ckpt_rows = 11 + 22  # ceil(700 / 64) + ceil(1348 / 64): the series' capacity rows past them are never written
    assert_same([t[:ckpt_rows] if i == 2 else t for i, t in enumerate(got)], [t[:ckpt_rows] if i == 2 else t for i, t in enumerate(want)])
    assert_same(got_grads, want_grads)
    if requires_grad:
        assert want_meta[0][1].startswith(f"GeneratedBackwardFor_cudnn_{op_name}")
        assert got_meta[0][1] == f"{eager_name}Backward"
    assert [m[0] for m in got_meta] == [m[0] for m in want_meta]
    if requires_grad and preceding:
        assert got_grads[5] is None  # under context parallelism initial_state gets no gradient


@pytest.mark.parametrize("family", FAMILIES)
def test_torch_without_the_private_queries_keeps_the_custom_op(family, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    from cudnn.linear_attention.ops import common

    public, op_name, _, channel_gate = FAMILIES[family]
    monkeypatch.setattr(common, "EAGER_PROBES", None)
    _, meta, _ = run(public, *inputs(4, False, channel_gate), True)
    assert meta[0][1].startswith(f"GeneratedBackwardFor_cudnn_{op_name}")


@pytest.mark.parametrize("family", FAMILIES)
def test_double_backward_raises_like_custom_op(family):
    """create_graph=True: the first-order gradients carry the op's graph on both paths, and differentiating them again raises
    the custom op's missing-formula error instead of silently treating the backward as constant."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    public, _, _, channel_gate = FAMILIES[family]
    leaves, cu, _, (do, dfs) = inputs(4, False, channel_gate)
    for custom_op in (True, False):
        xs = {k: v.detach().requires_grad_(True) for k, v in leaves.items()}
        with PassThrough() if custom_op else contextlib.nullcontext():
            o, fs = public(*(xs[k] for k in ("q", "k", "v", "g", "beta")), cu, initial_state=xs["initial_state"], output_final_state=True)[:2]
        gs = torch.autograd.grad([o, fs], [xs[k] for k in ("q", "k", "v", "g", "beta", "initial_state")], [do, dfs], create_graph=True)
        assert all(g.requires_grad for g in gs), custom_op
        with pytest.raises(RuntimeError, match="no autograd formula was registered"):
            torch.autograd.grad(sum(g.float().sum() for g in gs), [xs["q"]])


@pytest.mark.parametrize("family", FAMILIES)
def test_jit_trace_records_the_custom_op(family):
    """torch.jit.trace records dispatcher ops: the eager path would leave the trace with the allocations and no kernels."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    public, op_name, _, channel_gate = FAMILIES[family]
    leaves, cu, _, _ = inputs(4, False, channel_gate)
    args = tuple(leaves[k] for k in ("q", "k", "v", "g", "beta"))
    call = lambda *t: public(*t, cu, use_qk_l2norm_in_kernel=True)[0]  # noqa: E731
    with torch.no_grad():
        traced = torch.jit.trace(call, args, check_trace=False)
        assert f"cudnn::{op_name}" in {node.kind() for node in traced.graph.nodes()}
        assert torch.equal(traced(*args), call(*args))


@pytest.mark.parametrize("family", FAMILIES)
def test_profiler_sees_the_custom_ops(family):
    """Under the autograd profiler the public op takes the custom-op path, so a trace names cudnn::<op>_fwd / _bwd as on the
    op path, with no eager autograd node."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    public, op_name, _, channel_gate = FAMILIES[family]
    leaves, cu, _, (do, _) = inputs(4, False, channel_gate)
    xs = [leaves[k].detach().requires_grad_(True) for k in ("q", "k", "v", "g", "beta")]

    def step():
        o = public(*xs, cu, use_qk_l2norm_in_kernel=True)[0]
        torch.autograd.grad(o, xs, do)

    for _ in range(3):
        step()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        step()
    names = {e.name for e in prof.events()}
    bwd_name = op_name[: -len("_fwd")] + "_bwd"
    assert {f"cudnn::{op_name}", f"cudnn::{bwd_name}"} <= names
    assert not [n for n in names if "Eager" in n]


@pytest.mark.parametrize("family", FAMILIES)
def test_op_and_eager_paths_share_plans(family, monkeypatch):
    """The custom op and the eager path call the planned forward and backward with the same arguments: one plan each."""
    from cudnn import launch_plan

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    if not launch_plan.ENABLED or launch_plan.VERIFY:
        pytest.skip("this test counts plans: CUDNN_FRONTEND_LAUNCH_PLANS is 0 or verify")
    monkeypatch.setattr(launch_plan, "HOT_CALLS", 2)
    public, _, _, channel_gate = FAMILIES[family]
    module = kda if family == "kda" else gdn
    run_fwd, run_bwd = getattr(module, f"run_{family}_fwd"), getattr(module, f"run_{family}_bwd")
    run_fwd.plans.clear()
    run_bwd.plans.clear()
    leaves, cu, _, (do, _) = inputs(4, False, channel_gate)
    xs = [leaves[k].detach().requires_grad_(True) for k in ("q", "k", "v", "g", "beta")]
    for custom_op in (True, True, False, False):
        with PassThrough() if custom_op else contextlib.nullcontext():
            o = public(*xs, cu, use_qk_l2norm_in_kernel=True)[0]
            torch.autograd.grad(o, xs, do)
    assert [type(p).__name__ for p in run_fwd.plans.values()] == ["_Plan"]
    assert [type(p).__name__ for p in run_bwd.plans.values()] == ["_Plan"]
