# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The eager paths of the linear-attention ops (EagerKda, EagerGdn, EagerGdn2, EagerGdp, EagerMamba2, and the summary ops'
bodies) against the custom ops they bypass: the same outputs, gradients and autograd metadata; a torch function mode
keeps the call on the op."""

import contextlib

import pytest
import torch
from torch.overrides import TorchFunctionMode

from cudnn.linear_attention import ops as la_ops
from cudnn.linear_attention.ops import gated_delta_net, gdn, kda, kimi_delta_attention

from linear_attention import test_la as la
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
def test_op_and_eager_paths_share_recordings(family, monkeypatch):
    """The custom op and the eager path call the forward and backward with the same arguments: one recording each."""
    from cudnn import cpp_launcher

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    if not cpp_launcher.ENABLED or cpp_launcher.VERIFY:
        pytest.skip("this test counts recordings: CUDNN_FRONTEND_CPP_LAUNCHER is 0 or verify")
    monkeypatch.setattr(cpp_launcher, "HOT_CALLS", 2)
    public, _, _, channel_gate = FAMILIES[family]
    module = kda if family == "kda" else gdn
    run_fwd, run_bwd = getattr(module, f"run_{family}_fwd"), getattr(module, f"run_{family}_bwd")
    run_fwd.recordings.clear()
    run_bwd.recordings.clear()
    leaves, cu, _, (do, _) = inputs(4, False, channel_gate)
    xs = [leaves[k].detach().requires_grad_(True) for k in ("q", "k", "v", "g", "beta")]
    for custom_op in (True, True, False, False):
        with PassThrough() if custom_op else contextlib.nullcontext():
            o = public(*xs, cu, use_qk_l2norm_in_kernel=True)[0]
            torch.autograd.grad(o, xs, do)
    assert [type(p).__name__ for p in run_fwd.recordings.values()] == ["_Recording"]
    assert [type(p).__name__ for p in run_bwd.recordings.values()] == ["_Recording"]


# family: (its forward custom op's name, its eager Function), for the families driven through the suite's cases
OTHERS = {
    "gdn2": ("gated_delta_net_v2_fwd", "EagerGdn2"),
    "gdp": ("gated_delta_product_fwd", "EagerGdp"),
    "mamba2": ("mamba2_fwd", "EagerMamba2"),
}


def suite_step(family):
    """requires_grad -> (outputs, their autograd metadata, the leaves' gradients) of the family's op on one fixed case."""
    la.set_seed(7)
    case = la.make_case(family, torch.bfloat16, T=512, H=4)
    args = la.op_args(case)
    n = len(la.LEAF_NAMES[family])
    d_o = (torch.randn(la.op(family)(*args, plan_name=f"{family}_frost")[0].shape, device="cuda") * 0.1).to(torch.bfloat16)

    def step(requires_grad, create_graph=False):
        leaves = [a.detach().requires_grad_(requires_grad) for a in args[:n]]
        res = la.op(family)(*leaves, *args[n:], plan_name=f"{family}_frost")
        meta = [(t.requires_grad, type(t.grad_fn).__name__ if t.grad_fn is not None else None) for t in res]
        if not requires_grad:
            return res, meta, (), leaves
        return res, meta, torch.autograd.grad(res[0], leaves, d_o, allow_unused=True, create_graph=create_graph), leaves

    return step


@pytest.mark.parametrize("requires_grad", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize("family", OTHERS)
def test_eager_path_matches_custom_op_on_the_suite_cases(family, requires_grad):
    op_name, eager_name = OTHERS[family]
    step = suite_step(family)
    with PassThrough():
        want, want_meta, want_grads, _ = step(requires_grad)
    got, got_meta, got_grads, _ = step(requires_grad)
    assert_same(got, want)
    assert_same(got_grads, want_grads)
    kinds = {m[1] for m in want_meta if m[1]}, {m[1] for m in got_meta if m[1]}
    if requires_grad:
        assert any(kind.startswith(f"GeneratedBackwardFor_cudnn_{op_name}") for kind in kinds[0]) and f"{eager_name}Backward" in kinds[1]
    else:
        assert kinds == (set(), set())
    assert [m[0] for m in got_meta] == [m[0] for m in want_meta]


@pytest.mark.parametrize("family", OTHERS)
def test_the_suite_cases_keep_the_custom_op_without_the_private_queries(family, monkeypatch):
    from cudnn.linear_attention.ops import common

    monkeypatch.setattr(common, "EAGER_PROBES", None)
    _, meta, _, _ = suite_step(family)(True)
    assert any(m[1] and m[1].startswith(f"GeneratedBackwardFor_cudnn_{OTHERS[family][0]}") for m in meta)


@pytest.mark.parametrize("family", OTHERS)
def test_double_backward_raises_like_custom_op_on_the_suite_cases(family):
    step = suite_step(family)
    for custom_op in (True, False):
        with PassThrough() if custom_op else contextlib.nullcontext():
            _, _, gs, leaves = step(True, create_graph=True)
        assert all(g.requires_grad for g in gs if g is not None), custom_op
        with pytest.raises(RuntimeError, match="no autograd formula was registered"):
            torch.autograd.grad(sum(g.float().sum() for g in gs if g is not None), [leaves[0]])


@pytest.mark.parametrize("family", OTHERS)
def test_profiler_sees_the_custom_ops_on_the_suite_cases(family):
    op_name = OTHERS[family][0]
    step = suite_step(family)
    for _ in range(3):
        step(True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        step(True)
    names = {e.name for e in prof.events()}
    assert {f"cudnn::{op_name}", f"cudnn::{op_name[: -len('_fwd')]}_bwd"} <= names
    assert not [n for n in names if "Eager" in n]


def summaries(family, case, d_o):
    """Every tensor the family's summary and summary-backward ops return for ``case``."""
    if family == "mamba2":
        out = (la.mamba2_summary(case), la.mamba2_summary_bwd(case, d_o))
    else:
        out = (la.run_summary(case), la.run_summary_bwd(case, d_o, output_transition=True))
    return [t for part in out for t in (part if isinstance(part, tuple) else (part,))]


@pytest.mark.parametrize("family", ["kda", "gdn", "gdn2", "gdp", "mamba2"])
def test_summaries_match_their_custom_ops(family):
    la.set_seed(7)
    case = la.make_case(family, torch.bfloat16, T=512, H=4)
    o = la.op(family)(*la.op_args(case), plan_name=f"{family}_frost")[0]
    d_o = (torch.randn(o.shape, device="cuda") * 0.1).to(o.dtype)
    with PassThrough():
        want = summaries(family, case, d_o)
    got = summaries(family, case, d_o)
    assert_same(got, want)
    assert len(got) >= 3 and not any(t.requires_grad for t in got)


@pytest.mark.parametrize("family", ["kda", "gdn", "gdn2", "gdp"])
def test_a_summary_of_inputs_that_require_grad_carries_no_graph(family):
    la.set_seed(7)
    case = la.make_case(family, torch.bfloat16, T=512, H=4)
    args = [a.requires_grad_(True) if torch.is_tensor(a) and a.is_floating_point() else a for a in la.op_args(case)[1:]]
    fn = getattr(la_ops, la.SUMMARY_OPS[family])
    for custom_op in (True, False):
        with PassThrough() if custom_op else contextlib.nullcontext():
            out = fn(*args, plan_name=f"{family}_summary_frost")
        assert not any(t.requires_grad or t.grad_fn is not None for t in out), custom_op
