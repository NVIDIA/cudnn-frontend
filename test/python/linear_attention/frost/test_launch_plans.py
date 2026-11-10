# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA and GDN under launch plans (docs/utilities/launch_plans.md): the repeated forward and backward run from a plan and
match the unplanned implementation bitwise, also for other values (cu_seqlens included) at the recorded shapes."""

import math

import pytest
import torch

from cudnn import launch_plan
from cudnn.linear_attention.ops import gdn, kda

from linear_attention.test_la import frost_la_unavailable

requirement_error = frost_la_unavailable()
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)
if not launch_plan.ENABLED or launch_plan.VERIFY:
    pytest.skip("these tests count plans: CUDNN_FRONTEND_LAUNCH_PLANS is 0 or verify", allow_module_level=True)

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

D = 128
FAST = dict(use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True)
# name: (cu_seqlens of the recorded calls, cu_seqlens of the replay with other values, H, HV, options)
CASES = {
    "chain_state": ([0, 300, 1100, 1500, 2048], [0, 700, 900, 1800, 2048], 8, 8, dict(FAST, state=True, output_final_state=True)),
    "uncut_many_seqs": ([i * 272 for i in range(17)], [0] + [100 + 271 * i for i in range(15)] + [4352], 32, 32, FAST),
    "checkpoint": ([0, 1000, 2048], [0, 1500, 2048], 16, 16, dict(FAST, checkpoint_every_n_tokens=64)),
    "log_gate": ([0, 777, 2048], [0, 64, 2048], 16, 16, dict(use_qk_l2norm_in_kernel=True)),
    "gva_gate_params": ([0, 4000, 8192], [0, 123, 8192], 16, 32, dict(FAST, params=True)),
    "context_parallel": ([0, 1024, 2048], [0, 1536, 2048], 8, 8, dict(FAST, state=True, output_final_state=True, preceding=True)),
}


# family: (its planned forward and backward, its public op, whether its gate is per key channel)
FAMILIES = {
    "kda": (kda.run_kda_fwd, kda.run_kda_bwd, kda.kimi_delta_attention, True),
    "gdn": (gdn.run_gdn_fwd, gdn.run_gdn_bwd, gdn.gated_delta_net, False),
}


@pytest.fixture(autouse=True)
def blackwell(monkeypatch):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family Frost linear attention")
    monkeypatch.setattr(launch_plan, "HOT_CALLS", 2)  # record a signature's second call


@pytest.fixture(params=FAMILIES)
def family(request):
    return FAMILIES[request.param]


def inputs(seed, cu_list, H, HV, opts, channel_gate=True):
    """The forward's and backward's keyword arguments for one draw of values."""
    gen = torch.Generator().manual_seed(seed)
    T, N, HO = cu_list[-1], len(cu_list) - 1, max(H, HV)

    def draw(shape, dtype=torch.float32, scale=1.0, shift=0.0):
        return (torch.randn(shape, generator=gen) * scale + shift).to(dtype).cuda()

    fwd = dict(
        q=draw((T, H, D), torch.bfloat16),
        k=draw((T, H, D), torch.bfloat16),
        v=draw((T, HV, D), torch.bfloat16),
        g=draw((T, HO, D) if channel_gate else (T, HO), shift=-4.0 if opts.get("safe_gate") else -0.05),
        beta=draw((T, HO)),
        cu_seqlens=torch.tensor(cu_list, dtype=torch.int32, device="cuda"),
        scale=D**-0.5,
    )
    fwd.update({k: v for k, v in opts.items() if k not in ("state", "params", "preceding")})
    if opts.get("params"):
        fwd.update(a_log=draw((HO,), scale=0.1), dt_bias=draw((HO, D) if channel_gate else (HO,), scale=0.1))
    if opts.get("state"):
        fwd["initial_state"] = draw((N, HO, D, D), scale=0.05)
    if opts.get("preceding"):
        fwd.update(preceding_state_update=draw((2, HO, D, D), scale=0.05), preceding_transition=draw((2, HO, D, D), scale=0.01))
    bwd = {k: v for k, v in fwd.items() if k not in ("output_final_state", "preceding_state_update", "preceding_transition", "checkpoint_every_n_tokens")}
    bwd["dO"] = draw((T, HO, D), torch.bfloat16)
    if opts.get("output_final_state"):
        bwd["d_final_state"] = draw((N, HO, D, D), scale=0.05)
    return fwd, bwd


def assert_bitwise(got, want, checkpoint_rows=None):
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        if i == 2 and checkpoint_rows is not None:  # the checkpoint series' capacity rows past the last checkpoint are never written
            a, b = a[:checkpoint_rows], b[:checkpoint_rows]
        assert a.shape == b.shape and a.dtype == b.dtype, i
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), f"output {i} differs"


def planned_kinds(fn):
    return [type(p).__name__ for p in fn.plans.values()]


@pytest.mark.parametrize("case", CASES)
def test_planned_matches_unplanned(family, case):
    run_fwd, run_bwd, _, channel_gate = family
    cu_a, cu_b, H, HV, opts = CASES[case]
    run_fwd.plans.clear()
    run_bwd.plans.clear()
    every = opts.get("checkpoint_every_n_tokens", 0)

    def rows(cu):
        return sum(math.ceil((e - s) / every) for s, e in zip(cu, cu[1:])) if every else None

    def step(fwd, bwd, run_fwd, run_bwd):
        out = run_fwd(**fwd)
        extra = {}
        if every:
            extra = dict(state_checkpoints=out[2], checkpoint_every_n_tokens=every)
        if out[3].numel():
            extra["composed_initial_state"] = out[3]
        return out, run_bwd(**bwd, **extra)

    fwd_a, bwd_a = inputs(1, cu_a, H, HV, opts, channel_gate)
    first = step(fwd_a, bwd_a, run_fwd, run_bwd)  # runs the API
    step(fwd_a, bwd_a, run_fwd, run_bwd)  # records the plans
    planned = step(fwd_a, bwd_a, run_fwd, run_bwd)
    assert planned_kinds(run_fwd) == ["_Plan"] and planned_kinds(run_bwd) == ["_Plan"]
    for got, want in zip(planned, first):
        assert_bitwise(got, want, rows(cu_a))

    # other values, cu_seqlens included, at the recorded shapes: the plan runs, nothing host-side came from the values
    fwd_b, bwd_b = inputs(2, cu_b, H, HV, opts, channel_gate)
    planned = step(fwd_b, bwd_b, run_fwd, run_bwd)
    want = step(fwd_b, bwd_b, run_fwd.__wrapped__, run_bwd.__wrapped__)
    assert planned_kinds(run_fwd) == ["_Plan"] and planned_kinds(run_bwd) == ["_Plan"]
    for got, ref in zip(planned, want):
        assert_bitwise(got, ref, rows(cu_b))


def test_overwrite_state_planned_matches_unplanned(family):
    """The in-place op writes the final state into initial_state; the plan binds it as the argument it is."""
    run_fwd, _, _, channel_gate = family
    cu_a, cu_b = [0, 700, 2048], [0, 1300, 2048]
    run_fwd.plans.clear()

    def run(fwd, fn):
        state = fwd["initial_state"].clone()
        o = fn(**dict(fwd, initial_state=state, final_state_out=state))[0]
        return o, state

    fwd_a, _ = inputs(1, cu_a, 16, 16, dict(FAST, state=True, output_final_state=True), channel_gate)
    first = run(fwd_a, run_fwd)
    run(fwd_a, run_fwd)
    assert_bitwise(run(fwd_a, run_fwd), first)
    fwd_b, _ = inputs(2, cu_b, 16, 16, dict(FAST, state=True, output_final_state=True), channel_gate)
    assert_bitwise(run(fwd_b, run_fwd), run(fwd_b, run_fwd.__wrapped__))
    assert planned_kinds(run_fwd) == ["_Plan"]


def test_public_op_repeated_steps_are_planned(family):
    """Forward and autograd backward through the public op reach the plans; a broadcast dO (out.sum()) is materialized by
    the backward's Python, so that signature keeps running it."""
    run_fwd, run_bwd, public, channel_gate = family
    fwd, _ = inputs(1, [0, 1000, 2048], 8, 8, FAST, channel_gate)
    run_fwd.plans.clear()
    run_bwd.plans.clear()
    cu = fwd.pop("cu_seqlens")
    leaves = [fwd.pop(name).requires_grad_(True) for name in ("q", "k", "v", "g", "beta")]
    ones = torch.ones(2048, 8, D, dtype=torch.bfloat16, device="cuda")
    grads = []
    for _ in range(3):
        out, _ = public(*leaves, cu, **fwd)
        grads.append(torch.autograd.grad(out, leaves, ones))
    assert planned_kinds(run_fwd) == ["_Plan"] and planned_kinds(run_bwd) == ["_Plan"]
    assert_bitwise(grads[2], grads[0])
    for _ in range(3):
        out, _ = public(*leaves, cu, **fwd)
        broadcast = torch.autograd.grad(out.sum(), leaves)
    assert sorted(planned_kinds(run_bwd)) == ["_Plan", "int"] and -1 in run_bwd.plans.values()
    assert_bitwise(broadcast, grads[0])
