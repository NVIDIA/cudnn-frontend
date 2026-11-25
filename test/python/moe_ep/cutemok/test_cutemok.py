# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The inputs and the reference are derived from the test utilities of Mixture-of-Kittens
# (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""CuTe MoK: forward and backward against a PyTorch reference, the shared output gate against the layer functions,
CUDA graph replay, the plan's ownership of its workspace, declines and exports.

The GPU tests run on the ranks of the process group: one GPU under plain pytest (a one-rank NCCL group), expert
parallel under torchrun (``torchrun --nproc-per-node=8 -m pytest test_cutemok.py``).
"""

import dataclasses
import gc
import os
import subprocess
import sys
import textwrap
import weakref

import pytest
import torch
import torch.distributed as dist

from determinism_utils import assert_bitwise, assert_replays

import cudnn.moe_ep.cutemok as public
from cudnn._tupledict import TupleDict
from cudnn.moe_ep._cutemok import functional
from cudnn.moe_ep.cutemok import (
    CuteMoKConfig,
    CuteMoKPlan,
    CuteMoKSplitWeight,
    cutemok_backward,
    cutemok_forward,
    cutemok_prepare,
    cutemok_quantize_weights,
)
from cudnn.moe_ep._cutemok.cutedsl._arch import SUPPORTED_CAPABILITIES
from cudnn.moe_ep._cutemok.cutedsl._version_gate import CuteDslRequirementError, require_cutedsl

pytestmark = pytest.mark.L0

NAMES = ("CuteMoKConfig", "CuteMoKPlan", "CuteMoKSplitWeight", "cutemok_backward", "cutemok_forward", "cutemok_prepare", "cutemok_quantize_weights")
GRADIENT_NAMES = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")
LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, TOPK, MAX_TOKENS, TOKENS = 4, 1024, 512, 4, 2048, 1024
SWIGLU_LIMIT = 4.0
# (max absolute error, relative L1 error) over all ranks, against the BF16 reference (also for MXFP8 routed weights).
BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
EP_SIZES = (1, 4, 8, 16, 32, 64)


@pytest.fixture(scope="module")
def group():
    """(rank, world size, device) of an NCCL group: torchrun's when launched under it, else a one-rank group."""
    if not torch.cuda.is_available():
        pytest.skip("CuTe MoK requires a CUDA GPU")
    owned = not dist.is_initialized()
    if owned and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe MoK requires SM100 / SM103")
    try:
        require_cutedsl()
    except CuteDslRequirementError as error:
        pytest.skip(str(error))
    if owned and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", device_id=device)
    elif owned:
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1, device_id=device)
    elif dist.get_backend() != "nccl":
        pytest.skip("CuTe MoK requires an NCCL process group")
    if dist.get_world_size() not in EP_SIZES:
        if owned:
            dist.destroy_process_group()
        pytest.skip(f"CuTe MoK runs on {EP_SIZES} ranks")
    yield dist.get_rank(), dist.get_world_size(), device
    dist.barrier()
    if owned:
        dist.destroy_process_group()


def _inputs(rank, world_size, device, seed):
    """One rank's inputs: top-k of Gaussian router logits with softmax router weights, BF16 activations and weights
    scaled by 1/sqrt(fan-in)."""
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    logits = torch.randn(TOKENS, LOCAL_EXPERTS * world_size, generator=generator, device=device)
    values, top_experts = torch.topk(logits, TOPK, dim=1)
    router_weights = torch.softmax(values.float(), dim=-1)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.bfloat16) * scale

    x = randn(TOKENS, HIDDEN)
    shared = tuple(randn(*shape, scale=shape[-1] ** -0.5) for shape in ((INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)))
    routed = tuple(randn(LOCAL_EXPERTS, *w.shape, scale=w.shape[-1] ** -0.5) for w in shared)
    d_output = randn(TOKENS, HIDDEN, scale=HIDDEN**-0.5)
    return x, top_experts.to(torch.int32), router_weights, shared, routed, d_output


def _all_to_all(x, count, output_splits, input_splits):
    output = x.new_empty((count, *x.shape[1:]))
    dist.all_to_all_single(output, x, output_splits, input_splits)
    return output


def _swiglu(gate, up, limit):
    if limit is None:
        return torch.nn.functional.silu(gate) * up
    # The kernels keep the clamp's derivative at 1 on the boundary: clamp strictly outside the range only.
    gate, up = gate.float(), up.float()
    gate = torch.where(gate > limit, limit, gate)
    up = torch.where(up < -limit, -limit, torch.where(up > limit, limit, up))
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def _reference(x, top_experts, router_weights, shared, routed, d_output, limit):
    """The layer in PyTorch: tokens dispatched to their experts' ranks with all-to-all, each local expert's SwiGLU MLP
    in BF16, the results combined in FP32.  Returns the output and the gradients in the backward's order."""
    world_size, rank = dist.get_world_size(), dist.get_rank()
    tokens, hidden = x.shape
    topk, local_experts = top_experts.shape[1], routed[0].shape[0]
    experts = top_experts.flatten().long()
    valid = (experts >= 0).nonzero().flatten()
    destination = experts[valid] // local_experts
    order = valid[torch.argsort(destination, stable=True)]
    send_counts = torch.bincount(destination, minlength=world_size)
    all_counts = torch.empty(world_size, world_size, dtype=torch.int64, device=x.device)
    dist.all_gather_into_tensor(all_counts, send_counts)
    send, recv = send_counts.tolist(), all_counts[:, rank].tolist()
    recv_x = _all_to_all(x[order // topk], sum(recv), recv, send).requires_grad_()
    recv_experts = _all_to_all((experts[order] % local_experts).contiguous(), sum(recv), recv, send)

    gate_w, up_w, down_w = (w.detach().requires_grad_() for w in routed)
    recv_output = torch.zeros_like(recv_x)
    for expert in range(local_experts):
        rows = (recv_experts == expert).nonzero().flatten()
        hidden_act = _swiglu(recv_x[rows] @ gate_w[expert].T, recv_x[rows] @ up_w[expert].T, limit)
        recv_output = recv_output.index_copy(0, rows, hidden_act @ down_w[expert].T)
    returned = _all_to_all(recv_output.detach(), valid.numel(), send, recv)
    routed_output = torch.zeros(tokens * topk, hidden, dtype=returned.dtype, device=x.device)
    routed_output[order] = returned
    routed_output = routed_output.view(tokens, topk, hidden)

    x_shared = x.detach().requires_grad_()
    shared_w = [w.detach().requires_grad_() for w in shared]
    shared_output = _swiglu(x_shared @ shared_w[0].T, x_shared @ shared_w[1].T, limit) @ shared_w[2].T
    output = ((routed_output.float() * router_weights.unsqueeze(2)).sum(1) + shared_output.float()).to(torch.bfloat16)

    d_routed = (d_output.unsqueeze(1).float() * router_weights.unsqueeze(2)).to(torch.bfloat16)
    d_recv = _all_to_all(d_routed.reshape(-1, hidden)[order], sum(recv), recv, send).contiguous()
    d_recv_x, *d_routed_w = torch.autograd.grad(recv_output, (recv_x, gate_w, up_w, down_w), d_recv)
    returned_d_x = _all_to_all(d_recv_x, valid.numel(), send, recv)
    d_x_routed = torch.zeros(tokens * topk, hidden, dtype=returned_d_x.dtype, device=x.device)
    d_x_routed[order] = returned_d_x
    d_x_shared, *d_shared_w = torch.autograd.grad(shared_output, (x_shared, *shared_w), d_output)
    d_x = (d_x_routed.view(tokens, topk, hidden).float().sum(1) + d_x_shared.float()).to(torch.bfloat16)
    d_router = (d_output.unsqueeze(1).float() * routed_output.float()).sum(2).masked_fill(top_experts < 0, 0.0)
    return output, (d_x, d_router, *d_routed_w, *d_shared_w)


def _check(name, reference, actual, tolerance):
    """Max absolute and relative L1 error over every rank's tensor within ``tolerance``."""
    difference = (reference.float() - actual.float()).abs()
    sums = torch.stack((difference.double().sum(), reference.double().abs().sum()))
    largest = difference.max()
    dist.all_reduce(sums)
    dist.all_reduce(largest, op=dist.ReduceOp.MAX)
    error, magnitude = sums.tolist()
    absolute, relative = largest.item(), (0.0 if error == magnitude == 0.0 else error / magnitude)
    assert torch.isfinite(torch.tensor((absolute, relative))).all(), f"{name}: not finite"
    assert absolute <= tolerance[0] and relative <= tolerance[1], (
        f"{name}: max abs error {absolute:.4e} (bound {tolerance[0]}), relative L1 error {relative:.4e} " f"(bound {tolerance[1]})"
    )


def _assert_all_ranks(ok, message, device):
    """``assert ok, message`` on every rank together: the failures are counted over the ranks first (an all-reduce),
    so a check that fails on some ranks fails it on all of them instead of leaving the others in the next collective
    (the next call's kernels, or the next test's prepare)."""
    failed = torch.tensor(0 if ok else 1, dtype=torch.int32, device=device)
    dist.all_reduce(failed)
    failed = int(failed.item())
    assert failed == 0, f"{message if not ok else 'passed on this rank'} (failed on {failed} of {dist.get_world_size()} ranks)"


def _config():
    return CuteMoKConfig(fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5)


def _prepare(config, device, **options):
    return cutemok_prepare(
        config,
        dist.group.WORLD,
        device=device,
        max_tokens_per_rank=MAX_TOKENS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=LOCAL_EXPERTS,
        topk=TOPK,
        **options,
    )


# Each case is a plan of its own (the variants are compile-time choices of prepare).
CASES = {
    "bf16": {},
    "clamped_swiglu": dict(swiglu_clamped=True),
    "mxfp8": dict(routed_precision="mxfp8"),
    "no_shared_expert": dict(has_shared=False),
    "split_weights": dict(split_routed_weights=True, main_grad_dtype=torch.float32),
}


@pytest.mark.parametrize("case", list(CASES))
def test_forward_backward(group, case):
    rank, world_size, device = group
    config = _config()
    plan = _prepare(config, device, **CASES[case])
    x, top_experts, router_weights, shared, routed, d_output = _inputs(rank, world_size, device, 40)
    limit = SWIGLU_LIMIT if case == "clamped_swiglu" else None
    tolerance = MXFP8_TOLERANCE if case == "mxfp8" else BF16_TOLERANCE
    call_shared, forward_routed, backward_routed = shared, routed, routed
    main_grads = tables = None
    if case == "no_shared_expert":
        call_shared, shared = (None, None, None), tuple(torch.zeros_like(w) for w in shared)
    elif case == "mxfp8":
        quantized = cutemok_quantize_weights(*routed)
        assert isinstance(quantized, TupleDict) and list(quantized.keys()) == ["forward_weights", "backward_weights"]
        forward_routed, backward_routed = quantized
    elif case == "split_weights":
        # One allocation per expert: a combined FC1 [2I, H] (gate rows first) and FC2 [H, I], each with an FP32
        # main_grad of its own; the shared expert's main_grads are three FP32 buffers.
        gate, up, down = routed
        fc1 = [torch.cat((gate[e], up[e])) for e in range(LOCAL_EXPERTS)]
        fc2 = [down[e].clone() for e in range(LOCAL_EXPERTS)]
        fc1_weight, fc2_weight = CuteMoKSplitWeight.from_experts(fc1), CuteMoKSplitWeight.from_experts(fc2)
        forward_routed = backward_routed = (fc1_weight, fc1_weight, fc2_weight)
        fc1_grads = [torch.zeros(w.shape, dtype=torch.float32, device=device) for w in fc1]
        fc2_grads = [torch.zeros(w.shape, dtype=torch.float32, device=device) for w in fc2]
        shared_grads = [torch.zeros(w.shape, dtype=torch.float32, device=device) for w in shared]
        main_grads = (shared_grads[0], fc1_grads[0], shared_grads[1], fc1_grads[0], shared_grads[2], fc2_grads[0])
        fc1_table, fc2_table = (torch.tensor([b.data_ptr() for b in buffers], dtype=torch.int64, device=device) for buffers in (fc1_grads, fc2_grads))
        tables = (fc1_table, fc1_table, fc2_table)

    forward = cutemok_forward(x, top_experts, router_weights, *call_shared, *forward_routed, plan=plan, config=config, swiglu_limit=limit)
    output, schedule, saved = forward
    assert isinstance(forward, TupleDict) and list(forward.keys()) == ["output", "schedule", "forward_context"]
    grads = cutemok_backward(
        d_output,
        x,
        router_weights,
        *call_shared,
        *backward_routed,
        schedule,
        saved,
        plan=plan,
        config=config,
        main_grads=main_grads,
        main_grad_storage_tables=tables,
        swiglu_limit=limit,
    )
    assert grads[8] is None  # no shared output gate
    assert isinstance(grads, TupleDict) and list(grads.keys()) == [*GRADIENT_NAMES, "d_w_output_gate"]
    grads = list(grads)[:8]
    if case == "split_weights":
        grads[2:] = (
            torch.stack([g[:INTERMEDIATE] for g in fc1_grads]),
            torch.stack([g[INTERMEDIATE:] for g in fc1_grads]),
            torch.stack(fc2_grads),
            *shared_grads,
        )

    expected_output, expected_grads = _reference(x, top_experts, router_weights, shared, routed, d_output, limit)
    _check(f"{case}: output", expected_output, output, tolerance)
    for name, expected, actual in zip(GRADIENT_NAMES, expected_grads, grads, strict=True):
        if case == "no_shared_expert" and name.startswith("d_w_shared"):
            assert actual is None, name
        else:
            _check(f"{case}: {name}", expected, actual, tolerance)


def _gate_weight(rank, device, seed):
    """A shared output gate weight, bfloat16 [1, H] scaled by 1/sqrt(H)."""
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    return torch.randn(1, HIDDEN, generator=generator, device=device, dtype=torch.bfloat16) * HIDDEN**-0.5


@pytest.mark.parametrize("main_grad", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_shared_output_gate(group, precision, main_grad):
    """The shared output gate (prepare with ``shared_output_gate=True``; the gate weight in forward and backward)
    gives the layer functions' results bit for bit: the output, the eight gradients and the gate weight's FP32
    gradient, new or added into ``shared_output_gate_main_grad``.  A gated forward context needs the weight in
    the backward."""
    rank, world_size, device = group
    config = _config()
    plan = _prepare(config, device, shared_output_gate=True, routed_precision=precision)
    x, top_experts, router_weights, shared, routed, d_output = _inputs(rank, world_size, device, 60)
    weight = _gate_weight(rank, device, 61)
    forward_routed = backward_routed = routed
    if precision == "mxfp8":
        forward_routed, backward_routed = cutemok_quantize_weights(*routed)
    main_grads = [torch.full((1, HIDDEN), 0.5, dtype=torch.float32, device=device) if main_grad else None for _ in range(2)]

    output, schedule, saved = cutemok_forward(
        x, top_experts, router_weights, *shared, *forward_routed, plan=plan, config=config, shared_output_gate_weight=weight
    )
    with pytest.raises(ValueError, match="a gated forward context requires shared_output_gate_weight"):
        cutemok_backward(d_output, x, router_weights, *shared, *backward_routed, schedule, saved, plan=plan, config=config)
    grads = cutemok_backward(
        d_output,
        x,
        router_weights,
        *shared,
        *backward_routed,
        schedule,
        saved,
        plan=plan,
        config=config,
        shared_output_gate_weight=weight,
        shared_output_gate_main_grad=main_grads[0],
    )
    assert list(grads.keys()) == [*GRADIENT_NAMES, "d_w_output_gate"]
    if main_grad:
        assert grads["d_w_output_gate"] is main_grads[0]
    else:
        assert grads["d_w_output_gate"].dtype == torch.float32 and tuple(grads["d_w_output_gate"].shape) == (1, HIDDEN)

    workspace = functional.get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    layer_schedule = functional.build_schedule(workspace, config, top_experts, num_local_experts=LOCAL_EXPERTS)
    expected_output, layer_context = functional.forward(
        config, workspace, layer_schedule, x, router_weights, *shared, *forward_routed, shared_output_gate_weight=weight
    )
    expected = functional.backward(
        config,
        workspace,
        layer_schedule,
        layer_context,
        d_output,
        x,
        router_weights,
        *shared,
        *backward_routed,
        shared_output_gate_weight=weight,
        shared_output_gate_main_grad=main_grads[1],
    )
    for name, a, b in zip(("output", *GRADIENT_NAMES, "d_w_output_gate"), (expected_output, *expected), (output, *grads), strict=True):
        assert torch.equal(a, b), f"{precision}{' main_grad' if main_grad else ''}: {name} differs"
    torch.cuda.synchronize(device)
    dist.barrier()


@pytest.mark.parametrize("precision,gated", [("bf16", False), ("mxfp8", False), ("bf16", True)])
def test_cuda_graph(group, precision, gated):
    """Forward and backward captured once after one eager warm-up call, and replayed on new inputs written into the
    captured tensors (routing included): every replay matches an eager call bit for bit.  The captured calls
    allocate their results in the graph's pool.  ``gated``: with the shared output gate."""
    rank, world_size, device = group
    config = _config()
    plan = _prepare(config, device, routed_precision=precision, shared_output_gate=gated)
    x, top_experts, router_weights, shared, routed, d_output = _inputs(rank, world_size, device, 50)
    gate = dict(shared_output_gate_weight=_gate_weight(rank, device, 55)) if gated else {}
    forward_routed = backward_routed = routed
    if precision == "mxfp8":
        forward_routed, backward_routed = cutemok_quantize_weights(*routed)

    def step():
        output, schedule, saved = cutemok_forward(x, top_experts, router_weights, *shared, *forward_routed, plan=plan, config=config, **gate)
        grads = cutemok_backward(d_output, x, router_weights, *shared, *backward_routed, schedule, saved, plan=plan, config=config, **gate)
        return [output, *list(grads)[: 9 if gated else 8]]

    names = ("output", *GRADIENT_NAMES, *(("d_w_output_gate",) if gated else ()))
    step()  # warm-up
    if precision == "mxfp8":
        assert_replays(lambda: dict(zip(names, step(), strict=True)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for seed in (51, 52, 51):
        inputs = _inputs(rank, world_size, device, seed)
        for static, value in zip((x, top_experts, router_weights, d_output), (*inputs[:3], inputs[5])):
            static.copy_(value)
        eager = [t.clone() for t in step()]
        graph.replay()
        assert_bitwise(
            dict(zip(names, captured, strict=True)), dict(zip(names, eager, strict=True)), f"{precision}{' gated' if gated else ''}: seed {seed} vs eager"
        )
    del graph
    dist.barrier()


def _call(plan, config, layer):
    """One layer's forward and backward on ``plan``: the output and the eight gradients."""
    x, top_experts, router_weights, shared, routed, d_output = layer
    output, schedule, saved = cutemok_forward(x, top_experts, router_weights, *shared, *routed, plan=plan, config=config)
    grads = cutemok_backward(d_output, x, router_weights, *shared, *routed, schedule, saved, plan=plan, config=config)
    return [output, *list(grads)[:8]]


def test_plan_shared_by_layers(group):
    """Two plans of one configuration own separate workspaces; one plan serving two layers (both forwards, then
    both backwards, as a model runs them) gives each layer's results of a plan of its own, bit for bit."""
    rank, world_size, device = group
    config = _config()
    plans = [_prepare(config, device) for _ in range(2)]
    assert plans[0].workspace.x_buffer.data_ptr() != plans[1].workspace.x_buffer.data_ptr()
    layers = [_inputs(rank, world_size, device, seed) for seed in (70, 71)]
    expected = [_call(plan, config, layer) for plan, layer in zip(plans, layers)]

    shared_plan = plans[0]
    forwards = [
        cutemok_forward(x, top_experts, router_weights, *shared, *routed, plan=shared_plan, config=config)
        for x, top_experts, router_weights, shared, routed, _ in layers
    ]
    actual = {}
    for index in (1, 0):
        x, _, router_weights, shared, routed, d_output = layers[index]
        output, schedule, saved = forwards[index]
        grads = cutemok_backward(d_output, x, router_weights, *shared, *routed, schedule, saved, plan=shared_plan, config=config)
        actual[index] = [output, *list(grads)[:8]]
    for index in (0, 1):
        for name, a, b in zip(("output", *GRADIENT_NAMES), expected[index], actual[index], strict=True):
            assert torch.equal(a, b), f"layer {index}: {name} differs on the shared plan"
    torch.cuda.synchronize(device)
    dist.barrier()


def test_barrier_counter_wrap(group):
    """The cross-rank barrier's 32-bit counters wrap in a long run: with both counters set so that the first barrier
    of a call wraps its threshold to 0, and every rank but 0 held back before the call, the call still waits for all
    ranks and gives the results of a fresh plan, bit for bit.  (An unsigned compare lets rank 0 through on its own
    arrival there and read its peers' previous layer.)"""
    rank, world_size, device = group
    if world_size == 1:
        pytest.skip("one rank launches no barrier")
    config = _config()
    plan = _prepare(config, device)
    fresh = _prepare(config, device)  # kept until the end: a plan is dropped only after its work finished
    layers = [_inputs(rank, world_size, device, seed) for seed in (90, 90 + 2 * world_size)]
    expected = _call(fresh, config, layers[1])
    _call(plan, config, layers[0])  # the peers' buffers now hold layer 0
    torch.cuda.synchronize(device)
    dist.barrier()
    start = -world_size  # 2^32 - ep_size: the next barrier's threshold wraps to 0
    plan.workspace.barrier_target.fill_(start)
    plan.workspace.barrier_buffer.fill_(start)
    torch.cuda.synchronize(device)
    dist.barrier()
    if rank != 0:
        torch.cuda._sleep(200_000_000)  # hold the other ranks back: rank 0 reaches the barrier first
    actual = _call(plan, config, layers[1])
    for name, a, b in zip(("output", *GRADIENT_NAMES), expected, actual, strict=True):
        assert torch.equal(a, b), f"{name} differs after the barrier counter wrapped"
    torch.cuda.synchronize(device)
    # the call's barriers did cross the wrap: the target went from 2^32 - ep_size past 0
    assert 0 < int(plan.workspace.barrier_target.item()) < 2**31
    dist.barrier()


def test_plan_owns_workspace(group):
    """prepare does not fill the process-wide workspace cache of the layer functions; dropping the plan, after the
    calls on it finished on every rank, frees its symmetric buffers."""
    rank, world_size, device = group
    config = _config()
    cached = set(functional._WORKSPACE_CACHE)
    plan = _prepare(config, device)
    results = _call(plan, config, _inputs(rank, world_size, device, 80))
    assert set(functional._WORKSPACE_CACHE) == cached
    assert all(workspace is not plan.workspace for workspace in (*functional._WORKSPACE_CACHE.values(), *functional._WORKSPACE_LOOKUP.values()))
    buffers = [weakref.ref(getattr(plan.workspace, field.name)) for field in dataclasses.fields(plan.workspace) if field.name.endswith("_buffer")]
    del results
    torch.cuda.synchronize(device)
    dist.barrier()
    del plan
    gc.collect()
    assert len(buffers) == 8 and all(ref() is None for ref in buffers), "a workspace buffer outlived its plan"
    dist.barrier()


def _routed_forms(layer, precision):
    """(forward, backward) routed weights of a layer: the BF16 ones, or their MXFP8 forms."""
    routed = layer[4]
    return cutemok_quantize_weights(*routed) if precision == "mxfp8" else (routed, routed)


def _step(plan, config, layer, forms):
    """One layer's forward and backward on ``plan`` with the routed weights ``forms`` (``_routed_forms``): the output
    and the eight gradients."""
    x, top_experts, router_weights, shared, _, d_output = layer
    forward_routed, backward_routed = forms
    output, schedule, saved = cutemok_forward(x, top_experts, router_weights, *shared, *forward_routed, plan=plan, config=config)
    grads = cutemok_backward(d_output, x, router_weights, *shared, *backward_routed, schedule, saved, plan=plan, config=config)
    return [output, *list(grads)[:8]]


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_plan_memory(group, precision):
    """prepare keeps what later calls read -- the workspace and the backward's counters, none of it scaling with the
    macrobatch but the BF16 backward's router-weight storage; the scheduler's scratch and the forward's counters are
    per call, as in the layer functions -- and a forward + backward returns all of its scratch: after the first call
    (which builds the BF16 backward's launch storage) a call leaves the allocated bytes unchanged."""
    rank, world_size, device = group
    layer = _inputs(rank, world_size, device, 85)
    forms = _routed_forms(layer, precision)
    resident, peak = {}, {}
    for macrobatch in (4096, 8192):
        config = dataclasses.replace(_config(), macrobatch_size=macrobatch)
        torch.cuda.synchronize(device)
        gc.collect()
        before = torch.cuda.memory_allocated(device)
        plan = _prepare(config, device, routed_precision=precision)
        torch.cuda.synchronize(device)
        resident[macrobatch] = torch.cuda.memory_allocated(device) - before
        # Device buffers besides the workspace: a tensor or a tuple of them.
        held = {field.name for field in dataclasses.fields(plan) if field.name != "workspace" and isinstance(getattr(plan, field.name), (torch.Tensor, tuple))}
        _assert_all_ranks(held == {"backward_storage"}, f"{precision}: the plan holds {held}", device)
        _step(plan, config, layer, forms)
        torch.cuda.synchronize(device)
        settled = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        _step(plan, config, layer, forms)
        torch.cuda.synchronize(device)
        returned = torch.cuda.memory_allocated(device) == settled
        peak[macrobatch] = torch.cuda.max_memory_allocated(device) - settled
        _assert_all_ranks(returned, f"{precision}: a call kept device memory", device)
        dist.barrier()
        del plan
        gc.collect()
        dist.barrier()
    # The BF16 backward's router weights and partials: macrobatch * (1 + I / 128) FP32 values, and int32 counters
    # that do not grow with the macrobatch.
    bound = 4 * (8192 - 4096) * (1 + INTERMEDIATE // 128) if precision == "bf16" else 0
    _assert_all_ranks(resident[8192] - resident[4096] <= bound + 64 * 1024, f"{precision}: resident bytes {resident}", device)
    # The per-call scratch grows with the macrobatch instead: it is in the call's peak, not in the plan.
    _assert_all_ranks(peak[8192] > peak[4096], f"{precision}: call peaks {peak}", device)


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_call_scratch_not_read_across_calls(group, precision):
    """Nothing a forward leaves outside its results is read by the backward: with a NaN-filled allocation (which
    reuses the freed scratch) and another plan's forward between them, the results equal a forward and backward
    back to back, bit for bit."""
    rank, world_size, device = group
    config = _config()
    plan, other = (_prepare(config, device, routed_precision=precision) for _ in range(2))
    layer, other_layer = (_inputs(rank, world_size, device, seed) for seed in (90, 91))
    forms = _routed_forms(layer, precision)
    expected = [t.clone() for t in _step(plan, config, layer, forms)]

    x, top_experts, router_weights, shared, _, d_output = layer
    forward_routed, backward_routed = forms
    output, schedule, saved = cutemok_forward(x, top_experts, router_weights, *shared, *forward_routed, plan=plan, config=config)
    junk = torch.full((128 << 20,), float("nan"), dtype=torch.bfloat16, device=device)
    del junk
    ox, otop, orouter, oshared, _, _ = other_layer
    other_forward = cutemok_forward(ox, otop, orouter, *oshared, *_routed_forms(other_layer, precision)[0], plan=other, config=config)
    grads = cutemok_backward(d_output, x, router_weights, *shared, *backward_routed, schedule, saved, plan=plan, config=config)
    differ = [name for name, a, b in zip(("output", *GRADIENT_NAMES), expected, [output, *list(grads)[:8]], strict=True) if not torch.equal(a, b)]
    _assert_all_ranks(not differ, f"{precision}: {differ} differ with other work between forward and backward", device)
    del other_forward
    torch.cuda.synchronize(device)
    dist.barrier()


def _launch_plans(precision):
    """The megakernels' kept launch plans (their LaunchPlanCache values): a call that misses its cache adds one."""
    if precision == "mxfp8":
        from cudnn.moe_ep._cutemok.cutedsl import persistent_mxfp8_bwd, persistent_mxfp8_fwd

        caches = (persistent_mxfp8_fwd._FAST_PLANS, persistent_mxfp8_bwd._FAST_PLANS)
    else:
        from cudnn.moe_ep._cutemok.cutedsl import persistent_bf16_bwd, persistent_bf16_fwd

        caches = (persistent_bf16_fwd._LAUNCH_PLANS, *persistent_bf16_bwd._PLAN_CACHES)
    return [entry[2] for cache in caches for entry in cache._entries.values()]


def _free_blocks(device):
    """The caching allocator's free blocks that an allocation on the current stream can return: (state, size) of the
    blocks of the default pool's segments of that stream not in use (freed, or freed with pending stream uses).
    Without a large-pool block of 1 MiB or less: expandable segments (test/python/conftest.py enables them) split off
    such remainders, and a large-pool request exceeds 1 MiB, so one is reused only as part of a larger block."""
    stream = torch.cuda.current_stream(device).cuda_stream
    return [
        (block["state"], block["size"], segment["segment_type"])
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == device.index and segment["stream"] == stream and tuple(segment.get("segment_pool_id", (0, 0))) == (0, 0)
        for block in segment["blocks"]
        if block["state"] != "active_allocated" and (segment["segment_type"] == "small" or block["size"] > 1 << 20)
    ]


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_cache_hit_scratch_at_new_addresses(group, precision):
    """A call that hits the launch plans an earlier call built passes its own scratch, not the addresses the plans
    were built with: with every block the first forward + backward freed (its scratch among them) held by a
    sentinel during the second, the second call's results equal the first's bit for bit and the sentinel is
    untouched.  The plan has the shared output gate, so the gate's scratch (G * dY) reaches the backward
    megakernel too."""
    rank, world_size, device = group
    config = _config()
    gc.collect()
    torch.cuda.empty_cache()  # the earlier tests' cached blocks: the sentinel then holds only this test's
    plan = _prepare(config, device, routed_precision=precision, shared_output_gate=True)
    layer = _inputs(rank, world_size, device, 95)
    x, top_experts, router_weights, shared, _, d_output = layer
    forward_routed, backward_routed = _routed_forms(layer, precision)
    gate = dict(shared_output_gate_weight=_gate_weight(rank, device, 96))

    def step():
        output, schedule, saved = cutemok_forward(x, top_experts, router_weights, *shared, *forward_routed, plan=plan, config=config, **gate)
        grads = cutemok_backward(d_output, x, router_weights, *shared, *backward_routed, schedule, saved, plan=plan, config=config, **gate)
        return [output, *grads]

    earlier = _launch_plans(precision)
    expected = [t.clone() for t in step()]
    built = [p for p in _launch_plans(precision) if not any(p is q for q in earlier)]
    _assert_all_ranks(len(built) == 2, f"{precision}: the first call built {len(built)} megakernel launch plans, not 2", device)

    # Every block the first call freed is held: the allocator is best fit, so allocating the free blocks' sizes
    # largest first takes each block whole (a small-pool block in pieces of at most 1 MiB, the small pool's largest
    # request).  A few rounds also fill what the sentinel's own allocations leave free.  The second call's scratch
    # then comes from new segments, at addresses the first call never used.
    torch.cuda.synchronize(device)
    torch.empty(1, device=device)  # an allocation processes the frees whose stream uses have completed
    pattern, small = 0x5A, 1 << 20
    sentinel = []
    for _ in range(8):
        free = sorted(((size, kind) for state, size, kind in _free_blocks(device) if state == "inactive"), reverse=True)
        if not free:
            break
        for size, kind in free:
            piece = min(size, small) if kind == "small" else size
            sentinel += [torch.empty(piece, dtype=torch.uint8, device=device).fill_(pattern) for _ in range(size // piece)]
    left = _free_blocks(device)
    _assert_all_ranks(bool(sentinel) and not left, f"{precision}: free blocks left beside the sentinel: {left}", device)

    actual = step()
    kept = _launch_plans(precision)
    hit = all(any(p is q for q in kept) for p in built) and all(any(p is q for q in earlier + built) for p in kept)
    _assert_all_ranks(hit, f"{precision}: the second call did not run on the first call's launch plans", device)
    differ = [name for name, a, b in zip(("output", *GRADIENT_NAMES, "d_w_output_gate"), expected, actual, strict=True) if not torch.equal(a, b)]
    _assert_all_ranks(not differ, f"{precision}: {differ} differ on the cache hit", device)
    extrema = torch.stack([torch.stack((t.min(), t.max())) for t in sentinel])
    _assert_all_ranks(bool((extrema == pattern).all()), f"{precision}: the second call wrote into the first call's freed blocks", device)
    del sentinel
    torch.cuda.synchronize(device)
    dist.barrier()


def _host_plan(precision, shared_output_gate=False):
    """A plan object for the declines, which are raised before a plan's buffers are touched."""
    fields = {field.name: None for field in dataclasses.fields(CuteMoKPlan)}
    return CuteMoKPlan(**fields | {"routed_precision": precision, "shared_output_gate": shared_output_gate, "_views": {}, "_shapes": {}})


def test_declines():
    """Inputs the plan cannot serve raise before anything is launched (no GPU needed)."""
    x = torch.zeros(512, 256, dtype=torch.bfloat16)
    top_experts = torch.zeros(512, 2, dtype=torch.int32)
    router_weights = torch.zeros(512, 2, dtype=torch.float32)
    routed = [torch.zeros(1, 256, 256, dtype=torch.bfloat16)] * 3
    buffers = dict(plan=_host_plan("bf16"), config=CuteMoKConfig())

    def forward(x=x, top_experts=top_experts, router_weights=router_weights, routed=routed, **changes):
        cutemok_forward(x, top_experts, router_weights, None, None, None, *routed, **buffers | changes)

    for arguments, name in (
        (dict(x=x.float()), "x"),
        (dict(x=x.t().contiguous().t()), "x"),
        (dict(top_experts=top_experts.long()), "top_experts"),
        (dict(router_weights=router_weights.bfloat16()), "router_weights"),
        (dict(routed=[(w, w) for w in routed]), "routed_gate_weights: mxfp8 routed weights"),
        (dict(plan=_host_plan("mxfp8")), "routed_gate_weights: bf16 routed weights"),
    ):
        with pytest.raises(NotImplementedError, match=name):
            forward(**arguments)
    with pytest.raises(TypeError, match="cutemok_prepare"):
        forward(plan=object())
    with pytest.raises(NotImplementedError, match="grad_output"):
        cutemok_backward(x.float(), x, router_weights, None, None, None, *routed, None, None, **buffers)
    # The shared output gate: the weight is given exactly when the plan was prepared with it.
    weight = torch.zeros(1, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="pass shared_output_gate_weight"):
        forward(plan=_host_plan("bf16", shared_output_gate=True))
    with pytest.raises(ValueError, match="prepared without the shared output gate"):
        forward(shared_output_gate_weight=weight)
    with pytest.raises(ValueError, match="prepared without the shared output gate"):
        cutemok_backward(x, x, router_weights, None, None, None, *routed, None, None, **buffers, shared_output_gate_weight=weight)

    gate, up = torch.zeros(2, 256, 512, dtype=torch.bfloat16), torch.zeros(2, 256, 512, dtype=torch.bfloat16)
    down = torch.zeros(2, 512, 256, dtype=torch.bfloat16)
    odd = torch.zeros(2, 192, 512, dtype=torch.bfloat16)
    for arguments, error, match in (
        ((None, up, down), TypeError, "gate must be a torch.Tensor"),
        ((gate.float(), up, down), TypeError, "gate must be bfloat16"),
        ((gate, up[:, :128], down), ValueError, "up must have gate's shape"),
        ((odd, odd.clone(), torch.zeros(2, 512, 192, dtype=torch.bfloat16)), ValueError, "multiples of 128"),
        ((gate, up, down), ValueError, "CUDA device"),
    ):
        with pytest.raises(error, match=match):
            cutemok_quantize_weights(*arguments)


def test_public_exports():
    """The seven names are the public module's whole ``__all__`` and the same objects in ``cudnn.ops``; importing
    ``cudnn.ops`` imports neither the public module nor the implementation package."""
    assert sorted(public.__all__) == sorted(NAMES)
    import cudnn
    import cudnn.ops

    for name in NAMES:
        assert name in cudnn.ops.__all__ and getattr(cudnn.ops, name) is getattr(public, name), name
    code = f"""
        import sys
        import cudnn.ops
        assert "cudnn.moe_ep.cutemok" not in sys.modules and "cudnn.moe_ep._cutemok" not in sys.modules, "cudnn.ops imported the operator"
        assert all(name in cudnn.ops.__all__ for name in {NAMES!r})
        print("OK")
    """
    path = os.pathsep.join(filter(None, (os.path.dirname(os.path.dirname(cudnn.__file__)), os.environ.get("PYTHONPATH"))))
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=dict(os.environ, PYTHONPATH=path), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and result.stdout.startswith("OK"), result.stdout + result.stderr


@pytest.fixture
def toolchain_snapshot(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from cudnn.moe_ep._cutemok.cutedsl import persistent_bf16_bwd as backward

    root = tmp_path / "dsl" / "cutlass"
    specs = {
        name: SimpleNamespace(origin=str(root.joinpath(*name.split(".")[1:], "__init__.py")), submodule_search_locations=[str(root)])
        for name in ("cutlass", "cutlass.cute", "cutlass.base_dsl")
    }
    find_spec = backward.importlib.util.find_spec
    monkeypatch.setattr(backward.importlib.util, "find_spec", lambda name: specs[name] if name in specs else find_spec(name))
    monkeypatch.setattr(backward, "_device_capability", lambda device: backward.TARGET_CAPABILITY)
    monkeypatch.setattr(torch.version, "cuda", "13.4")
    return backward, specs


@pytest.mark.parametrize("extra_roots", [0, 1, 2])
def test_toolchain_allows_namespace_extensions(toolchain_snapshot, tmp_path, extra_roots):
    backward, specs = toolchain_snapshot
    specs["cutlass"].submodule_search_locations.extend(str(tmp_path / f"extension{i}" / "cutlass") for i in range(extra_roots))
    backward._runtime_toolchain_snapshot(torch.device("cuda:0"))


@pytest.mark.parametrize("module", ["cutlass.cute", "cutlass.base_dsl"])
def test_toolchain_rejects_mixed_dsl_modules(toolchain_snapshot, tmp_path, module):
    backward, specs = toolchain_snapshot
    specs[module].origin = str(tmp_path.joinpath("other_dsl", *module.split("."), "__init__.py"))
    with pytest.raises(backward.ToolchainMismatchError, match="exactly one active CUTLASS DSL origin"):
        backward._runtime_toolchain_snapshot(torch.device("cuda:0"))


@pytest.mark.parametrize("module", ["cutlass", "cutlass.cute", "cutlass.base_dsl"])
def test_toolchain_requires_concrete_dsl_origins(toolchain_snapshot, module):
    backward, specs = toolchain_snapshot
    specs[module].origin = None
    with pytest.raises(backward.ToolchainMismatchError, match=f"cannot determine CUTLASS DSL origin for {module}"):
        backward._runtime_toolchain_snapshot(torch.device("cuda:0"))
