# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 BT64 KDA: local derivative precision and byte stability.

Q/K/V and their gradients are BF16; gates, beta and states are FP32. Pin the
CUDA provider: a fallback would not exercise the local derivative under test.
"""

import pytest
import torch

from cudnn.linear_attention import kimi_delta_attention
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from linear_attention.determinism import check_reference, pack, sequences, split
from linear_attention.hopper.test_kda_sm90_cuda import ENGINE, requires_hopper

pytestmark = [pytest.mark.L0, requires_hopper]
LEAVES = ("q", "k", "v", "g", "beta", "initial_state")


def run(inputs, *, scale=None, snapshot_result=True):
    leaves = [inputs[name].detach().clone().requires_grad_() for name in LEAVES]
    out, state = kimi_delta_attention(
        *leaves[:5],
        inputs["cu"],
        initial_state=leaves[-1],
        output_final_state=True,
        scale=scale,
        batch_invariant=True,
        plan_name=ENGINE,
    )[:2]
    grads = torch.autograd.grad((out, state), leaves, (inputs["do"], inputs["dfs"]))
    result = dict(zip(("out", "final_state", *(f"d{name}" for name in LEAVES)), (out, state, *grads)))
    return snapshot(result) if snapshot_result else result


@pytest.mark.parametrize("heads", [2, 8], ids=["bv32", "bv16"])
@pytest.mark.parametrize("lengths", [(1, 15, 16, 17, 63, 64, 65, 0), (257,), (256,)], ids=["ragged", "single-tail", "single-full"])
def test_reference_and_replay(heads, lengths):
    inputs = pack(sequences("kda", lengths, heads=heads, key_dim=128, value_dim=128))
    before = snapshot(inputs)
    result = assert_replays(lambda: run(inputs))
    check_reference("kda", inputs, result, True)
    assert_bitwise(inputs, before, "inputs unchanged")


@pytest.mark.parametrize("length", [65, 257], ids=["one-boundary", "four-boundaries"])
@pytest.mark.parametrize("seed", [1, 713, 2026])
def test_correlated_gate_gradient(length, seed):
    """Signed local contributions must not lose the small gate derivative.

    Repeated normalized keys, beta=1/2 and no decay create a cancellation
    case without CP or a nearly-zero reference norm. The bound is separate
    from the ordinary BF16 output tolerance and checks the FP32 gate gradient.
    """
    from linear_attention import test_la as la

    inputs = pack(sequences("kda", (length,), heads=2, key_dim=128, value_dim=128, seed=seed))
    inputs["k"].zero_()
    inputs["k"][..., :2] = 0.70703125
    inputs["q"].copy_(inputs["k"])
    inputs["g"].zero_()
    inputs["beta"].fill_(0.5)
    inputs["initial_state"].zero_()
    before = snapshot(inputs)
    got = assert_replays(lambda: run(inputs), repeats=3)
    check_reference("kda", inputs, got, True)
    leaves = {name: inputs[name].cpu().double().unsqueeze(0) for name in LEAVES[:-1]}
    leaves["g"].requires_grad_()
    out, state = la.reference_call("kda", leaves, initial_state=inputs["initial_state"].cpu().double(), cu_seqlens=inputs["cu"].cpu())
    want = torch.autograd.grad((out, state), leaves["g"], (inputs["do"].cpu().double().unsqueeze(0), inputs["dfs"].cpu().double()))[0][0]
    assert want.norm() > 0.1
    error = (got["dg"].cpu().double() - want).norm() / want.norm()
    assert error < 0.02, f"correlated dG relative-L2 error {error.item():.6f}"
    assert_bitwise(inputs, before, "correlated inputs unchanged")


@pytest.mark.parametrize("heads", [2, 8], ids=["bv32", "bv16"])
@pytest.mark.parametrize("lengths", [(257, 0, 65, 17), (17001, 0, 129, 17)], ids=["short", "long"])
def test_batch_invariance(heads, lengths):
    samples = sequences("kda", lengths, heads=heads, key_dim=128, value_dim=128)
    expected = split("kda", run(pack(samples)), samples)
    for order in ((0,), (2,), (3, 2, 1, 0), (0, 0), (1, 3, 0)):
        selected = [samples[i] for i in order]
        actual = split("kda", run(pack(selected)), selected)
        for index, piece in zip(order, actual):
            assert_bitwise(piece, expected[index], f"order={order}, sequence={index}")
    changed = sequences("kda", (lengths[0],), heads=heads, key_dim=128, value_dim=128, seed=19)[0]
    actual = split("kda", run(pack([changed, *samples[1:]])), samples)
    for index in range(1, len(samples)):
        assert_bitwise(actual[index], expected[index], f"changed neighbor, sequence={index}")


def test_golden():
    inputs = pack(sequences("kda", (17, 3), key_dim=128, value_dim=128))
    result = assert_replays(lambda: run(inputs), repeats=3)
    check_reference("kda", inputs, result, True)
    assert_golden("kda-sm90-bt64-local-fwd-bwd-bf16", inputs, result)


def test_long_replay_with_concurrent_work():
    inputs = pack(sequences("kda", (4097, 129, 0, 65), heads=8, key_dim=128, value_dim=128))
    expected = run(inputs)
    other = torch.cuda.Stream()
    junk = torch.empty(1024, 1024, device="cuda")
    other.wait_stream(torch.cuda.current_stream())
    for _ in range(8):
        with torch.cuda.stream(other):
            junk.fill_(0.125)
            junk.square_()
        assert_bitwise(run(inputs), expected, "fresh buffers with concurrent work")
    torch.cuda.current_stream().wait_stream(other)


def test_scan_warm_stress():
    """A graph retains outputs/workspace while the scan wraps its three stages."""
    from warm_stress import StressCase, assert_warm_stress

    inputs = pack(sequences("kda", (1025, 513, 257, 129), heads=8, key_dim=128, value_dim=128))
    # Capture the public call, including autograd, without snapshot's CPU checks.
    run(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run(inputs, snapshot_result=False)
    try:
        case = StressCase("kda-sm90-bt64-local", graph.replay, outputs, thread_blocks=256, refills_per_launch=14)
        assert_warm_stress(case)
    finally:
        torch.cuda.synchronize()
        graph.reset()
