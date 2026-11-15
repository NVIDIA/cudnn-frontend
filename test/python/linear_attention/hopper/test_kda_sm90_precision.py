# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 CUDA KDA: FP32 additive state-gradient precision and byte stability.

Q/K/V and their gradients are BF16; gates, beta and states are FP32. Pin the
CUDA provider: a fallback would not exercise the P workspace under test.
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


@pytest.mark.parametrize("heads", [1, 8], ids=["bv32", "bv16"])
@pytest.mark.parametrize("lengths", [(2,), (64,), (65,), (257,), (0, 2, 65)], ids=["tail2", "full64", "tail65", "refill257", "ragged-empty"])
def test_p_preserves_cancellation_residual(heads, lengths):
    """Q_g^T dO = 1 + 2^-10 cancels an incoming -1, leaving exactly 2^-10.

    Keys are e1 and queries e0, so the state update/projection cannot modify
    this coordinate. All operands and the expected residual are exactly
    representable. BF16 storage of P makes d_initial_state zero instead.
    """
    inputs = pack(sequences("kda", lengths, heads=heads, key_dim=128, value_dim=128))
    for name in LEAVES + ("do", "dfs"):
        inputs[name].zero_()
    inputs["k"][..., 1] = 1
    inputs["beta"].fill_(0.5)
    want = torch.zeros_like(inputs["initial_state"])
    offset = 0
    for index, length in enumerate(lengths):
        if length:
            inputs["q"][offset : offset + 2, :, 0] = 1
            inputs["do"][offset, :, 0] = 1
            inputs["do"][offset + 1, :, 0] = 2**-10
            inputs["dfs"][index, :, 0, 0] = -1
            want[index, :, 0, 0] = 2**-10
        offset += length
    got = run(inputs, scale=1.0)
    torch.testing.assert_close(got["dinitial_state"], want, atol=0, rtol=0)
    for name, value in got.items():
        if name != "dinitial_state":
            assert torch.count_nonzero(value) == 0, name


@pytest.mark.parametrize("heads", [2, 8], ids=["bv32", "bv16"])
@pytest.mark.parametrize("lengths", [(1, 15, 16, 17, 63, 64, 65, 0), (257,), (256,)], ids=["ragged", "single-tail", "single-full"])
def test_reference_and_replay(heads, lengths):
    inputs = pack(sequences("kda", lengths, heads=heads, key_dim=128, value_dim=128))
    before = snapshot(inputs)
    result = assert_replays(lambda: run(inputs))
    check_reference("kda", inputs, result, True)
    assert_bitwise(inputs, before, "inputs unchanged")


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
    assert_golden("kda-sm90-fp32-p-fwd-bwd-bf16", inputs, result)


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
        case = StressCase("kda-sm90-fp32-p", graph.replay, outputs, thread_blocks=256, refills_per_launch=14)
        assert_warm_stress(case)
    finally:
        torch.cuda.synchronize()
        graph.reset()
