# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism of ``gdn_hopper_cuda`` (the SM90 Gated DeltaNet engine), forward and backward.

Contract under test: ``cudnn.linear_attention.gated_delta_net`` pinned to
``plan_name="gdn_hopper_cuda"``, bf16 q/k/v (L2-normalised q and k), fp32 g,
beta and state, K = V = 128, packed THD with int32 ``cu_seqlens``, a carried
initial state and ``output_final_state=True``. Outputs ``out`` and
``final_state``; gradients ``dq``, ``dk``, ``dv``, ``dg``, ``dbeta`` and
``dinitial_state``. FP16 is not served by this engine (it declines the graph).

Covered: raw-byte replay with fresh buffers and concurrent GPU work in both
directions, across the launch geometries the engine switches on (forward split
into pieces at 132 or fewer (sequence, head) pairs, the 64-column backward scan
with its aliased workspace at 32 or more pairs, one long sequence, more work
items than SMs, chunk boundaries, an empty sequence); isolation of a sequence
from its neighbours' data at fixed geometry; backward batch invariance;
upstream-gradient sources; a small golden per direction, checked against an
fp64 reference first.

Not provided: forward batch invariance. With 132 or fewer (sequence, head)
pairs the forward cuts each sequence into ``264 // pairs`` pieces, and a piece
warms up from a truncated history, so a sequence's output bytes can depend on
how many other sequences share the launch. On H100 the change was rare and
small (one of 614,400 output elements, relative difference 1.5e-5, for one
sequence alone against packed with three others at 16 heads on a softplus
gate; none on the other gates measured), but it exists, so no forward batch
case is asserted here. The engine declines ``batch_invariant=True``.
Shared-memory release ordering (Rule 11) is not covered by replay; see
``test_gdn_sm90_cuda.py``.
"""

import pytest
import torch
import torch.nn.functional as F

import cudnn  # noqa: F401 -- import-order requirement, see test/python/conftest.py
from cudnn.linear_attention import gated_delta_net
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from linear_attention.reference_gdn import gdn_reference, rms_ratio

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0), reason="the sm90 CUDA GDN engine needs Hopper"),
]

ENGINE = "gdn_hopper_cuda"
DIM = 128
GRADS = ("dq", "dk", "dv", "dg", "dbeta", "dinitial_state")
LEAVES = ("q", "k", "v", "g", "beta", "initial_state")
TOL = {"out": 2e-2, "final_state": 2e-2, "dinitial_state": 6e-2}  # rms ratios as in test_la.py, gradients 4e-2


def sequences(lengths, heads=2, gate="init", seed=20261112):
    """One dict of inputs and upstream gradients per sequence, drawn on the CPU so the data does not follow the GPU."""
    samples = []
    for index, length in enumerate(lengths):
        gen = torch.Generator().manual_seed(seed + index)
        q = F.normalize(torch.randn(length, heads, DIM, generator=gen), dim=-1)
        k = F.normalize(torch.randn(length, heads, DIM, generator=gen), dim=-1)
        v = torch.randn(length, heads, DIM, generator=gen) * 0.5
        x = torch.randn(length, heads, generator=gen)
        if gate == "init":  # a freshly initialised model: -A * softplus(0.5 x + 1), A ~ U(1, 16)
            g = -(1 + 15 * torch.rand(heads, generator=gen)) * F.softplus(0.5 * x + 1.0)
        else:  # slow: about -0.02 per token, the forward's walk-back case
            g = -0.02 * F.softplus(x)
        sample = {
            "q": q.bfloat16(),
            "k": k.bfloat16(),
            "v": v.bfloat16(),
            "g": g.float(),
            "beta": torch.sigmoid(torch.randn(length, heads, generator=gen)).float(),
            "initial_state": torch.randn(1, heads, DIM, DIM, generator=gen) * 0.05,
            "do": torch.randn(length, heads, DIM, generator=gen).bfloat16(),
            "dfs": torch.randn(1, heads, DIM, DIM, generator=gen) * 0.1,
        }
        samples.append({name: value.cuda() for name, value in sample.items()})
    return samples


def pack(samples):
    tensors = {name: torch.cat([sample[name] for sample in samples]).contiguous() for name in samples[0]}
    lengths = [sample["q"].shape[0] for sample in samples]
    tensors["cu"] = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    return tensors


def run(inputs, backward):
    """Fresh leaves and outputs every call; the result is snapshotted before the next launch."""
    leaves = {name: inputs[name].detach().clone().requires_grad_(backward) for name in LEAVES}
    out, state = gated_delta_net(
        *(leaves[name] for name in LEAVES[:5]),
        inputs["cu"].clone(),
        initial_state=leaves["initial_state"],
        output_final_state=True,
        plan_name=ENGINE,
    )[:2]
    result = {"out": out, "final_state": state}
    if backward:
        grads = torch.autograd.grad((out, state), tuple(leaves.values()), (inputs["do"], inputs["dfs"]))
        result.update((f"d{name}", value) for name, value in zip(LEAVES, grads))
    return snapshot(result)


def gradients(result):
    return {name: result[name] for name in GRADS}


def split(result, samples):
    """Per-sequence slices of a packed result: token rows for out and the token gradients, state rows by index."""
    pieces, offset = [], 0
    for index, sample in enumerate(samples):
        length = sample["q"].shape[0]
        piece = {}
        for name, value in result.items():
            piece[name] = value[index : index + 1] if name in ("final_state", "dinitial_state") else value[offset : offset + length]
        pieces.append(piece)
        offset += length
    return pieces


def check_reference(inputs, result, backward):
    """An fp64 recurrence, independent of the byte comparisons."""
    leaves = {name: inputs[name].detach().double().requires_grad_(backward) for name in LEAVES}
    out, final = gdn_reference(*(leaves[name].unsqueeze(0) for name in LEAVES[:5]), initial_state=leaves["initial_state"], cu_seqlens=inputs["cu"])
    expected = {"out": out[0], "final_state": final}
    if backward:
        grads = torch.autograd.grad((out, final), tuple(leaves.values()), (inputs["do"].double().unsqueeze(0), inputs["dfs"].double()))
        expected.update((f"d{name}", value) for name, value in zip(LEAVES, grads))
    for name, want in expected.items():
        tol = TOL.get(name, 4e-2)
        err = rms_ratio(result[name], want)
        assert err <= tol, f"{name}: rms ratio {err:.3e} against the fp64 reference exceeds {tol}"


def geometry_cases():
    sms = torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else 132
    return [
        pytest.param((1000, 37, 515), 16, id="split-fwd-wide-scan"),  # 48 pairs: forward splits, backward runs the 64-column scan
        pytest.param((64,) * 40, 4, id="unsplit-fwd-wide-scan"),  # 160 pairs: no split
        pytest.param((257,), 2, id="split-fwd-narrow-scan"),  # 2 pairs: deep split, 32-column scan
        pytest.param((17001,), 2, id="long-sequence"),
        pytest.param((129, 17, 65, 1), sms // 4 + 1, id="more-work-items-than-sms"),
        pytest.param((1, 15, 16, 17, 63, 64, 65), 2, id="chunk-boundaries"),
    ]


@pytest.fixture(autouse=True)
def _fresh_rng():
    with torch.random.fork_rng():
        yield


@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
@pytest.mark.parametrize("gate", ["init", "slow"])
def test_replay(backward, gate):
    """Eight launches with fresh buffers and a busy side stream agree byte for byte; the inputs are not touched."""
    inputs = pack(sequences((65, 0, 17, 1), gate=gate))
    before = snapshot(inputs)
    assert_replays(lambda: run(inputs, backward))
    assert_bitwise(inputs, before, "inputs changed")


@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
@pytest.mark.parametrize("lengths, heads", geometry_cases())
def test_replay_across_launch_geometries(backward, lengths, heads):
    inputs = pack(sequences(lengths, heads))
    assert_replays(lambda: run(inputs, backward), repeats=4)


@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
def test_empty_sequence_passes_its_state_through(backward):
    """A zero-length sequence returns its initial state, and its state gradient is the upstream state gradient."""
    samples = sequences((65, 0, 17))
    result = split(run(pack(samples), backward), samples)[1]
    assert_bitwise({"state": result["final_state"]}, {"state": samples[1]["initial_state"]}, "empty sequence")
    if backward:
        assert_bitwise({"ds": result["dinitial_state"]}, {"ds": samples[1]["dfs"]}, "empty sequence backward")


@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
@pytest.mark.parametrize("heads", [2, 16], ids=["split-fwd-narrow-scan", "split-fwd-wide-scan"])
def test_neighbor_isolation(backward, heads):
    """Changing the neighbours' data, with every length and the target's data and upstream gradients fixed, leaves the target's bytes alone."""
    samples = sequences((65, 17, 300), heads)
    expected = split(run(pack(samples), backward), samples)[1]
    replacements = sequences((65, 17, 300), heads, seed=20261113)
    changed = [replacements[0], samples[1], replacements[2]]
    actual = split(run(pack(changed), backward), changed)[1]
    assert_bitwise(actual, expected, "changed neighbours, fixed target and geometry")


@pytest.mark.parametrize("heads", [2, 16], ids=["narrow-scan", "wide-scan"])
def test_backward_batch_invariance(heads):
    """The gradients of one sequence do not depend on what else is in the batch.

    Only the gradients are compared: the forward is not batch invariant (see the module docstring), and the
    backward recomputes its intermediates from the inputs rather than from the forward's outputs.
    """
    samples = sequences((65, 17, 300, 1), heads)
    expected = [gradients(part) for part in split(run(pack(samples), True), samples)]
    for order in ((0,), (2, 0), (3, 2, 1, 0), (1, 3, 0, 2), (0, 0)):
        selected = [samples[index] for index in order]
        actual = split(run(pack(selected), True), selected)
        for index, part in zip(order, actual):
            assert_bitwise(gradients(part), expected[index], f"batch {order}, sequence {index}")


@pytest.mark.parametrize("zero_gradient", ["do", "dfs"], ids=["bprop-state-loss-only", "bprop-output-loss-only"])
def test_gradient_source(zero_gradient):
    samples = sequences((17, 3))
    for sample in samples:
        sample[zero_gradient].zero_()
    inputs = pack(samples)
    result = assert_replays(lambda: run(inputs, True), repeats=4)
    check_reference(inputs, result, True)


@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
def test_golden(backward):
    inputs = pack(sequences((17, 3)))
    actual = assert_replays(lambda: run(inputs, backward), repeats=3)
    check_reference(inputs, actual, backward)
    assert_golden(f"gdn-sm90-cuda-{'bprop' if backward else 'fprop'}", inputs, actual)
