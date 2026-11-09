# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared forward/backward contract for the four FROST delta-rule families."""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from . import test_la as la
from .conftest import gen_qkv


def sequences(variant, lengths, dtype=torch.bfloat16, heads=2, *, value_heads=None, key_dim=64, value_dim=64, num_householder=3, seed=20261104):
    samples = []
    n = num_householder if variant == "gdp" else 1
    value_heads = heads if value_heads is None else value_heads
    output_heads = max(heads, value_heads)
    for index, length in enumerate(lengths):
        # CPU draws keep the dataset independent of the GPU's SM count.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + index)
            q, k, v = gen_qkv(1, length * n, heads, value_heads, key_dim, value_dim, dtype, device="cpu")
            q = q[:, ::n].contiguous()
            gates = la.gen_gates(variant, 1, length, output_heads, key_dim, value_dim, dtype, n=n, device="cpu")
            sample = {"q": q[0], "k": k[0], "v": v[0], **{name: value[0] for name, value in gates.items()}}
            sample["initial_state"] = torch.randn(1, output_heads, value_dim, key_dim) * 0.05
            sample["do"] = torch.randn(length, output_heads, value_dim).to(dtype)
            sample["dfs"] = torch.randn(1, output_heads, value_dim, key_dim) * 0.1
        samples.append({name: value.cuda() for name, value in sample.items()})
    return samples


def pack(samples):
    tensors = {name: torch.cat([sample[name] for sample in samples]) for name in samples[0]}
    lengths = [sample["q"].shape[0] for sample in samples]
    tensors["cu"] = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    return tensors


def run(variant, inputs, backward):
    names = (*la.LEAF_NAMES[variant], "initial_state")
    leaves = {name: inputs[name].detach().clone().requires_grad_(backward) for name in names}
    kwargs = dict(
        cu_seqlens=inputs["cu"].clone(),
        initial_state=leaves["initial_state"],
        output_final_state=True,
        batch_invariant=True,
        plan_name=f"{variant}_frost",
    )
    if variant == "gdp":
        kwargs["num_householder"] = inputs["k"].shape[0] // inputs["q"].shape[0]
    out, state = la.op(variant)(*(leaves[name] for name in la.LEAF_NAMES[variant]), **kwargs)
    result = {"out": out, "final_state": state}
    if backward:
        grads = torch.autograd.grad((out, state), tuple(leaves.values()), (inputs["do"], inputs["dfs"]))
        result.update((f"d{name}", value) for name, value in zip(names, grads))
    return snapshot(result)


def check_reference(variant, inputs, result, backward):
    names = la.LEAF_NAMES[variant]
    leaves = {name: inputs[name].detach().cpu().double().unsqueeze(0).requires_grad_(backward) for name in names}
    state = inputs["initial_state"].detach().cpu().double().requires_grad_(backward)
    n = inputs["k"].shape[0] // inputs["q"].shape[0] if variant == "gdp" else 1
    out, final = la.reference_call(variant, leaves, n=n, initial_state=state, cu_seqlens=inputs["cu"].cpu())
    expected = {"out": out[0], "final_state": final}
    if backward:
        grads = torch.autograd.grad((out, final), (*leaves.values(), state), (inputs["do"].cpu().double().unsqueeze(0), inputs["dfs"].cpu().double()))
        expected.update((f"d{name}", value[0]) for name, value in zip(names, grads[:-1]))
        expected["dinitial_state"] = grads[-1]
    for name, want in expected.items():
        tol = la.STATE_GRAD_TOL if name == "dinitial_state" else la.BWD_TOL[inputs["q"].dtype] if name.startswith("d") else la.FWD_TOL[inputs["q"].dtype]
        la.assert_rms_close(name, result[name].cpu(), want, tol)


def split(variant, result, samples):
    pieces, offset = [], 0
    n = sum(sample["k"].shape[0] for sample in samples) // sum(sample["q"].shape[0] for sample in samples) if variant == "gdp" else 1
    for index, sample in enumerate(samples):
        length = sample["q"].shape[0]
        piece = {}
        for name, value in result.items():
            if name in ("final_state", "dinitial_state"):
                piece[name] = value[index : index + 1]
            else:
                factor = n if name in ("dk", "dv", "dbeta") else 1
                piece[name] = value[offset * factor : (offset + length) * factor]
        pieces.append(piece)
        offset += length
    return pieces


def check_partitions(variant, samples, backward):
    expected = split(variant, assert_replays(lambda: run(variant, pack(samples), backward), repeats=4), samples)
    orders = [(index,) for index in range(len(samples))]
    orders += [tuple(reversed(range(len(samples)))), (0, 0)]
    for order in orders:
        selected = [samples[index] for index in order]
        actual = split(variant, run(variant, pack(selected), backward), selected)
        for index, part in zip(order, actual):
            assert_bitwise(part, expected[index], f"batch {order}, sequence {index}")


class LinearAttentionDeterminism:
    @pytest.fixture(autouse=True)
    def supported_runtime(self):
        reason = la.frost_la_unavailable()
        if reason:
            pytest.skip(reason)
        la.clear_caches()
        try:
            with torch.random.fork_rng():
                yield
        finally:
            la.clear_caches()

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    def test_replay(self, backward, dtype):
        inputs = pack(sequences(self.variant, (65, 0, 17, 1), dtype))
        before = snapshot(inputs)
        assert_replays(lambda: run(self.variant, inputs, backward))
        assert_bitwise(inputs, before, "inputs changed")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    def test_batch_invariance(self, backward, dtype):
        samples = sequences(self.variant, (65, 0, 17, 1), dtype)
        expected = split(self.variant, run(self.variant, pack(samples), backward), samples)
        for order in ((0,), (2, 0), (3, 2, 1, 0), (1, 3, 0, 2)):
            selected = [samples[index] for index in order]
            actual = split(self.variant, run(self.variant, pack(selected), backward), selected)
            for index, part in zip(order, actual):
                assert_bitwise(part, expected[index], f"batch {order}, sequence {index}")
        assert_bitwise({"state": expected[1]["final_state"]}, {"state": samples[1]["initial_state"]}, "empty sequence")
        if backward:
            assert_bitwise({"ds": expected[1]["dinitial_state"]}, {"ds": samples[1]["dfs"]}, "empty sequence backward")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_golden(self, backward):
        inputs = pack(sequences(self.variant, (17, 3)))
        actual = assert_replays(lambda: run(self.variant, inputs, backward), repeats=3)
        check_reference(self.variant, inputs, actual, backward)
        assert_golden(f"{self.variant}-{'bprop' if backward else 'fprop'}", inputs, actual)

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_long_batch_invariance(self, backward):
        samples = sequences(self.variant, (17001, 17), heads=2)
        expected = assert_replays(lambda: run(self.variant, pack(samples), backward), repeats=4)
        expected = split(self.variant, expected, samples)
        for index, sample in enumerate(samples):
            actual = run(self.variant, pack([sample]), backward)
            assert_bitwise(actual, expected[index], f"long sequence {index}")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_many_work_items_replay(self, backward):
        heads = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count // 4 + 1
        inputs = pack(sequences(self.variant, (129, 17, 65, 1), heads=heads))
        assert_replays(lambda: run(self.variant, inputs, backward))

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    def test_tile_boundaries(self, backward, dtype):
        samples = sequences(self.variant, (1, 15, 16, 17, 63, 64, 65), dtype)
        check_partitions(self.variant, samples, backward)

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize(
        "heads,value_heads,key_dim,value_dim",
        [(1, 4, 64, 128), (4, 1, 128, 64), (2, 2, 128, 128)],
        ids=["qk1-v4-k64-v128", "qk4-v1-k128-v64", "qk2-v2-k128-v128"],
    )
    def test_head_layouts(self, backward, heads, value_heads, key_dim, value_dim):
        samples = sequences(self.variant, (17, 3), heads=heads, value_heads=value_heads, key_dim=key_dim, value_dim=value_dim)
        inputs = pack(samples)
        check_reference(self.variant, inputs, run(self.variant, inputs, backward), backward)
        check_partitions(self.variant, samples, backward)

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_neighbor_isolation(self, backward):
        samples = sequences(self.variant, (65, 17, 3))
        expected = split(self.variant, run(self.variant, pack(samples), backward), samples)[1]
        replacements = sequences(self.variant, (65, 17, 3), seed=20261105)
        changed = [replacements[0], samples[1], replacements[2]]
        for selected in (changed, list(reversed(changed))):
            actual = split(self.variant, run(self.variant, pack(selected), backward), selected)[1]
            assert_bitwise(actual, expected, "changed neighbors, fixed target and upstream gradients")

    @pytest.mark.L0
    @pytest.mark.parametrize("zero_gradient", ["do", "dfs"], ids=["bprop-state-loss-only", "bprop-output-loss-only"])
    def test_gradient_source(self, zero_gradient):
        samples = sequences(self.variant, (17, 3))
        for sample in samples:
            sample[zero_gradient].zero_()
        inputs = pack(samples)
        check_reference(self.variant, inputs, run(self.variant, inputs, True), True)
        check_partitions(self.variant, samples, True)
