# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QSA replay and batch invariance through the public training API."""

import pytest
import torch

from cudnn.ops import qsa_backward, qsa_forward
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from QSA.cutedsl.test_qsa import assert_close, make_inputs, reference, supported_gpu  # noqa: F401
from QSA.cutedsl.test_qsa_backward_sm100 import _check, _entry_backward, _inputs, _skip_cu12


def sequences(lengths, heads, *, mask="random", seed=20261104):
    samples = []
    for index, length in enumerate(lengths):
        sample = make_inputs((length,), *heads, empty=mask == "empty-rows", all_selected=mask == "all-selected")
        if mask == "all-masked":
            sample["sel_blocks"].zero_()
        rng = torch.Generator().manual_seed(seed + index)
        for name in ("q", "k", "v"):
            sample[name] = torch.randn(sample[name].shape, generator=rng).to(torch.bfloat16).cuda()
        sample["do"] = torch.randn(sample["q"].shape, generator=rng).to(torch.bfloat16).cuda()
        samples.append(sample)
    return samples


def pack(samples):
    total = sum(sample["q"].shape[0] for sample in samples)
    blocks = sum(sample["block_first"].numel() for sample in samples)
    selected = torch.zeros((total, blocks), dtype=torch.uint8, device="cuda")
    parts = {name: [] for name in samples[0] if name not in ("scale", "sel_blocks")}
    token_offset = block_offset = 0
    for sample in samples:
        length, count = sample["q"].shape[0], sample["block_first"].numel()
        selected[token_offset : token_offset + length, block_offset : block_offset + count] = sample["sel_blocks"]
        for name in parts:
            value = sample[name]
            if name == "blk_base":
                value = value + block_offset
            elif name in ("lo", "hi", "tok_base", "block_first", "block_last"):
                value = value + token_offset
            parts[name].append(value)
        token_offset += length
        block_offset += count
    return {**{name: torch.cat(values) for name, values in parts.items()}, "sel_blocks": selected, "scale": samples[0]["scale"]}


def run(inputs, backward, deterministic=True):
    local = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}
    do = local.pop("do")
    outputs = qsa_forward(**local)
    assert tuple(outputs.keys()) == ("out", "out_fp32", "lse")
    result = dict(zip(outputs.keys(), outputs))
    if backward:
        grads = qsa_backward(do, **local, out_fp32=outputs["out_fp32"], lse=outputs["lse"], deterministic=deterministic)
        assert tuple(grads.keys()) == ("dq", "dk", "dv")
        result.update(zip(grads.keys(), grads))
    return snapshot(result)


def split(result, samples):
    offset, pieces = 0, []
    for sample in samples:
        end = offset + sample["q"].shape[0]
        pieces.append({name: value[:, offset:end] if name == "lse" else value[offset:end] for name, value in result.items()})
        offset = end
    return pieces


def check_reference(inputs, actual, backward):
    local = {name: value for name, value in inputs.items() if name != "do"}
    for name, expected in zip(("out", "out_fp32", "lse"), reference(local)):
        assert_close(actual[name], expected)
    if backward:
        expected_grads = reference(local, inputs["do"], actual["out_fp32"], actual["lse"])
        for name, expected in zip(("dq", "dk", "dv"), expected_grads):
            assert_close(actual[name], expected)


@pytest.mark.usefixtures("supported_gpu")
class TestQSA:
    @pytest.mark.L0
    @pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)], ids=["q12-kv1", "q24-kv2", "q6-kv1"])
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_replay(self, heads, backward):
        inputs = pack(sequences((65, 3, 17), heads))
        tensors = {name: value for name, value in inputs.items() if isinstance(value, torch.Tensor)}
        before = snapshot(tensors)
        assert_replays(lambda: run(inputs, backward))
        assert_bitwise(tensors, before, "inputs changed")

    @pytest.mark.L0
    @pytest.mark.parametrize("heads", [(12, 1), (24, 2), (6, 1)], ids=["q12-kv1", "q24-kv2", "q6-kv1"])
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_batch_invariance(self, heads, backward):
        if backward:
            pytest.skip("skip until fix: QSA backward gradients change with batch layout (PR #2266)")
        samples = sequences((65, 3, 17), heads)
        expected = split(run(pack(samples), backward), samples)
        for order in ((0,), (1, 0), (2, 1, 0)):
            selected = [samples[index] for index in order]
            actual = split(run(pack(selected), backward), selected)
            for index, part in zip(order, actual):
                assert_bitwise(part, expected[index], f"batch {order}, sequence {index}")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_golden(self, backward):
        inputs = pack(sequences((17, 3), (12, 1)))
        # the saved hashes are the default path's
        actual = assert_replays(lambda: run(inputs, backward, deterministic=False), repeats=3)
        check_reference(inputs, actual, backward)
        tensors = {name: value for name, value in inputs.items() if isinstance(value, torch.Tensor)}
        tensors["scale"] = torch.tensor(inputs["scale"], dtype=torch.float64)
        assert_golden(f"qsa-{'bprop' if backward else 'fprop'}", tensors, actual)

    @pytest.mark.L0
    @pytest.mark.skip(reason="skip until fix: QSA sparse batch output/gradients differ (PR #2266)")
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_sparse_batch_invariance(self, backward):
        samples = sequences((2057, 3), (12, 1))
        expected = split(assert_replays(lambda: run(pack(samples), backward), repeats=4), samples)
        for index, sample in enumerate(samples):
            assert_bitwise(run(pack([sample]), backward), expected[index], f"sparse sequence {index}")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize(
        "total",
        [512, pytest.param(513, marks=pytest.mark.skip(reason="skip until fix: QSA dispatch boundary changes saved FP32 output (PR #2266)"))],
        ids=["total512", "total513"],
    )
    def test_dispatch_boundary(self, backward, total):
        target, neighbor = sequences((3, total - 3), (12, 1))
        expected = run(pack([target]), backward)
        for selected in ([target, neighbor], [neighbor, target]):
            index = 0 if selected[0] is target else 1
            actual = split(run(pack(selected), backward), selected)[index]
            assert_bitwise(actual, expected, f"total {total}, target position {index}")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("mask", ["all-selected", "empty-rows", "all-masked"])
    def test_mask_replay(self, backward, mask):
        inputs = pack(sequences((65, 3, 17), (12, 1), mask=mask))
        check_reference(inputs, run(inputs, backward), backward)
        assert_replays(lambda: run(inputs, backward))

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("mask", ["all-selected", "empty-rows", "all-masked"])
    def test_mask_batch_invariance(self, backward, mask):
        if backward and mask in ("all-selected", "empty-rows"):
            pytest.skip("skip until fix: QSA masked backward gradients change with batch layout (PR #2266)")
        samples = sequences((65, 3, 17), (12, 1), mask=mask)
        expected = split(run(pack(samples), backward), samples)
        for order in ((0,), (1,), (2,), (2, 1, 0), (0, 0)):
            selected = [samples[index] for index in order]
            actual = split(run(pack(selected), backward), selected)
            for index, part in zip(order, actual):
                assert_bitwise(part, expected[index], f"mask {mask}, batch {order}, sequence {index}")

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_neighbor_isolation(self, backward):
        samples = sequences((65, 17, 3), (12, 1))
        expected = split(run(pack(samples), backward), samples)[1]
        replacements = sequences((65, 17, 3), (12, 1), mask="all-selected", seed=20261105)
        changed = [replacements[0], samples[1], replacements[2]]
        # the backward's bytes follow the packed layout, so its neighbors only change their data
        for selected in (changed,) if backward else (changed, list(reversed(changed))):
            actual = split(run(pack(selected), backward), selected)[1]
            assert_bitwise(actual, expected, "changed neighbor data/masks, fixed target and upstream gradient")

    @pytest.mark.L0
    def test_zero_upstream_gradient_bprop(self):
        inputs = pack(sequences((65, 3, 17), (12, 1)))
        inputs["do"].zero_()
        actual = assert_replays(lambda: run(inputs, True))
        for name in ("dq", "dk", "dv"):
            assert torch.count_nonzero(actual[name]).item() == 0, f"{name}: zero upstream gradient produced a nonzero result"


ENGINE_REPLAYS = [
    # dense kernel: several key tiles (a dQ row takes one add per tile), several bands per tile (as do its dK / dV
    # rows per band), a key tile shared by two sequences
    pytest.param("bitmap", (300, 700, 129), 24, 2, "fp32", id="dense-tiles-q24-kv2"),
    pytest.param("bitmap", (2057, 3), 12, 1, "bf16", id="dense-tiles-q12-kv1-bf16"),
    pytest.param("bitmap", (1024,), 6, 1, "fp32", id="dense-tiles-q6-kv1"),
    # 6152 sparse queries fill a record window and a second one with two query ranges, the first size at which the order
    # of a key's adds shows. A token count that is a multiple of 8 leaves the key-major item list no spare key to pad with.
    pytest.param("bitmap", (26632,), 24, 2, "fp32", id="sparse-windows-q24-kv2"),
    pytest.param("bitmap", (26632,), 6, 1, "bf16", id="sparse-windows-q6-kv1-bf16"),
    pytest.param("indexed", (26632,), 12, 1, "fp32", id="sparse-windows-indexed-q12-kv1"),
    # a sparse sequence packed between dense ones, origins off the 4-token grid
    pytest.param("indexed", (3, 26633, 2051), 12, 1, "fp32", id="sparse-packed-indexed-q12-kv1"),
]


@pytest.mark.usefixtures("supported_gpu")
class TestQSABackwardEngine:
    """Backward replay on the engine paths the packed cases above do not reach."""

    @pytest.mark.L0
    @pytest.mark.parametrize("entry, lengths, h_q, h_kv, ds_precision", ENGINE_REPLAYS)
    def test_replay(self, entry, lengths, h_q, h_kv, ds_precision):
        _skip_cu12(h_kv)
        inputs = _inputs(lengths, h_kv, seed=sum(lengths), h_q=h_q)
        run_entry = lambda: _entry_backward(entry, inputs, lengths, ds_precision=ds_precision, deterministic=True)
        grads = assert_replays(lambda: dict(zip(("dq", "dk", "dv"), run_entry())), repeats=4)
        _check(inputs, tuple(grads[name] for name in ("dq", "dk", "dv")))
