# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise replay and batch invariance through the public DSA training APIs."""

import pytest
import torch

from cudnn import DSA
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from deepseek_sparse_attention.cutedsl.dsa_reference import check_ref_dsa_sparse_attention_backward, ref_sparse_attention_forward

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
ALLOWED_INF = {"max_logits": -torch.inf, "lse": torch.inf, "lse_indexer": torch.inf}
SHAPES = [(64, 512, 64), (64, 576, 64), (128, 512, 128)]
SHAPE_IDS = ["h64-d512-k64", "h64-d576-k64", "h128-d512-k128"]


@pytest.fixture(autouse=True)
def supported_runtime():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("DSA determinism suite requires SM100, SM103 or SM107")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("DSA requires CuTe DSL")
    reason = cutedsl_requirement_error("DSA") or cutedsl_arch_requirement_error(torch.cuda.get_device_capability())
    if reason:
        pytest.skip(reason)


def sequences(lengths=(17, 3), *, heads=64, dim=512, topk=64, dtype=torch.bfloat16, selection="full", seed=20261104):
    samples = []
    for index, length in enumerate(lengths):
        rng = torch.Generator().manual_seed(seed + index)
        kv_length = topk + 33 + index
        sample = {
            "q": (torch.randn(length, heads, dim, generator=rng) * 0.1).to(dtype),
            "kv": (torch.randn(kv_length, dim, generator=rng) * 0.1).to(dtype),
            "attn_sink": torch.linspace(-0.5, 0.5, heads),
            "dout": torch.randn(length, heads, 512, generator=rng).to(dtype),
            "topk_idxs": torch.stack([torch.randperm(kv_length, generator=rng)[:topk] for _ in range(length)]).int(),
        }
        if selection in ("ragged", "empty"):
            sample["topk_length"] = torch.arange(length, dtype=torch.int32) % (topk + 1)
            if selection == "empty":
                sample["topk_length"].zero_()
            else:
                sample["topk_length"][-1] = topk
        if selection == "invalid":
            sample["topk_idxs"][:, ::7] = -7
            sample["topk_idxs"][:, 1::7] = kv_length + 3
        samples.append({name: value.cuda() for name, value in sample.items()})
    return samples


def pack(samples, gradient_target=None):
    total_kv = sum(sample["kv"].shape[0] for sample in samples)
    parts = {name: [] for name in samples[0] if name != "attn_sink"}
    offset = 0
    for index, sample in enumerate(samples):
        assert_bitwise({"sink": sample["attn_sink"]}, {"sink": samples[0]["attn_sink"]}, "shared sink")
        for name in parts:
            value = sample[name]
            if name == "topk_idxs":
                valid = (value >= 0) & (value < sample["kv"].shape[0])
                value = torch.where(valid, value + offset, torch.where(value < 0, value, total_kv + value))
            elif name == "dout" and gradient_target is not None and index != gradient_target:
                value = torch.zeros_like(value)
            parts[name].append(value)
        offset += sample["kv"].shape[0]
    return {**{name: torch.cat(values) for name, values in parts.items()}, "attn_sink": samples[0]["attn_sink"]}


def run(inputs, backward, indexer_topk=0):
    local = {name: value.clone() for name, value in inputs.items()}
    before = snapshot(local)
    dout = local.pop("dout")
    output = DSA.sparse_attention_forward_wrapper(**local, indexer_topk=indexer_topk)
    assert tuple(output.keys()) == ("out", "max_logits", "lse", "lse_indexer")
    assert (output["lse_indexer"] is None) == (indexer_topk == 0)
    result = {name: value for name, value in zip(output.keys(), output) if value is not None}
    idx = local["topk_idxs"]
    valid = (idx >= 0) & (idx < local["kv"].shape[0])
    if "topk_length" in local:
        valid &= torch.arange(idx.shape[1], device=idx.device)[None] < local["topk_length"][:, None]
    empty = ~valid.any(dim=1)
    for name, sentinel in ALLOWED_INF.items():
        if name in result:
            empty_stat = ~valid[:, :indexer_topk].any(dim=1) if name == "lse_indexer" else empty
            assert torch.equal(result[name] == sentinel, empty_stat[:, None].expand_as(result[name])), f"{name}: wrong empty-row sentinel"
    assert torch.count_nonzero(result["out"][empty]).item() == 0, "empty row produced a nonzero output"
    if backward:
        grads = DSA.sparse_attention_backward_wrapper(**local, out=output["out"], dout=dout, lse=output["lse"], deterministic=True)
        assert tuple(grads.keys()) == ("dq", "dkv", "d_sink")
        result.update(zip(grads.keys(), grads))
    assert_bitwise({**local, "dout": dout}, before, "kernel inputs changed")
    return snapshot(result, allowed_inf=ALLOWED_INF)


def check_reference(inputs, actual, backward, indexer_topk=0):
    local = {name: value for name, value in inputs.items() if name != "dout"}
    expected = ref_sparse_attention_forward(**local, indexer_topk=indexer_topk, return_full=True)
    for name, value in zip(("out", "max_logits", "lse", "lse_indexer"), expected):
        if value is not None:
            atol, rtol = (8e-4, 3.01 / 128) if name == "out" else (1e-6, 2.01 / 65536)
            torch.testing.assert_close(actual[name].float(), value.float(), atol=atol, rtol=rtol)
    if backward:
        check_ref_dsa_sparse_attention_backward(
            **local,
            out=actual["out"],
            dout=inputs["dout"],
            lse=actual["lse"],
            dq_actual=actual["dq"],
            dkv_actual=actual["dkv"],
            d_sink_actual=actual["d_sink"],
            atol=5e-2,
            rtol=5e-2,
        )


def piece(result, samples, index):
    q_start = sum(sample["q"].shape[0] for sample in samples[:index])
    kv_start = sum(sample["kv"].shape[0] for sample in samples[:index])
    selected = {}
    for name, value in result.items():
        if name == "d_sink":
            selected[name] = value
        elif name == "dkv":
            selected[name] = value[kv_start : kv_start + samples[index]["kv"].shape[0]]
        else:
            selected[name] = value[q_start : q_start + samples[index]["q"].shape[0]]
    return selected


def check_batch(samples, backward, indexer_topk=0, targets=None):
    for target in range(len(samples)) if targets is None else targets:
        # Isolate the target's loss because d_sink is shared across sequences.
        expected = run(pack([samples[target]]), backward, indexer_topk)
        for order in (tuple(range(len(samples))), tuple(reversed(range(len(samples)))), (target, target)):
            selected = [samples[index] for index in order]
            position = order.index(target)
            actual = run(pack(selected, gradient_target=position if backward else None), backward, indexer_topk)
            assert_bitwise(piece(actual, selected, position), expected, f"batch {order}, sequence {target}")


class TestDSA:
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("heads,dim,topk", SHAPES, ids=SHAPE_IDS)
    def test_replay(self, backward, dtype, heads, dim, topk):
        inputs = pack(sequences(heads=heads, dim=dim, topk=topk, dtype=dtype))
        actual = assert_replays(lambda: run(inputs, backward), allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, backward)

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize("heads,dim,topk", SHAPES, ids=SHAPE_IDS)
    def test_batch_invariance(self, backward, dtype, heads, dim, topk):
        if backward and dtype == torch.bfloat16 and (heads, dim, topk) == (128, 512, 128):
            pytest.skip("skip until fix: DSA H128 BF16 dKV changes with batch layout (PR #2266)")
        check_batch(sequences(heads=heads, dim=dim, topk=topk, dtype=dtype), backward)

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("selection", ["ragged", "invalid", "empty", "tail65", "prefix512"])
    def test_selection(self, backward, selection):
        topk = 512 if selection == "prefix512" else 65 if selection == "tail65" else 64
        prefix = 512 if selection == "prefix512" else 0
        samples = sequences((5, 3), topk=topk, selection=selection)
        inputs = pack(samples)
        actual = assert_replays(lambda: run(inputs, backward, prefix), allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, backward, prefix)
        check_batch(samples, backward, prefix)

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize(
        "heads,total", [(64, 128), (64, 129), (64, 257), (128, 74), (128, 75)], ids=["h64-q128", "h64-q129", "h64-q257", "h128-q74", "h128-q75"]
    )
    def test_query_boundary(self, backward, heads, total):
        if backward and (heads, total) in ((64, 129), (64, 257), (128, 74), (128, 75)):
            pytest.skip("skip until fix: DSA query boundary changes dKV across batches (PR #2266)")
        samples = sequences((17, total - 17), heads=heads, topk=128 if heads == 128 else 64)
        assert_replays(lambda: run(pack(samples), backward), repeats=4, allowed_inf=ALLOWED_INF)
        check_batch(samples, backward, targets=(0,))

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_neighbor_isolation(self, backward):
        samples = sequences()
        expected = piece(run(pack(samples, gradient_target=0 if backward else None), backward), samples, 0)
        changed = [samples[0], sequences(seed=20261105)[1]]
        actual = piece(run(pack(changed, gradient_target=0 if backward else None), backward), changed, 0)
        assert_bitwise(actual, expected, "changed neighbor, fixed target loss")

    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    def test_golden(self, backward):
        inputs = pack(sequences((3, 2), topk=16))
        actual = assert_replays(lambda: run(inputs, backward), repeats=3, allowed_inf=ALLOWED_INF)
        check_reference(inputs, actual, backward)
        assert_golden(f"dsa-{'bprop' if backward else 'fprop'}", inputs, actual, allowed_inf=ALLOWED_INF)
