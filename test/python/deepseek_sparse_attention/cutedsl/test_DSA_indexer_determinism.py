# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise replay and batch invariance for indexer scores and gradients."""

import pytest
import torch

from cudnn import DSA
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    _batched_ratio_causal_mask,
    _dense_indexer_predict_distribution,
    _indexer_predict_distribution,
    check_ref_dense_indexer_backward,
    check_ref_indexer_backward,
    check_ref_indexer_forward,
)
from deepseek_sparse_attention.cutedsl.test_DSA_determinism import supported_runtime

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
SCORE_INF = {"scores": -torch.inf}
SM_SCALE = 128**-0.5
BACKWARD_CASES = [("sparse", 32), ("sparse", 64), ("dense", 64)]
BACKWARD_IDS = ["sparse-h32", "sparse-h64", "dense-h64"]


def forward_sample(*, queries=17, keys=129, heads=64, weight_dtype=torch.bfloat16, offset=8, seed=20261104, device="cuda"):
    rng = torch.Generator().manual_seed(seed)
    return {
        "q": (torch.randn(1, queries, heads, 128, generator=rng) * 0.25).to(device=device, dtype=torch.bfloat16),
        "k": (torch.randn(1, keys, 1, 128, generator=rng) * 0.25).to(device=device, dtype=torch.bfloat16),
        "w": (torch.rand(1, queries, heads, generator=rng) + 0.1).to(device=device, dtype=weight_dtype),
        "q_causal_offsets": torch.tensor([offset], dtype=torch.int32, device=device),
    }


def batch(samples):
    return {name: torch.cat([sample[name] for sample in samples]) for name in samples[0]}


def forward_run(inputs, ratio=1, **options):
    local = {name: value.clone() for name, value in inputs.items()}
    before = snapshot(local)
    output = DSA.indexer_forward_wrapper(**local, ratio=ratio, **options)
    assert tuple(output.keys()) == ("scores",)
    assert_bitwise(local, before, "forward inputs changed")
    return snapshot({"scores": output["scores"]}, allowed_inf=SCORE_INF)


def forward_reference(inputs, actual, ratio=1):
    check_ref_indexer_forward(
        inputs["q"],
        inputs["k"],
        inputs["w"],
        actual["scores"],
        ratio,
        q_causal_offsets=inputs["q_causal_offsets"],
        compute_dtype=torch.float64,
    )


def backward_sample(kind, *, queries=17, keys=256, heads=64, topk=128, ratio=1, invalid=False, seed=20261104):
    source = forward_sample(queries=queries, keys=keys, heads=heads, seed=seed, device="cpu")
    q, k, w = source["q"], source["k"].squeeze(2), source["w"]
    rng = torch.Generator().manual_seed(seed + 100)
    inputs = {"index_q": q, "index_k": k, "weights": w}
    if kind == "sparse":
        indices = torch.stack([torch.randperm(keys, generator=rng)[:topk] for _ in range(queries)]).unsqueeze(0).int()
        if invalid:
            indices[..., ::7] = -1
        predict = _indexer_predict_distribution(q.float(), k.float(), w.float(), indices, SM_SCALE)
        target = torch.rand(1, queries, topk, generator=rng).masked_fill(indices < 0, 0)
        inputs.update(topk_indices=indices, index_score=predict, attn_score=target / target.sum(-1, keepdim=True))
    else:
        offsets = source["q_causal_offsets"]
        _, scores = _dense_indexer_predict_distribution(q.float(), k.float(), w.float(), SM_SCALE, ratio, offsets, return_scores=True)
        valid = _batched_ratio_causal_mask(queries, keys, ratio, "cpu", 1, offsets)
        target = torch.rand(1, queries, keys, generator=rng).masked_fill(~valid, 0)
        inputs.update(
            index_score=scores.masked_fill(~valid, 0),
            index_lse=torch.logsumexp(scores, -1),
            attn_score=target,
            attn_l1norm=target.sum(-1),
            q_causal_offsets=offsets,
        )
    return {name: value.contiguous().cuda() for name, value in inputs.items()}


def backward_run(inputs, kind, dk_dtype=torch.bfloat16, *, ratio=1, global_indices=False, planes=96, grad_loss=0.75):
    local = {name: value.clone() for name, value in inputs.items()}
    if global_indices:
        offsets = torch.arange(local["index_q"].shape[0], device="cuda", dtype=torch.int32)[:, None, None] * local["index_k"].shape[1]
        ids = local["topk_indices"]
        local["topk_indices"] = torch.where(ids >= 0, ids + offsets, ids)
    mutable = ("attn_score",) if kind == "sparse" else ("attn_score", "index_score")
    before = snapshot({name: value for name, value in local.items() if name not in mutable})
    grads = {
        "d_index_q": torch.full_like(local["index_q"], torch.nan),
        "d_weights": torch.full_like(local["weights"], torch.nan),
        "d_index_k": torch.full_like(local["index_k"], torch.nan, dtype=dk_dtype),
    }
    b, queries = local["index_q"].shape[:2]
    options = dict(grad_loss=torch.tensor(grad_loss, device="cuda", dtype=torch.float32), loss_coeff=float(b * queries), sm_scale=SM_SCALE, **grads)
    if kind == "sparse":
        from cudnn.deepseek_sparse_attention.indexer_backward.deterministic_dk_sm100 import deterministic_dk_workspace_bytes

        size = deterministic_dk_workspace_bytes(b, local["index_k"].shape[1], 128, planes)
        options.update(
            workspace=torch.full((size,), 0xA5, device="cuda", dtype=torch.uint8),
            deterministic=True,
            deterministic_dk_planes=planes,
            topk_indices_global=global_indices,
        )
        output = DSA.indexer_backward_wrapper(**local, **options)
    else:
        output = DSA.dense_indexer_backward_wrapper(**local, ratio=ratio, **options)
    assert tuple(output.keys()) == ("d_index_q", "d_weights", "d_index_k")
    assert_bitwise({name: value for name, value in local.items() if name not in mutable}, before, "backward inputs changed")
    return snapshot({**{name: output[name] for name in grads}, **{name: local[name] for name in mutable}})


def backward_reference(inputs, actual, kind, ratio=1):
    common = dict(
        index_q=inputs["index_q"],
        weights=inputs["weights"],
        index_k=inputs["index_k"],
        d_index_q_actual=actual["d_index_q"],
        d_weights_actual=actual["d_weights"],
        d_index_k_actual=actual["d_index_k"],
        sm_scale=SM_SCALE,
        grad_scale=0.75,
        cos_min=0.99,
        rms_rel_max=0.02,
    )
    if kind == "sparse":
        check_ref_indexer_backward(**common, attn_score=inputs["attn_score"], index_score=inputs["index_score"], topk_indices=inputs["topk_indices"])
    else:
        check_ref_dense_indexer_backward(
            **common, attn_score_raw=inputs["attn_score"], attn_l1norm=inputs["attn_l1norm"], ratio=ratio, q_causal_offsets=inputs["q_causal_offsets"]
        )


def check_batch(samples, run):
    for target in range(len(samples)):
        expected = run(samples[target])
        for order in ((0, 1), (1, 0), (target, target)):
            actual = run(batch([samples[index] for index in order]))
            position = order.index(target)
            assert_bitwise({name: value[position : position + 1] for name, value in actual.items()}, expected, f"batch {order}, sequence {target}")


class TestDSAIndexerForward:
    @pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
    @pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
    @pytest.mark.parametrize("ratio", [1, 4], ids=["ratio1", "ratio4"])
    def test_replay(self, heads, weight_dtype, ratio):
        inputs = batch([forward_sample(heads=heads, weight_dtype=weight_dtype, seed=20261104 + i) for i in range(2)])
        actual = assert_replays(lambda: forward_run(inputs, ratio), allowed_inf=SCORE_INF)
        forward_reference(inputs, actual, ratio)

    @pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
    @pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
    @pytest.mark.parametrize("ratio", [1, 4], ids=["ratio1", "ratio4"])
    def test_batch_invariance(self, heads, weight_dtype, ratio):
        samples = [forward_sample(heads=heads, weight_dtype=weight_dtype, seed=20261104 + i) for i in range(2)]
        check_batch(samples, lambda inputs: forward_run(inputs, ratio))

    @pytest.mark.parametrize(
        "queries,keys,offset", [(127, 127, 8), (128, 128, 8), (129, 129, 8), (17, 129, -17)], ids=["q127-k127", "q128-k128", "q129-k129", "empty-rows"]
    )
    def test_boundary(self, queries, keys, offset):
        samples = [forward_sample(queries=queries, keys=keys, offset=offset, seed=20261104 + i) for i in range(2)]
        inputs = batch(samples)
        actual = assert_replays(lambda: forward_run(inputs), repeats=4, allowed_inf=SCORE_INF)
        forward_reference(inputs, actual)
        check_batch(samples, forward_run)

    @pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32], ids=["w-bf16", "w-fp32"])
    def test_ragged_batch(self, weight_dtype):
        samples = [forward_sample(queries=q, keys=k, weight_dtype=weight_dtype, seed=20261104 + i) for i, (q, k) in enumerate(((3, 7), (17, 129)))]
        expected = [forward_run(sample, 4) for sample in samples]
        for sample, result in zip(samples, expected):
            forward_reference(sample, result, 4)
        for order in ((0, 1), (1, 0), (0, 0)):
            chosen = [samples[index] for index in order]
            q_lengths = [sample["q"].shape[1] for sample in chosen]
            k_lengths = [sample["k"].shape[1] for sample in chosen]
            inputs = {name: torch.cat([sample[name].squeeze(0) for sample in chosen]) for name in ("q", "k", "w")}
            inputs.update(q_causal_offsets=torch.cat([sample["q_causal_offsets"] for sample in chosen]))
            for name, lengths in (("cu_seqlens_q", q_lengths), ("cu_seqlens_k", k_lengths)):
                inputs[name] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
            actual = assert_replays(lambda: forward_run(inputs, 4, max_seqlen_q=max(q_lengths), max_seqlen_k=max(k_lengths)), repeats=4, allowed_inf=SCORE_INF)[
                "scores"
            ]
            start = 0
            for index, q, k in zip(order, q_lengths, k_lengths):
                assert_bitwise({"scores": actual[start : start + q, :k].unsqueeze(0)}, expected[index], f"ragged {order}, sequence {index}")
                assert torch.isneginf(actual[start : start + q, k:]).all(), "ragged padding must stay masked"
                start += q

    def test_neighbor_isolation(self):
        target = forward_sample()
        first = forward_run(batch([target, forward_sample(seed=20261105)]))
        second = forward_run(batch([target, forward_sample(seed=20261106)]))
        assert_bitwise({"scores": first["scores"][:1]}, {"scores": second["scores"][:1]}, "changed neighbor")

    def test_golden(self):
        inputs = forward_sample(queries=3)
        actual = assert_replays(lambda: forward_run(inputs), repeats=3, allowed_inf=SCORE_INF)
        forward_reference(inputs, actual)
        assert_golden("dsa-indexer-fprop", inputs, actual, allowed_inf=SCORE_INF)


class TestDSAIndexerBackward:
    @pytest.mark.parametrize("kind,heads", BACKWARD_CASES, ids=BACKWARD_IDS)
    @pytest.mark.parametrize("dk_dtype", [torch.bfloat16, torch.float32], ids=["dk-bf16", "dk-fp32"])
    def test_replay(self, kind, heads, dk_dtype):
        inputs = batch([backward_sample(kind, heads=heads, seed=20261104 + i) for i in range(2)])
        backward_reference(inputs, backward_run(inputs, kind, dk_dtype), kind)
        assert_replays(lambda: backward_run(inputs, kind, dk_dtype))

    @pytest.mark.parametrize("kind,heads", BACKWARD_CASES, ids=BACKWARD_IDS)
    @pytest.mark.parametrize("dk_dtype", [torch.bfloat16, torch.float32], ids=["dk-bf16", "dk-fp32"])
    def test_batch_invariance(self, kind, heads, dk_dtype):
        samples = [backward_sample(kind, heads=heads, seed=20261104 + i) for i in range(2)]
        check_batch(samples, lambda inputs: backward_run(inputs, kind, dk_dtype))

    @pytest.mark.parametrize("global_indices", [False, True], ids=["local-ids", "global-ids"])
    def test_sparse_invalid_slots(self, global_indices):
        samples = [backward_sample("sparse", invalid=True, seed=20261104 + i) for i in range(2)]
        inputs = batch(samples)
        run = lambda values: backward_run(values, "sparse", torch.float32, global_indices=global_indices)
        actual = assert_replays(lambda: run(inputs))
        backward_reference(inputs, actual, "sparse")
        check_batch(samples, run)

    @pytest.mark.parametrize("topk", [896, 2048], ids=["topk896", "topk2048"])
    def test_sparse_persistent_rows(self, topk):
        samples = [backward_sample("sparse", queries=193, keys=2048, topk=topk, seed=20261104 + i) for i in range(2)]
        actual = assert_replays(lambda: backward_run(samples[0], "sparse", torch.float32), repeats=4)
        backward_reference(samples[0], actual, "sparse")
        check_batch(samples, lambda inputs: backward_run(inputs, "sparse", torch.float32))

    @pytest.mark.parametrize("kind", ["sparse", "dense"])
    def test_zero_loss_gradient(self, kind):
        inputs = backward_sample(kind)
        actual = assert_replays(lambda: backward_run(inputs, kind, torch.float32, grad_loss=0))
        for name in ("d_index_q", "d_weights", "d_index_k"):
            assert torch.count_nonzero(actual[name]).item() == 0, f"{name}: nonzero gradient for zero loss"

    @pytest.mark.parametrize("kind", ["sparse", "dense"])
    def test_neighbor_isolation(self, kind):
        target = backward_sample(kind)
        first = backward_run(batch([target, backward_sample(kind, seed=20261105)]), kind, torch.float32)
        second = backward_run(batch([target, backward_sample(kind, seed=20261106)]), kind, torch.float32)
        assert_bitwise({name: value[:1] for name, value in first.items()}, {name: value[:1] for name, value in second.items()}, "changed neighbor")

    @pytest.mark.parametrize("kind", ["sparse", "dense"])
    def test_golden(self, kind):
        inputs = backward_sample(kind, queries=1)
        actual = assert_replays(lambda: backward_run(inputs, kind, torch.float32), repeats=3)
        backward_reference(inputs, actual, kind)
        assert_golden(f"dsa-indexer-{kind}-bprop", inputs, actual)
