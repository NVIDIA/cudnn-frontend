# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexer FP8/MXFP8 replay with fixed quantized data and scale bytes."""

import pytest
import torch

from cudnn import DSA
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from deepseek_sparse_attention.cutedsl import mxfp8_scale_reference
from deepseek_sparse_attention.cutedsl.dsa_reference import ref_indexer_forward
from deepseek_sparse_attention.cutedsl.dsa_utils import (
    expand_mxfp8_scale,
    make_random_mxfp8_scale,
    pack_mxfp8_scales_thd,
    quantize_fp8_1x128,
    quantize_mxfp8,
)
from deepseek_sparse_attention.cutedsl.test_DSA_indexer_determinism import forward_sample

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
ALLOWED_INF = {"scores": -torch.inf, "lse": -torch.inf}


@pytest.fixture(params=["mxfp8", "fp8"], ids=["sm100-sm103-sm107-mxfp8", "sm90-fp8"])
def precision(request):
    if not torch.cuda.is_available():
        pytest.skip("Indexer FP8 requires CUDA")
    capability = torch.cuda.get_device_capability()
    admitted = ((10, 0), (10, 3), (10, 7)) if request.param == "mxfp8" else ((9, 0),)
    if capability not in admitted:
        pytest.skip(f"{request.param} requires {admitted}; found {capability}")
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error, cutedsl_state

    if not cutedsl_state()[0]:
        pytest.skip("Indexer FP8 requires CuTe DSL")
    reason = cutedsl_requirement_error("DSA indexer FP8") or cutedsl_arch_requirement_error(capability)
    if reason:
        pytest.skip(reason)
    return request.param


def sample(precision, **options):
    source = forward_sample(device="cpu", **options)
    for index, name in enumerate(("q", "k")):
        if precision == "mxfp8":
            scales = make_random_mxfp8_scale(
                (*source[name].shape[:-1], 4),
                device="cpu",
                seed=options.get("seed", 20261104) + 100 + index,
                exponent_min=-3,
                exponent_max=2,
            )
            source[name] = quantize_mxfp8(source[name], scales)
        else:
            source[name], scales = quantize_fp8_1x128(source[name])
        source[f"{name}_scale"] = scales
    return {name: value.cuda() for name, value in source.items()}


def pack(samples, precision, ragged=False):
    if ragged:
        result = {name: torch.cat([item[name].squeeze(0) for item in samples]) for name in ("q", "k", "w", "q_scale", "k_scale")}
        for name, key in (("cu_seqlens_q", "q"), ("cu_seqlens_k", "k")):
            lengths = [item[key].shape[1] for item in samples]
            result[name] = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
            result[f"max_seqlen_{key}"] = max(lengths)
    else:
        result = {name: torch.cat([item[name] for item in samples]) for name in ("q", "k", "w", "q_scale", "k_scale")}
    result["q_causal_offsets"] = torch.cat([item["q_causal_offsets"] for item in samples])
    if precision == "mxfp8":
        heads = samples[0]["q"].shape[2]
        if ragged:
            result["q_scale"], result["k_scale"], result["cu_seqlens_q_scale_padded"], result["cu_seqlens_k_scale_padded"] = pack_mxfp8_scales_thd(
                result["q_scale"],
                result["k_scale"],
                result["cu_seqlens_q"],
                result["cu_seqlens_k"],
                heads,
            )
        else:
            result["q_scale"] = mxfp8_scale_reference.pack_q_scale_bshd(result["q_scale"], heads)
            result["k_scale"] = mxfp8_scale_reference.pack_k_scale_bshd(result["k_scale"])
    return result


def run(inputs, precision, ratio=1):
    local = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}
    before = {name: value.clone() for name, value in local.items() if isinstance(value, torch.Tensor)}
    for name, value in before.items():
        assert torch.isfinite(value.float()).all(), f"{name}: non-finite input"
    output = DSA.indexer_forward_wrapper(**local, precision=precision, ratio=ratio, return_lse=precision == "fp8")
    expected_keys = ("scores", "lse") if precision == "fp8" else ("scores",)
    assert tuple(output.keys()) == expected_keys
    assert_bitwise({name: local[name] for name in before}, before, "quantized inputs/scales changed")
    return snapshot({name: output[name] for name in expected_keys}, allowed_inf=ALLOWED_INF)


def reference(item, actual, precision, ratio=1):
    values = {}
    for name in ("q", "k"):
        scale = expand_mxfp8_scale(item[f"{name}_scale"], 128) if precision == "mxfp8" else item[f"{name}_scale"].unsqueeze(-1)
        values[name] = item[name].float() * scale
    expected = {
        "scores": ref_indexer_forward(
            values["q"],
            values["k"],
            item["w"],
            ratio,
            q_causal_offsets=item["q_causal_offsets"],
            compute_dtype=torch.float64,
        )
    }
    if precision == "fp8":
        expected["lse"] = torch.logsumexp(expected["scores"], -1)
    for name, want in expected.items():
        assert torch.equal(torch.isneginf(actual[name]), torch.isneginf(want)), f"{name}: incorrect mask"
        tolerance = 5e-3 if precision == "mxfp8" or name == "lse" else 2e-3
        torch.testing.assert_close(actual[name], want, atol=tolerance, rtol=tolerance)


def check_batch(samples, precision, ratio=1, ragged=False):
    expected = [run(pack([item], precision), precision, ratio) for item in samples]
    for item, actual in zip(samples, expected):
        reference(item, actual, precision, ratio)
    for order in ((0, 1), (1, 0), (0, 0), (1, 1)):
        chosen = [samples[index] for index in order]
        actual = run(pack(chosen, precision, ragged), precision, ratio)
        start = 0
        for position, index in enumerate(order):
            queries, keys = samples[index]["q"].shape[1], samples[index]["k"].shape[1]
            if ragged:
                piece = {name: value[start : start + queries].unsqueeze(0) for name, value in actual.items()}
                assert torch.isneginf(piece["scores"][..., keys:]).all(), "ragged padding must stay masked"
                piece["scores"] = piece["scores"][..., :keys]
            else:
                piece = {name: value[position : position + 1] for name, value in actual.items()}
            assert_bitwise(piece, expected[index], f"batch {order}, sequence {index}")
            start += queries


class TestDSAIndexerFP8:
    @pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
    @pytest.mark.parametrize("ratio", [1, 4], ids=["ratio1", "ratio4"])
    def test_replay(self, precision, heads, ratio):
        item = sample(precision, heads=heads)
        inputs = pack([item], precision)
        actual = assert_replays(lambda: run(inputs, precision, ratio), allowed_inf=ALLOWED_INF)
        reference(item, actual, precision, ratio)

    @pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
    @pytest.mark.parametrize("ratio", [1, 4], ids=["ratio1", "ratio4"])
    def test_batch_invariance(self, precision, heads, ratio):
        check_batch([sample(precision, heads=heads, seed=20261104 + index) for index in range(2)], precision, ratio)

    @pytest.mark.parametrize(
        "queries,keys,offset",
        [(127, 127, 8), (128, 128, 8), (129, 129, 8), (17, 129, -17)],
        ids=["q127-k127", "q128-k128", "q129-k129", "empty-rows"],
    )
    def test_boundary(self, precision, queries, keys, offset):
        samples = [sample(precision, queries=queries, keys=keys, offset=offset, seed=20261104 + index) for index in range(2)]
        inputs = pack([samples[0]], precision)
        actual = assert_replays(lambda: run(inputs, precision), repeats=4, allowed_inf=ALLOWED_INF)
        reference(samples[0], actual, precision)
        check_batch(samples, precision)

    @pytest.mark.parametrize("heads", [32, 64], ids=["h32", "h64"])
    def test_ragged_batch(self, precision, heads):
        samples = [sample(precision, queries=q, keys=k, heads=heads, seed=20261104 + i) for i, (q, k) in enumerate(((3, 7), (17, 129)))]
        inputs = pack(samples, precision, ragged=True)
        assert_replays(lambda: run(inputs, precision, 4), repeats=4, allowed_inf=ALLOWED_INF)
        check_batch(samples, precision, ratio=4, ragged=True)

    def test_neighbor_isolation(self, precision):
        target = sample(precision)
        first = run(pack([target, sample(precision, seed=20261105)], precision), precision)
        second = run(pack([target, sample(precision, seed=20261106)], precision), precision)
        assert_bitwise({name: value[:1] for name, value in first.items()}, {name: value[:1] for name, value in second.items()}, "changed neighbor/scales")

    def test_scale_change(self, precision):
        original = sample(precision)
        changed = {name: value.clone() for name, value in original.items()}
        changed["q_scale"] = (changed["q_scale"].float() * 2).to(changed["q_scale"].dtype)
        assert_bitwise({"q": changed["q"], "k": changed["k"]}, {"q": original["q"], "k": original["k"]}, "fixed FP8 payload")
        first = run(pack([original], precision), precision)
        inputs = pack([changed], precision)
        second = assert_replays(lambda: run(inputs, precision), repeats=4, allowed_inf=ALLOWED_INF)
        reference(original, first, precision)
        reference(changed, second, precision)
        finite = torch.isfinite(first["scores"])
        assert torch.count_nonzero(first["scores"][finite] != second["scores"][finite]).item() > 0, "scale update had no effect"

    def test_golden(self, precision):
        item = sample(precision, queries=3)
        inputs = pack([item], precision)
        actual = assert_replays(lambda: run(inputs, precision), repeats=3, allowed_inf=ALLOWED_INF)
        reference(item, actual, precision)
        assert_golden(f"dsa-indexer-{precision}-fprop", inputs, actual, allowed_inf=ALLOWED_INF)
