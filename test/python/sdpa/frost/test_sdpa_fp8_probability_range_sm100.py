# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
"""FP8 probability casts retain collectively significant low-probability keys."""

import math

import cudnn
import pytest
import torch

from determinism_utils import assert_bitwise, snapshot
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_fwd_fp8_sm100 as fp8

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("owner,cga", [("multi_q", 1), ("multi_q", 2), ("dual_kv", 1)])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_peaked_row_retains_tail_mass(monkeypatch, owner, cga, in_key, out_dtype):
    # Both ends have an anchor, so either traversal direction sees the peak
    # before the interior keys. Splitting would hide this loss by giving some
    # partitions only the flat tail, so pin an unsplit implementation.
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs

    sq, skv, d = 16, 32768, 128
    q = torch.zeros(1, 1, sq, d, device="cuda")
    k = torch.zeros(1, 1, skv, d, device="cuda")
    v = torch.zeros_like(k)
    q[..., 0] = 1
    v[..., 1] = 1
    for anchor in (0, skv - 1):
        k[:, :, anchor, 0] = 1
        v[:, :, anchor, :] = 0
        v[:, :, anchor, 0] = 1
    name = "sdpa_fwd_sm100_d128_fp8_" + owner
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    engine_id = family.offered_ids()[name]
    calls = []
    execute = cudnn.pygraph.execute

    def pin(graph, _names, **_kwargs):
        knobs = SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
        graph.create_execution_plan(engine_id, knobs)
        graph.select_plan(len(graph.plans) - 1)
        return graph.plans[-1]

    def record(graph, vp, workspace, *args, **kwargs):
        calls.append((graph, vp, workspace))
        return execute(graph, vp, workspace, *args, **kwargs)

    monkeypatch.setattr(fp8, "_select_engine", pin)
    monkeypatch.setattr(cudnn.pygraph, "execute", record)
    result = fp8._run(1, 1, 1, sq, skv, in_key, out_dtype, scale=10.0, sdpa_kwargs={}, qkv_float=(q, k, v), return_lse=True)
    graph, vp, workspace = calls[-1]
    assert graph.selected_engine.name == name
    tail = (skv - 2) / (2 * math.exp(10) + skv - 2)
    # The shared FP8 oracle rounds P itself; use the mathematical attention
    # value here so matching its quantization cannot hide collective tail loss.
    expected = torch.zeros_like(result.output, dtype=torch.float32)
    expected[..., 0] = 1 - tail
    expected[..., 1] = tail
    torch.testing.assert_close(result.output.float(), expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(result.stats, result.reference_stats, rtol=1e-5, atol=1e-4)
    assert math.isfinite(result.amax)
    assert abs(result.amax - max(tail, 1 - tail)) <= 2e-3 + 2e-2 * max(tail, 1 - tail)
    baseline = snapshot({"o": result.output, "stats": result.stats})
    # Capture through the same public call and stream contract as inference.
    handle = cudnn.create_handle()
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture):
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            execute(graph, vp, workspace, handle=handle)
        result.output.fill_(float("nan"))
        result.stats.fill_(float("nan"))
        capture.replay()
        assert_bitwise({"o": result.output, "stats": result.stats}, baseline, "peaked row replay")
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


def _probability_fixture(owner, cga, in_key, b):
    from cudnn.engines.manifest import MANIFEST
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    from test_sdpa_prepared_fp8 import _case

    name = "sdpa_fwd_sm100_d128_fp8_" + owner
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    knobs = SdpaFwdKnobs(tile_m=128, tile_n=128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
    dtype = torch.float8_e4m3fn if in_key == "e4m3" else torch.float8_e5m2
    graph, vp, workspace, bufs, _ = _case(
        b=b,
        heads=(1, 1),
        sq=16,
        skv=32768,
        dtype=dtype,
        record=(family.offered_ids()[name], knobs.to_public()),
    )
    assert graph.selected_engine.name == name
    for role in ("q", "k", "v"):
        bufs[role].zero_()
    bufs["q"][..., 0] = 1
    for anchor in (0, 32767):
        bufs["k"][:, :, anchor, 0] = 1
    bufs["v"][..., 1] = 1
    for anchor in (0, 32767):
        bufs["v"][:, :, anchor, 1] = 0
        bufs["v"][:, :, anchor, 0] = 1
    for role in ("descale_q", "descale_v", "scale_s", "descale_s", "scale_o"):
        bufs[role].fill_(1)
    bufs["descale_k"].fill_(10 * math.sqrt(128))
    logit = float(bufs["descale_k"].cpu()[0]) / math.sqrt(128)
    tail = 32766 / (2 * math.exp(logit) + 32766)
    expected_o = torch.zeros_like(bufs["o"], dtype=torch.float32)
    expected_o[..., 0], expected_o[..., 1] = 1 - tail, tail
    expected = {
        "o": expected_o,
        "lse": torch.full_like(bufs["lse"], math.log(2 * math.exp(logit) + 32766)),
        "amax_o": torch.full_like(bufs["amax_o"], max(tail, 1 - tail)),
    }
    return graph, vp, workspace, bufs, expected


def _check_probability_outputs(outputs, expected):
    torch.testing.assert_close(outputs["o"].float(), expected["o"], rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(outputs["lse"], expected["lse"], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(outputs["amax_o"], expected["amax_o"], rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("owner,cga", [("multi_q", 1), ("multi_q", 2), ("dual_kv", 1)])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
def test_probability_range_replay_and_batch_isolation(owner, cga, in_key):
    from determinism_utils import assert_replays

    graph, vp, workspace, bufs, expected = _probability_fixture(owner, cga, in_key, 1)
    inputs = snapshot({name: bufs[name] for name in ("q", "k", "v", "descale_q", "descale_k", "descale_v", "scale_o")})
    outputs = {name: bufs[name] for name in expected}
    handle = cudnn.create_handle()
    capture = torch.cuda.CUDAGraph()
    try:
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)

        def run():
            for value in outputs.values():
                value.fill_(float("nan"))
            graph.execute(vp, workspace, handle=handle)
            _check_probability_outputs(outputs, expected)
            return outputs

        baseline = assert_replays(run)
        with torch.cuda.graph(capture):
            cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
            graph.execute(vp, workspace, handle=handle)
        for value in outputs.values():
            value.fill_(float("nan"))
        capture.replay()
        assert_bitwise(outputs, baseline, "retained O/Stats/Amax replay")
        assert_bitwise({name: bufs[name] for name in inputs}, inputs, "inputs remain unchanged")
        # Preserve the actual quantized target payload and scale bytes across
        # batch insertion, duplication, movement and a changed neighbor.
        batched, pack, scratch, bigger, _ = _probability_fixture(owner, cga, in_key, 3)
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        for targets, neighbor_value in (((1, 2), 0.125), ((0, 2), 0.25)):
            bigger["q"].fill_(0)
            bigger["k"].fill_(0)
            bigger["v"].fill_(neighbor_value)
            for index in targets:
                for role in ("q", "k", "v"):
                    bigger[role][index : index + 1].copy_(inputs[role])
            for role in ("descale_q", "descale_k", "descale_v", "scale_o"):
                bigger[role].copy_(inputs[role])
            for role in outputs:
                bigger[role].fill_(float("nan"))
            batched.execute(pack, scratch, handle=handle)
            for index in targets:
                actual = {name: bigger[name][index : index + 1] for name in ("o", "lse")}
                actual["amax_o"] = bigger["amax_o"]
                _check_probability_outputs(actual, expected)
                assert_bitwise(actual, baseline, f"batch target {index}, neighbor {neighbor_value}")
    finally:
        capture.reset()
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("owner,cga", [("multi_q", 1), ("multi_q", 2), ("dual_kv", 1)])
def test_probability_cast_keeps_lazy_rescale_headroom(owner, cga):
    graph, vp, workspace, bufs, _ = _probability_fixture(owner, cga, "e4m3", 1)
    # Adjacent tile peaks differ by 0.5 nat (about 0.721 bit): one rise fits
    # the 0.75-bit allowance, while the next must update the row maximum.
    levels = torch.arange(32768, device="cuda") // 128 % 4
    bufs["k"].zero_()
    bufs["k"][0, 0, :, 0].copy_(levels.to(bufs["k"].dtype))
    bufs["v"].fill_(1)
    bufs["descale_k"].fill_(0.5 * math.sqrt(128))
    step = float(bufs["descale_k"].cpu()[0]) / math.sqrt(128)
    expected = {
        "o": torch.ones_like(bufs["o"], dtype=torch.float32),
        "lse": torch.full_like(bufs["lse"], math.log(8192 * sum(math.exp(step * x) for x in range(4)))),
        "amax_o": torch.ones_like(bufs["amax_o"]),
    }
    outputs = {name: bufs[name] for name in expected}
    handle = cudnn.create_handle()
    try:
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        for _ in range(2):
            for value in outputs.values():
                value.fill_(float("nan"))
            graph.execute(vp, workspace, handle=handle)
            _check_probability_outputs(outputs, expected)
    finally:
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("owner,cga", [("multi_q", 1), ("multi_q", 2), ("dual_kv", 1)])
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
def test_probability_range_golden(owner, cga, in_key):
    from determinism_utils import assert_golden

    graph, vp, workspace, bufs, expected = _probability_fixture(owner, cga, in_key, 1)
    outputs = {name: bufs[name] for name in expected}
    handle = cudnn.create_handle()
    try:
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        graph.execute(vp, workspace, handle=handle)
        _check_probability_outputs(outputs, expected)
        baseline = snapshot(outputs)
        for value in outputs.values():
            value.fill_(float("nan"))
        graph.execute(vp, workspace, handle=handle)
        assert_bitwise(outputs, baseline, "golden repeat")
        _check_probability_outputs(outputs, expected)
        assert_golden(
            f"sdpa_fp8_probability_{owner}_cga{cga}_{in_key}_bf16",
            {name: bufs[name] for name in ("q", "k", "v", "descale_q", "descale_k", "descale_v", "scale_o")},
            outputs,
        )
    finally:
        cudnn.destroy_handle(handle)


@pytest.mark.parametrize("owner,cga", [("multi_q", 1), ("multi_q", 2), ("dual_kv", 1)])
def test_probability_range_warm_refill(owner, cga):
    from warm_stress import StressCase, assert_warm_stress, stress_validation

    graph, vp, workspace, bufs, expected = _probability_fixture(owner, cga, "e4m3", 1)
    outputs = {name: bufs[name] for name in expected}
    handle = cudnn.create_handle()
    cold_launches = 0
    try:
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)

        def launch():
            nonlocal cold_launches
            graph.execute(vp, workspace, handle=handle)
            # Check the stress helper's actual cold baseline against the
            # mathematical oracle, before it is used for byte comparisons.
            if cold_launches < 2:
                _check_probability_outputs(outputs, expected)
                cold_launches += 1

        validation = stress_validation(cluster=cga)
        assert_warm_stress(
            StressCase(
                name=f"fp8_probability_{owner}_cga{cga}",
                launch=launch,
                outputs=outputs,
                thread_blocks=cga,
                refills_per_launch=128 if owner == "dual_kv" else 256,
            ),
            validation=validation,
            warm=512,
            poisoned=512,
        )
        _check_probability_outputs(outputs, expected)
    finally:
        cudnn.destroy_handle(handle)
