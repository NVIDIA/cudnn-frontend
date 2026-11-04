# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense exact-D128 SM120 per-tensor FP8 folds a split sink exactly once."""

import math
from dataclasses import replace

import pytest
import torch

import cudnn
from frost_test_utils import requires_blackwell_geforce, requires_dsl
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_sm120_fp8_sink_split_exact_dense_domain(dtype):
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, mismatch
    from test_sdpa_fwd_split_kv_sm120 import _sink_decode_facts

    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm120_fp8")
    knobs = SdpaFwdKnobs(sched_policy=0, split_kv=4, pack_gqa=False)
    facts = _sink_decode_facts(d_qk=128, d_v=128, dtype=dtype, dtype_o=cudnn.data_type.HALF, is_fp8=True)
    assert mismatch(caps, facts, knobs) is None
    for over in (
        dict(d_qk=112),
        dict(d_v=112),
        dict(d_qk=64, d_v=64),
        dict(d_qk=256, d_v=256),
        dict(d_qk=512, d_v=512),
        dict(padded=True),
        dict(thd=True),
        dict(has_paged_kv=True),
        dict(is_fp8=False, is_mxfp8=True),
    ):
        row = caps if not over.get("is_mxfp8") else next(s.capabilities for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm120_mxfp8")
        reason = mismatch(row, replace(facts, **over), knobs)
        assert reason is not None, over
        if "d_qk" in over or "d_v" in over:
            assert mismatch(caps, replace(facts, has_sink=False, **over), knobs) is None


def _check(bufs, *, sq, skv, stats_log2):
    # Keep the established FP8 output/Amax tolerances; Stats also checks that
    # only the final combine converts the natural-log partials to base two.
    shared._check({k: v for k, v in bufs.items() if k != "lse"}, thd=False, sq=sq, skv=skv)
    if "lse" in bufs:
        _, stats, _ = shared._reference(bufs, thd=False, sq=sq, skv=skv)
        torch.testing.assert_close(bufs["lse"], stats * (math.log2(math.e) if stats_log2 else 1), atol=2e-3, rtol=2e-4)
    assert torch.all(bufs["o_storage"][..., 128:] == 12)


@pytest.mark.parametrize("split", [2, 9])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_sm120_fp8_sink_split_rebind_replay(split, dtype, output, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    seen = []
    load = api_dsl._load_sm120_kernel_module

    def record(flavor, params, **kwargs):
        seen.append((params, kwargs))
        return load(flavor, params, **kwargs)

    monkeypatch.setattr(api_dsl, "_load_sm120_kernel_module", record)
    sq, skv, stats_log2 = (1 if split == 2 else 17), 256, split == 9
    sink = torch.tensor([-120, math.log(skv), math.log(skv) + 2, 120], device="cuda").view(1, 4, 1, 1)
    g, vp, ws, bufs, tensors = shared._case(
        arch="sm120",
        sq=sq,
        skv=skv,
        split_kv=split,
        dtype=dtype,
        output_dtype=output,
        sink=sink,
        output_padding=16,
        sdpa_kwargs=dict(stats_use_log2=stats_log2),
    )
    assert any(p.split_kv == split and not p.has_sink and kw["fp8"] and kw["pertensor"] for p, kw in seen)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared.spec.has_sink and prepared.spec.split == split and prepared.spec.combine is not None
    assert prepared.spec.combine.o.dtype in ("float16", "bfloat16")
    # A positive V mean makes duplicate sink mass a large O/Amax error.
    bufs["v"].copy_((bufs["v"].float() * 0.2 + 1).to(dtype))
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "lse", "amax_o", "sinks", "descale_q", "descale_k", "descale_v", "scale_o"):
                bufs[name] = bufs[name].clone()
                vp[tensors[name]] = bufs[name]
            bufs["q"].copy_((bufs["q"].float() * -0.7).to(dtype))
            # Every head now has material sink mass, including the head that
            # sets global Amax_O; a repeated fold cannot hide behind head 0.
            bufs["sinks"].fill_(math.log(skv) + 1)
            bufs["descale_q"].fill_(1.7)
            bufs["descale_v"].fill_(0.3)
            bufs["scale_o"].fill_(1.9)
            bufs["o_storage"] = bufs["o_storage"].clone()
            bufs["o"] = bufs["o_storage"][..., :128].transpose(1, 2)
            vp[tensors["o"]] = bufs["o"]
            ws = torch.empty_like(ws)
        g.execute(vp, ws)
        _check(bufs, sq=sq, skv=skv, stats_log2=stats_log2)
        with shared._cuda_graph() as capture:
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                g.execute(vp, ws)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(capture):
                torch.cuda.set_sync_debug_mode("error")
                try:
                    g.execute(vp, ws)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            # A later execute rebinds storage without redirecting captured pointers.
            fresh = {name: value.clone() for name, value in bufs.items()}
            fresh["o"] = fresh["o_storage"][..., :128].transpose(1, 2)
            rebound = {t: fresh[name] for name, t in tensors.items() if name in fresh}
            fresh["sinks"].add_(2)
            fresh_ws = torch.empty_like(ws)
            g.execute(rebound, fresh_ws)
            _check(fresh, sq=sq, skv=skv, stats_log2=stats_log2)
            bufs["sinks"].sub_(0.5)
            bufs["scale_o"].fill_(1.2)
            bufs["v"].copy_((bufs["v"].float() * 0.8).to(dtype))
            for name in ("o", "lse", "amax_o"):
                bufs[name].fill_(float("nan"))
            capture.replay()
            _check(bufs, sq=sq, skv=skv, stats_log2=stats_log2)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_fp8_sink_split_packed_keyless_rows(dtype, output):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    sq, skv = 257, 128
    # The standalone contract serves Sq > Skv; the graph's shared validator
    # rejects that bottom-right geometry independently of engine eligibility.
    bufs = {
        "q": torch.zeros((2, sq, 4, 128), device="cuda", dtype=dtype).transpose(1, 2),
        "k": torch.zeros((2, skv, 2, 128), device="cuda", dtype=dtype).transpose(1, 2),
        "v": torch.ones((2, skv, 2, 128), device="cuda", dtype=dtype).transpose(1, 2),
        "o": torch.empty((2, sq, 4, 128), device="cuda", dtype=output).transpose(1, 2),
        "lse": torch.empty((2, 4, sq), device="cuda"),
        "sinks": torch.tensor([-120, 0, 5, 120], device="cuda", dtype=torch.float32).view(1, 4, 1, 1),
        "amax_o": torch.empty(1, device="cuda"),
    }
    api = SdpaFwdDslSm120(
        **{"sample_" + name: bufs[name] for name in ("q", "k", "v", "o", "lse")},
        has_sink=True,
        pertensor_fp8=True,
        is_causal=True,
        causal_bottom_right=True,
        stats_log2=True,
        split_kv=9,
        sched_policy=0,
        pack_gqa=True,
    )
    assert api.check_support()
    ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)
    api.compile()
    assert api._prepared_fp8 and api._dense_spec.has_sink and api._dense_spec.combine is not None
    for name, value in (("descale_q", 0.8), ("descale_k", 0.9), ("descale_v", 0.7), ("scale_o", 1.3)):
        bufs[name] = torch.full((1,), value, device="cuda")

    def run():
        api.execute(
            **{name + "_tensor": bufs[name] for name in ("q", "k", "v", "o", "lse")},
            **{name: bufs[name] for name in ("sinks", "amax_o", "descale_q", "descale_k", "descale_v", "scale_o")},
            workspace=ws,
        )

    def check(value):
        # Exact zero-score oracle in float64: each visible key has unit mass.
        count = (torch.arange(sq, device="cuda", dtype=torch.float64) + skv - sq + 1).clamp(0, skv).view(1, 1, sq)
        log_count = count.log()
        stats = torch.logaddexp(log_count, bufs["sinks"].double().view(1, 4, 1))
        unscaled = (log_count - stats).exp() * bufs["descale_v"].double() * value
        ref = (unscaled * bufs["scale_o"].double()).unsqueeze(-1).expand(2, 4, sq, 128)
        torch.testing.assert_close(bufs["o"].float(), ref.float(), atol=0.03, rtol=0.03)
        torch.testing.assert_close(bufs["lse"], (stats * math.log2(math.e)).expand(2, 4, sq).float(), atol=2e-3, rtol=2e-4)
        torch.testing.assert_close(bufs["amax_o"][0], unscaled.amax().float(), atol=0.03, rtol=0.03)
        assert torch.count_nonzero(bufs["o"][:, :, : sq - skv].float()) == 0

    run()
    check(1)
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            run()
        bufs["sinks"].add_(2)
        bufs["v"].fill_(0.5)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        check(0.5)


def test_sm120_fp8_sink_split_without_optional_outputs():
    sink = torch.full((1, 4, 1, 1), math.log(256), device="cuda")
    g, vp, ws, bufs, _ = shared._case(
        arch="sm120",
        sq=1,
        skv=256,
        split_kv=9,
        stats=False,
        amax=False,
        output_dtype=torch.float8_e5m2,
        sink=sink,
        output_padding=16,
    )
    g.execute(vp, ws)
    _check(bufs, sq=1, skv=256, stats_log2=False)


@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_sm120_fp8_sink_split_standalone(output):
    from test_sdpa_prepared_fp8_split import test_prepared_fp8_split_standalone_default_scales

    test_prepared_fp8_split_standalone_default_scales("sm120", 128, 128, output, sink=True)
