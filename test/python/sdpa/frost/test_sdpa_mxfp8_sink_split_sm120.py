# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 dense exact-D128 MXFP8 split partials exclude the attention sink."""

import math
from dataclasses import replace

import pytest
import torch
import cudnn

from frost_test_utils import requires_blackwell_geforce, requires_dsl
import test_sdpa_prepared_mxfp8 as shared
from test_sdpa_prepared_fp8 import _cuda_graph

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dtype", [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2])
def test_sm120_mxfp8_sink_split_domain(dtype):
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, SdpaFwdKnobs, mismatch
    from test_sdpa_fwd_split_kv_sm120 import _sink_decode_facts

    caps = next(s.capabilities for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm120_mxfp8")
    knobs = SdpaFwdKnobs(sched_policy=0, split_kv=4, pack_gqa=False)
    facts = _sink_decode_facts(d_qk=128, d_v=128, dtype=dtype, dtype_o=cudnn.data_type.HALF, is_mxfp8=True)
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
        dict(o_block_scale=32),
        dict(has_epilogue_gate=True),
    ):
        assert mismatch(caps, replace(facts, **over), knobs) is not None, over


def _check(bufs, *, sq, skv):
    shared._check(bufs, thd=False, b=2, sq=sq, skv=skv)
    assert torch.all(bufs["o_storage"][..., 128:] == 12)


@pytest.mark.parametrize("split", [2, 9])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_sm120_mxfp8_sink_split_rebind_replay(split, dtype, output, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    seen = []
    load = api_dsl._load_sm120_kernel_module

    def record(flavor, params, **kwargs):
        seen.append((params, kwargs))
        return load(flavor, params, **kwargs)

    monkeypatch.setattr(api_dsl, "_load_sm120_kernel_module", record)
    sq, skv = (1 if split == 2 else 17), 256
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
        explicit_plan=True,
    )
    assert any(p.split_kv == split and not p.has_sink and kw["fp8"] and not kw["pertensor"] for p, kw in seen)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared.spec.has_sink and prepared.spec.split == split and prepared.spec.combine is not None
    assert prepared.spec.quant.sf_sizes
    # Positive mean makes duplicate sink mass visible in both O and Amax.
    bufs["v"].copy_((bufs["v"].float() * 0.2 + 1).to(dtype))
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "sf_q", "sf_k", "sf_v", "lse", "amax_o", "sinks"):
                bufs[name] = bufs[name].clone()
                vp[tensors[name]] = bufs[name]
            bufs["q"].copy_((bufs["q"].float() * -0.7).to(dtype))
            bufs["sinks"].fill_(math.log(skv) + 1)
            shared._change_scales(bufs)
            bufs["o_storage"] = bufs["o_storage"].clone()
            bufs["o"] = bufs["o_storage"][..., :128].transpose(1, 2)
            vp[tensors["o"]] = bufs["o"]
            ws = torch.empty_like(ws)
        g.execute(vp, ws)
        _check(bufs, sq=sq, skv=skv)
        with _cuda_graph() as capture:
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
            fresh = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in bufs.items()}
            fresh["o"] = fresh["o_storage"][..., :128].transpose(1, 2)
            rebound = {t: fresh[name] for name, t in tensors.items() if name in fresh}
            fresh["sinks"].add_(2)
            fresh_ws = torch.empty_like(ws)
            g.execute(rebound, fresh_ws)
            _check(fresh, sq=sq, skv=skv)
            bufs["sinks"].sub_(0.5)
            shared._change_scales(bufs)
            bufs["v"].copy_((bufs["v"].float() * 0.8).to(dtype))
            for name in ("o", "lse", "amax_o"):
                bufs[name].fill_(float("nan"))
            capture.replay()
            _check(bufs, sq=sq, skv=skv)


def test_sm120_mxfp8_sink_split_optional_outputs():
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
        explicit_plan=True,
    )
    g.execute(vp, ws)
    _check(bufs, sq=1, skv=256)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("stats_log2", [False, True], ids=["ln", "log2"])
def test_sm120_mxfp8_sink_split_standalone_keyless(dtype, stats_log2):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    sq, skv = 257, 128
    _, _, _, bufs, _ = shared._case(arch="sm120", sq=sq, skv=skv, dtype=dtype, output_padding=16, explicit_plan=True)
    bufs["q"].zero_()
    bufs["k"].zero_()
    bufs["sinks"] = torch.tensor([-120, -float("inf"), 5, 120], device="cuda").view(1, 4, 1, 1)
    api = SdpaFwdDslSm120(
        **{"sample_" + name: bufs[name] for name in ("q", "k", "v", "o", "lse")},
        has_sink=True,
        is_causal=True,
        causal_bottom_right=True,
        stats_log2=stats_log2,
        split_kv=9,
        sched_policy=0,
        pack_gqa=True,
        tile_m=64 if stats_log2 else 128,
        tile_n=64 if stats_log2 else 128,
    )
    assert api.check_support()
    api.compile()
    assert api._prepared_mxfp8 and api._dense_spec.has_sink
    ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)

    def run():
        api.execute(
            **{name + "_tensor": bufs[name] for name in ("q", "k", "v", "o", "lse")},
            **{name: bufs[name] for name in ("sf_q", "sf_k", "sf_v", "sinks", "amax_o")},
            workspace=ws,
        )

    def check():
        q_rows = torch.arange(sq, device="cuda").view(sq, 1)
        k_cols = torch.arange(skv, device="cuda").view(1, skv)
        scores = torch.where(k_cols <= q_rows + skv - sq, 0.0, -float("inf")).double().view(1, 1, sq, skv)
        stats = torch.logaddexp(scores.logsumexp(-1), bufs["sinks"].double().view(1, 4, 1))
        weights = torch.where(torch.isfinite(stats.unsqueeze(-1)), (scores - stats.unsqueeze(-1)).exp(), 0)
        v = bufs["v"].double() * bufs["dq_v"].double() * bufs["factor_v"]
        expected = weights @ v.repeat_interleave(2, 1)
        torch.testing.assert_close(bufs["o"].double(), expected, atol=0.03, rtol=0.03)
        torch.testing.assert_close(bufs["lse"].double(), (stats * (math.log2(math.e) if stats_log2 else 1)).expand(2, 4, sq), atol=0.01, rtol=0.01)
        torch.testing.assert_close(bufs["amax_o"][0].double(), expected.abs().amax(), atol=0.03, rtol=0.03)
        assert torch.count_nonzero(bufs["o"][:, :, : sq - skv].float()) == 0
        assert torch.all(bufs["o_storage"][..., 128:] == 12)

    run()
    check()
    with _cuda_graph() as capture:
        with torch.cuda.graph(capture):
            run()
        bufs["sinks"].add_(2)
        shared._change_scales(bufs)
        bufs["v"].copy_((bufs["v"].float() * 0.7).to(dtype))
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        check()
