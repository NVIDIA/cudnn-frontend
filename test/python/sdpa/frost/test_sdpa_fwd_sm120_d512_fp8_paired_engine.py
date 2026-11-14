# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 wide-head FP8 / MXFP8 records name and fix their body across rebuild, rebind and retained capture."""

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

_FP8, _FP8_ID, _FP8_LEGACY_ID = "sdpa_fwd_sm120_d512_fp8_paired", 20533, 20507
_MXFP8, _MXFP8_LEGACY_ID = "sdpa_fwd_sm120_d512_mxfp8_paired", 20518


def _identity(graph, name, template, split):
    assert graph.selected_engine.name == name
    compiled = graph._compiled_plans[graph._plan_index]._compiled
    assert compiled.kernel_template == template
    metadata = compiled.implementation_config
    assert (metadata["query_tile"], metadata["kv_tile"], metadata["split_kv"]) == (64, 64, split)
    return metadata


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split", [1, 3])
def test_fp8_public_record_rebuild_rebind_and_retained_capture(monkeypatch, dtype, output, split):
    from cudnn.sdpa.fwd import heuristics
    import test_sdpa_prepared_fp8 as shared

    case = dict(arch="sm120", d=512, sq=97, skv=301, dtype=dtype, output_dtype=output, split_kv=split)
    g, vp, ws, bufs, tensors = shared._case(**case)
    record = g.get_engine_and_knobs_at_index(g._plan_index)
    assert record[0] == _FP8_ID
    metadata = _identity(g, _FP8, "prefill_d512_fp8", split)
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=97, skv=301)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    rebuilt, rvp, rws, rbufs, rtensors = shared._case(**case, record=record)
    assert rebuilt.get_engine_and_knobs_at_index(rebuilt._plan_index) == record
    assert _identity(rebuilt, _FP8, "prefill_d512_fp8", split) == metadata

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(capture, stream=stream):
            rebuilt.execute(rvp, rws)
        torch.cuda.current_stream().wait_stream(stream)
        rbufs["descale_q"].fill_(0.5)
        rbufs["scale_o"].fill_(1.7)
        for name in ("o", "lse", "amax_o"):
            rbufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(rbufs, thd=False, sq=97, skv=301)
    finally:
        capture.reset()
    for name in ("q", "k", "v", "o", "lse", "amax_o", "descale_q", "scale_o"):
        rbufs[name] = rbufs[name].clone()
        rvp[rtensors[name]] = rbufs[name]
    rebuilt.execute(rvp, torch.empty_like(rws))
    shared._check(rbufs, thd=False, sq=97, skv=301)


def test_fp8_retired_legacy_record_is_declined_and_envelope_keeps_legacy():
    import test_sdpa_prepared_fp8 as shared

    g, *_ = shared._case(arch="sm120", d=512, sq=97, skv=301)
    _, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="owned by|No execution plans"):
        shared._case(arch="sm120", d=512, sq=97, skv=301, record=(_FP8_LEGACY_ID, knobs))
    envelope, vp, ws, bufs, _ = shared._case(arch="sm120", d=384, dv=320, sq=97, skv=301)
    assert envelope.selected_engine.name == "sdpa_fwd_prefill_sm120_fp8"
    assert envelope._compiled_plans[envelope._plan_index]._compiled.implementation_config is None


@pytest.mark.parametrize("split", [None, 3])
def test_mxfp8_selected_body_and_retired_legacy_record(monkeypatch, split):
    import cudnn
    import test_sdpa_fwd_mxfp8_sm120 as mx

    graphs = []
    select = mx._select_engine

    def recording(graph, *args, **kwargs):
        graphs.append(graph)
        return select(graph, *args, **kwargs)

    monkeypatch.setattr(mx, "_select_engine", recording)
    res = mx._run(2, 4, 2, 97, 301, d_qk=512, split_kv=split)
    mx._check(res, d_qk=512)
    (g,) = graphs
    _identity(g, _MXFP8, "prefill_d512_mxfp8", g.plans[g._plan_index].knobs.split_kv or 1)
    _, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    g.create_execution_plan(_MXFP8_LEGACY_ID, knobs)
    g.select_plan(len(g.plans) - 1)
    with pytest.raises((NotImplementedError, RuntimeError, cudnn.cudnnGraphNotSupportedError), match="owned by"):
        g.check_support()
