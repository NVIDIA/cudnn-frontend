# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 general-tile FP8 / MXFP8 records name their body and replay it without selection."""

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]

_FP8, _FP8_ID, _FP8_LEGACY_ID = "sdpa_fwd_sm120_fp8_general", 20542, 20507
_MXFP8, _MXFP8_ID, _MXFP8_LEGACY_ID = "sdpa_fwd_sm120_mxfp8_general", 20543, 20518


def _identity(graph, name, template, split):
    assert graph.selected_engine.name == name
    compiled = graph._compiled_plans[graph._plan_index]._compiled
    assert compiled.kernel_template == template
    metadata = compiled.implementation_config
    assert metadata["split_kv"] == split
    return metadata


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("d,dv,split", [(128, 128, 1), (64, 64, 3), (192, 128, 1), (256, 256, 2)])
def test_fp8_public_record_names_and_replays_its_body(monkeypatch, dtype, d, dv, split):
    from cudnn.sdpa.fwd import heuristics
    import test_sdpa_prepared_fp8 as shared

    case = dict(arch="sm120", d=d, dv=dv, sq=97, skv=301, dtype=dtype, split_kv=split)
    g, vp, ws, bufs, _ = shared._case(**case)
    record = g.get_engine_and_knobs_at_index(g._plan_index)
    assert record[0] == _FP8_ID
    metadata = _identity(g, _FP8, "prefill_fp8", split)
    g.execute(vp, ws)
    shared._check(bufs, thd=False, sq=97, skv=301)

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit public replay consulted candidate selection")

    monkeypatch.setattr(heuristics, "recommend", forbidden)
    rebuilt, rvp, rws, rbufs, _ = shared._case(**case, record=record)
    assert rebuilt.get_engine_and_knobs_at_index(rebuilt._plan_index) == record
    assert _identity(rebuilt, _FP8, "prefill_fp8", split) == metadata
    rebuilt.execute(rvp, rws)
    shared._check(rbufs, thd=False, sq=97, skv=301)


def test_fp8_retired_legacy_record_is_declined_and_envelope_keeps_legacy():
    import test_sdpa_prepared_fp8 as shared

    g, *_ = shared._case(arch="sm120", d=128, sq=97, skv=301)
    _, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match="owned by|No execution plans"):
        shared._case(arch="sm120", d=128, sq=97, skv=301, record=(_FP8_LEGACY_ID, knobs))
    envelope, vp, ws, bufs, _ = shared._case(arch="sm120", d=80, sq=97, skv=301)
    assert envelope.selected_engine.name == "sdpa_fwd_prefill_sm120_fp8"
    envelope.execute(vp, ws)
    shared._check(bufs, thd=False, sq=97, skv=301)


@pytest.mark.parametrize("d_qk,d_v,split", [(128, 128, None), (64, 64, 3), (256, 128, None)])
def test_mxfp8_selected_body_and_retired_legacy_record(monkeypatch, d_qk, d_v, split):
    import cudnn
    import test_sdpa_fwd_mxfp8_sm120 as mx

    graphs = []
    select = mx._select_engine

    def recording(graph, *args, **kwargs):
        graphs.append(graph)
        return select(graph, *args, **kwargs)

    monkeypatch.setattr(mx, "_select_engine", recording)
    res = mx._run(2, 4, 2, 97, 301, d_qk=d_qk, d_v=d_v, split_kv=split)
    mx._check(res, d_qk=d_qk)
    (g,) = graphs
    _identity(g, _MXFP8, "prefill_mxfp8", g.plans[g._plan_index].knobs.split_kv or 1)
    assert g.get_engine_and_knobs_at_index(g._plan_index)[0] == _MXFP8_ID
    _, knobs = g.get_engine_and_knobs_at_index(g._plan_index)
    g.create_execution_plan(_MXFP8_LEGACY_ID, knobs)
    g.select_plan(len(g.plans) - 1)
    with pytest.raises((NotImplementedError, RuntimeError, cudnn.cudnnGraphNotSupportedError), match="owned by"):
        g.check_support()
