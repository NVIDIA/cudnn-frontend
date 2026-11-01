# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D192 physical origins stay independent from logical sequence lengths."""

import pytest


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("stats_layout", ["token", "head"])
@pytest.mark.parametrize("cga,policy,total_hints", [(2, 0, False), (2, 1, True), (1, 0, True)])
def test_independent_origins(dtype_name, stats_layout, cga, policy, total_hints, monkeypatch):
    import cudnn
    import torch
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("native SM100/SM103/SM107 D192 physical origins")
    engine = "sdpa_fwd_prefill_sm107" if torch.cuda.get_device_capability() == (10, 7) else "sdpa_fwd_prefill_sm100"
    original = cudnn.pygraph.create_execution_plan

    def select_schedule(graph, engine_id, knobs=None):
        return original(graph, engine_id, {**(knobs or {}), cudnn.knob_type.SCHED_POLICY: policy})

    monkeypatch.setattr(cudnn.pygraph, "create_execution_plan", select_schedule)
    run((192, 128), dtype_name, stats_layout, total_hints, False, 1, engine_name=engine, cga=cga)


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("stats_layout", ["token", "head"])
@pytest.mark.parametrize("splits", [2, 9])
def test_split_independent_origins(dtype_name, stats_layout, splits):
    import torch
    from test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as run

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("native SM100/SM103/SM107 D192 split physical origins")
    engine = "sdpa_fwd_prefill_sm107" if torch.cuda.get_device_capability() == (10, 7) else "sdpa_fwd_prefill_sm100"
    run((192, 128), dtype_name, stats_layout, True, False, splits, engine_name=engine, cga=1)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)])
def test_split_origins_require_matching_extension(monkeypatch, cc):
    import cudnn
    from cudnn.frost import buffers
    from cudnn.sdpa.fwd import engines
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    monkeypatch.setattr(buffers, "_cutedsl_has_sm107", lambda: True)
    facts = SdpaGraphFacts(
        b=1,
        h_q=4,
        h_kv=4,
        s_q=128,
        s_kv=2048,
        d_qk=192,
        d_v=128,
        dtype=cudnn.data_type.BFLOAT16,
        thd=True,
        device_cc=cc,
        device_sm_count=148,
    )
    name = "sdpa_fwd_prefill_sm107" if cc == (10, 7) else "sdpa_fwd_prefill_sm100"
    caps = next(spec.capabilities for spec in engines.ENGINE_SPECS if spec.name == name)
    split = engines.SdpaFwdKnobs(cga=1, split_kv=2, pack_gqa=0, sched_policy=0)
    assert engines.mismatch(caps, facts, split) is None
    binder = cudnn._pybind_module._SdpaThdBinder

    class PreviousExtension:
        def __getattr__(self, attr):
            if attr == "supports_nonpaged_d192_split_physical_origins":
                return False
            return getattr(binder, attr)

    monkeypatch.setattr(cudnn._pybind_module, "_SdpaThdBinder", PreviousExtension())
    assert "nonpaged D192 split physical origins" in engines.mismatch(caps, facts, split)
    assert engines.mismatch(caps, facts, engines.SdpaFwdKnobs(cga=1, split_kv=1, pack_gqa=0, sched_policy=0)) is None
