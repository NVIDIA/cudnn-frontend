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
