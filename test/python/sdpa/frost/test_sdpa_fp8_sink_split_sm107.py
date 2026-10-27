# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Rubin FP8 split partials exclude sinks until the final combine."""

from dataclasses import replace

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128)])
@pytest.mark.parametrize("split", [2, 9])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_fp8_sink_split_rebind_replay(d, dv, split, dtype, output, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    seen = []
    original = api_dsl._load_kernel_template

    def record(filename, params, tag):
        seen.append(filename)
        return original(filename, params, tag)

    monkeypatch.setattr(api_dsl, "_load_kernel_template", record)
    sq = 1 if split == 2 else 17
    sink = torch.linspace(-120, 120, 4, device="cuda").view(1, 4, 1, 1)
    g, vp, ws, bufs, tensors = shared._case(arch="sm107", d=d, dv=dv, sq=sq, skv=256, split_kv=split, dtype=dtype, output_dtype=output, sink=sink)
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=split == 9 and d != 192, sched_policy=split % 3))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    expected = "sm107/prefill_d192_d128_fp8.py" if d == 192 else "sm107/prefill_d128_fp8.py"
    assert expected in seen and not any(name.startswith("sm100/") for name in seen)
    prepared = g._compiled_plans[g._plan_index]._prepared
    assert prepared.spec.has_sink and prepared.spec.split == split and prepared.spec.combine is not None
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "o", "lse", "amax_o", "sinks", "descale_q", "scale_o"):
                bufs[name] = bufs[name].clone()
                vp[tensors[name]] = bufs[name]
            bufs["sinks"].fill_(5)
            bufs["scale_o"].fill_(1.9)
            ws = torch.empty_like(ws)
        g.execute(vp, ws)
        shared._check(bufs, thd=False, sq=sq, skv=256)
        with shared._cuda_graph() as capture:
            torch.cuda.set_sync_debug_mode("error")
            try:
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                g.execute(vp, ws)
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            finally:
                torch.cuda.set_sync_debug_mode("default")
            with torch.cuda.graph(capture):
                g.execute(vp, ws)
            fresh = {name: value.clone() for name, value in bufs.items()}
            rebound = dict(vp)
            for name, tensor in tensors.items():
                if name in fresh:
                    rebound[tensor] = fresh[name]
            fresh["sinks"].fill_(7)
            fresh_ws = torch.empty_like(ws)
            g.execute(rebound, fresh_ws)
            shared._check(fresh, thd=False, sq=sq, skv=256)
            bufs["sinks"].fill_(-3)
            for name in ("o", "lse", "amax_o"):
                bufs[name].fill_(float("nan"))
            capture.replay()
            shared._check(bufs, thd=False, sq=sq, skv=256)


@pytest.mark.parametrize("d,dv", [(64, 64), (128, 128), (192, 128)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_fp8_sink_split_standalone(d, dv, output):
    from test_sdpa_prepared_fp8_split import test_prepared_fp8_split_standalone_default_scales

    test_prepared_fp8_split_standalone_default_scales("sm107", d, dv, output, sink=True)


@pytest.mark.parametrize("d", [256, 512])
def test_native_fp8_sink_keeps_independent_split_domain(d):
    sink = torch.zeros((1, 4, 1, 1), device="cuda")
    with pytest.raises(NotImplementedError, match="split"):
        shared._case(arch="sm107", d=d, sq=16, skv=256, split_kv=4, sink=sink)
