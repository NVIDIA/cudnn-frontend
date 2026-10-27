# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D192 FP8 packing preserves outputs, dynamic bindings and captures."""

from dataclasses import replace

import pytest
import torch

from frost_test_utils import requires_dsl, requires_rubin
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_rubin]


@pytest.mark.parametrize("b", [1, 3])
@pytest.mark.parametrize("split,sched,sink", [(1, 0, False), (4, 1, True), (9, 2, False)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_d192_fp8_pack_rebind_replay(b, split, sched, sink, dtype, output, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    seen = []
    load = api_dsl._load_kernel_template

    def record(filename, params, tag):
        seen.append(filename)
        return load(filename, params, tag)

    monkeypatch.setattr(api_dsl, "_load_kernel_template", record)
    d, dv = 192, 128
    sq = 129 if split == 1 else 17
    sinks = torch.linspace(-5, 8, 4, device="cuda").view(1, 4, 1, 1) if sink else None
    g, vp, ws, bufs, tensors = shared._case(arch="sm107", b=b, d=d, dv=dv, sq=sq, skv=256, split_kv=split, dtype=dtype, output_dtype=output, sink=sinks)
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=True, sched_policy=sched))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    assert "sm107/prefill_d192_d128_fp8.py" in seen
    assert not any(name.startswith("sm100/") for name in seen)
    assert g._compiled_plans[g._plan_index]._prepared is not None
    g.execute(vp, ws)
    shared._check(bufs, thd=False, b=b, sq=sq, skv=256)
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        fresh = {name: value.clone() for name, value in bufs.items()}
        rebound = {tensor: fresh[name] for name, tensor in tensors.items()}
        fresh["descale_q"].fill_(1.1)
        fresh["scale_o"].fill_(0.6)
        if sink:
            fresh["sinks"].fill_(6)
        fresh_ws = torch.empty_like(ws)
        g.execute(rebound, fresh_ws)
        shared._check(fresh, thd=False, b=b, sq=sq, skv=256)
        bufs["descale_v"].fill_(0.3)
        if sink:
            bufs["sinks"].fill_(-4)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(bufs, thd=False, b=b, sq=sq, skv=256)


@pytest.mark.parametrize("d,dv", [(160, 160), (224, 208)])
def test_native_d192_pack_retains_d256_padding_decline(d, dv):
    import cudnn

    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="no kernel-flavor envelope covers"):
        shared._case(arch="sm107", d=d, dv=dv, sq=17, skv=256)
