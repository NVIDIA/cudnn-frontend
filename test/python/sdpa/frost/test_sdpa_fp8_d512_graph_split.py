# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The existing FP8 D512 split kernel is also reachable through graph plans."""

from dataclasses import replace

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


@pytest.mark.parametrize("d,dv", [(512, 512), (384, 320)])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize(
    "sink,split,pack,dtype",
    [(False, 2, False, torch.float8_e4m3fn), (False, 9, True, torch.float8_e5m2), (True, 4, True, torch.float8_e4m3fn), (True, 9, False, torch.float8_e5m2)],
)
def test_d512_fp8_graph_split_rebind_replay(d, dv, output, sink, split, pack, dtype):
    sinks = torch.linspace(-12, 12, 4, device="cuda").view(1, 4, 1, 1) if sink else None
    g, vp, ws, bufs, tensors = shared._case(d=d, dv=dv, sq=17, skv=256, split_kv=split, dtype=dtype, output_dtype=output, sink=sinks, output_padding=16)
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=pack, sched_policy=0))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    assert g._compiled_plans[g._plan_index]._prepared.spec.split == split
    for iteration in range(2):
        if iteration:
            for name in ("q", "k", "v", "lse", "amax_o", "descale_q", "scale_o") + (("sinks",) if sink else ()):
                bufs[name] = bufs[name].clone()
                vp[tensors[name]] = bufs[name]
            bufs["o_storage"] = torch.full_like(bufs["o_storage"], 12)
            bufs["o"] = bufs["o_storage"][..., :dv].transpose(1, 2)
            vp[tensors["o"]] = bufs["o"]
            bufs["descale_q"].fill_(0.5)
            bufs["scale_o"].fill_(1.9)
            ws = torch.empty_like(ws)
        g.execute(vp, ws)
        shared._check(bufs, thd=False, sq=17, skv=256)
        with shared._cuda_graph() as capture:
            with torch.cuda.graph(capture):
                g.execute(vp, ws)
            if sink:
                bufs["sinks"].fill_(5 if iteration else -3)
            bufs["descale_q"].fill_(0.7)
            for name in ("o", "lse", "amax_o"):
                bufs[name].fill_(float("nan"))
            capture.replay()
            shared._check(bufs, thd=False, sq=17, skv=256)
        assert torch.all(bufs["o_storage"][..., dv:] == 12)
