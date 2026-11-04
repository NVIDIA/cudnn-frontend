# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D192 FP8 envelopes zero-fill head tails without reading caller storage slack."""

from dataclasses import replace

import pytest
import torch

from frost_test_utils import requires_blackwell, requires_dsl
import test_sdpa_prepared_fp8 as shared

pytestmark = [pytest.mark.L0, requires_blackwell, requires_dsl]


@pytest.mark.parametrize("d,dv", [(144, 128), (160, 112), (176, 64), (192, 16)])
@pytest.mark.parametrize("split,pack,sched,sink", [(1, False, 0, False), (1, True, 1, False), (4, True, 2, True)])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_d192_padded_heads_rebind_replay(d, dv, split, pack, sched, sink, dtype, output, monkeypatch):
    from cudnn.sdpa.fwd import api_dsl

    arch = "sm107" if torch.cuda.get_device_capability() == (10, 7) else "sm100"
    seen = []
    load = api_dsl._load_kernel_template

    def record(filename, params, tag, part=None):
        seen.append(filename)
        return load(filename, params, tag, part=part)

    monkeypatch.setattr(api_dsl, "_load_kernel_template", record)
    b, sq, skv = 2, 129 if split == 1 else 17, 256
    sinks = torch.linspace(-5, 8, 4, device="cuda").view(1, 4, 1, 1) if sink else None
    g, vp, ws, bufs, tensors = shared._case(
        arch=arch,
        b=b,
        d=d,
        dv=dv,
        sq=sq,
        skv=skv,
        split_kv=split,
        dtype=dtype,
        output_dtype=output,
        sink=sinks,
        padded=True,
        output_padding=16,
    )
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=pack, sched_policy=sched))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    assert f"{arch}/prefill_d192_d128_fp8.py" in seen
    assert all(name.startswith(arch + "/") for name in seen)
    for name in ("q", "k", "v"):
        view = bufs[name]
        storage = view._base
        assert storage.shape[-1] == view.shape[-1] + 16
        storage[..., view.shape[-1] :].fill_(float("nan"))
        view.copy_((view.float() * 2.5).to(dtype))
    g.execute(vp, ws)
    shared._check(bufs, thd=False, b=b, sq=sq, skv=skv)
    assert torch.all(bufs["o_storage"][..., dv:] == 12)
    with shared._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        fresh = {}
        for name, value in bufs.items():
            if name == "o_storage":
                continue
            fresh[name] = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
            fresh[name].copy_(value)
        fresh["descale_q"].fill_(1.1)
        fresh["scale_o"].fill_(0.6)
        if sink:
            fresh["sinks"].fill_(6)
        rebound = {tensor: fresh[name] for name, tensor in tensors.items()}
        fresh_ws = torch.empty_like(ws)
        g.execute(rebound, fresh_ws)
        shared._check(fresh, thd=False, b=b, sq=sq, skv=skv)
        bufs["descale_v"].fill_(0.3)
        if sink:
            bufs["sinks"].fill_(-4)
        for name in ("o", "lse", "amax_o"):
            bufs[name].fill_(float("nan"))
        capture.replay()
        shared._check(bufs, thd=False, b=b, sq=sq, skv=skv)
        assert torch.all(bufs["o_storage"][..., dv:] == 12)
