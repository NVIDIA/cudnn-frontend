# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native D512 FP8 split contracts, collected through the SM107 CI entry."""

from dataclasses import replace

import test_sdpa_prepared_fp8 as dense
from sdpa_fp8_d256_padding import _check, _clone_with_slack, _record_route

import pytest
import torch
from sdpa_fp8_d512_pack_sm107 import _run as _run_packed

# Head envelope, GQA group, sequence extents, mask/sink and schedule, split, packing.
_CASES = [
    ((512, 512, 2, 17, 257, False, False, None, False, 0), 9, False),
    ((384, 336, 4, 129, 512, True, False, None, True, 1), 4, True),
    ((272, 496, 8, 33, 257, True, True, None, True, 0), 4, True),
    ((496, 272, 16, 17, 128, True, False, 64, True, 1), 9, True),
    ((320, 448, 32, 129, 256, True, True, 64, True, 0), 3, False),
    ((512, 336, 4, 127, 256, False, False, None, True, 1), 3, True),
    ((384, 512, 8, 1, 4096, False, False, None, False, 1), 8, True),
    ((512, 512, 2, 257, 128, True, False, None, False, 0), 2, False),
    ((384, 384, 64, 3, 2048, False, False, None, True, 1), 4, True),
    ((512, 512, 128, 17, 257, True, True, None, True, 0), 8, True),
]


@pytest.mark.parametrize(
    "case,split,pack", _CASES, ids=["empty_chunks", "causal", "narrow_qk", "narrow_v", "swa", "sink", "short_q", "masked_rows", "group64", "group128"]
)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("output", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_native_d512_split_rebind_replay(case, split, pack, dtype, output, monkeypatch):
    _run_packed(case, dtype, output, monkeypatch, split=split, pack=pack)


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_native_d512_split_half_adapter(pack, dtype, monkeypatch):
    _run_packed(_CASES[1][0], dtype, torch.bfloat16, monkeypatch, half=True, adapter=True, split=4, pack=pack)


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("stats,amax", [(False, False), (False, True), (True, False), (True, True)])
def test_native_d512_split_nomask_optional(pack, dtype, stats, amax, monkeypatch, skv=128):
    """MASK_NONE still releases empty splits; optional Stats/Amax stay optional."""
    seen = _record_route(monkeypatch)
    g, vp, _, bufs, tensors = dense._case(
        arch="sm107",
        d=384,
        dv=336,
        b=2,
        sq=17,
        skv=skv,
        dtype=dtype,
        stats=stats,
        amax=amax,
        padded=True,
        output_padding=16,
        sdpa_kwargs=dict(stats_use_log2=stats),
    )
    selected = g.plans[g._plan_index]
    g.create_execution_plan(selected.engine_id, replace(selected.knobs, pack_gqa=pack, split_kv=9, sched_policy=int(pack), cga=2))
    g.select_plan(len(g.plans) - 1)
    g.check_support()
    g.build_plans()
    ws = torch.empty(max(1, g.get_workspace_size()), dtype=torch.uint8, device="cuda")
    assert seen and set(seen) == {"sm107/prefill_d512_fp8.py"}, seen
    for name in ("q", "k", "v"):
        value = bufs[name]
        value._base[..., value.shape[-1] :].fill_(float("nan"))
    check = dict(split=9, pack=pack, stats_log2=stats, q_tile_rows=256)
    g.execute(vp, ws)
    _check(bufs, **check)
    torch.cuda.set_sync_debug_mode("error")
    try:
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        g.execute(vp, ws)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with dense._cuda_graph() as capture:
        with torch.cuda.graph(capture):
            g.execute(vp, ws)
        fresh = {name: _clone_with_slack(value, float("nan")) for name, value in bufs.items() if name != "o_storage"}
        fresh["scale_o"].fill_(0.6)
        fresh["descale_q"].fill_(1.1)
        g.execute({tensor: fresh[name] for name, tensor in tensors.items()}, torch.empty_like(ws))
        _check(fresh, **check)
        bufs["descale_v"].fill_(0.3)
        for name in ("o", "lse", "amax_o"):
            if name in bufs:
                bufs[name].fill_(float("nan"))
        ws.fill_(255)
        capture.replay()
        _check(bufs, **check)
        assert torch.all(bufs["o_storage"][..., 336:] == 12)


@pytest.mark.parametrize("sink", [False, True])
@pytest.mark.parametrize("output", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_d512_split_omitted_scales(sink, output):
    from test_sdpa_prepared_fp8_split import test_prepared_fp8_split_standalone_default_scales

    test_prepared_fp8_split_standalone_default_scales("sm107", 512, 512, output, sink=sink)


@pytest.mark.parametrize("pack", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_native_d512_split_scalar_kv_tail(pack, dtype, monkeypatch):
    """A scalar KV tail needs no caller-provided per-batch length tensors."""
    test_native_d512_split_nomask_optional(pack, dtype, True, True, monkeypatch, skv=257)
