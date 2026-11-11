# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM120 FP8 uses each port's physical origin without changing logical work."""

import pytest
import torch

from frost_test_utils import requires_blackwell_geforce, requires_dsl
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_sm120 import _check_invariance

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


def _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
    _check_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
    expected = {name: values[name].contiguous().view(torch.uint8).clone() for name in ("o", "stats")}
    plan = graph._compiled_plans[graph._plan_index]
    prepared = plan._prepared
    pack = {nodes[name]: value for name, value in values.items()}
    plan._prepared, plan.takes_variant_pack = None, False
    try:
        values["o"].fill_(-17)
        values["stats"].fill_(-29)
        graph.execute(pack, workspace, **overrides)
        for name, bits in expected.items():
            assert torch.equal(values[name].contiguous().view(torch.uint8), bits), name
    finally:
        plan._prepared, plan.takes_variant_pack = prepared, True
    torch.cuda.set_sync_debug_mode("error")
    try:
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.execute(pack, workspace, **overrides)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("dims", [64, 128, (192, 128), 256, 512, (384, 512)])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(dims, dtype, stats):
    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm120_fp8", extra_checks=_check_routes_and_invariance)


@pytest.mark.parametrize("dims", [128, 512])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_packed_origins_rebind_replay(dims, dtype):
    _check_origins(dims, dtype, "token", False, True, 1, engine_name="sdpa_fwd_prefill_sm120_fp8", extra_checks=_check_routes_and_invariance)


@pytest.mark.parametrize("dims", [128, 512])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_origin_golden(dims, dtype, stats):
    from determinism_utils import assert_golden

    record = {}

    def collect(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
        ql, kl, origins = state
        if origins["q"][0] == 0:
            record["inputs"] = {
                name: flat(values[name])[origins[name][0] : origins[name][0] + (ql[0] if name == "q" else kl[0])].clone() for name in ("q", "k", "v")
            }
            for name in ("descale_q", "descale_k", "descale_v", "scale_s", "descale_s", "scale_o"):
                record["inputs"][name] = values[name].clone()
            stats_view = values["stats"].T if stats_layout == "head" else values["stats"]
            record["outputs"] = {
                "o": flat(values["o"])[origins["o"][0] : origins["o"][0] + ql[0]].clone(),
                "stats": stats_view[origins["stats"][0] : origins["stats"][0] + ql[0]].clone(),
            }

    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm120_fp8", extra_checks=collect)
    assert_golden(f"sdpa-sm120-fp8-thd-origins-d{dims}-{dtype}-{stats}", record["inputs"], record["outputs"])
