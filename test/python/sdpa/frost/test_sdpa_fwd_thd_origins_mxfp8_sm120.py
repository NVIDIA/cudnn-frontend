# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""MXFP8 data origins are independent of logical scale-tile coordinates."""

import pytest

from frost_test_utils import requires_blackwell_geforce, requires_dsl
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dims", [64, 128, (192, 128), 256, 512, (384, 512)])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(dims, dtype, stats):
    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm120_mxfp8", extra_checks=_check_routes_and_invariance)


@pytest.mark.parametrize("dims", [128, 512])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_origins_without_total_hints(dims, dtype):
    _check_origins(dims, dtype, "token", False, False, 1, engine_name="sdpa_fwd_prefill_sm120_mxfp8", extra_checks=_check_routes_and_invariance)


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
            for name in ("q", "k", "v"):
                tiles = ((ql[0] if name == "q" else kl[0]) + 127) // 128
                record["inputs"]["sf_" + name] = values["sf_" + name][:, :tiles].clone()
            stats_view = values["stats"].T if stats_layout == "head" else values["stats"]
            record["outputs"] = {
                "o": flat(values["o"])[origins["o"][0] : origins["o"][0] + ql[0]].clone(),
                "stats": stats_view[origins["stats"][0] : origins["stats"][0] + ql[0]].clone(),
            }

    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm120_mxfp8", extra_checks=collect)
    assert_golden(f"sdpa-sm120-mxfp8-thd-origins-d{dims}-{dtype}-{stats}", record["inputs"], record["outputs"])
