# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""MXFP8 data origins are independent of logical scale-tile coordinates."""

import pytest

from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance
from sdpa.frost.test_sdpa_physical_origins_shared import _check_native_origins

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]


def _check_native_routes(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
    width = nodes["q"].get_dim()[-1]
    assert graph._compiled_plans[graph._plan_index]._compiled.kernel_template == f"prefill_d{width}_mxfp8"
    _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)


@pytest.mark.parametrize("dims,cga", [(128, 2), (512, 1)], ids=["d128-cga2", "d512-cga1"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(dims, dtype, stats, cga):
    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm100_mxfp8", cga=cga, extra_checks=_check_native_routes)


@pytest.mark.parametrize("dims,cga", [(128, 2), (512, 1)], ids=["d128-cga2", "d512-cga1"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_origins_without_total_hints(dims, dtype, cga):
    _check_origins(dims, dtype, "token", False, False, 1, engine_name="sdpa_fwd_prefill_sm100_mxfp8", cga=cga, extra_checks=_check_native_routes)


@pytest.mark.parametrize("dims,cga", [(128, 2), (512, 1)], ids=["d128-cga2", "d512-cga1"])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_origin_golden(dims, dtype, stats, cga):
    from determinism_utils import assert_golden

    record = {}

    def collect(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        _check_native_routes(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
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

    _check_origins(dims, dtype, stats, True, False, 1, engine_name="sdpa_fwd_prefill_sm100_mxfp8", cga=cga, extra_checks=collect)
    assert_golden(f"sdpa-sm100-mxfp8-thd-origins-d{dims}-{dtype}-{stats}-cga{cga}", record["inputs"], record["outputs"])


@pytest.mark.parametrize("dims", [128, 512])
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_shared_physical_origins(dims, dtype, request, cudnn_handle):
    _check_native_origins(dtype, dims, request, cudnn_handle, mxfp8=True, engine="sdpa_fwd_prefill_sm100_mxfp8")
