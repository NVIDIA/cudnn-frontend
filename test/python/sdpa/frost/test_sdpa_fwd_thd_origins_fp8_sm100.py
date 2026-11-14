# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""SM100 FP8 THD uses independent physical origins and logical live bounds."""

import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance
from sdpa.frost.test_sdpa_physical_origins_shared import _check_native_origins

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]

_ENGINE = "sdpa_fwd_prefill_sm100_fp8"
_DTYPES = ("float8_e4m3fn", "float8_e5m2")


@pytest.mark.parametrize("dtype", _DTYPES)
def test_shared_physical_origins(dtype, request, cudnn_handle):
    _check_native_origins(dtype, 128, request, cudnn_handle, engine=_ENGINE)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(dtype, stats):
    # The existing nonpaged unsplit FP8 THD contract uses two CTAs.
    _check_origins(128, dtype, stats, True, False, 1, engine_name=_ENGINE, cga=2, extra_checks=_check_routes_and_invariance)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_origins_without_total_hints(dtype):
    _check_origins(128, dtype, "token", False, False, 1, engine_name=_ENGINE, cga=2, extra_checks=_check_routes_and_invariance)


@pytest.mark.parametrize("dims", [128, 256])
def test_half_origin_execution_routes(dims):
    # The same standalone forwarding serves existing half map hosts.
    _check_origins(dims, "bfloat16", "token", True, False, 1, cga=2, extra_checks=_check_routes_and_invariance)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_origin_golden(dtype, stats):
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

    _check_origins(128, dtype, stats, True, False, 1, engine_name=_ENGINE, cga=2, extra_checks=collect)
    assert_golden(f"sdpa-sm100-fp8-thd-origins-d128-{dtype}-{stats}", record["inputs"], record["outputs"])
