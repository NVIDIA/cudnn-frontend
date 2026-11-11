# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native Rubin FP8 THD addresses each port independently of live lengths."""

import pytest

from determinism_utils import assert_golden, assert_replays
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance
from sdpa.frost.test_sdpa_physical_origins_shared import _check_native_origins

_ENGINE = "sdpa_fwd_prefill_sm107_fp8"
_DTYPES = ("float8_e4m3fn", "float8_e5m2")


def _check_route(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
    plan = graph._compiled_plans[graph._plan_index]
    module = plan._compiled.kernel_module
    assert module.__file__.endswith("sm107/prefill_d128_fp8.py")
    assert module.CFG.CTA_MMA == 2 and module.CFG.SPLIT_KV == 1
    assert plan._prepared.spec.origin_specs is not None
    assert plan._prepared.spec.native is not None
    _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
    pack = {nodes[name]: value for name, value in values.items()}

    def run():
        values["o"].fill_(-17)
        values["stats"].fill_(-29)
        graph.execute(pack, workspace, **overrides)
        return {name: values[name] for name in ("o", "stats")}

    assert_replays(run, repeats=3, allowed_inf={"stats": -float("inf")})


@pytest.mark.parametrize("dtype", _DTYPES)
def test_shared_physical_origins(dtype, request, cudnn_handle):
    _check_native_origins(dtype, 128, request, cudnn_handle, engine=_ENGINE)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(dtype, stats, monkeypatch):
    monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
    _check_origins(128, dtype, stats, True, False, 1, engine_name=_ENGINE, cga=2, extra_checks=_check_route)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_origins_without_total_hints(dtype, monkeypatch):
    monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
    _check_origins(128, dtype, "token", False, False, 1, engine_name=_ENGINE, cga=2, extra_checks=_check_route)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_origin_golden(dtype, stats, monkeypatch):
    monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
    record = {}

    def collect(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        _check_route(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
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
    assert_golden(f"sdpa-sm107-fp8-thd-origins-d128-{dtype}-{stats}", record["inputs"], record["outputs"])
