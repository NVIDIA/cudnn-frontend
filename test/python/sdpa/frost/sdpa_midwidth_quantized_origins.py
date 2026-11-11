# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D192/D256 quantized data origins preserve logical quantization coordinates."""

import pytest

from determinism_utils import assert_golden, assert_replays
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance


def origin_checks(family):
    engine = f"sdpa_fwd_prefill_sm100_{family}"

    def check_route(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        width = nodes["q"].get_dim()[-1]
        flavor = "d192_d128" if width == 192 else "d256"
        plan = graph._compiled_plans[graph._plan_index]
        assert plan._compiled.kernel_template == f"prefill_{flavor}_{family}"
        assert plan._prepared.spec.origin_specs is not None and plan._prepared.spec.native is not None
        _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
        pack = {nodes[name]: value for name, value in values.items()}

        def run():
            values["o"].fill_(-17)
            values["stats"].fill_(-29)
            graph.execute(pack, workspace, **overrides)
            return {name: values[name] for name in ("o", "stats")}

        assert_replays(run, repeats=3, allowed_inf={"stats": -float("inf")})

    def shape_dtype(fn):
        fn = pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])(fn)
        return pytest.mark.parametrize("dims,cga", [((192, 128), 1), ((192, 128), 2), (256, 1)], ids=["d192-v128-cga1", "d192-v128-cga2", "d256-cga1"])(fn)

    @shape_dtype
    @pytest.mark.parametrize("stats", ["token", "head"])
    def test_independent_origins_rebind_replay(dims, cga, dtype, stats, monkeypatch):
        monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
        _check_origins(dims, dtype, stats, True, False, 1, engine_name=engine, cga=cga, extra_checks=check_route)

    @shape_dtype
    def test_origins_without_total_hints(dims, cga, dtype, monkeypatch):
        monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
        _check_origins(dims, dtype, "token", False, False, 1, engine_name=engine, cga=cga, extra_checks=check_route)

    @shape_dtype
    @pytest.mark.parametrize("stats", ["token", "head"])
    def test_origin_golden(dims, cga, dtype, stats, monkeypatch):
        monkeypatch.setenv("FROST_THD_CLUSTERS", "2")
        record = {}

        def collect(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
            check_route(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
            ql, kl, origins = state
            if origins["q"][0] == 0:
                record["inputs"] = {
                    name: flat(values[name])[origins[name][0] : origins[name][0] + (ql[0] if name == "q" else kl[0])].clone() for name in ("q", "k", "v")
                }
                if family == "mxfp8":
                    for name in ("q", "k", "v"):
                        tiles = ((ql[0] if name == "q" else kl[0]) + 127) // 128
                        record["inputs"]["sf_" + name] = values["sf_" + name][:, :tiles].clone()
                else:
                    for name in ("descale_q", "descale_k", "descale_v", "scale_s", "descale_s", "scale_o"):
                        record["inputs"][name] = values[name].clone()
                stats_view = values["stats"].T if stats_layout == "head" else values["stats"]
                record["outputs"] = {
                    "o": flat(values["o"])[origins["o"][0] : origins["o"][0] + ql[0]].clone(),
                    "stats": stats_view[origins["stats"][0] : origins["stats"][0] + ql[0]].clone(),
                }

        _check_origins(dims, dtype, stats, True, False, 1, engine_name=engine, cga=cga, extra_checks=collect)
        shape = "192x128" if isinstance(dims, tuple) else str(dims)
        assert_golden(f"sdpa-sm100-{family}-thd-origins-d{shape}-{dtype}-{stats}-cga{cga}", record["inputs"], record["outputs"])

    return tuple(map(staticmethod, (test_independent_origins_rebind_replay, test_origins_without_total_hints, test_origin_golden)))
