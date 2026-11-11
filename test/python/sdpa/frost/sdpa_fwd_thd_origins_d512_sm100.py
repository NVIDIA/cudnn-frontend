# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D512 half and per-tensor FP8 keep logical lengths separate from origins."""

import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell
from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance
from sdpa.frost.test_sdpa_physical_origins_shared import _check_native_origins

pytestmark = [pytest.mark.L0, requires_dsl, requires_pre_rubin_blackwell]

_LEGS = [
    pytest.param("float16", False, id="fp16-role-split"),
    pytest.param("bfloat16", False, id="bf16-role-split"),
    pytest.param("float16", True, id="fp16-2x2"),
    pytest.param("bfloat16", True, id="bf16-2x2"),
    pytest.param("float8_e4m3fn", False, id="e4m3"),
    pytest.param("float8_e5m2", False, id="e5m2"),
]


def _configure(monkeypatch, dtype, two_by_two):
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", two_by_two)
    fp8 = dtype.startswith("float8_")
    engine = "sdpa_fwd_prefill_sm100_fp8" if fp8 else "sdpa_fwd_prefill_sm100"
    template = "prefill_d512_fp8" if fp8 else "prefill_d512_f16_2x2" if two_by_two else "prefill_d512_f16"
    return engine, template


def _verify_template(graph, template):
    assert graph._compiled_plans[graph._plan_index]._compiled.kernel_template == template


@pytest.mark.parametrize("dtype,two_by_two", [leg for leg in _LEGS if leg.values[0] != "bfloat16"])
def test_shared_physical_origins(monkeypatch, dtype, two_by_two, request, cudnn_handle):
    engine, _ = _configure(monkeypatch, dtype, two_by_two)
    _check_native_origins(dtype, 512, request, cudnn_handle, engine=engine)


@pytest.mark.parametrize("dtype,two_by_two", _LEGS)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(monkeypatch, dtype, two_by_two, stats):
    _check_case(monkeypatch, dtype, two_by_two, stats, total_hints=True)


@pytest.mark.parametrize("dtype,two_by_two", _LEGS)
def test_origins_without_total_hints(monkeypatch, dtype, two_by_two):
    _check_case(monkeypatch, dtype, two_by_two, "token", total_hints=False)


@pytest.mark.parametrize("dtype,two_by_two", [leg for leg in _LEGS if not leg.values[0].startswith("float8_")])
@pytest.mark.parametrize("dims", [(384, 448), (264, 504)], ids=["d384-v448", "d264-v504"])
def test_half_head_envelope_origins(monkeypatch, dtype, two_by_two, dims):
    _check_case(monkeypatch, dtype, two_by_two, "token", total_hints=True, dims=dims)


def _check_case(monkeypatch, dtype, two_by_two, stats, *, total_hints, collect=None, dims=512):
    engine, template = _configure(monkeypatch, dtype, two_by_two)

    def extra(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        _verify_template(graph, template)
        _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)
        if collect is not None:
            collect(values, state, flat, stats_layout)

    _check_origins(
        dims,
        dtype,
        stats,
        total_hints,
        False,
        1,
        engine_name=engine,
        cga=2,
        extra_checks=extra,
        origin_element_unit=8 if isinstance(dims, tuple) else None,
    )


@pytest.mark.parametrize("dtype,two_by_two", _LEGS)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_origin_golden(monkeypatch, dtype, two_by_two, stats):
    from determinism_utils import assert_golden

    record = {}

    def collect(values, state, flat, stats_layout):
        ql, kl, origins = state
        if origins["q"][0] == 0:
            record["inputs"] = {
                name: flat(values[name])[origins[name][0] : origins[name][0] + (ql[0] if name == "q" else kl[0])].clone() for name in ("q", "k", "v")
            }
            for name in ("descale_q", "descale_k", "descale_v", "scale_s", "descale_s", "scale_o"):
                if name in values:
                    record["inputs"][name] = values[name].clone()
            stats_view = values["stats"].T if stats_layout == "head" else values["stats"]
            record["outputs"] = {
                "o": flat(values["o"])[origins["o"][0] : origins["o"][0] + ql[0]].clone(),
                "stats": stats_view[origins["stats"][0] : origins["stats"][0] + ql[0]].clone(),
            }

    _check_case(monkeypatch, dtype, two_by_two, stats, total_hints=True, collect=collect)
    arm = "2x2" if two_by_two else "role-split"
    assert_golden(f"sdpa-sm100-d512-thd-origins-{dtype}-{arm}-{stats}", record["inputs"], record["outputs"])


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn", "float8_e5m2"])
def test_stats_slot_waits_for_all_warps(monkeypatch, tmp_path, dtype):
    """Delay nonleader readers while the same persistent cluster reuses Stats."""
    from pathlib import Path

    from cudnn.sdpa.fwd import api_dsl
    from frost_test_utils import process_watchdog

    monkeypatch.setenv("FROST_THD_CLUSTERS", "1")
    original_load = api_dsl.load_template
    served = []

    def delayed(path, params, tag="template", **kwargs):
        source_path = Path(path)
        if source_path.parent.name == "sm100" and source_path.name in ("prefill_d512_f16.py", "prefill_d512_fp8.py"):
            source = source_path.read_text()
            anchor = "            ell_addr = sStats_xfer_raw.subview(tid_in_wg)\n"
            assert source.count(anchor) == 1
            delay = (
                "            if ~is_lead_warp:\n"
                "                stats_read_start = cute.arch.globaltimer()\n"
                "                while cute.arch.globaltimer() - stats_read_start < cutlass.Int64(100_000):\n"
                "                    pass\n"
            )
            target = tmp_path / source_path.name
            target.write_text(source.replace(anchor, delay + anchor))
            module = original_load(str(target), params, tag, **kwargs)
            assert module.THD_PERSISTENT
            served.append(source_path.name)
            return module
        return original_load(path, params, tag, **kwargs)

    monkeypatch.setattr(api_dsl, "load_template", delayed)
    # Existing helper checks independent origins, numerical O/Stats, bitwise
    # batch invariance, changed bindings and retained live/empty graph replay.
    with process_watchdog(120.0, "D512 Stats slot reader join"):
        _check_case(monkeypatch, dtype, False, "token", total_hints=True)
    assert served, "the delayed role-split template did not serve the graph"
