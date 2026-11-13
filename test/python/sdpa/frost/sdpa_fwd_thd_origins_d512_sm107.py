# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Native SM107 half D512 physical origins in both retained kernel bodies."""

import pytest

from sdpa.frost.test_sdpa_fwd_thd_origins import test_independent_thd_origins_rebind_and_capture as _check_origins
from sdpa.frost.test_sdpa_fwd_thd_origins_fp8_sm120 import _check_routes_and_invariance
from sdpa.frost.test_sdpa_physical_origins_shared import _check_native_origins

_LEGS = [
    pytest.param("float16", False, id="fp16-role-split"),
    pytest.param("bfloat16", False, id="bf16-role-split"),
    pytest.param("float16", True, id="fp16-2x2"),
    pytest.param("bfloat16", True, id="bf16-2x2"),
]


def _configure(monkeypatch, two_by_two):
    from cudnn.sdpa.fwd import api_dsl

    monkeypatch.setattr(api_dsl, "D512_2X2", two_by_two)
    return "prefill_d512_f16_2x2" if two_by_two else "prefill_d512_f16"


@pytest.mark.parametrize("two_by_two", [False, True], ids=["role-split", "2x2"])
def test_shared_physical_origins(monkeypatch, two_by_two, request, cudnn_handle):
    _configure(monkeypatch, two_by_two)
    _check_native_origins("float16", 512, request, cudnn_handle, engine="sdpa_fwd_prefill_sm107")


@pytest.mark.parametrize("dtype,two_by_two", _LEGS)
@pytest.mark.parametrize("stats", ["token", "head"])
def test_independent_origins_rebind_replay(monkeypatch, dtype, two_by_two, stats):
    _check_case(monkeypatch, dtype, two_by_two, stats, total_hints=True)


@pytest.mark.parametrize("dtype,two_by_two", _LEGS)
def test_origins_without_total_hints(monkeypatch, dtype, two_by_two):
    _check_case(monkeypatch, dtype, two_by_two, "token", total_hints=False)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("stats", ["token", "head"])
def test_packed_origins_decline(monkeypatch, dtype, stats):
    # This repair preserves the existing native THD packing boundary.
    with pytest.raises(NotImplementedError, match="Rubin half PackGQA"):
        _check_case(monkeypatch, dtype, True, stats, total_hints=True, pack=True)


@pytest.mark.parametrize("dtype,two_by_two", _LEGS)
def test_half_head_envelope_declines(monkeypatch, dtype, two_by_two):
    with pytest.raises(NotImplementedError, match="head-dim envelope is dense-only"):
        _check_case(monkeypatch, dtype, two_by_two, "token", total_hints=True, dims=(264, 504))


def _check_case(monkeypatch, dtype, two_by_two, stats, *, total_hints, dims=512, pack=False):
    template = _configure(monkeypatch, two_by_two)

    def extra(graph, nodes, values, state, workspace, overrides, flat, stats_layout):
        compiled = graph._compiled_plans[graph._plan_index]._compiled
        assert compiled.kernel_template == template
        import inspect

        api = inspect.getclosurevars(compiled.default_stream).nonlocals["api"]
        assert "/sm107/" in api._k_mod.__file__
        _check_routes_and_invariance(graph, nodes, values, state, workspace, overrides, flat, stats_layout)

    _check_origins(
        dims,
        dtype,
        stats,
        total_hints,
        pack,
        1,
        engine_name="sdpa_fwd_prefill_sm107",
        cga=2,
        extra_checks=extra,
        origin_element_unit=8 if isinstance(dims, tuple) else None,
    )
