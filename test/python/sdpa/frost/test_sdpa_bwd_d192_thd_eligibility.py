# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""D192 packed half admission must not widen dense-only FP8 eligibility."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.sdpa.bwd import engines
from test_sdpa_bwd_d128_thd_eligibility import compiler_version, facts as d128_facts

pytestmark = pytest.mark.L0


def facts(**kwargs):
    values = dict(d_qk=192, d_v=128, scale=192**-0.5)
    values.update(kwargs)
    value = d128_facts(**values)
    ports = []
    for role, shape, stride in value.port_layouts:
        width = 128 if role in ("v", "o", "dO", "dV") else 192
        ports.append((role, (*shape[:3], width), (stride[0], width, value.h_q * width, 1)))
    return replace(value, port_layouts=tuple(ports))


def reason(value, engine="sdpa_bwd_sm100_d192"):
    spec = next(s for s in engines.ENGINE_SPECS if s.name == engine)
    return engines.mismatch(spec.capabilities, value)


@pytest.mark.parametrize("packing", ["TH1", "HN"])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_half_thd_admitted(packing, dtype, cc):
    assert reason(facts(packing=packing, dtype=dtype, device_cc=cc)) is None


@pytest.mark.parametrize(
    "overrides,token",
    [
        ({"h_kv": 1}, "GQA"),
        ({"scale": 0}, "attn_scale = 0"),
        ({"scale": -0.1}, "positive"),
        ({"scale": float("inf")}, "positive"),
        ({"device_cc": (10, 1)}, "10.0 / 10.3"),
        ({"max_total_seq_len_q": None}, "positive"),
        ({"max_total_seq_len_kv": 0}, "positive"),
        ({"s_q": 2**31 - 127}, "tile rounding"),
        ({"max_total_seq_len_q": 2**32}, "dimension"),
        ({"b": 65536}, "grid"),
        ({"deterministic": True}, "deterministic"),
        ({"causal": True, "bottom_right": True}, "bottom-right"),
        ({"window_left": 128, "causal": True}, "window"),
    ],
)
def test_unsupported_declarations_decline(overrides, token):
    assert token.lower() in reason(facts(**overrides)).lower()


def test_fp8_thd_stays_declined():
    value = facts(dtype=cudnn.data_type.FP8_E4M3, is_fp8=True)
    assert reason(value, "sdpa_bwd_sm100_d192_fp8") is not None


def test_head_major_stats_must_cover_physical_q_capacity():
    value = facts(packing="HN")
    value.stats_t.get_stride = lambda: (1024, 258, 1, 1)
    assert "physical Q-token capacity" in reason(value)


def test_qk_and_value_token_strides_are_independent():
    value = facts()
    ports = tuple((r, sh, (st[0], 192, value.h_q * 192, 1) if r == "v" else st) for r, sh, st in value.port_layouts)
    assert reason(replace(value, port_layouts=ports)) is not None


def test_old_dsl_declines_before_kernel_import(monkeypatch):
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.7.4")))
    assert "nvidia-cutlass-dsl >= 4.8.0" in reason(facts())
