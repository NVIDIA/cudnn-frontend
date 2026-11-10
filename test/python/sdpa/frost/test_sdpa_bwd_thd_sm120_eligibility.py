# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM120 packed backward width, GQA and tile-resource declaration boundaries."""

from dataclasses import replace

import pytest

import cudnn
from cudnn.sdpa.bwd import engines
from test_sdpa_bwd_d128_thd_eligibility import compiler_version, facts as d128_facts

pytestmark = pytest.mark.L0


def facts(**overrides):
    value = d128_facts(**overrides)
    ports = []
    for role in ("q", "k", "v", "o", "dO", "dQ", "dK", "dV"):
        q_side = role in ("q", "o", "dO", "dQ")
        h = value.h_q if q_side else value.h_kv
        n = value.s_q if q_side else value.s_kv
        d = value.d_v if role in ("v", "o", "dO", "dV") else value.d_qk
        ports.append((role, (value.b, h, n, d), (n * h * d, d, h * d, 1)))
    return replace(value, port_layouts=tuple(ports))


def reason(**overrides):
    value = facts(device_cc=(12, 0), **overrides)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    return engines.mismatch(spec.capabilities, value)


@pytest.mark.parametrize("packing", ["TH1", "HN"])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_frost_thd_sm120_admission(packing, dtype):
    assert reason(packing=packing, dtype=dtype) is None


@pytest.mark.parametrize(
    "overrides,token",
    [
        ({"d_qk": 257, "d_v": 128}, "256"),
        ({"d_qk": 127, "d_v": 120}, "8"),
        ({"deterministic": True}, "deterministic"),
        ({"window_left": 64}, "window"),
        ({"has_sink": True}, "sink"),
        ({"scale": -0.1}, "positive"),
        ({"scale": float("inf")}, "positive"),
        ({"max_total_seq_len_q": None}, "positive"),
        ({"b": 65536}, "grid"),
        ({"s_kv": 2**31 - 63}, "rounding"),
    ],
)
def test_frost_thd_sm120_declines_outside_scope(overrides, token):
    assert token.lower() in reason(**overrides).lower()


@pytest.mark.parametrize("device_cc", [(12, 0), (12, 1)])
def test_frost_thd_sm120_family_admission(device_cc):
    value = facts(device_cc=device_cc)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    assert engines.mismatch(spec.capabilities, value) is None


def test_frost_dense_sm120_eligibility_unchanged():
    assert reason(thd=False, padded=False, d_qk=64, d_v=64, h_kv=1) is None


@pytest.mark.parametrize("tile_m,tile_n,accepted", [(64, 64, True), (32, 32, True), (64, 32, True), (None, None, True), (32, 64, False), (64, 128, False)])
def test_frost_thd_tile_domain_is_exact(tile_m, tile_n, accepted):
    value = facts(device_cc=(12, 0))
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    result = engines.mismatch(spec.capabilities, value, engines.SdpaBwdKnobs(tile_m=tile_m, tile_n=tile_n))
    assert (result is None) == accepted


@pytest.mark.parametrize("dims", [(64, 64), (128, 128), (192, 128), (256, 256), (248, 184)])
def test_frost_thd_gqa_wide_admission(dims):
    assert reason(d_qk=dims[0], d_v=dims[1], h_kv=1) is None


@pytest.mark.parametrize("tiles,accepted", [((None, None), True), ((32, 32), True), ((32, 64), False), ((64, 32), False), ((64, 64), False)])
def test_frost_thd_wide_tile_resource_boundary(tiles, accepted):
    value = facts(device_cc=(12, 0), d_qk=256, d_v=256, h_kv=1)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    result = engines.mismatch(spec.capabilities, value, engines.SdpaBwdKnobs(tile_m=tiles[0], tile_n=tiles[1]))
    assert (result is None) == accepted


@pytest.mark.parametrize("d,expected", [(128, (64, 64)), (192, (32, 32)), (256, (32, 32))])
def test_frost_thd_listed_tiles_replay_native_geometry(d, expected):
    from cudnn.sdpa.bwd.heuristics import default_knobs

    value = facts(device_cc=(12, 0), d_qk=d, d_v=d)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    knobs = default_knobs(spec, value)
    assert (knobs.tile_m, knobs.tile_n) == expected
    assert engines.mismatch(spec.capabilities, value, knobs) is None
    assert knobs.to_public() == {cudnn.knob_type.TILE_M: expected[0], cudnn.knob_type.TILE_N: expected[1]}


@pytest.mark.parametrize("batch", [2, 65535])
def test_frost_thd_sm120_q_envelope_launch_boundary(batch):
    from cudnn.sdpa.bwd.sm120_thd_contract import validate_bounds

    # Independent-origin delta and dQ cast launch over the declared B*SQ,
    # even when empty sequences let the packed physical capacity stay tiny.
    last_safe = (2**31 - 1) // batch
    validate_bounds(batch, 1, 1, last_safe, 1, 128, 128, 1, 1, 128**-0.5)
    assert reason(b=batch, h_q=1, h_kv=1, s_q=last_safe, max_total_seq_len_q=1, max_total_seq_len_kv=1) is None
    with pytest.raises(ValueError, match="Int32"):
        validate_bounds(batch, 1, 1, last_safe + 1, 1, 128, 128, 1, 1, 128**-0.5)
    assert "Int32" in reason(b=batch, h_q=1, h_kv=1, s_q=last_safe + 1, max_total_seq_len_q=1, max_total_seq_len_kv=1)
