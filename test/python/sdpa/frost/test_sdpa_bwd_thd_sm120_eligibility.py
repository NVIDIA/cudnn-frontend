# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SM120 packed addition has a strict D128 MHA declaration boundary."""

import pytest

import cudnn
from cudnn.sdpa.bwd import engines
from test_sdpa_bwd_d128_thd_eligibility import compiler_version, facts as d128_facts

pytestmark = pytest.mark.L0


def reason(**overrides):
    value = d128_facts(device_cc=(12, 0), **overrides)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    return engines.mismatch(spec.capabilities, value)


@pytest.mark.parametrize("packing", ["TH1", "HN"])
@pytest.mark.parametrize("dtype", [cudnn.data_type.HALF, cudnn.data_type.BFLOAT16])
def test_frost_thd_sm120_admission(packing, dtype):
    assert reason(packing=packing, dtype=dtype) is None


@pytest.mark.parametrize(
    "overrides,token",
    [
        ({"h_kv": 1}, "MHA"),
        ({"d_qk": 64, "d_v": 64}, "128"),
        ({"d_qk": 192, "d_v": 128}, "128"),
        ({"deterministic": True}, "deterministic"),
        ({"causal": True, "bottom_right": True}, "bottom-right"),
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
    value = d128_facts(device_cc=device_cc)
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    assert engines.mismatch(spec.capabilities, value) is None


def test_frost_dense_sm120_eligibility_unchanged():
    assert reason(thd=False, padded=False, d_qk=64, d_v=64, h_kv=1) is None


@pytest.mark.parametrize("tile_m,tile_n,accepted", [(64, 64, True), (None, None, True), (32, 64, False), (64, 128, False)])
def test_frost_thd_tile_domain_is_exact(tile_m, tile_n, accepted):
    value = d128_facts(device_cc=(12, 0))
    spec = next(s for s in engines.ENGINE_SPECS if s.name == "sdpa_bwd_sm120")
    result = engines.mismatch(spec.capabilities, value, engines.SdpaBwdKnobs(tile_m=tile_m, tile_n=tile_n))
    assert (result is None) == accepted


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
