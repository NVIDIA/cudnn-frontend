# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Declaration boundaries for native SM120 FP8 backward."""

import pytest
import cudnn
from cudnn.sdpa.bwd.sm120_fp8_contract import environment_error, validate_bounds
import test_sdpa_bwd_fp8_sm107 as fp8

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("dims", [(8, 8), (64, 64), (128, 128), (192, 128), (256, 256), (504, 248), (512, 512)])
@pytest.mark.parametrize("deterministic", [False, True])
def test_graph_admission(monkeypatch, cc, dims, deterministic):
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(engines, "cutedsl_arch_requirement_error", lambda cc: None)
    reason = fp8._decline_reason(
        monkeypatch,
        engine="sdpa_bwd_sm120_fp8",
        cc=cc,
        d=dims[0],
        d_v=dims[1],
        h=6,
        hkv=2,
        s=19,
        skv=37,
        deterministic=deterministic,
        causal=True,
        left_bound=7,
    )
    assert reason is None, reason


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d=520),
        dict(d=63),
        dict(d=64, d_v=128),
        dict(bottom_right=True),
        dict(right_bound=3),
        dict(padded=True),
        dict(sink=True),
        dict(fp8=cudnn.data_type.FP8_E5M2),
        dict(scale=0.0),
        dict(scale=float("inf")),
    ],
)
def test_graph_decline(monkeypatch, kwargs):
    reason = fp8._decline_reason(monkeypatch, engine="sdpa_bwd_sm120_fp8", cc=(12, 0), **kwargs)
    assert reason is not None


@pytest.mark.parametrize("cc,version,token", [((10, 7), "4.8.0", "SM120"), ((12, 0), "4.7.0", "4.8.0"), ((12, 1), "4.6.2", "4.8.0")])
def test_compiler_and_arch_floor(cc, version, token):
    assert token in environment_error(cc, ("nvidia-cutlass-dsl", version))


@pytest.mark.parametrize(
    "overrides,token",
    [
        ({"batch": 65536}, "65535"),
        ({"skv": 2**31 - 128}, "Int32"),
        ({"sq": 2**31 // 128, "heads": 128}, "delta"),
        ({"window_left": -1}, "nonnegative"),
        ({"window_left": True}, "nonnegative"),
        ({"kv_heads": 3}, "ratio"),
    ],
)
def test_native_bounds(overrides, token):
    kw = dict(batch=1, heads=4, kv_heads=2, sq=19, skv=37, dq=128, dv=128, scale=None, window_left=None)
    kw.update(overrides)
    with pytest.raises(ValueError, match=token):
        validate_bounds(**kw)
