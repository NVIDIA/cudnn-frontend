# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Graph declaration bounds for native SM120 block-scale backward."""

import pytest
import cudnn
import test_sdpa_bwd_mxfp8_sm100 as mx

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def native(monkeypatch):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(mx, "_ENGINE", "sdpa_bwd_sm120_mxfp8")
    monkeypatch.setattr(ga, "_device_cc", lambda: (12, 0))
    monkeypatch.setattr(ga, "_device_sm_count", lambda: 16)
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", "4.8.0")))
    monkeypatch.setattr(engines, "cutedsl_arch_requirement_error", lambda cc: None)


@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("dims", [(8, 8), (64, 64), (128, 128), (192, 128), (256, 256), (504, 248), (512, 512)])
@pytest.mark.parametrize("deterministic", [False, True])
def test_graph_admission(monkeypatch, cc, dims, deterministic):
    from cudnn.sdpa import graph_analyzer as ga

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    assert (
        mx._decline_reason(d=dims[0], dv=dims[1], hq=6, hkv=2, sq=19, skv=37, use_causal_mask=True, left_bound=7, use_deterministic_algorithm=deterministic)
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(d=520),
        dict(d=63),
        dict(d=64, dv=128),
        dict(with_sink=True),
        dict(use_causal_mask_bottom_right=True),
        dict(right_bound=3),
        dict(use_padding_mask=True, seq_len_dims=(1, 1, 1, 1)),
        dict(fp8=cudnn.data_type.FP8_E5M2),
        dict(scale=0.0),
        dict(scale=float("inf")),
        dict(b=65536),
        dict(skv=2**31 - 128),
    ],
)
def test_graph_decline(kwargs):
    assert mx._decline_reason(**kwargs) is not None


@pytest.mark.parametrize("cc,version", [((10, 7), "4.8.0"), ((12, 0), "4.7.0"), ((12, 1), "4.6.2")])
def test_arch_and_version_decline(monkeypatch, cc, version):
    from cudnn.sdpa import graph_analyzer as ga
    from cudnn.sdpa.bwd import engines

    monkeypatch.setattr(ga, "_device_cc", lambda: cc)
    monkeypatch.setattr(engines, "cutedsl_state", lambda: (True, ("nvidia-cutlass-dsl", version)))
    assert mx._decline_reason() is not None


@pytest.mark.parametrize("requested", [True, ("amax_dQ",), ("amax_dK",), ("amax_dV",)])
def test_amax_outputs_admitted(requested):
    assert mx._decline_reason(d=192, dv=128, declare_amax=requested) is None
