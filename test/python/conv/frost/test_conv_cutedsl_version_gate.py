# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only CuTeDSL version gates for the Frost convolution variants."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cudnn.conv.frost import _cutedsl
from cudnn.frost import buffers

pytestmark = pytest.mark.L0


@pytest.fixture
def dsl_state(monkeypatch):
    def set_state(installed, version):
        monkeypatch.setattr(buffers, "_DSL_STATE", (installed, version))

    return set_state


def test_cutedsl_4_8_serves_block_scale_but_declines_dense(dsl_state) -> None:
    dsl_state(True, ("nvidia-cutlass-dsl", "4.8.0.dev0"))

    assert _cutedsl.requirement_error("block-scale", _cutedsl.BLOCK_SCALE_CUTEDSL_MIN_VERSION) is None
    dense_error = _cutedsl.requirement_error("dense", _cutedsl.DENSE_CUTEDSL_MIN_VERSION)
    assert "requires nvidia-cutlass-dsl >= 4.9" in dense_error
    assert "found 4.8.0.dev0" in dense_error


def test_plan_support_checks_apply_the_flavor_floor_first(dsl_state) -> None:
    from cudnn.conv.frost.engine import _Sm100FrostBlockScaleConvPlan, _Sm100FrostConvPlan

    dsl_state(True, ("nvidia-cutlass-dsl", "4.8.0.dev0"))
    with pytest.raises(NotImplementedError, match=r"dense convolution requires nvidia-cutlass-dsl >= 4\.9; found 4\.8\.0\.dev0"):
        _Sm100FrostConvPlan.check_support(None)

    dsl_state(True, ("nvidia-cutlass-dsl", "4.7.0"))
    with pytest.raises(NotImplementedError, match=r"block-scale convolution requires nvidia-cutlass-dsl >= 4\.8; found 4\.7\.0"):
        _Sm100FrostBlockScaleConvPlan.check_support(None)


@pytest.mark.parametrize(
    "block_scale_data,version,error",
    (
        (None, "4.8.0.dev0", r"dense convolution requires nvidia-cutlass-dsl >= 4\.9"),
        (object(), "4.7.0", r"block-scale convolution requires nvidia-cutlass-dsl >= 4\.8"),
    ),
    ids=("dense", "block-scale"),
)
def test_build_plan_gates_before_importing_the_template(dsl_state, monkeypatch, block_scale_data, version, error) -> None:
    from cudnn.conv.frost import engine

    analysis = SimpleNamespace(
        conv_node=SimpleNamespace(params={}),
        image=SimpleNamespace(dim=(1, 64, 1, 1, 1)),
        weight=SimpleNamespace(dim=(64, 64, 1, 1, 1)),
        output=object(),
        block_scale_data=block_scale_data,
    )
    dsl_state(True, ("nvidia-cutlass-dsl", version))
    monkeypatch.setattr(engine, "analyze", lambda graph: analysis)
    monkeypatch.setattr(engine, "build_device", lambda device: nullcontext())

    with pytest.raises(NotImplementedError, match=error):
        engine.FrostConvEngine().build_plan(None, None)


def test_missing_cutedsl_declines_both_variants(dsl_state) -> None:
    dsl_state(False, None)

    for minimum_version in (_cutedsl.BLOCK_SCALE_CUTEDSL_MIN_VERSION, _cutedsl.DENSE_CUTEDSL_MIN_VERSION):
        assert "not installed" in _cutedsl.requirement_error("frost_conv", minimum_version)
