# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTLASS DSL version-gate tests for Rubin MegaMoE."""

import pytest


@pytest.mark.L0
@pytest.mark.parametrize("version", ["4.5.0", "4.6.1", "4.7.0"])
def test_rubin_cutedsl_gate_rejects_public_wheels_below_4_8(
    monkeypatch,
    version,
):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _cutedsl

    monkeypatch.setattr(_cutedsl, "_public_cutedsl_version", lambda: version)

    with pytest.raises(
        RuntimeError,
        match=r"nvidia-cutlass-dsl>=4\.8\.0",
    ):
        _cutedsl.require_rubin_cutedsl()


@pytest.mark.L0
@pytest.mark.parametrize("version", ["4.8.0", "4.8.0rc1", "4.9.0"])
def test_rubin_cutedsl_gate_accepts_4_8_or_newer(monkeypatch, version):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _cutedsl

    monkeypatch.setattr(_cutedsl, "_public_cutedsl_version", lambda: version)

    _cutedsl.require_rubin_cutedsl()


@pytest.mark.L0
@pytest.mark.parametrize(
    "module_name, function_name",
    [
        (
            "cudnn.moe_ep._megamoe_backend.mxfp8._compile",
            "prepare_kernel",
        ),
        (
            "cudnn.moe_ep._megamoe_backend.mxfp8._backward_compile",
            "prepare_backward_kernel",
        ),
    ],
)
def test_rubin_prepare_gates_before_cuda_initialization(
    monkeypatch,
    module_name,
    function_name,
):
    module = __import__(module_name, fromlist=[function_name])

    class GateReached(RuntimeError):
        pass

    def reject():
        raise GateReached

    monkeypatch.setattr(module, "require_rubin_cutedsl", reject)

    with pytest.raises(GateReached):
        getattr(module, function_name)(None, None, None)
