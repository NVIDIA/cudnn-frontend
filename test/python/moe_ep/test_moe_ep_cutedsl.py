# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""CUTLASS DSL version-gate tests for Rubin MegaMoE."""

import pytest
import torch


@pytest.mark.L0
@pytest.mark.parametrize(
    ("data_capacity", "scale_capacity", "expected_capacity"),
    ((128, 128, 128), (256, 128, 256), (128, 256, 256)),
)
def test_deterministic_token_comm_scale_capacity_covers_data_pool(
    data_capacity,
    scale_capacity,
    expected_capacity,
):
    from types import SimpleNamespace

    from cudnn.moe_ep._megamoe_backend.cutedsl_src.communication.nvlink_domain.token_comm_deterministic import (
        TokenCommDeterministic,
    )

    comm = object.__new__(TokenCommDeterministic)
    comm._router = SimpleNamespace(
        worst_case_token_count=data_capacity,
        receive_capacity=lambda padding_block: SimpleNamespace(padded_route_count=scale_capacity),
    )
    comm.sf_padding_block = 128

    assert comm.worst_case_sf_token_count == expected_capacity


@pytest.mark.L0
@pytest.mark.parametrize(
    ("ready_padded_capacity", "ready_slots"),
    ((256, 1), (512, 2), (1024, 4), (2304, 9)),
)
def test_deterministic_token_comm_uses_ready_slot_padding_contract(
    ready_padded_capacity,
    ready_slots,
):
    from types import SimpleNamespace

    from cudnn.moe_ep._megamoe_backend.cutedsl_src.communication.nvlink_domain.token_comm_deterministic import (
        TokenCommDeterministic,
    )

    comm = object.__new__(TokenCommDeterministic)
    comm._router = SimpleNamespace(
        receive_capacity=lambda padding_block: SimpleNamespace(padded_route_count=ready_padded_capacity),
    )
    comm.tokens_per_fc1_ready_slot = 256

    assert comm.max_fc1_ready_slot_count == ready_slots


@pytest.mark.L0
def test_rubin_cutedsl_gate_rejects_public_wheels_below_4_8(
    monkeypatch,
):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _cutedsl

    monkeypatch.setattr(_cutedsl, "_public_cutedsl_version", lambda: "4.7.0")

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
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _compile_common

    class GateReached(RuntimeError):
        pass

    def reject():
        raise GateReached

    monkeypatch.setattr(_compile_common, "require_rubin_cutedsl", reject)

    with pytest.raises(GateReached):
        getattr(module, function_name)(None, None, None)


@pytest.mark.L0
@pytest.mark.parametrize("context", ["forward", "backward"])
def test_rubin_environment_errors_include_compile_context(
    monkeypatch,
    context,
):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _compile_common

    monkeypatch.setattr(_compile_common, "require_rubin_cutedsl", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: (10, 0),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"Rubin MXFP8 {context} preparation",
    ):
        _compile_common._prepare_rubin_environment(
            torch.device("cuda", 0),
            2,
            context=context,
        )


@pytest.mark.L0
def test_rubin_environment_rejects_incompatible_arch_override(
    monkeypatch,
):
    from cudnn.moe_ep._megamoe_backend.mxfp8 import _compile_common

    monkeypatch.setattr(_compile_common, "require_rubin_cutedsl", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: (10, 7),
    )
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")

    with pytest.raises(
        RuntimeError,
        match=r"CUTE_DSL_ARCH.*forward.*sm_100a",
    ):
        _compile_common._prepare_rubin_environment(
            torch.device("cuda", 0),
            2,
            context="forward",
        )
