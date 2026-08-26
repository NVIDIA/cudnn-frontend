# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Core MoE EP backward contract, parity, and distributed tests."""

from __future__ import annotations

import inspect
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from cudnn.moe_ep import MoeEp
from cudnn.moe_ep._contracts import ForwardConfig
from cudnn.moe_ep._megamoe_backend import _capability
from cudnn.moe_ep._megamoe_backend.mxfp8._backend import Mxfp8Backend
from cudnn.moe_ep._megamoe_backend.mxfp8._backward_layout import (
    Mxfp8BackwardLayout,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._backward_staging import (
    _stage_fc1_preact,
    stage_backward,
)
from cudnn.moe_ep._megamoe_backend._workspace import WorkspaceRequirements
from cudnn.moe_ep._megamoe_backend.mxfp8._backward_dispatch import (
    Mxfp8BackwardRedispatch,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._backward_dprob import (
    return_grad_topk_weights,
)
from cudnn.moe_ep._tuning import MoeEpTuningConfig
from cudnn.moe_ep._validation import validate_backward
from moe_ep.moe_ep_backward_support import (
    _assert_backward_matches,
    _expected_backward,
    _grad_output,
    _reference_backward,
)
from moe_ep.moe_ep_distributed_workers import (
    _distributed_backward_worker,
)
from moe_ep.moe_ep_forward_support import (
    _forward_config,
    _require_distributed_sm107,
    _sm107_device,
)
from moe_ep.moe_ep_reference import (
    MoeFormat,
    backward_combine_round_trip,
    forward_combine_round_trip,
)
from moe_ep.moe_ep_test_data import (
    make_forward_inputs,
    quantize_mxfp8,
)


def _config(**overrides) -> ForwardConfig:
    values = {
        "num_experts": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "top_k": 2,
        "experts_per_rank": 2,
        "ep_size": 1,
        "ep_rank": 0,
        "ep_group": None,
        "ep_global_ranks": (),
        "max_tokens_per_rank": 4,
        "output_format": "bf16",
        "combine_format": "bf16",
        "apply_topk_in_fc1": True,
        "gate_up_clamp": None,
        "generate_c": True,
        "token_padding_size": 128,
        "sf_padding_size": 128,
        "tuning": MoeEpTuningConfig(),
    }
    values.update(overrides)
    return ForwardConfig(**values)


def _inputs():
    activation = torch.randn(2, 128, dtype=torch.bfloat16)
    fc1_weight = torch.randn(2, 128, 512, dtype=torch.bfloat16)
    fc2_weight = torch.randn(2, 256, 128, dtype=torch.bfloat16)
    topk_idx = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
    topk_weights = torch.randn(2, 2, dtype=torch.float32)
    return activation, fc1_weight, fc2_weight, topk_idx, topk_weights


def _validate_backward(
    config,
    grad_output,
    args,
    fc1_c,
    route_metadata,
):
    return validate_backward(
        config,
        grad_output,
        *args[1:],
        fc1_c,
        route_metadata,
    )


# Validation, layout, staging, and backend contracts.


@pytest.mark.L0
def test_validate_backward_builds_typed_request_and_checks_stash():
    config = _config()
    args = _inputs()
    grad_output = torch.randn(2, 128)
    fc1_c = torch.randn(3, 512, dtype=torch.bfloat16)
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )

    request = _validate_backward(
        config,
        grad_output,
        args,
        fc1_c,
        route_metadata,
    )

    assert request.config is config
    assert request.fc1_weight is args[1]
    assert request.topk_idx is args[3]
    assert request.local_routes == 3
    with pytest.raises(ValueError, match="fc1_c shape must be"):
        _validate_backward(
            config,
            grad_output,
            args,
            fc1_c[:2],
            route_metadata,
        )
    with pytest.raises(TypeError, match="route_metadata must have dtype"):
        _validate_backward(
            config,
            grad_output,
            args,
            fc1_c,
            route_metadata.to(torch.int64),
        )


@pytest.mark.L0
def test_mxfp8_backward_layout_builds_public_preactivation_lut():
    config = _config()
    args = _inputs()
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        route_metadata,
    )

    layout = Mxfp8BackwardLayout.from_request(request)

    assert layout.preact_row_lut[0, 0, 0].item() == 0
    assert layout.preact_row_lut[0, 1, 1].item() == 1
    assert layout.preact_row_lut[0, 1, 0].item() == 2


@pytest.mark.L0
def test_mxfp8_backward_stages_compact_preactivation_into_pool_rows():
    config = _config()
    args = _inputs()
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    fc1_c = torch.arange(
        3 * 512,
        dtype=torch.float32,
    ).reshape(3, 512).to(torch.bfloat16)
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        fc1_c,
        route_metadata,
    )
    layout = Mxfp8BackwardLayout.from_request(request)
    pool_capacity = 256
    fc1_preact = torch.empty(
        pool_capacity,
        512,
        dtype=torch.bfloat16,
    )
    prepared = SimpleNamespace(
        config=SimpleNamespace(
            intermediate=256,
            num_experts=2,
            token_padding_block=128,
        ),
        kernel=SimpleNamespace(
            token_comm=SimpleNamespace(
                router_data_cta_count=1,
                router_warps_per_cta=4,
            )
        ),
        pool_token_capacity=pool_capacity,
    )

    _stage_fc1_preact(request, layout, prepared, fc1_preact)

    staged = fc1_preact
    gate, up = fc1_c.split(256, dim=1)
    expected = torch.stack(
        (gate.reshape(3, 8, 32), up.reshape(3, 8, 32)),
        dim=2,
    ).reshape(3, 512)
    torch.testing.assert_close(staged[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(staged[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(staged[128], expected[2], rtol=0, atol=0)
    assert staged[2:128].eq(0).all()
    assert staged[129:].eq(0).all()


@pytest.mark.L0
def test_mxfp8_backward_stages_sources_in_destination_ring_order():
    metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=torch.int64,
    )
    fc1_c = torch.stack(
        (
            torch.cat(
                (
                    torch.full((32,), 10, dtype=torch.bfloat16),
                    torch.full((32,), 11, dtype=torch.bfloat16),
                )
            ),
            torch.cat(
                (
                    torch.full((32,), 20, dtype=torch.bfloat16),
                    torch.full((32,), 21, dtype=torch.bfloat16),
                )
            ),
        )
    )
    request = SimpleNamespace(
        config=SimpleNamespace(
            ep_rank=1,
            ep_size=2,
            max_tokens_per_rank=1,
            top_k=1,
        ),
        route_metadata=metadata,
        fc1_c=fc1_c,
    )
    layout = SimpleNamespace(
        preact_row_lut=torch.tensor([[[0]], [[1]]], dtype=torch.int32)
    )
    pool_capacity = 128
    fc1_preact = torch.empty(
        pool_capacity,
        64,
        dtype=torch.bfloat16,
    )
    prepared = SimpleNamespace(
        config=SimpleNamespace(
            intermediate=32,
            num_experts=1,
            token_padding_block=128,
        ),
        kernel=SimpleNamespace(
            token_comm=SimpleNamespace(
                router_data_cta_count=1,
                router_warps_per_cta=4,
            )
        ),
        pool_token_capacity=pool_capacity,
    )

    _stage_fc1_preact(request, layout, prepared, fc1_preact)

    staged = fc1_preact
    # Destination rank 1 receives source rank 1 before wrapped source rank 0.
    expected = torch.stack(
        (fc1_c[:, :32], fc1_c[:, 32:]),
        dim=1,
    ).reshape(2, 64)
    torch.testing.assert_close(staged[0], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(staged[1], expected[0], rtol=0, atol=0)


@pytest.mark.L0
def test_mxfp8_backward_stages_source_routes_in_router_vector_order():
    metadata = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 2, 0],
            [0, 0, 3, 1],
            [0, 0, 4, 0],
        ],
        dtype=torch.int64,
    )
    fc1_c = torch.arange(5, dtype=torch.bfloat16).view(5, 1).expand(
        5,
        64,
    ).contiguous()
    request = SimpleNamespace(
        config=SimpleNamespace(
            ep_rank=0,
            ep_size=1,
            max_tokens_per_rank=5,
            top_k=2,
        ),
        route_metadata=metadata,
        fc1_c=fc1_c,
    )
    preact_row_lut = torch.full((1, 5, 2), -1, dtype=torch.int32)
    preact_row_lut[
        metadata[:, 1],
        metadata[:, 2],
        metadata[:, 3],
    ] = torch.arange(5, dtype=torch.int32)
    layout = SimpleNamespace(preact_row_lut=preact_row_lut)
    pool_capacity = 128
    fc1_preact = torch.empty(
        pool_capacity,
        64,
        dtype=torch.bfloat16,
    )
    prepared = SimpleNamespace(
        config=SimpleNamespace(
            intermediate=32,
            num_experts=1,
            token_padding_block=128,
        ),
        kernel=SimpleNamespace(
            token_comm=SimpleNamespace(
                router_data_cta_count=1,
                router_warps_per_cta=4,
            )
        ),
        pool_token_capacity=pool_capacity,
    )

    _stage_fc1_preact(request, layout, prepared, fc1_preact)

    staged = fc1_preact
    # Int32 router loads four adjacent routes per thread, then stable-sorts by
    # register round and lane: flat routes 0,4,8 precede 3,7.
    assert staged[:5, 0].tolist() == [0, 2, 4, 1, 3]


@pytest.mark.L0
def test_mxfp8_backward_workspace_regions_are_explicit_and_symmetric():
    requirements = WorkspaceRequirements.for_mxfp8(
        _config(),
        kernel_local_workspace_bytes=64,
        kernel_shared_workspace_bytes=128,
        backward_fc1_preact_bytes=1024,
        backward_dprob_bytes=32,
        backward_aux_data_bytes=512,
        backward_aux_scale_bytes=256,
    )
    symmetric = {
        region.name: region for region in requirements.symmetric_regions
    }
    local = {region.name: region for region in requirements.local_regions}

    assert symmetric["backward_dprob"].nbytes == 32
    assert local["backward_fc1_preact"].nbytes == 1024
    assert local["backward_fc1_preact"].alignment == 128
    assert local["backward_aux_data"].nbytes == 512
    assert local["backward_aux_scale"].nbytes == 256

    with pytest.raises(ValueError, match="must be enabled together"):
        WorkspaceRequirements.for_mxfp8(
            _config(),
            kernel_local_workspace_bytes=64,
            kernel_shared_workspace_bytes=128,
            backward_dprob_bytes=32,
        )


@pytest.mark.L0
def test_rubin_adapter_source_tracks_current_kernel_signatures():
    from cudnn.moe_ep._megamoe_backend.mxfp8 import (
        _backward_compile,
        _compile,
    )

    forward_source = inspect.getsource(_compile.prepare_kernel)
    backward_source = inspect.getsource(_backward_compile.prepare_backward_kernel)
    runtime_source = inspect.getsource(
        _backward_compile.build_backward_runtime_kwargs
    )

    assert "apply_topk_in_fc1=config.apply_topk_in_fc1" not in forward_source
    assert "gate_up_clamp=config.gate_up_clamp" in backward_source
    assert "dfc2_recompute=dfc2_recompute" in backward_source
    assert "dfc2_col_output=dfc2_col_output" in backward_source
    assert "enable_grad_y2_col_quant=enable_grad_y2_col_quant" in backward_source
    assert '"fc1_preact":' in runtime_source
    overflow_runtime_source = runtime_source.split(
        '"overflow_flag":',
        1,
    )[1].split('"dprob":', 1)[0]
    assert "dynamic_layout=False" in overflow_runtime_source
    for output_name in (
        "dprob",
        "fc1_recompute",
        "fc1_recompute_sf",
        "fc1_col_output",
        "fc1_col_output_sf",
        "grad_y2",
        "grad_y2_sf",
    ):
        assert f'"{output_name}":' in runtime_source


@pytest.mark.L0
def test_stage_backward_exposes_fixed_aux_shapes_and_resets_symmetric_dprob():
    config = _config()
    args = _inputs()
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        route_metadata,
    )
    layout = Mxfp8BackwardLayout.from_request(request)
    aux_shapes = {
        "dprob": (4, 2),
        "fc1_recompute": (8, 256),
        "fc1_recompute_sf": (1, 256),
        "fc1_col_output": (8, 512),
        "fc1_col_output_sf": (1, 512),
        "grad_y2": (8, 128),
        "grad_y2_sf": (32,),
    }
    kernel = SimpleNamespace(
        token_comm=SimpleNamespace(
            router_data_cta_count=1,
            router_warps_per_cta=4,
        ),
        get_fc1_preact_shape=lambda: (256, 512),
        get_aux_output_shapes=lambda: aux_shapes,
    )
    prepared = SimpleNamespace(
        config=SimpleNamespace(
            max_tokens_per_rank=4,
            hidden=128,
            top_k=2,
            intermediate=256,
            num_experts=2,
            combine_format="bf16",
            token_padding_block=128,
        ),
        kernel=kernel,
        pool_token_capacity=256,
        pre_reduced_activation_offset=0,
        pre_reduced_activation_bytes_per_token=4,
        pre_reduced_activation_sf_offset=None,
        pre_reduced_activation_sf_bytes_per_token=0,
        local_workspace_zero_bytes=0,
        shared_workspace_zero_bytes=0,
        dfc2_recompute=False,
        dfc2_col_output=False,
        enable_grad_y2_col_quant=False,
    )
    symmetric_dprob = torch.full((32,), 0x7F, dtype=torch.uint8)
    resources = SimpleNamespace(
        workspace=SimpleNamespace(
            symmetric={
                "activation_data": torch.empty(4 * 128, dtype=torch.uint8),
                "activation_scale": torch.empty(4 * 16, dtype=torch.uint8),
                "topk_weights": torch.empty(4 * 2 * 4, dtype=torch.uint8),
                "output_data": torch.empty(4 * 128 * 2, dtype=torch.uint8),
                "backward_dprob": symmetric_dprob,
                "kernel_shared_workspace": torch.empty(
                    64,
                    dtype=torch.uint8,
                ),
            },
            local={
                "topk_idx": torch.empty(4 * 2 * 4, dtype=torch.uint8),
                "overflow_flag": torch.empty(4, dtype=torch.uint8),
                "backward_fc1_preact": torch.empty(
                    256 * 512 * 2,
                    dtype=torch.uint8,
                ),
                "backward_aux_data": torch.empty(
                    8 * 512,
                    dtype=torch.uint8,
                ),
                "backward_aux_scale": torch.empty(
                    512,
                    dtype=torch.uint8,
                ),
                "kernel_local_workspace": torch.empty(
                    64,
                    dtype=torch.uint8,
                ),
            },
        )
    )

    inputs = stage_backward(request, layout, prepared, resources)

    assert inputs.fc1_preact.shape == (256, 512)
    assert inputs.fc1_preact.dtype is torch.bfloat16
    assert inputs.dprob.shape == (4, 2)
    assert inputs.dprob.dtype is torch.float32
    assert inputs.dprob.eq(0).all()
    assert inputs.fc1_recompute.shape == aux_shapes["fc1_recompute"]
    assert inputs.fc1_recompute.dtype is torch.float8_e4m3fn
    assert inputs.fc1_recompute_sf.shape == aux_shapes["fc1_recompute_sf"]
    assert inputs.fc1_recompute_sf.dtype is torch.float8_e8m0fnu
    assert inputs.fc1_col_output.shape == aux_shapes["fc1_col_output"]
    assert inputs.fc1_col_output_sf.shape == aux_shapes["fc1_col_output_sf"]
    assert (
        inputs.fc1_recompute.data_ptr()
        == inputs.fc1_col_output.data_ptr()
    )
    assert (
        inputs.fc1_recompute_sf.data_ptr()
        == inputs.fc1_col_output_sf.data_ptr()
    )
    assert inputs.grad_y2.shape == aux_shapes["grad_y2"]
    assert inputs.grad_y2_sf.shape == aux_shapes["grad_y2_sf"]
    assert inputs.grad_y2.data_ptr() == inputs.fc1_recompute.data_ptr()
    assert inputs.grad_y2_sf.data_ptr() == inputs.fc1_recompute_sf.data_ptr()

    operands_config = replace(
        config,
        backward_wgrad_mode="operands",
        token_padding_size=256,
    )
    operands_request = replace(request, config=operands_config)
    operands_aux_shapes = {
        "dprob": (4, 2),
        "fc1_recompute": (512, 256),
        "fc1_recompute_sf": (8, 256),
        "fc1_col_output": (512, 512),
        "fc1_col_output_sf": (8, 512),
        "grad_y2": (512, 128),
        "grad_y2_sf": (256 // 32 * 128,),
    }
    operands_kernel = SimpleNamespace(
        token_comm=kernel.token_comm,
        get_fc1_preact_shape=lambda: (512, 512),
        get_aux_output_shapes=lambda: operands_aux_shapes,
    )
    operands_prepared = SimpleNamespace(
        **{
            **vars(prepared),
            "config": SimpleNamespace(
                **{
                    **vars(prepared.config),
                    "token_padding_block": 256,
                }
            ),
            "kernel": operands_kernel,
            "pool_token_capacity": 512,
            "dfc2_recompute": True,
            "dfc2_col_output": True,
            "enable_grad_y2_col_quant": True,
        }
    )
    operands_local = dict(resources.workspace.local)
    operands_local["backward_fc1_preact"] = torch.empty(
        512 * 512 * 2,
        dtype=torch.uint8,
    )
    operands_resources = SimpleNamespace(
        workspace=SimpleNamespace(
            symmetric=resources.workspace.symmetric,
            local=operands_local,
        )
    )

    operand_inputs = stage_backward(
        operands_request,
        Mxfp8BackwardLayout.from_request(operands_request),
        operands_prepared,
        operands_resources,
    )

    assert operand_inputs.fc1_recompute.shape == (512, 256)
    assert operand_inputs.fc1_col_output.shape == (512, 512)
    assert operand_inputs.grad_y2.shape == (512, 128)
    assert operand_inputs.grad_y2_sf.shape == (
        256 // 32 * 128,
    )
    assert (
        operand_inputs.fc1_recompute.data_ptr()
        != operands_local["backward_aux_data"].data_ptr()
    )
    assert (
        operand_inputs.fc1_col_output.data_ptr()
        != operands_local["backward_aux_data"].data_ptr()
    )
    assert operand_inputs.fc1_recompute.eq(0).all()
    assert operand_inputs.fc1_col_output.eq(0).all()
    assert operand_inputs.grad_y2.eq(0).all()
    assert operand_inputs.fc1_preact[128:256].eq(0).all()
    assert torch.equal(
        operand_inputs.fc1_preact[256],
        torch.stack(
            (
                request.fc1_c[2, :256].reshape(8, 32),
                request.fc1_c[2, 256:].reshape(8, 32),
            ),
            dim=1,
        ).reshape(512),
    )


@pytest.mark.L0
@pytest.mark.parametrize("apply_topk_in_fc1", [True, False])
def test_mxfp8_backward_recomputes_semantic_grad_topk_weights(
    apply_topk_in_fc1,
):
    args = list(_inputs())
    fc2_weight = torch.zeros_like(args[2])
    fc2_weight[0, 0, 0] = 2
    fc2_weight[1, 0, 0] = 4
    args[2] = fc2_weight
    config = _config(apply_topk_in_fc1=apply_topk_in_fc1)
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    fc1_c = torch.zeros(3, 512, dtype=torch.bfloat16)
    fc1_c[:, 0] = 1
    fc1_c[:, 256] = 1
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        fc1_c,
        route_metadata,
    )
    redispatched_grad_output = torch.zeros(3, 128)
    redispatched_grad_output[:, 0] = torch.tensor([1.0, 2.0, 3.0])

    grad_topk = return_grad_topk_weights(
        request,
        redispatched_grad_output,
    )
    silu_one = torch.sigmoid(torch.tensor(1.0))
    torch.testing.assert_close(
        grad_topk,
        silu_one * torch.tensor([[2.0, 0.0], [12.0, 4.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.L0
def test_mxfp8_backward_dprob_recompute_applies_gate_up_clamp():
    args = list(_inputs())
    fc2_weight = torch.zeros_like(args[2])
    fc2_weight[0, 0, 0] = 2
    fc2_weight[1, 0, 0] = 4
    args[2] = fc2_weight
    config = _config(gate_up_clamp=0.5)
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    fc1_c = torch.zeros(3, 512, dtype=torch.bfloat16)
    fc1_c[:, 0] = 2
    fc1_c[:, 256] = 2
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        fc1_c,
        route_metadata,
    )
    redispatched_grad_output = torch.zeros(3, 128)
    redispatched_grad_output[:, 0] = torch.tensor([1.0, 2.0, 3.0])

    grad_topk = return_grad_topk_weights(
        request,
        redispatched_grad_output,
    )

    clamped_hidden = 0.25 * torch.sigmoid(torch.tensor(0.5))
    torch.testing.assert_close(
        grad_topk,
        clamped_hidden * torch.tensor([[2.0, 0.0], [12.0, 4.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.L0
def test_mxfp8_grad_output_redispatch_uses_public_route_order():
    config = _config()
    args = _inputs()
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        route_metadata,
    )

    actual = Mxfp8BackwardRedispatch(request).run()
    expected_rows = torch.tensor([0, 1, 1], dtype=torch.int64)

    torch.testing.assert_close(
        actual.grad_output,
        request.grad_output.index_select(0, expected_rows).float(),
        rtol=0,
        atol=0,
    )


@pytest.mark.L0
def test_moe_ep_backward_delegates_validated_request(monkeypatch):
    import cudnn.moe_ep._backend as backend_seam

    args = _inputs()
    grad_output = torch.randn(2, 128)
    fc1_c = torch.randn(3, 512, dtype=torch.bfloat16)
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    expected = (
        torch.empty(2, 128, dtype=torch.float32),
        torch.empty(2, 2, dtype=torch.float32),
    )

    class Backend:
        request = None

        def backward(self, request):
            self.request = request
            return expected

        def close(self):
            pass

    instance = Backend()
    monkeypatch.setattr(backend_seam, "validate_config", lambda config: None)
    monkeypatch.setattr(
        backend_seam,
        "validate_backward_request",
        lambda request: None,
    )
    monkeypatch.setattr(
        backend_seam,
        "create_backend",
        lambda config, device: instance,
    )

    operator = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=4,
        generate_c=True,
    )
    actual = operator.backward(
        grad_output,
        *args[1:],
        fc1_c,
        route_metadata,
    )

    assert actual is expected
    assert len(actual) == 2
    assert instance.request is not None
    assert instance.request.local_routes == 3
    assert instance.request.fc1_c is fc1_c
    assert instance.request.fc1_weight is args[1]


@pytest.mark.L0
def test_moe_ep_backward_accepts_explicit_stashes_in_reordered_calls(
    monkeypatch,
):
    import cudnn.moe_ep._backend as backend_seam

    calls = []

    class Backend:
        def backward(self, request):
            calls.append(request)
            marker = request.fc1_c[0, 0].float().reshape(1)
            return marker, marker

        def close(self):
            pass

    backend = Backend()
    monkeypatch.setattr(backend_seam, "validate_config", lambda config: None)
    monkeypatch.setattr(
        backend_seam,
        "validate_backward_request",
        lambda request: None,
    )
    monkeypatch.setattr(
        backend_seam,
        "create_backend",
        lambda config, device: backend,
    )
    operator = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=4,
        generate_c=True,
    )
    args = _inputs()
    route_metadata = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
        dtype=torch.int32,
    )
    stash_a = torch.full((3, 512), 1.0, dtype=torch.bfloat16)
    stash_b = torch.full((3, 512), 2.0, dtype=torch.bfloat16)

    markers = []
    for stash in (stash_a, stash_b, stash_b, stash_a):
        result = operator.backward(
            torch.randn(2, 128),
            *args[1:],
            stash,
            route_metadata,
        )
        markers.append(result[0].item())

    assert markers == [1.0, 2.0, 2.0, 1.0]
    assert calls[0].fc1_c is stash_a
    assert calls[1].fc1_c is stash_b
    assert calls[2].fc1_c is stash_b
    assert calls[3].fc1_c is stash_a


@pytest.mark.L0
def test_moe_ep_backward_requires_generate_c():
    args = _inputs()
    operator = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=4,
        generate_c=False,
    )

    with pytest.raises(RuntimeError, match="generate_c=True"):
        operator.backward(
            torch.randn(2, 128),
            *args[1:],
            torch.randn(3, 512, dtype=torch.bfloat16),
            torch.zeros(3, 4, dtype=torch.int32),
        )


@pytest.mark.L0
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_format": "mxfp8"}, "output_format='bf16'"),
        ({"apply_topk_in_fc1": False}, "apply_topk_in_fc1=True"),
    ],
)
def test_mxfp8_backward_capability_rejects_unsupported_config(
    monkeypatch,
    overrides,
    message,
):
    config = _config(**overrides)
    args = _inputs()
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        torch.zeros(3, 4, dtype=torch.int32),
    )
    monkeypatch.setattr(_capability, "_validate_device", lambda device: None)
    monkeypatch.setattr(
        _capability,
        "_is_cuda_stream_capturing",
        lambda device: False,
    )

    with pytest.raises(
        NotImplementedError,
        match=message,
    ):
        _capability.validate_backward_request(request)


@pytest.mark.L0
def test_mxfp8_backward_capability_accepts_gate_up_clamp(monkeypatch):
    config = _config(gate_up_clamp=1.0)
    args = _inputs()
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        torch.zeros(3, 4, dtype=torch.int32),
    )
    monkeypatch.setattr(_capability, "_validate_device", lambda device: None)
    monkeypatch.setattr(
        _capability,
        "_is_cuda_stream_capturing",
        lambda device: False,
    )

    _capability.validate_backward_request(request)


@pytest.mark.L0
def test_mxfp8_backward_capability_accepts_ep_above_16(monkeypatch):
    config = _config(ep_size=32, ep_rank=31)
    args = _inputs()
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        torch.zeros(3, 4, dtype=torch.int32),
    )
    monkeypatch.setattr(_capability, "_validate_device", lambda device: None)
    monkeypatch.setattr(
        _capability,
        "_is_cuda_stream_capturing",
        lambda device: False,
    )

    _capability.validate_backward_request(request)


@pytest.mark.L0
def test_mxfp8_backward_capability_rejects_cuda_graph_capture(monkeypatch):
    config = _config()
    args = _inputs()
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        torch.zeros(3, 4, dtype=torch.int32),
    )
    monkeypatch.setattr(_capability, "_validate_device", lambda device: None)
    monkeypatch.setattr(
        _capability,
        "_is_cuda_stream_capturing",
        lambda device: True,
    )

    with pytest.raises(NotImplementedError, match="CUDA graph capture"):
        _capability.validate_backward_request(request)


@pytest.mark.L0
def test_mxfp8_backward_delegates_to_explicit_executor(monkeypatch):
    import cudnn.moe_ep._megamoe_backend.mxfp8._backend as backend_module

    config = _config()
    args = _inputs()
    request = _validate_backward(
        config,
        torch.randn(2, 128),
        args,
        torch.randn(3, 512, dtype=torch.bfloat16),
        torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
            dtype=torch.int32,
        ),
    )
    expected = tuple(torch.empty(0) for _ in range(2))

    class Executor:
        def __init__(self, actual_config, actual_device):
            assert actual_config is config
            assert actual_device == torch.device("cpu")

        def run(self, actual_request):
            assert actual_request is request
            return expected

        def close(self):
            pass

    monkeypatch.setattr(
        backend_module,
        "Mxfp8BackwardExecutor",
        Executor,
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    class Stream:
        def wait_event(self, event):
            del event

    class Event:
        def record(self, stream):
            del stream

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    backend = Mxfp8Backend(config, torch.device("cpu"))
    assert backend.backward(request) is expected


# Single-rank and distributed backward numerical parity.


def _make_reentrant_case_b(args, device):
    activation = quantize_mxfp8(
        args[0].dequantize(dtype=torch.float32) + 0.25,
        axis=1,
    )
    topk_idx = torch.tensor(
        [[0, 0], [-1, -1], [0, -1], [0, 0], [0, -1]],
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.tensor(
        [
            [0.625, 0.375],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.75, 0.25],
            [1.0, 0.0],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    return activation, args[1], args[2], topk_idx, topk_weights


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("combine_format", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "gate_up_clamp",
    [None, 0.5],
    ids=["unclamped", "clamped"],
)
def test_mxfp8_backward_ep1_matches_reference_and_resets_workspace(
    combine_format,
    gate_up_clamp,
):
    device = _sm107_device()
    args = make_forward_inputs(device)
    config = _forward_config(
        generate_c=True,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
    )
    reference = _reference_backward(config)
    grad_output = _grad_output(device, args[3].shape[0], seed=20260817)

    with MoeEp(**config) as op:
        _, fc1_c, route_metadata = op(*args)
        stash = (fc1_c, route_metadata)
        expected = _expected_backward(reference, grad_output, args, stash)

        first = op.backward(grad_output, *args[1:], *stash)
        second = op.backward(grad_output, *args[1:], *stash)
        torch.cuda.synchronize(device)

    _assert_backward_matches(first, expected, args[3])
    _assert_backward_matches(second, expected, args[3])


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_mxfp8_backward_ep1_uses_explicit_stash_after_reordered_forwards():
    device = _sm107_device()
    args_a = make_forward_inputs(device)
    args_b = _make_reentrant_case_b(args_a, device)
    config = _forward_config(generate_c=True)
    reference = _reference_backward(config)
    grad_a = _grad_output(device, args_a[3].shape[0], seed=20260818)
    grad_b = _grad_output(device, args_b[3].shape[0], seed=20260819)

    with MoeEp(**config) as op:
        _, fc1_c_a, metadata_a = op(*args_a)
        _, fc1_c_b, metadata_b = op(*args_b)
        cases = (
            (grad_b, args_b, (fc1_c_b, metadata_b)),
            (grad_a, args_a, (fc1_c_a, metadata_a)),
            (grad_b, args_b, (fc1_c_b, metadata_b)),
            (grad_a, args_a, (fc1_c_a, metadata_a)),
        )
        results = []
        for grad_output, args, stash in cases:
            expected = _expected_backward(reference, grad_output, args, stash)
            actual = op.backward(grad_output, *args[1:], *stash)
            results.append((actual, expected, args[3]))
        torch.cuda.synchronize(device)

    for actual, expected, topk_idx in results:
        _assert_backward_matches(actual, expected, topk_idx)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("world_size", [2, 4], ids=["ep2", "ep4"])
@pytest.mark.parametrize("combine_format", ["bf16", "mxfp8"])
def test_mxfp8_backward_multi_gpu_matches_reference(
    world_size,
    combine_format,
    tmp_path,
):
    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = (
        tmp_path
        / f"{combine_format}_combine_mxfp8_backward_ep{world_size}.init"
    )
    mp.spawn(
        _distributed_backward_worker,
        args=(world_size, str(init_file), combine_format),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_mxfp8_backward_ep2_gate_up_clamp_matches_reference(tmp_path):
    world_size = 2
    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / "bf16_combine_clamped_mxfp8_backward_ep2.init"
    mp.spawn(
        _distributed_backward_worker,
        args=(world_size, str(init_file), "bf16", 0.5),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L0
def test_forward_and_backward_mxfp8_combine_are_direct_fp32():
    generator = torch.Generator().manual_seed(20260820)
    accumulator = torch.randn(4, 128, generator=generator) * 3.25

    backward = backward_combine_round_trip(
        accumulator,
        MoeFormat.MXFP8,
    )
    forward = forward_combine_round_trip(
        accumulator,
        MoeFormat.MXFP8,
    )

    torch.testing.assert_close(backward, forward, rtol=0, atol=0)
