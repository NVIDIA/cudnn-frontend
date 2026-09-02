# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stateless MoE EP training contracts."""

from __future__ import annotations

from dataclasses import fields

import cudnn
import pytest
import torch

from cudnn.moe_ep import (
    BlockScaledTensor,
    MoeEp,
    MoeEpBackwardWeightStaging,
    MoeEpBackwardWeights,
    MoeEpExecutionLane,
    MoeEpForwardWeights,
    MoeEpNativeBackwardWeights,
    MoeEpNativeForwardWeights,
    MoeEpNativeWeight,
    MoeEpNativeWeightLayout,
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingWgradOperands,
    pack_backward_weights,
    pack_forward_weights,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_resources import (
    Mxfp8TrainingState,
    _build_training_abi_facts,
    _harmonize_symmetric_regions,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_execute import _stage_input
from cudnn.moe_ep._megamoe_backend.mxfp8._training_weights import (
    backward_native_to_kernel,
    forward_native_to_kernel,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_wgrad import (
    assemble_training_wgrad_operands,
)
from cudnn.moe_ep._megamoe_backend._workspace import (
    BufferRegion,
    WorkspaceRequirements,
    WorkspaceViews,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._fingerprint import canonical_json_sha256
from cudnn.moe_ep._validation import (
    validate_native_backward_weights,
    validate_native_forward_weights,
    validate_training_backward_outputs,
    validate_training_forward_outputs,
    validate_training_forward_state,
    validate_training_input,
    validate_training_non_aliasing,
)
from cudnn.moe_ep.api import _resolve_training_device
from moe_ep.moe_ep_test_support import (
    _allocate_stateless_training_outputs,
    _allocate_training_weight_staging,
    _assert_backward_matches,
    _assert_grouped_wgrads_match_reference,
    _assert_matches_reference,
    _assert_wgrads_match_reference,
    _dense_wgrads_from_grouped_kernel,
    _fixed_training_reference,
    _fixed_training_weights,
    _grad_output,
    _interleave_fc1_wgrad,
    _sm107_device,
    _training_abi_prepared,
    _training_config,
    _training_prepared_pair,
    make_forward_inputs,
    quantize_mxfp8,
)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _blocked_scale_elements(rows: int, columns: int) -> int:
    return _round_up(rows, 128) * _round_up(columns, 4)


def _native_forward(
    config,
    *,
    device: torch.device = torch.device("cpu"),
) -> MoeEpNativeForwardWeights:
    e = config.experts_per_rank
    h = config.hidden_size
    i = config.intermediate_size
    fc1 = torch.empty_strided(
        (e, h, 2 * i),
        (h * 2 * i, 1, h),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    fc2 = torch.empty_strided(
        (e, i, h),
        (i * h, 1, i),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    return MoeEpNativeForwardWeights(
        fc1=MoeEpNativeWeight(
            fc1,
            torch.empty(
                (e, _blocked_scale_elements(2 * i, h // 32)),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            MoeEpNativeWeightLayout.FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1,
        ),
        fc2=MoeEpNativeWeight(
            fc2,
            torch.empty(
                (e, _blocked_scale_elements(h, i // 32)),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            MoeEpNativeWeightLayout.FORWARD_FC2_K_MAJOR_V1,
        ),
    )


def _native_backward(
    config,
    *,
    device: torch.device = torch.device("cpu"),
) -> MoeEpNativeBackwardWeights:
    e = config.experts_per_rank
    h = config.hidden_size
    i = config.intermediate_size
    return MoeEpNativeBackwardWeights(
        w2_transpose=MoeEpNativeWeight(
            torch.empty(
                (e, h, i),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty(
                (e, _blocked_scale_elements(i, h // 32)),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            MoeEpNativeWeightLayout.BACKWARD_W2_TRANSPOSE_V1,
        ),
        w1_transpose=MoeEpNativeWeight(
            torch.empty(
                (e, 2 * i, h),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.empty(
                (e, _blocked_scale_elements(h, 2 * i // 32)),
                dtype=torch.float8_e8m0fnu,
                device=device,
            ),
            MoeEpNativeWeightLayout.BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1,
        ),
    )


def _source_weights(config):
    e = config.experts_per_rank
    h = config.hidden_size
    i = config.intermediate_size

    def block_scaled(shape):
        scale_shape = list(shape)
        scale_shape[1] //= 32
        return BlockScaledTensor(
            data=torch.empty(shape, dtype=torch.float8_e4m3fn),
            scale=torch.empty(scale_shape, dtype=torch.float8_e8m0fnu),
            format="mxfp8",
            logical_shape=shape,
            axis=1,
        )

    return (
        MoeEpForwardWeights(
            block_scaled((e, h, 2 * i)),
            block_scaled((e, i, h)),
        ),
        MoeEpBackwardWeights(
            block_scaled((e, h, i)),
            block_scaled((e, 2 * i, h)),
        ),
    )


@pytest.mark.L0
def test_only_stateless_training_types_are_public():
    removed = (
        "MoeEpTrainingResources",
        "MoeEpTrainingSlot",
        "MoeEpTrainingWeights",
    )
    for name in removed:
        assert not hasattr(cudnn, name)
    for name in (
        "MoeEpForwardWeights",
        "MoeEpBackwardWeights",
        "MoeEpNativeWeight",
        "MoeEpTrainingForwardOutputs",
        "MoeEpTrainingBackwardOutputs",
        "pack_forward_weights",
        "pack_backward_weights",
    ):
        assert hasattr(cudnn, name)


@pytest.mark.L0
def test_training_device_prefers_explicit_then_current(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert _resolve_training_device(None) == torch.device("cuda:2")
    assert _resolve_training_device("cuda") == torch.device("cuda:2")
    assert _resolve_training_device(1) == torch.device("cuda:1")
    with pytest.raises(ValueError, match="must be CUDA"):
        _resolve_training_device("cpu")


@pytest.mark.L0
def test_training_input_rejects_noncontiguous_plain_tensor():
    config = _training_config(weight_interleave_size=32)
    activation = torch.empty((config.hidden_size, 2), dtype=torch.bfloat16).t()
    topk_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    topk_weights = torch.ones((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="activation must be contiguous"):
        validate_training_input(
            config,
            "activation",
            activation,
            topk_idx,
            topk_weights,
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_training_bundle_fields_match_public_contracts():
    dummy = object()
    assert [field.name for field in fields(MoeEpForwardWeights)] == ["fc1", "fc2"]
    assert [field.name for field in fields(MoeEpBackwardWeights)] == [
        "w2_transpose",
        "w1_transpose",
    ]
    assert MoeEpForwardWeights(dummy, dummy).fc1 is dummy
    assert MoeEpBackwardWeights(dummy, dummy).w2_transpose is dummy
    assert [field.name for field in fields(MoeEpTrainingForwardOutputs)] == [
        "fc1_preact",
        "output",
        "fc1_a",
        "fc1_sfa",
        "valid_route_counts",
        "expert_offsets",
    ]
    assert [field.name for field in fields(MoeEpTrainingBackwardOutputs)] == [
        "grad_activation",
        "dprob",
        "fc1_b",
        "fc1_sfb",
        "fc2_a",
        "fc2_sfa",
        "fc2_b",
        "fc2_sfb",
    ]


@pytest.mark.L0
def test_native_weight_validation_and_kernel_views_are_zero_copy():
    config = _training_config(weight_interleave_size=32)
    forward = _native_forward(config)
    backward = _native_backward(config)

    assert validate_native_forward_weights(config, forward) == torch.device("cpu")
    assert validate_native_backward_weights(config, backward) == torch.device("cpu")

    forward_kernel = forward_native_to_kernel(forward)
    backward_kernel = backward_native_to_kernel(backward)
    assert forward_kernel.fc1_weight.data_ptr() == forward.fc1.payload.data_ptr()
    assert forward_kernel.fc1_weight_sf.data_ptr() == forward.fc1.scale.data_ptr()
    assert forward_kernel.fc2_weight.data_ptr() == forward.fc2.payload.data_ptr()
    assert backward_kernel.fc1_weight.data_ptr() == backward.w2_transpose.payload.data_ptr()
    assert backward_kernel.fc2_weight_sf.data_ptr() == backward.w1_transpose.scale.data_ptr()


@pytest.mark.L0
def test_standalone_weight_packers_write_only_caller_staging():
    config = _training_config(weight_interleave_size=32)
    source = _source_weights(config)
    forward_out, backward_out = _allocate_training_weight_staging(source)

    native_forward = pack_forward_weights(source[0], out=forward_out)
    native_backward = pack_backward_weights(source[1], out=backward_out)

    assert native_forward.fc1.payload is forward_out.fc1_payload
    assert native_forward.fc2.scale is forward_out.fc2_scale
    assert native_backward.w2_transpose.payload is backward_out.w2_transpose_payload
    assert native_backward.w1_transpose.scale is backward_out.w1_transpose_scale
    validate_native_forward_weights(config, native_forward)
    validate_native_backward_weights(config, native_backward)


@pytest.mark.L0
def test_weight_packing_rejects_source_staging_alias():
    config = _training_config(weight_interleave_size=32)
    source = _source_weights(config)
    _, backward_out = _allocate_training_weight_staging(source)
    aliased_out = MoeEpBackwardWeightStaging(
        w2_transpose_payload=source[1].w2_transpose.data,
        w2_transpose_scale=backward_out.w2_transpose_scale,
        w1_transpose_payload=backward_out.w1_transpose_payload,
        w1_transpose_scale=backward_out.w1_transpose_scale,
    )

    with pytest.raises(ValueError, match="must not alias"):
        pack_backward_weights(source[1], out=aliased_out)


@pytest.mark.L0
def test_mxfp8_training_input_bypasses_quantization_stager():
    class RejectingStager:
        def stage(self, *args, **kwargs):
            raise AssertionError("MXFP8 input must bypass the quantization stager")

    token_count = 2
    hidden = 32
    top_k = 2
    value = BlockScaledTensor(
        data=torch.ones((token_count, hidden), dtype=torch.float8_e4m3fn),
        scale=torch.ones((token_count, hidden // 32), dtype=torch.float8_e8m0fnu),
        format="mxfp8",
        logical_shape=(token_count, hidden),
        axis=1,
    )
    topk_idx = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.75, 0.25], [1.0, 0.0]], dtype=torch.float32)
    activation_data = torch.empty((4, hidden), dtype=torch.float8_e4m3fn)
    activation_sf = torch.empty((4, hidden // 32), dtype=torch.float8_e8m0fnu)
    routing_idx = torch.empty((4, top_k), dtype=torch.int32)
    routing_weights = torch.empty((4, top_k), dtype=torch.float32)

    _stage_input(
        type("Owner", (), {"stager": RejectingStager()})(),
        value,
        topk_idx,
        topk_weights,
        activation_data,
        activation_sf,
        routing_idx,
        routing_weights,
    )

    torch.testing.assert_close(activation_data[:token_count], value.data)
    torch.testing.assert_close(activation_sf[:token_count], value.scale)
    torch.testing.assert_close(routing_idx[:token_count], topk_idx)
    torch.testing.assert_close(routing_weights[:token_count], topk_weights)
    assert activation_data[token_count:].eq(0).all()
    assert activation_sf[token_count:].view(torch.uint8).eq(0).all()
    assert routing_idx[token_count:].eq(-1).all()
    assert routing_weights[token_count:].eq(0).all()


@pytest.mark.L0
def test_native_execution_rejects_compact_or_wrong_layout_scales():
    config = _training_config(weight_interleave_size=32)
    native = _native_forward(config)
    bad = MoeEpNativeForwardWeights(
        fc1=MoeEpNativeWeight(
            native.fc1.payload,
            torch.empty(
                (config.experts_per_rank, config.hidden_size // 32, 2 * config.intermediate_size),
                dtype=torch.float8_e8m0fnu,
            ),
            native.fc1.layout_id,
        ),
        fc2=native.fc2,
    )
    with pytest.raises(ValueError, match=r"weights\.fc1\.scale shape"):
        validate_native_forward_weights(config, bad)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("phase", "missing"),
    (
        ("forward", "fc1_preact"),
        ("forward", "output"),
        ("forward", "fc1_a"),
        ("forward", "fc1_sfa"),
        ("forward", "valid_route_counts"),
        ("forward", "expert_offsets"),
        ("backward", "grad_activation"),
        ("backward", "dprob"),
        ("backward", "fc1_b"),
        ("backward", "fc1_sfb"),
        ("backward", "fc2_a"),
        ("backward", "fc2_sfa"),
        ("backward", "fc2_b"),
        ("backward", "fc2_sfb"),
    ),
)
def test_training_output_requirements_reject_missing_fields(phase, missing):
    requirement = ((1,), (1,), torch.float32, 1)
    if phase == "forward":
        names = ("output", "fc1_preact", "fc1_a", "fc1_sfa", "valid_route_counts", "expert_offsets")
        values = {name: torch.empty(1) for name in names}
        values[missing] = None
        output = MoeEpTrainingForwardOutputs(**values)
        validate = validate_training_forward_outputs
    else:
        names = ("grad_activation", "dprob", "fc1_b", "fc1_sfb", "fc2_a", "fc2_sfa", "fc2_b", "fc2_sfb")
        values = {name: torch.empty(1) for name in names}
        values[missing] = None
        output = MoeEpTrainingBackwardOutputs(**values)
        validate = validate_training_backward_outputs
    requirements = {name: requirement for name in names}
    with pytest.raises(TypeError, match=rf"out\.{missing} must be a torch.Tensor"):
        validate(output, requirements, device=torch.device("cpu"))


@pytest.mark.L0
def test_training_output_types_remain_optional_before_validation():
    with pytest.raises(TypeError, match="fc1_preact"):
        MoeEpTrainingForwardOutputs()
    forward = MoeEpTrainingForwardOutputs(fc1_preact=torch.empty(1))
    backward = MoeEpTrainingBackwardOutputs()
    assert forward.output is None
    assert backward.grad_activation is None


@pytest.mark.L0
def test_training_forward_state_validation_uses_output_contract_names():
    requirement = ((1,), (1,), torch.float32, 1)
    requirements = {
        name: requirement
        for name in (
            "fc1_preact",
            "fc1_a",
            "fc1_sfa",
            "valid_route_counts",
            "expert_offsets",
        )
    }
    with pytest.raises(
        TypeError,
        match=r"out\.fc1_a must be a torch.Tensor",
    ):
        validate_training_forward_state(
            fc1_preact=torch.empty(1),
            fc1_a=None,
            fc1_sfa=torch.empty(1),
            valid_route_counts=torch.empty(1),
            expert_offsets=torch.empty(1),
            requirements=requirements,
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_training_backward_rejects_missing_output_bundle_after_prepare():
    op = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=4,
        max_recv_size_per_rank=4,
        weight_interleave_size=32,
    )
    lane = MoeEpExecutionLane(0, op._operator_token)
    op._training_state = object()
    op._training_requirements = {}
    op._training_lanes = (lane,)
    op._forward_backend_device = torch.device("cpu")
    with pytest.raises(TypeError, match="out must be a MoeEpTrainingBackwardOutputs"):
        op.training_backward(
            lane,
            torch.empty((0, 128), dtype=torch.bfloat16),
            torch.empty((0, 2), dtype=torch.int32),
            torch.empty((0, 2), dtype=torch.float32),
            weights=None,
            fc1_preact=torch.empty((0, 512), dtype=torch.bfloat16),
            out=None,
        )


@pytest.mark.L0
def test_wgrad_assembly_returns_only_caller_owned_views():
    buffers = {
        name: torch.empty(1)
        for name in (
            "fc1_a",
            "fc1_sfa",
            "fc1_b",
            "fc1_sfb",
            "fc2_a",
            "fc2_sfa",
            "fc2_b",
            "fc2_sfb",
            "valid_route_counts",
            "expert_offsets",
        )
    }
    backward = MoeEpTrainingBackwardOutputs(
        fc1_b=buffers["fc1_b"],
        fc1_sfb=buffers["fc1_sfb"],
        fc2_a=buffers["fc2_a"],
        fc2_sfa=buffers["fc2_sfa"],
        fc2_b=buffers["fc2_b"],
        fc2_sfb=buffers["fc2_sfb"],
    )
    operands = assemble_training_wgrad_operands(
        fc1_a=buffers["fc1_a"],
        fc1_sfa=buffers["fc1_sfa"],
        valid_route_counts=buffers["valid_route_counts"],
        expert_offsets=buffers["expert_offsets"],
        backward=backward,
    )
    assert isinstance(operands, MoeEpTrainingWgradOperands)
    for name in buffers:
        assert getattr(operands, name).data_ptr() == buffers[name].data_ptr()


@pytest.mark.L0
def test_private_training_state_has_no_bound_weights_or_wgrad_exporter():
    config = _training_config(weight_interleave_size=32)
    forward, backward = _training_prepared_pair(config)
    state = Mxfp8TrainingState(
        config,
        torch.device("cpu"),
        forward,
        backward,
        lane_count=2,
    )
    assert not hasattr(state, "weight_bindings")
    assert not hasattr(state, "wgrad_exporter")
    assert not hasattr(state, "slot_count")
    assert state.lane_count == 2
    assert all(
        "fc1_preact" not in region.name
        for region in (
            *state.requirements.symmetric_regions,
            *state.requirements.local_regions,
        )
    )

    requirements = state.public_requirements()
    assert requirements["fc1_a"] == (
        (config.hidden_size, forward.pool_token_capacity),
        (forward.pool_token_capacity, 1),
        torch.float8_e4m3fn,
        128,
    )
    assert requirements["fc1_b"][1] == (2 * config.intermediate_size, 1)
    assert requirements["fc2_a"][1] == (1, config.intermediate_size)
    assert requirements["fc2_b"][1] == (1, forward.pool_token_capacity)


@pytest.mark.L0
def test_private_training_workspace_keeps_only_live_lane_scratch():
    config = _training_config(weight_interleave_size=32)
    forward, backward = _training_prepared_pair(config)
    state = Mxfp8TrainingState(
        config,
        torch.device("cpu"),
        forward,
        backward,
        lane_count=2,
    )
    flat = WorkspaceViews(
        token_count=0,
        symmetric={region.name: torch.empty(region.nbytes, dtype=torch.uint8) for region in state.requirements.symmetric_regions},
        local={region.name: torch.empty(region.nbytes, dtype=torch.uint8) for region in state.requirements.local_regions},
        peer_mapping=object(),
    )

    first = state._lane_scratch_views(flat, 0)
    second = state._lane_scratch_views(flat, 1)
    forward_workspace = state._phase_workspace(
        flat,
        forward.workspace_requirements,
        lane=0,
        phase="forward",
    )
    backward_workspace = state._phase_workspace(
        flat,
        backward.workspace_requirements,
        lane=0,
        phase="backward",
    )
    assert "col_quant_data" not in forward_workspace.local
    assert "col_quant_sf" not in forward_workspace.local
    assert "kernel_local_workspace" in forward_workspace.local
    assert "backward_aux_data" in backward_workspace.local
    assert "backward_aux_scale" in backward_workspace.local
    for field in fields(first):
        first_value = getattr(first, field.name)
        second_value = getattr(second, field.name)
        if isinstance(first_value, torch.Tensor):
            assert first_value.data_ptr() != second_value.data_ptr()
    names = {region.name for region in (*state.requirements.symmetric_regions, *state.requirements.local_regions)}
    removed = (
        "valid_route_counts",
        "expert_offsets",
        "fc1_recompute",
        "fc1_recompute_sf",
        "fc1_col_output",
        "fc1_col_output_sf",
        "grad_y2",
        "grad_y2_sf",
        "col_quant_data",
        "col_quant_sf",
    )
    assert not any(any(name.endswith(removed_name) for removed_name in removed) for name in names)
    for lane in range(2):
        assert f"lane.{lane}.fallback.local.routing_topk_idx" in names
        assert f"lane.{lane}.fallback.symmetric.routing_topk_weights" in names
        assert f"lane.{lane}.backward.local.backward_aux_data" in names
        assert f"lane.{lane}.backward.local.backward_aux_scale" in names
        assert f"lane.{lane}.forward.symmetric.output_data" in names
        assert f"lane.{lane}.backward.symmetric.output_data" in names
        assert f"lane.{lane}.backward.symmetric.backward_dprob" in names


@pytest.mark.L0
def test_training_views_require_col_quant_snapshot():
    config = _training_config(weight_interleave_size=32)
    forward, backward = _training_prepared_pair(config)
    forward.col_quant_sizes_offset = None
    state = Mxfp8TrainingState(
        config,
        torch.device("cpu"),
        forward,
        backward,
        lane_count=1,
    )
    with pytest.raises(RuntimeError, match="persistent col-quant expert-size snapshot"):
        state.views(lane=0, token_count=0)


@pytest.mark.L0
def test_training_abi_fingerprint_covers_lanes_and_native_layouts():
    config = _training_config(
        ep_size=2,
        ep_global_ranks=(0, 1),
        weight_interleave_size=32,
    )
    forward = _training_abi_prepared("forward")
    backward = _training_abi_prepared("backward")
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=4,
        symmetric_regions=(BufferRegion("symmetric", 256),),
        local_regions=(BufferRegion("local", 128),),
    )
    first = _build_training_abi_facts(
        config,
        forward,
        backward,
        requirements,
        lane_count=1,
        source_tree_digest="source",
    )
    repeated = _build_training_abi_facts(
        config,
        forward,
        backward,
        requirements,
        lane_count=1,
        source_tree_digest="source",
    )
    changed_lanes = _build_training_abi_facts(
        config,
        forward,
        backward,
        requirements,
        lane_count=2,
        source_tree_digest="source",
    )

    assert first["schema_version"] == 2
    assert first["native_weight_layouts"] == [layout.value for layout in MoeEpNativeWeightLayout]
    assert canonical_json_sha256(first) == canonical_json_sha256(repeated)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed_lanes)


@pytest.mark.L0
def test_training_workspace_harmonizes_symmetric_regions(monkeypatch):
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=1,
        symmetric_regions=(
            BufferRegion("first", 1),
            BufferRegion("second", 257),
        ),
        local_regions=(BufferRegion("local", 1),),
    )
    runtime = type("Runtime", (), {"world_size": 2, "group": object()})()

    def all_reduce(tensor, *, op, group):
        assert group is runtime.group
        if tensor.numel() == 2 and op == torch.distributed.ReduceOp.MAX:
            tensor.copy_(torch.tensor([257, 257], dtype=torch.int64))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    harmonized = _harmonize_symmetric_regions(
        requirements,
        runtime,
        torch.device("cpu"),
    )

    assert tuple(region.nbytes for region in harmonized.symmetric_regions) == (
        257,
        257,
    )
    assert harmonized.local_regions == requirements.local_regions


@pytest.mark.L0
def test_training_contract_rejects_cross_bundle_aliases():
    storage = torch.empty(16)
    with pytest.raises(ValueError, match="out must not alias saved"):
        validate_training_non_aliasing(
            {
                "saved": storage[:8],
                "out": storage[4:12],
            }
        )


@pytest.mark.L0
def test_training_methods_require_prepare_and_do_not_expose_cleanup():
    op = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=4,
        max_recv_size_per_rank=4,
        weight_interleave_size=32,
    )
    assert hasattr(op, "prepare_training")
    assert hasattr(op, "training_forward")
    assert hasattr(op, "training_backward")
    assert not hasattr(op, "prepare_training_resources")
    assert not hasattr(op, "refresh_weights")
    assert not hasattr(op, "finalize_overflow")
    with pytest.raises(RuntimeError, match="prepare_training"):
        op.training_forward(
            object(),
            torch.empty((0, 128), dtype=torch.bfloat16),
            torch.empty((0, 2), dtype=torch.int32),
            torch.empty((0, 2), dtype=torch.float32),
            weights=_native_forward(_training_config(weight_interleave_size=32)),
            out=MoeEpTrainingForwardOutputs(
                fc1_preact=torch.empty((0, 512), dtype=torch.bfloat16),
            ),
        )

    conventional = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=4,
        max_recv_size_per_rank=4,
    )
    with pytest.raises(ValueError, match="weight_interleave_size=32"):
        conventional.prepare_training()


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_stateless_training_ep1_eager_and_cuda_graph_match_reference():
    device = _sm107_device()
    base_args = make_forward_inputs(device)
    args = (
        base_args[0].dequantize(torch.bfloat16),
        base_args[1],
        base_args[2],
        base_args[3],
        base_args[4].float().contiguous(),
    )
    grad_output = _grad_output(device, args[0].shape[0], seed=20260902)
    expected = _fixed_training_reference(
        args,
        grad_output,
        combine_format="bf16",
        gate_up_clamp=None,
    )
    source_weights = _fixed_training_weights(args)

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=args[0].shape[0],
        max_recv_size_per_rank=args[0].shape[0] * args[3].shape[1],
        drop_on_overflow=True,
        combine_format="bf16",
        weight_interleave_size=32,
    ) as op:
        requirements = op.prepare_training(lane_count=1, device=device)
        forward_staging, backward_staging = _allocate_training_weight_staging(source_weights)
        native_forward = op.pack_forward_weights(
            source_weights[0],
            out=forward_staging,
        )
        native_backward = op.pack_backward_weights(
            source_weights[1],
            out=backward_staging,
        )
        forward_out, backward_out = _allocate_stateless_training_outputs(
            requirements,
            device,
        )
        lane = op.training_lanes[0]

        def run():
            y = op.training_forward(
                lane,
                args[0],
                args[3],
                args[4],
                weights=native_forward,
                out=forward_out,
            )
            dx, dprob, operands = op.training_backward(
                lane,
                grad_output,
                args[3],
                args[4],
                weights=native_backward,
                fc1_preact=forward_out.fc1_preact,
                fc1_a=forward_out.fc1_a,
                fc1_sfa=forward_out.fc1_sfa,
                valid_route_counts=forward_out.valid_route_counts,
                expert_offsets=forward_out.expert_offsets,
                out=backward_out,
            )
            assert operands is not None
            return y, dx, dprob, operands

        def assert_matches(actual):
            y, dx, dprob, operands = actual
            _assert_matches_reference(y, expected[0])
            _assert_backward_matches(
                (dx, dprob),
                (expected[1], expected[2]),
                args[3],
            )
            _assert_wgrads_match_reference(
                operands,
                expected[3],
                weight_interleave_size=32,
            )

        eager = run()
        grouped_wgrads = _dense_wgrads_from_grouped_kernel(eager[3])
        torch.cuda.synchronize(device)
        assert_matches(eager)
        expected_fc1_wgrad, expected_fc2_wgrad = expected[3].dense_wgrads()
        _assert_grouped_wgrads_match_reference(
            grouped_wgrads,
            (
                _interleave_fc1_wgrad(expected_fc1_wgrad),
                expected_fc2_wgrad,
            ),
            reference_name="the independent PyTorch MXFP8 reference",
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        pointers = tuple(tensor.data_ptr() for bundle in (forward_out, backward_out) for tensor in vars(bundle).values() if tensor is not None)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize(device)
            assert pointers == tuple(tensor.data_ptr() for bundle in (forward_out, backward_out) for tensor in vars(bundle).values() if tensor is not None)
            assert_matches(captured)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_native_io_mxfp8_cuda_graph_replay():
    """Exercise native MXFP8 I/O and weight contracts with graph replay."""

    device = _sm107_device()
    base_args = make_forward_inputs(device)
    activation = base_args[0]
    topk_idx = base_args[3]
    topk_weights = base_args[4].float().contiguous()
    args = (
        activation,
        base_args[1],
        base_args[2],
        topk_idx,
        topk_weights,
    )
    grad_output_plain = _grad_output(
        device,
        activation.shape[0],
        seed=20260903,
    )
    grad_output = quantize_mxfp8(grad_output_plain, axis=1)
    expected = _fixed_training_reference(
        args,
        grad_output.dequantize(torch.float32),
        combine_format="bf16",
        gate_up_clamp=None,
    )

    op = MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=activation.shape[0],
        max_recv_size_per_rank=activation.shape[0] * topk_idx.shape[1],
        drop_on_overflow=True,
        output_format="bf16",
        combine_format="bf16",
        weight_interleave_size=32,
    )
    try:
        requirements = op.prepare_training(lane_count=1, device=device)
        forward_out, backward_out = _allocate_stateless_training_outputs(
            requirements,
            device,
        )
        lane = op.training_lanes[0]

        # The production call below receives only native packs. The existing
        # fallback packer is used once here as a test oracle to create known-good
        # native contents, then copied into independent caller-owned tensors.
        source_weights = _fixed_training_weights(args)
        forward_staging, backward_staging = _allocate_training_weight_staging(
            source_weights
        )
        packed_forward = op.pack_forward_weights(
            source_weights[0],
            out=forward_staging,
        )
        packed_backward = op.pack_backward_weights(
            source_weights[1],
            out=backward_staging,
        )

        def clone_native_tensor(tensor):
            return torch.empty_strided(
                tensor.shape,
                tensor.stride(),
                dtype=tensor.dtype,
                device=tensor.device,
            ).copy_(tensor)

        native_forward = MoeEpNativeForwardWeights(
            fc1=MoeEpNativeWeight(
                clone_native_tensor(packed_forward.fc1.payload),
                clone_native_tensor(packed_forward.fc1.scale),
                MoeEpNativeWeightLayout.FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1,
            ),
            fc2=MoeEpNativeWeight(
                clone_native_tensor(packed_forward.fc2.payload),
                clone_native_tensor(packed_forward.fc2.scale),
                MoeEpNativeWeightLayout.FORWARD_FC2_K_MAJOR_V1,
            ),
        )
        native_backward = MoeEpNativeBackwardWeights(
            w2_transpose=MoeEpNativeWeight(
                clone_native_tensor(packed_backward.w2_transpose.payload),
                clone_native_tensor(packed_backward.w2_transpose.scale),
                MoeEpNativeWeightLayout.BACKWARD_W2_TRANSPOSE_V1,
            ),
            w1_transpose=MoeEpNativeWeight(
                clone_native_tensor(packed_backward.w1_transpose.payload),
                clone_native_tensor(packed_backward.w1_transpose.scale),
                MoeEpNativeWeightLayout.BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1,
            ),
        )
        assert native_forward.fc1.payload.data_ptr() != packed_forward.fc1.payload.data_ptr()
        assert native_backward.w1_transpose.scale.data_ptr() != packed_backward.w1_transpose.scale.data_ptr()

        def run():
            output = op.training_forward(
                lane,
                activation,
                topk_idx,
                topk_weights,
                weights=native_forward,
                out=forward_out,
            )
            grad_activation, dprob, operands = op.training_backward(
                lane,
                grad_output,
                topk_idx,
                topk_weights,
                weights=native_backward,
                fc1_preact=forward_out.fc1_preact,
                fc1_a=forward_out.fc1_a,
                fc1_sfa=forward_out.fc1_sfa,
                valid_route_counts=forward_out.valid_route_counts,
                expert_offsets=forward_out.expert_offsets,
                out=backward_out,
            )
            return output, grad_activation, dprob, operands

        def assert_matches(result):
            output, grad_activation, dprob, operands = result
            assert output.data_ptr() == forward_out.output.data_ptr()
            assert grad_activation.data_ptr() == backward_out.grad_activation.data_ptr()
            assert dprob.data_ptr() == backward_out.dprob.data_ptr()
            _assert_matches_reference(output, expected[0])
            _assert_backward_matches(
                (grad_activation, dprob),
                (expected[1], expected[2]),
                topk_idx,
            )
            _assert_wgrads_match_reference(
                operands,
                expected[3],
                weight_interleave_size=32,
            )

        eager = run()
        grouped_wgrads = _dense_wgrads_from_grouped_kernel(eager[3])
        torch.cuda.synchronize(device)
        assert_matches(eager)
        expected_fc1_wgrad, expected_fc2_wgrad = expected[3].dense_wgrads()
        _assert_grouped_wgrads_match_reference(
            grouped_wgrads,
            (
                _interleave_fc1_wgrad(expected_fc1_wgrad),
                expected_fc2_wgrad,
            ),
            reference_name="the independent PyTorch MXFP8 reference",
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        output_pointers = (
            forward_out.output.data_ptr(),
            backward_out.grad_activation.data_ptr(),
            backward_out.dprob.data_ptr(),
        )
        native_weight_pointers = (
            native_forward.fc1.payload.data_ptr(),
            native_forward.fc1.scale.data_ptr(),
            native_forward.fc2.payload.data_ptr(),
            native_forward.fc2.scale.data_ptr(),
            native_backward.w2_transpose.payload.data_ptr(),
            native_backward.w2_transpose.scale.data_ptr(),
            native_backward.w1_transpose.payload.data_ptr(),
            native_backward.w1_transpose.scale.data_ptr(),
        )
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize(device)
            assert output_pointers == (
                forward_out.output.data_ptr(),
                backward_out.grad_activation.data_ptr(),
                backward_out.dprob.data_ptr(),
            )
            assert native_weight_pointers == (
                native_forward.fc1.payload.data_ptr(),
                native_forward.fc1.scale.data_ptr(),
                native_forward.fc2.payload.data_ptr(),
                native_forward.fc2.scale.data_ptr(),
                native_backward.w2_transpose.payload.data_ptr(),
                native_backward.w2_transpose.scale.data_ptr(),
                native_backward.w1_transpose.payload.data_ptr(),
                native_backward.w1_transpose.scale.data_ptr(),
            )
            assert_matches(captured)
    finally:
        op.close()
