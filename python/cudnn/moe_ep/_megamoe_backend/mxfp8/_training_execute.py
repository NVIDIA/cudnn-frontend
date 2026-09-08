# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Ordinary/capturable stateless launch path over private lane resources."""

from __future__ import annotations

import torch

from ..._types import (
    BlockScaledTensor,
    MoeEpNativeBackwardWeights,
    MoeEpNativeForwardWeights,
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingForwardOutputs,
    MoeEpTrainingWgradOperands,
    MoeTensor,
)
from .._runtime import _runtime_debug
from .._workspace import padded_mxfp8_scale_columns
from ._adapter import (
    Mxfp8LaunchInputs,
    _typed_view,
)
from ._backward_compile import (
    Mxfp8BackwardLaunchInputs,
    build_backward_runtime_kwargs,
    compile_backward_or_get,
)
from ._compile import compile_or_get
from ._launch import build_runtime_kwargs
from ._training_resources import (
    Mxfp8TrainingExecutionViews,
    Mxfp8TrainingState,
)
from ._training_weights import (
    backward_native_to_kernel,
    forward_native_to_kernel,
)
from ._training_wgrad import assemble_training_wgrad_operands


def _zero_pre_reduced(inputs, prepared) -> None:
    capacity = prepared.config.max_tokens_per_rank
    offset = prepared.pre_reduced_activation_offset
    bytes_per_token = prepared.pre_reduced_activation_bytes_per_token
    if offset is not None and bytes_per_token:
        inputs.shared_workspace.narrow(
            0,
            offset,
            capacity * bytes_per_token,
        ).zero_()
    sf_offset = prepared.pre_reduced_activation_sf_offset
    sf_bytes_per_token = prepared.pre_reduced_activation_sf_bytes_per_token
    if sf_offset is not None and sf_bytes_per_token:
        inputs.shared_workspace.narrow(
            0,
            sf_offset,
            capacity * sf_bytes_per_token,
        ).zero_()


def _activation_views(
    execution: Mxfp8TrainingExecutionViews,
    *,
    backward: bool,
    capacity: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace = execution.backward.workspace if backward else execution.forward.workspace
    return (
        _typed_view(
            workspace.symmetric["activation_data"],
            torch.float8_e4m3fn,
            (capacity, hidden),
        ),
        _typed_view(
            workspace.symmetric["activation_scale"],
            torch.float8_e8m0fnu,
            (capacity, padded_mxfp8_scale_columns(hidden)),
        ),
    )


def _stage_input(
    state: Mxfp8TrainingState,
    value: MoeTensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    activation_data: torch.Tensor,
    activation_sf: torch.Tensor,
    routing_topk_idx: torch.Tensor,
    routing_topk_weights: torch.Tensor,
) -> None:
    """Stage into private symmetric memory, bypassing quantization for MXFP8."""

    if not isinstance(value, BlockScaledTensor):
        state.stager.stage(
            value,
            topk_idx,
            topk_weights,
            activation_data,
            activation_sf,
            routing_topk_idx,
            routing_topk_weights,
        )
        return

    token_count = int(value.logical_shape[0])
    scale_columns = int(value.scale.shape[1])
    activation_data.zero_()
    activation_sf.zero_()
    routing_topk_idx.fill_(-1)
    routing_topk_weights.zero_()
    if token_count == 0:
        return
    activation_data[:token_count].copy_(value.data)
    activation_sf[:token_count, :scale_columns].copy_(value.scale)
    routing_topk_idx[:token_count].copy_(topk_idx)
    routing_topk_weights[:token_count].copy_(topk_weights)


def _write_expert_offsets(
    execution: Mxfp8TrainingExecutionViews,
    padding: int,
    counts: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    snapshot = execution.forward_expert_size_snapshot
    counts.copy_(snapshot)
    torch.add(counts, padding - 1, out=offsets)
    torch.div(offsets, padding, rounding_mode="floor", out=offsets)
    offsets.mul_(padding)
    torch.cumsum(offsets, dim=0, out=offsets)


def launch_training_forward(
    state: Mxfp8TrainingState,
    execution: Mxfp8TrainingExecutionViews,
    activation: MoeTensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    weights: MoeEpNativeForwardWeights,
    out: MoeEpTrainingForwardOutputs,
) -> torch.Tensor:
    """Launch one stateless forward over caller-owned outputs."""

    prepared = state.forward_prepared
    config = prepared.config
    capacity = config.max_tokens_per_rank
    scratch = execution.scratch
    token_count = int(activation.logical_shape[0] if isinstance(activation, BlockScaledTensor) else activation.shape[0])
    _runtime_debug(
        "training-forward.begin",
        lane=scratch.index,
        token_count=token_count,
    )
    activation_data, activation_sf = _activation_views(
        execution,
        backward=False,
        capacity=capacity,
        hidden=config.hidden,
    )
    _runtime_debug("training-forward.stage.begin", lane=scratch.index)
    _stage_input(
        state,
        activation,
        topk_idx,
        topk_weights,
        activation_data,
        activation_sf,
        scratch.routing_topk_idx,
        scratch.routing_topk_weights,
    )
    _runtime_debug("training-forward.stage.end", lane=scratch.index)

    assert out.output is not None
    assert out.fc1_a is not None
    assert out.fc1_sfa is not None
    assert out.valid_route_counts is not None
    assert out.expert_offsets is not None
    fc1_preact = out.fc1_preact
    col_quant_data = out.fc1_a.transpose(0, 1)
    col_quant_sf = out.fc1_sfa.view(torch.uint8).reshape(-1)
    expected_elements = int(prepared.col_quant_sf_elements)
    if col_quant_sf.numel() != expected_elements:
        raise ValueError("out.fc1_sfa storage does not match the forward producer ABI: " f"{col_quant_sf.numel()} != {expected_elements}")
    valid_route_counts = out.valid_route_counts
    expert_offsets = out.expert_offsets

    scratch.forward_output.zero_()
    scratch.forward_overflow.zero_()
    col_quant_data.zero_()
    # E8M0 byte 127 encodes scale 1.0. The producer only overwrites active
    # expert segments, so the unused grouped-WGrad capacity must stay neutral.
    col_quant_sf.fill_(127)
    _runtime_debug("training-forward.reset.end", lane=scratch.index)

    workspace = execution.forward.workspace
    inputs = Mxfp8LaunchInputs(
        activation=activation_data,
        activation_sf=activation_sf,
        topk_indices=scratch.routing_topk_idx,
        topk_scores=scratch.routing_topk_weights,
        weights=forward_native_to_kernel(weights),
        fc1_c=fc1_preact,
        output_data=scratch.forward_output,
        col_quant_data=col_quant_data,
        col_quant_sf=col_quant_sf,
        overflow_flag=scratch.forward_overflow,
        local_workspace=workspace.local["kernel_local_workspace"],
        shared_workspace=workspace.symmetric["kernel_shared_workspace"],
        token_count=token_count,
    )
    _zero_pre_reduced(inputs, prepared)
    _runtime_debug("training-forward.compile.begin", lane=scratch.index)
    compiled = compile_or_get(
        prepared,
        inputs,
        execution.forward,
    )
    _runtime_debug("training-forward.compile.end", lane=scratch.index)
    _runtime_debug("training-forward.launch.begin", lane=scratch.index)
    compiled.callable(**build_runtime_kwargs(inputs, execution.forward))
    _runtime_debug("training-forward.launch.end", lane=scratch.index)
    _runtime_debug("training-forward.offsets.begin", lane=scratch.index)
    _write_expert_offsets(
        execution,
        config.token_padding_block,
        valid_route_counts,
        expert_offsets,
    )
    _runtime_debug("training-forward.offsets.end", lane=scratch.index)
    state.apply_overflow(lane=scratch.index, phase="forward")

    output = out.output[:token_count]
    output.copy_(scratch.forward_output[:token_count])
    _runtime_debug("training-forward.end", lane=scratch.index)
    return output


def launch_training_backward(
    state: Mxfp8TrainingState,
    execution: Mxfp8TrainingExecutionViews,
    grad_output: MoeTensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    weights: MoeEpNativeBackwardWeights,
    fc1_preact: torch.Tensor,
    fc1_a: torch.Tensor | None,
    fc1_sfa: torch.Tensor | None,
    valid_route_counts: torch.Tensor | None,
    expert_offsets: torch.Tensor | None,
    out: MoeEpTrainingBackwardOutputs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    MoeEpTrainingWgradOperands,
]:
    """Launch one stateless backward using explicit caller-owned saved state."""

    prepared = state.backward_prepared
    config = prepared.config
    capacity = config.max_tokens_per_rank
    scratch = execution.scratch
    token_count = int(grad_output.logical_shape[0] if isinstance(grad_output, BlockScaledTensor) else grad_output.shape[0])
    _runtime_debug(
        "training-backward.begin",
        lane=scratch.index,
        token_count=token_count,
    )
    activation_data, activation_sf = _activation_views(
        execution,
        backward=True,
        capacity=capacity,
        hidden=config.hidden,
    )
    _runtime_debug("training-backward.stage.begin", lane=scratch.index)
    _stage_input(
        state,
        grad_output,
        topk_idx,
        topk_weights,
        activation_data,
        activation_sf,
        scratch.routing_topk_idx,
        scratch.routing_topk_weights,
    )
    _runtime_debug("training-backward.stage.end", lane=scratch.index)

    assert fc1_a is not None
    assert fc1_sfa is not None
    assert valid_route_counts is not None
    assert expert_offsets is not None
    assert out.grad_activation is not None
    assert out.dprob is not None
    assert out.fc1_b is not None
    assert out.fc1_sfb is not None
    assert out.fc2_a is not None
    assert out.fc2_sfa is not None
    assert out.fc2_b is not None
    assert out.fc2_sfb is not None
    fc1_recompute = out.fc2_a.transpose(0, 1)
    fc1_recompute_sf = out.fc2_sfa
    fc1_col_output = out.fc1_b
    fc1_col_output_sf = out.fc1_sfb
    grad_y2 = out.fc2_b
    grad_y2_sf = out.fc2_sfb.view(torch.uint8).reshape(-1)

    scratch.backward_output.zero_()
    scratch.backward_overflow.zero_()
    scratch.dprob.zero_()
    fc1_recompute.zero_()
    fc1_recompute_sf.view(torch.uint8).fill_(127)
    fc1_col_output.zero_()
    fc1_col_output_sf.view(torch.uint8).fill_(127)
    grad_y2.zero_()
    grad_y2_sf.fill_(127)
    _runtime_debug("training-backward.reset.end", lane=scratch.index)

    workspace = execution.backward.workspace
    kernel_weights = backward_native_to_kernel(weights)
    inputs = Mxfp8BackwardLaunchInputs(
        grad_out=activation_data,
        grad_out_sf=activation_sf,
        topk_idx=scratch.routing_topk_idx,
        topk_weights=scratch.routing_topk_weights,
        fc1_weight=kernel_weights.fc1_weight,
        fc1_weight_sf=kernel_weights.fc1_weight_sf,
        fc2_weight=kernel_weights.fc2_weight,
        fc2_weight_sf=kernel_weights.fc2_weight_sf,
        beta=state.beta,
        fc1_preact=fc1_preact,
        output_activation=scratch.backward_output,
        overflow_flag=scratch.backward_overflow,
        dprob=scratch.dprob,
        fc1_recompute=fc1_recompute,
        fc1_recompute_sf=fc1_recompute_sf,
        fc1_col_output=fc1_col_output,
        fc1_col_output_sf=fc1_col_output_sf,
        grad_y2=grad_y2,
        grad_y2_sf=grad_y2_sf,
        local_workspace=workspace.local["kernel_local_workspace"],
        shared_workspace=workspace.symmetric["kernel_shared_workspace"],
        token_count=token_count,
    )
    _zero_pre_reduced(inputs, prepared)
    _runtime_debug("training-backward.compile.begin", lane=scratch.index)
    compiled = compile_backward_or_get(
        prepared,
        inputs,
        execution.backward,
    )
    _runtime_debug("training-backward.compile.end", lane=scratch.index)
    _runtime_debug("training-backward.launch.begin", lane=scratch.index)
    compiled.callable(**build_backward_runtime_kwargs(inputs, execution.backward))
    _runtime_debug("training-backward.launch.end", lane=scratch.index)
    state.apply_overflow(lane=scratch.index, phase="backward")

    grad_activation = out.grad_activation[:token_count]
    grad_activation.copy_(scratch.backward_output[:token_count])

    dprob = out.dprob[:token_count]
    dprob.copy_(scratch.dprob[:token_count])

    operands = assemble_training_wgrad_operands(
        fc1_a=fc1_a,
        fc1_sfa=fc1_sfa,
        expert_offsets=expert_offsets,
        valid_route_counts=valid_route_counts,
        backward=out,
    )
    _runtime_debug("training-backward.end", lane=scratch.index)
    return grad_activation, dprob, operands


__all__ = [
    "launch_training_backward",
    "launch_training_forward",
]
