# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Ordinary/capturable launch path over fixed MXFP8 training resources."""

from __future__ import annotations

import torch

from ..._types import MoeEpTrainingWgradOperands
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
    Mxfp8TrainingResourceOwner,
)


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


def _write_expert_offsets(
    execution: Mxfp8TrainingExecutionViews,
    padding: int,
) -> None:
    snapshot = execution.forward_expert_size_snapshot
    if snapshot is None:
        raise RuntimeError("training forward requires the persistent expert-size snapshot")
    counts = execution.slot.valid_route_counts
    offsets = execution.slot.expert_offsets
    counts.copy_(snapshot)
    torch.add(counts, padding - 1, out=offsets)
    torch.div(offsets, padding, rounding_mode="floor", out=offsets)
    offsets.mul_(padding)
    torch.cumsum(offsets, dim=0, out=offsets)


def launch_training_forward(
    owner: Mxfp8TrainingResourceOwner,
    execution: Mxfp8TrainingExecutionViews,
    activation: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Stage and launch one fixed-slot forward without host-visible routing."""

    prepared = owner.forward_prepared
    config = prepared.config
    capacity = config.max_tokens_per_rank
    slot = execution.slot
    _runtime_debug(
        "training-forward.begin",
        slot=execution.slot.index,
        token_count=int(activation.shape[0]),
    )
    activation_data, activation_sf = _activation_views(
        execution,
        backward=False,
        capacity=capacity,
        hidden=config.hidden,
    )
    _runtime_debug("training-forward.stage.begin", slot=execution.slot.index)
    owner.stager.stage(
        activation,
        topk_idx,
        topk_weights,
        activation_data,
        activation_sf,
        slot.routing_topk_idx,
        slot.routing_topk_weights,
    )
    _runtime_debug("training-forward.stage.end", slot=execution.slot.index)
    slot.forward_output.zero_()
    slot.forward_overflow.zero_()
    if slot.col_quant_data is not None:
        slot.col_quant_data.zero_()
    if slot.col_quant_sf is not None:
        slot.col_quant_sf.zero_()
    _runtime_debug("training-forward.reset.end", slot=execution.slot.index)

    workspace = execution.forward.workspace
    inputs = Mxfp8LaunchInputs(
        activation=activation_data,
        activation_sf=activation_sf,
        topk_indices=slot.routing_topk_idx,
        topk_scores=slot.routing_topk_weights,
        weights=owner.weight_bindings.forward,
        fc1_c=slot.fc1_preact,
        output_data=slot.forward_output,
        col_quant_data=slot.col_quant_data,
        col_quant_sf=slot.col_quant_sf,
        overflow_flag=slot.forward_overflow,
        local_workspace=workspace.local["kernel_local_workspace"],
        shared_workspace=workspace.symmetric["kernel_shared_workspace"],
        token_count=int(activation.shape[0]),
    )
    _zero_pre_reduced(inputs, prepared)
    _runtime_debug("training-forward.compile.begin", slot=execution.slot.index)
    compiled = compile_or_get(
        prepared,
        inputs,
        execution.forward,
    )
    _runtime_debug("training-forward.compile.end", slot=execution.slot.index)
    _runtime_debug("training-forward.launch.begin", slot=execution.slot.index)
    compiled.callable(**build_runtime_kwargs(inputs, execution.forward))
    _runtime_debug("training-forward.launch.end", slot=execution.slot.index)
    _runtime_debug("training-forward.offsets.begin", slot=execution.slot.index)
    _write_expert_offsets(execution, config.token_padding_block)
    _runtime_debug("training-forward.offsets.end", slot=execution.slot.index)
    _runtime_debug("training-forward.end", slot=execution.slot.index)
    return slot.forward_output[: inputs.token_count]


def launch_training_backward(
    owner: Mxfp8TrainingResourceOwner,
    execution: Mxfp8TrainingExecutionViews,
    grad_output: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    MoeEpTrainingWgradOperands,
]:
    """Stage and launch one fixed-slot backward using forward's raw pool."""

    prepared = owner.backward_prepared
    config = prepared.config
    capacity = config.max_tokens_per_rank
    slot = execution.slot
    token_count = int(grad_output.shape[0])
    _runtime_debug(
        "training-backward.begin",
        slot=execution.slot.index,
        token_count=token_count,
    )
    activation_data, activation_sf = _activation_views(
        execution,
        backward=True,
        capacity=capacity,
        hidden=config.hidden,
    )
    _runtime_debug("training-backward.stage.begin", slot=execution.slot.index)
    owner.stager.stage(
        grad_output,
        slot.routing_topk_idx[:token_count],
        slot.routing_topk_weights[:token_count],
        activation_data,
        activation_sf,
        slot.routing_topk_idx,
        slot.routing_topk_weights,
    )
    _runtime_debug("training-backward.stage.end", slot=execution.slot.index)

    slot.backward_output.zero_()
    slot.grad_activation.zero_()
    slot.backward_overflow.zero_()
    slot.dprob.zero_()
    slot.fc1_recompute.zero_()
    slot.fc1_recompute_sf.view(torch.uint8).fill_(127)
    slot.fc1_col_output.zero_()
    slot.fc1_col_output_sf.view(torch.uint8).fill_(127)
    slot.grad_y2.zero_()
    slot.grad_y2_sf.fill_(127)
    _runtime_debug("training-backward.reset.end", slot=execution.slot.index)

    workspace = execution.backward.workspace
    weights = owner.weight_bindings.backward
    inputs = Mxfp8BackwardLaunchInputs(
        grad_out=activation_data,
        grad_out_sf=activation_sf,
        topk_idx=slot.routing_topk_idx,
        topk_weights=slot.routing_topk_weights,
        fc1_weight=weights.fc1_weight,
        fc1_weight_sf=weights.fc1_weight_sf,
        fc2_weight=weights.fc2_weight,
        fc2_weight_sf=weights.fc2_weight_sf,
        beta=owner.beta,
        fc1_preact=slot.fc1_preact,
        output_activation=slot.backward_output,
        overflow_flag=slot.backward_overflow,
        dprob=slot.dprob,
        fc1_recompute=slot.fc1_recompute,
        fc1_recompute_sf=slot.fc1_recompute_sf,
        fc1_col_output=slot.fc1_col_output,
        fc1_col_output_sf=slot.fc1_col_output_sf,
        grad_y2=slot.grad_y2,
        grad_y2_sf=slot.grad_y2_sf,
        local_workspace=workspace.local["kernel_local_workspace"],
        shared_workspace=workspace.symmetric["kernel_shared_workspace"],
        token_count=token_count,
    )
    _zero_pre_reduced(inputs, prepared)
    _runtime_debug("training-backward.compile.begin", slot=execution.slot.index)
    compiled = compile_backward_or_get(
        prepared,
        inputs,
        execution.backward,
    )
    _runtime_debug("training-backward.compile.end", slot=execution.slot.index)
    _runtime_debug("training-backward.launch.begin", slot=execution.slot.index)
    compiled.callable(**build_backward_runtime_kwargs(inputs, execution.backward))
    _runtime_debug("training-backward.launch.end", slot=execution.slot.index)
    slot.grad_activation.copy_(slot.backward_output)
    _runtime_debug("training-backward.wgrad-export.begin", slot=execution.slot.index)
    operands = owner.wgrad_exporter.export(slot)
    _runtime_debug("training-backward.wgrad-export.end", slot=execution.slot.index)
    _runtime_debug("training-backward.end", slot=execution.slot.index)
    return (
        slot.grad_activation[:token_count],
        slot.dprob[:token_count],
        operands,
    )


__all__ = [
    "launch_training_backward",
    "launch_training_forward",
]
