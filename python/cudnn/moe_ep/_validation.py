# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure public-contract validation for :mod:`cudnn.moe_ep`."""

from __future__ import annotations

from typing import Tuple

import torch

from ._contracts import Fc1WeightLayout, ForwardConfig, ValidatedForwardRequest
from ._types import (
    BlockScaledTensor,
    MoeEpTrainingWeights,
    MoeFormat,
    MoeTensor,
)


def _replace_axis(
    shape: Tuple[int, ...],
    axis: int,
    extent: int,
) -> Tuple[int, ...]:
    result = list(shape)
    result[axis] = extent
    return tuple(result)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise RuntimeError(f"this PyTorch build does not provide torch.{name}")
    return dtype


def _logical_shape(tensor: MoeTensor) -> Tuple[int, ...]:
    if isinstance(tensor, BlockScaledTensor):
        return tensor.logical_shape
    if isinstance(tensor, torch.Tensor):
        return tuple(tensor.shape)
    raise ValueError(f"expected torch.Tensor or BlockScaledTensor, got {type(tensor).__name__}")


def _tensor_device(tensor: MoeTensor) -> torch.device:
    if isinstance(tensor, BlockScaledTensor):
        return tensor.device
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    raise ValueError(f"expected torch.Tensor or BlockScaledTensor, got {type(tensor).__name__}")


def _validate_strided(name: str, tensor: torch.Tensor) -> None:
    if tensor.layout is not torch.strided:
        raise ValueError(f"{name} must use torch.strided layout, got {tensor.layout}")


def _validate_tensor_representation(
    name: str,
    tensor: MoeTensor,
    expected_logical_shape: Tuple[int, ...],
) -> None:
    logical_shape = _logical_shape(tensor)
    if logical_shape != expected_logical_shape:
        raise ValueError(f"{name} logical shape must be {expected_logical_shape}, " f"got {logical_shape}")
    if isinstance(tensor, torch.Tensor):
        _validate_strided(name, tensor)
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}")
        return
    if not isinstance(tensor, BlockScaledTensor):
        raise ValueError(f"{name} must be a torch.Tensor or BlockScaledTensor, " f"got {type(tensor).__name__}")
    if tensor.axis != 1:
        raise ValueError(f"{name} block-scaled axis must be 1, got {tensor.axis}")
    _validate_strided(f"{name}.data", tensor.data)
    _validate_strided(f"{name}.scale", tensor.scale)

    logical_extent = expected_logical_shape[tensor.axis]
    if tensor.format is MoeFormat.MXFP8:
        payload_extent = logical_extent
        block_size = 32
        expected_data_dtype = _require_torch_dtype("float8_e4m3fn")
        expected_scale_dtype = _require_torch_dtype("float8_e8m0fnu")
    else:
        payload_extent = _ceil_div(logical_extent, 2)
        block_size = 16
        expected_data_dtype = torch.uint8
        expected_scale_dtype = _require_torch_dtype("float8_e4m3fn")
    expected_data_shape = _replace_axis(
        expected_logical_shape,
        tensor.axis,
        payload_extent,
    )
    expected_scale_shape = _replace_axis(
        expected_logical_shape,
        tensor.axis,
        _ceil_div(logical_extent, block_size),
    )
    if tuple(tensor.data.shape) != expected_data_shape:
        raise ValueError(f"{name}.data shape must be {expected_data_shape}, " f"got {tuple(tensor.data.shape)}")
    if tuple(tensor.scale.shape) != expected_scale_shape:
        raise ValueError(f"{name}.scale shape must be {expected_scale_shape}, " f"got {tuple(tensor.scale.shape)}")
    if tensor.data.dtype != expected_data_dtype:
        raise ValueError(f"{name}.data must have dtype {expected_data_dtype}, " f"got {tensor.data.dtype}")
    if tensor.scale.dtype != expected_scale_dtype:
        raise ValueError(f"{name}.scale must have dtype {expected_scale_dtype}, " f"got {tensor.scale.dtype}")


def _validate_expert_ids(
    config: ForwardConfig,
    topk_idx: torch.Tensor,
) -> None:
    valid_experts = topk_idx.reshape(-1)
    valid_experts = valid_experts[valid_experts != -1]
    if valid_experts.numel() > 0 and bool(((valid_experts < 0) | (valid_experts >= config.num_experts)).any().item()):
        raise ValueError("topk_idx contains out-of-range expert ids")


def _validate_routes(
    config: ForwardConfig,
    token_count: int,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    validate_expert_ids: bool,
) -> None:
    if not isinstance(topk_idx, torch.Tensor):
        raise ValueError(f"topk_idx must be a torch.Tensor, got {type(topk_idx).__name__}")
    if not isinstance(topk_weights, torch.Tensor):
        raise ValueError("topk_weights must be a torch.Tensor, " f"got {type(topk_weights).__name__}")
    _validate_strided("topk_idx", topk_idx)
    _validate_strided("topk_weights", topk_weights)
    route_shape = (token_count, config.top_k)
    if tuple(topk_idx.shape) != route_shape:
        raise ValueError(f"topk_idx shape must be {route_shape}, got {tuple(topk_idx.shape)}")
    if tuple(topk_weights.shape) != route_shape:
        raise ValueError(f"topk_weights shape must be {route_shape}, " f"got {tuple(topk_weights.shape)}")
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_idx must have dtype torch.int32 or torch.int64, " f"got {topk_idx.dtype}")
    if not topk_weights.is_floating_point():
        raise ValueError(f"topk_weights must be floating point, got {topk_weights.dtype}")
    if config.max_tokens_per_rank is not None and token_count > config.max_tokens_per_rank:
        raise ValueError(f"token count {token_count} exceeds " f"max_tokens_per_rank={config.max_tokens_per_rank}")
    if validate_expert_ids:
        _validate_expert_ids(config, topk_idx)


def validate_forward(
    config: ForwardConfig,
    activation: MoeTensor,
    fc1_weight: MoeTensor,
    fc2_weight: MoeTensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    validate_expert_ids: bool = True,
) -> ValidatedForwardRequest:
    """Validate inference-forward semantics without importing a device backend."""

    activation_shape = _logical_shape(activation)
    if len(activation_shape) != 2 or activation_shape[1] != config.hidden_size:
        raise ValueError(f"activation logical shape must be (T, {config.hidden_size}), " f"got {activation_shape}")
    token_count = activation_shape[0]
    _validate_tensor_representation("activation", activation, activation_shape)
    _validate_tensor_representation(
        "fc1_weight",
        fc1_weight,
        (
            config.experts_per_rank,
            config.hidden_size,
            2 * config.intermediate_size,
        ),
    )
    _validate_tensor_representation(
        "fc2_weight",
        fc2_weight,
        (
            config.experts_per_rank,
            config.intermediate_size,
            config.hidden_size,
        ),
    )
    if config.fc1_weight_layout is Fc1WeightLayout.GATE_UP_INTERLEAVED_32 and (
        not isinstance(fc1_weight, BlockScaledTensor)
        or fc1_weight.format is not MoeFormat.MXFP8
    ):
        raise ValueError(
            "weight_interleave_size=32 requires an MXFP8 BlockScaledTensor "
            "for fc1_weight"
        )
    _validate_routes(
        config,
        token_count,
        topk_idx,
        topk_weights,
        validate_expert_ids=False,
    )
    device = _tensor_device(activation)
    for name, tensor in (
        ("fc1_weight", fc1_weight),
        ("fc2_weight", fc2_weight),
        ("topk_idx", topk_idx),
        ("topk_weights", topk_weights),
    ):
        tensor_device = _tensor_device(tensor)
        if tensor_device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor_device}")
    if device.type == "cuda":
        with torch.cuda.device(device):
            capturing = torch.cuda.is_current_stream_capturing()
    else:
        capturing = False
    if validate_expert_ids and not capturing:
        _validate_expert_ids(config, topk_idx)
    return ValidatedForwardRequest(
        config=config,
        activation=activation,
        fc1_weight=fc1_weight,
        fc2_weight=fc2_weight,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        token_count=token_count,
        device=device,
    )


def validate_training_weights(
    config: ForwardConfig,
    weights: MoeEpTrainingWeights,
) -> torch.device:
    """Validate fixed MXFP8 weight bindings used by training resources."""

    def has_supported_layout(tensor: torch.Tensor) -> bool:
        if tensor.is_contiguous():
            return True
        if tensor.ndim != 3:
            return False
        experts, reduction, output = tensor.shape
        return tensor.stride() == (reduction * output, 1, reduction)

    def is_compact_k_major(tensor: torch.Tensor) -> bool:
        if tensor.ndim != 3:
            return False
        experts, reduction, output = tensor.shape
        return tensor.stride() == (reduction * output, 1, reduction)

    if not isinstance(weights, MoeEpTrainingWeights):
        raise TypeError("weights must be a MoeEpTrainingWeights, " f"got {type(weights).__name__}")
    expected = (
        (
            "weights.forward_fc1",
            weights.forward_fc1,
            (
                config.experts_per_rank,
                config.hidden_size,
                2 * config.intermediate_size,
            ),
        ),
        (
            "weights.forward_fc2",
            weights.forward_fc2,
            (
                config.experts_per_rank,
                config.intermediate_size,
                config.hidden_size,
            ),
        ),
        (
            "weights.backward_w2_transpose",
            weights.backward_w2_transpose,
            (
                config.experts_per_rank,
                config.hidden_size,
                config.intermediate_size,
            ),
        ),
        (
            "weights.backward_w1_transpose",
            weights.backward_w1_transpose,
            (
                config.experts_per_rank,
                2 * config.intermediate_size,
                config.hidden_size,
            ),
        ),
    )
    for name, tensor, shape in expected:
        _validate_tensor_representation(name, tensor, shape)
        if not isinstance(tensor, BlockScaledTensor):
            raise TypeError(f"{name} must be an MXFP8 BlockScaledTensor for " "fixed training resources")
        if tensor.format is not MoeFormat.MXFP8:
            raise NotImplementedError(f"{name} must use format='mxfp8', got {tensor.format.value!r}")
        if name in (
            "weights.backward_w2_transpose",
            "weights.backward_w1_transpose",
        ) and not tensor.data.is_contiguous():
            raise ValueError(
                f"{name} data must be contiguous for fixed training weight binding"
            )
        if not has_supported_layout(tensor.data) or not has_supported_layout(tensor.scale):
            raise ValueError(
                f"{name} data and scale must be contiguous or compact K-major "
                "for fixed training weight binding"
            )
    if config.fc1_weight_layout is Fc1WeightLayout.GATE_UP_INTERLEAVED_32 and not (
        is_compact_k_major(weights.forward_fc1.data)
        and is_compact_k_major(weights.forward_fc2.data)
        and weights.backward_w2_transpose.data.is_contiguous()
        and weights.backward_w1_transpose.data.is_contiguous()
    ):
        raise ValueError(
            "weight_interleave_size=32 requires compact K-major forward "
            "weights and contiguous backward transpose weights"
        )
    device = weights.forward_fc1.device
    for name, tensor, _shape in expected[1:]:
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return device


__all__ = ["validate_forward", "validate_training_weights"]
