# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure public-contract validation for :mod:`cudnn.moe_ep`."""

from __future__ import annotations

from typing import Mapping, Tuple

import torch

from ._contracts import Fc1WeightLayout, ForwardConfig, ValidatedForwardRequest
from ._math import round_up
from ._types import (
    BlockScaledTensor,
    MoeEpBackwardWeights,
    MoeEpForwardWeights,
    MoeEpNativeBackwardWeights,
    MoeEpNativeForwardWeights,
    MoeEpNativeWeight,
    MoeEpNativeWeightLayout,
    MoeEpTrainingBackwardOutputs,
    MoeEpTrainingForwardOutputs,
    MoeFormat,
    MoeTensor,
    _block_scaled_representation,
)

_SourceWeightSpec = tuple[str, BlockScaledTensor, Tuple[int, ...]]
_NativeWeightSpec = tuple[
    str,
    MoeEpNativeWeight,
    MoeEpNativeWeightLayout,
    Tuple[int, ...],
    Tuple[int, ...],
    int,
]


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

    (
        expected_data_shape,
        expected_scale_shape,
        expected_data_dtype,
        expected_scale_dtype,
    ) = _block_scaled_representation(
        tensor.format,
        expected_logical_shape,
        tensor.axis,
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
        not isinstance(fc1_weight, BlockScaledTensor) or fc1_weight.format is not MoeFormat.MXFP8
    ):
        raise ValueError("weight_interleave_size=32 requires an MXFP8 BlockScaledTensor " "for fc1_weight")
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


def _validate_source_weight(
    name: str,
    tensor: BlockScaledTensor,
    shape: Tuple[int, ...],
) -> None:
    _validate_tensor_representation(name, tensor, shape)
    if not isinstance(tensor, BlockScaledTensor):
        raise TypeError(f"{name} must be an MXFP8 BlockScaledTensor")
    if tensor.format is not MoeFormat.MXFP8:
        raise NotImplementedError(f"{name} must use format='mxfp8', got {tensor.format.value!r}")
    if not tensor.data.is_contiguous() or not tensor.scale.is_contiguous():
        raise ValueError(f"{name} data and scale must be contiguous")


def _validate_source_weight_pair(
    expected: tuple[_SourceWeightSpec, _SourceWeightSpec],
) -> torch.device:
    for name, tensor, shape in expected:
        _validate_source_weight(name, tensor, shape)
    device = expected[0][1].device
    second_name, second_tensor, _ = expected[1]
    if second_tensor.device != device:
        raise ValueError(f"{second_name} must be on {device}, got {second_tensor.device}")
    return device


def validate_forward_source_weights(
    config: ForwardConfig,
    weights: MoeEpForwardWeights,
) -> torch.device:
    """Validate source weights used by allocation-free forward packing."""

    if not isinstance(weights, MoeEpForwardWeights):
        raise TypeError("weights must be a MoeEpForwardWeights, " f"got {type(weights).__name__}")
    expected = (
        ("weights.fc1", weights.fc1, (config.experts_per_rank, config.hidden_size, 2 * config.intermediate_size)),
        ("weights.fc2", weights.fc2, (config.experts_per_rank, config.intermediate_size, config.hidden_size)),
    )
    return _validate_source_weight_pair(expected)


def validate_backward_source_weights(
    config: ForwardConfig,
    weights: MoeEpBackwardWeights,
) -> torch.device:
    """Validate source weights used by allocation-free backward packing."""

    if not isinstance(weights, MoeEpBackwardWeights):
        raise TypeError("weights must be a MoeEpBackwardWeights, " f"got {type(weights).__name__}")
    expected = (
        ("weights.w2_transpose", weights.w2_transpose, (config.experts_per_rank, config.hidden_size, config.intermediate_size)),
        ("weights.w1_transpose", weights.w1_transpose, (config.experts_per_rank, 2 * config.intermediate_size, config.hidden_size)),
    )
    return _validate_source_weight_pair(expected)


def _blocked_scale_elements(raw_rows: int, raw_columns: int) -> int:
    return round_up(raw_rows, 128) * round_up(raw_columns, 4)


def _validate_native_weight(
    name: str,
    weight: MoeEpNativeWeight,
    *,
    layout_id: MoeEpNativeWeightLayout,
    payload_shape: Tuple[int, ...],
    payload_stride: Tuple[int, ...],
    scale_elements: int,
    scale_dtype: torch.dtype,
    device: torch.device | None,
) -> torch.device:
    if not isinstance(weight, MoeEpNativeWeight):
        raise TypeError(f"{name} must be a MoeEpNativeWeight, got {type(weight).__name__}")
    if weight.layout_id is not layout_id:
        raise ValueError(f"{name}.layout_id must be {layout_id.value!r}, " f"got {weight.layout_id.value!r}")
    _validate_strided(f"{name}.payload", weight.payload)
    _validate_strided(f"{name}.scale", weight.scale)
    if tuple(weight.payload.shape) != payload_shape:
        raise ValueError(f"{name}.payload shape must be {payload_shape}, " f"got {tuple(weight.payload.shape)}")
    if tuple(weight.payload.stride()) != payload_stride:
        raise ValueError(f"{name}.payload stride must be {payload_stride}, " f"got {tuple(weight.payload.stride())}")
    expected_payload_dtype = _require_torch_dtype("float8_e4m3fn")
    if weight.payload.dtype is not expected_payload_dtype:
        raise ValueError(f"{name}.payload must have dtype {expected_payload_dtype}, " f"got {weight.payload.dtype}")
    expected_scale_shape = (payload_shape[0], scale_elements)
    if tuple(weight.scale.shape) != expected_scale_shape:
        raise ValueError(f"{name}.scale shape must be {expected_scale_shape}, " f"got {tuple(weight.scale.shape)}")
    if not weight.scale.is_contiguous():
        raise ValueError(f"{name}.scale must be contiguous")
    if weight.scale.dtype is not scale_dtype:
        raise ValueError(f"{name}.scale must have dtype {scale_dtype}, " f"got {weight.scale.dtype}")
    for field_name, tensor in (
        ("payload", weight.payload),
        ("scale", weight.scale),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name}.{field_name} must be at least 16-byte aligned")
    if device is not None and weight.device != device:
        raise ValueError(f"{name} must be on {device}, got {weight.device}")
    return weight.device


def _validate_native_weight_pair(
    expected: tuple[_NativeWeightSpec, _NativeWeightSpec],
    *,
    scale_dtype: torch.dtype,
    device: torch.device | None,
) -> torch.device:
    resolved = device
    for name, weight, layout_id, payload_shape, payload_stride, scale_elements in expected:
        resolved = _validate_native_weight(
            name,
            weight,
            layout_id=layout_id,
            payload_shape=payload_shape,
            payload_stride=payload_stride,
            scale_elements=scale_elements,
            scale_dtype=scale_dtype,
            device=resolved,
        )
    assert resolved is not None
    return resolved


def validate_native_forward_weights(
    config: ForwardConfig,
    weights: MoeEpNativeForwardWeights,
    *,
    device: torch.device | None = None,
) -> torch.device:
    if not isinstance(weights, MoeEpNativeForwardWeights):
        raise TypeError("weights must be a MoeEpNativeForwardWeights, " f"got {type(weights).__name__}")
    if config.fc1_weight_layout is not Fc1WeightLayout.GATE_UP_INTERLEAVED_32:
        raise ValueError("native training weights require weight_interleave_size=32")
    experts = config.experts_per_rank
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    sf_dtype = _require_torch_dtype("float8_e8m0fnu")
    expected = (
        (
            "weights.fc1",
            weights.fc1,
            MoeEpNativeWeightLayout.FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1,
            (experts, hidden, 2 * intermediate),
            (hidden * 2 * intermediate, 1, hidden),
            _blocked_scale_elements(2 * intermediate, hidden // 32),
        ),
        (
            "weights.fc2",
            weights.fc2,
            MoeEpNativeWeightLayout.FORWARD_FC2_K_MAJOR_V1,
            (experts, intermediate, hidden),
            (intermediate * hidden, 1, intermediate),
            _blocked_scale_elements(hidden, intermediate // 32),
        ),
    )
    return _validate_native_weight_pair(
        expected,
        scale_dtype=sf_dtype,
        device=device,
    )


def validate_native_backward_weights(
    config: ForwardConfig,
    weights: MoeEpNativeBackwardWeights,
    *,
    device: torch.device | None = None,
) -> torch.device:
    if not isinstance(weights, MoeEpNativeBackwardWeights):
        raise TypeError("weights must be a MoeEpNativeBackwardWeights, " f"got {type(weights).__name__}")
    if config.fc1_weight_layout is not Fc1WeightLayout.GATE_UP_INTERLEAVED_32:
        raise ValueError("native training weights require weight_interleave_size=32")
    experts = config.experts_per_rank
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    sf_dtype = _require_torch_dtype("float8_e8m0fnu")
    expected = (
        (
            "weights.w2_transpose",
            weights.w2_transpose,
            MoeEpNativeWeightLayout.BACKWARD_W2_TRANSPOSE_V1,
            (experts, hidden, intermediate),
            (hidden * intermediate, intermediate, 1),
            _blocked_scale_elements(intermediate, hidden // 32),
        ),
        (
            "weights.w1_transpose",
            weights.w1_transpose,
            MoeEpNativeWeightLayout.BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1,
            (experts, 2 * intermediate, hidden),
            (2 * intermediate * hidden, hidden, 1),
            _blocked_scale_elements(hidden, 2 * intermediate // 32),
        ),
    )
    return _validate_native_weight_pair(
        expected,
        scale_dtype=sf_dtype,
        device=device,
    )


def validate_training_input(
    config: ForwardConfig,
    name: str,
    value: MoeTensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    device: torch.device,
) -> int:
    logical_shape = _logical_shape(value)
    if len(logical_shape) != 2 or logical_shape[1] != config.hidden_size:
        raise ValueError(f"{name} logical shape must be (T, {config.hidden_size}), " f"got {logical_shape}")
    _validate_tensor_representation(name, value, logical_shape)
    if isinstance(value, BlockScaledTensor):
        if value.format is not MoeFormat.MXFP8:
            raise NotImplementedError(f"{name} only supports MXFP8 block scaling")
        if not value.data.is_contiguous() or not value.scale.is_contiguous():
            raise ValueError(f"{name} MXFP8 data and scale must be contiguous")
    elif value.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be BF16, FP32, or an MXFP8 BlockScaledTensor")
    elif not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    token_count = logical_shape[0]
    if device.type == "cuda":
        with torch.cuda.device(device):
            capturing = torch.cuda.is_current_stream_capturing()
    else:
        capturing = False
    _validate_routes(
        config,
        token_count,
        topk_idx,
        topk_weights,
        validate_expert_ids=not capturing,
    )
    if topk_idx.dtype is not torch.int32 or not topk_idx.is_contiguous():
        raise TypeError("training topk_idx must be contiguous torch.int32")
    if topk_weights.dtype is not torch.float32 or not topk_weights.is_contiguous():
        raise TypeError("training topk_weights must be contiguous torch.float32")
    tensors = (
        (name, value),
        ("topk_idx", topk_idx),
        ("topk_weights", topk_weights),
    )
    for tensor_name, tensor in tensors:
        tensor_device = _tensor_device(tensor)
        if tensor_device != device:
            raise ValueError(f"{tensor_name} must be on {device}, got {tensor_device}")
    return token_count


def _tensor_byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    byte_start = tensor.data_ptr()
    max_element_offset = sum(
        (int(extent) - 1) * int(step)
        for extent, step in zip(tensor.shape, tensor.stride())
        if int(extent) > 0
    )
    byte_end = byte_start + (
        0
        if tensor.numel() == 0
        else (max_element_offset + 1) * tensor.element_size()
    )
    return byte_start, byte_end


def _assert_no_overlap(
    name: str,
    byte_range: tuple[int, int],
    ranges: list[tuple[int, int, str]],
) -> None:
    byte_start, byte_end = byte_range
    for other_start, other_end, other_name in ranges:
        if byte_start < other_end and other_start < byte_end:
            raise ValueError(f"{name} must not alias {other_name}")


def _validate_named_buffers(
    tensors: Mapping[str, object],
    requirements: Mapping[str, tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, int]],
    *,
    device: torch.device,
) -> None:
    ranges: list[tuple[int, int, str]] = []
    for name, requirement in requirements.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"out.{name} must be a torch.Tensor")
        shape, stride, dtype, alignment = requirement
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"out.{name} shape must be {tuple(shape)}, got {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != tuple(stride):
            raise ValueError(f"out.{name} stride must be {tuple(stride)}, " f"got {tuple(tensor.stride())}")
        if tensor.dtype is not dtype:
            raise ValueError(f"out.{name} dtype must be {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"out.{name} must be on {device}, got {tensor.device}")
        if tensor.data_ptr() % alignment:
            raise ValueError(f"out.{name} must be {alignment}-byte aligned")
        qualified_name = f"out.{name}"
        byte_start, byte_end = _tensor_byte_range(tensor)
        _assert_no_overlap(qualified_name, (byte_start, byte_end), ranges)
        ranges.append((byte_start, byte_end, qualified_name))


def _validate_output_buffers(
    output: object,
    requirements: Mapping[str, tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, int]],
    *,
    device: torch.device,
) -> None:
    _validate_named_buffers(
        {name: getattr(output, name) for name in requirements},
        requirements,
        device=device,
    )


def validate_training_non_aliasing(
    tensors: Mapping[str, torch.Tensor | None],
) -> None:
    """Reject overlapping caller inputs, saved state, weights, and outputs."""

    ranges: list[tuple[int, int, str]] = []
    for name, tensor in tensors.items():
        if tensor is None or tensor.numel() == 0:
            continue
        byte_start, byte_end = _tensor_byte_range(tensor)
        _assert_no_overlap(name, (byte_start, byte_end), ranges)
        ranges.append((byte_start, byte_end, name))


def validate_training_forward_outputs(
    output: MoeEpTrainingForwardOutputs,
    requirements: Mapping[str, tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, int]],
    *,
    device: torch.device,
) -> None:
    if not isinstance(output, MoeEpTrainingForwardOutputs):
        raise TypeError("out must be a MoeEpTrainingForwardOutputs, " f"got {type(output).__name__}")
    _validate_output_buffers(output, requirements, device=device)


def validate_training_forward_state(
    *,
    fc1_preact: torch.Tensor,
    fc1_a: torch.Tensor | None,
    fc1_sfa: torch.Tensor | None,
    valid_route_counts: torch.Tensor | None,
    expert_offsets: torch.Tensor | None,
    requirements: Mapping[str, tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, int]],
    device: torch.device,
) -> None:
    _validate_named_buffers(
        {
            "fc1_preact": fc1_preact,
            "fc1_a": fc1_a,
            "fc1_sfa": fc1_sfa,
            "valid_route_counts": valid_route_counts,
            "expert_offsets": expert_offsets,
        },
        requirements,
        device=device,
    )


def validate_training_backward_outputs(
    output: MoeEpTrainingBackwardOutputs,
    requirements: Mapping[str, tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, int]],
    *,
    device: torch.device,
) -> None:
    if not isinstance(output, MoeEpTrainingBackwardOutputs):
        raise TypeError("out must be a MoeEpTrainingBackwardOutputs, " f"got {type(output).__name__}")
    _validate_output_buffers(output, requirements, device=device)


__all__ = [
    "validate_backward_source_weights",
    "validate_forward",
    "validate_forward_source_weights",
    "validate_native_backward_weights",
    "validate_native_forward_weights",
    "validate_training_backward_outputs",
    "validate_training_forward_outputs",
    "validate_training_forward_state",
    "validate_training_input",
    "validate_training_non_aliasing",
]
