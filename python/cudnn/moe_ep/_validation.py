# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pure public-contract validation for :mod:`cudnn.moe_ep`.

This module intentionally depends only on PyTorch and the lightweight public
API types. It must not import CuTeDSL, CUDA Python, NVSHMEM, or the private
MegaMoE runtime.
"""

from __future__ import annotations

from typing import Tuple

import torch

from ._contracts import (
    ForwardConfig,
    ValidatedBackwardRequest,
    ValidatedForwardRequest,
)
from ._types import (
    BlockScaledTensor,
    MoeEpWgradForwardStash,
    MoeFormat,
    MoeTensor,
)


def _replace_axis(shape: Tuple[int, ...], axis: int, extent: int) -> Tuple[int, ...]:
    result = list(shape)
    result[axis] = extent
    return tuple(result)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


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
    raise ValueError(
        f"expected torch.Tensor or BlockScaledTensor, got {type(tensor).__name__}"
    )


def _tensor_device(tensor: MoeTensor) -> torch.device:
    if isinstance(tensor, BlockScaledTensor):
        return tensor.device
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    raise ValueError(
        f"expected torch.Tensor or BlockScaledTensor, got {type(tensor).__name__}"
    )


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
        raise ValueError(
            f"{name} logical shape must be {expected_logical_shape}, got {logical_shape}"
        )

    if isinstance(tensor, torch.Tensor):
        _validate_strided(name, tensor)
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating point, got {tensor.dtype}")
        return

    if not isinstance(tensor, BlockScaledTensor):
        raise ValueError(
            f"{name} must be a torch.Tensor or BlockScaledTensor, "
            f"got {type(tensor).__name__}"
        )

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
        raise ValueError(
            f"{name}.data shape must be {expected_data_shape}, "
            f"got {tuple(tensor.data.shape)}"
        )
    if tuple(tensor.scale.shape) != expected_scale_shape:
        raise ValueError(
            f"{name}.scale shape must be {expected_scale_shape}, "
            f"got {tuple(tensor.scale.shape)}"
        )
    if tensor.data.dtype != expected_data_dtype:
        raise ValueError(
            f"{name}.data must have dtype {expected_data_dtype}, "
            f"got {tensor.data.dtype}"
        )
    if tensor.scale.dtype != expected_scale_dtype:
        raise ValueError(
            f"{name}.scale must have dtype {expected_scale_dtype}, "
            f"got {tensor.scale.dtype}"
        )


def _validate_expert_ids(
    config: ForwardConfig,
    topk_idx: torch.Tensor,
) -> None:
    valid_experts = topk_idx.reshape(-1)
    valid_experts = valid_experts[valid_experts != -1]
    if valid_experts.numel() > 0 and bool(
        (
            (valid_experts < 0)
            | (valid_experts >= config.num_experts)
        ).any().item()
    ):
        raise ValueError("topk_idx contains out-of-range expert ids")


def _validate_routes(
    config: ForwardConfig,
    token_count: int,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    validate_expert_ids: bool,
) -> None:
    """Validate the public routing plane shared by forward and backward."""

    if not isinstance(topk_idx, torch.Tensor):
        raise ValueError(
            f"topk_idx must be a torch.Tensor, got {type(topk_idx).__name__}"
        )
    if not isinstance(topk_weights, torch.Tensor):
        raise ValueError(
            "topk_weights must be a torch.Tensor, "
            f"got {type(topk_weights).__name__}"
        )
    _validate_strided("topk_idx", topk_idx)
    _validate_strided("topk_weights", topk_weights)

    route_shape = (token_count, config.top_k)
    if tuple(topk_idx.shape) != route_shape:
        raise ValueError(
            f"topk_idx shape must be {route_shape}, got {tuple(topk_idx.shape)}"
        )
    if tuple(topk_weights.shape) != route_shape:
        raise ValueError(
            f"topk_weights shape must be {route_shape}, "
            f"got {tuple(topk_weights.shape)}"
        )
    if topk_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "topk_idx must have dtype torch.int32 or torch.int64, "
            f"got {topk_idx.dtype}"
        )
    if not topk_weights.is_floating_point():
        raise ValueError(
            f"topk_weights must be floating point, got {topk_weights.dtype}"
        )
    if (
        config.max_tokens_per_rank is not None
        and token_count > config.max_tokens_per_rank
    ):
        raise ValueError(
            f"token count {token_count} exceeds "
            f"max_tokens_per_rank={config.max_tokens_per_rank}"
        )
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
    """Validate public forward semantics without importing a device backend."""

    activation_shape = _logical_shape(activation)
    if len(activation_shape) != 2 or activation_shape[1] != config.hidden_size:
        raise ValueError(
            f"activation logical shape must be (T, {config.hidden_size}), "
            f"got {activation_shape}"
        )
    token_count = activation_shape[0]
    expected_fc1 = (
        config.experts_per_rank,
        config.hidden_size,
        2 * config.intermediate_size,
    )
    expected_fc2 = (
        config.experts_per_rank,
        config.intermediate_size,
        config.hidden_size,
    )
    _validate_tensor_representation("activation", activation, activation_shape)
    _validate_tensor_representation("fc1_weight", fc1_weight, expected_fc1)
    _validate_tensor_representation("fc2_weight", fc2_weight, expected_fc2)

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

    # Boolean compaction plus the host-visible ``item()`` below is not CUDA
    # graph capturable. Eager calls (including the mandatory pre-capture
    # warmup) retain strict validation. During capture/replay, callers must
    # preserve that validated invariant: every route is -1 or a valid global
    # expert ID.
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


def _validate_wgrad_forward_stash(
    config: ForwardConfig,
    stash: MoeEpWgradForwardStash,
    route_metadata: torch.Tensor,
    device: torch.device,
) -> None:
    """Validate caller-owned forward operands and their route identity."""

    if not isinstance(stash, MoeEpWgradForwardStash):
        raise TypeError(
            "wgrad_forward_stash must be a MoeEpWgradForwardStash, "
            f"got {type(stash).__name__}"
        )

    tensors = (
        ("fc1_a", stash.fc1_a),
        ("fc1_sfa", stash.fc1_sfa),
        ("expert_offsets", stash.expert_offsets),
        ("valid_route_counts", stash.valid_route_counts),
        ("route_metadata", stash.route_metadata),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"wgrad_forward_stash.{name} must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
        _validate_strided(f"wgrad_forward_stash.{name}", tensor)
        if tensor.device != device:
            raise ValueError(
                f"wgrad_forward_stash.{name} must be on {device}, "
                f"got {tensor.device}"
            )

    e4m3_dtype = _require_torch_dtype("float8_e4m3fn")
    e8m0_dtype = _require_torch_dtype("float8_e8m0fnu")
    if stash.fc1_a.dtype is not e4m3_dtype:
        raise TypeError(
            "wgrad_forward_stash.fc1_a must have dtype "
            f"{e4m3_dtype}, got {stash.fc1_a.dtype}"
        )
    if stash.fc1_sfa.dtype is not e8m0_dtype:
        raise TypeError(
            "wgrad_forward_stash.fc1_sfa must have dtype "
            f"{e8m0_dtype}, got {stash.fc1_sfa.dtype}"
        )

    expert_shape = (config.experts_per_rank,)
    for name, tensor in (
        ("expert_offsets", stash.expert_offsets),
        ("valid_route_counts", stash.valid_route_counts),
    ):
        if tuple(tensor.shape) != expert_shape:
            raise ValueError(
                f"wgrad_forward_stash.{name} shape must be {expert_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype is not torch.int32:
            raise TypeError(
                f"wgrad_forward_stash.{name} must have dtype torch.int32, "
                f"got {tensor.dtype}"
            )

    if tuple(stash.route_metadata.shape) != tuple(route_metadata.shape):
        raise ValueError(
            "wgrad_forward_stash.route_metadata shape must match "
            "route_metadata"
        )
    if stash.route_metadata.dtype is not torch.int32:
        raise TypeError(
            "wgrad_forward_stash.route_metadata must have dtype torch.int32, "
            f"got {stash.route_metadata.dtype}"
        )
    if not torch.equal(stash.route_metadata, route_metadata):
        raise ValueError(
            "wgrad_forward_stash route identity does not match route_metadata"
        )

    offsets = [int(value) for value in stash.expert_offsets.cpu().tolist()]
    counts = [int(value) for value in stash.valid_route_counts.cpu().tolist()]
    previous = 0
    for expert, (offset, count) in enumerate(zip(offsets, counts)):
        padded_routes = offset - previous
        if offset < previous:
            raise ValueError(
                "wgrad_forward_stash.expert_offsets must be non-decreasing"
            )
        if count < 0 or count > padded_routes:
            raise ValueError(
                "wgrad_forward_stash.valid_route_counts must fit each "
                f"expert's padded range; expert {expert} has count={count} "
                f"and capacity={padded_routes}"
            )
        expected_padded_routes = _round_up(
            count,
            config.token_padding_size,
        )
        if padded_routes != expected_padded_routes:
            raise ValueError(
                "wgrad_forward_stash expert ranges must use the canonical "
                f"{config.token_padding_size}-row padding; expert {expert} "
                f"has capacity={padded_routes}, expected="
                f"{expected_padded_routes}"
            )
        previous = offset

    padded_route_count = offsets[-1] if offsets else 0
    expected_fc1_a = (config.hidden_size, padded_route_count)
    if tuple(stash.fc1_a.shape) != expected_fc1_a:
        raise ValueError(
            "wgrad_forward_stash.fc1_a shape must be "
            f"{expected_fc1_a}, got {tuple(stash.fc1_a.shape)}"
        )
    if padded_route_count % 32:
        raise ValueError(
            "wgrad_forward_stash padded route count must be divisible by 32"
        )
    expected_fc1_sfa = (
        _round_up(config.hidden_size, 128),
        _round_up(padded_route_count // 32, 4),
    )
    if tuple(stash.fc1_sfa.shape) != expected_fc1_sfa:
        raise ValueError(
            "wgrad_forward_stash.fc1_sfa shape must be "
            f"{expected_fc1_sfa}, got {tuple(stash.fc1_sfa.shape)}"
        )
    if padded_route_count and not stash.fc1_a.is_contiguous():
        raise ValueError(
            "wgrad_forward_stash.fc1_a must use compact (K, 1) strides"
        )
    if not stash.fc1_sfa.is_contiguous():
        raise ValueError(
            "wgrad_forward_stash.fc1_sfa must be contiguous"
        )
    for name, tensor, alignment in (
        ("fc1_a", stash.fc1_a, 16),
        ("fc1_sfa", stash.fc1_sfa, 16),
        ("expert_offsets", stash.expert_offsets, 4),
        ("valid_route_counts", stash.valid_route_counts, 4),
    ):
        if tensor.data_ptr() % alignment:
            raise ValueError(
                f"wgrad_forward_stash.{name} must be "
                f"{alignment}-byte aligned"
            )

    local_routes = int(route_metadata.shape[0])
    if sum(counts) != local_routes:
        raise ValueError(
            "wgrad_forward_stash.valid_route_counts must sum to the "
            "route_metadata row count"
        )
    if local_routes:
        local_experts = route_metadata[:, 0].to(torch.int64)
        if bool(
            (
                (local_experts < 0)
                | (local_experts >= config.experts_per_rank)
            ).any().item()
        ):
            raise ValueError(
                "route_metadata contains out-of-range local expert ids"
            )
        metadata_counts = torch.bincount(
            local_experts,
            minlength=config.experts_per_rank,
        )
        expected_counts = stash.valid_route_counts.to(torch.int64)
        if not torch.equal(metadata_counts, expected_counts):
            raise ValueError(
                "wgrad_forward_stash.valid_route_counts do not match "
                "route_metadata"
            )
        expected_experts = torch.repeat_interleave(
            torch.arange(
                config.experts_per_rank,
                dtype=torch.int64,
                device=device,
            ),
            expected_counts,
            output_size=local_routes,
        )
        if not torch.equal(local_experts, expected_experts):
            raise ValueError(
                "route_metadata rows must be grouped by local expert"
            )

        src_ranks = route_metadata[:, 1]
        src_tokens = route_metadata[:, 2]
        src_slots = route_metadata[:, 3]
        if bool(((src_ranks < 0) | (src_ranks >= config.ep_size)).any().item()):
            raise ValueError("route_metadata contains out-of-range source ranks")
        if bool((src_tokens < 0).any().item()):
            raise ValueError("route_metadata contains negative source tokens")
        if config.max_tokens_per_rank is not None and bool(
            (src_tokens >= config.max_tokens_per_rank).any().item()
        ):
            raise ValueError("route_metadata contains out-of-range source tokens")
        if bool(((src_slots < 0) | (src_slots >= config.top_k)).any().item()):
            raise ValueError("route_metadata contains out-of-range source slots")


def validate_backward(
    config: ForwardConfig,
    grad_output: torch.Tensor,
    fc1_weight: MoeTensor,
    fc2_weight: MoeTensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    fc1_c: torch.Tensor,
    route_metadata: torch.Tensor,
    *,
    wgrad_forward_stash: MoeEpWgradForwardStash | None = None,
) -> ValidatedBackwardRequest:
    """Validate public backward semantics without importing a device backend."""

    if not isinstance(grad_output, torch.Tensor):
        raise TypeError(
            "grad_output must be a torch.Tensor, "
            f"got {type(grad_output).__name__}"
        )
    _validate_strided("grad_output", grad_output)
    if grad_output.ndim != 2 or grad_output.shape[1] != config.hidden_size:
        raise ValueError(
            f"grad_output shape must be (T, {config.hidden_size}), "
            f"got {tuple(grad_output.shape)}"
        )
    if not grad_output.is_floating_point():
        raise TypeError(
            f"grad_output must be floating point, got {grad_output.dtype}"
        )
    token_count = int(grad_output.shape[0])
    expected_fc1 = (
        config.experts_per_rank,
        config.hidden_size,
        2 * config.intermediate_size,
    )
    expected_fc2 = (
        config.experts_per_rank,
        config.intermediate_size,
        config.hidden_size,
    )
    _validate_tensor_representation("fc1_weight", fc1_weight, expected_fc1)
    _validate_tensor_representation("fc2_weight", fc2_weight, expected_fc2)

    _validate_routes(
        config,
        token_count,
        topk_idx,
        topk_weights,
        validate_expert_ids=True,
    )

    if not isinstance(route_metadata, torch.Tensor):
        raise TypeError(
            "route_metadata must be a torch.Tensor, "
            f"got {type(route_metadata).__name__}"
        )
    _validate_strided("route_metadata", route_metadata)
    if route_metadata.ndim != 2 or route_metadata.shape[1] != 4:
        raise ValueError(
            "route_metadata shape must be (local_routes, 4), "
            f"got {tuple(route_metadata.shape)}"
        )
    if route_metadata.dtype is not torch.int32:
        raise TypeError(
            "route_metadata must have dtype torch.int32, "
            f"got {route_metadata.dtype}"
        )
    local_routes = int(route_metadata.shape[0])
    expected_fc1_c = (local_routes, 2 * config.intermediate_size)
    if not isinstance(fc1_c, torch.Tensor):
        raise TypeError(
            f"fc1_c must be a torch.Tensor, got {type(fc1_c).__name__}"
        )
    _validate_strided("fc1_c", fc1_c)
    if tuple(fc1_c.shape) != expected_fc1_c:
        raise ValueError(
            f"fc1_c shape must be {expected_fc1_c}, got {tuple(fc1_c.shape)}"
        )
    if fc1_c.dtype is not torch.bfloat16:
        raise TypeError(
            f"fc1_c must have dtype torch.bfloat16, got {fc1_c.dtype}"
        )

    device = grad_output.device
    for name, tensor in (
        ("fc1_weight", fc1_weight),
        ("fc2_weight", fc2_weight),
        ("topk_idx", topk_idx),
        ("topk_weights", topk_weights),
        ("grad_output", grad_output),
        ("fc1_c", fc1_c),
        ("route_metadata", route_metadata),
    ):
        tensor_device = _tensor_device(tensor)
        if tensor_device != device:
            raise ValueError(
                f"{name} must be on {device}, got {tensor_device}"
            )

    if config.backward_wgrad_mode == "operands":
        _validate_wgrad_forward_stash(
            config,
            wgrad_forward_stash,
            route_metadata,
            device,
        )
    elif wgrad_forward_stash is not None:
        raise ValueError(
            "wgrad_forward_stash is only accepted when "
            "backward_wgrad_mode='operands'"
        )

    return ValidatedBackwardRequest(
        config=config,
        grad_output=grad_output,
        fc1_weight=fc1_weight,
        fc2_weight=fc2_weight,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        fc1_c=fc1_c,
        route_metadata=route_metadata,
        token_count=token_count,
        local_routes=local_routes,
        device=device,
        wgrad_forward_stash=wgrad_forward_stash,
    )


__all__ = ["validate_backward", "validate_forward"]
