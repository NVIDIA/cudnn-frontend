# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight public tensor and format types for :mod:`cudnn.moe_ep`."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

import torch

from ._math import ceil_div


class MoeFormat(str, Enum):
    """Data formats supported by the MoE+EP interface."""

    BF16 = "bf16"
    MXFP8 = "mxfp8"
    NVFP4 = "nvfp4"


def parse_format(value: Union[MoeFormat, str]) -> MoeFormat:
    """Normalize a public format value."""

    if isinstance(value, MoeFormat):
        return value
    try:
        return MoeFormat(value.lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(item.value for item in MoeFormat)
        raise ValueError(f"unsupported format {value!r}; expected one of: {choices}") from exc


def _normalize_axis(axis: int, ndim: int) -> int:
    if isinstance(axis, bool):
        raise ValueError(f"axis must be an integer, got {axis!r}")
    try:
        axis = operator.index(axis)
    except TypeError as exc:
        raise ValueError(f"axis must be an integer, got {axis!r}") from exc
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of range for a {ndim}-D tensor")
    return normalized


def _block_scaled_representation(
    fmt: MoeFormat,
    logical_shape: Tuple[int, ...],
    axis: int,
) -> tuple[Tuple[int, ...], Tuple[int, ...], torch.dtype, torch.dtype]:
    if fmt is MoeFormat.BF16:
        raise ValueError("BlockScaledTensor only represents mxfp8 or nvfp4")
    logical_extent = logical_shape[axis]
    block_size = 32 if fmt is MoeFormat.MXFP8 else 16
    payload_extent = (
        logical_extent
        if fmt is MoeFormat.MXFP8
        else ceil_div(logical_extent, 2)
    )
    data_shape = list(logical_shape)
    data_shape[axis] = payload_extent
    scale_shape = list(logical_shape)
    scale_shape[axis] = ceil_div(logical_extent, block_size)

    e4m3_dtype = getattr(torch, "float8_e4m3fn", None)
    if e4m3_dtype is None:
        raise RuntimeError("this PyTorch build does not provide torch.float8_e4m3fn")
    if fmt is MoeFormat.MXFP8:
        scale_dtype = getattr(torch, "float8_e8m0fnu", None)
        if scale_dtype is None:
            raise RuntimeError(
                "this PyTorch build does not provide torch.float8_e8m0fnu"
            )
        data_dtype = e4m3_dtype
    else:
        data_dtype = torch.uint8
        scale_dtype = e4m3_dtype
    return tuple(data_shape), tuple(scale_shape), data_dtype, scale_dtype


@dataclass(frozen=True)
class BlockScaledTensor:
    """Data-plus-scale result returned for MXFP8 and NVFP4 outputs."""

    data: torch.Tensor
    scale: torch.Tensor
    format: Union[MoeFormat, str]
    logical_shape: Tuple[int, ...]
    axis: int = -1

    def __post_init__(self) -> None:
        if not isinstance(self.data, torch.Tensor):
            raise ValueError(f"data must be a torch.Tensor, got {type(self.data).__name__}")
        if not isinstance(self.scale, torch.Tensor):
            raise ValueError(f"scale must be a torch.Tensor, got {type(self.scale).__name__}")
        fmt = parse_format(self.format)
        if fmt is MoeFormat.BF16:
            raise ValueError("BlockScaledTensor only represents mxfp8 or nvfp4")
        if self.data.device != self.scale.device:
            raise ValueError(f"data device {self.data.device} does not match " f"scale device {self.scale.device}")
        try:
            raw_logical_shape = tuple(self.logical_shape)
        except TypeError as exc:
            raise ValueError("logical_shape must be an iterable of integers") from exc
        logical_shape = []
        for dim in raw_logical_shape:
            if isinstance(dim, bool):
                raise ValueError(f"logical_shape dimensions must be integers, got {dim!r}")
            try:
                dim = operator.index(dim)
            except TypeError as exc:
                raise ValueError(f"logical_shape dimensions must be integers, got {dim!r}") from exc
            if dim < 0:
                raise ValueError(f"logical_shape dimensions must be non-negative, got {dim}")
            logical_shape.append(dim)
        normalized_shape = tuple(logical_shape)
        axis = _normalize_axis(self.axis, len(normalized_shape))
        (
            expected_data_shape,
            expected_scale_shape,
            expected_data_dtype,
            expected_scale_dtype,
        ) = _block_scaled_representation(fmt, normalized_shape, axis)
        if tuple(self.data.shape) != expected_data_shape:
            raise ValueError(f"{fmt.value} data shape must be {expected_data_shape}, " f"got {tuple(self.data.shape)}")
        if tuple(self.scale.shape) != expected_scale_shape:
            raise ValueError(f"{fmt.value} scale shape must be {expected_scale_shape}, " f"got {tuple(self.scale.shape)}")
        if self.data.dtype is not expected_data_dtype:
            raise ValueError(f"{fmt.value} data must have dtype {expected_data_dtype}, " f"got {self.data.dtype}")
        if self.scale.dtype is not expected_scale_dtype:
            raise ValueError(f"{fmt.value} scale must have dtype {expected_scale_dtype}, " f"got {self.scale.dtype}")
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "logical_shape", normalized_shape)
        object.__setattr__(self, "axis", axis)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.logical_shape

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def block_size(self) -> int:
        return 32 if self.format is MoeFormat.MXFP8 else 16

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode the logical, unswizzled block-scaled representation."""

        logical_extent = self.logical_shape[self.axis]
        scale = self.scale.movedim(self.axis, -1).float()
        expanded_scale = scale.repeat_interleave(
            self.block_size,
            dim=-1,
        )[..., :logical_extent]

        if self.format is MoeFormat.MXFP8:
            values = self.data.movedim(self.axis, -1).float()
        else:
            packed = self.data.movedim(self.axis, -1)
            low = packed & 0x0F
            high = packed >> 4
            codes = torch.stack((low, high), dim=-1).flatten(-2)[..., :logical_extent]
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ],
                dtype=torch.float32,
                device=packed.device,
            )
            values = table[codes.long()]

        return (values * expanded_scale).movedim(-1, self.axis).to(dtype)


class MoeEpNativeWeightLayout(str, Enum):
    """Versioned kernel-native MXFP8 weight layouts."""

    FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1 = "mxfp8.forward_fc1.gate_up_interleaved_32.blocked_sf.v1"
    FORWARD_FC2_K_MAJOR_V1 = "mxfp8.forward_fc2.k_major.blocked_sf.v1"
    BACKWARD_W2_TRANSPOSE_V1 = "mxfp8.backward_w2_transpose.contiguous.blocked_sf.v1"
    BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1 = "mxfp8.backward_w1_transpose.gate_up_interleaved_32.blocked_sf.v1"


@dataclass(frozen=True)
class MoeEpForwardWeights:
    """Logical gate-then-up MXFP8 sources accepted by the fallback packer."""

    fc1: BlockScaledTensor
    fc2: BlockScaledTensor


@dataclass(frozen=True)
class MoeEpBackwardWeights:
    """Logical gate-then-up MXFP8 transpose sources for the fallback packer."""

    w2_transpose: BlockScaledTensor
    w1_transpose: BlockScaledTensor


@dataclass(frozen=True)
class MoeEpNativeWeight:
    """One kernel-executable payload plus Rubin blocked/interleaved scales."""

    payload: torch.Tensor
    scale: torch.Tensor
    layout_id: Union[MoeEpNativeWeightLayout, str]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, torch.Tensor):
            raise TypeError("payload must be a torch.Tensor, " f"got {type(self.payload).__name__}")
        if not isinstance(self.scale, torch.Tensor):
            raise TypeError("scale must be a torch.Tensor, " f"got {type(self.scale).__name__}")
        if self.payload.device != self.scale.device:
            raise ValueError(f"payload device {self.payload.device} does not match " f"scale device {self.scale.device}")
        try:
            layout_id = MoeEpNativeWeightLayout(self.layout_id)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(layout.value for layout in MoeEpNativeWeightLayout)
            raise ValueError(f"unsupported native weight layout_id {self.layout_id!r}; " f"expected one of: {choices}") from exc
        object.__setattr__(self, "layout_id", layout_id)

    @property
    def device(self) -> torch.device:
        return self.payload.device


@dataclass(frozen=True)
class MoeEpNativeForwardWeights:
    """Independent kernel-native weights consumed by one forward call."""

    fc1: MoeEpNativeWeight
    fc2: MoeEpNativeWeight


@dataclass(frozen=True)
class MoeEpNativeBackwardWeights:
    """Independent kernel-native transpose weights consumed by one backward."""

    w2_transpose: MoeEpNativeWeight
    w1_transpose: MoeEpNativeWeight


@dataclass(frozen=True)
class MoeEpForwardWeightStaging:
    """Caller-owned destinations used by forward weight materialization."""

    fc1_payload: torch.Tensor
    fc1_scale: torch.Tensor
    fc2_payload: torch.Tensor
    fc2_scale: torch.Tensor


@dataclass(frozen=True)
class MoeEpBackwardWeightStaging:
    """Caller-owned destinations used by backward weight materialization."""

    w2_transpose_payload: torch.Tensor
    w2_transpose_scale: torch.Tensor
    w1_transpose_payload: torch.Tensor
    w1_transpose_scale: torch.Tensor


@dataclass(frozen=True)
class MoeEpTrainingForwardOutputs:
    """Caller-owned forward destinations."""

    fc1_preact: torch.Tensor
    output: torch.Tensor | None = None
    fc1_a: torch.Tensor | None = None
    fc1_sfa: torch.Tensor | None = None
    valid_route_counts: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None


@dataclass(frozen=True)
class MoeEpTrainingBackwardOutputs:
    """Caller-owned backward and final grouped-WGrad destinations."""

    grad_activation: torch.Tensor | None = None
    dprob: torch.Tensor | None = None
    fc1_b: torch.Tensor | None = None
    fc1_sfb: torch.Tensor | None = None
    fc2_a: torch.Tensor | None = None
    fc2_sfa: torch.Tensor | None = None
    fc2_b: torch.Tensor | None = None
    fc2_sfb: torch.Tensor | None = None


@dataclass(frozen=True)
class MoeEpTrainingWgradOperands:
    """Non-owning views over caller-owned grouped-WGrad operand buffers."""

    fc1_a: torch.Tensor
    fc1_sfa: torch.Tensor
    fc1_b: torch.Tensor
    fc1_sfb: torch.Tensor
    fc2_a: torch.Tensor
    fc2_sfa: torch.Tensor
    fc2_b: torch.Tensor
    fc2_sfb: torch.Tensor
    expert_offsets: torch.Tensor
    valid_route_counts: torch.Tensor


@dataclass(frozen=True)
class MoeEpExecutionLane:
    """Operator-bound index of one mutable per-stream execution lane."""

    index: int
    _operator_token: object


MoeTensor = Union[torch.Tensor, BlockScaledTensor]


__all__ = [
    "BlockScaledTensor",
    "MoeEpExecutionLane",
    "MoeEpBackwardWeightStaging",
    "MoeEpBackwardWeights",
    "MoeEpForwardWeightStaging",
    "MoeEpForwardWeights",
    "MoeEpNativeBackwardWeights",
    "MoeEpNativeForwardWeights",
    "MoeEpNativeWeight",
    "MoeEpNativeWeightLayout",
    "MoeEpTrainingBackwardOutputs",
    "MoeEpTrainingForwardOutputs",
    "MoeEpTrainingWgradOperands",
    "MoeFormat",
    "MoeTensor",
    "parse_format",
]
