# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Lightweight public tensor and format types for :mod:`cudnn.moe_ep`."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

import torch


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
        raise ValueError(
            f"unsupported format {value!r}; expected one of: {choices}"
        ) from exc


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
            raise ValueError(
                f"data must be a torch.Tensor, got {type(self.data).__name__}"
            )
        if not isinstance(self.scale, torch.Tensor):
            raise ValueError(
                f"scale must be a torch.Tensor, got {type(self.scale).__name__}"
            )
        fmt = parse_format(self.format)
        if fmt is MoeFormat.BF16:
            raise ValueError("BlockScaledTensor only represents mxfp8 or nvfp4")
        if self.data.device != self.scale.device:
            raise ValueError(
                f"data device {self.data.device} does not match "
                f"scale device {self.scale.device}"
            )
        try:
            raw_logical_shape = tuple(self.logical_shape)
        except TypeError as exc:
            raise ValueError(
                "logical_shape must be an iterable of integers"
            ) from exc
        logical_shape = []
        for dim in raw_logical_shape:
            if isinstance(dim, bool):
                raise ValueError(
                    f"logical_shape dimensions must be integers, got {dim!r}"
                )
            try:
                dim = operator.index(dim)
            except TypeError as exc:
                raise ValueError(
                    f"logical_shape dimensions must be integers, got {dim!r}"
                ) from exc
            if dim < 0:
                raise ValueError(
                    f"logical_shape dimensions must be non-negative, got {dim}"
                )
            logical_shape.append(dim)
        normalized_shape = tuple(logical_shape)
        axis = _normalize_axis(self.axis, len(normalized_shape))
        logical_extent = normalized_shape[axis]
        block_size = 32 if fmt is MoeFormat.MXFP8 else 16
        payload_extent = (
            logical_extent
            if fmt is MoeFormat.MXFP8
            else (logical_extent + 1) // 2
        )
        scale_extent = (logical_extent + block_size - 1) // block_size
        expected_data_shape = list(normalized_shape)
        expected_data_shape[axis] = payload_extent
        expected_scale_shape = list(normalized_shape)
        expected_scale_shape[axis] = scale_extent
        expected_data_shape = tuple(expected_data_shape)
        expected_scale_shape = tuple(expected_scale_shape)
        if tuple(self.data.shape) != expected_data_shape:
            raise ValueError(
                f"{fmt.value} data shape must be {expected_data_shape}, "
                f"got {tuple(self.data.shape)}"
            )
        if tuple(self.scale.shape) != expected_scale_shape:
            raise ValueError(
                f"{fmt.value} scale shape must be {expected_scale_shape}, "
                f"got {tuple(self.scale.shape)}"
            )
        e4m3_dtype = getattr(torch, "float8_e4m3fn", None)
        if e4m3_dtype is None:
            raise RuntimeError("this PyTorch build does not provide torch.float8_e4m3fn")
        if fmt is MoeFormat.MXFP8:
            expected_data_dtype = e4m3_dtype
            expected_scale_dtype = getattr(torch, "float8_e8m0fnu", None)
            if expected_scale_dtype is None:
                raise RuntimeError(
                    "this PyTorch build does not provide torch.float8_e8m0fnu"
                )
        else:
            expected_data_dtype = torch.uint8
            expected_scale_dtype = e4m3_dtype
        if self.data.dtype is not expected_data_dtype:
            raise ValueError(
                f"{fmt.value} data must have dtype {expected_data_dtype}, "
                f"got {self.data.dtype}"
            )
        if self.scale.dtype is not expected_scale_dtype:
            raise ValueError(
                f"{fmt.value} scale must have dtype {expected_scale_dtype}, "
                f"got {self.scale.dtype}"
            )
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
            codes = torch.stack((low, high), dim=-1).flatten(-2)[
                ..., :logical_extent
            ]
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


@dataclass(frozen=True)
class MoeEpWgradForwardStash:
    """Caller-owned forward state required to form expert-local wgrads.

    ``fc1_a`` and ``fc1_sfa`` represent the MXFP8 ``x.T`` operand. Valid
    routes for each local expert occupy the beginning of its padded range;
    ``expert_offsets`` contains cumulative padded end offsets and
    ``valid_route_counts`` contains the corresponding unpadded row counts.
    Scale factors use the blocked layout consumed by grouped wgrad, with
    logical 1x32 scaling and physical 128x4 scale tiles.
    ``route_metadata`` is the compact identity table returned by forward,
    using ``(local_expert, src_rank, src_token, src_slot)`` rows. It validates
    that the stash belongs to the matching routed call; it is not padded or
    row-aligned with the operands' K dimension.
    """

    fc1_a: torch.Tensor
    fc1_sfa: torch.Tensor
    expert_offsets: torch.Tensor
    valid_route_counts: torch.Tensor
    route_metadata: torch.Tensor


@dataclass(frozen=True)
class MoeEpWgradOperands:
    """Caller-owned MXFP8 operands for expert-local grouped wgrad GEMMs.

    The represented operations are ``dW1 = fc1_a @ fc1_b`` and
    ``dW2 = fc2_a @ fc2_b``. For total padded route extent ``K``, their
    logical shapes are ``fc1_a=(H,K)``, ``fc1_b=(K,2I)``,
    ``fc2_a=(I,K)``, and ``fc2_b=(K,H)``. Each scale tensor uses grouped
    wgrad's blocked 1x32 layout: ``(round_up(non-K,128), round_up(K/32,4))``.
    The shared expert metadata has the same meaning as in
    :class:`MoeEpWgradForwardStash`.

    Attributes:
        fc1_a: E4M3 data for the FC1 A operand, logically ``x.T`` with shape
            ``(H, K)``. ``x`` is the activation dispatched to each local
            expert. The K dimension concatenates the experts' independently
            padded route ranges.
        fc1_sfa: E8M0 scales for ``fc1_a``. Each logical scale covers 32
            consecutive K elements of one hidden-feature row.
        fc1_b: E4M3 data for the FC1 B operand, logically
            ``dC=[d_gate | d_up]`` with shape ``(K, 2I)``. ``dC`` is the
            gradient of the pre-SwiGLU FC1 accumulator; columns use the public
            gate-then-up order rather than the kernel's internal strip
            interleave.
        fc1_sfb: E8M0 scales for ``fc1_b``. Each logical scale covers 32
            consecutive K rows for one gate/up feature column.
        fc2_a: E4M3 data for the FC2 A operand, logically ``(p*h).T`` with
            shape ``(I, K)``. ``h=SwiGLU(C)`` and ``p`` is the route's FP32
            router score, applied exactly once before column quantization.
        fc2_sfa: E8M0 scales for ``fc2_a``. Each logical scale covers 32
            consecutive K elements of one intermediate-feature row.
        fc2_b: E4M3 data for the FC2 B operand, logically unweighted ``dY``
            with shape ``(K, H)``. ``dY`` is the routed FC2 output gradient.
        fc2_sfb: E8M0 scales for ``fc2_b``. Each logical scale covers 32
            consecutive K rows for one hidden-feature column.
        expert_offsets: Int32 cumulative padded K-end offset for every local
            expert. Adjacent equal offsets represent an empty expert.
        valid_route_counts: Int32 unpadded route count for every local expert;
            valid rows occupy the beginning of each padded expert range.
        route_metadata: Compact Int32 route identity table with columns
            ``(local_expert, src_rank, src_token, src_slot)``. It identifies
            the routed call but is not padded or row-aligned with K.
    """

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
    route_metadata: torch.Tensor


MoeTensor = Union[torch.Tensor, BlockScaledTensor]


__all__ = [
    "BlockScaledTensor",
    "MoeEpWgradForwardStash",
    "MoeEpWgradOperands",
    "MoeFormat",
    "MoeTensor",
    "parse_format",
]
