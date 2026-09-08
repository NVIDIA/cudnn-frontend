# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stable, allocation-free layout staging for pre-quantized training weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._contracts import Fc1WeightLayout
from ..._math import round_up
from ..._types import (
    BlockScaledTensor,
    MoeEpBackwardWeightStaging,
    MoeEpBackwardWeights,
    MoeEpForwardWeightStaging,
    MoeEpForwardWeights,
    MoeEpNativeBackwardWeights,
    MoeEpNativeForwardWeights,
    MoeEpNativeWeight,
    MoeEpNativeWeightLayout,
)
from ..._validation import validate_training_non_aliasing
from ._adapter import Mxfp8Weights


def _copy_blocked_scales_plain(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    raw_rows: int,
    raw_columns: int,
) -> None:
    """Pack public ``(E,Kblocks,N)`` scales for a non-interleaved weight."""

    experts = source.shape[0]
    if tuple(source.shape) != (experts, raw_columns, raw_rows):
        raise ValueError("plain training scale shape mismatch: " f"{tuple(source.shape)} != " f"{(experts, raw_columns, raw_rows)}")
    if raw_rows % 128 or raw_columns % 4:
        raise ValueError("training scale pack requires rows divisible by 128 and " "columns divisible by 4")
    row_blocks = raw_rows // 128
    column_blocks = raw_columns // 4
    source_view = (
        source.view(
            torch.uint8,
        )
        .view(
            experts,
            column_blocks,
            4,
            row_blocks,
            4,
            32,
        )
        .permute(0, 3, 1, 5, 4, 2)
    )
    target.view(torch.uint8).view(
        experts,
        row_blocks,
        column_blocks,
        32,
        4,
        4,
    ).copy_(source_view)


def _copy_gate_up_interleaved_last(
    target: torch.Tensor,
    source: torch.Tensor,
    intermediate: int,
) -> None:
    """Copy ``(E,K,gate||up)`` into 32-column gate/up strip order."""

    experts, reduction, gate_up = source.shape
    if target.shape != source.shape or gate_up != 2 * intermediate:
        raise ValueError("forward FC1 training weight shape mismatch")
    pairs = intermediate // 32
    source_view = source.view(experts, reduction, 2, pairs, 32).permute(0, 1, 3, 2, 4)
    target_view = target.as_strided(
        (experts, reduction, pairs, 2, 32),
        (
            target.stride(0),
            target.stride(1),
            64 * target.stride(2),
            32 * target.stride(2),
            target.stride(2),
        ),
    )
    target_view.copy_(source_view)


def _copy_gate_up_interleaved_reduction(
    target: torch.Tensor,
    source: torch.Tensor,
    intermediate: int,
) -> None:
    """Copy ``(E,gate||up,N)`` into 32-row gate/up strip order."""

    experts, gate_up, output = source.shape
    if target.shape != source.shape or gate_up != 2 * intermediate:
        raise ValueError("backward W1-transpose training weight shape mismatch")
    pairs = intermediate // 32
    source_view = source.view(experts, 2, pairs, 32, output).permute(0, 2, 1, 3, 4)
    target.view(experts, pairs, 2, 32, output).copy_(source_view)


def _copy_blocked_scales_gate_up_rows(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    intermediate: int,
    reduction_blocks: int,
) -> None:
    """Pack forward FC1 scales after 32-row gate/up interleave."""

    experts = source.shape[0]
    raw_rows = 2 * intermediate
    if intermediate % 64 or reduction_blocks % 4:
        raise ValueError("intermediate and reduction block alignment are invalid")
    source_view = source.view(torch.uint8).view(experts, reduction_blocks // 4, 4, 2, raw_rows // 128, 2, 32).permute(0, 4, 1, 6, 5, 3, 2)
    target.view(torch.uint8).view(experts, raw_rows // 128, reduction_blocks // 4, 32, 2, 2, 4).copy_(source_view)


def _copy_blocked_scales_gate_up_columns(
    target: torch.Tensor,
    source: torch.Tensor,
    *,
    intermediate: int,
    output: int,
) -> None:
    """Pack backward W1-transpose scales with interleaved K blocks."""

    experts = source.shape[0]
    reduction_blocks = intermediate // 32
    if output % 128 or reduction_blocks % 2:
        raise ValueError("output and reduction block alignment are invalid")
    source_view = source.view(torch.uint8).view(experts, 2, reduction_blocks // 2, 2, output // 128, 4, 32).permute(0, 4, 2, 6, 5, 3, 1)
    target.view(torch.uint8).view(experts, output // 128, reduction_blocks // 2, 32, 4, 2, 2).copy_(source_view)


@dataclass(frozen=True)
class Mxfp8BackwardWeights:
    """Kernel names follow the two backward FC stages."""

    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor


def _expect_staging_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout is not torch.strided:
        raise ValueError(f"{name} must use torch.strided layout, got {tensor.layout}")
    if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride:
        raise ValueError(f"{name} must have shape={shape}, stride={stride}; got " f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}")
    if tensor.dtype is not dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be at least 16-byte aligned")


def _expect_scale_staging(
    name: str,
    tensor: torch.Tensor,
    *,
    experts: int,
    raw_rows: int,
    raw_columns: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    elements = round_up(raw_rows, 128) * round_up(raw_columns, 4)
    _expect_staging_tensor(
        name,
        tensor,
        shape=(experts, elements),
        stride=(elements, 1),
        dtype=dtype,
        device=device,
    )


def forward_native_to_kernel(weights: MoeEpNativeForwardWeights) -> Mxfp8Weights:
    """Create kernel views without allocating, copying, or retaining inputs."""

    return Mxfp8Weights(
        fc1_weight=weights.fc1.payload,
        fc1_weight_sf=weights.fc1.scale,
        fc2_weight=weights.fc2.payload,
        fc2_weight_sf=weights.fc2.scale,
    )


def backward_native_to_kernel(
    weights: MoeEpNativeBackwardWeights,
) -> Mxfp8BackwardWeights:
    """Create kernel views without allocating, copying, or retaining inputs."""

    return Mxfp8BackwardWeights(
        fc1_weight=weights.w2_transpose.payload,
        fc1_weight_sf=weights.w2_transpose.scale,
        fc2_weight=weights.w1_transpose.payload,
        fc2_weight_sf=weights.w1_transpose.scale,
    )


def materialize_forward(
    weights: MoeEpForwardWeights,
    *,
    out: MoeEpForwardWeightStaging,
    fc1_weight_layout: Fc1WeightLayout,
) -> MoeEpNativeForwardWeights:
    """Materialize source forward weights into caller-owned native storage."""

    if fc1_weight_layout is not Fc1WeightLayout.GATE_UP_INTERLEAVED_32:
        raise ValueError("native training materialization requires weight_interleave_size=32")
    if not isinstance(weights, MoeEpForwardWeights):
        raise TypeError("weights must be a MoeEpForwardWeights")
    if not isinstance(out, MoeEpForwardWeightStaging):
        raise TypeError("out must be a MoeEpForwardWeightStaging")
    fc1 = weights.fc1
    fc2 = weights.fc2
    for name, value in (("weights.fc1", fc1), ("weights.fc2", fc2)):
        if not isinstance(value, BlockScaledTensor) or value.format.value != "mxfp8" or value.axis != 1:
            raise TypeError(f"{name} must be an axis-1 MXFP8 BlockScaledTensor")
        if not value.data.is_contiguous() or not value.scale.is_contiguous():
            raise ValueError(f"{name} data and scale must be contiguous")
    experts, hidden, gate_up = fc1.data.shape
    intermediate = fc2.data.shape[1]
    if tuple(fc2.data.shape) != (experts, intermediate, hidden) or gate_up != 2 * intermediate:
        raise ValueError("forward source weight shapes are inconsistent")
    if fc2.device != fc1.device:
        raise ValueError("forward source weights must share one device")
    sf_dtype = torch.float8_e8m0fnu
    _expect_staging_tensor(
        "out.fc1_payload",
        out.fc1_payload,
        shape=tuple(fc1.data.shape),
        stride=(hidden * gate_up, 1, hidden),
        dtype=fc1.data.dtype,
        device=fc1.device,
    )
    _expect_staging_tensor(
        "out.fc2_payload",
        out.fc2_payload,
        shape=tuple(fc2.data.shape),
        stride=(intermediate * hidden, 1, intermediate),
        dtype=fc2.data.dtype,
        device=fc1.device,
    )
    _expect_scale_staging(
        "out.fc1_scale",
        out.fc1_scale,
        experts=experts,
        raw_rows=gate_up,
        raw_columns=hidden // 32,
        dtype=sf_dtype,
        device=fc1.device,
    )
    _expect_scale_staging(
        "out.fc2_scale",
        out.fc2_scale,
        experts=experts,
        raw_rows=hidden,
        raw_columns=intermediate // 32,
        dtype=sf_dtype,
        device=fc1.device,
    )
    validate_training_non_aliasing(
        {
            "weights.fc1.data": fc1.data,
            "weights.fc1.scale": fc1.scale,
            "weights.fc2.data": fc2.data,
            "weights.fc2.scale": fc2.scale,
            "out.fc1_payload": out.fc1_payload,
            "out.fc1_scale": out.fc1_scale,
            "out.fc2_payload": out.fc2_payload,
            "out.fc2_scale": out.fc2_scale,
        }
    )
    _copy_gate_up_interleaved_last(out.fc1_payload, fc1.data, intermediate)
    _copy_blocked_scales_gate_up_rows(
        out.fc1_scale,
        fc1.scale,
        intermediate=intermediate,
        reduction_blocks=hidden // 32,
    )
    out.fc2_payload.copy_(fc2.data)
    _copy_blocked_scales_plain(
        out.fc2_scale,
        fc2.scale,
        raw_rows=hidden,
        raw_columns=intermediate // 32,
    )
    return MoeEpNativeForwardWeights(
        fc1=MoeEpNativeWeight(
            out.fc1_payload,
            out.fc1_scale,
            MoeEpNativeWeightLayout.FORWARD_FC1_GATE_UP_INTERLEAVED_32_V1,
        ),
        fc2=MoeEpNativeWeight(
            out.fc2_payload,
            out.fc2_scale,
            MoeEpNativeWeightLayout.FORWARD_FC2_K_MAJOR_V1,
        ),
    )


def materialize_backward(
    weights: MoeEpBackwardWeights,
    *,
    out: MoeEpBackwardWeightStaging,
    fc1_weight_layout: Fc1WeightLayout,
) -> MoeEpNativeBackwardWeights:
    """Materialize source backward weights into caller-owned native storage."""

    if fc1_weight_layout is not Fc1WeightLayout.GATE_UP_INTERLEAVED_32:
        raise ValueError("native training materialization requires weight_interleave_size=32")
    if not isinstance(weights, MoeEpBackwardWeights):
        raise TypeError("weights must be a MoeEpBackwardWeights")
    if not isinstance(out, MoeEpBackwardWeightStaging):
        raise TypeError("out must be a MoeEpBackwardWeightStaging")
    w2t = weights.w2_transpose
    w1t = weights.w1_transpose
    for name, value in (
        ("weights.w2_transpose", w2t),
        ("weights.w1_transpose", w1t),
    ):
        if not isinstance(value, BlockScaledTensor) or value.format.value != "mxfp8" or value.axis != 1:
            raise TypeError(f"{name} must be an axis-1 MXFP8 BlockScaledTensor")
        if not value.data.is_contiguous() or not value.scale.is_contiguous():
            raise ValueError(f"{name} data and scale must be contiguous")
    experts, hidden, intermediate = w2t.data.shape
    if tuple(w1t.data.shape) != (experts, 2 * intermediate, hidden):
        raise ValueError("backward source weight shapes are inconsistent")
    if w1t.device != w2t.device:
        raise ValueError("backward source weights must share one device")
    _expect_staging_tensor(
        "out.w2_transpose_payload",
        out.w2_transpose_payload,
        shape=tuple(w2t.data.shape),
        stride=(hidden * intermediate, intermediate, 1),
        dtype=w2t.data.dtype,
        device=w2t.device,
    )
    _expect_staging_tensor(
        "out.w1_transpose_payload",
        out.w1_transpose_payload,
        shape=tuple(w1t.data.shape),
        stride=(2 * intermediate * hidden, hidden, 1),
        dtype=w1t.data.dtype,
        device=w2t.device,
    )
    sf_dtype = torch.float8_e8m0fnu
    _expect_scale_staging(
        "out.w2_transpose_scale",
        out.w2_transpose_scale,
        experts=experts,
        raw_rows=intermediate,
        raw_columns=hidden // 32,
        dtype=sf_dtype,
        device=w2t.device,
    )
    _expect_scale_staging(
        "out.w1_transpose_scale",
        out.w1_transpose_scale,
        experts=experts,
        raw_rows=hidden,
        raw_columns=2 * intermediate // 32,
        dtype=sf_dtype,
        device=w2t.device,
    )
    validate_training_non_aliasing(
        {
            "weights.w2_transpose.data": w2t.data,
            "weights.w2_transpose.scale": w2t.scale,
            "weights.w1_transpose.data": w1t.data,
            "weights.w1_transpose.scale": w1t.scale,
            "out.w2_transpose_payload": out.w2_transpose_payload,
            "out.w2_transpose_scale": out.w2_transpose_scale,
            "out.w1_transpose_payload": out.w1_transpose_payload,
            "out.w1_transpose_scale": out.w1_transpose_scale,
        }
    )
    out.w2_transpose_payload.copy_(w2t.data)
    _copy_blocked_scales_plain(
        out.w2_transpose_scale,
        w2t.scale,
        raw_rows=intermediate,
        raw_columns=hidden // 32,
    )
    _copy_gate_up_interleaved_reduction(
        out.w1_transpose_payload,
        w1t.data,
        intermediate,
    )
    _copy_blocked_scales_gate_up_columns(
        out.w1_transpose_scale,
        w1t.scale,
        intermediate=intermediate,
        output=hidden,
    )
    return MoeEpNativeBackwardWeights(
        w2_transpose=MoeEpNativeWeight(
            out.w2_transpose_payload,
            out.w2_transpose_scale,
            MoeEpNativeWeightLayout.BACKWARD_W2_TRANSPOSE_V1,
        ),
        w1_transpose=MoeEpNativeWeight(
            out.w1_transpose_payload,
            out.w1_transpose_scale,
            MoeEpNativeWeightLayout.BACKWARD_W1_TRANSPOSE_GATE_UP_INTERLEAVED_32_V1,
        ),
    )


__all__ = [
    "Mxfp8BackwardWeights",
    "backward_native_to_kernel",
    "forward_native_to_kernel",
    "materialize_backward",
    "materialize_forward",
]
