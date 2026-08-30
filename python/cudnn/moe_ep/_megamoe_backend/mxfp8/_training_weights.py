# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Stable, allocation-free layout staging for pre-quantized training weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..._types import BlockScaledTensor, MoeEpTrainingWeights
from ._adapter import Mxfp8Weights


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _empty_k_major_like(tensor: torch.Tensor) -> torch.Tensor:
    experts, reduction, output = tensor.shape
    return torch.empty_strided(
        tensor.shape,
        (reduction * output, 1, reduction),
        dtype=tensor.dtype,
        device=tensor.device,
    )


def _empty_blocked_scales(
    source: BlockScaledTensor,
    *,
    raw_rows: int,
    raw_columns: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    experts = source.data.shape[0]
    packed_bytes = _round_up(raw_rows, 128) * _round_up(raw_columns, 4)
    return torch.empty(
        (experts, packed_bytes),
        dtype=torch.uint8,
        device=source.device,
    ).view(dtype)


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
    source_view = (
        source.view(torch.uint8)
        .view(experts, reduction_blocks // 4, 4, 2, raw_rows // 128, 2, 32)
        .permute(0, 4, 1, 6, 5, 3, 2)
    )
    target.view(torch.uint8).view(
        experts, raw_rows // 128, reduction_blocks // 4, 32, 2, 2, 4
    ).copy_(source_view)


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
    source_view = (
        source.view(torch.uint8)
        .view(experts, 2, reduction_blocks // 2, 2, output // 128, 4, 32)
        .permute(0, 4, 2, 6, 5, 3, 1)
    )
    target.view(torch.uint8).view(
        experts, output // 128, reduction_blocks // 2, 32, 4, 2, 2
    ).copy_(source_view)


@dataclass(frozen=True)
class Mxfp8BackwardWeights:
    """Kernel names follow the two backward FC stages."""

    fc1_weight: torch.Tensor
    fc1_weight_sf: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_sf: torch.Tensor


class Mxfp8TrainingWeightBindings:
    """Direct data bindings with persistent kernel-native scale staging."""

    def __init__(self, weights: MoeEpTrainingWeights) -> None:
        self.weights = weights
        fwd_fc1 = weights.forward_fc1
        fwd_fc2 = weights.forward_fc2
        bwd_w2t = weights.backward_w2_transpose
        bwd_w1t = weights.backward_w1_transpose
        self._uses_direct_weight_bindings = (
            fwd_fc1.data.stride(1) == 1
            and fwd_fc2.data.stride(1) == 1
            and bwd_w2t.data.is_contiguous()
            and bwd_w1t.data.is_contiguous()
        )

        self.forward = Mxfp8Weights(
            fc1_weight=(
                fwd_fc1.data
                if self._uses_direct_weight_bindings
                else _empty_k_major_like(fwd_fc1.data)
            ),
            fc1_weight_sf=_empty_blocked_scales(
                fwd_fc1,
                raw_rows=fwd_fc1.data.shape[2],
                raw_columns=fwd_fc1.data.shape[1] // 32,
                dtype=torch.uint8,
            ),
            fc2_weight=(
                fwd_fc2.data
                if self._uses_direct_weight_bindings
                else _empty_k_major_like(fwd_fc2.data)
            ),
            fc2_weight_sf=_empty_blocked_scales(
                fwd_fc2,
                raw_rows=fwd_fc2.data.shape[2],
                raw_columns=fwd_fc2.data.shape[1] // 32,
                dtype=torch.uint8,
            ),
        )
        self.backward = Mxfp8BackwardWeights(
            fc1_weight=(
                bwd_w2t.data
                if self._uses_direct_weight_bindings
                else torch.empty_like(bwd_w2t.data)
            ),
            fc1_weight_sf=_empty_blocked_scales(
                bwd_w2t,
                raw_rows=bwd_w2t.data.shape[2],
                raw_columns=bwd_w2t.data.shape[1] // 32,
                dtype=torch.float8_e8m0fnu,
            ),
            fc2_weight=(
                bwd_w1t.data
                if self._uses_direct_weight_bindings
                else torch.empty_like(bwd_w1t.data)
            ),
            fc2_weight_sf=_empty_blocked_scales(
                bwd_w1t,
                raw_rows=bwd_w1t.data.shape[2],
                raw_columns=bwd_w1t.data.shape[1] // 32,
                dtype=torch.float8_e8m0fnu,
            ),
        )
        self.refresh()

    def refresh(self) -> None:
        """Enqueue fixed-address layout copies; safe to record in a graph."""

        fwd_fc1 = self.weights.forward_fc1
        fwd_fc2 = self.weights.forward_fc2
        bwd_w2t = self.weights.backward_w2_transpose
        bwd_w1t = self.weights.backward_w1_transpose

        intermediate = fwd_fc2.data.shape[1]
        if self._uses_direct_weight_bindings:
            _copy_blocked_scales_plain(
                self.forward.fc1_weight_sf,
                fwd_fc1.scale,
                raw_rows=fwd_fc1.data.shape[2],
                raw_columns=fwd_fc1.data.shape[1] // 32,
            )
        else:
            _copy_gate_up_interleaved_last(
                self.forward.fc1_weight,
                fwd_fc1.data,
                intermediate,
            )
            _copy_blocked_scales_gate_up_rows(
                self.forward.fc1_weight_sf,
                fwd_fc1.scale,
                intermediate=intermediate,
                reduction_blocks=fwd_fc1.data.shape[1] // 32,
            )
            self.forward.fc2_weight.copy_(fwd_fc2.data)
        _copy_blocked_scales_plain(
            self.forward.fc2_weight_sf,
            fwd_fc2.scale,
            raw_rows=fwd_fc2.data.shape[2],
            raw_columns=fwd_fc2.data.shape[1] // 32,
        )

        if not self._uses_direct_weight_bindings:
            self.backward.fc1_weight.copy_(bwd_w2t.data)
        _copy_blocked_scales_plain(
            self.backward.fc1_weight_sf,
            bwd_w2t.scale,
            raw_rows=bwd_w2t.data.shape[2],
            raw_columns=bwd_w2t.data.shape[1] // 32,
        )
        if self._uses_direct_weight_bindings:
            _copy_blocked_scales_plain(
                self.backward.fc2_weight_sf,
                bwd_w1t.scale,
                raw_rows=bwd_w1t.data.shape[2],
                raw_columns=bwd_w1t.data.shape[1] // 32,
            )
        else:
            _copy_gate_up_interleaved_reduction(
                self.backward.fc2_weight,
                bwd_w1t.data,
                intermediate,
            )
            _copy_blocked_scales_gate_up_columns(
                self.backward.fc2_weight_sf,
                bwd_w1t.scale,
                intermediate=intermediate,
                output=bwd_w1t.data.shape[2],
            )


__all__ = [
    "Mxfp8BackwardWeights",
    "Mxfp8TrainingWeightBindings",
]
