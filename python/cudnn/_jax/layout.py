# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Dependency-free axis and layout conversion for CUTLASS JAX calls."""

from __future__ import annotations

from operator import index
from typing import TypeVar

AxisValueT = TypeVar("AxisValueT")


def mode_from_layout(layout: str, *, kernel_axes: str) -> tuple[int, ...]:
    """Map named kernel axes to their positions in a public JAX layout.

    Axis names are single, case-sensitive characters. The returned mapping
    follows ``mode[kernel_axis] = public_axis``.
    """

    if not isinstance(layout, str):
        raise TypeError(f"layout must be a string, got {layout!r}")
    if not isinstance(kernel_axes, str):
        raise TypeError(f"kernel_axes must be a string, got {kernel_axes!r}")
    if len(layout) != len(kernel_axes):
        raise ValueError(
            f"layout rank must match kernel_axes: got {layout!r} and {kernel_axes!r}"
        )
    if len(set(layout)) != len(layout):
        raise ValueError(f"layout axes must be unique, got {layout!r}")
    if len(set(kernel_axes)) != len(kernel_axes):
        raise ValueError(f"kernel_axes must be unique, got {kernel_axes!r}")
    if set(layout) != set(kernel_axes):
        raise ValueError(
            f"layout must contain exactly the axes in {kernel_axes!r}, got {layout!r}"
        )
    return tuple(layout.index(axis) for axis in kernel_axes)


def normalize_mode(rank: int, mode: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """Return and validate a kernel-axis to public-axis mapping.

    ``mode[kernel_axis]`` is the corresponding axis in the public JAX array.
    Omitting ``mode`` keeps the public and kernel axis orders identical.
    """

    if isinstance(rank, bool):
        raise TypeError(f"rank must be an integer, got {rank!r}")
    try:
        rank = index(rank)
    except TypeError as error:
        raise TypeError(f"rank must be an integer, got {rank!r}") from error
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")

    if mode is None:
        return tuple(range(rank))

    normalized = []
    for axis in mode:
        if isinstance(axis, bool):
            raise TypeError(f"mode entries must be integers, got {axis!r}")
        try:
            normalized.append(index(axis))
        except TypeError as error:
            raise TypeError(f"mode entries must be integers, got {axis!r}") from error
    normalized_mode = tuple(normalized)
    if tuple(sorted(normalized_mode)) != tuple(range(rank)):
        raise ValueError(
            f"mode must be a permutation of [0, {rank - 1}], got {normalized_mode}"
        )
    return normalized_mode


def to_canonical_axes(
    public_values: tuple[AxisValueT, ...],
    mode: tuple[int, ...] | None = None,
) -> tuple[AxisValueT, ...]:
    """Reorder public-axis values into canonical kernel-axis order."""

    public_values = tuple(public_values)
    mode = normalize_mode(len(public_values), mode)
    return tuple(public_values[public_axis] for public_axis in mode)


def to_public_axes(
    canonical_values: tuple[AxisValueT, ...],
    mode: tuple[int, ...] | None = None,
) -> tuple[AxisValueT, ...]:
    """Reorder canonical kernel-axis values into public JAX axis order."""

    canonical_values = tuple(canonical_values)
    mode = normalize_mode(len(canonical_values), mode)
    canonical_axis_by_public_axis = [0] * len(mode)
    for canonical_axis, public_axis in enumerate(mode):
        canonical_axis_by_public_axis[public_axis] = canonical_axis
    return tuple(
        canonical_values[canonical_axis_by_public_axis[public_axis]]
        for public_axis in range(len(mode))
    )


def compact_stride(
    shape: tuple[int, ...], stride_order: tuple[int, ...]
) -> tuple[int, ...]:
    """Return compact strides for dimensions ordered fastest to slowest."""

    stride = [0] * len(shape)
    running = 1
    for dimension in stride_order:
        stride[dimension] = running
        running *= max(shape[dimension], 1)
    return tuple(stride)


def stride_order_to_public(
    canonical_stride_order: tuple[int, ...],
    mode: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Map a fastest-to-slowest canonical axis order to public array axes."""

    canonical_stride_order = tuple(canonical_stride_order)
    mode = normalize_mode(len(canonical_stride_order), mode)
    if tuple(sorted(canonical_stride_order)) != tuple(
        range(len(canonical_stride_order))
    ):
        raise ValueError(
            "canonical_stride_order must be a permutation of "
            f"[0, {len(canonical_stride_order) - 1}], got {canonical_stride_order}"
        )
    return tuple(mode[canonical_axis] for canonical_axis in canonical_stride_order)


def to_cutlass_layout(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    stride_order: tuple[int, ...],
    *,
    mode: tuple[int, ...] | None = None,
    name: str = "tensor",
) -> tuple[int, ...]:
    """Return CUTLASS stride ranks indexed by public JAX array axes.

    The descriptor arguments are in canonical kernel-axis order. ``mode``
    maps those axes back to the public array axes consumed by TensorSpec.
    """

    if stride != compact_stride(shape, stride_order):
        raise ValueError(
            f"JAX TensorSpec cannot represent non-compact stride {stride} for {name}"
        )

    layout = [0] * len(stride_order)
    for rank, dimension in enumerate(stride_order):
        layout[dimension] = rank
    return to_public_axes(tuple(layout), mode)


__all__ = [
    "compact_stride",
    "mode_from_layout",
    "normalize_mode",
    "stride_order_to_public",
    "to_canonical_axes",
    "to_cutlass_layout",
    "to_public_axes",
]
