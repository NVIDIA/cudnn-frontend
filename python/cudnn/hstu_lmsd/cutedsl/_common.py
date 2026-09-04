# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and device helpers for the HSTU LMSD kernels."""

import math
import struct

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

ALIGNMENT_BYTES = 16
LOG2E = 1.4426950408889634

MASK_LMSD = 0x1
MASK_X = 0x2
MASK_SILU = 0x4


def round_to_float32(value: float) -> float:
    """Round a host scalar exactly as a CuTe DSL Float32 argument."""
    return struct.unpack("f", struct.pack("f", value))[0]


def normalize_dropout_ratio(dropout_ratio: float) -> float:
    """Validate and normalize a dropout ratio to its kernel representation."""
    raw_dropout_ratio = float(dropout_ratio)
    if not math.isfinite(raw_dropout_ratio) or not 0.0 <= raw_dropout_ratio < 1.0:
        raise ValueError(f"dropout_ratio must be finite and in [0, 1), got {raw_dropout_ratio}")
    normalized_dropout_ratio = round_to_float32(raw_dropout_ratio)
    if not 0.0 <= normalized_dropout_ratio < 1.0:
        raise ValueError("dropout_ratio must remain in [0, 1) after float32 conversion, " f"got {raw_dropout_ratio} -> {normalized_dropout_ratio}")
    return normalized_dropout_ratio


def keep_threshold32(dropout_ratio: float) -> int:
    """Return the first uint32 Philox sample kept by the dropout comparison."""
    dropout_ratio_f32 = normalize_dropout_ratio(dropout_ratio)
    scale = round_to_float32(2.0**-32)
    low, high = 0, 1 << 32
    while low < high:
        midpoint = (low + high) // 2
        if round_to_float32(round_to_float32(midpoint) * scale) > dropout_ratio_f32:
            high = midpoint
        else:
            low = midpoint + 1
    return low


@dsl_user_op
def domain_offset_i64(
    coord: cute.Coord,
    tensor: cute.Tensor,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    """Rebase a tensor with a 64-bit byte offset, preserving its layout."""
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride)
    element_offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + element_offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)
