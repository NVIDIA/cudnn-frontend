# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constants and device helpers for the HSTU LMSD kernels."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op

ALIGNMENT_BYTES = 16
LOG2E = 1.4426950408889634

MASK_LMSD = 0x1
MASK_X = 0x2
MASK_SILU = 0x4


def keep_threshold32(dropout_ratio: float) -> int:
    """Return the first uint32 Philox sample kept by the dropout comparison."""
    import struct

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    dropout_ratio_f32 = f32(dropout_ratio)
    scale = f32(2.0**-32)
    low, high = 0, 1 << 32
    while low < high:
        midpoint = (low + high) // 2
        if f32(f32(midpoint) * scale) > dropout_ratio_f32:
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
