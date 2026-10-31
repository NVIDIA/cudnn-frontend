# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure matrix storage geometry, independent of dtypes and runtime buffers."""

from math import prod


def fold_batch_layout(shape, stride, *, writable=False, allow_identity_broadcast=False):
    """Return the equivalent (batch, row, column) layout, or reject a non-affine fold.

    Matrix axes remain unchanged. Unit batch axes never contribute an address;
    every other batch axis must follow one common linear batch pitch. Gaps
    between matrices and strided/transposed matrix axes are allowed.
    ``allow_identity_broadcast`` permits zero batch pitch only when no rank
    folding is needed; writable storage must still be non-overlapping.
    """
    shape, stride = tuple(map(int, shape)), tuple(map(int, stride))
    if len(shape) < 2 or len(stride) != len(shape) or any(d <= 0 for d in shape) or any(s < 0 for s in stride):
        raise ValueError(f"matrix batch view requires rank >= 2, positive extents and nonnegative strides; got {shape}, {stride}")
    prefix = shape[:-2]
    batch = prod(prefix)
    active = [i for i, dim in enumerate(prefix) if dim != 1]
    pitch = stride[active[-1]] if active else stride[len(prefix) - 1] if prefix else max(shape[-2] * stride[-2], shape[-1] * stride[-1])
    zero_pitch_allowed = allow_identity_broadcast and len(shape) == 3
    if active and ((pitch == 0 and not zero_pitch_allowed) or any(stride[i] != pitch * prod(prefix[i + 1 :]) for i in active)):
        raise ValueError(f"matrix batch axes are not affine and cannot be folded without copying: shape={shape}, stride={stride}")
    span = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    if max(batch, pitch, span, *shape, *stride) > (1 << 63) - 1:
        raise ValueError("matrix batch view exceeds signed 64-bit geometry")
    if writable:
        occupied = 1
        for s, d in sorted((s, d) for d, s in zip(shape, stride) if d > 1):
            if s < occupied:
                raise ValueError("matrix output view requires non-overlapping storage")
            occupied += (d - 1) * s
    return (batch, *shape[-2:]), (pitch, *stride[-2:])


def reshape_layout(shape, stride, target):
    """Compute a lexicographic zero-copy reshape using contiguous stride chunks.

    Unit axes do not break a chunk. Matrix padding and noncontiguous layouts
    are accepted only when the requested shape does not merge across a gap.
    No tensor/framework calls, allocation or address dereference is involved.
    """
    shape, stride, target = tuple(map(int, shape)), tuple(map(int, stride)), tuple(map(int, target))
    limit = (1 << 63) - 1
    if not shape or not target or len(shape) != len(stride) or any(d <= 0 for d in (*shape, *target)) or any(s < 0 for s in stride):
        raise ValueError("reshape view requires positive extents and nonnegative strides")
    if prod(shape) != prod(target):
        raise ValueError("reshape view must preserve element count")
    if max(prod(shape), 1 + sum((d - 1) * s for d, s in zip(shape, stride)), *shape, *target, *stride) > limit:
        raise ValueError("reshape view exceeds signed 64-bit geometry")
    if shape == target:
        return target, stride
    result = [0] * len(target)
    axis = len(target) - 1
    chunk_stride, chunk_size, view_size = stride[-1], 1, 1
    for i in range(len(shape) - 1, -1, -1):
        chunk_size *= shape[i]
        boundary = i == 0 or (shape[i - 1] != 1 and stride[i - 1] != chunk_size * chunk_stride)
        if not boundary:
            continue
        while axis >= 0 and (view_size < chunk_size or target[axis] == 1):
            result[axis] = view_size * chunk_stride
            view_size *= target[axis]
            axis -= 1
        if view_size != chunk_size:
            raise ValueError("reshape view would require copying across a stride gap")
        if i > 0:
            chunk_stride, chunk_size, view_size = stride[i - 1], 1, 1
    if axis != -1 or max(result) > limit:
        raise ValueError("reshape view cannot represent requested shape in signed 64-bit geometry")
    return target, tuple(result)


def operand_view_layout(shape, stride, steps):
    """Apply a prepared sequence of logical reshape and last-two-axis views."""
    shape, stride = tuple(shape), tuple(stride)
    for kind, argument, declared_stride in steps:
        if kind == "reshape":
            shape, stride = reshape_layout(shape, stride, argument)
            if declared_stride and any(d > 1 and x != y for d, x, y in zip(shape, stride, declared_stride)):
                raise ValueError("reshape declared strides do not match a logical zero-copy view")
        elif kind == "transpose":
            shape, stride = tuple(shape[i] for i in argument), tuple(stride[i] for i in argument)
        else:
            raise ValueError(f"unsupported operand view {kind!r}")
    return shape, stride
