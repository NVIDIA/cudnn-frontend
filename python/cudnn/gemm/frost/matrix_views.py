# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checked, metadata-only folding of leading matrix batch axes."""

from math import prod

from cudnn.frost import buffers

_INT64_MAX = (1 << 63) - 1


def matrix_layout(shape, stride, *, writable=False):
    """Return the equivalent (batch, row, column) layout, or reject a non-affine fold.

    Matrix axes remain unchanged. Unit batch axes never contribute an address;
    every other batch axis must follow one common linear batch pitch. Gaps
    between matrices and strided/transposed matrix axes are allowed.
    """
    shape, stride = tuple(map(int, shape)), tuple(map(int, stride))
    if len(shape) < 2 or len(stride) != len(shape) or any(d <= 0 for d in shape) or any(s < 0 for s in stride):
        raise ValueError(f"matrix batch view requires rank >= 2, positive extents and nonnegative strides; got {shape}, {stride}")
    prefix = shape[:-2]
    batch = prod(prefix)
    active = [i for i, dim in enumerate(prefix) if dim != 1]
    pitch = stride[active[-1]] if active else max(shape[-2] * stride[-2], shape[-1] * stride[-1])
    if active and (pitch <= 0 or any(stride[i] != pitch * prod(prefix[i + 1 :]) for i in active)):
        raise ValueError(f"matrix batch axes are not affine and cannot be folded without copying: shape={shape}, stride={stride}")
    span = 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    if max(batch, pitch, span, *shape, *stride) > _INT64_MAX:
        raise ValueError("matrix batch view exceeds signed 64-bit geometry")
    if writable:
        occupied = 1
        for s, d in sorted((s, d) for d, s in zip(shape, stride) if d > 1):
            if s < occupied:
                raise ValueError("matrix output view requires non-overlapping storage")
            occupied += (d - 1) * s
    return (batch, *shape[-2:]), (pitch, *stride[-2:])


def matrix_shape(shape, prefix):
    """Check the original rank and each batch axis, not merely their product."""
    shape = tuple(map(int, shape))
    if len(shape) != len(prefix) + 2 or shape[:-2] != prefix:
        raise ValueError(f"matrix batch view expected original batch axes {prefix}, got shape={shape}")
    return (prod(prefix), *shape[-2:])


def matrix_view(tensor, prefix, *, writable=False):
    """Fold only checked batch axes; never use a copying reshape on caller data."""
    shape = matrix_shape(tensor.shape, prefix)
    matrix_layout(tensor.shape, tensor.stride(), writable=writable)
    if hasattr(tensor, "flatten_batch"):
        return tensor.flatten_batch()  # native OperandBuffer: exact strided metadata
    if isinstance(tensor, buffers.DeviceView):
        return tensor.reshape(shape)  # DeviceView is contiguous by contract
    # Torch's view (unlike reshape) raises when a storage copy would be needed.
    return tensor.view(shape)


def canonical_operands(rules, operands, output_indices=()):
    if not rules:
        return operands
    views = list(operands)
    for index, prefix in rules:
        views[index] = matrix_view(views[index], prefix, writable=index in output_indices)
    return views
