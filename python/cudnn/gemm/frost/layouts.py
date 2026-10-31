# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure strided-view geometry, independent of dtypes and runtime buffers."""

from dataclasses import dataclass, replace
from math import prod
from operator import index


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
    shape, stride, target = tuple(map(index, shape)), tuple(map(index, stride)), tuple(map(index, target))
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


def transpose_layout(shape, stride, permutation):
    """Permute any logical axes; matrix-engine restrictions belong to lowering."""
    shape, stride, permutation = tuple(map(index, shape)), tuple(map(index, stride)), tuple(map(index, permutation))
    if not shape or len(shape) != len(stride) or sorted(permutation) != list(range(len(shape))):
        raise ValueError("transpose view requires a full permutation and matching shape/strides")
    if any(d <= 0 for d in shape) or any(s < 0 for s in stride):
        raise ValueError("transpose view requires positive extents and nonnegative strides")
    if max(*shape, *stride, prod(shape), 1 + sum((d - 1) * s for d, s in zip(shape, stride))) > (1 << 63) - 1:
        raise ValueError("transpose view exceeds signed 64-bit geometry")
    return tuple(shape[i] for i in permutation), tuple(stride[i] for i in permutation)


@dataclass(frozen=True)
class ViewStep:
    """One logical view, including optional declared reshape strides.

    Reshape preserves lexicographic element order; transpose preserves axes by
    permutation. Positive extents and nonnegative strides are the supported
    geometry domain. Zero-element, inferred and negative-stride views decline.
    """

    kind: str
    argument: tuple[int, ...]
    declared_stride: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "argument", tuple(map(index, self.argument)))
        object.__setattr__(self, "declared_stride", tuple(map(index, self.declared_stride)))
        if self.kind not in ("reshape", "transpose"):
            raise ValueError(f"unsupported operand view {self.kind!r}")
        if self.kind == "transpose" and self.declared_stride:
            raise ValueError("transpose view cannot declare reshape strides")
        if self.declared_stride and (len(self.declared_stride) != len(self.argument) or any(s < 0 for s in self.declared_stride)):
            raise ValueError("reshape declared strides must match target rank and be nonnegative")

    def layout(self, shape, stride):
        if self.kind == "transpose":
            return transpose_layout(shape, stride, self.argument)
        shape, stride = reshape_layout(shape, stride, self.argument)
        if self.declared_stride and any(d > 1 and x != y for d, x, y in zip(shape, stride, self.declared_stride)):
            raise ValueError("reshape declared strides do not match a logical zero-copy view")
        return shape, stride


@dataclass(frozen=True)
class OperandView:
    """Prepared index-map composition, never a pointer or storage owner.

    Every intermediate step is proved at prepare time. Matching runtime layout
    reuses the result; a different stride is re-proved through every step, not
    merely checked against the final storage span. Source shape stays fixed.
    """

    source_shape: tuple[int, ...]
    source_stride: tuple[int, ...]
    steps: tuple[ViewStep, ...]
    target_shape: tuple[int, ...]
    target_stride: tuple[int, ...]
    batch_prefix: tuple[int, ...] | None = None
    alternate_steps: tuple[tuple[ViewStep, ...], ...] = ()

    @classmethod
    def prepare(cls, shape, stride, steps, *, fold_batch=False):
        shape, stride, steps = tuple(map(index, shape)), tuple(map(index, stride)), tuple(steps)
        target, target_stride = reshape_layout(shape, stride, shape)
        for step in steps:
            target, target_stride = step.layout(target, target_stride)
        prefix = target[:-2] if fold_batch else None
        if fold_batch:
            target, target_stride = fold_batch_layout(target, target_stride)
        return cls(shape, stride, steps, target, target_stride, prefix)

    def folded(self):
        """Include the matrix ABI fold in the single prepared runtime view."""
        return replace(self.prepare(self.source_shape, self.source_stride, self.steps, fold_batch=True), alternate_steps=self.alternate_steps)

    def merged(self, other):
        """Deduplicate equivalent storage maps, retaining every path's legality."""
        if (self.source_shape, self.source_stride, self.target_shape, self.target_stride) != (
            other.source_shape,
            other.source_stride,
            other.target_shape,
            other.target_stride,
        ):
            raise ValueError("shared reshape input requires incompatible operand views")
        paths = tuple(dict.fromkeys((self.steps, *self.alternate_steps, other.steps, *other.alternate_steps)))
        return replace(self, alternate_steps=paths[1:])

    def check_shape(self, shape):
        if tuple(shape) != self.source_shape:
            raise ValueError(f"reshape operand requires prepared source shape {self.source_shape}, got {tuple(shape)}")
        return self.target_shape

    def layout(self, shape, stride):
        self.check_shape(shape)
        stride = tuple(stride)
        if stride == self.source_stride:
            return self.target_shape, self.target_stride
        current = self.prepare(shape, stride, self.steps, fold_batch=self.batch_prefix is not None)
        for steps in self.alternate_steps:
            alternate = self.prepare(shape, stride, steps, fold_batch=self.batch_prefix is not None)
            if (alternate.target_shape, alternate.target_stride) != (current.target_shape, current.target_stride):
                raise ValueError("shared reshape input requires incompatible runtime operand views")
        return current.target_shape, current.target_stride
