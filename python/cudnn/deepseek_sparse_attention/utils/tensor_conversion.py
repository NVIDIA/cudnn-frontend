# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common CuTe tensor conversion helpers."""

from __future__ import annotations

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


def get_broadcast_dims(tensor: torch.Tensor) -> tuple[bool, ...]:
    """Return dimensions broadcast via stride 0."""
    return tuple(stride == 0 for stride in tensor.stride())


def fake_cute_tensor(dtype, rank: int, assumed_align: int = 16):
    """Compile-time stand-in carrying the type ``to_cute_tensor`` gives a compact tensor.

    The runtime binding is ``from_dlpack(...).mark_layout_dynamic()``: dynamic
    shape, dynamic 64-bit strides and a unit innermost stride. A kernel compiled
    on this fake therefore accepts the real tensors at launch, and plan-time
    compiles touch no device memory (recipe R11).
    """
    shape = tuple(cute.sym_int() for _ in range(rank))
    stride = tuple(cute.sym_int64() for _ in range(rank - 1)) + (1,)
    return cute.runtime.make_fake_tensor(dtype, shape, stride, assumed_align=assumed_align)


def to_cute_tensor(
    t: torch.Tensor,
    assumed_align: int = 16,
    leading_dim: int = -1,
    fully_dynamic: bool = False,
    enable_tvm_ffi: bool = True,
    divisibility=None,
):
    """Convert a torch tensor to a CuTe tensor for TVM FFI."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility is not None:
        tensor = tensor.mark_compact_shape_dynamic(mode=leading_dim, stride_order=t.dim_order(), divisibility=divisibility)
    return tensor
