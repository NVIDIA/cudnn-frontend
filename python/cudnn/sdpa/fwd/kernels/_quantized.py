# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Scalar reduction lifecycle shared by prepared quantized SDPA hosts."""

import cutlass
import cutlass.cute as cute


@cute.jit
def _initialize_split_amax(amax):
    """One producer clears Amax before the following combine kernel reduces it.

    Split attention never reduces Amax itself. The stream dependency from all
    attention CTAs to combine orders this store without a separate reset launch.
    """
    tx, ty, tz = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    if (tx == 0) & (ty == 0) & (tz == 0) & (bx == 0) & (by == 0) & (bz == 0):
        amax.iterator.store(amax.element_type(0))


@cute.jit
def _scale_or_one(scale):
    """None-specialized identity when the final combine owns output scaling."""
    value = cutlass.Float32(1.0)
    if cutlass.const_expr(scale is not None):
        value = cutlass.Float32(cutlass.make_array_view(scale)[0])
    return value


@cute.kernel
def _reset_amax_kernel(amax: cute.Pointer):
    amax.store(cutlass.Float32(0.0))


_reset_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _unscale_amax_kernel(amax: cute.Pointer, scale: cute.Pointer):
    amax.store(amax.load() / scale.load())


_unscale_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
