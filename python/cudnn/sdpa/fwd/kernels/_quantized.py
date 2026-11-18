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


@cute.jit
def _descale_qk_negate_bit(descale_q, descale_k):
    """tcgen05 a_negate (bit 13) when the device descale_q * descale_k is negative (#1435)."""
    bit = cutlass.Int32(0)
    if cutlass.Float32(cutlass.make_array_view(descale_q)[0]) * cutlass.Float32(cutlass.make_array_view(descale_k)[0]) < cutlass.Float32(0.0):
        bit = cutlass.Int32(1 << 13)
    return bit


@cute.kernel
def _reset_amax_kernel(amax: cute.Pointer):
    amax.store(cutlass.Float32(0.0))


_reset_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _unscale_amax_kernel(amax: cute.Pointer, scale: cute.Pointer):
    amax.store(amax.load() / scale.load())


_unscale_amax_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _public_s_scale_factors(scale_s, descale_s, cast_bias: cutlass.Constexpr, rescale_threshold: cutlass.Constexpr):
    """Convert public S scales into the existing online-softmax coordinates.

    The row sum and Stats retain the kernel internal cast-bias units. Only P
    entering low-precision MMA receives the public scale; its matching output
    factor restores those internal units and applies the public descale.
    Headroom uses the same rescale allowance as the running row maximum.
    None preserves the standalone legacy entry until its caller opts in.
    """
    p_gain = cutlass.Float32(1.0)
    o_gain = cutlass.Float32(1.0)
    if cutlass.const_expr((scale_s is None) != (descale_s is None)):
        raise ValueError("scale_s and descale_s must be supplied together")
    if cutlass.const_expr(scale_s is not None):
        units = 2.0 ** (cast_bias + rescale_threshold)
        s_value = cutlass.Float32(scale_s) if cutlass.const_expr(not isinstance(scale_s, cute.Tensor)) else _scale_or_one(scale_s)
        ds_value = cutlass.Float32(descale_s) if cutlass.const_expr(not isinstance(descale_s, cute.Tensor)) else _scale_or_one(descale_s)
        p_gain = s_value * cutlass.Float32(1.0 / units)
        o_gain = ds_value * cutlass.Float32(units)
    return p_gain, o_gain


@cute.jit
def _launch_public_s(kernel: cutlass.Constexpr, kernel_args, kernel_kwargs, launch_kwargs, scale_s, descale_s, allow_fixed: cutlass.Constexpr):
    """Precompile the standard host pair and retain arbitrary runtime scalars.

    Both branches are emitted during plan compilation. No device scalar is
    inspected on the host, and a runtime value never becomes a compile key.
    The main body still owns the conversion from public to internal units.
    """
    if cutlass.const_expr(
        allow_fixed and scale_s is not None and descale_s is not None and not isinstance(scale_s, cute.Tensor) and not isinstance(descale_s, cute.Tensor)
    ):
        if (scale_s == cutlass.Float32(256.0)) & (descale_s == cutlass.Float32(1.0 / 256.0)):
            kernel(*kernel_args, **kernel_kwargs, scale_s_t=None, descale_s_t=None, fixed_s_pair=(256.0, 1.0 / 256.0)).launch(**launch_kwargs)
        else:
            kernel(*kernel_args, **kernel_kwargs, scale_s_t=scale_s, descale_s_t=descale_s).launch(**launch_kwargs)
    else:
        kernel(*kernel_args, **kernel_kwargs, scale_s_t=scale_s, descale_s_t=descale_s).launch(**launch_kwargs)
