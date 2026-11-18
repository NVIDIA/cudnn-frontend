# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-layout norm quantization, shared by forward and backward.

The backward topology follows cuDNN's combined post-pass: read the rounded
intermediate once and emit rowwise and columnwise payloads from those values.
A thread owns one column of a 32-row tile; a warp owns a row's 32-element
scale block. Both non-transposed payload stores are coalesced.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import div_rn_f32, fmax_f32, opaque_f32_bits
from cutlass.cute.arch.nvvm_wrappers import inline_ptx


@cute.jit
def _scale(amax, fp8_max: cutlass.Constexpr):
    # Norm's block_scale_utils.cuh clamps to FLT_MIN before upward E8M0
    # conversion: a zero block emits 0x01. Attention's helper emits 0x00.
    raw = fmax_f32(div_rn_f32(amax, opaque_f32_bits(0x47600000 if fp8_max == 57344.0 else 0x43E00000)), opaque_f32_bits(0x00800000))
    packed = inline_ptx(
        "cvt.rp.satfinite.ue8m0x2.f32 $0, $1, $1;",
        write_only_types=[cutlass.Uint16],
        read_only_args=[raw],
    )
    byte = cutlass.Int32(packed) & 255
    scale = (byte << 23).bitcast(cutlass.Float32)
    if amax == opaque_f32_bits(0x7F800000):
        scale = amax
    return scale, byte.to(cutlass.Uint8)


@cute.jit
def _fp8_byte(value, e5m2: cutlass.Constexpr):
    if cutlass.const_expr(e5m2):
        packed = inline_ptx(
            "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $1;",
            write_only_types=[cutlass.Uint16],
            read_only_args=[value],
        )
    else:
        packed = inline_ptx(
            "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
            write_only_types=[cutlass.Uint16],
            read_only_args=[value],
        )
    return packed.to(cutlass.Uint8)


@cute.kernel
def _mxfp8_kernel(x, row, row_sf, col, col_sf, tile_n: cutlass.Constexpr, e5m2: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    nx = block % (x.shape[2] // tile_n)
    rows = cute.ceil_div(x.shape[1], 32)
    my = (block // (x.shape[2] // tile_n)) % rows
    batch = cutlass.Int64(block // ((x.shape[2] // tile_n) * rows))
    j = cutlass.Int64(nx) * tile_n + lane
    r0 = cutlass.Int64(my) * 32
    vals = cute.make_rmem_tensor((32,), cutlass.Float32)
    amax_col = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(32):
        value = cutlass.Float32(0.0)
        if r0 + i < x.shape[1]:
            value = x[batch, r0 + i, j].to(cutlass.Float32)
        vals[i] = value
        amax_col = fmax_f32(amax_col, cute.math.abs(value))
    fp8_max: cutlass.Constexpr = 57344.0 if e5m2 else 448.0
    if cutlass.const_expr(col is not None):
        scale_col, byte_col = _scale(amax_col, fp8_max)
        col_sf[batch, my, j] = byte_col
    for i in cutlass.range_constexpr(32):
        value = vals[i]
        amax_row = cute.math.abs(value)
        for shift in cutlass.range_constexpr(5):
            other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax_row, 1 << shift, 31, kind=nvvm.Shfl.BFLY))
            amax_row = fmax_f32(amax_row, other)
        scale_row, byte_row = _scale(amax_row, fp8_max)
        if r0 + i < x.shape[1]:
            row[batch, r0 + i, j] = _fp8_byte(div_rn_f32(value, scale_row), e5m2)
            if lane % 32 == 0:
                row_sf[batch, r0 + i, j // 32] = byte_row
            if cutlass.const_expr(col is not None):
                code = _fp8_byte(div_rn_f32(value, scale_col), e5m2)
                col[batch, r0 + i, j] = code


_mxfp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def mxfp8(x, row, row_sf, col, col_sf, e5m2: cutlass.Constexpr, stream: cuda.CUstream):
    assert x.shape[2] % 32 == 0
    assert col is None or x.shape[1] % 32 == 0
    tile_n: cutlass.Constexpr = 128 if x.shape[2] % 128 == 0 else (64 if x.shape[2] % 64 == 0 else 32)
    tiles: cutlass.Constexpr = (x.shape[2] // tile_n) * cute.ceil_div(x.shape[1], 32) * x.shape[0]
    assert 0 < tiles < 2**31
    _mxfp8_kernel(x, row, row_sf, col, col_sf, tile_n, e5m2).launch(grid=(tiles, 1, 1), block=(tile_n, 1, 1), stream=stream)


@cute.kernel
def _nvfp4_kernel(x, row, row_sf, tile_n: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    nx = block % (x.shape[2] // tile_n)
    rows = cute.ceil_div(x.shape[1], 4)
    my = (block // (x.shape[2] // tile_n)) % rows
    batch = cutlass.Int64(block // ((x.shape[2] // tile_n) * rows))
    j = cutlass.Int64(nx) * tile_n + lane
    r0 = cutlass.Int64(my) * 4
    for i in cutlass.range_constexpr(4):
        value = cutlass.Float32(0.0)
        if r0 + i < x.shape[1]:
            value = x[batch, r0 + i, j].to(cutlass.Float32)
        amax = cute.math.abs(value)
        # Independent 16-lane reductions: two NVFP4 blocks per warp.
        for shift in cutlass.range_constexpr(4):
            other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, amax, 1 << shift, 31, kind=nvvm.Shfl.BFLY))
            amax = fmax_f32(amax, other)
        raw = fmax_f32(div_rn_f32(amax, opaque_f32_bits(0x40C00000)), opaque_f32_bits(0x00800000))
        # Norm's backend contract uses the raw FP32 scale for the payload,
        # then rounds the separately emitted scale to E4M3.
        scaled = div_rn_f32(value, raw)
        partner = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, scaled, 1, 31, kind=nvvm.Shfl.BFLY))
        packed = inline_ptx(
            "{ .reg .b8 code; cvt.rn.satfinite.e2m1x2.f32 code, $2, $1; mov.b32 $0, {code, code, code, code}; }",
            write_only_types=[cutlass.Uint32],
            read_only_args=[scaled, partner],
        ).to(cutlass.Uint8)
        if r0 + i < x.shape[1]:
            if lane % 2 == 0:
                row[batch, r0 + i, j // 2] = packed
            if lane % 16 == 0:
                row_sf[batch, r0 + i, j // 16] = _fp8_byte(raw, False)


_nvfp4_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def nvfp4(x, row, row_sf, col, col_sf, e5m2: cutlass.Constexpr, stream: cuda.CUstream):
    assert x.shape[2] % 32 == 0 and col is None and col_sf is None
    tile_n: cutlass.Constexpr = 128 if x.shape[2] % 128 == 0 else (64 if x.shape[2] % 64 == 0 else 32)
    tiles: cutlass.Constexpr = (x.shape[2] // tile_n) * cute.ceil_div(x.shape[1], 4) * x.shape[0]
    assert 0 < tiles < 2**31
    _nvfp4_kernel(x, row, row_sf, tile_n).launch(grid=(tiles, 1, 1), block=(tile_n, 1, 1), stream=stream)
