# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operand access, gradient stores and the delta pre-pass shared by the SM120 FP8 and MXFP8 backward bodies."""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import fmax_f32, fp32_to_fp8x2
from cudnn.sdpa.bwd.kernels.sm100.bprop_d192_dkdv_fp8 import _fold_amax, _scalar

FP8 = cutlass.Float8E4M3FN
M, C = 16, 128


@cute.jit
def _zero_word():
    # mma.sync accepts register operands only. A fully masked static tail
    # must not fold to an immediate 0 in the inline PTX instruction.
    return cute.arch.inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32], read_only_args=[])


@cute.jit
def _word(t, batch, row, head, col):
    bits = _zero_word()
    if row < t.shape[1] and col < t.shape[3]:
        off = cutlass.Int64(batch) * t.stride[0] + cutlass.Int64(row) * t.stride[1] + cutlass.Int64(head) * t.stride[2] + col
        values = (t.iterator.raw_ptr() + off).load(count=4, alignment=4)
        bits = values.bitcast(cutlass.Int32)[0]
    return bits


@cute.jit
def _column_word(t, batch, row, head, col):
    bits = _zero_word().bitcast(cutlass.Uint32)
    for i in cutlass.range_constexpr(4):
        byte = cutlass.Uint8(0)
        if row + i < t.shape[1] and col < t.shape[3]:
            off = cutlass.Int64(batch) * t.stride[0] + cutlass.Int64(row + i) * t.stride[1] + cutlass.Int64(head) * t.stride[2] + col
            byte = (t.iterator.raw_ptr() + off).load().bitcast(cutlass.Uint8)
        bits = bits | (cutlass.Uint32(byte) << (8 * i))
    return bits.bitcast(cutlass.Int32)


@cute.jit
def _store(accum, out, batch, head, owner, chunk, factor, out_scale, amax: Optional[cute.Tensor]):
    lane = cute.arch.lane_idx()
    maximum = cutlass.Float32(0.0)
    for nf in cutlass.range_constexpr(C // 8):
        for i in cutlass.range_constexpr(4):
            row = owner * M + lane // 4 + (8 if i >= 2 else 0)
            col = chunk * C + nf * 8 + lane % 4 * 2 + i % 2
            if row < out.shape[1] and col < out.shape[3]:
                val = accum[nf * 4 + i] * factor
                maximum = fmax_f32(maximum, cute.math.abs(val))
                if cutlass.const_expr(out.element_type == FP8):
                    bits = cutlass.Uint8(fp32_to_fp8x2(val * out_scale, cutlass.Float32(0.0)))
                    off = cutlass.Int64(batch) * out.stride[0] + cutlass.Int64(row) * out.stride[1] + cutlass.Int64(head) * out.stride[2] + col
                    (out.iterator.raw_ptr() + off).store(bits.bitcast(FP8))
                else:
                    off = cutlass.Int64(batch) * out.stride[0] + cutlass.Int64(row) * out.stride[1] + cutlass.Int64(head) * out.stride[2] + col
                    (out.iterator.raw_ptr() + off).store(val.to(out.element_type))
    if cutlass.const_expr(amax is not None):
        _fold_amax(maximum, amax)


@cute.kernel
def delta_kernel(o, do, delta, descale_o, descale_do, amaxes: tuple):
    block, head, batch = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    row, lane = block * 8 + tid // 16, tid % 16
    if block == 0 and head == 0 and batch == 0 and tid == 0:
        for tensor in amaxes:
            if cutlass.const_expr(tensor is not None):
                tensor[0] = cutlass.Float32(0.0)
    val = cutlass.Float32(0.0)
    if row < o.shape[1]:
        for i in cutlass.range_constexpr((o.shape[3] + 15) // 16):
            col = i * 16 + lane
            if col < o.shape[3]:
                oo = cutlass.Int64(batch) * o.stride[0] + cutlass.Int64(row) * o.stride[1] + cutlass.Int64(head) * o.stride[2] + col
                od = cutlass.Int64(batch) * do.stride[0] + cutlass.Int64(row) * do.stride[1] + cutlass.Int64(head) * do.stride[2] + col
                val = val + cutlass.Float32((o.iterator.raw_ptr() + oo).load()) * cutlass.Float32((do.iterator.raw_ptr() + od).load())
    for step in cutlass.range_constexpr(4):
        val = val + nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=val, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
    if lane == 0:
        if cutlass.const_expr(descale_o is not None):
            val = val * _scalar(descale_o) * _scalar(descale_do)
        delta[batch, head, row] = val


delta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_host(o, do, delta, descale_o, descale_do, amax_dq, amax_dk, amax_dv, amax_dp, stream):
    delta_kernel(o, do, delta, descale_o, descale_do, (amax_dq, amax_dk, amax_dv, amax_dp)).launch(
        grid=(delta.shape[2] // 8, o.shape[2], o.shape[0]), block=(128, 1, 1), stream=stream
    )
