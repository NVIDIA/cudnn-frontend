# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped-head gradient reduction (GVA/GQA) for linear-attention backward.

With grouped heads a backward kernel that emits per-output-head gradients at
``HO`` heads leaves the true per-native-head gradient as the sum over each
head's group of ``r = HO // H`` consecutive output heads:

    out[t, i, ...] = sum_{j < r} in[t, i * r + j, ...]

Flat 1-D grid over output words: each thread owns one 16-byte chunk of an
f16/bf16 row (one 4-byte word of an fp32 fold, or of a row that is not whole
chunks), gathers it from all ``r`` group heads (coalesced, strided by
``inner_words``), accumulates in fp32, and stores it back.  Head count and
group size are runtime values.
Serves the f16/bf16 ``[total, HO, D]`` tensor grads (dQ/dK for GVA, dK/dV for
GQA) and the fp32 ``[total, HO]`` Gate/Beta grads.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global_v4

USE_PDL = True

BLOCK = 1024  # 4-byte words per CTA
ROW_CHUNKS = 16  # 16-byte chunks of a 128-element 16-bit row


@cute.kernel
def frost_head_reduce(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    total_words: cutlass.Int64,
    out_row_words: cutlass.Int64,
    out_head_words: cutlass.Int64,
    h_count: cute.FastDivmodDivisorV2,
    r: cutlass.Int32,
    inner_words: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    thread_words = cutlass.const_expr(4 if io_dtype != cutlass.Float32 and inner_words % 4 == 0 else 1)
    gw = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx)) * cutlass.Int64(thread_words)
    if gw < total_words:
        seg = gw // cutlass.Int64(inner_words)
        w_off = gw - seg * cutlass.Int64(inner_words)
        base = seg * (cutlass.Int64(r) * cutlass.Int64(inner_words)) + w_off
        t_idx, h_idx = divmod(seg.to(cutlass.Int32), h_count)
        out_off = cutlass.Int64(t_idx) * out_row_words + cutlass.Int64(h_idx) * out_head_words + w_off
        if cutlass.const_expr(io_dtype == cutlass.Float32):
            in_p = cute.recast_ptr(mIn.iterator, dtype=cutlass.Float32)
            out_p = cute.recast_ptr(mOut.iterator, dtype=cutlass.Float32)
            acc = cutlass.Float32(0.0)
            addr = base
            for i in cutlass.range(r):
                v = (in_p + addr).load()
                acc = v if i == 0 else acc + v
                addr = addr + cutlass.Int64(inner_words)
            (out_p + out_off).store(acc)
        elif cutlass.const_expr(thread_words == 4):
            out_row = cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32) + out_off
            addr = mIn.iterator.toint() + base * cutlass.Int64(4)
            acc = [cutlass.Float32(0.0)] * 8
            for part in cutlass.range((r + cutlass.Int32(3)) // cutlass.Int32(4)):
                loads = [list(ld_global_v4(addr, cutlass.Int32))]
                for j in cutlass.range_constexpr(1, 4):
                    words = [cutlass.Int32(0)] * 4
                    if part * cutlass.Int32(4) + cutlass.Int32(j) < r:
                        words = list(ld_global_v4(addr + cutlass.Int64(j * inner_words * 4), cutlass.Int32))
                    loads.append(words)
                for j in cutlass.range_constexpr(4):
                    summed = []
                    for m in cutlass.range_constexpr(4):
                        lo, hi = f16x2_to_f32(loads[j][m], dtype=io_dtype)
                        summed.append(acc[2 * m] + lo)
                        summed.append(acc[2 * m + 1] + hi)
                    acc = summed
                addr = addr + cutlass.Int64(16 * inner_words)
            for m in cutlass.range_constexpr(4):
                (out_row + m).store(fp32_to_fp16(acc[2 * m], acc[2 * m + 1], dtype=io_dtype))
        else:
            in_p = cute.recast_ptr(mIn.iterator, dtype=cutlass.Int32)
            out_p = cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32)
            acc_lo = cutlass.Float32(0.0)
            acc_hi = cutlass.Float32(0.0)
            addr = base
            for i in cutlass.range(r):
                lo, hi = f16x2_to_f32((in_p + addr).load(), dtype=io_dtype)
                acc_lo = lo if i == 0 else acc_lo + lo
                acc_hi = hi if i == 0 else acc_hi + hi
                addr = addr + cutlass.Int64(inner_words)
            (out_p + out_off).store(fp32_to_fp16(acc_lo, acc_hi, dtype=io_dtype))


@cute.jit
def launch(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    total_words: cutlass.Int64,
    out_row_words: cutlass.Int64,
    out_head_words: cutlass.Int64,
    grid_x: cutlass.Int32,
    h_count: cutlass.Int32,
    r: cutlass.Int32,
    inner_words: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    frost_head_reduce(mIn, mOut, total_words, out_row_words, out_head_words, cute.FastDivmodDivisorV2(h_count), r, inner_words, io_dtype).launch(
        grid=(grid_x, 1, 1),
        block=(BLOCK // (4 if io_dtype != cutlass.Float32 and inner_words % 4 == 0 else 1), 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


frost_head_reduce.set_name_prefix("cudnn", remove_cutlass_symbol=False)
