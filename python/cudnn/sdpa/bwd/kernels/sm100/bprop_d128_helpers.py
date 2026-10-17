# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense aligned d128 delta, dQ initialization and native output conversion."""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16

THREADS = 128
ROWS_PER_CTA = 8
LANES_PER_ROW = 16
ELEMENTS_PER_LANE = 8
RN = cute.math.RoundingMode.NEAREST_EVEN


@cute.kernel
def dot_do_o_kernel(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, dq_accum: cute.Tensor, dot_factor: cutlass.Float32):
    """Eight rows per CTA, FP32 FTZ pair reduction, and dQ initialization."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * ROWS_PER_CTA + cutlass.Int64(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    dq_offset = (
        cutlass.Int64(batch) * cutlass.Int64(dq_accum.stride[0])
        + row * cutlass.Int64(dq_accum.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2])
        + column
    )
    o_offset = cutlass.Int64(batch) * cutlass.Int64(o.stride[0]) + row * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
    do_offset = (
        cutlass.Int64(batch) * cutlass.Int64(do.stride[0]) + row * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
    )
    delta_offset = (
        cutlass.Int64(batch) * cutlass.Int64(delta.stride[0]) + cutlass.Int64(head) * cutlass.Int64(delta.stride[1]) + row * cutlass.Int64(delta.stride[2])
    )
    zero4 = cutlass.Vector.from_elements((cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0)), cutlass.Float32)
    (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
    (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
    do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=16)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    for step in cutlass.range_constexpr(4):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=1 << (3 - step), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, dot_factor, rounding=RN, ftz=True)
    if tid % LANES_PER_ROW == 0:
        (delta.iterator.raw_ptr() + delta_offset).store(dot)


dot_do_o_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_zero_host(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, dq_accum: cute.Tensor, stream: cuda_driver.CUstream):
    """One dense no-dropout launch; sequence lengths must be multiples of 128."""
    dot_do_o_kernel(o, do, delta, dq_accum, cutlass.Float32(1.0)).launch(
        grid=(o.shape[1] // ROWS_PER_CTA, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.kernel
def cast_dq_d128_kernel(dq_accum: cute.Tensor, dq_out: cute.Tensor, io_dtype: cutlass.Constexpr):
    """Convert eight FP32 elements per lane without rescaling."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * ROWS_PER_CTA + cutlass.Int64(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    src_offset = (
        cutlass.Int64(batch) * cutlass.Int64(dq_accum.stride[0])
        + row * cutlass.Int64(dq_accum.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2])
        + column
    )
    dst_offset = (
        cutlass.Int64(batch) * cutlass.Int64(dq_out.stride[0])
        + row * cutlass.Int64(dq_out.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq_out.stride[2])
        + column
    )
    lo = (dq_accum.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
    hi = (dq_accum.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
    packed = cutlass.Vector.from_elements(
        (
            fp32_to_fp16(lo[0], lo[1], dtype=io_dtype),
            fp32_to_fp16(lo[2], lo[3], dtype=io_dtype),
            fp32_to_fp16(hi[0], hi[1], dtype=io_dtype),
            fp32_to_fp16(hi[2], hi[3], dtype=io_dtype),
        ),
        cutlass.Int32,
    )
    Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)


cast_dq_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def cast_dq_host(dq_accum: cute.Tensor, dq_out: cute.Tensor, io_dtype: cutlass.Constexpr, stream: cuda_driver.CUstream):
    """One native vector cast launch; output rows must be 16-byte aligned."""
    cast_dq_d128_kernel(dq_accum, dq_out, io_dtype).launch(
        grid=(dq_out.shape[1] // ROWS_PER_CTA, dq_out.shape[0], dq_out.shape[2]), block=(THREADS, 1, 1), stream=stream
    )
