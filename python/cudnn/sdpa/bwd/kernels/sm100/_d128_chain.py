# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared delta, conversion and ordered-fold launches for small-half backward."""

from typing import Optional
import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16

THREADS = 128
ROWS_PER_CTA = 8
LANES_PER_ROW = 16
ELEMENTS_PER_LANE = 8
RN = cute.math.RoundingMode.NEAREST_EVEN


@cute.kernel
def dot_do_o_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: Optional[cute.Tensor],
    dot_factor: cutlass.Float32,
    lanes_per_row: cutlass.Constexpr[int] = LANES_PER_ROW,
    rows_per_cta: cutlass.Constexpr[int] = ROWS_PER_CTA,
):
    """``rows_per_cta`` rows per CTA, FP32 FTZ pair reduction, and dQ initialization when a dQ scratch is given."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * rows_per_cta + cutlass.Int64(tid // lanes_per_row)
    column = cutlass.Int64(tid % lanes_per_row) * ELEMENTS_PER_LANE
    # Tail rows past S_q read the last O / dO row (in bounds, uniform across
    # the shuffle group) and store delta = 0; their dQ rows are still zeroed.
    valid = row < cutlass.Int64(o.shape[1])
    io_row = cutlass.Int64(arith.select(valid.ir_value(), row.ir_value(), cutlass.Int64(o.shape[1] - 1).ir_value()))
    o_offset = (
        cutlass.Int64(batch) * cutlass.Int64(o.stride[0]) + io_row * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
    )
    do_offset = (
        cutlass.Int64(batch) * cutlass.Int64(do.stride[0]) + io_row * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
    )
    delta_offset = (
        cutlass.Int64(batch) * cutlass.Int64(delta.stride[0]) + cutlass.Int64(head) * cutlass.Int64(delta.stride[1]) + row * cutlass.Int64(delta.stride[2])
    )
    if cutlass.const_expr(dq_accum is not None):
        dq_offset = (
            cutlass.Int64(batch) * cutlass.Int64(dq_accum.stride[0])
            + row * cutlass.Int64(dq_accum.stride[1])
            + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2])
            + column
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
    for step in cutlass.range_constexpr(lanes_per_row.bit_length() - 1):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=lanes_per_row >> (step + 1), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, dot_factor, rounding=RN, ftz=True)
    dot = cutlass.Float32(arith.select(valid.ir_value(), dot.ir_value(), cutlass.Float32(0.0).ir_value()))
    if tid % lanes_per_row == 0:
        (delta.iterator.raw_ptr() + delta_offset).store(dot)


dot_do_o_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_zero_host(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, dq_accum: cute.Tensor, stream: cuda_driver.CUstream):
    """One dense no-dropout launch over dQ's rounded rows (a multiple of 128)."""
    lanes = o.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    dot_do_o_kernel(o, do, delta, dq_accum, cutlass.Float32(1.0), lanes, rows).launch(
        grid=(dq_accum.shape[1] // rows, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.jit
def delta_only_host(o: cute.Tensor, do: cute.Tensor, delta: cute.Tensor, stream: cuda_driver.CUstream):
    """One dense launch over delta's rounded rows (a multiple of 128), without a dQ scratch."""
    lanes = o.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    dot_do_o_kernel(o, do, delta, None, cutlass.Float32(1.0), lanes, rows).launch(
        grid=(delta.shape[2] // rows, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.kernel
def cast_dq_d128_kernel(
    dq_accum: cute.Tensor,
    dq_out: cute.Tensor,
    io_dtype: cutlass.Constexpr,
    lanes_per_row: cutlass.Constexpr[int] = LANES_PER_ROW,
    rows_per_cta: cutlass.Constexpr[int] = ROWS_PER_CTA,
):
    """Convert eight FP32 elements per lane without rescaling."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * rows_per_cta + cutlass.Int64(tid // lanes_per_row)
    column = cutlass.Int64(tid % lanes_per_row) * ELEMENTS_PER_LANE
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
    if row < cutlass.Int64(dq_out.shape[1]):
        Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)


cast_dq_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def fold_dkv_d128_kernel(
    part: cute.Tensor,
    out: cute.Tensor,
    splits: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    lanes_per_row: cutlass.Constexpr[int] = LANES_PER_ROW,
    rows_per_cta: cutlass.Constexpr[int] = ROWS_PER_CTA,
):
    """Sum the ``splits`` fp32 partial planes of a dK / dV row in plane order and convert eight elements per lane."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    row = cutlass.Int64(block) * rows_per_cta + cutlass.Int64(tid // lanes_per_row)
    column = cutlass.Int64(tid % lanes_per_row) * ELEMENTS_PER_LANE
    acc = [cutlass.Float32(0.0) for _ in range(ELEMENTS_PER_LANE)]
    for s in cutlass.range_constexpr(splits):
        src_offset = (
            (cutlass.Int64(batch) * splits + s) * cutlass.Int64(part.stride[0])
            + row * cutlass.Int64(part.stride[1])
            + cutlass.Int64(head) * cutlass.Int64(part.stride[2])
            + column
        )
        lo = (part.iterator.raw_ptr() + src_offset).load(count=4, alignment=16)
        hi = (part.iterator.raw_ptr() + src_offset + 4).load(count=4, alignment=16)
        for j in cutlass.range_constexpr(4):
            acc[j] = cute.math.add(acc[j], lo[j], rounding=RN, ftz=True)
            acc[4 + j] = cute.math.add(acc[4 + j], hi[j], rounding=RN, ftz=True)
    packed = cutlass.Vector.from_elements(
        (
            fp32_to_fp16(acc[0], acc[1], dtype=io_dtype),
            fp32_to_fp16(acc[2], acc[3], dtype=io_dtype),
            fp32_to_fp16(acc[4], acc[5], dtype=io_dtype),
            fp32_to_fp16(acc[6], acc[7], dtype=io_dtype),
        ),
        cutlass.Int32,
    )
    dst_offset = (
        cutlass.Int64(batch) * cutlass.Int64(out.stride[0]) + row * cutlass.Int64(out.stride[1]) + cutlass.Int64(head) * cutlass.Int64(out.stride[2]) + column
    )
    if row < cutlass.Int64(out.shape[1]):
        Pointer(cutlass.make_array_view(out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)


fold_dkv_d128_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def fold_dkv_host(part: cute.Tensor, out: cute.Tensor, splits: cutlass.Constexpr[int], stream: cuda_driver.CUstream):
    """One fold launch over the output rows; output rows must be 16-byte aligned."""
    lanes = out.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    fold_dkv_d128_kernel(part, out, splits, out.element_type, lanes, rows).launch(
        grid=((out.shape[1] + rows - 1) // rows, out.shape[0], out.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.jit
def cast_dq_host(dq_accum: cute.Tensor, dq_out: cute.Tensor, io_dtype: cutlass.Constexpr, stream: cuda_driver.CUstream):
    """One native vector cast launch; output rows must be 16-byte aligned."""
    lanes = dq_out.shape[3] // ELEMENTS_PER_LANE
    rows = THREADS // lanes
    cast_dq_d128_kernel(dq_accum, dq_out, io_dtype, lanes, rows).launch(
        grid=((dq_out.shape[1] + rows - 1) // rows, dq_out.shape[0], dq_out.shape[2]), block=(THREADS, 1, 1), stream=stream
    )
