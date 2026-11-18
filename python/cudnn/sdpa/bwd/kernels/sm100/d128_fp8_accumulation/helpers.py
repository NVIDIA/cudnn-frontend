# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense d128 per-tensor FP8 delta, dQ initialization, amax reset and dQ conversion."""

from typing import Optional

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from cudnn.frost.tile_dsl.pointwise import abs_max_tree, fmax_f32, fp32_to_fp16, fp32_to_fp8x2, opaque_f32_zero, pack_fp8x2_pairs

HEAD_DIM = globals().get("FROST_TEMPLATE_PARAMS", 128)
THREADS = 128
ELEMENTS_PER_LANE = 8
LANES_PER_ROW = HEAD_DIM // ELEMENTS_PER_LANE
ROWS_PER_CTA = THREADS // LANES_PER_ROW
CAST_ROW_PASSES = 4
CAST_ROWS_PER_CTA = ROWS_PER_CTA * CAST_ROW_PASSES
RN = cute.math.RoundingMode.NEAREST_EVEN
FP8 = cutlass.Float8E4M3FN


@cute.jit
def _scalar(tensor):
    return cutlass.Float32(cutlass.make_array_view(tensor)[0])


@cute.jit
def _reset(tensor: Optional[cute.Tensor]):
    if cutlass.const_expr(tensor is not None):
        (tensor.iterator.raw_ptr()).store(cutlass.Float32(0.0))


@cute.kernel
def dot_do_o_fp8_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: cute.Tensor,
    descale_o: cute.Tensor,
    descale_do: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
):
    """Eight rows per CTA, FP32 FTZ pair reduction in true units, and dQ initialization."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    if block == 0 and batch == 0 and head == 0 and tid == 0:
        _reset(amax_dq)
        _reset(amax_dk)
        _reset(amax_dv)
        _reset(amax_dp)
    row = cutlass.Int64(block) * ROWS_PER_CTA + cutlass.Int64(tid // LANES_PER_ROW)
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    valid = row < cutlass.Int64(o.shape[1])
    io_row = cutlass.Int64(arith.select(valid.ir_value(), row.ir_value(), cutlass.Int64(o.shape[1] - 1).ir_value()))
    dq_offset = (
        cutlass.Int64(batch) * cutlass.Int64(dq_accum.stride[0])
        + row * cutlass.Int64(dq_accum.stride[1])
        + cutlass.Int64(head) * cutlass.Int64(dq_accum.stride[2])
        + column
    )
    o_offset = (
        cutlass.Int64(batch) * cutlass.Int64(o.stride[0]) + io_row * cutlass.Int64(o.stride[1]) + cutlass.Int64(head) * cutlass.Int64(o.stride[2]) + column
    )
    do_offset = (
        cutlass.Int64(batch) * cutlass.Int64(do.stride[0]) + io_row * cutlass.Int64(do.stride[1]) + cutlass.Int64(head) * cutlass.Int64(do.stride[2]) + column
    )
    delta_offset = (
        cutlass.Int64(batch) * cutlass.Int64(delta.stride[0]) + cutlass.Int64(head) * cutlass.Int64(delta.stride[1]) + row * cutlass.Int64(delta.stride[2])
    )
    zero4 = cutlass.Vector.from_elements((cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0), cutlass.Float32(0.0)), cutlass.Float32)
    (dq_accum.iterator.raw_ptr() + dq_offset).store(zero4, alignment=16)
    (dq_accum.iterator.raw_ptr() + dq_offset + 4).store(zero4, alignment=16)
    do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=8)
    o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=8)
    pairs = []
    for j in cutlass.range_constexpr(ELEMENTS_PER_LANE // 2):
        high = cute.math.mul(do_values[2 * j + 1].to(cutlass.Float32), o_values[2 * j + 1].to(cutlass.Float32), rounding=RN, ftz=True)
        pairs.append(cute.math.fma(do_values[2 * j].to(cutlass.Float32), o_values[2 * j].to(cutlass.Float32), high, rounding=RN, ftz=True))
    dot = cute.math.add(pairs[0], pairs[1], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[2], rounding=RN, ftz=True)
    dot = cute.math.add(dot, pairs[3], rounding=RN, ftz=True)
    for step in cutlass.range_constexpr(LANES_PER_ROW.bit_length() - 1):
        peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=dot, offset=LANES_PER_ROW >> (1 + step), mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
        dot = cute.math.add(dot, peer, rounding=RN, ftz=True)
    dot = cute.math.mul(dot, _scalar(descale_o) * _scalar(descale_do), rounding=RN, ftz=True)
    dot = cutlass.Float32(arith.select(valid.ir_value(), dot.ir_value(), cutlass.Float32(0.0).ir_value()))
    if tid % LANES_PER_ROW == 0:
        (delta.iterator.raw_ptr() + delta_offset).store(dot)


dot_do_o_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_zero_fp8_host(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    dq_accum: cute.Tensor,
    descale_o: cute.Tensor,
    descale_do: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    stream: cuda_driver.CUstream,
):
    """One dense launch over dQ's rounded rows (a multiple of 128)."""
    dot_do_o_fp8_kernel(o, do, delta, dq_accum, descale_o, descale_do, amax_dq, amax_dk, amax_dv, amax_dp).launch(
        grid=(dq_accum.shape[1] // ROWS_PER_CTA, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )


@cute.kernel
def cast_dq_fp8_kernel(
    dq_accum: cute.Tensor,
    dq_out: cute.Tensor,
    descale_dp: cute.Tensor,
    descale_k: cute.Tensor,
    scale_dq: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    grad_dtype: cutlass.Constexpr,
):
    """Convert eight FP32 elements per lane and row pass: descale, fold amax_dQ, scale, cast."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    fp8_out = cutlass.const_expr(grad_dtype == FP8)
    fold = cutlass.const_expr(amax_dq is not None)
    factor = _scalar(descale_dp) * _scalar(descale_k)
    out_scale = cutlass.Float32(1.0)
    if cutlass.const_expr(fp8_out):
        out_scale = _scalar(scale_dq)
    amax = opaque_f32_zero()
    column = cutlass.Int64(tid % LANES_PER_ROW) * ELEMENTS_PER_LANE
    for pass_index in cutlass.range_constexpr(CAST_ROW_PASSES):
        row = cutlass.Int64(block) * CAST_ROWS_PER_CTA + cutlass.Int64(pass_index * ROWS_PER_CTA) + cutlass.Int64(tid // LANES_PER_ROW)
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
        true_values = [cutlass.Float32(x) * factor for x in (lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3])]
        if cutlass.const_expr(fold):
            amax = fmax_f32(amax, abs_max_tree(true_values))
        if row < cutlass.Int64(dq_out.shape[1]):
            if cutlass.const_expr(fp8_out):
                scaled = [x * out_scale for x in true_values]
                packed = cutlass.Vector.from_elements(
                    (
                        pack_fp8x2_pairs(fp32_to_fp8x2(scaled[0], scaled[1], dtype=FP8), fp32_to_fp8x2(scaled[2], scaled[3], dtype=FP8)),
                        pack_fp8x2_pairs(fp32_to_fp8x2(scaled[4], scaled[5], dtype=FP8), fp32_to_fp8x2(scaled[6], scaled[7], dtype=FP8)),
                    ),
                    cutlass.Int32,
                )
                Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=8)
            else:
                packed = cutlass.Vector.from_elements(
                    (
                        fp32_to_fp16(true_values[0], true_values[1], dtype=grad_dtype),
                        fp32_to_fp16(true_values[2], true_values[3], dtype=grad_dtype),
                        fp32_to_fp16(true_values[4], true_values[5], dtype=grad_dtype),
                        fp32_to_fp16(true_values[6], true_values[7], dtype=grad_dtype),
                    ),
                    cutlass.Int32,
                )
                Pointer(cutlass.make_array_view(dq_out).data_ptr() + dst_offset, dtype=cutlass.Int32).store(packed, alignment=16)
    if cutlass.const_expr(fold):
        for step in cutlass.range_constexpr(5):
            peer = nvvm.shfl_sync(thread_mask=0xFFFFFFFF, val=amax, offset=1 << step, mask_and_clamp=0x1F, kind=nvvm.Shfl.BFLY)
            amax = fmax_f32(amax, peer)
        partial = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.smem)
        if tid % 32 == 0:
            partial.subview(tid // 32).store(amax)
        nvvm.barrier_cta_sync()
        if tid == 0:
            folded = fmax_f32(fmax_f32(partial.subview(0).load(), partial.subview(1).load()), fmax_f32(partial.subview(2).load(), partial.subview(3).load()))
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, Pointer(amax_dq.iterator.raw_ptr(), dtype=cutlass.Int32), folded.bitcast(cutlass.Int32))


cast_dq_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def cast_dq_fp8_host(
    dq_accum: cute.Tensor,
    dq_out: cute.Tensor,
    descale_dp: cute.Tensor,
    descale_k: cute.Tensor,
    scale_dq: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    grad_dtype: cutlass.Constexpr,
    stream: cuda_driver.CUstream,
):
    """One conversion launch over dQ's rounded rows (a multiple of 128)."""
    cast_dq_fp8_kernel(dq_accum, dq_out, descale_dp, descale_k, scale_dq, amax_dq, grad_dtype).launch(
        grid=(dq_accum.shape[1] // CAST_ROWS_PER_CTA, dq_out.shape[0], dq_out.shape[2]), block=(THREADS, 1, 1), stream=stream
    )
