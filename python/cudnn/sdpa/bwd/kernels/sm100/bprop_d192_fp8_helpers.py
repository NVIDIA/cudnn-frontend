# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense D_QK = 192 / D_V = 128 per-tensor FP8 delta and amax reset."""

from typing import Optional

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm

THREADS = 128
ROWS_PER_CTA = 8
LANES_PER_ROW = 16
ELEMENTS_PER_LANE = 8
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
def dot_do_o_d192_fp8_kernel(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    descale_o: cute.Tensor,
    descale_do: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
):
    """Eight rows per CTA: the true-unit O ."""
    block, batch, head = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    if block == 0 and batch == 0 and head == 0 and tid == 0:
        _reset(amax_dq)
        _reset(amax_dk)
        _reset(amax_dv)
        _reset(amax_dp)
    row = cutlass.Int64(block) * ROWS_PER_CTA + cutlass.Int64(tid // LANES_PER_ROW)
    lane = tid % LANES_PER_ROW
    column = cutlass.Int64(lane) * ELEMENTS_PER_LANE
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
    do_values = (do.iterator.raw_ptr() + do_offset).load(count=ELEMENTS_PER_LANE, alignment=8)
    o_values = (o.iterator.raw_ptr() + o_offset).load(count=ELEMENTS_PER_LANE, alignment=8)
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
    dot = cute.math.mul(dot, _scalar(descale_o) * _scalar(descale_do), rounding=RN, ftz=True)
    dot = cutlass.Float32(arith.select(valid.ir_value(), dot.ir_value(), cutlass.Float32(0.0).ir_value()))
    if lane == 0:
        (delta.iterator.raw_ptr() + delta_offset).store(dot)


dot_do_o_d192_fp8_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def delta_d192_fp8_host(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    descale_o: cute.Tensor,
    descale_do: cute.Tensor,
    amax_dq: Optional[cute.Tensor],
    amax_dk: Optional[cute.Tensor],
    amax_dv: Optional[cute.Tensor],
    amax_dp: Optional[cute.Tensor],
    stream: cuda_driver.CUstream,
):
    """One dense launch over delta's rounded rows (a multiple of 128)."""
    dot_do_o_d192_fp8_kernel(o, do, delta, descale_o, descale_do, amax_dq, amax_dk, amax_dv, amax_dp).launch(
        grid=(delta.shape[2] // ROWS_PER_CTA, o.shape[0], o.shape[2]), block=(THREADS, 1, 1), stream=stream
    )
