# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped-head gradient reduction (GVA/GQA) for linear-attention backward.

With grouped heads a backward kernel that emits per-output-head gradients at
``HO`` heads leaves the true per-native-head gradient as the sum over each
head's group of ``r = HO // H`` consecutive output heads:

    out[t, i, ...] = sum_{j < r} in[t, i * r + j, ...]

Flat 1-D grid over output words: each thread owns one 4-byte word (a packed
f16x2/bf16x2 pair, or one fp32 element), gathers it from all ``r`` group
heads (coalesced, strided by ``inner_words``), accumulates in fp32, and
stores one word back.  Head count and group size are runtime values; the
group loop runs over all ``r`` heads with a select on the first so the
compiler's runtime unroll (8/4/2/1 blocks) issues a group's loads together.
Serves the f16/bf16 ``[total, HO, D]`` tensor grads (dQ/dK for GVA, dK/dV for
GQA) and the fp32 ``[total, HO]`` Gate/Beta grads.

The Mamba-2 form (``head_group_reduce_dot_product``) folds the per-head dQ / dK
rows to the B / C groups and dots every head's row with its group's Q / K row on
the way, the per-token fp32 dot the dt-gate gradient needs.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cuda.bindings.driver as cuda

from cutlass.cute.runtime import from_dlpack

from .host import get_dtype
from cudnn.frost.device import current_device
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fadd2, fp32_to_fp16, opaque_f32_zero
from cudnn.frost.tile_dsl.tma import ld_global_v4, st_global_v4
from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids

USE_PDL = True

BLOCK = 256
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
    gw = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
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
        block=(BLOCK, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


compiled_cache = {}


def head_group_reduce(src, dst, *, stream) -> None:
    """Reduce ``src (total, HO, D)`` into ``dst (total, H, D)``, or the
    rank-2 ``(total, HO)`` into ``(total, H)``, by summing each group of
    ``r = HO // H`` consecutive heads (fp32 accumulation).

    ``src`` is contiguous (kernel-internal wide buffer); ``dst`` needs a
    stride-1 innermost dim with free outer strides (f16/bf16 outer strides
    must be even, word-pair stores). Same-dtype (f16/bf16, or fp32),
    DLPack-compatible CUDA tensors; the f16/bf16 inner extent ``D`` must be
    even.  Compile-cache-and-replay per ``(dtype, rank, D)``; head counts are runtime."""
    if len(src.shape) == 2:
        total, HO = src.shape
        D = 1
        H = dst.shape[1]
    else:
        total, HO, D = src.shape
        H = dst.shape[1]
    io_dtype = get_dtype(src.dtype)
    is_fp32 = io_dtype == cutlass.Float32
    r = HO // H
    inner_words = D if is_fp32 else D // 2
    total_words = total * H * inner_words
    grid_x = -(-total_words // BLOCK)
    cu_stream = cuda.CUstream(int(stream))

    dst_strides = tuple(dst.stride())
    out_row_words = dst_strides[0] if is_fp32 else dst_strides[0] // 2
    out_head_words = (dst_strides[1] if is_fp32 else dst_strides[1] // 2) if len(dst.shape) == 3 else 1

    key = (str(src.dtype).split(".")[-1], len(src.shape), D, current_device())
    if key not in compiled_cache:

        src_c = from_dlpack(src, assumed_align=4).mark_layout_dynamic(leading_dim=len(src.shape) - 1)
        compiled_cache[key] = cute.compile(
            launch,
            src_c,
            from_dlpack(dst, assumed_align=4).mark_layout_dynamic(leading_dim=len(dst.shape) - 1),
            cutlass.Int64(total_words),
            cutlass.Int64(out_row_words),
            cutlass.Int64(out_head_words),
            cutlass.Int32(grid_x),
            cutlass.Int32(H),
            cutlass.Int32(r),
            inner_words,
            io_dtype,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    compiled_cache[key](src, dst, total_words, out_row_words, out_head_words, grid_x, H, r, cu_stream)


@cute.kernel
def frost_mamba2_head_group_reduce_dot_product(
    mSrc: cute.Tensor,
    mOp: cute.Tensor,
    mDst: cute.Tensor | None,
    mDot: cute.Tensor,
    n_threads: cutlass.Int32,
    groups: cute.FastDivmodDivisorV2,
    r: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
) -> None:
    """Thread (row, group, chunk): the group's ``r`` head rows of ``mSrc`` [rows, HO, 128] summed into ``mDst``
    [rows, G, 128] and dotted with the group's ``mOp`` row into ``mDot`` [rows, HO]; the 16 chunk lanes of a row reduce
    the dots by shuffle."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    gw = bidx * cutlass.Int32(BLOCK) + tidx
    valid = gw < n_threads
    gw_c = gw if valid else cutlass.Int32(0)
    c = gw_c % cutlass.Int32(ROW_CHUNKS)
    row, g = divmod(gw_c // cutlass.Int32(ROW_CHUNKS), groups)
    k0 = cutlass.Int64(c * cutlass.Int32(8))
    zero = opaque_f32_zero()
    op_addr = mOp.iterator.toint() + (cutlass.Int64(row) * cutlass.Int64(mOp.stride[0]) + cutlass.Int64(g) * cutlass.Int64(mOp.stride[1]) + k0) * cutlass.Int64(
        2
    )
    op_words = ld_global_v4(op_addr, cutlass.Int32)
    op = []
    for j in cutlass.range_constexpr(4):
        lo, hi = f16x2_to_f32(op_words[j], dtype=io_dtype)
        op += [lo, hi]
    acc = [zero for _ in range(8)]
    dots = []
    source_base = mSrc.iterator.toint() + (
        cutlass.Int64(row) * cutlass.Int64(mSrc.stride[0]) + cutlass.Int64(g * cutlass.Int32(r)) * cutlass.Int64(mSrc.stride[1]) + k0
    ) * cutlass.Int64(2)
    source_step = cutlass.Int64(mSrc.stride[1]) * cutlass.Int64(2)
    for i in cutlass.range_constexpr(r):
        words = ld_global_v4(source_base + cutlass.Int64(i) * source_step, cutlass.Int32)
        dot = zero
        for j in cutlass.range_constexpr(4):
            lo, hi = f16x2_to_f32(words[j], dtype=io_dtype)
            acc[2 * j], acc[2 * j + 1] = fadd2(acc[2 * j], acc[2 * j + 1], lo, hi)
            dot = dot + op[2 * j] * lo + op[2 * j + 1] * hi
        dots.append(dot)
    for i in cutlass.range_constexpr(r):
        dot = dots[i]
        for offset in [8, 4, 2, 1]:
            dot = dot + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, dot, offset, 31, kind=nvvm.Shfl.BFLY))
        dots[i] = dot
    if valid:
        if c == 0:
            for i in cutlass.range_constexpr(r):
                mDot[row, g * cutlass.Int32(r) + cutlass.Int32(i)] = dots[i]
        if cutlass.const_expr(mDst is not None):
            out = [fp32_to_fp16(acc[2 * j], acc[2 * j + 1], dtype=io_dtype) for j in range(4)]
            destination_address = mDst.iterator.toint() + (
                cutlass.Int64(row) * cutlass.Int64(mDst.stride[0]) + cutlass.Int64(g) * cutlass.Int64(mDst.stride[1]) + k0
            ) * cutlass.Int64(2)
            st_global_v4(destination_address, out, cutlass.Int32)


@cute.jit
def head_group_reduce_dot_product_launch(
    mSrc: cute.Tensor,
    mOp: cute.Tensor,
    mDst: cute.Tensor | None,
    mDot: cute.Tensor,
    n_threads: cutlass.Int32,
    groups: cutlass.Int32,
    r: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    frost_mamba2_head_group_reduce_dot_product(mSrc, mOp, mDst, mDot, n_threads, cute.FastDivmodDivisorV2(groups), r, mSrc.element_type).launch(
        grid=(cute.ceil_div(n_threads, BLOCK), 1, 1), block=(BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL
    )


def head_group_reduce_dot_product(source, op, destination, dot, *, stream) -> None:
    """``destination`` [rows, G, 128] = the per-group head sum of ``source`` [rows, HO, 128] (skipped when ``destination`` is None), ``dot``
    [rows, HO] fp32 = <op[row, group], source[row, head]>; ``op`` [rows, G, 128]."""
    rows, h_o, k = (int(dim) for dim in source.shape)
    groups = int(op.shape[1])
    if k != ROW_CHUNKS * 8 or h_o % groups:
        raise ValueError(
            f"head_group_reduce_dot_product: expected [rows, HO, 128] with HO a multiple of the groups, got {tuple(source.shape)} over {groups} groups"
        )
    r = h_o // groups
    n_threads = rows * groups * ROW_CHUNKS
    cu_stream = cuda.CUstream(int(stream))
    key = ("head_group_reduce_dot_product", str(source.dtype), r, destination is not None, current_device())
    if key not in compiled_cache:
        traced = [from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=2) if t is not None else None for t in (source, op, destination)]
        traced.append(from_dlpack(dot, assumed_align=4).mark_layout_dynamic(leading_dim=1))
        compiled_cache[key] = cute.compile(
            head_group_reduce_dot_product_launch, *traced, cutlass.Int32(n_threads), cutlass.Int32(groups), r, cu_stream, options="--enable-tvm-ffi"
        )
    compiled_cache[key](source, op, destination, dot, n_threads, groups, cu_stream)


frost_head_reduce.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_head_group_reduce_dot_product.set_name_prefix("cudnn", remove_cutlass_symbol=False)
