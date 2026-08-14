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
stores one word back.  Serves the f16/bf16 ``[total, HO, D]`` tensor grads
(dQ/dK for GVA, dK/dV for GQA) and the fp32 ``[total, HO]`` Gate/Beta grads.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from cutlass.cute.runtime import from_dlpack

from .host import get_dtype
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16

BLOCK = 256


@cute.kernel
def head_reduce_kernel(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    total_words: cutlass.Int64,
    out_row_words: cutlass.Int64,
    out_head_words: cutlass.Int64,
    h_count: cutlass.Constexpr[int],
    r: cutlass.Constexpr[int],
    inner_words: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    gw = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(cutlass.Int32(tidx))
    if gw < total_words:
        seg = gw // cutlass.Int64(inner_words)
        w_off = gw - seg * cutlass.Int64(inner_words)
        base = seg * cutlass.Int64(r * inner_words) + w_off
        t_idx = seg // cutlass.Int64(h_count)
        h_idx = seg - t_idx * cutlass.Int64(h_count)
        out_off = t_idx * out_row_words + h_idx * out_head_words + w_off
        if cutlass.const_expr(io_dtype == cutlass.Float32):
            in_p = cute.recast_ptr(mIn.iterator, dtype=cutlass.Float32)
            out_p = cute.recast_ptr(mOut.iterator, dtype=cutlass.Float32)
            acc = (in_p + base).load()
            for i in cutlass.range_constexpr(r - 1):
                acc = acc + (in_p + (base + (i + 1) * inner_words)).load()
            (out_p + out_off).store(acc)
        else:
            in_p = cute.recast_ptr(mIn.iterator, dtype=cutlass.Int32)
            out_p = cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32)
            acc_lo, acc_hi = f16x2_to_f32((in_p + base).load(), dtype=io_dtype)
            for i in cutlass.range_constexpr(r - 1):
                lo, hi = f16x2_to_f32((in_p + (base + (i + 1) * inner_words)).load(), dtype=io_dtype)
                acc_lo = acc_lo + lo
                acc_hi = acc_hi + hi
            (out_p + out_off).store(fp32_to_fp16(acc_lo, acc_hi, dtype=io_dtype))


@cute.jit
def launch(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    total_words: cutlass.Int64,
    out_row_words: cutlass.Int64,
    out_head_words: cutlass.Int64,
    grid_x: cutlass.Int32,
    h_count: cutlass.Constexpr[int],
    r: cutlass.Constexpr[int],
    inner_words: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    head_reduce_kernel(mIn, mOut, total_words, out_row_words, out_head_words, h_count, r, inner_words, io_dtype).launch(
        grid=(grid_x, 1, 1),
        block=(BLOCK, 1, 1),
        stream=stream,
    )


compiled_cache = {}


def head_group_reduce(src, dst, *, stream) -> None:
    """Reduce ``src (total, HO, D)`` into ``dst (total, H, D)`` — or the
    rank-2 ``(total, HO)`` into ``(total, H)`` — by summing each group of
    ``r = HO // H`` consecutive heads (fp32 accumulation).

    ``src`` is contiguous (kernel-internal wide buffer); ``dst`` needs a
    stride-1 innermost dim with free outer strides (f16/bf16 outer strides
    must be even — word-pair stores). Same-dtype (f16/bf16, or fp32),
    DLPack-compatible CUDA tensors; the f16/bf16 inner extent ``D`` must be
    even.  Compile-cache-and-replay per ``(dtype, HO, H, D)``."""
    if len(src.shape) == 2:
        total, HO = src.shape
        D = 1
        H = dst.shape[1]
        shape_ok = len(dst.shape) == 2 and dst.shape[0] == total
    else:
        total, HO, D = src.shape
        H = dst.shape[1]
        shape_ok = len(dst.shape) == 3 and dst.shape[0] == total and dst.shape[2] == D
    if not shape_ok:
        raise ValueError(f"head_group_reduce: shape mismatch {tuple(src.shape)} -> {tuple(dst.shape)}")
    if H <= 0 or HO % H != 0 or HO <= H:
        raise ValueError(f"head_group_reduce: bad head group HO={HO} H={H}")
    io_dtype = get_dtype(src.dtype)
    if str(src.dtype).split(".")[-1] != str(dst.dtype).split(".")[-1]:
        raise ValueError(f"head_group_reduce: dtype mismatch {src.dtype} vs {dst.dtype}")
    is_fp32 = io_dtype == cutlass.Float32
    if not is_fp32 and D % 2 != 0:
        raise ValueError(f"head_group_reduce: D={D} must be even for f16/bf16")
    r = HO // H
    inner_words = D if is_fp32 else D // 2
    total_words = total * H * inner_words
    grid_x = -(-total_words // BLOCK)
    cu_stream = cuda.CUstream(int(stream))

    dst_strides = tuple(dst.stride())
    if not is_fp32 and any(st % 2 != 0 for st, sz in zip(dst_strides[:-1], dst.shape[:-1]) if sz != 1):
        raise ValueError(f"head_group_reduce: f16/bf16 dst outer strides must be even (word-pair stores), got {dst_strides}")
    if dst.shape[-1] != 1 and dst_strides[-1] != 1:
        raise ValueError(f"head_group_reduce: dst innermost dim must be stride-1, got strides {dst_strides}")
    out_row_words = dst_strides[0] if is_fp32 else dst_strides[0] // 2
    out_head_words = (dst_strides[1] if is_fp32 else dst_strides[1] // 2) if len(dst.shape) == 3 else 1

    key = (str(src.dtype).split(".")[-1], HO, H, D)
    if key not in compiled_cache:

        src_c = from_dlpack(src, assumed_align=4)
        src_c.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(src.shape))), divisibility=1)
        compiled_cache[key] = cute.compile(
            launch,
            src_c,
            from_dlpack(dst, assumed_align=4).mark_layout_dynamic(leading_dim=len(dst.shape) - 1),
            cutlass.Int64(total_words),
            cutlass.Int64(out_row_words),
            cutlass.Int64(out_head_words),
            cutlass.Int32(grid_x),
            H,
            r,
            inner_words,
            io_dtype,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    compiled_cache[key](src, dst, total_words, out_row_words, out_head_words, grid_x, cu_stream)
