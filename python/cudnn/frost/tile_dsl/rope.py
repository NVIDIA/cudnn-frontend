# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
import cutlass.cute as cute

from .swizzle import swizzle_xor_128b


@cute.jit
def rope_rotate_smem_tile(
    sbuf,
    rope_cs_ptr,
    row_base,
    *,
    rows: cutlass.Constexpr[int],
    d_qk: cutlass.Constexpr[int],
    tidx,
    threads: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr[int] = 2,
):
    d2 = d_qk // 2
    epr = d_qk
    n_pairs = rows * d2
    assert n_pairs % threads == 0, f"RoPE: rows*d2 ({n_pairs}) must be divisible by threads ({threads})"
    per_thread = n_pairs // threads
    for u in cutlass.range_constexpr(per_thread):
        pid = tidx + cutlass.Int32(u * threads)
        row = pid // d2
        i = pid % d2
        pos64 = (row_base + row).to(cutlass.Int64)
        cs_idx = pos64 * cutlass.Int64(d2 * 2) + i.to(cutlass.Int64) * cutlass.Int64(2)
        c = rope_cs_ptr[cs_idx]
        s = rope_cs_ptr[cs_idx + cutlass.Int64(1)]
        swz_lo = swizzle_xor_128b(row, i, elem_bytes=elem_bytes)
        swz_hi = swizzle_xor_128b(row, i + cutlass.Int32(d2), elem_bytes=elem_bytes)
        off_lo = row * cutlass.Int32(epr) + swz_lo
        off_hi = row * cutlass.Int32(epr) + swz_hi
        x_lo = sbuf[off_lo].to(cutlass.Float32)
        x_hi = sbuf[off_hi].to(cutlass.Float32)
        o_lo = x_lo * c - x_hi * s
        o_hi = x_hi * c + x_lo * s
        sbuf[off_lo] = o_lo.to(io_dtype)
        sbuf[off_hi] = o_hi.to(io_dtype)
