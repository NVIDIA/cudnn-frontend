# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
import cutlass.cute as cute


@cute.jit
def swizzle_xor_128b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_elems = 16 // elem_bytes
    chunk_idx = col_elem // chunk_elems
    in_chunk = col_elem % chunk_elems
    swz_chunk = chunk_idx ^ (row & 7)
    return swz_chunk * chunk_elems + in_chunk


@cute.jit
def swizzle_lin_128b(lin, *, row_stride_log2: cutlass.Constexpr[int], elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_log2 = cutlass.const_expr((16 // elem_bytes).bit_length() - 1)
    shift = cutlass.const_expr(row_stride_log2 - chunk_log2)
    mask = cutlass.const_expr(0x7 << chunk_log2)
    return lin ^ ((lin >> shift) & mask)


@cute.jit
def swizzle_lin_S(lin, *, bbits: cutlass.Constexpr[int], mbase: cutlass.Constexpr[int], sshift: cutlass.Constexpr[int]):
    yyy = (lin >> cutlass.const_expr(mbase + sshift)) & cutlass.const_expr((1 << bbits) - 1)
    return lin ^ (yyy << cutlass.const_expr(mbase))
