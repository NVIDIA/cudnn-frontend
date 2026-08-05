# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
import cutlass.cute as cute


@cute.jit
def get_swizzled_col(
    row: cutlass.Int32,
    col: cutlass.Int32,
    row_stride: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Return the physical SMEM column for an XOR-swizzled row-major tile.

    The XOR is applied at the 16-byte boundary for all element widths.
    ``elem_bytes`` selects the element-domain shift and swizzle chunk size.
    """
    row_stride_bytes = row_stride * elem_bytes
    chunk_bytes = 32
    sw_bits = 1
    row_shift = 2
    if row_stride_bytes % 128 == 0:
        chunk_bytes = 128
        sw_bits = 3
        row_shift = 0
    elif row_stride_bytes % 64 == 0:
        chunk_bytes = 64
        sw_bits = 2
        row_shift = 1
    chunk_size = chunk_bytes // elem_bytes
    elems_per_16b = 16 // elem_bytes
    sw_base = elems_per_16b.bit_length() - 1
    chunk = col // chunk_size
    col_in_chunk = col % chunk_size
    bit_msk = (1 << sw_bits) - 1
    return chunk * chunk_size + (col_in_chunk ^ (((row >> row_shift) & bit_msk) << sw_base))


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
