# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared warp-level primitives for the FROST SM120 SDPA backward kernels."""

from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from cudnn.frost.tile_dsl.mma import mma_m16n8k16_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


_LOG2E = 1.4426950408889634
_COPY_ELEMS = 8  # 16-byte gmem<->smem chunk (8 fp16/bf16)


@cute.jit
def tile_ptr(
    sbuf,
    row: cutlass.Int32,
    col: cutlass.Int32,
    *,
    chunk_elems: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
):
    """Element pointer into a swizzled smem tile."""
    chunk = col // chunk_elems
    in_col = col % chunk_elems
    off = chunk * (rows * chunk_elems) + row * chunk_elems + swizzle_xor(row, in_col, chunk_elems, 2)
    return sbuf.subview(off).data_ptr()


@cute.jit
def pack_half2(lo, hi, dtype: cutlass.Constexpr[Type[cutlass.Numeric]]):
    """Pack two fp32 into a 2-element io-dtype vector (one 4 B store)."""
    return cutlass.Vector.from_elements((lo.to(dtype), hi.to(dtype)), dtype)


@cute.jit
def load_a_frag(
    sbuf,
    k_chunk: cutlass.Constexpr[int],
    row0,
    lane,
    *,
    rows: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
):
    """ldmatrix.x4 one (16 x 16) row-major A fragment."""
    row = row0 + lane % 16
    col = k_chunk * 16 + (lane // 16) * 8
    return prims.ldmatrix(tile_ptr(sbuf, row, col, chunk_elems=chunk_elems, rows=rows), 4, prims.MMALayout.ROW)


@cute.jit
def load_a_frag_transposed(
    sbuf,
    k_chunk: cutlass.Constexpr[int],
    col0,
    lane,
    *,
    rows: cutlass.Constexpr[int],
    chunk_elems: cutlass.Constexpr[int],
):
    """ldmatrix.trans.x4: A[M, K] from a tile stored physically [K, M]."""
    row = k_chunk * 16 + lane % 16
    col = col0 + (lane // 16) * 8
    return prims.ldmatrix(tile_ptr(sbuf, row, col, chunk_elems=chunk_elems, rows=rows), 4, prims.MMALayout.COL)


@cute.jit
def copy16_smem_to_gmem(sptr, gptr):
    """One 16-byte smem->gmem chunk."""
    v = sptr.load(count=8)
    gptr.store(v, alignment=16)


@cute.jit
def mma_bstream(
    acc,
    a_frag,
    sB,
    *,
    b_k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    b_trans: cutlass.Constexpr[bool],
    b_rows: cutlass.Constexpr[int],
    b_chunk_elems: cutlass.Constexpr[int],
    lane,
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
    col_base=0,
    row_base=0,
):
    """One k=16 step of an (M x N) MMA, B streamed from smem via
    ldmatrix.x4 (2 adjacent 8-column n-frags per fetch).

    acc:    ``(M//16) * (N//8) * 4`` fp32, m-block major then n-frag.
    a_frag: ``(M//16) * 4`` Int32 (one 16x16 A fragment per m-block).
    """
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    PAIRS = N_FRAGS // 2
    a_stride = len(a_frag) // M_BLOCKS

    if cutlass.const_expr(b_trans):
        b_row = lane % 16
        n_offset = lane // 16
        layout_flag = prims.MMALayout.COL
    else:
        b_row = lane % 8
        b_col_subchunk = (lane // 8) % 2
        n_offset = lane // 16
        layout_flag = prims.MMALayout.ROW

    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        if cutlass.const_expr(b_trans):
            row = b_k_step * 16 + b_row
            col = (n_frag + n_offset) * 8 + col_base
        else:
            row = row_base + (n_frag + n_offset) * 8 + b_row
            col = b_k_step * 16 + b_col_subchunk * 8 + col_base
        b_ptr = tile_ptr(sB, row, col, chunk_elems=b_chunk_elems, rows=b_rows)
        b_v = prims.ldmatrix(b_ptr, 4, layout_flag)
        for m_block in cutlass.range_constexpr(M_BLOCKS):
            a_off = m_block * a_stride
            for half in cutlass.range_constexpr(2):
                s = (m_block * N_FRAGS + n_frag + half) * 4
                c0, c1, c2, c3 = mma_m16n8k16_f32(
                    a_frag[a_off + 0],
                    a_frag[a_off + 1],
                    a_frag[a_off + 2],
                    a_frag[a_off + 3],
                    b_v[half * 2 + 0],
                    b_v[half * 2 + 1],
                    acc[s + 0],
                    acc[s + 1],
                    acc[s + 2],
                    acc[s + 3],
                    ab_dtype,
                )
                acc[s + 0] = c0
                acc[s + 1] = c1
                acc[s + 2] = c2
                acc[s + 3] = c3


@cute.jit
def mma_abregs(
    acc,
    a_frag,
    b_frag,
    *,
    b_k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    """One k=16 MMA step with both operands resident in registers."""
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    PAIRS = N_FRAGS // 2
    a_stride = len(a_frag) // M_BLOCKS
    b_k_stride = PAIRS * 4

    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        b_off = b_k_step * b_k_stride + pair * 4
        for m_block in cutlass.range_constexpr(M_BLOCKS):
            a_off = m_block * a_stride
            for half in cutlass.range_constexpr(2):
                s = (m_block * N_FRAGS + n_frag + half) * 4
                c0, c1, c2, c3 = mma_m16n8k16_f32(
                    a_frag[a_off + 0],
                    a_frag[a_off + 1],
                    a_frag[a_off + 2],
                    a_frag[a_off + 3],
                    b_frag[b_off + half * 2 + 0],
                    b_frag[b_off + half * 2 + 1],
                    acc[s + 0],
                    acc[s + 1],
                    acc[s + 2],
                    acc[s + 3],
                    ab_dtype,
                )
                acc[s + 0] = c0
                acc[s + 1] = c1
                acc[s + 2] = c2
                acc[s + 3] = c3
