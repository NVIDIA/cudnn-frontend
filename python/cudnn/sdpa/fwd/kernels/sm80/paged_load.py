# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80 vector copies with independent K/V page-table addressing."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

from cutlass.base_dsl.typing import Pointer
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import load_tile_2d

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.jit
def load_kv_tile(
    smem_dst,
    gmem_src,
    rows: cutlass.Constexpr[int],
    elems_per_row: cutlass.Constexpr[int],
    gmem_row_stride_elems,
    tidx,
    *,
    num_threads: cutlass.Constexpr[int],
    elems_per_copy: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    cache: nvvm.LoadCacheModifier = nvvm.LoadCacheModifier.CG,
    swizzle: cutlass.Constexpr[bool] = False,
    cp_size_bytes=None,
    valid_rows=None,
    valid_cols=None,
    row_base=None,
    page_table=None,
    page_size: cutlass.Constexpr[int] = 0,
    pool_base=None,
    page_stride=None,
    batch=None,
):
    if cutlass.const_expr(page_table is None):
        load_tile_2d(
            smem_dst,
            gmem_src,
            rows,
            elems_per_row,
            gmem_row_stride_elems,
            tidx,
            num_threads=num_threads,
            elems_per_copy=elems_per_copy,
            elem_bytes=elem_bytes,
            cache=cache,
            swizzle=swizzle,
            cp_size_bytes=cp_size_bytes,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
            row_base=row_base,
        )
    else:
        bytes_per_copy = elems_per_copy * elem_bytes
        chunks_per_row = elems_per_row // elems_per_copy
        if cutlass.const_expr(bytes_per_copy != 16 or rows * chunks_per_row % num_threads):
            raise ValueError("SM80 paged loads require complete 16-byte copy waves")
        table = Pointer(cutlass.make_array_view(page_table).data_ptr(), dtype=page_table.element_type)
        for i in cutlass.range_constexpr(rows * chunks_per_row // num_threads):
            chunk = i * num_threads + tidx
            row = chunk // chunks_per_row
            col = (chunk % chunks_per_row) * elems_per_copy
            logical_row = row_base + row
            live = cutlass.Int32((logical_row >= 0) & (logical_row < valid_rows))
            if cutlass.const_expr(valid_cols is not None):
                live = live * cutlass.Int32(col + elems_per_copy <= valid_cols)
            if cutlass.const_expr(cp_size_bytes is not None):
                live = live * cutlass.Int32(cp_size_bytes != 0)
            # An inactive prefetch or empty sequence must not read its page
            # table, even though cp.async itself would zero-fill the copy.
            offset = cutlass.Int64(0)
            if live != 0:
                table_offset = cutlass.Int64(batch) * cutlass.Int64(page_table.stride[0])
                table_offset += cutlass.Int64(logical_row // page_size) * cutlass.Int64(page_table.stride[1])
                page = cutlass.Int64(table[table_offset])
                offset = page * cutlass.Int64(page_stride)
                offset += cutlass.Int64(logical_row % page_size) * cutlass.Int64(gmem_row_stride_elems)
                offset += cutlass.Int64(col)
            smem_col = swizzle_xor_128b(row, col, elem_bytes=elem_bytes) if cutlass.const_expr(swizzle) else col
            dst = smem_dst.subview(row * elems_per_row + smem_col)
            nvvm.cp_async_shared_global(dst, pool_base + offset, bytes_per_copy, cache, cp_size=live * bytes_per_copy)
