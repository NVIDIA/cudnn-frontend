# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native row mapping for SM80 CTAs sharing one KV head across Q heads."""

import hashlib
import math
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.jit
def row_offset(row, pack_g: cutlass.Constexpr[int], row_stride, head_stride):
    return cutlass.Int64(row // pack_g) * cutlass.Int64(row_stride) + cutlass.Int64(row % pack_g) * cutlass.Int64(head_stride)


@cute.jit
def load_q(
    smem,
    source,
    row_base,
    row_stride,
    head_stride,
    tidx,
    valid_rows,
    valid_cols,
    rows: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    pack_g: cutlass.Constexpr[int],
):
    chunks = width // 8
    copy_threads = math.gcd(rows * chunks, threads)
    active = True if cutlass.const_expr(copy_threads == threads) else tidx < copy_threads
    if active:
        for wave in cutlass.range_constexpr(rows * chunks // copy_threads):
            chunk = wave * copy_threads + tidx
            row, col = chunk // chunks, (chunk % chunks) * 8
            packed_row = row_base + row
            live = cutlass.Int32(packed_row // pack_g < valid_rows)
            if cutlass.const_expr(valid_cols is not None):
                live = live * cutlass.Int32(col + 8 <= valid_cols)
            offset = row_offset(packed_row, pack_g, row_stride, head_stride) + cutlass.Int64(col)
            # Invalid rows zero-fill; keep the unused source inside the operand.
            if live == 0:
                offset = cutlass.Int64(0)
            dst = smem.subview(row * width + swizzle_xor_128b(row, col, elem_bytes=2))
            nvvm.cp_async_shared_global(dst, source + offset, 16, nvvm.LoadCacheModifier.CG, cp_size=live * 16)


@cute.jit
def fold_sinks(
    row_max,
    row_sum,
    accum,
    sinks,
    row_base,
    head_base,
    warp_m_base,
    lane_row,
    m_blocks: cutlass.Constexpr[int],
    n_frags: cutlass.Constexpr[int],
    pack_g: cutlass.Constexpr[int],
    natural: cutlass.Constexpr[bool],
):
    ptr = Pointer(cutlass.make_array_view(sinks).data_ptr(), dtype=sinks.element_type)
    for block in cutlass.range_constexpr(m_blocks):
        for half in cutlass.range_constexpr(2):
            row = row_base + warp_m_base + block * 16 + lane_row + half * 8
            head = head_base + row % pack_g
            sink = cutlass.Float32(ptr[cutlass.Int64(head) * cutlass.Int64(sinks.stride[0])])
            if cutlass.const_expr(natural):
                sink = sink * cutlass.Float32(1.4426950408889634)
            state = block * 2 + half
            if sink == cutlass.Float32.inf:
                row_max[state] = cutlass.Float32.inf
                row_sum[state] = cutlass.Float32(1.0)
                for n in cutlass.range_constexpr(n_frags):
                    for c in cutlass.range_constexpr(2):
                        accum[block * n_frags * 4 + n * 4 + half * 2 + c] = cutlass.Float32(0.0)
            elif sink != -cutlass.Float32.inf:
                lifted = cutlass.Float32(cute.math.max(row_max[state], sink, ftz=True))
                scale = cute.math.exp2(row_max[state] - lifted, fastmath=True)
                row_sum[state] = row_sum[state] * scale + cute.math.exp2(sink - lifted, fastmath=True)
                row_max[state] = lifted
                for n in cutlass.range_constexpr(n_frags):
                    for c in cutlass.range_constexpr(2):
                        at = block * n_frags * 4 + n * 4 + half * 2 + c
                        accum[at] = accum[at] * scale
