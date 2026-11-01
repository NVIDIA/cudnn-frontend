# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-input safety flags, tile bounds, and exact packed key masks."""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key
from .. import _metadata

DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FROST_SOURCE_DIGEST = DIGEST
TILE_M, MAX_SPAN = 128, 8192


@cute.kernel
def frost_qsa_scan_kv(k: cute.Tensor, v: cute.Tensor, partials: cute.Tensor, range_safe: cutlass.Constexpr = False):
    tid = cute.arch.thread_idx()[0]
    block = cute.arch.block_idx()[0]
    bad = Int32(0)
    for step in cutlass.range_constexpr(8):
        index = Int64(block) * 1024 + tid + step * 128
        if index < k.shape[0]:
            kb = Float32(k[index]).bitcast(Int32) & Int32(0x7F800000)
            vb = Float32(v[index]).bitcast(Int32) & Int32(0x7F800000)
            if (kb == Int32(0x7F800000)) or (vb == Int32(0x7F800000)):
                bad = Int32(1)
            # Hopper uses BF16 probabilities in dense tiles; extreme V needs the FP32 gather path.
            if cutlass.const_expr(range_safe):
                if cute.math.abs(Float32(v[index])) > 1e30:
                    bad = Int32(1)
    bad = cute.arch.warp_reduction_max(bad)
    if tid % 32 == 0:
        partials[Int64(block) * 4 + tid // 32] = bad


frost_qsa_scan_kv.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_qsa_tile_bounds(lo: cute.Tensor, hi: cute.Tensor, partials: cute.Tensor, modes: cute.Tensor, ranges: cute.Tensor):
    tid = cute.arch.thread_idx()[0]
    tile = cute.arch.block_idx()[0]
    row = tile * TILE_M + tid
    total = Int32(lo.shape[0])
    low, high, bad = total, Int32(-1), Int32(0)
    if row < total:
        left = cute.math.max(lo[row], Int32(0))
        right = cute.math.min(hi[row], total - 1)
        if left <= right:
            low, high = left, right
    for index in cutlass.range(tid, Int32(partials.shape[0]), 128):
        bad = bad | partials[index]
    low = cute.arch.warp_reduction(low, lambda x, y: cute.math.min(x, y))
    high = cute.arch.warp_reduction_max(high)
    bad = cute.arch.warp_reduction_max(bad)
    scratch = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(12), byte_alignment=4)
    if tid % 32 == 0:
        scratch[tid // 32] = low
        scratch[4 + tid // 32] = high
        scratch[8 + tid // 32] = bad
    cute.arch.sync_threads()
    if tid == 0:
        left, right, unsafe = total, Int32(-1), Int32(0)
        for warp in cutlass.range_constexpr(4):
            left = cute.math.min(left, scratch[warp])
            right = cute.math.max(right, scratch[4 + warp])
            unsafe = unsafe | scratch[8 + warp]
        begin, chunks = Int32(0), Int32(1)
        if left <= right:
            begin = left & Int32(-128)
            end = (right + 128) & Int32(-128)
            chunks = (end - begin) // 128
            if end - begin > MAX_SPAN:
                unsafe = Int32(1)
        modes[tile] = unsafe
        ranges[tile, 0], ranges[tile, 1] = begin, chunks


frost_qsa_tile_bounds.set_name_prefix("cudnn", remove_cutlass_symbol=True)


MAX_SORT = 4096


@cute.kernel
def frost_qsa_schedule(modes: cute.Tensor, ranges: cute.Tensor, order: cute.Tensor):
    """Rank query tiles by descending dense work so the largest launch first."""
    tid = Int32(cute.arch.thread_idx()[0])
    count = Int32(modes.shape[0])
    keys = utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout(MAX_SORT), byte_alignment=16)
    if count <= Int32(MAX_SORT):
        for tile in cutlass.range(tid, count, 128):
            work = Int32(0)
            if modes[tile] == 0:
                work = ranges[tile, 1]
            keys[tile] = work
        cute.arch.sync_threads()
        for tile in cutlass.range(tid, count, 128):
            mine = keys[tile]
            rank = Int32(0)
            for other in cutlass.range(count):
                theirs = keys[other]
                if (theirs > mine) or ((theirs == mine) and (other < tile)):
                    rank = rank + 1
            order[rank] = tile
    else:
        for tile in cutlass.range(tid, count, 128):
            order[tile] = tile


frost_qsa_schedule.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def frost_qsa_pack_bits(
    sel: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    bases: cute.Tensor,
    origins: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    bits: cute.Tensor,
    modes: cute.Tensor,
    ranges: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    vectorized: cutlass.Constexpr = False,
):
    tid = cute.arch.thread_idx()[0]
    row = Int64(cute.arch.block_idx()[0]) * 4 + tid // 32
    lane = Int32(tid % 32)
    total = Int32(bits.shape[0])
    if row < bits.shape[0]:
        tile = Int32(row // TILE_M)
        if modes[tile] == 0:
            first_word = (ranges[tile, 0] // 128) * 4
            stop_word = first_word + ranges[tile, 1] * 4
            low = cute.math.max(lo[row], Int32(0))
            high = cute.math.min(hi[row], total - 1)
            origin, base = origins[row], bases[row]
            for word_index in cutlass.range(first_word + lane, stop_word, 32, unroll=1):
                word = Int32(0)
                key0 = Int32(word_index) * 32
                if (low <= high) and (high >= key0) and (low <= key0 + 31):
                    block0 = base + ((key0 - origin) >> 2)
                    for group in cutlass.range_constexpr(9):
                        block = block0 + group
                        if (block >= 0) and (block < sel.shape[1]):
                            if sel[row, Int64(block)] != 0:
                                key = origin + 4 * (block - base)
                                left = cute.math.max(cute.math.max(key, key0), cute.math.max(low, first[block]))
                                right = cute.math.min(cute.math.min(key + 3, key0 + 31), cute.math.min(high, last[block]))
                                if left <= right:
                                    mask = ((Int32(1) << (right - left + 1)) - 1) << (left - key0)
                                    word = word | mask
                bits[row, Int64(word_index)] = word
        else:
            _metadata._compact_row(sel, lo, hi, first, last, ids, counts, row, lane, vectorized)


frost_qsa_pack_bits.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def prepare(
    k,
    v,
    sel,
    lo,
    hi,
    bases,
    origins,
    first,
    last,
    partials,
    modes,
    ranges,
    bits,
    order,
    ids,
    counts,
    stream: cuda.CUstream,
    vectorized: cutlass.Constexpr = False,
    range_safe: cutlass.Constexpr = False,
):
    frost_qsa_scan_kv(k, v, partials, range_safe).launch(grid=(cute.ceil_div(k.shape[0], 1024), 1, 1), block=(128, 1, 1), stream=stream)
    frost_qsa_tile_bounds(lo, hi, partials, modes, ranges).launch(grid=(cute.ceil_div(lo.shape[0], TILE_M), 1, 1), block=(128, 1, 1), stream=stream)
    frost_qsa_schedule(modes, ranges, order).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
    frost_qsa_pack_bits(sel, lo, hi, bases, origins, first, last, bits, modes, ranges, ids, counts, vectorized).launch(
        grid=(cute.ceil_div(bits.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream
    )


def compile_kernel(arch):
    def tensor(dtype, shape, alignment=4):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    total, blocks, words, elems, pieces, tiles = (cute.sym_int64() for _ in range(6))
    kv = tensor(cutlass.BFloat16, (elems,), 16)
    row, span = tensor(Int32, (total,)), tensor(Int32, (blocks,))
    return compile_cached(
        prepare,
        kv,
        kv,
        tensor(cutlass.Uint8, (total, blocks), 1),
        row,
        row,
        row,
        row,
        span,
        span,
        tensor(Int32, (pieces,)),
        tensor(Int32, (tiles,)),
        tensor(Int32, (tiles, 2)),
        tensor(Int32, (total, words)),
        tensor(Int32, (tiles,)),
        tensor(Int32, (total, blocks)),
        row,
        make_fake_stream(use_tvm_ffi_env_stream=False),
        arch == 107,
        cache_key=template_key(globals(), {"arch": arch, "source": DIGEST, "metadata": _metadata.FROST_SOURCE_DIGEST}, "prepare"),
        symbol="qsa_hybrid_prepare",
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
    )
