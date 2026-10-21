# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact each selection row on the launch stream."""

import hashlib
from functools import lru_cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cuda.bindings import driver as cuda

from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


@cute.jit
def _compact_chunk(sel, lo, hi, first, last, ids, row, lane, base, count, vectorized: cutlass.Constexpr):
    """Compact 16 blocks per lane in stable order."""
    packed = None
    if cutlass.const_expr(vectorized):
        offset = Int64(row) * Int64(sel.shape[1]) + Int64(base)
        packed = (sel.iterator.raw_ptr() + offset).load(count=16, alignment=1)

    low, high = lo[row], hi[row]
    chosen = []
    for item in cutlass.range_constexpr(16):
        block = base + item
        take = Boolean(False)
        if block < sel.shape[1]:
            marked = Boolean(False)
            if cutlass.const_expr(vectorized):
                marked = packed[item] != 0
            else:
                marked = sel[Int64(row), Int64(block)] != 0
            if marked:
                take = (last[block] >= low) & (first[block] <= high) & (low <= high)
        chosen.append(take)

    hits = Int32(0)
    for item in cutlass.range_constexpr(16):
        if chosen[item]:
            hits = hits + 1

    inclusive = hits
    for offset in (1, 2, 4, 8, 16):
        prior = cute.arch.shuffle_sync_up(inclusive, offset)
        if lane >= offset:
            inclusive = inclusive + prior
    output = count + inclusive - hits
    local = Int32(0)
    for item in cutlass.range_constexpr(16):
        if chosen[item]:
            ids[Int64(row), Int64(output + local)] = base + item
            local = local + 1
    return count + Int32(cute.arch.shuffle_sync(inclusive, 31))


@cute.jit
def _compact_row(
    sel: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    row: Int64,
    lane: Int32,
    vectorized: cutlass.Constexpr = False,
):
    low, high = lo[row], hi[row]
    count = Int32(0)
    if cutlass.const_expr(vectorized):
        bulk = (Int32(sel.shape[1]) // 512) * 512
        for start in cutlass.range(0, bulk, 512, unroll=1):
            count = _compact_chunk(sel, lo, hi, first, last, ids, row, lane, start + lane * 16, count, True)
        count = _compact_chunk(sel, lo, hi, first, last, ids, row, lane, bulk + lane * 16, count, False)
    else:
        lower_lanes = (Uint32(1) << Uint32(lane)) - Uint32(1)
        for start in cutlass.range(0, Int32(sel.shape[1]), 32, unroll=1):
            block = start + lane
            selected = Boolean(False)
            if block < sel.shape[1]:
                selected = (sel[Int64(row), Int64(block)] != 0) & (last[block] >= low) & (first[block] <= high) & (low <= high)
            ballot = Uint32(cute.arch.vote_ballot_sync(selected))
            if selected:
                offset = count + Int32(cute.arch.popc(ballot & lower_lanes))
                ids[Int64(row), Int64(offset)] = block
            count = count + Int32(cute.arch.popc(ballot))
    if lane == 0:
        counts[row] = count


@cute.kernel
def frost_qsa_compact(
    sel: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    mode,
    vectorized: cutlass.Constexpr = False,
):
    tid = cute.arch.thread_idx()[0]
    rows_per_warp = cutlass.const_expr(8 if mode is not None else 1)
    row0 = cute.arch.block_idx()[0] * (4 * rows_per_warp) + tid // 32
    lane = tid % 32
    active = row0 < sel.shape[0]
    if cutlass.const_expr(mode is not None):
        if active:
            active = mode[Int64(row0 // 128)] != 0
    if active:
        for step in cutlass.range(rows_per_warp, unroll=1):
            row = row0 + step * 4
            if row < sel.shape[0]:
                _compact_row(sel, lo, hi, first, last, ids, counts, Int64(row), Int32(lane), vectorized)


frost_qsa_compact.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _compact_host(sel, lo, hi, first, last, ids, counts, stream: cuda.CUstream, vectorized: cutlass.Constexpr = False):
    frost_qsa_compact(sel, lo, hi, first, last, ids, counts, None, vectorized).launch(
        grid=(cute.ceil_div(sel.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream
    )


@cute.jit
def _compact_gated_host(sel, lo, hi, first, last, ids, counts, mode, stream: cuda.CUstream, vectorized: cutlass.Constexpr = False):
    frost_qsa_compact(sel, lo, hi, first, last, ids, counts, mode, vectorized).launch(
        grid=(cute.ceil_div(sel.shape[0], 32), 1, 1), block=(128, 1, 1), stream=stream
    )


@lru_cache(maxsize=None)
def _compile_args(gated):
    """Cache storage-free signatures; compiled objects use the shared cache."""
    total, blocks = cute.sym_int64(), cute.sym_int64()

    def tensor(dtype, shape, alignment=4):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    sel = tensor(cutlass.Uint8, (total, blocks), 1)
    rows, spans = tensor(Int32, (total,)), tensor(Int32, (blocks,))
    ids = tensor(Int32, (total, blocks))
    args = (sel, rows, rows, spans, spans, ids, rows)
    if gated:
        args += (tensor(Int32, (cute.sym_int64(),)),)
    return (*args, make_fake_stream(use_tvm_ffi_env_stream=False))


def compile_kernel(arch, gated=False):
    key = template_key(globals(), locals(), "compile_kernel")
    return compile_cached(
        _compact_gated_host if gated else _compact_host,
        *_compile_args(gated),
        arch == 107,
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_compact",
    )


@cute.kernel
def frost_qsa_compact_group(
    sel: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    total: Int32,
    NQ: cutlass.Constexpr[int],
    BLK_BITS: cutlass.Constexpr[int],
):
    """Union the selections of NQ consecutive queries; tag each block with its owners."""
    tid = cute.arch.thread_idx()[0]
    group = cute.arch.block_idx()[0] * 4 + tid // 32
    lane = tid % 32
    if group < ids.shape[0]:
        base = Int32(group) * Int32(NQ)
        rows, lows, highs, live = [], [], [], []
        for i in cutlass.range_constexpr(NQ):
            row = base + i
            inside = row < total
            if not inside:
                row = total - 1
            rows.append(row)
            lows.append(lo[row])
            highs.append(hi[row])
            live.append(inside)
        count = Int32(0)
        lower_lanes = (Uint32(1) << Uint32(lane)) - Uint32(1)
        for start in cutlass.range(0, Int32(sel.shape[1]), 32, unroll=1):
            block = start + lane
            mask = Int32(0)
            if block < sel.shape[1]:
                begin, end = first[block], last[block]
                for i in cutlass.range_constexpr(NQ):
                    taken = live[i] & (end >= lows[i]) & (begin <= highs[i]) & (lows[i] <= highs[i])
                    if taken:
                        taken = sel[Int64(rows[i]), Int64(block)] != 0
                    if taken:
                        mask = mask | Int32(1 << i)
            selected = mask != Int32(0)
            ballot = Uint32(cute.arch.vote_ballot_sync(selected))
            if selected:
                offset = count + Int32(cute.arch.popc(ballot & lower_lanes))
                ids[Int64(group), Int64(offset)] = block | (mask << Int32(BLK_BITS))
            count = count + Int32(cute.arch.popc(ballot))
        if lane == 0:
            counts[group] = count


frost_qsa_compact_group.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _compact_group_host(
    sel, lo, hi, first, last, ids, counts, total: Int32, stream: cuda.CUstream, NQ: cutlass.Constexpr[int], BLK_BITS: cutlass.Constexpr[int]
):
    frost_qsa_compact_group(sel, lo, hi, first, last, ids, counts, total, NQ, BLK_BITS).launch(
        grid=(cute.ceil_div(ids.shape[0], 4), 1, 1), block=(128, 1, 1), stream=stream
    )


@lru_cache(maxsize=None)
def _compile_group_args():
    """Cache storage-free grouped signatures."""
    total, blocks, groups = cute.sym_int64(), cute.sym_int64(), cute.sym_int64()

    def tensor(dtype, shape, alignment=4):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    sel = tensor(cutlass.Uint8, (total, blocks), 1)
    rows, spans = tensor(Int32, (total,)), tensor(Int32, (blocks,))
    ids = tensor(Int32, (groups, blocks))
    gcounts = tensor(Int32, (groups,))
    return (
        sel,
        rows,
        rows,
        spans,
        spans,
        ids,
        gcounts,
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
    )


def compile_group_kernel(arch, nq, blk_bits):
    key = template_key(globals(), {"arch": arch, "nq": nq, "blk_bits": blk_bits}, "compile_group_kernel")
    return compile_cached(
        _compact_group_host,
        *_compile_group_args(),
        nq,
        blk_bits,
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_compact_group",
    )
