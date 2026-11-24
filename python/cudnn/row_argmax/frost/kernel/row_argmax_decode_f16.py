# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The row argmax DECODE kernel (bf16 in; one int32 or int64 result per row out; one launch): per row the lowest column of the
row maximum, or the entry of an id table at that column.  Config in ``row_argmax_decode_config.py``, the host entry in
``row_argmax/frost/row_argmax_decode.py``.  Arch-portable PTX (global loads / stores, ``setp``, ``selp``, the warp reductions of
``tile_dsl.rowwise``, ``bar.sync``); the arch axis is an admission fact of the host entry, not a directory.

THREAD BLOCK: one per row; ``cfg.THREADS`` threads in ``cfg.WARPS`` FULL warps; thread ``t`` owns the ``cfg.PASSES`` vectors
``t + THREADS x pass`` of its row -- the eight contiguous elements at column ``8 (t + THREADS x pass)`` each.  Every thread has the
same role up to the reduction; thread 0 alone writes the row's result.

PIPELINE, in order, for every thread:

1. ``cfg.PASSES`` 128-bit global loads of the thread's vectors (bf16 -> fp32 is exact), in ascending column order.
2. A running ``(best, index)`` from ``(-inf, 0)``: per element ``take = value > best``, then two selects.  The compare is ORDERED
   and STRICT: a NaN is never taken (so ``best`` is never a NaN), and of equal values the first -- the lowest column -- stays.
3. ``block_reduce((best, index), ..., kind="max_index")``: the block's maximum and the LOWEST ``index`` among the threads that hold
   it, in every thread.  A row without a value above ``-inf`` leaves every thread at ``(-inf, 0)``: column 0.
4. Thread 0: without an id table ONE 32-bit global store of the column; with one, one load of the table's element at the column
   (32 or 64 bits by the table kind) and one store of those bits.

THE RULE IS THE CONTRACT, AND IT IS EXACT.  There is no arithmetic on a value: compares, selects and bit copies only.  The scan
must stay a compare and a select (a floating-point maximum instruction in its place would return the other operand for a NaN and
lose the column), and the compare must stay strict (``>=`` would move a tie to the highest column).  ``+0`` and ``-0`` compare
equal here and inside the reduction, so the lowest zero column wins whichever zero the reduction reports as the maximum.

BATCH INVARIANCE AND REPLAY.  A thread block reads its own row and, in thread 0, one element of the shared table; nothing of
another row; the reduction never leaves the block; there is no atomic and no cross-row scratch.  The row index enters ADDRESSES
only.

BARRIER TABLE.  No mbarrier, no cluster, no ``cp.async`` group, no loop with a run-time trip count.  ONE synchronization object:

    barrier                         kind                 participants      issuing lanes            guard
    the thread-block barrier (3)    named barrier 0      cfg.THREADS       WARPS x 32, once each    none

``SUM(issuing lanes) = 32 x WARPS == THREADS``: every thread the barrier counts reaches it exactly once on every path.  The
barrier and the slot accesses of the table below are emitted by ``block_reduce``; the kernel reaches that call with every thread,
outside any run-time branch.  Thread 0's branch sits behind it and holds no barrier.

SHARED-MEMORY TABLE.  ONE buffer; no tensor memory, no descriptor:

    buffer          dtype x elems        writer                              reader                             refilled in a launch
    warp results    fp32 x 2 x WARPS     lane 0 of warp w: word w (its       lanes 0 .. WARPS - 1 of every      no: written once, read
                                         maximum) and WARPS + w (the bits    warp: words lane and WARPS +       once -> nothing
                                         of its index), before the barrier   lane, behind the barrier           releases it

Lanes address the words directly at a 4-byte stride (one word per lane in an access phase); no swizzle because no descriptor
reads it.

PDL SITES: none.  The launch is ordinary (no programmatic-stream-serialization attribute) and the kernel holds neither a wait nor
a release: it starts when its stream predecessor has completed.

OPERANDS are device ADDRESSES (``Int64``) and one byte stride: the kernel addresses flat and reads no extent, so no shape, stride
or alignment of an operand enters the compiled kernel (the host entry checks them before every launch).  The table kind is a
compile-time argument beside the config: it selects the load and the store width of step 4.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.memory

from cudnn.frost.tile_dsl.rowwise import block_reduce, row_load8, slot_words
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v2, st_global, st_global_v2

from .row_argmax_decode_config import INT_BYTES, RESULT_DTYPE, VEC, VEC_BYTES, WARP, validate_cfg, validate_table_kind

FP32, INT32, INT64, BF16 = cutlass.Float32, cutlass.Int32, cutlass.Int64, cutlass.BFloat16
NEGATIVE_INFINITY_BITS = 0xFF800000


@cute.kernel
def frost_row_argmax(
    cfg: cutlass.Constexpr,
    table: cutlass.Constexpr,  # the id table kind: "none", "int32" or "int64"
    x: cutlass.Int64,  # device addresses: bf16 [rows, WIDTH], a row contiguous
    x_row: cutlass.Int64,  # byte stride between the rows of x
    ids: cutlass.Int64,  # the id table, WIDTH elements of the table kind's dtype, contiguous; unread without a table
    out: cutlass.Int64,  # one result per row, contiguous: int32, or the table's dtype
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane = tid % WARP
    warp = tid // WARP
    slots = cutlass.memory.SmemAllocator().allocate_tensor(FP32, cute.make_layout(slot_words("max_index", cfg.WARPS)), byte_alignment=16)
    row64 = row.to(INT64)
    mine = x + row64 * x_row + tid.to(INT64) * INT64(VEC_BYTES)  # the thread's vector of the first pass

    best = cutlass.Uint32(NEGATIVE_INFINITY_BITS).bitcast(FP32)
    index = INT32(0)
    for scan in cutlass.range_constexpr(cfg.PASSES):
        values = row_load8(mine + INT64(scan * cfg.THREADS * VEC_BYTES), dtype=BF16)
        column = (tid + scan * cfg.THREADS) * VEC
        for element in cutlass.range_constexpr(VEC):
            # rule contract: an ordered, strict compare and two selects -- a NaN is never taken, the lowest column of equals stays
            take = values[element] > best
            best = values[element] if take else best
            index = (column + element) if take else index

    # rule contract: the lowest index among the threads that hold the block's maximum.  The call holds the ONE thread-block barrier:
    # all cfg.THREADS threads reach it here, outside any run-time branch.
    _, winner = block_reduce((best, index), lane, warp, slots, cfg.WARPS, kind="max_index", order=None, index_limit=cfg.WIDTH)

    if tid == 0:
        result = out + row64 * INT64(INT_BYTES[RESULT_DTYPE[table]])
        if cutlass.const_expr(table == "none"):
            st_global(result, winner, INT32)
        if cutlass.const_expr(table == "int32"):
            st_global(result, ld_global(ids + winner.to(INT64) * INT64(INT_BYTES["int32"]), INT32), INT32)
        if cutlass.const_expr(table == "int64"):  # a bit copy of the 64-bit id as its two 32-bit halves
            low, high = ld_global_v2(ids + winner.to(INT64) * INT64(INT_BYTES["int64"]), INT32)
            st_global_v2(result, [low, high], INT32)


frost_row_argmax.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def launch(
    cfg: cutlass.Constexpr,
    table: cutlass.Constexpr,
    x: cutlass.Int64,
    x_row: cutlass.Int64,
    ids: cutlass.Int64,
    out: cutlass.Int64,
    rows: cutlass.Int32,
    stream: cuda.CUstream,
):
    """One ordinary launch: grid ``(rows, 1, 1)``, ``cfg.THREADS`` threads.  ``rows``, every address and the row stride are runtime
    arguments; the config and the table kind are the compile key.  A config the body does not have (a width that is not whole
    passes) and an unknown table kind are refused here, at trace time, before any operation is emitted."""
    validate_cfg(cfg)
    validate_table_kind(table)
    frost_row_argmax(cfg, table, x, x_row, ids, out).launch(grid=(rows, 1, 1), block=(cfg.THREADS, 1, 1), stream=stream)
