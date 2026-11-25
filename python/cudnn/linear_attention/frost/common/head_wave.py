# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Queues for independent KDA head waves; each work item retains its full recurrence."""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm

from cudnn.frost.tile_dsl.barrier import wait_on_dependent_grids
from .split_k import write_item


@cute.jit
def write_wave_items(divisor: cutlass.Constexpr[int], b_t: cutlass.Constexpr[int], phase, heads_out, cu_pieces, main_rows, dest, count):
    """Emit a head wave, placing pieces of at least 32 chunks before short ones.

    All 256 CTA threads participate. A prefix scan gives each row a unique
    destination: long rows grow from the front, short rows from the back.
    Only queue order changes; piece/head identities and recurrence bounds do not.
    The blocking CTA barriers publish and retire the eight warp counts before
    their next use. No atomics or additional global scratch are needed.
    """
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    warp = tid // 32
    warp_counts = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    front = cutlass.Int32(0)
    back = cutlass.Int32(0)
    for start in cutlass.range(0, count, 256):
        row = start + tid
        valid = row < count
        is_long = cutlass.Int32(0)
        if valid:
            slot = row // (heads_out // divisor)
            length = cu_pieces[slot + 1] - cu_pieces[slot]
            is_long = cutlass.Int32(1) if length >= 32 * b_t else cutlass.Int32(0)
        inclusive = is_long
        for offset in [1, 2, 4, 8, 16]:
            other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, inclusive, offset, 0, kind=nvvm.Shfl.UP))
            inclusive = inclusive + (other if lane >= offset else cutlass.Int32(0))
        if lane == 31:
            warp_counts[warp] = inclusive
        nvvm.barrier_cta_sync()
        total = cutlass.Int32(0)
        prefix = inclusive
        for w in cutlass.range_constexpr(8):
            value = warp_counts[w]
            total = total + value
            prefix = prefix + (value if w < warp else cutlass.Int32(0))
        if valid:
            dst = front + prefix - 1 if is_long != 0 else count - 1 - (back + tid - prefix)
            write_item(True, b_t, heads_out, cu_pieces, None, dest, dst, row * divisor + phase, chain=True, mRowBase=main_rows)
        front = front + total
        valid_rows = count - start
        valid_rows = valid_rows if valid_rows < 256 else cutlass.Int32(256)
        back = back + valid_rows - total
        nvvm.barrier_cta_sync()


@cute.kernel
def head_wave_items(
    divisor: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    phase: cutlass.Int32,
    heads_out: cutlass.Int32,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    source_count: cute.Tensor,
    dest: cute.Tensor,
    dest_count: cute.Tensor,
    scheduler_series: cute.Tensor,
    scheduler_bwd: cute.Tensor,
):
    """Regenerate one independent-head wave from the unchanged piece geometry.

    Every row owns one complete (piece, head) recurrence. Each output row has
    exactly one writer; the next same-stream kernel waits for this launch.
    Only summary storage whose last consumer has finished is reused.
    """
    wait_on_dependent_grids()
    tid = cute.arch.thread_idx()[0]
    count = source_count[0] // divisor
    if tid == 0:
        dest_count[0] = count
        scheduler_series[0] = 0
        scheduler_series[1] = 0
        scheduler_bwd[0] = 0
        scheduler_bwd[1] = 0
    write_wave_items(divisor, b_t, phase, heads_out, cu_pieces, main_rows, dest, count)


head_wave_items.set_name_prefix("cudnn", remove_cutlass_symbol=False)


@cute.kernel
def state_wave_items(
    divisor: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    phase: cutlass.Int32,
    heads_out: cutlass.Int32,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    scheduler_all: cute.Tensor,
):
    """Regenerate both queues for a complete state wave, without changing piece boundaries.

    Counts are separate from the immutable row-base tables, whose final entries
    retain the full-head totals. Every queue row has one writer. Reset every
    scheduler ticket pair only after the preceding wave has completed.
    """
    wait_on_dependent_grids()
    tid = cute.arch.thread_idx()[0]
    n = main_rows[main_rows.shape[0] - 1] // divisor
    ns = summary_rows[summary_rows.shape[0] - 1] // divisor
    if tid == 0:
        main_count[0] = n
        summary_count[0] = ns
    if tid < scheduler_all.shape[0]:
        scheduler_all[tid] = 0
    write_wave_items(divisor, b_t, phase, heads_out, cu_pieces, main_rows, work_items, n)
    for row in cutlass.range(tid, ns, 256):
        write_item(
            True, b_t, heads_out, cu_pieces, None, work_items_summary, row, row * divisor + phase, chain=True, mRowBase=summary_rows, mSlotRows=main_rows
        )


state_wave_items.set_name_prefix("cudnn", remove_cutlass_symbol=False)
