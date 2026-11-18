# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Forward communication prefix: the dynamic task stream of X pulls (dispatch) and Y pushes (combine).

Private device module of ``_persistent_bf16_fwd_mega``.  The warp slots and the primitives are shared with
the backward (``_dynamic_task_stream``; read its docstring first); this module supplies the forward stream, the
task description and the load / store addresses (``_prepare``).

The stream (``_stream_task``: task number -> (direction, macrobatch, tile row)), in dependency order:
    pulls of the last macrobatch (processed first; nothing to wait for)
    for m = last .. 1:
        pulls of m - 1 on the ring positions (minibatches) m never used (a short last macrobatch)
        for each minibatch j of m, from the last to the first (the order the compute takes them):
            the pushes of (m, j) and the pulls of (m - 1, j) alternate tile row by tile row
            (both wait for the same thing: the routed Down of (m, j), which produced the Y rows and
            freed the ring position)
    pushes of macrobatch 0
One macrobatch: all pulls, then all pushes.  The X of macrobatch 0, the last one the stream processes, stays in
the ring for the backward.
With the combine in the routed Down epilogue (``epilogue_combine``) the stream keeps its pulls only, in the same order
(``_pull_stream_task``): the epilogue stores every finished row into the source rank's combine buffer itself.

Protocol (warp w of the COMM_WARPS of a communication CTA; it owns slot w and holds one tile at a time):
    task = w x comm SMs + CTA                      # the first task is dealt statically
    while task < tasks:
        direction, macro, row = _stream_task(task)
        if not ready(task):                        # routed Down of the minibatch it depends on
            flush arrive; wait
        for col_block in 0 .. col_blocks-1:
            prepare(row, col_block)                # every read of the tile, while the previous store is in flight
            wait slot w read; load(-> slot w); count (deferred); issue store(slot w)
        task = atomic_add(head, 1) + COMM_WARPS x comm SMs     # the next number, while the store is in flight
    flush arrive

* The head is the word after y_routed_done's 128-row blocks (zero at launch).
* Arrivals are batched per arrive unit (pull: minibatch; push: 128-row block, the granularity of
  y_routed_done): the count is released when the unit of the finished tile changes; a pull count after
  the warp's writes are complete, a push count after its stores have read the slot (y_routed_done gates
  the reuse of the local Y rows, which the load already read; the peers read the combine after the kernel).
* A task that is not ready stays with the warp that took it: every task depends only on tasks with
  smaller numbers (and on the compute of those), so the smallest unfinished task is always ready.

Every size, count, index and expected count is a runtime value.
"""

import dataclasses

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from . import _dynamic_task_stream as comm
from ._fwd_device_helpers import _ceil_div_i32, _peer_address
from .persistent_bf16_fwd_contract import CLUSTER_SIZE, COMM_TILE_SHAPES, MLP_TILE_COLUMNS, MLP_TILE_ROWS

PULL = comm.PULL
PUSH = comm.PUSH
DONE_BLOCK_ROWS = MLP_TILE_ROWS // CLUSTER_SIZE  # y_routed_done granularity (128 rows)
assert all(DONE_BLOCK_ROWS % shape.rows == 0 for shape in COMM_TILE_SHAPES)  # a tile counts toward one block

# Interface read by the mega kernel: shared memory of the communication CTA.  The tile shape
# (``CommSlots.shape``, compile time) also sizes the readiness counts the compute CTAs wait for: one arrival
# per tile (``comm.row_tiles`` x ``comm.column_blocks``).
ARENA_BYTES = comm.ARENA_BYTES
MBARRIERS = comm.MBARRIERS


@dataclasses.dataclass(frozen=True)
class FwdCommArgs:
    """Global tensors and runtime scalars of the forward prefix (built by the mega kernel)."""

    x_peer_table: cute.Tensor  # X base address of every rank (pull source)
    combine_peer_table: cute.Tensor  # combine-buffer base address of every rank (push destination)
    schedule_peer_rank: cute.Tensor  # per routed row: source rank, -1 for a padding row
    schedule_peer_token_idx: cute.Tensor  # per routed row: route index on the source rank
    x_routed: cute.Tensor  # local X ring (pull destination)
    y_routed: cute.Tensor  # local Y ring (push source)
    x_routed_ready: cute.Tensor  # [minibatch] pulled tiles (Gate/Up waits)
    y_routed_ready: cute.Tensor  # [minibatch] routed Down tiles (push waits)
    y_routed_done: cute.Tensor  # [128-row block] pushed tiles (next macrobatch's Down waits)
    num_tokens: Int32
    macrobatch_size: Int32
    minibatch_size: Int32
    hidden_size: Int32
    topk: Int32
    has_tail: bool = False  # compile time: H has a 128-wide tail, one more Down tile per row block


# ---------------------------------------------------------------- shapes (all runtime)


@cute.jit
def _macro_rows(macro: Int32, num_tokens: Int32, macrobatch_size: Int32) -> Int32:
    """Routed rows of ``macro`` (the last macrobatch may be short)."""
    return cutlass.min(Int32(macrobatch_size), num_tokens - macro * Int32(macrobatch_size))


@cute.jit
def _down_tiles(minibatch: Int32, num_tokens: Int32, minibatch_size: Int32, hidden_size: Int32, has_tail: cutlass.Constexpr = False) -> Int32:
    """y_routed_ready count of a complete minibatch: all routed Down tiles of both cluster CTAs."""
    rows = cutlass.min(Int32(minibatch_size), num_tokens - minibatch * Int32(minibatch_size))
    column_blocks = Int32(hidden_size // MLP_TILE_COLUMNS)
    if cutlass.const_expr(has_tail):
        # A 128-wide H tail is one more Down tile, arrived on by both cluster CTAs like every other.
        column_blocks = _ceil_div_i32(hidden_size, Int32(MLP_TILE_COLUMNS))
    return _ceil_div_i32(rows, Int32(MLP_TILE_ROWS)) * column_blocks * Int32(CLUSTER_SIZE)


# ---------------------------------------------------------------- task description (the direction-specific part)


@cute.jit
def _wait_target(
    direction: Int32, macro: Int32, row_start: Int32, last_macro: Int32, num_tokens: Int32, macrobatch_size: Int32, minibatch_size: Int32
) -> Int32:
    """Minibatch whose y_routed_ready the segment waits for, or -1 (no dependency).
    PULL: routed Down of the previous macrobatch consumed these X-ring rows (none for the last
    macrobatch, or when the previous macrobatch had no such rows).  PUSH: routed Down of this
    macrobatch produced these Y rows."""
    minibatch = Int32(-1)
    if direction == Int32(PULL):
        if macro < last_macro:
            previous = macro + Int32(1)
            if row_start < _macro_rows(previous, num_tokens, macrobatch_size):
                minibatch = (previous * Int32(macrobatch_size) + row_start) // Int32(minibatch_size)
    else:
        minibatch = (macro * Int32(macrobatch_size) + row_start) // Int32(minibatch_size)
    return minibatch


@cute.jit
def _arrive_unit(direction: Int32, macro: Int32, row_start: Int32, macrobatch_size: Int32, minibatch_size: Int32) -> Int32:
    """Counter index a finished tile counts toward, or -1 (nobody waits).  PULL: x_routed_ready[minibatch]
    (Gate/Up waits for the minibatch's tiles).  PUSH: y_routed_done[128-row block] (the next
    macrobatch's Down reuses the Y rows; nothing to signal for the last-processed macrobatch)."""
    unit = Int32(-1)
    if direction == Int32(PULL):
        unit = (macro * Int32(macrobatch_size) + row_start) // Int32(minibatch_size)
    else:
        if macro > Int32(0):
            unit = (macro * Int32(macrobatch_size) + row_start) // Int32(DONE_BLOCK_ROWS)
    return unit


# ---------------------------------------------------------------- arrive address


@cute.jit
def _arrive_address(args: FwdCommArgs, direction: Int32, macro: Int32, row: Int32) -> Int64:
    """Counter word a finished tile counts toward (0: nobody waits): PULL x_routed_ready[unit],
    PUSH y_routed_done[unit] (``_arrive_unit``)."""
    unit = _arrive_unit(direction, macro, row, args.macrobatch_size, args.minibatch_size)
    address = Int64(0)
    if direction == Int32(PULL):
        address = comm.counter_address(args.x_routed_ready, unit)
    else:
        address = comm.counter_address(args.y_routed_done, unit)
    return address


# ---------------------------------------------------------------- prepare (lanes)


@cute.jit
def _prepare(args: FwdCommArgs, slots: comm.CommSlots, direction: Int32, macro: Int32, row: Int32, col_block: Int32) -> comm.TileCopy:
    """This lane's copy of the tile (row ``row`` of ``macro``, column block ``col_block``), every read of its load
    and store done here, while the warp's previous store is in flight.  PULL: from the peer X row (the route's
    token) into the local X ring (an unrouted row is zeroed); PUSH: from the local Y row to the peer combine
    buffer (an unrouted row moves nothing).  The lanes past the tile's rows move nothing."""
    lane_idx = Int32(cute.arch.lane_idx())
    row_bytes = slots.tile_bytes(args.hidden_size, col_block)
    column = Int64(col_block) * Int64(slots.shape.row_chunk_bytes)
    schedule_row = macro * args.macrobatch_size + row + lane_idx
    peer_rank = slots.lane_entry(args.schedule_peer_rank, schedule_row, lane_idx)
    routed = Int32(cute.arch.popc(cute.arch.vote_ballot_sync(peer_rank >= Int32(0))))
    ring_row = Int64(row + lane_idx) * Int64(args.hidden_size) * Int64(2) + column
    source = Int64(0)
    destination = Int64(0)
    zero = Int32(0)
    wait_writes = Int32(0)  # PUSH: y_routed_done gates the reuse of the local Y rows
    if direction == Int32(PULL):
        wait_writes = Int32(1)  # Gate/Up reads the x_routed rows
        if slots.owns_row(lane_idx) != Int32(0):
            destination = args.x_routed.iterator.toint() + ring_row
            if peer_rank >= Int32(0):
                route_index = args.schedule_peer_token_idx[schedule_row]
                source = _peer_address(args.x_peer_table, peer_rank) + Int64(route_index // args.topk) * Int64(args.hidden_size) * Int64(2) + column
            else:
                zero = Int32(1)
    else:
        if peer_rank >= Int32(0):
            route_index = args.schedule_peer_token_idx[schedule_row]
            source = args.y_routed.iterator.toint() + ring_row
            destination = _peer_address(args.combine_peer_table, peer_rank) + Int64(route_index) * Int64(args.hidden_size) * Int64(2) + column
    return comm.TileCopy(
        source=source,
        destination=destination,
        zero=zero,
        row_bytes=row_bytes,
        tx_bytes=routed * row_bytes,
        side_address=Int64(0),
        side_value=Float32(0.0),
        arrive=_arrive_address(args, direction, macro, row),
        wait_writes=wait_writes,
    )


# ---------------------------------------------------------------- the stream


@cute.jit
def _reverse_tile(position: Int32, count: Int32, mini_tiles: Int32) -> Int32:
    """Tile row at ``position`` of a macrobatch of ``count`` tile rows taken minibatch by minibatch from the last
    (possibly short) one to the first, tile rows in order inside a minibatch (the compute takes the minibatches in
    this order)."""
    minibatches = (count + mini_tiles - Int32(1)) // mini_tiles
    first = count - (minibatches - Int32(1)) * mini_tiles  # tile rows of the last minibatch, taken first
    tile = (minibatches - Int32(1)) * mini_tiles + position
    if position >= first:
        k = position - first
        tile = (minibatches - Int32(2) - k // mini_tiles) * mini_tiles + k % mini_tiles
    return tile


@cute.jit
def _stream_task(index: Int32, last_macro: Int32, last_tiles: Int32, full_tiles: Int32, mini_tiles: Int32) -> tuple[Int32, Int32, Int32]:
    """(direction, macrobatch, tile row) of task ``index`` of the stream (module docstring): all pulls of the last
    macrobatch (``last_tiles`` tile rows, every other macrobatch ``full_tiles``); then for each macrobatch
    m = last .. 1: the pulls of m - 1 on the ring positions (minibatches) m never used, then minibatch by minibatch
    the pushes of minibatch j of m and the pulls of minibatch j of m - 1 (the same ring position, freed by m's Down
    of j), alternating tile row by tile row; then the pushes of macrobatch 0.  Inside a macrobatch the minibatches
    go from the last to the first (``_reverse_tile``), as the compute takes them.
    2 x (last_tiles + last_macro x full_tiles) tasks."""
    direction = Int32(PULL)
    macro = last_macro
    tile = _reverse_tile(index, last_tiles, mini_tiles)
    if index >= last_tiles:
        j = index - last_tiles
        m = last_macro
        count = last_tiles  # tile rows of m (whose pushes this block moves)
        if j >= last_tiles + full_tiles:
            j2 = j - last_tiles - full_tiles
            block = j2 // (Int32(2) * full_tiles)
            m = last_macro - Int32(1) - block
            count = full_tiles
            j = j2 - block * Int32(2) * full_tiles
        direction = Int32(PUSH)
        macro = m
        tile = _reverse_tile(j, count, mini_tiles)
        if m > Int32(0):
            minibatches = (count + mini_tiles - Int32(1)) // mini_tiles
            first = count - (minibatches - Int32(1)) * mini_tiles  # tile rows of m's last minibatch
            unused = full_tiles - minibatches * mini_tiles  # tile rows of m - 1 on ring positions m never used
            if j < unused:
                direction = Int32(PULL)
                macro = m - Int32(1)
                tile = (full_tiles // mini_tiles - Int32(1) - j // mini_tiles) * mini_tiles + j % mini_tiles
            else:
                # step k: the pushes of m's minibatch (minibatches - 1 - k) (step 0: its short last one), then
                # mini_tiles pulls of m - 1 on the same ring position
                jj = j - unused
                step = Int32(0)
                position = jj
                pushes = first
                push_base = Int32(0)
                if jj >= first + mini_tiles:
                    k = jj - first - mini_tiles
                    step = Int32(1) + k // (Int32(2) * mini_tiles)
                    position = k % (Int32(2) * mini_tiles)
                    pushes = mini_tiles
                    push_base = first + (step - Int32(1)) * mini_tiles
                # inside the step pushes and pulls alternate tile row by tile row (push, pull, push, ...), the remaining
                # pulls after the pushes run out: consecutive task numbers mix both kinds
                pull_row = position - pushes
                is_push = Int32(0)
                if position < Int32(2) * pushes:
                    pull_row = position // Int32(2)
                    if position % Int32(2) == Int32(0):
                        is_push = Int32(1)
                if is_push == Int32(1):
                    tile = _reverse_tile(push_base + position // Int32(2), count, mini_tiles)
                else:
                    direction = Int32(PULL)
                    macro = m - Int32(1)
                    tile = (minibatches - Int32(1) - step) * mini_tiles + pull_row
    return direction, macro, tile


@cute.jit
def _pull_stream_task(index: Int32, last_macro: Int32, last_tiles: Int32, full_tiles: Int32, mini_tiles: Int32) -> tuple[Int32, Int32, Int32]:
    """The pulls of ``_stream_task`` alone, in its order (``epilogue_combine``): the pulls of the last macrobatch,
    then those of every earlier macrobatch, each from the last minibatch to the first; the ring positions a short
    last macrobatch never used come first, as in the full stream.  last_tiles + last_macro x full_tiles tasks."""
    macro = last_macro
    tile = _reverse_tile(index, last_tiles, mini_tiles)
    if index >= last_tiles:
        k = index - last_tiles
        macro = last_macro - Int32(1) - k // full_tiles
        tile = _reverse_tile(k % full_tiles, full_tiles, mini_tiles)
    return Int32(PULL), macro, tile


@cute.jit
def _wait_index(args: FwdCommArgs, direction: Int32, macro: Int32, row: Int32, last_macro: Int32) -> Int32:
    """y_routed_ready index the task of row ``row`` of ``macro`` waits for, or -1 (``_wait_target`` of its
    minibatch)."""
    row_start = row // args.minibatch_size * args.minibatch_size
    return _wait_target(direction, macro, row_start, last_macro, args.num_tokens, args.macrobatch_size, args.minibatch_size)


@cute.jit
def run_comm_prefix(args: FwdCommArgs, slots: comm.CommSlots, comm_cta: Int32, num_comm_sms: Int32, epilogue_combine: cutlass.Constexpr = False) -> None:
    """The dynamic task stream (module docstring): each of the COMM_WARPS warps takes its first task statically
    (warp x comm SMs + CTA) and every later one with an atomic add on the head, so a warp that finishes early takes
    more; a warp waits for a taken task that is not ready, then moves its tiles one at a time through its own
    slot.  ``epilogue_combine`` (compile time): the routed Down epilogue pushes the combine, the stream moves its
    pulls only."""
    comm.init(slots)

    num_tokens = args.num_tokens
    macrobatch_size = args.macrobatch_size
    last_macro = _ceil_div_i32(num_tokens, macrobatch_size) - Int32(1)
    last_tiles = comm.row_tiles(slots.shape, _macro_rows(last_macro, num_tokens, macrobatch_size))
    full_tiles = comm.row_tiles(slots.shape, macrobatch_size)
    mini_tiles = comm.row_tiles(slots.shape, args.minibatch_size)
    if last_macro == Int32(0):
        mini_tiles = full_tiles  # one macrobatch: minibatches in order (_reverse_tile is then the identity)
    if cutlass.const_expr(epilogue_combine):
        tasks = last_tiles + last_macro * full_tiles
    else:
        tasks = Int32(2) * (last_tiles + last_macro * full_tiles)
    col_blocks = comm.column_blocks(slots.shape, args.hidden_size)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    warps_total = Int32(comm.COMM_WARPS) * num_comm_sms
    head_address = comm.counter_address(args.y_routed_done, args.y_routed_done.shape[0] - Int32(1))
    slot_phase = Int32(0)  # the phase of the warp's slot mbarrier
    batch = comm.no_arrivals()  # finished tiles not yet arrived
    if warp_idx < Int32(comm.COMM_WARPS):
        task = warp_idx * num_comm_sms + comm_cta
        while task < tasks:
            if cutlass.const_expr(epilogue_combine):
                direction, macro, tile_row = _pull_stream_task(task, last_macro, last_tiles, full_tiles, mini_tiles)
            else:
                direction, macro, tile_row = _stream_task(task, last_macro, last_tiles, full_tiles, mini_tiles)
            row = tile_row * Int32(slots.shape.rows)
            target = _wait_index(args, direction, macro, row, last_macro)
            if target >= Int32(0):
                required = _down_tiles(target, args.num_tokens, args.minibatch_size, args.hidden_size, args.has_tail)
                if comm.reached(args.y_routed_ready, target, required) == Int32(0):
                    # not ready: arrive what we finished, then wait
                    batch = batch.flush()
                    comm.wait(args.y_routed_ready, target, required)
            for col_block in cutlass.range(col_blocks):
                # every read of the tile while the previous store is in flight, then the load at once
                copy = _prepare(args, slots, direction, macro, row, col_block)
                comm.wait_slot_read()  # the previous tile's stores have read slot w
                batch = comm.move(slots, copy, warp_idx, slot_phase, batch)
                slot_phase = slot_phase ^ Int32(1)
            # the next task of the stream after the statically dealt first ones (the last stores are in flight)
            taken = Int32(0)
            if cute.arch.lane_idx() == Int32(0):
                head = cute.make_ptr(Int32, head_address, cute.AddressSpace.gmem, assumed_align=4)
                taken = cute.arch.atomic_add(head, Int32(1), sem="relaxed", scope="gpu")
            task = cute.arch.shuffle_sync(taken, 0) + warps_total
        batch.flush()
    # in-flight stores (the last ones of every warp, and those of tiles nobody waits for) complete before the CTA exits
    cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.sync_threads()
