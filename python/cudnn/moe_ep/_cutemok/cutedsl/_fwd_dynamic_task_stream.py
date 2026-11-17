# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Forward communication prefix: the dynamic task stream of X pulls (dispatch) and Y pushes (combine).

Private device module of ``_persistent_bf16_fwd_mega``.  The slot pool and the primitives are shared with
the backward (``_dynamic_task_stream``; read its docstring first); this module supplies the forward stream, the
task description and the load / store addresses.

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

Protocol (one warp of COMM_WARPS per communication CTA):
    task = warp x comm SMs + CTA                   # the first task is dealt statically
    while task < tasks:
        direction, macro, row = _stream_task(task)
        if not ready(task):                        # routed Down of the minibatch it depends on
            store(prev); release; flush arrive; wait (holding no slot)
        for col_block in 0 .. col_blocks-1:
            slot = acquire(pool)                   # first come; blocks only while holding no slot
            load(row, col_block -> slot); store(prev); release(prev)
        task = atomic_add(head, 1) + COMM_WARPS x comm SMs     # the next number of the stream
    store(prev); release; flush arrive

* The head is the word after y_routed_done's 128-row blocks (zero at launch).
* Arrivals are batched per arrive unit (pull: minibatch; push: 128-row block, the granularity of
  y_routed_done): the count is released when the unit of the finished tile changes.
* A task that is not ready stays with the warp that took it: every task depends only on tasks with
  smaller numbers (and on the compute of those), so the smallest unfinished task is always ready.

Every size, count, index and expected count is a runtime value.
"""

import dataclasses

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from . import _dynamic_task_stream as pool
from ._fwd_device_helpers import _ceil_div_i32, _peer_address
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw, zero_shared_bulk
from .persistent_bf16_fwd_contract import CLUSTER_SIZE, COMM_TILE_SHAPES, MLP_TILE_COLUMNS, MLP_TILE_ROWS

PULL = pool.PULL
PUSH = pool.PUSH
DONE_BLOCK_ROWS = MLP_TILE_ROWS // CLUSTER_SIZE  # y_routed_done granularity (128 rows)
assert all(DONE_BLOCK_ROWS % shape.rows == 0 for shape in COMM_TILE_SHAPES)  # a tile counts toward one block

# Interface read by the mega kernel: shared memory of the communication CTA.  The tile shape
# (``SlotPool.shape``, compile time) also sizes the readiness counts the compute CTAs wait for: one arrival
# per tile (``pool.row_tiles`` x ``pool.column_blocks``).
ARENA_BYTES = pool.ARENA_BYTES
MBARRIERS = pool.MBARRIERS
SCRATCH_WORDS = pool.SCRATCH_WORDS


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
def _arrive_address(args: FwdCommArgs, tile: pool.InFlightTile) -> Int64:
    """Counter word a finished tile counts toward (0: nobody waits): PULL x_routed_ready[unit],
    PUSH y_routed_done[unit] (``_arrive_unit``)."""
    unit = _arrive_unit(tile.task, tile.batch, tile.row, args.macrobatch_size, args.minibatch_size)
    address = Int64(0)
    if tile.task == Int32(PULL):
        address = pool.counter_address(args.x_routed_ready, unit)
    else:
        address = pool.counter_address(args.y_routed_done, unit)
    return address


# ---------------------------------------------------------------- load / store (lanes)


@cute.jit
def _load(args: FwdCommArgs, slot_pool: pool.SlotPool, tile: pool.InFlightTile) -> None:
    """load: arm the slot's mbarrier and issue this lane's row load: PULL from the peer X row, PUSH from
    the local Y row (unrouted rows, and the lanes past the tile's rows, load nothing)."""
    lane_idx = Int32(cute.arch.lane_idx())
    tile_bytes = slot_pool.tile_bytes(args.hidden_size, tile.col_block)
    schedule_row = tile.batch * args.macrobatch_size + tile.row + lane_idx
    row_chunk = slot_pool.row_chunk(tile.slot, lane_idx)
    mbarrier = slot_pool.mbarriers + tile.slot
    peer_rank = slot_pool.lane_entry(args.schedule_peer_rank, schedule_row, lane_idx)
    routed = Int32(cute.arch.popc(cute.arch.vote_ballot_sync(peer_rank >= Int32(0))))
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, routed * tile_bytes)
    cute.arch.sync_warp()
    if peer_rank >= Int32(0):
        source = (
            args.y_routed.iterator.toint()
            + Int64(tile.row + lane_idx) * Int64(args.hidden_size) * Int64(2)
            + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
        )
        if tile.task == Int32(PULL):
            route_index = args.schedule_peer_token_idx[schedule_row]
            source = (
                _peer_address(args.x_peer_table, peer_rank)
                + Int64(route_index // args.topk) * Int64(args.hidden_size) * Int64(2)
                + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
            )
        tma_load_1d_raw(row_chunk, source, mbarrier, tile_bytes)


@cute.jit
def _store_row(
    args: FwdCommArgs,
    slot_pool: pool.SlotPool,
    tile: pool.InFlightTile,
    lane_idx: Int32,
    schedule_row: Int32,
    peer_rank: Int32,
    row_chunk: cute.Pointer,
    tile_bytes: Int32,
) -> None:
    """One lane's part of ``_store``, after the slot's loads arrived: store this lane's row chunk and wait for
    the store."""
    if tile.task == Int32(PULL):
        if peer_rank < Int32(0):
            # one st.bulk per lane (word-by-word stores would hit one bank for all lanes)
            zero_shared_bulk(row_chunk, slot_pool.shape.row_chunk_bytes)
            cute.arch.fence_proxy("async.shared", space="cta")
        destination = (
            args.x_routed.iterator.toint()
            + Int64(tile.row + lane_idx) * Int64(args.hidden_size) * Int64(2)
            + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
        )
        tma_store_1d_raw(destination, row_chunk, tile_bytes)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)  # x_routed_ready may be released after this tile
    else:
        if peer_rank >= Int32(0):
            route_index = args.schedule_peer_token_idx[schedule_row]
            destination = (
                _peer_address(args.combine_peer_table, peer_rank)
                + Int64(route_index) * Int64(args.hidden_size) * Int64(2)
                + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
            )
            tma_store_1d_raw(destination, row_chunk, tile_bytes)
            cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def _store(args: FwdCommArgs, slot_pool: pool.SlotPool, tile: pool.InFlightTile, slot_phase: Int32) -> None:
    """store: wait for the slot's loads, then PULL: store the row into the local X ring (unrouted rows
    zero) and wait for the store; PUSH: store the row to the peer combine buffer and wait until
    the slot has been read.  The slot is reusable when this returns: the whole warp waits for the loads and
    syncs after the stores, the lanes past the tile's rows store nothing in between."""
    lane_idx = Int32(cute.arch.lane_idx())
    tile_bytes = slot_pool.tile_bytes(args.hidden_size, tile.col_block)
    schedule_row = tile.batch * args.macrobatch_size + tile.row + lane_idx
    row_chunk = slot_pool.row_chunk(tile.slot, lane_idx)
    peer_rank = slot_pool.lane_entry(args.schedule_peer_rank, schedule_row, lane_idx)
    cute.arch.mbarrier_wait(slot_pool.mbarriers + tile.slot, slot_phase)
    if cutlass.const_expr(slot_pool.shape.rows == pool.WARP_LANES):
        _store_row(args, slot_pool, tile, lane_idx, schedule_row, peer_rank, row_chunk, tile_bytes)
    else:
        if lane_idx < Int32(slot_pool.shape.rows):
            _store_row(args, slot_pool, tile, lane_idx, schedule_row, peer_rank, row_chunk, tile_bytes)
    cute.arch.sync_warp()


@cute.jit
def _finish(args: FwdCommArgs, slot_pool: pool.SlotPool, tile: pool.InFlightTile, batch: pool.ArriveBatch) -> pool.ArriveBatch:
    """store(tile); release(tile.slot); count the tile toward its counter (the batch is released
    when the counter changes)."""
    slot_phase = pool.phase(slot_pool, tile.slot)
    _store(args, slot_pool, tile, slot_phase)
    pool.release(slot_pool, tile.slot, slot_phase)
    return batch.add(_arrive_address(args, tile))


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
def run_comm_prefix(args: FwdCommArgs, slot_pool: pool.SlotPool, comm_cta: Int32, num_comm_sms: Int32, epilogue_combine: cutlass.Constexpr = False) -> None:
    """The dynamic task stream (module docstring): each of the COMM_WARPS warps takes its first task statically
    (warp x comm SMs + CTA) and every later one with an atomic add on the head, so a warp that finishes early takes
    more; a warp waits, holding no slot, for a taken task that is not ready.  ``epilogue_combine`` (compile time): the
    routed Down epilogue pushes the combine, the stream moves its pulls only."""
    pool.init(slot_pool)

    num_tokens = args.num_tokens
    macrobatch_size = args.macrobatch_size
    last_macro = _ceil_div_i32(num_tokens, macrobatch_size) - Int32(1)
    last_tiles = pool.row_tiles(slot_pool.shape, _macro_rows(last_macro, num_tokens, macrobatch_size))
    full_tiles = pool.row_tiles(slot_pool.shape, macrobatch_size)
    mini_tiles = pool.row_tiles(slot_pool.shape, args.minibatch_size)
    if last_macro == Int32(0):
        mini_tiles = full_tiles  # one macrobatch: minibatches in order (_reverse_tile is then the identity)
    if cutlass.const_expr(epilogue_combine):
        tasks = last_tiles + last_macro * full_tiles
    else:
        tasks = Int32(2) * (last_tiles + last_macro * full_tiles)
    col_blocks = pool.column_blocks(slot_pool.shape, args.hidden_size)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    warps_total = Int32(pool.COMM_WARPS) * num_comm_sms
    head_address = pool.counter_address(args.y_routed_done, args.y_routed_done.shape[0] - Int32(1))
    prev = pool.no_tile()  # the tile whose store is not finished yet
    batch = pool.no_arrivals()  # finished tiles not yet released
    if warp_idx < Int32(pool.COMM_WARPS):
        task = warp_idx * num_comm_sms + comm_cta
        while task < tasks:
            if cutlass.const_expr(epilogue_combine):
                direction, macro, tile_row = _pull_stream_task(task, last_macro, last_tiles, full_tiles, mini_tiles)
            else:
                direction, macro, tile_row = _stream_task(task, last_macro, last_tiles, full_tiles, mini_tiles)
            row = tile_row * Int32(slot_pool.shape.rows)
            target = _wait_index(args, direction, macro, row, last_macro)
            if target >= Int32(0):
                required = _down_tiles(target, args.num_tokens, args.minibatch_size, args.hidden_size, args.has_tail)
                if pool.reached(args.y_routed_ready, target, required) == Int32(0):
                    # not ready: release what we hold, then wait
                    if prev.slot >= Int32(0):
                        batch = _finish(args, slot_pool, prev, batch)
                        prev = prev.stored()
                    batch = batch.flush()
                    pool.wait(args.y_routed_ready, target, required)
            for col_block in cutlass.range(col_blocks):
                # acquire: block only while holding no slot
                slot = pool.acquire(slot_pool, Int32(0))
                if slot < Int32(0):
                    if prev.slot >= Int32(0):
                        batch = _finish(args, slot_pool, prev, batch)
                        prev = prev.stored()
                    slot = pool.acquire(slot_pool, Int32(1))
                tile = pool.InFlightTile(slot=slot, task=direction, batch=macro, row=row, col_block=col_block)
                _load(args, slot_pool, tile)
                if prev.slot >= Int32(0):
                    batch = _finish(args, slot_pool, prev, batch)
                prev = tile
            # the next task of the stream after the statically dealt first ones
            taken = Int32(0)
            if cute.arch.lane_idx() == Int32(0):
                head = cute.make_ptr(Int32, head_address, cute.AddressSpace.gmem, assumed_align=4)
                taken = cute.arch.atomic_add(head, Int32(1), sem="relaxed", scope="gpu")
            task = cute.arch.shuffle_sync(taken, 0) + warps_total
        if prev.slot >= Int32(0):
            batch = _finish(args, slot_pool, prev, batch)
        batch.flush()
    # in-flight peer stores must complete before the CTA exits
    cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.sync_threads()
