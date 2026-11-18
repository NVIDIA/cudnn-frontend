# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Forward communication: the BF16 forward's hooks on the communication task stream (``_comm_stream.run_stream``).

Private device module of ``_persistent_bf16_fwd_mega``.  The loop, the per-warp buffers and the arrivals are the
shared stream's (``_comm_stream``; read its docstring first); this module supplies what a block is: the task order,
what a block waits for, where it is loaded from, what is done with it once it landed and which counter it arrives
on.  A block is one communication tile of the kernel's ``comm_tile_shape`` (contract CommTileShape) in the warp's
one 32 KiB buffer, each row one bulk copy of 1 KiB (32 x 512) or 2 KiB (16 x 1024).  A task is one tile row of one
macrobatch, claimed STREAM_GROUP_BLOCKS column blocks at a time.

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

A block:
    pull  wait  routed Down of the previous macrobatch on the same X-ring rows (none for the last macrobatch)
          load  the peer's X row (route / topk) per lane, a padding row zero-filled (``_comm_stream.pull_row``)
          then  bulk-store the rows into the local X ring; arrive on x_routed_ready[minibatch]
    push  wait  routed Down of this minibatch (y_routed_ready)
          load  the local Y row per lane (``_comm_stream.load_local_row``)
          then  bulk-store the rows into the source rank's combine buffer; arrive on y_routed_done[128-row
                block] (none for macrobatch 0: nobody overwrites its Y rows)

* The head is the word after y_routed_done's 128-row blocks (zero at launch).
* What the compute side waits for: one arrival per block, x_routed_ready[minibatch] = block rows of the
  minibatch x ``column_blocks``, y_routed_done[128-row block] = its block rows x ``column_blocks``
  (CommTileShape.row_tiles / column_blocks).
* A task that is not ready stays with the warp that took it: every task depends only on tasks with
  smaller numbers (and on the compute of those), so the smallest unfinished task is always ready.

Every size, count, index and expected count is a runtime value.
"""

import enum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from . import _comm_stream
from ._comm_stream import (
    STREAM_GROUP_BLOCKS,
    STREAM_PULL,
    STREAM_WARP_BYTES,
    STREAM_WARPS,
    _divide_u32,
    _reciprocal_u32,
)
from ._fwd_device_helpers import _ceil_div_i32, _peer_address
from .persistent_bf16_fwd_contract import CLUSTER_SIZE, MLP_TILE_COLUMNS, MLP_TILE_ROWS

PULL = STREAM_PULL
PUSH = _comm_stream.STREAM_PUSH
DONE_BLOCK_ROWS = MLP_TILE_ROWS // CLUSTER_SIZE  # y_routed_done granularity (128 rows); a block counts toward one

# Interface read by the mega kernel: shared memory of the communication CTA (the buffers alias the compute arena).
ARENA_BYTES = STREAM_WARPS * STREAM_WARP_BYTES
MBARRIERS = STREAM_WARPS  # one per warp: its one buffer (``block_shape`` buffers = 1)


@cute.jit
def column_blocks(hidden_size: Int32, columns: cutlass.Constexpr) -> Int32:
    """Column blocks of one row (``columns`` the tile shape's, compile time): the blocks of a task (the last may be
    short), and per block row the arrivals a readiness counter gets."""
    return _ceil_div_i32(hidden_size, Int32(columns))


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
    direction: Int32, macro: Int32, mini: Int32, row_start: Int32, last_macro: Int32, num_tokens: Int32, macrobatch_size: Int32, minibatches_per_macro: Int32
) -> Int32:
    """Minibatch whose y_routed_ready a block of minibatch ``mini`` of ``macro`` (first row ``row_start`` inside the
    macrobatch) waits for, or -1 (no dependency).  PULL: routed Down of the previous macrobatch consumed these X-ring
    rows (none for the last macrobatch, or when the previous macrobatch had no such rows).  PUSH: routed Down of this
    macrobatch produced these Y rows."""
    minibatch = Int32(-1)
    if direction == Int32(PULL):
        if macro < last_macro:
            previous = macro + Int32(1)
            if row_start < _macro_rows(previous, num_tokens, macrobatch_size):
                minibatch = previous * minibatches_per_macro + mini
    else:
        minibatch = macro * minibatches_per_macro + mini
    return minibatch


@cute.jit
def _arrive_unit(direction: Int32, macro: Int32, row_start: Int32, macrobatch_size: Int32, minibatch_size: Int32, reciprocal_minibatch: Int32) -> Int32:
    """Counter index a finished block counts toward, or -1 (nobody waits).  PULL: x_routed_ready[minibatch]
    (Gate/Up waits for the minibatch's blocks).  PUSH: y_routed_done[128-row block] (the next
    macrobatch's Down reuses the Y rows; nothing to signal for the last-processed macrobatch)."""
    unit = Int32(-1)
    if direction == Int32(PULL):
        unit = _divide_u32(macro * Int32(macrobatch_size) + row_start, minibatch_size, reciprocal_minibatch)
    else:
        if macro > Int32(0):
            unit = (macro * Int32(macrobatch_size) + row_start) // Int32(DONE_BLOCK_ROWS)
    return unit


# ---------------------------------------------------------------- the stream order
#
# ``geometry`` (``ForwardStream.run`` builds it once): (last_macro, last_tiles, full_tiles, mini_tiles,
# last_minibatches, full_minibatches, reciprocal_mini_tiles, reciprocal_full_tiles, reciprocal_2f, reciprocal_2mini)
# -- the block rows (tiles of the order) of the last / every other macrobatch and of a minibatch, the minibatches of the last / every
# other macrobatch, and the ``_reciprocal_u32`` of mini_tiles, full_tiles, 2 x full_tiles and 2 x mini_tiles (no
# division per task).


@cute.jit
def _reverse_tile(position: Int32, count: Int32, minibatches: Int32, mini_tiles: Int32, reciprocal_mini_tiles: Int32) -> Int32:
    """Tile row at ``position`` of a macrobatch of ``count`` tile rows (``minibatches`` minibatches) taken minibatch by
    minibatch from the last (possibly short) one to the first, tile rows in order inside a minibatch (the compute
    takes the minibatches in this order)."""
    first = count - (minibatches - Int32(1)) * mini_tiles  # tile rows of the last minibatch, taken first
    tile = (minibatches - Int32(1)) * mini_tiles + position
    if position >= first:
        k = position - first
        q = _divide_u32(k, mini_tiles, reciprocal_mini_tiles)
        tile = (minibatches - Int32(2) - q) * mini_tiles + (k - q * mini_tiles)
    return tile


@cute.jit
def _stream_task(index: Int32, geometry: tuple) -> tuple[Int32, Int32, Int32]:
    """(direction, macrobatch, tile row) of task ``index`` of the stream (module docstring): all pulls of the last
    macrobatch (``last_tiles`` tile rows, every other macrobatch ``full_tiles``); then for each macrobatch
    m = last .. 1: the pulls of m - 1 on the ring positions (minibatches) m never used, then minibatch by minibatch
    the pushes of minibatch j of m and the pulls of minibatch j of m - 1 (the same ring position, freed by m's Down
    of j), alternating tile row by tile row; then the pushes of macrobatch 0.  Inside a macrobatch the minibatches
    go from the last to the first (``_reverse_tile``), as the compute takes them.
    2 x (last_tiles + last_macro x full_tiles) tasks."""
    last_macro, last_tiles, full_tiles, mini_tiles, last_minibatches, full_minibatches, reciprocal_mini_tiles, _, reciprocal_2f, reciprocal_2mini = geometry
    direction = Int32(PULL)
    macro = last_macro
    tile = _reverse_tile(index, last_tiles, last_minibatches, mini_tiles, reciprocal_mini_tiles)
    if index >= last_tiles:
        j = index - last_tiles
        m = last_macro
        count = last_tiles  # tile rows of m (whose pushes this block moves)
        minibatches = last_minibatches
        if j >= last_tiles + full_tiles:
            j2 = j - last_tiles - full_tiles
            block = _divide_u32(j2, Int32(2) * full_tiles, reciprocal_2f)
            m = last_macro - Int32(1) - block
            count = full_tiles
            minibatches = full_minibatches
            j = j2 - block * Int32(2) * full_tiles
        direction = Int32(PUSH)
        macro = m
        tile = _reverse_tile(j, count, minibatches, mini_tiles, reciprocal_mini_tiles)
        if m > Int32(0):
            first = count - (minibatches - Int32(1)) * mini_tiles  # tile rows of m's last minibatch
            unused = full_tiles - minibatches * mini_tiles  # tile rows of m - 1 on ring positions m never used
            if j < unused:
                direction = Int32(PULL)
                macro = m - Int32(1)
                q = _divide_u32(j, mini_tiles, reciprocal_mini_tiles)
                tile = (full_minibatches - Int32(1) - q) * mini_tiles + (j - q * mini_tiles)
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
                    q = _divide_u32(k, Int32(2) * mini_tiles, reciprocal_2mini)
                    step = Int32(1) + q
                    position = k - q * (Int32(2) * mini_tiles)
                    pushes = mini_tiles
                    push_base = first + (step - Int32(1)) * mini_tiles
                # inside the step pushes and pulls alternate tile row by tile row (push, pull, push, ...), the remaining
                # pulls after the pushes run out: consecutive task numbers mix both kinds
                pull_row = position - pushes
                is_push = Int32(0)
                if position < Int32(2) * pushes:
                    pull_row = position >> Int32(1)
                    if (position & Int32(1)) == Int32(0):
                        is_push = Int32(1)
                if is_push == Int32(1):
                    tile = _reverse_tile(push_base + (position >> Int32(1)), count, minibatches, mini_tiles, reciprocal_mini_tiles)
                else:
                    direction = Int32(PULL)
                    macro = m - Int32(1)
                    tile = (minibatches - Int32(1) - step) * mini_tiles + pull_row
    return direction, macro, tile


@cute.jit
def _pull_stream_task(index: Int32, geometry: tuple) -> tuple[Int32, Int32, Int32]:
    """The pulls of ``_stream_task`` alone, in its order (``epilogue_combine``): the pulls of the last macrobatch,
    then those of every earlier macrobatch, each from the last minibatch to the first; the ring positions a short
    last macrobatch never used come first, as in the full stream.  last_tiles + last_macro x full_tiles tasks."""
    (
        last_macro,
        last_tiles,
        full_tiles,
        mini_tiles,
        last_minibatches,
        full_minibatches,
        reciprocal_mini_tiles,
        reciprocal_full_tiles,
        reciprocal_2f,
        reciprocal_2mini,
    ) = geometry
    macro = last_macro
    tile = _reverse_tile(index, last_tiles, last_minibatches, mini_tiles, reciprocal_mini_tiles)
    if index >= last_tiles:
        k = index - last_tiles
        q = _divide_u32(k, full_tiles, reciprocal_full_tiles)
        macro = last_macro - Int32(1) - q
        tile = _reverse_tile(k - q * full_tiles, full_tiles, full_minibatches, mini_tiles, reciprocal_mini_tiles)
    return Int32(PULL), macro, tile


class _Fo(enum.IntEnum):
    """Positions in the hooks' operand list (``ForwardStream.run`` builds it, ``_comm_stream.run_stream`` hands it to
    every hook)."""

    X_PEER_TABLE = 0
    COMBINE_PEER_TABLE = 1
    X_ROUTED = 2
    Y_ROUTED = 3
    X_ROUTED_READY = 4
    Y_ROUTED_READY = 5
    Y_ROUTED_DONE = 6
    LAST_MACRO = 7
    GEOMETRY = 8  # the stream order's tuple (``_stream_task``)
    NUM_TOKENS = 9
    HIDDEN_SIZE = 10
    TOPK = 11
    MACROBATCH_SIZE = 12
    MINIBATCH_SIZE = 13
    MINIBATCHES_PER_MACRO = 14
    RECIPROCAL_MINIBATCH = 15
    RECIPROCAL_TOPK = 16
    COUNT = 17


class ForwardStream:
    """The BF16 forward's hooks on ``_comm_stream.run_stream`` (module docstring).  ``has_tail`` (a 128-wide H tail),
    ``epilogue_combine`` (the routed Down epilogue pushes the combine, the stream moves its pulls only) and
    ``comm_tile_shape`` (the contract's CommTileShape: a block is one tile, ``block_rows`` x ``block_columns``)
    are the kernel's compile-time parameters."""

    def __init__(self, has_tail: bool, epilogue_combine: bool, comm_tile_shape):
        self.has_tail = bool(has_tail)
        self.epilogue_combine = bool(epilogue_combine)
        self.block_rows = comm_tile_shape.rows
        self.block_columns = comm_tile_shape.columns
        self._stream_block = _comm_stream.block_shape(self.block_rows, self.block_columns, 1)  # one tile per warp
        assert MBARRIERS == STREAM_WARPS * self._stream_block[2]  # one barrier per buffer
        assert self.block_rows <= _comm_stream.WARP_LANES and DONE_BLOCK_ROWS % self.block_rows == 0

    _run_stream = _comm_stream.run_stream

    @cute.jit
    def _stream_decode(self, operands: list, task: Int32) -> tuple[Int32, Int32, Int32]:
        """Decode hook: stream task -> (direction, macrobatch, block row), ``_stream_task`` (``_pull_stream_task``
        with ``epilogue_combine``)."""
        if cutlass.const_expr(self.epilogue_combine):
            return _pull_stream_task(task, operands[_Fo.GEOMETRY])
        return _stream_task(task, operands[_Fo.GEOMETRY])

    @cute.jit
    def _stream_schedule_row(self, operands: list, macro: Int32, row: Int32) -> Int32:
        """Schedule-row hook: the schedule index of row ``row`` of macrobatch ``macro``."""
        return macro * Int32(operands[_Fo.MACROBATCH_SIZE]) + row

    @cute.jit
    def _stream_ready_lane0(self, operands: list, direction: Int32, macro: Int32, row: Int32) -> Int32:
        """Readiness hook, lane 0's answer whether the block waits for nothing more (the other lanes return 1):
        y_routed_ready of ``_wait_target``'s minibatch holds every routed Down tile of it (``_down_tiles``)."""
        ready = Int32(1)
        if cute.arch.lane_idx() == Int32(0):
            num_tokens = operands[_Fo.NUM_TOKENS]
            minibatch_size = operands[_Fo.MINIBATCH_SIZE]
            mini = _divide_u32(row, minibatch_size, operands[_Fo.RECIPROCAL_MINIBATCH])  # minibatch inside the macrobatch
            target = _wait_target(
                direction,
                macro,
                mini,
                mini * minibatch_size,
                operands[_Fo.LAST_MACRO],
                num_tokens,
                operands[_Fo.MACROBATCH_SIZE],
                operands[_Fo.MINIBATCHES_PER_MACRO],
            )
            if target >= Int32(0):
                required = _down_tiles(target, num_tokens, minibatch_size, operands[_Fo.HIDDEN_SIZE], self.has_tail)
                y_routed_ready = operands[_Fo.Y_ROUTED_READY]
                if cute.arch.load(y_routed_ready.iterator + target, Int32, sem="acquire", scope="gpu") < required:
                    ready = Int32(0)
        return ready

    @cute.jit
    def _stream_issue(
        self,
        operands: list,
        direction: Int32,
        row: Int32,
        column_tile: Int32,
        buffer: cute.Pointer,
        mbarrier: cute.Pointer,
        lane: Int32,
        peer: Int32,
        route: Int32,
    ) -> None:
        """Issue hook: a block into ``buffer`` on ``mbarrier`` with the lane's schedule entries (``peer``, ``route``
        of the block's row ``lane``) already read: a pull of the peer's X row ``route / topk``, a push's read of
        the local Y row ``row + lane``."""
        hidden_size = operands[_Fo.HIDDEN_SIZE]
        # The addresses first: the peer-table read and the division overlap the previous block's stores, which may
        # still be reading the buffer (a lane past the block's rows has peer -1 and reads nothing).
        peer_buffer = Int64(0)  # a padding lane's (peer -1) is never read
        source_row = Int32(0)
        if direction == Int32(PULL):
            if peer >= Int32(0):
                peer_buffer = _peer_address(operands[_Fo.X_PEER_TABLE], peer)
                source_row = _divide_u32(route, operands[_Fo.TOPK], operands[_Fo.RECIPROCAL_TOPK])
        # The buffer's previous block: its stores must have read it (the wait left open by _stream_process).
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        if lane < Int32(self.block_rows):
            if direction == Int32(PULL):
                _comm_stream.pull_row(buffer, mbarrier, lane, column_tile, peer, source_row, hidden_size, peer_buffer, self.has_tail, self.block_columns)
            else:
                _comm_stream.load_local_row(
                    buffer, mbarrier, lane, operands[_Fo.Y_ROUTED].iterator.toint(), row, column_tile, peer, hidden_size, self.has_tail, self.block_columns
                )
        else:
            cute.arch.mbarrier_arrive(mbarrier)  # a lane past the block's rows: no row, the barrier counts every lane

    @cute.jit
    def _stream_arrival(self, operands: list, direction: Int32, macro: Int32, row: Int32) -> Int64:
        """Arrival-word hook: the counter a processed block arrives on (``_arrive_unit``), 0 for none."""
        unit = _arrive_unit(direction, macro, row, operands[_Fo.MACROBATCH_SIZE], operands[_Fo.MINIBATCH_SIZE], operands[_Fo.RECIPROCAL_MINIBATCH])
        address = Int64(0)
        if unit >= Int32(0):
            if direction == Int32(PULL):
                address = (operands[_Fo.X_ROUTED_READY].iterator + unit).toint()
            else:
                address = (operands[_Fo.Y_ROUTED_DONE].iterator + unit).toint()
        return address

    @cute.jit
    def _stream_process(
        self,
        operands: list,
        direction: Int32,
        macro: Int32,
        row: Int32,
        column: Int32,
        buffer: cute.Pointer,
        lane: Int32,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
    ) -> None:
        """Process hook, the block's bytes landed in ``buffer``: a pull block's rows (a padding row zero) go to the
        local X ring row ``row + lane`` (the writes completed by ``_stream_drain`` before the arrival is released); a
        push block's rows go to the source ranks' combine buffers.  The stores are not waited for here: the next
        issue into this buffer waits for their read (``_stream_issue``), a pull's arrival for their writes
        (``_stream_drain``)."""
        hidden_size = operands[_Fo.HIDDEN_SIZE]
        if lane < Int32(self.block_rows):  # a lane past the block's rows stores nothing
            if direction == Int32(PULL):
                x_routed = operands[_Fo.X_ROUTED]
                _comm_stream.store_row(
                    buffer,
                    lane,
                    x_routed.iterator.toint() + Int64(row + lane) * Int64(hidden_size) * Int64(2),
                    column,
                    hidden_size,
                    self.has_tail,
                    self.block_columns,
                )
            else:
                schedule_row = self._stream_schedule_row(operands, macro, row) + lane
                peer = schedule_peer_rank[schedule_row]
                if peer >= Int32(0):
                    route = schedule_peer_token_idx[schedule_row]
                    _comm_stream.store_row(
                        buffer,
                        lane,
                        _peer_address(operands[_Fo.COMBINE_PEER_TABLE], peer) + Int64(route) * Int64(hidden_size) * Int64(2),
                        column,
                        hidden_size,
                        self.has_tail,
                        self.block_columns,
                    )
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _stream_drain(self, operands: list, direction: Int32) -> None:
        """Drain hook, before the pending arrivals of ``direction`` are released.  Pulls: complete this lane's bulk
        stores (the X-ring rows counted on x_routed_ready are read by the Gate/Up TMA loads once it is released); one
        wait per release, so the stores of a minibatch's blocks overlap.  Pushes: nothing -- y_routed_done frees Y
        rows the block's load already read; the peers read the combine buffer after the kernel."""
        if direction == Int32(PULL):
            cute.arch.cp_async_bulk_wait_group(0)

    @cute.jit
    def run(
        self,
        x_peer_table: cute.Tensor,
        combine_peer_table: cute.Tensor,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        x_routed: cute.Tensor,
        y_routed: cute.Tensor,
        x_routed_ready: cute.Tensor,
        y_routed_ready: cute.Tensor,
        y_routed_done: cute.Tensor,
        arena: cute.Pointer,
        mbarriers: cute.Pointer,
        num_tokens: Int32,
        macrobatch_size: Int32,
        minibatch_size: Int32,
        hidden_size: Int32,
        topk: Int32,
        comm_cta: Int32,
        num_comm_sms: Int32,
    ) -> None:
        """The communication CTA's forward: the stream buffers' barriers, then the task stream over every
        macrobatch (``_comm_stream.run_stream`` with this class's hooks; ``x_routed`` / ``y_routed`` are the X / Y
        rings, ``hidden_size`` their width)."""
        thread = cute.arch.thread_idx()[0]
        if thread == Int32(0):
            for barrier in cutlass.range_constexpr(MBARRIERS):
                cute.arch.mbarrier_init(mbarriers + Int32(barrier), _comm_stream.WARP_LANES)  # one arrival per lane
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        last_macro = _ceil_div_i32(num_tokens, macrobatch_size) - Int32(1)
        # Block-row tiles: num_tokens and the macrobatch are multiples of the block rows (the schedule pads every expert to
        # 256 rows, the minibatch is a multiple of 256).
        last_tiles = _macro_rows(last_macro, num_tokens, macrobatch_size) // Int32(self.block_rows)
        full_tiles = macrobatch_size // Int32(self.block_rows)
        mini_tiles = minibatch_size // Int32(self.block_rows)
        if last_macro == Int32(0):
            mini_tiles = full_tiles  # one macrobatch: minibatches in order (_reverse_tile is then the identity)
        # The stream order's geometry, its divisions done once here (none per task).
        reciprocal_mini_tiles = _reciprocal_u32(mini_tiles)
        geometry = (
            last_macro,
            last_tiles,
            full_tiles,
            mini_tiles,
            _ceil_div_i32(last_tiles, mini_tiles),
            full_tiles // mini_tiles,
            reciprocal_mini_tiles,
            _reciprocal_u32(full_tiles),
            _reciprocal_u32(Int32(2) * full_tiles),
            _reciprocal_u32(Int32(2) * mini_tiles),
        )
        tasks = last_tiles + last_macro * full_tiles
        if cutlass.const_expr(not self.epilogue_combine):
            tasks = Int32(2) * tasks
        blocks_per_row = column_blocks(hidden_size, self.block_columns)
        group_blocks = Int32(STREAM_GROUP_BLOCKS)
        groups = _ceil_div_i32(blocks_per_row, group_blocks)  # the stream is numbered per group of column blocks
        head = cute.make_ptr(Int32, (y_routed_done.iterator + (y_routed_done.shape[0] - Int32(1))).toint(), cute.AddressSpace.gmem, assumed_align=4)
        operands = [None] * _Fo.COUNT  # the hooks' operands, by position
        operands[_Fo.X_PEER_TABLE] = x_peer_table
        operands[_Fo.COMBINE_PEER_TABLE] = combine_peer_table
        operands[_Fo.X_ROUTED] = x_routed
        operands[_Fo.Y_ROUTED] = y_routed
        operands[_Fo.X_ROUTED_READY] = x_routed_ready
        operands[_Fo.Y_ROUTED_READY] = y_routed_ready
        operands[_Fo.Y_ROUTED_DONE] = y_routed_done
        operands[_Fo.LAST_MACRO] = last_macro
        operands[_Fo.GEOMETRY] = geometry
        operands[_Fo.NUM_TOKENS] = num_tokens
        operands[_Fo.HIDDEN_SIZE] = hidden_size
        operands[_Fo.TOPK] = topk
        operands[_Fo.MACROBATCH_SIZE] = macrobatch_size
        operands[_Fo.MINIBATCH_SIZE] = minibatch_size
        operands[_Fo.MINIBATCHES_PER_MACRO] = macrobatch_size // minibatch_size
        operands[_Fo.RECIPROCAL_MINIBATCH] = _reciprocal_u32(minibatch_size)
        operands[_Fo.RECIPROCAL_TOPK] = _reciprocal_u32(topk)
        assert all(operand is not None for operand in operands)
        self._run_stream(
            operands,
            schedule_peer_rank,
            schedule_peer_token_idx,
            head,
            arena,
            mbarriers,
            tasks * groups,
            groups,
            _reciprocal_u32(groups),
            _reciprocal_u32(group_blocks),
            blocks_per_row,
            comm_cta,
            num_comm_sms,
            Int32(0),
        )
