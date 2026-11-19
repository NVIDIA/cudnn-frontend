# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Backward communication: the BF16 backward's hooks on the communication task stream (``_comm_stream.run_stream``).

Private device module of ``_persistent_bf16_bwd_mega``.  The loop, the per-warp buffers and the arrivals are the shared
stream's (``_comm_stream``; read its docstring first), as in the forward ``_fwd_dynamic_task_stream``; this module
supplies the task order and what a block is.  A block is one communication tile of the kernel's ``comm_tile_shape``
(in the warp's one 32 KiB buffer) of one generation
(minibatch); the hooks address it by (macrobatch, macrobatch-local row), the row inside the ring:

    task        generations   wait (per generation)                       then (per block)            arrive
    X(g) pull   g >= mpm      ring position free: routed_buffers_done      bulk-store into the X ring   replayed_x_routed_ready[g]
                (replay)      [g-mpm] >= slot_reuse_required and
                              wgrad_read_consumed[g-mpm] >= experts(g-mpm) x 3 x wgrad_tiles
    dY(g) pull  every g       the same (none for g < mpm)                  bulk-store into the dY ring  d_y_routed_ready[g]
                              (column block 0 also copies the row's router score into the ring at issue)
    dX(g) push  every g       d_x_routed_ready[g] >= Gate/Up dgrad tiles   bulk-store to the peer       reverse_dispatch_done[g]
                              (column block 0 also writes the row's d_router sum to the peer)   (none when g + mpm >= G)

A pull loads the peer's row (route / topk; a padding row, or a source rank at or past the runtime EP size, is
zero-filled) and a push reads the local dX ring row.

The stream (``_stream_task``), generations in order (ring position g % mpm):
    dY pulls of generations 0 .. min(mpm, G) - 1  (the first macrobatch; its X is still in the ring from the forward)
    with several macrobatches: for g = mpm .. G - 1 one step per generation, the X rows of g then its dY rows, with
      the dX pushes of g - lag mixed in, lag = max(mpm - 1, 1): a push is ready once its generation's Gate/Up dgrad
      is done, about one generation before the pulls of g + mpm may reuse its ring position, so it goes one step
      earlier than that (``_step_task``); the pushes of generations 0 .. mpm - lag - 1 come before the first step
    the dX pushes of the last lag generations (all generations with one macrobatch)
recompute_forward_context (compile time; the forward context holds no routed activations): the first macrobatch is
replayed too, its part of the stream is the X rows then the dY rows of each of its generations.
Every task depends only on tasks with smaller numbers and on the compute of those, so the smallest unfinished task is
always ready (no deadlock).  The head is the word after reverse_dispatch_done's generations (zero at launch).

What the compute side waits for: one arrival per block, d_y_routed_ready / replayed_x_routed_ready /
reverse_dispatch_done of a generation = its block rows x column blocks (``comm_blocks``).  The router score of a
row is written at the issue of the dY block of column block 0, before that block's d_y_routed_ready arrival, so every
compute task that waits d_y_routed_ready sees it (no separate router-score readiness counter).

Every size, count, index and expected count is a runtime value.
"""

import enum

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from . import _comm_stream
from ._bwd_device_helpers import _ceil_div_i32, _load_peer_float, _peer_address, _store_peer_float
from ._comm_stream import (
    STREAM_GROUP_BLOCKS,
    STREAM_WARP_BYTES,
    STREAM_WARPS,
    _divide_u32,
    _reciprocal_u32,
)
from .persistent_bf16_bwd_contract import CLUSTER_SIZE, MLP_TILE_M, MLP_TILE_N

TASK_X = 0  # replay pull
TASK_DY = 1  # pull
TASK_DX = 2  # push

# Interface read by the mega kernel: shared memory of the communication CTA (the buffers alias the compute arena).
ARENA_BYTES = STREAM_WARPS * STREAM_WARP_BYTES
MBARRIERS = STREAM_WARPS  # one per warp: its one buffer (``block_shape`` buffers = 1)


@cute.jit
def comm_blocks(rows: Int32, hidden_size: Int32, comm_tile_shape: cutlass.Constexpr) -> Int32:
    """Blocks of ``rows`` routed rows (a generation's, per direction) for the kernel's tile shape (compile time): block
    rows x column blocks, one arrival each."""
    return (
        rows // Int32(comm_tile_shape.rows) * _ceil_div_i32(hidden_size, Int32(comm_tile_shape.columns))  # whole tile rows: the stream counts them the same way
    )


# ---------------------------------------------------------------- shapes (all runtime)


@cute.jit
def _generation_rows(g: Int32, num_tokens: Int32, minibatch_size: Int32) -> Int32:
    """Routed rows of generation (minibatch) ``g``; the last one may be short."""
    return cutlass.min(Int32(minibatch_size), num_tokens - g * Int32(minibatch_size))


@cute.jit
def _gate_up_dgrad_tiles(g: Int32, num_tokens: Int32, minibatch_size: Int32, hidden_size: Int32, has_tail: cutlass.Constexpr = False) -> Int32:
    """d_x_routed_ready count of a complete generation: every routed Gate/Up dgrad tile, both cluster CTAs."""
    rows = _generation_rows(g, num_tokens, minibatch_size)
    column_blocks = Int32(hidden_size // MLP_TILE_N)
    if cutlass.const_expr(has_tail):
        # A 128-wide H tail is one more dgrad tile, arrived on by both cluster CTAs like every other.
        column_blocks = _ceil_div_i32(hidden_size, Int32(MLP_TILE_N))
    return (rows // Int32(MLP_TILE_M)) * column_blocks * Int32(CLUSTER_SIZE)


# ---------------------------------------------------------------- arrive unit


@cute.jit
def _arrive_unit(task: Int32, g: Int32, total_generations: Int32, mpm: Int32) -> Int32:
    """Counter index a finished tile counts toward, or -1 (nobody waits).  The counter itself is
    chosen by ``task`` in ``_stream_arrival``: X -> replayed_x_routed_ready[g], dY -> d_y_routed_ready[g],
    dX -> reverse_dispatch_done[g] (only while a later generation reuses the dX slot)."""
    unit = g
    if task == Int32(TASK_DX):
        if g + mpm >= total_generations:
            unit = Int32(-1)
    return unit


# ---------------------------------------------------------------- the stream order


@cute.jit
def _step_task(k: Int32, pulls: Int32, pushes: Int32) -> tuple[Int32, Int32]:
    """(task, tile row) at position ``k`` of a step with ``pulls`` X / dY rows of generation g and ``pushes`` dX rows of
    g - lag: the X half first (replay needs X before dY), then the dY half, each with half the pushes mixed in
    (X0 P0 X1 | X2 P1 X3 | ... then dY0 P dY1 | ...), then the pushes left over."""
    used = (pulls + Int32(1)) // Int32(2)  # pushes mixed into one half
    half = pulls + used
    task = Int32(TASK_DX)
    tile_row = Int32(2) * used + (k - Int32(2) * half)
    if k < Int32(2) * half:
        h = Int32(0)  # the half: k < 2 x half, so k // half is 0 or 1
        if k >= half:
            h = Int32(1)
        q = k - h * half
        group = q // Int32(3)
        position = q - group * Int32(3)
        tile_row = h * used + group
        if position != Int32(1):
            task = Int32(TASK_X)
            if h == Int32(1):
                task = Int32(TASK_DY)
            tile_row = Int32(2) * group
            if position == Int32(2):
                tile_row = tile_row + Int32(1)
    return task, tile_row


@cute.jit
def _stream_task(index: Int32, geometry: tuple, recompute_forward_context: cutlass.Constexpr = False) -> tuple[Int32, Int32, Int32]:
    """(task, generation, tile row) of task ``index`` of the backward stream (module docstring).  Generations
    0 .. G-1 in order (ring position g % mpm; ``mini_tiles`` tile rows each, the last one ``last_tiles``).  A dX push
    of g is ready once the Gate/Up dgrad of g is done, about one generation before the pulls of g + mpm may reuse its
    ring position, so the pushes of g ride with the pulls of g + lag, lag = max(mpm - 1, 1) (at least one step later
    than the pulls of g, which the push depends on):
      the dY pulls of generations 0 .. min(mpm, G) - 1 (the first macrobatch; its X is still in the ring from the
      forward);
      with several macrobatches, the dX pushes of generations 0 .. mpm - lag - 1;
      then for g = mpm .. G-1 one step per ring position (``_step_task``): the X rows of g, then its dY rows, each half
      with half the dX pushes of g - lag mixed in (pull g waits for the readers of g - mpm, push g - lag for its Gate/Up
      dgrad);
      then the dX pushes of generations G - lag .. G-1 (all generations with one macrobatch).
    3 x sum(rows) - (X rows the first macrobatch does not pull) tasks.
    ``recompute_forward_context``: the first macrobatch pulls its X as well, per generation its X rows then its dY
    rows.
    ``geometry`` (``BackwardStream.run`` builds it once): (generations, mpm, mini_tiles, last_tiles, and the
    ``_reciprocal_u32`` of mini_tiles, 2 x mini_tiles and 3 x mini_tiles: no division per task)."""
    generations, mpm, mini_tiles, last_tiles, reciprocal_mini, reciprocal_2mini, reciprocal_3mini = geometry
    task = Int32(TASK_DY)
    g = _divide_u32(index, mini_tiles, reciprocal_mini)
    tile_row = index - g * mini_tiles
    first_gens = cutlass.min(mpm, generations)
    first_size = (first_gens - Int32(1)) * mini_tiles + mini_tiles
    if first_gens == generations:
        first_size = (first_gens - Int32(1)) * mini_tiles + last_tiles
    if cutlass.const_expr(recompute_forward_context):
        # Two halves per generation, X then dY (only the stream's last generation can be short, and
        # nothing follows it in this part).
        first_size = Int32(2) * first_size
        g = _divide_u32(index, Int32(2) * mini_tiles, reciprocal_2mini)
        tile_row = index - g * (Int32(2) * mini_tiles)
        generation_tiles = mini_tiles
        if g == generations - Int32(1):
            generation_tiles = last_tiles
        task = Int32(TASK_X)
        if tile_row >= generation_tiles:
            task = Int32(TASK_DY)
            tile_row = tile_row - generation_tiles
    if index >= first_size:
        j = index - first_size
        tail_g0 = Int32(0)
        in_tail = Int32(1)
        if generations > mpm:
            lag = cutlass.max(mpm - Int32(1), Int32(1))
            early_pushes = (mpm - lag) * mini_tiles
            tail_g0 = generations - lag
            if j < early_pushes:
                in_tail = Int32(0)
                task = Int32(TASK_DX)
                g = _divide_u32(j, mini_tiles, reciprocal_mini)
                tile_row = j - g * mini_tiles
            else:
                j = j - early_pushes
                full_steps = generations - Int32(1) - mpm
                step_size = Int32(3) * mini_tiles
                if j < full_steps * step_size:
                    in_tail = Int32(0)
                    s = _divide_u32(j, step_size, reciprocal_3mini)
                    g = mpm + s
                    task, tile_row = _step_task(j - s * step_size, mini_tiles, mini_tiles)
                    if task == Int32(TASK_DX):
                        g = g - lag
                else:
                    j = j - full_steps * step_size
                    if j < Int32(2) * last_tiles + mini_tiles:
                        in_tail = Int32(0)
                        g = generations - Int32(1)
                        task, tile_row = _step_task(j, last_tiles, mini_tiles)
                        if task == Int32(TASK_DX):
                            g = g - lag
                    else:
                        j = j - (Int32(2) * last_tiles + mini_tiles)
        if in_tail == Int32(1):
            task = Int32(TASK_DX)
            q = _divide_u32(j, mini_tiles, reciprocal_mini)
            g = tail_g0 + q
            tile_row = j - q * mini_tiles
    return task, g, tile_row


@cute.jit
def _macro_tile(generation: Int32, tile: Int32, mpm: Int32, mini_tiles: Int32, reciprocal_mpm: Int32) -> tuple[Int32, Int32]:
    """(macrobatch, tile row inside it) of tile row ``tile`` of ``generation``: the macrobatch-local row is the ring
    row (ring position generation % mpm); ``reciprocal_mpm`` is ``_reciprocal_u32(mpm)``."""
    macro = _divide_u32(generation, mpm, reciprocal_mpm)
    return macro, (generation - macro * mpm) * mini_tiles + tile


@cute.jit
def _generation_of(macro: Int32, row: Int32, mpm: Int32, minibatch_size: Int32, reciprocal_minibatch: Int32) -> Int32:
    """The generation (minibatch) of row ``row`` of macrobatch ``macro`` (``_macro_tile``'s inverse);
    ``reciprocal_minibatch`` is ``_reciprocal_u32(minibatch_size)``."""
    return macro * mpm + _divide_u32(row, minibatch_size, reciprocal_minibatch)


class StreamOperand(enum.IntEnum):
    """Positions in the hooks' operand list (``BackwardStream.run`` builds it, ``_comm_stream.run_stream`` hands it to
    every hook)."""

    X_PEER_TABLE = 0
    D_Y_PEER_TABLE = 1
    D_X_PEER_TABLE = 2
    ROUTER_PEER_TABLE = 3
    D_ROUTER_PEER_TABLE = 4
    X_ROUTED = 5
    D_Y_ROUTED = 6
    D_X_ROUTED = 7
    ROUTER_WEIGHTS = 8
    D_ROUTER_PARTIALS = 9
    MINIBATCH_EXPERT_OFFSETS = 10
    D_Y_ROUTED_READY = 11
    D_X_ROUTED_READY = 12
    REPLAYED_X_ROUTED_READY = 13
    ROUTED_BUFFERS_DONE = 14
    WGRAD_READ_CONSUMED = 15
    REVERSE_DISPATCH_DONE = 16
    NUM_TOKENS = 17
    MACROBATCH_SIZE = 18
    MINIBATCH_SIZE = 19
    MPM = 20
    HIDDEN_SIZE = 21
    TOPK = 22
    S_TILES = 23
    SLOT_REUSE_REQUIRED = 24
    WGRAD_TILES = 25
    EP_RANKS = 26
    GENERATIONS = 27
    GEOMETRY = 28  # the stream order's tuple (``_stream_task``)
    RECIPROCAL_MPM = 29
    RECIPROCAL_MINIBATCH = 30
    RECIPROCAL_TOPK = 31
    COUNT = 32


@cute.jit
def _source_rank(peer_rank: Int32, ep_size: Int32) -> Int32:
    """The schedule's source rank of a row, -1 (a padding row: nothing moved, zeros pulled) for a padding row or a rank
    at or past the runtime EP size."""
    rank = peer_rank
    if peer_rank >= ep_size:
        rank = Int32(-1)
    return rank


class BackwardStream:
    """The BF16 backward's hooks on ``_comm_stream.run_stream`` (module docstring).  ``has_tail`` (a 128-wide H tail),
    ``recompute_forward_context`` (the first macrobatch is replayed too) and ``comm_tile_shape`` (the contract's
    CommTileShape: a block is one tile, ``block_rows`` x ``block_columns``) are the kernel's compile-time
    parameters."""

    def __init__(self, has_tail: bool, recompute_forward_context: bool, comm_tile_shape):
        self.has_tail = bool(has_tail)
        self.recompute_forward_context = bool(recompute_forward_context)
        self.block_rows = comm_tile_shape.rows
        self.block_columns = comm_tile_shape.columns
        self._stream_block = _comm_stream.block_shape(self.block_rows, self.block_columns, 1)  # one tile per warp
        assert MBARRIERS == STREAM_WARPS * self._stream_block[2]  # one barrier per buffer
        assert self.block_rows <= _comm_stream.WARP_LANES

    _run_stream = _comm_stream.run_stream

    @cute.jit
    def _generation(self, operands: list, macro: Int32, row: Int32) -> Int32:
        """The generation (minibatch) of row ``row`` of macrobatch ``macro`` (``_generation_of``)."""
        return _generation_of(macro, row, operands[StreamOperand.MPM], operands[StreamOperand.MINIBATCH_SIZE], operands[StreamOperand.RECIPROCAL_MINIBATCH])

    @cute.jit
    def _stream_decode(self, operands: list, task: Int32) -> tuple[Int32, Int32, Int32]:
        """Decode hook: stream task -> (task kind, macrobatch, block row of the macrobatch), from ``_stream_task``'s
        (task kind, generation, tile of the generation)."""
        geometry = operands[StreamOperand.GEOMETRY]
        kind, generation, tile = _stream_task(task, geometry, self.recompute_forward_context)
        macro, macro_tile = _macro_tile(generation, tile, operands[StreamOperand.MPM], geometry[2], operands[StreamOperand.RECIPROCAL_MPM])
        return kind, macro, macro_tile

    @cute.jit
    def _stream_schedule_row(self, operands: list, macro: Int32, row: Int32) -> Int32:
        """Schedule-row hook: the macrobatch's first row plus ``row`` (macrobatch-local, which is the ring row)."""
        return macro * operands[StreamOperand.MACROBATCH_SIZE] + row

    @cute.jit
    def _stream_ready_lane0(self, operands: list, kind: Int32, macro: Int32, row: Int32) -> Int32:
        """Readiness hook, lane 0's answer whether the block waits for nothing more (the other lanes return 1): a dX
        push of generation g for g's Gate/Up dgrad (d_x_routed_ready); an X / dY pull of g >= mpm for the readers of
        ring position g % mpm, generation g - mpm, to be done with it (routed_buffers_done, and wgrad_read_consumed of
        every expert overlapping it)."""
        ready = Int32(1)
        if cute.arch.lane_idx() == Int32(0):
            g = self._generation(operands, macro, row)
            mpm = operands[StreamOperand.MPM]
            if kind == Int32(TASK_DX):
                required = _gate_up_dgrad_tiles(
                    g, operands[StreamOperand.NUM_TOKENS], operands[StreamOperand.MINIBATCH_SIZE], operands[StreamOperand.HIDDEN_SIZE], self.has_tail
                )
                if cute.arch.load(operands[StreamOperand.D_X_ROUTED_READY].iterator + g, Int32, sem="acquire", scope="gpu") < required:
                    ready = Int32(0)
            elif g >= mpm:
                previous = g - mpm
                if (
                    cute.arch.load(operands[StreamOperand.ROUTED_BUFFERS_DONE].iterator + previous, Int32, sem="acquire", scope="gpu")
                    < operands[StreamOperand.SLOT_REUSE_REQUIRED]
                ):
                    ready = Int32(0)
                else:
                    offsets = operands[StreamOperand.MINIBATCH_EXPERT_OFFSETS]
                    experts = Int32(offsets[previous + Int32(1)]) - Int32(offsets[previous])
                    if (
                        cute.arch.load(operands[StreamOperand.WGRAD_READ_CONSUMED].iterator + previous, Int32, sem="acquire", scope="gpu")
                        < experts * Int32(3) * operands[StreamOperand.WGRAD_TILES]
                    ):
                        ready = Int32(0)
        return ready

    @cute.jit
    def _stream_issue(
        self, operands: list, kind: Int32, row: Int32, column_tile: Int32, buffer: cute.Pointer, mbarrier: cute.Pointer, lane: Int32, peer: Int32, route: Int32
    ) -> None:
        """Issue hook: a block into ``buffer`` on ``mbarrier`` with the lane's schedule entries (``peer``, ``route`` of
        the block's row ``lane``) already read.  X / dY: the peer's token row ``route / topk`` (a padding row
        zero-filled); column block 0 of dY also copies the route's router score into ring row ``row + lane`` (an exact
        zero for a padding row).  dX: the local dX ring row ``row + lane``."""
        hidden_size = operands[StreamOperand.HIDDEN_SIZE]
        # The addresses and the router score first: the peer reads overlap the previous block's stores, which may
        # still be reading the buffer (a lane past the block's rows has peer -1 and reads nothing).
        rank = _source_rank(peer, operands[StreamOperand.EP_RANKS])
        peer_buffer = Int64(0)  # a padding lane's (rank -1) is never read
        source_row = Int32(0)
        score = Float32(0.0)  # a padding row's router score: exact zero
        if kind != Int32(TASK_DX):
            if rank >= Int32(0):
                # dY is per token, like X
                source_row = _divide_u32(route, operands[StreamOperand.TOPK], operands[StreamOperand.RECIPROCAL_TOPK])
                if kind == Int32(TASK_X):
                    peer_buffer = _peer_address(operands[StreamOperand.X_PEER_TABLE], rank)
                else:
                    peer_buffer = _peer_address(operands[StreamOperand.D_Y_PEER_TABLE], rank)
                    if column_tile == Int32(0):
                        score = _load_peer_float(operands[StreamOperand.ROUTER_PEER_TABLE], rank, route)
        # The buffer's previous block: its stores must have read it (the wait left open by _stream_process).
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        if lane < Int32(self.block_rows):
            if kind == Int32(TASK_DX):
                _comm_stream.load_local_row(
                    buffer,
                    mbarrier,
                    lane,
                    operands[StreamOperand.D_X_ROUTED].iterator.toint(),
                    row,
                    column_tile,
                    rank,
                    hidden_size,
                    self.has_tail,
                    self.block_columns,
                )
            else:
                _comm_stream.pull_row(buffer, mbarrier, lane, column_tile, rank, source_row, hidden_size, peer_buffer, self.has_tail, self.block_columns)
                if kind == Int32(TASK_DY):
                    if column_tile == Int32(0):
                        operands[StreamOperand.ROUTER_WEIGHTS][row + lane] = score
        else:
            cute.arch.mbarrier_arrive(mbarrier)  # a lane past the block's rows: no row, the barrier counts every lane

    @cute.jit
    def _stream_arrival(self, operands: list, kind: Int32, macro: Int32, row: Int32) -> Int64:
        """Arrival-word hook: replayed_x_routed_ready[g] (X), d_y_routed_ready[g] (dY), reverse_dispatch_done[g] (dX,
        only while a later generation reuses the dX ring position; 0 otherwise: nobody waits)."""
        g = self._generation(operands, macro, row)
        unit = _arrive_unit(kind, g, operands[StreamOperand.GENERATIONS], operands[StreamOperand.MPM])
        address = Int64(0)
        if unit >= Int32(0):
            if kind == Int32(TASK_X):
                address = (operands[StreamOperand.REPLAYED_X_ROUTED_READY].iterator + unit).toint()
            elif kind == Int32(TASK_DY):
                address = (operands[StreamOperand.D_Y_ROUTED_READY].iterator + unit).toint()
            else:
                address = (operands[StreamOperand.REVERSE_DISPATCH_DONE].iterator + unit).toint()
        return address

    @cute.jit
    def _stream_process(
        self,
        operands: list,
        kind: Int32,
        macro: Int32,
        row: Int32,
        column: Int32,
        buffer: cute.Pointer,
        lane: Int32,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
    ) -> None:
        """Process hook, the block's bytes landed in ``buffer``: an X / dY block's rows (a padding row zero) go to the
        local ring row ``row + lane`` (the writes completed by ``_stream_drain`` before the arrival is released); a dX
        block's rows go to the source ranks' dX buffers (column block 0 also writes the row's d_router sum to the
        peer).  The stores are not waited for here: the next issue into this buffer waits for their read
        (``_stream_issue``), a pull's arrival for their writes (``_stream_drain``)."""
        hidden_size = operands[StreamOperand.HIDDEN_SIZE]
        if lane < Int32(self.block_rows):  # a lane past the block's rows stores nothing
            if kind == Int32(TASK_DX):
                schedule_row = self._stream_schedule_row(operands, macro, row) + lane
                rank = _source_rank(schedule_peer_rank[schedule_row], operands[StreamOperand.EP_RANKS])
                if rank >= Int32(0):
                    route = schedule_peer_token_idx[schedule_row]
                    _comm_stream.store_row(
                        buffer,
                        lane,
                        _peer_address(operands[StreamOperand.D_X_PEER_TABLE], rank) + Int64(route) * Int64(hidden_size) * Int64(2),
                        column,
                        hidden_size,
                        self.has_tail,
                        self.block_columns,
                    )
                    if column == Int32(0):
                        d_router_partials = operands[StreamOperand.D_ROUTER_PARTIALS]
                        d_router = Float32(0.0)
                        for partial in cutlass.range(operands[StreamOperand.S_TILES]):
                            d_router = d_router + Float32(d_router_partials[row + lane, partial])
                        _store_peer_float(operands[StreamOperand.D_ROUTER_PEER_TABLE], rank, route, d_router)
            else:
                ring = operands[StreamOperand.D_Y_ROUTED]
                if kind == Int32(TASK_X):
                    ring = operands[StreamOperand.X_ROUTED]
                _comm_stream.store_row(
                    buffer,
                    lane,
                    ring.iterator.toint() + Int64(row + lane) * Int64(hidden_size) * Int64(2),
                    column,
                    hidden_size,
                    self.has_tail,
                    self.block_columns,
                )
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def _stream_drain(self, operands: list, kind: Int32) -> None:
        """Drain hook, before the pending arrivals of ``kind`` are released.  X / dY pulls: complete this lane's bulk
        stores (the ring rows counted on replayed_x_routed_ready / d_y_routed_ready are read by the compute TMA loads
        once it is released); one wait per release.  dX pushes: nothing -- reverse_dispatch_done frees dX ring rows the
        block's load already read; the peers read their dX after the kernel."""
        if kind != Int32(TASK_DX):
            cute.arch.cp_async_bulk_wait_group(0)

    @cute.jit
    def run(
        self,
        operands: list,
        schedule_peer_rank: cute.Tensor,
        schedule_peer_token_idx: cute.Tensor,
        arena: cute.Pointer,
        mbarriers: cute.Pointer,
        comm_cta: Int32,
        num_comm_sms: Int32,
    ) -> None:
        """The communication CTA's backward: the stream buffers' barriers, then the task stream over every generation
        (``_comm_stream.run_stream`` with this class's hooks).  ``operands``: the caller's tensors and runtime scalars
        by ``StreamOperand`` position up to EP_RANKS; the stream's shape and divisors (from GENERATIONS on) are filled
        here."""
        thread = cute.arch.thread_idx()[0]
        if thread == Int32(0):
            for barrier in cutlass.range_constexpr(MBARRIERS):
                cute.arch.mbarrier_init(mbarriers + Int32(barrier), _comm_stream.WARP_LANES)  # one arrival per lane
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        num_tokens = operands[StreamOperand.NUM_TOKENS]
        minibatch_size = operands[StreamOperand.MINIBATCH_SIZE]
        mpm = operands[StreamOperand.MPM]
        generations = _ceil_div_i32(num_tokens, minibatch_size)
        # Block rows: the schedule pads every expert to 256 rows, the minibatch is a multiple of 256.
        mini_tiles = minibatch_size // Int32(self.block_rows)
        last_tiles = _generation_rows(generations - Int32(1), num_tokens, minibatch_size) // Int32(self.block_rows)
        rows_total = (generations - Int32(1)) * mini_tiles + last_tiles
        x_rows = Int32(0)  # X pulls: generations mpm .. G-1
        if generations > mpm:
            x_rows = rows_total - mpm * mini_tiles
        if cutlass.const_expr(self.recompute_forward_context):
            x_rows = rows_total  # every generation
        tasks = Int32(2) * rows_total + x_rows
        blocks_per_row = _ceil_div_i32(operands[StreamOperand.HIDDEN_SIZE], Int32(self.block_columns))
        group_blocks = Int32(STREAM_GROUP_BLOCKS)
        groups = _ceil_div_i32(blocks_per_row, group_blocks)  # the stream is numbered per group of column blocks
        reverse_dispatch_done = operands[StreamOperand.REVERSE_DISPATCH_DONE]
        head = cute.make_ptr(
            Int32, (reverse_dispatch_done.iterator + (reverse_dispatch_done.shape[0] - Int32(1))).toint(), cute.AddressSpace.gmem, assumed_align=4
        )
        operands[StreamOperand.GENERATIONS] = generations
        # The stream order's geometry and the hooks' divisors, their divisions done once here (none per task).
        operands[StreamOperand.GEOMETRY] = (
            generations,
            mpm,
            mini_tiles,
            last_tiles,
            _reciprocal_u32(mini_tiles),
            _reciprocal_u32(Int32(2) * mini_tiles),
            _reciprocal_u32(Int32(3) * mini_tiles),
        )
        operands[StreamOperand.RECIPROCAL_MPM] = _reciprocal_u32(mpm)
        operands[StreamOperand.RECIPROCAL_MINIBATCH] = _reciprocal_u32(minibatch_size)
        operands[StreamOperand.RECIPROCAL_TOPK] = _reciprocal_u32(operands[StreamOperand.TOPK])
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
