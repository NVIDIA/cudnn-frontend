# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Backward communication prefix: the dynamic task stream of X replay pulls, dY pulls (reverse combine) and
dX pushes (reverse dispatch).

Private device module of ``_persistent_bf16_bwd_mega``.  The same protocol as the forward ``_fwd_dynamic_task_stream``
(the slot pool and primitives are ``_dynamic_task_stream``; the loop has the same shape: take a task, wait for it
holding no slot, per tile acquire / load / store(prev) / release, batched arrive, take the next number).  Only the
stream, the task description and the load / store addresses differ:

    task        generations   rows                  wait (per generation)                 arrive (per tile)
    X(g) pull   g >= mpm      rows(g)/R tile rows   ring position free: routed_buffers_done  replayed_x_routed_ready[g]
                (replay)                            [g-mpm] >= slot_reuse_required and
                                                    wgrad_read_consumed[g-mpm] >= experts(g-mpm) x 3 x wgrad_tiles
    dY(g) pull  every g       rows(g)/R tile rows   the same (none for g < mpm)             d_y_routed_ready[g]
                              (col_block 0 lanes also copy the row's router score into the ring)
    dX(g) push  every g       rows(g)/R tile rows   d_x_routed_ready[g] >= Gate/Up dgrad    reverse_dispatch_done[g]
                              (col_block 0 lanes also write the row's d_router sum to the peer) (none when g + mpm >= G)

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

Expected counts are per R-row x C-column tile, the tile shape of the kernel variant (``SlotPool.shape``, compile
time, one of the contract's COMM_TILE_SHAPES); the compute-side waits of
d_y_routed_ready / replayed_x_routed_ready / reverse_dispatch_done use the same tile.  The router score of a row is
written by the dY tile of column block 0 before that tile's d_y_routed_ready arrival, so every compute task that waits
d_y_routed_ready sees it (no separate router-score readiness counter).

The loop is written out per direction (not shared with the forward module): the two streams differ and a device
function cannot be passed to another as an argument.  Tensors and runtime scalars travel in ``BwdCommArgs``.  Every
size, count, index and expected count is a runtime value.
"""

import dataclasses

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64

from . import _dynamic_task_stream as pool
from ._bwd_device_helpers import _ceil_div_i32, _load_peer_float, _peer_address, _store_peer_float
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw, zero_shared_bulk
from .persistent_bf16_bwd_contract import CLUSTER_SIZE, MLP_TILE_M, MLP_TILE_N

PULL = pool.PULL
PUSH = pool.PUSH
TASK_X = 0  # replay pull
TASK_DY = 1  # pull
TASK_DX = 2  # push

# Interface read by the mega kernel (shared memory of the communication CTA).
ARENA_BYTES = pool.ARENA_BYTES
MBARRIERS = pool.MBARRIERS
SCRATCH_WORDS = pool.SCRATCH_WORDS


@dataclasses.dataclass(frozen=True)
class BwdCommArgs:
    """Global tensors and runtime scalars of the backward prefix (built by the mega kernel)."""

    x_peer_table: cute.Tensor  # X base address of every rank (replay pull source)
    d_y_peer_table: cute.Tensor  # dY base address of every rank (pull source)
    d_x_peer_table: cute.Tensor  # dX base address of every rank (push destination)
    router_peer_table: cute.Tensor  # router-score base address of every rank
    d_router_peer_table: cute.Tensor  # d_router base address of every rank
    schedule_peer_rank: cute.Tensor  # per routed row: source rank, -1 for a padding row
    schedule_peer_token_idx: cute.Tensor  # per routed row: route index on the source rank
    x_routed: cute.Tensor  # local X ring (replay destination)
    d_y_routed: cute.Tensor  # local dY ring (pull destination)
    d_x_routed: cute.Tensor  # local dX ring (push source)
    router_weights: cute.Tensor  # router score per ring row
    d_router_partials: cute.Tensor  # [ring row, s tile] d_router partial sums
    minibatch_expert_offsets: cute.Tensor  # experts of each generation (Wgrad reader count)
    d_y_routed_ready: cute.Tensor  # [generation] pulled dY tiles
    d_x_routed_ready: cute.Tensor  # [generation] Gate/Up dgrad tiles (push waits)
    replayed_x_routed_ready: cute.Tensor  # [generation] replayed X tiles
    routed_buffers_done: cute.Tensor  # [generation] compute readers done with the slot
    wgrad_read_consumed: cute.Tensor  # [generation] Wgrad readers done with the slot
    reverse_dispatch_done: cute.Tensor  # [generation] pushed dX tiles
    num_tokens: Int32
    macrobatch_size: Int32
    minibatch_size: Int32
    mpm: Int32  # minibatches per macrobatch (ring slots)
    hidden_size: Int32
    topk: Int32
    s_tiles: Int32
    slot_reuse_required: Int32
    wgrad_tiles: Int32
    ep_size: Int32
    has_tail: bool = False  # compile time: H has a 128-wide tail, one more dgrad tile per row block
    recompute_forward_context: bool = False  # compile time: the first macrobatch's X is pulled (replayed) too


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


# ---------------------------------------------------------------- task description (the direction-specific part)


@cute.jit
def _segment_reached(args: BwdCommArgs, direction: Int32, g: Int32) -> Int32:
    """Warp-uniform: 1 if segment ``g`` may start now (its wait targets are complete).
    PULL: slot reuse — the compute readers of generation g - mpm released the dY/X slot
    (routed_buffers_done >= slot_reuse_required, compute readers only) and every Wgrad task of it
    read the slot (wgrad_read_consumed); nothing to wait for when g < mpm.
    PUSH: routed Gate/Up dgrad of g produced the dX rows (d_x_routed_ready)."""
    reached = Int32(1)
    if direction == Int32(PULL):
        if g >= args.mpm:
            previous = g - args.mpm
            experts = Int32(args.minibatch_expert_offsets[previous + Int32(1)]) - Int32(args.minibatch_expert_offsets[previous])
            reached = pool.reached(args.routed_buffers_done, previous, args.slot_reuse_required)
            if reached == Int32(1):
                reached = pool.reached(args.wgrad_read_consumed, previous, experts * Int32(3) * args.wgrad_tiles)
    else:
        reached = pool.reached(args.d_x_routed_ready, g, _gate_up_dgrad_tiles(g, args.num_tokens, args.minibatch_size, args.hidden_size, args.has_tail))
    return reached


@cute.jit
def _segment_wait(args: BwdCommArgs, direction: Int32, g: Int32) -> None:
    """Whole warp: block until segment ``g`` may start (same targets as ``_segment_reached``)."""
    if direction == Int32(PULL):
        if g >= args.mpm:
            previous = g - args.mpm
            experts = Int32(args.minibatch_expert_offsets[previous + Int32(1)]) - Int32(args.minibatch_expert_offsets[previous])
            pool.wait(args.routed_buffers_done, previous, args.slot_reuse_required)
            pool.wait(args.wgrad_read_consumed, previous, experts * Int32(3) * args.wgrad_tiles)
    else:
        pool.wait(args.d_x_routed_ready, g, _gate_up_dgrad_tiles(g, args.num_tokens, args.minibatch_size, args.hidden_size, args.has_tail))


@cute.jit
def _arrive_unit(task: Int32, g: Int32, total_generations: Int32, mpm: Int32) -> Int32:
    """Counter index a finished tile counts toward, or -1 (nobody waits).  The counter itself is
    chosen by ``task`` in ``_arrive_address``: X -> replayed_x_routed_ready[g], dY -> d_y_routed_ready[g],
    dX -> reverse_dispatch_done[g] (only while a later generation reuses the dX slot)."""
    unit = g
    if task == Int32(TASK_DX):
        if g + mpm >= total_generations:
            unit = Int32(-1)
    return unit


# ---------------------------------------------------------------- arrive address


@cute.jit
def _arrive_address(args: BwdCommArgs, tile: pool.InFlightTile, total_generations: Int32) -> Int64:
    """Counter word a finished tile counts toward (0: nobody waits)."""
    unit = _arrive_unit(tile.task, tile.batch, total_generations, args.mpm)
    address = Int64(0)
    if tile.task == Int32(TASK_X):
        address = pool.counter_address(args.replayed_x_routed_ready, unit)
    elif tile.task == Int32(TASK_DY):
        address = pool.counter_address(args.d_y_routed_ready, unit)
    else:
        address = pool.counter_address(args.reverse_dispatch_done, unit)
    return address


# ---------------------------------------------------------------- load / store (lanes)


@cute.jit
def _routed(args: BwdCommArgs, peer_rank: Int32) -> Boolean:
    """A schedule row is routed when its source rank is a valid EP rank (padding rows are -1)."""
    routed = Boolean(peer_rank >= Int32(0))
    if peer_rank >= args.ep_size:
        routed = Boolean(False)
    return routed


@cute.jit
def _load_row(
    args: BwdCommArgs,
    slot_pool: pool.SlotPool,
    tile: pool.InFlightTile,
    schedule_row: Int32,
    ring_row: Int32,
    peer_rank: Int32,
    routed: Boolean,
    row_chunk: cute.Pointer,
    mbarrier: cute.Pointer,
    tile_bytes: Int32,
) -> None:
    """One lane's part of ``_load``, after the slot's mbarrier was armed: issue this lane's row read."""
    if routed:
        route_index = args.schedule_peer_token_idx[schedule_row]
        if tile.task == Int32(TASK_X):
            source = (
                _peer_address(args.x_peer_table, peer_rank)
                + Int64(route_index // args.topk) * Int64(args.hidden_size) * Int64(2)
                + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
            )
            tma_load_1d_raw(row_chunk, source, mbarrier, tile_bytes)
        elif tile.task == Int32(TASK_DY):
            source = (
                _peer_address(args.d_y_peer_table, peer_rank)
                + Int64(route_index // args.topk) * Int64(args.hidden_size) * Int64(2)  # dY is per token, like X
                + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
            )
            tma_load_1d_raw(row_chunk, source, mbarrier, tile_bytes)
            if tile.col_block == Int32(0):
                args.router_weights[ring_row] = _load_peer_float(args.router_peer_table, peer_rank, route_index)
        else:
            source = (
                args.d_x_routed.iterator.toint()
                + Int64(ring_row) * Int64(args.hidden_size) * Int64(2)
                + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
            )
            tma_load_1d_raw(row_chunk, source, mbarrier, tile_bytes)
    else:
        if tile.task == Int32(TASK_DY):
            if tile.col_block == Int32(0):
                args.router_weights[ring_row] = Float32(0.0)  # padding row: exact zero score


@cute.jit
def _load(args: BwdCommArgs, slot_pool: pool.SlotPool, tile: pool.InFlightTile) -> None:
    """load: arm the slot's mbarrier and issue this lane's row read.  X / dY: the peer token row
    (route_index // topk; column block 0 of dY also copies the route's router score into the
    ring); dX: local d_x ring row.  Unrouted rows load nothing; the lanes past the tile's rows have no row
    (they only take part in the ballot and the sync)."""
    lane_idx = Int32(cute.arch.lane_idx())
    tile_bytes = slot_pool.tile_bytes(args.hidden_size, tile.col_block)
    schedule_row = tile.batch * args.minibatch_size + tile.row + lane_idx
    ring_row = (tile.batch % args.mpm) * args.minibatch_size + tile.row + lane_idx
    row_chunk = slot_pool.row_chunk(tile.slot, lane_idx)
    mbarrier = slot_pool.mbarriers + tile.slot
    peer_rank = slot_pool.lane_entry(args.schedule_peer_rank, schedule_row, lane_idx)
    routed = _routed(args, peer_rank)
    count = Int32(cute.arch.popc(cute.arch.vote_ballot_sync(routed)))
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, count * tile_bytes)
    cute.arch.sync_warp()
    if cutlass.const_expr(slot_pool.shape.rows == pool.WARP_LANES):
        _load_row(args, slot_pool, tile, schedule_row, ring_row, peer_rank, routed, row_chunk, mbarrier, tile_bytes)
    else:
        if lane_idx < Int32(slot_pool.shape.rows):
            _load_row(args, slot_pool, tile, schedule_row, ring_row, peer_rank, routed, row_chunk, mbarrier, tile_bytes)


@cute.jit
def _store_row(
    args: BwdCommArgs,
    slot_pool: pool.SlotPool,
    tile: pool.InFlightTile,
    schedule_row: Int32,
    ring_row: Int32,
    peer_rank: Int32,
    routed: Boolean,
    row_chunk: cute.Pointer,
    tile_bytes: Int32,
) -> None:
    """One lane's part of ``_store``, after the slot's loads arrived: store this lane's row chunk and wait for
    the store."""
    if tile.task != Int32(TASK_DX):
        if not routed:
            # one st.bulk per lane (word stores: one bank for all lanes)
            zero_shared_bulk(row_chunk, slot_pool.shape.row_chunk_bytes)
            cute.arch.fence_proxy("async.shared", space="cta")
        ring_base = args.d_y_routed.iterator.toint()
        if tile.task == Int32(TASK_X):
            ring_base = args.x_routed.iterator.toint()
        destination = ring_base + Int64(ring_row) * Int64(args.hidden_size) * Int64(2) + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
        tma_store_1d_raw(destination, row_chunk, tile_bytes)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)  # the ring row is complete when the tile is counted
    else:
        if routed:
            route_index = args.schedule_peer_token_idx[schedule_row]
            destination = (
                _peer_address(args.d_x_peer_table, peer_rank)
                + Int64(route_index) * Int64(args.hidden_size) * Int64(2)
                + Int64(tile.col_block) * Int64(slot_pool.shape.row_chunk_bytes)
            )
            tma_store_1d_raw(destination, row_chunk, tile_bytes)
            cute.arch.cp_async_bulk_commit_group()
            if tile.col_block == Int32(0):
                d_router = Float32(0.0)
                for partial in cutlass.range(args.s_tiles):
                    d_router = d_router + Float32(args.d_router_partials[ring_row, partial])
                _store_peer_float(args.d_router_peer_table, peer_rank, route_index, d_router)
        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def _store(args: BwdCommArgs, slot_pool: pool.SlotPool, tile: pool.InFlightTile, slot_phase: Int32) -> None:
    """store: wait for the slot's loads, then X / dY: store the row into the local ring (unrouted
    rows zero) and wait for the store; dX: store the row to the peer dX buffer (column block 0
    also writes the row's d_router sum to the peer) and wait until the slot has been read.  The whole warp
    waits for the loads and syncs after the stores; the lanes past the tile's rows store nothing in between."""
    lane_idx = Int32(cute.arch.lane_idx())
    tile_bytes = slot_pool.tile_bytes(args.hidden_size, tile.col_block)
    schedule_row = tile.batch * args.minibatch_size + tile.row + lane_idx
    ring_row = (tile.batch % args.mpm) * args.minibatch_size + tile.row + lane_idx
    row_chunk = slot_pool.row_chunk(tile.slot, lane_idx)
    peer_rank = slot_pool.lane_entry(args.schedule_peer_rank, schedule_row, lane_idx)
    routed = _routed(args, peer_rank)
    cute.arch.mbarrier_wait(slot_pool.mbarriers + tile.slot, slot_phase)
    if cutlass.const_expr(slot_pool.shape.rows == pool.WARP_LANES):
        _store_row(args, slot_pool, tile, schedule_row, ring_row, peer_rank, routed, row_chunk, tile_bytes)
    else:
        if lane_idx < Int32(slot_pool.shape.rows):
            _store_row(args, slot_pool, tile, schedule_row, ring_row, peer_rank, routed, row_chunk, tile_bytes)
    cute.arch.sync_warp()


@cute.jit
def _finish(args: BwdCommArgs, slot_pool: pool.SlotPool, tile: pool.InFlightTile, batch: pool.ArriveBatch, total_generations: Int32) -> pool.ArriveBatch:
    """store(tile); release(tile.slot); count the tile toward its counter (the batch is released
    when the counter changes)."""
    slot_phase = pool.phase(slot_pool, tile.slot)
    _store(args, slot_pool, tile, slot_phase)
    pool.release(slot_pool, tile.slot, slot_phase)
    return batch.add(_arrive_address(args, tile, total_generations))


# ---------------------------------------------------------------- the stream


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
        h = k // half
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
def _stream_task(
    index: Int32, generations: Int32, mpm: Int32, mini_tiles: Int32, last_tiles: Int32, recompute_forward_context: cutlass.Constexpr = False
) -> tuple[Int32, Int32, Int32]:
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
    rows."""
    task = Int32(TASK_DY)
    g = index // mini_tiles
    tile_row = index - g * mini_tiles
    first_gens = cutlass.min(mpm, generations)
    first_size = (first_gens - Int32(1)) * mini_tiles + mini_tiles
    if first_gens == generations:
        first_size = (first_gens - Int32(1)) * mini_tiles + last_tiles
    if cutlass.const_expr(recompute_forward_context):
        # Two halves per generation, X then dY (only the stream's last generation can be short, and
        # nothing follows it in this part).
        first_size = Int32(2) * first_size
        g = index // (Int32(2) * mini_tiles)
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
                g = j // mini_tiles
                tile_row = j - g * mini_tiles
            else:
                j = j - early_pushes
                full_steps = generations - Int32(1) - mpm
                step_size = Int32(3) * mini_tiles
                if j < full_steps * step_size:
                    in_tail = Int32(0)
                    s = j // step_size
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
            g = tail_g0 + j // mini_tiles
            tile_row = j - (j // mini_tiles) * mini_tiles
    return task, g, tile_row


@cute.jit
def run_comm_prefix(args: BwdCommArgs, slot_pool: pool.SlotPool, comm_cta: Int32, num_comm_sms: Int32) -> None:
    """The dynamic task stream (module docstring): each of the COMM_WARPS warps takes its first task statically
    (warp x comm SMs + CTA) and every later one with an atomic add on the head; a warp waits, holding no slot, for a
    taken task that is not ready."""
    pool.init(slot_pool)

    mpm = args.mpm
    total_generations = _ceil_div_i32(args.num_tokens, args.minibatch_size)
    mini_tiles = pool.row_tiles(slot_pool.shape, args.minibatch_size)
    last_tiles = pool.row_tiles(slot_pool.shape, _generation_rows(total_generations - Int32(1), args.num_tokens, args.minibatch_size))
    rows_total = (total_generations - Int32(1)) * mini_tiles + last_tiles
    x_rows = Int32(0)  # X pulls: generations mpm .. G-1
    if total_generations > mpm:
        x_rows = rows_total - mpm * mini_tiles
    if cutlass.const_expr(args.recompute_forward_context):
        x_rows = rows_total  # every generation
    tasks = Int32(2) * rows_total + x_rows
    col_blocks = pool.column_blocks(slot_pool.shape, args.hidden_size)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    warps_total = Int32(pool.COMM_WARPS) * num_comm_sms
    head_address = pool.counter_address(args.reverse_dispatch_done, args.reverse_dispatch_done.shape[0] - Int32(1))
    prev = pool.no_tile()  # the tile whose store is not finished yet
    batch = pool.no_arrivals()  # finished tiles not yet released
    if warp_idx < Int32(pool.COMM_WARPS):
        task_index = warp_idx * num_comm_sms + comm_cta
        while task_index < tasks:
            task, g, tile_row = _stream_task(task_index, total_generations, mpm, mini_tiles, last_tiles, args.recompute_forward_context)
            direction = Int32(PULL)
            if task == Int32(TASK_DX):
                direction = Int32(PUSH)
            if _segment_reached(args, direction, g) == Int32(0):
                # not ready: release what we hold, then wait
                if prev.slot >= Int32(0):
                    batch = _finish(args, slot_pool, prev, batch, total_generations)
                    prev = prev.stored()
                batch = batch.flush()
                _segment_wait(args, direction, g)
            for col_block in cutlass.range(col_blocks):
                # acquire: block only while holding no slot
                slot = pool.acquire(slot_pool, Int32(0))
                if slot < Int32(0):
                    if prev.slot >= Int32(0):
                        batch = _finish(args, slot_pool, prev, batch, total_generations)
                        prev = prev.stored()
                    slot = pool.acquire(slot_pool, Int32(1))
                tile = pool.InFlightTile(slot=slot, task=task, batch=g, row=tile_row * Int32(slot_pool.shape.rows), col_block=col_block)
                _load(args, slot_pool, tile)
                if prev.slot >= Int32(0):
                    batch = _finish(args, slot_pool, prev, batch, total_generations)
                prev = tile
            # the next task of the stream after the statically dealt first ones
            taken = Int32(0)
            if cute.arch.lane_idx() == Int32(0):
                head = cute.make_ptr(Int32, head_address, cute.AddressSpace.gmem, assumed_align=4)
                taken = cute.arch.atomic_add(head, Int32(1), sem="relaxed", scope="gpu")
            task_index = cute.arch.shuffle_sync(taken, 0) + warps_total
        if prev.slot >= Int32(0):
            batch = _finish(args, slot_pool, prev, batch, total_generations)
        batch.flush()
    # in-flight peer stores must complete before the CTA exits
    cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.sync_threads()
