# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Backward communication prefix: the dynamic task stream of X replay pulls, dY pulls (reverse combine) and
dX pushes (reverse dispatch).

Private device module of ``_persistent_bf16_bwd_mega``.  The same protocol as the forward ``_fwd_dynamic_task_stream``
(the warp slots and primitives are ``_dynamic_task_stream``; the loop has the same shape: take a task, wait for it,
per tile prepare its addresses while the previous store is in flight / wait for the slot to be read / load into the
warp's own slot / count (deferred) / issue the store, batched arrive (X / dY after the writes complete, dX after the
slot is read), take the next number while the store is in flight).  Only the stream, the task description and the
load / store addresses (``_prepare``) differ:

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

Expected counts are per R-row x C-column tile, the tile shape of the kernel variant (``CommSlots.shape``, compile
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

from . import _dynamic_task_stream as comm
from ._bwd_device_helpers import _ceil_div_i32, _load_peer_float, _peer_address
from .persistent_bf16_bwd_contract import CLUSTER_SIZE, MLP_TILE_M, MLP_TILE_N

PULL = comm.PULL
PUSH = comm.PUSH
TASK_X = 0  # replay pull
TASK_DY = 1  # pull
TASK_DX = 2  # push

# Interface read by the mega kernel (shared memory of the communication CTA).
ARENA_BYTES = comm.ARENA_BYTES
MBARRIERS = comm.MBARRIERS


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
            reached = comm.reached(args.routed_buffers_done, previous, args.slot_reuse_required)
            if reached == Int32(1):
                reached = comm.reached(args.wgrad_read_consumed, previous, experts * Int32(3) * args.wgrad_tiles)
    else:
        reached = comm.reached(args.d_x_routed_ready, g, _gate_up_dgrad_tiles(g, args.num_tokens, args.minibatch_size, args.hidden_size, args.has_tail))
    return reached


@cute.jit
def _segment_wait(args: BwdCommArgs, direction: Int32, g: Int32) -> None:
    """Whole warp: block until segment ``g`` may start (same targets as ``_segment_reached``)."""
    if direction == Int32(PULL):
        if g >= args.mpm:
            previous = g - args.mpm
            experts = Int32(args.minibatch_expert_offsets[previous + Int32(1)]) - Int32(args.minibatch_expert_offsets[previous])
            comm.wait(args.routed_buffers_done, previous, args.slot_reuse_required)
            comm.wait(args.wgrad_read_consumed, previous, experts * Int32(3) * args.wgrad_tiles)
    else:
        comm.wait(args.d_x_routed_ready, g, _gate_up_dgrad_tiles(g, args.num_tokens, args.minibatch_size, args.hidden_size, args.has_tail))


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
def _arrive_address(args: BwdCommArgs, task: Int32, g: Int32, total_generations: Int32) -> Int64:
    """Counter word a finished tile counts toward (0: nobody waits)."""
    unit = _arrive_unit(task, g, total_generations, args.mpm)
    address = Int64(0)
    if task == Int32(TASK_X):
        address = comm.counter_address(args.replayed_x_routed_ready, unit)
    elif task == Int32(TASK_DY):
        address = comm.counter_address(args.d_y_routed_ready, unit)
    else:
        address = comm.counter_address(args.reverse_dispatch_done, unit)
    return address


# ---------------------------------------------------------------- prepare (lanes)


@cute.jit
def _routed(args: BwdCommArgs, peer_rank: Int32) -> Boolean:
    """A schedule row is routed when its source rank is a valid EP rank (padding rows are -1)."""
    routed = Boolean(peer_rank >= Int32(0))
    if peer_rank >= args.ep_size:
        routed = Boolean(False)
    return routed


@cute.jit
def _prepare(args: BwdCommArgs, slots: comm.CommSlots, task: Int32, g: Int32, row: Int32, col_block: Int32, total_generations: Int32) -> comm.TileCopy:
    """This lane's copy of the tile (row ``row`` of generation ``g``, column block ``col_block``), every read of its
    load and store done here, while the warp's previous store is in flight.  X / dY: from the peer token row
    (route_index // topk) into the local ring (an unrouted row is zeroed); column block 0 of dY also copies the
    route's router score into the ring (0 for a padding row).  dX: from the local d_x ring row to the peer dX
    buffer; column block 0 also sums the row's d_router partials for the peer (``side_value``).  The lanes past the
    tile's rows move nothing.  The router score and the d_router partials are read here, after the task's
    dependency, as before (in the load / the store)."""
    lane_idx = Int32(cute.arch.lane_idx())
    row_bytes = slots.tile_bytes(args.hidden_size, col_block)
    column = Int64(col_block) * Int64(slots.shape.row_chunk_bytes)
    schedule_row = g * args.minibatch_size + row + lane_idx
    ring_row = (g % args.mpm) * args.minibatch_size + row + lane_idx
    ring_offset = Int64(ring_row) * Int64(args.hidden_size) * Int64(2) + column
    peer_rank = slots.lane_entry(args.schedule_peer_rank, schedule_row, lane_idx)
    routed = _routed(args, peer_rank)
    count = Int32(cute.arch.popc(cute.arch.vote_ballot_sync(routed)))
    owns_row = slots.owns_row(lane_idx)
    source = Int64(0)
    destination = Int64(0)
    zero = Int32(0)
    side_address = Int64(0)
    side_value = Float32(0.0)
    wait_writes = Int32(0)  # dX: reverse_dispatch_done gates the reuse of the local dX rows
    if task != Int32(TASK_DX):
        wait_writes = Int32(1)  # the compute reads the X / dY ring rows
        if owns_row != Int32(0):
            ring_base = args.d_y_routed.iterator.toint()
            peer_base = Int64(0)
            if task == Int32(TASK_X):
                ring_base = args.x_routed.iterator.toint()
            destination = ring_base + ring_offset
            if routed:
                route_index = args.schedule_peer_token_idx[schedule_row]
                if task == Int32(TASK_X):
                    peer_base = _peer_address(args.x_peer_table, peer_rank)
                else:
                    peer_base = _peer_address(args.d_y_peer_table, peer_rank)  # dY is per token, like X
                    if col_block == Int32(0):
                        args.router_weights[ring_row] = _load_peer_float(args.router_peer_table, peer_rank, route_index)
                source = peer_base + Int64(route_index // args.topk) * Int64(args.hidden_size) * Int64(2) + column
            else:
                zero = Int32(1)
                if task == Int32(TASK_DY):
                    if col_block == Int32(0):
                        args.router_weights[ring_row] = Float32(0.0)  # padding row: exact zero score
    else:
        if routed:
            route_index = args.schedule_peer_token_idx[schedule_row]
            source = args.d_x_routed.iterator.toint() + ring_offset
            destination = _peer_address(args.d_x_peer_table, peer_rank) + Int64(route_index) * Int64(args.hidden_size) * Int64(2) + column
            if col_block == Int32(0):
                for partial in cutlass.range(args.s_tiles):
                    side_value = side_value + Float32(args.d_router_partials[ring_row, partial])
                side_address = _peer_address(args.d_router_peer_table, peer_rank) + Int64(route_index) * Int64(4)
    return comm.TileCopy(
        source=source,
        destination=destination,
        zero=zero,
        row_bytes=row_bytes,
        tx_bytes=count * row_bytes,
        side_address=side_address,
        side_value=side_value,
        arrive=_arrive_address(args, task, g, total_generations),
        wait_writes=wait_writes,
    )


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
def run_comm_prefix(args: BwdCommArgs, slots: comm.CommSlots, comm_cta: Int32, num_comm_sms: Int32) -> None:
    """The dynamic task stream (module docstring): each of the COMM_WARPS warps takes its first task statically
    (warp x comm SMs + CTA) and every later one with an atomic add on the head; a warp waits for a taken task that is
    not ready, then moves its tiles one at a time through its own slot."""
    comm.init(slots)

    mpm = args.mpm
    total_generations = _ceil_div_i32(args.num_tokens, args.minibatch_size)
    mini_tiles = comm.row_tiles(slots.shape, args.minibatch_size)
    last_tiles = comm.row_tiles(slots.shape, _generation_rows(total_generations - Int32(1), args.num_tokens, args.minibatch_size))
    rows_total = (total_generations - Int32(1)) * mini_tiles + last_tiles
    x_rows = Int32(0)  # X pulls: generations mpm .. G-1
    if total_generations > mpm:
        x_rows = rows_total - mpm * mini_tiles
    if cutlass.const_expr(args.recompute_forward_context):
        x_rows = rows_total  # every generation
    tasks = Int32(2) * rows_total + x_rows
    col_blocks = comm.column_blocks(slots.shape, args.hidden_size)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    warps_total = Int32(comm.COMM_WARPS) * num_comm_sms
    head_address = comm.counter_address(args.reverse_dispatch_done, args.reverse_dispatch_done.shape[0] - Int32(1))
    slot_phase = Int32(0)  # the phase of the warp's slot mbarrier
    batch = comm.no_arrivals()  # finished tiles not yet arrived
    if warp_idx < Int32(comm.COMM_WARPS):
        task_index = warp_idx * num_comm_sms + comm_cta
        while task_index < tasks:
            task, g, tile_row = _stream_task(task_index, total_generations, mpm, mini_tiles, last_tiles, args.recompute_forward_context)
            direction = Int32(PULL)
            if task == Int32(TASK_DX):
                direction = Int32(PUSH)
            if _segment_reached(args, direction, g) == Int32(0):
                # not ready: arrive what we finished, then wait
                batch = batch.flush()
                _segment_wait(args, direction, g)
            for col_block in cutlass.range(col_blocks):
                # every read of the tile while the previous store is in flight, then the load at once
                copy = _prepare(args, slots, task, g, tile_row * Int32(slots.shape.rows), col_block, total_generations)
                comm.wait_slot_read()  # the previous tile's stores have read slot w
                batch = comm.move(slots, copy, warp_idx, slot_phase, batch)
                slot_phase = slot_phase ^ Int32(1)
            # the next task of the stream after the statically dealt first ones (the last stores are in flight)
            taken = Int32(0)
            if cute.arch.lane_idx() == Int32(0):
                head = cute.make_ptr(Int32, head_address, cute.AddressSpace.gmem, assumed_align=4)
                taken = cute.arch.atomic_add(head, Int32(1), sem="relaxed", scope="gpu")
            task_index = cute.arch.shuffle_sync(taken, 0) + warps_total
        batch.flush()
    # in-flight stores (the last ones of every warp, and those of tiles nobody waits for) complete before the CTA exits
    cute.arch.cp_async_bulk_wait_group(0)
    cute.arch.sync_threads()
