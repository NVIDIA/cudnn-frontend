# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Warp slots of the communication CTAs, shared by the forward and backward task streams.

Private device module of ``_fwd_dynamic_task_stream`` and ``_bwd_dynamic_task_stream``.  Both run the dynamic task
stream on every communication CTA: every pull / push tile row of the kernel is one task of a single
linear stream in dependency order; the COMM_WARPS warps of every communication CTA are identical, each
takes its first task statically (warp x comm SMs + CTA) and every later one with an atomic add on a
global head, decodes it, waits while it is not ready, and moves it.  Warp ``w`` owns slot ``w``: nobody else
uses it, and the warp holds one tile at a time.  This module holds the part that does not depend on the
direction: the slots, the tile geometry, the copy of one prepared tile and the leader-side wait / arrive
primitives.  The direction modules supply the stream (which task is number i), the task description (what it
waits for, which counter a finished tile counts toward) and the per-lane addresses of a tile (``_prepare``).

One warp (leader = one elected lane; lanes = the lanes that own a tile row, lane = row):
    take a task; wait(counter[idx] >= n)   leader polls with a nanosleep backoff, the warp syncs behind it
    per column block:
        prepare(tile)              every schedule / table / peer read of the tile's load and store (TileCopy)
        wait_slot_read             lanes wait until their earlier TMA writes have read slot w (bulk wait_group.read)
        load(tile -> slot w)       lanes issue one TMA read each (one row chunk) into the slot's mbarrier
        count(tile)                deferred, batched per counter (ArriveBatch): an arrive first waits, a pull batch
                                   for the completion of the warp's writes (bulk wait_group), a push batch until
                                   its stores have read the slot (bulk wait_group.read), then leader
                                   atom.release.gpu
        store(slot w -> dest)      lanes wait the mbarrier, issue one TMA write each (not waited)
    take the next task (claim, decode, dependency wait) while the last stores are in flight
While a tile's store is in flight the warp does everything for the next tile but its TMA: the claim of the next
task (or the next column block of this one), its decode, its dependency and its prepare; then wait_slot_read and
the next load are issued back to back.

Device state is grouped in frozen dataclasses: ``CommSlots`` (the CTA's slot storage), ``TileCopy`` (one lane's
prepared copy of a tile) and ``ArriveBatch`` (finished tiles not yet arrived).  The direction modules group their
tensors and runtime scalars the same way (``FwdCommArgs`` / ``BwdCommArgs``).  A slot's mbarrier phase is kept by
its warp in a register.

The tile shape (contract ``CommTileShape``: rows x columns of one 32 KiB slot) is the compile-time parameter of
the kernel instance, held by ``CommSlots``: 32 x 512 (every lane owns a row and moves 1 KiB of it per bulk copy)
or 16 x 1024 (lanes 0..15 own a row and move 2 KiB of it per bulk copy; lanes 16..31 move nothing and only take
part in the warp-wide ballots and syncs).  The slot layout, the bytes of a copy and the tile counts (``row_tiles``,
``column_blocks``) are derived from it; the slot bytes, the slot count and the 224 KiB arena are the same for
every shape; every size, count and index is a runtime value.
"""

import dataclasses

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

# The counter wait (relaxed poll + nanosleep(16) + fence.acquire.gpu).
from ._bwd_device_helpers import _counter_wait_gpu
from ._bwd_device_helpers import _ceil_div_i32, _store_global_b32
from ._tma_1d import tma_load_1d_raw, tma_store_1d_raw, zero_shared_bulk
from .persistent_bf16_fwd_contract import COMM_SLOT_BYTES, COMM_TILE_SHAPES, CommTileShape

WARP_LANES = 32
SLOT_BYTES = COMM_SLOT_BYTES  # one tile of any shape (the contract checks that the shapes agree)
assert all(shape.rows <= WARP_LANES for shape in COMM_TILE_SHAPES)  # lane = row
COMM_WARPS = 7  # warps 0..6 of a communication CTA move tiles, warp w through slot w (the rest of the CTA idles)
PULL = 0  # direction of a tile: peer -> local ring
PUSH = 1  # local -> peer

# Shared-memory sizes of the communication CTA (read by the mega kernels).
ARENA_BYTES = COMM_WARPS * SLOT_BYTES  # 224 KiB
MBARRIERS = COMM_WARPS  # one transaction barrier per slot
assert ARENA_BYTES <= 229376


@dataclasses.dataclass(frozen=True)
class CommSlots:
    """Shared-memory storage of the slots: one transaction mbarrier per slot and the slot arena (aliases the
    compute arena); and the tile shape (compile time, one of the contract's COMM_TILE_SHAPES), which lays the
    slots out: slot ``w`` (warp ``w``'s) is ``shape.slot_bytes`` at ``w x shape.slot_bytes``, lane
    ``r < shape.rows`` owns the ``shape.row_chunk_bytes`` at ``r x shape.row_chunk_bytes`` of it."""

    mbarriers: cute.Pointer
    arena: cute.Pointer
    shape: CommTileShape

    @cute.jit
    def row_chunk(self, slot: Int32, lane_idx: Int32) -> cute.Pointer:
        """This lane's row chunk of ``slot``."""
        return self.arena + slot * Int32(self.shape.slot_bytes) + lane_idx * Int32(self.shape.row_chunk_bytes)

    @cute.jit
    def tile_bytes(self, hidden_size: Int32, col_block: Int32) -> Int32:
        """Bytes of one row chunk of column block ``col_block`` (the last block may be short): one bulk copy,
        and per routed row the transaction bytes of the slot's mbarrier."""
        return cutlass.min(Int32(self.shape.columns), hidden_size - col_block * Int32(self.shape.columns)) * Int32(2)

    @cute.jit
    def lane_entry(self, table: cute.Tensor, index: Int32, lane_idx: Int32) -> Int32:
        """``table[index]``, the schedule entry of this lane's row; -1 (not routed) for a lane past the tile's
        rows, which reads nothing."""
        if cutlass.const_expr(self.shape.rows == WARP_LANES):
            entry = table[index]
        else:
            entry = Int32(-1)
            if lane_idx < Int32(self.shape.rows):
                entry = Int32(table[index])
        return entry

    @cute.jit
    def owns_row(self, lane_idx: Int32) -> Int32:
        """1 if this lane has a row of the tile (every lane of a 32-row tile)."""
        owns = Int32(1)
        if cutlass.const_expr(self.shape.rows != WARP_LANES):
            if lane_idx >= Int32(self.shape.rows):
                owns = Int32(0)
        return owns


@dataclasses.dataclass(frozen=True)
class TileCopy:
    """One lane's copy of a tile, prepared (``_prepare`` of the direction module) while the warp's previous store is
    in flight; the load and the store read nothing else.  ``source``: the global address of this lane's row chunk
    (0: the lane loads nothing, an unrouted row or a lane past the tile's rows); ``destination``: where the row
    chunk goes (0: the lane stores nothing); ``zero``: 1 to zero the lane's row chunk of the slot before the store
    (a pull row that is not routed); ``row_bytes``: bytes of the row chunk (the column block's); ``tx_bytes``:
    warp-uniform transaction bytes of the slot's mbarrier (routed rows x ``row_bytes``); ``side_address`` /
    ``side_value``: one 4-byte value the lane stores after its row (0: none; the backward dX row's router-weight
    gradient); ``arrive`` / ``wait_writes``: warp-uniform, the counter the tile counts toward and its completion
    (``ArriveBatch``)."""

    source: Int64
    destination: Int64
    zero: Int32
    row_bytes: Int32
    tx_bytes: Int32
    side_address: Int64
    side_value: Float32
    arrive: Int64
    wait_writes: Int32


@cute.jit
def row_tiles(shape: cutlass.Constexpr, rows: Int32) -> Int32:
    """Tile rows of ``rows`` routed rows (a multiple of 256): the tasks of a stream per direction, and with
    ``column_blocks`` the arrivals of a readiness counter over those rows (one per tile)."""
    return rows // Int32(shape.rows)


@cute.jit
def column_blocks(shape: cutlass.Constexpr, hidden_size: Int32) -> Int32:
    """Column blocks of one row: the tiles of a task (the last block may be short)."""
    return _ceil_div_i32(hidden_size, Int32(shape.columns))


@dataclasses.dataclass(frozen=True)
class ArriveBatch:
    """Finished tiles not yet arrived: ``count`` arrivals for the counter word at ``address``
    (0: nobody waits).  ``wait_writes`` 1: the counter's waiters read what the tiles wrote (a pull into the
    local ring), the arrive waits for the writes; 0: the counter gates the reuse of the local rows the tiles
    read (a push to a peer; those rows were read by the load before its store was issued), the arrive waits
    for the slot reads only.  Warp-uniform; every method is called by the whole warp."""

    address: Int64
    count: Int32
    wait_writes: Int32

    @cute.jit
    def fire(self) -> None:
        """arrive: every lane waits for its stores (``wait_writes``: complete, else read their slot), then the
        leader releases the batched count (atom.release.gpu)."""
        if self.address != Int64(0):
            if self.wait_writes != Int32(0):
                cute.arch.cp_async_bulk_wait_group(0)  # pull: compute tasks of this kernel read the ring rows next
            else:
                # push: read only -- the local rows were read by the load; the peers read after the end-of-kernel
                # drain and barrier
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                counter = cute.make_ptr(Int32, self.address, cute.AddressSpace.gmem, assumed_align=4)
                cute.arch.atomic_add(counter, self.count, sem="release", scope="gpu")
        cute.arch.sync_warp()

    @cute.jit
    def add(self, address: Int64, wait_writes: Int32) -> "ArriveBatch":
        """Count one finished tile toward the counter at ``address`` (``wait_writes``: the counter's, see the
        class); the pending count is arrived first when the counter changes."""
        pending_address = self.address
        pending_count = self.count
        if address != self.address:
            self.fire()
            pending_address = address
            pending_count = Int32(0)
        return ArriveBatch(address=pending_address, count=pending_count + Int32(1), wait_writes=wait_writes)

    @cute.jit
    def flush(self) -> "ArriveBatch":
        """Arrive the pending count (before blocking on a wait, and at the end)."""
        self.fire()
        return no_arrivals()


@cute.jit
def no_arrivals() -> ArriveBatch:
    return ArriveBatch(address=Int64(0), count=Int32(0), wait_writes=Int32(0))


@cute.jit
def counter_address(counter: cute.Tensor, index: Int32) -> Int64:
    """Address of ``counter[index]`` (the identity of an arrive unit), 0 for index -1 (nobody waits)."""
    address = Int64(0)
    if index >= Int32(0):
        address = (counter.iterator + index).toint()
    return address


@cute.jit
def init(slots: CommSlots) -> None:
    """Whole CTA: initialise the slot barriers (phase 0)."""
    thread_idx = cute.arch.thread_idx()[0]
    if thread_idx < Int32(COMM_WARPS):
        cute.arch.mbarrier_init(slots.mbarriers + thread_idx, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()


@cute.jit
def wait_slot_read() -> None:
    """Whole warp: wait until the warp's stores have read its slot (their writes may still be in flight); the slot
    is free for the next load."""
    cute.arch.cp_async_bulk_wait_group(0, read=True)  # read only: reusing the smem slot needs no write completion
    cute.arch.sync_warp()


@cute.jit
def load(slots: CommSlots, copy: TileCopy, slot: Int32) -> None:
    """load, right after ``wait_slot_read``: arm the slot's mbarrier and issue this lane's row read from the prepared
    source (no global read here)."""
    mbarrier = slots.mbarriers + slot
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, copy.tx_bytes)
    cute.arch.sync_warp()
    if copy.source != Int64(0):
        tma_load_1d_raw(slots.row_chunk(slot, Int32(cute.arch.lane_idx())), copy.source, mbarrier, copy.row_bytes)


@cute.jit
def store(slots: CommSlots, copy: TileCopy, slot: Int32, slot_phase: Int32) -> None:
    """store: wait for the slot's loads, zero an unrouted pull row, issue this lane's row store to the prepared
    destination (not waited: the slot is reusable after ``wait_slot_read``; before the tile's arrival a pull waits
    for the writes, a push for the reads), then the lane's side value.  The whole warp waits for the loads and syncs
    after the stores."""
    row_chunk = slots.row_chunk(slot, Int32(cute.arch.lane_idx()))
    cute.arch.mbarrier_wait(slots.mbarriers + slot, slot_phase)
    if copy.zero != Int32(0):
        # one st.bulk per lane (word-by-word stores would hit one bank for all lanes)
        zero_shared_bulk(row_chunk, slots.shape.row_chunk_bytes)
        cute.arch.fence_proxy("async.shared", space="cta")
    if copy.destination != Int64(0):
        tma_store_1d_raw(copy.destination, row_chunk, copy.row_bytes)
        cute.arch.cp_async_bulk_commit_group()
    if copy.side_address != Int64(0):
        _store_global_b32(copy.side_address, copy.side_value.bitcast(Uint32))
    cute.arch.sync_warp()


@cute.jit
def move(slots: CommSlots, copy: TileCopy, slot: Int32, slot_phase: Int32, batch: ArriveBatch) -> ArriveBatch:
    """load(tile -> slot); count the tile toward its counter (deferred: the batch is arrived when the counter
    changes; a pull batch after its writes are complete, a push batch after its stores have read the slot); issue
    its store (not waited)."""
    load(slots, copy, slot)
    batch = batch.add(copy.arrive, copy.wait_writes)
    store(slots, copy, slot, slot_phase)
    return batch


@cute.jit
def reached(counter: cute.Tensor, index: Int32, required: Int32) -> Int32:
    """Warp-uniform: 1 if counter[index] >= required already (lane 0 reads, the value is broadcast;
    elect_one does not fix the lane a shuffle can read from).  The warp barrier orders the other lanes' later
    memory accesses after lane 0's acquire (a shuffle alone does not)."""
    seen = Int32(0)
    if cute.arch.lane_idx() == Int32(0):
        if cute.arch.load(counter.iterator + index, Int32, sem="acquire", scope="gpu") >= required:
            seen = Int32(1)
    seen = cute.arch.shuffle_sync(seen, 0)
    cute.arch.sync_warp()
    return seen


@cute.jit
def wait(counter: cute.Tensor, index: Int32, required: Int32) -> None:
    """Leader blocks until counter[index] >= required; the warp syncs behind it."""
    with cute.arch.elect_one():
        _counter_wait_gpu(counter, index, required)
    cute.arch.sync_warp()
