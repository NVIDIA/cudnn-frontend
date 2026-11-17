# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Slot pool of the communication CTAs, shared by the forward and backward task streams.

Private device module of ``_fwd_dynamic_task_stream`` and ``_bwd_dynamic_task_stream``.  Both run the dynamic task
stream on every communication CTA: every pull / push tile row of the kernel is one task of a single
linear stream in dependency order; the COMM_WARPS warps of every communication CTA are identical, each
takes its first task statically (warp x comm SMs + CTA) and every later one with an atomic add on a
global head, decodes it, waits (holding no slot) while it is not ready, and moves it.  This module holds
the part that does not depend on the direction: the slot pool, the tile geometry and the leader-side
wait / acquire / release primitives.  The direction modules supply the stream (which task is number i),
the task description (what it waits for, which counter a finished tile counts toward) and the load /
store addresses.

Primitives (leader = one elected lane; lanes = the lanes that own a tile row, lane = row):
    wait(counter[idx] >= n)   leader polls with a nanosleep backoff, the warp syncs behind it
    arrive(counter[idx], n)   leader atom.release.gpu, batched per counter (ArriveBatch)
    acquire(pool)             lane 0 CAS on the pool word, slot broadcast by shuffle
    release(slot)             leader flips the slot's mbarrier phase, atomic_or returns the slot
    load(tile -> slot)        lanes issue one TMA read each (one row chunk) into the slot's mbarrier (no wait)
    store(slot -> dest)       lanes wait the mbarrier, issue one TMA write each, wait the writes

Device state is grouped in frozen dataclasses: ``SlotPool`` (the CTA's pool storage), ``InFlightTile``
(the loaded tile whose store is pending) and ``ArriveBatch`` (finished tiles not yet released).  The
direction modules group their tensors and runtime scalars the same way (``FwdCommArgs`` / ``BwdCommArgs``).

Slots are taken per use by whichever warp needs one, first come, with no partition between pull and
push: the stream order already keeps the awaited tiles ahead.  Each warp overlaps two tiles, so
COMM_WARPS x 2 > NUM_SLOTS and the pool stays full whichever direction has work.

The tile shape (contract ``CommTileShape``: rows x columns of one 32 KiB slot) is the compile-time parameter of
the kernel instance, held by ``SlotPool``: 32 x 512 (every lane owns a row and moves 1 KiB of it per bulk copy)
or 16 x 1024 (lanes 0..15 own a row and move 2 KiB of it per bulk copy; lanes 16..31 move nothing and only take
part in the warp-wide ballots and syncs).  The slot layout, the bytes of a copy and the tile counts (``row_tiles``,
``column_blocks``) are derived from it; the slot bytes, the slot count and the 224 KiB arena are the same for
every shape; every size, count and index is a runtime value.
"""

import dataclasses

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64

# The counter wait (relaxed poll + nanosleep(16) + fence.acquire.gpu) and nanosleep.
from ._bwd_device_helpers import _counter_wait_gpu, _nanosleep_compat
from ._bwd_device_helpers import _ceil_div_i32
from .persistent_bf16_fwd_contract import COMM_SLOT_BYTES, COMM_TILE_SHAPES, CommTileShape

WARP_LANES = 32
SLOT_BYTES = COMM_SLOT_BYTES  # one tile of any shape (the contract checks that the shapes agree)
assert all(shape.rows <= WARP_LANES for shape in COMM_TILE_SHAPES)  # lane = row
NUM_SLOTS = 7  # 224 KiB pool
COMM_WARPS = 7  # warps 0..6 of a communication CTA move tiles (identical; the rest of the CTA idles)
PULL = 0  # direction of a tile: peer -> local ring
PUSH = 1  # local -> peer

# Pool word: bits 0..NUM_SLOTS-1 free slots.
_POOL_WORD = 0
_PHASE_WORD0 = 1  # per-slot mbarrier phase (a slot changes owner)
assert NUM_SLOTS < 31 and COMM_WARPS <= 8

# Shared-memory sizes of the communication CTA (read by the mega kernels).
ARENA_BYTES = NUM_SLOTS * SLOT_BYTES
MBARRIERS = NUM_SLOTS  # one transaction barrier per slot
SCRATCH_WORDS = _PHASE_WORD0 + NUM_SLOTS  # pool word + per-slot phases
assert ARENA_BYTES <= 229376


@dataclasses.dataclass(frozen=True)
class SlotPool:
    """Shared-memory storage of the pool: one transaction mbarrier per slot, the scratch words
    (pool word + per-slot phases) and the slot arena (aliases the compute arena); and the tile shape
    (compile time, one of the contract's COMM_TILE_SHAPES), which lays the slots out: slot ``s`` is
    ``shape.slot_bytes`` at ``s x shape.slot_bytes``, lane ``r < shape.rows`` owns the ``shape.row_chunk_bytes``
    at ``r x shape.row_chunk_bytes`` of it."""

    mbarriers: cute.Pointer
    scratch: cute.Tensor
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


@dataclasses.dataclass(frozen=True)
class InFlightTile:
    """The tile a warp has loaded and not stored yet (``slot`` -1: none).  ``task``: the direction's task
    kind (forward: PULL / PUSH; backward: X / dY / dX).  ``batch``: the batch whose rows the tile carries
    (forward: the macrobatch; backward: the generation).  ``row``: the tile's first row inside that
    batch (a row of the macrobatch / generation, not a global schedule row)."""

    slot: Int32
    task: Int32
    batch: Int32
    row: Int32
    col_block: Int32

    @cute.jit
    def stored(self) -> "InFlightTile":
        """The same warp after the tile's store: holds nothing."""
        return InFlightTile(slot=Int32(-1), task=self.task, batch=self.batch, row=self.row, col_block=self.col_block)


@cute.jit
def row_tiles(shape: cutlass.Constexpr, rows: Int32) -> Int32:
    """Tile rows of ``rows`` routed rows (a multiple of 256): the tasks of a stream per direction, and with
    ``column_blocks`` the arrivals of a readiness counter over those rows (one per tile)."""
    return rows // Int32(shape.rows)


@cute.jit
def column_blocks(shape: cutlass.Constexpr, hidden_size: Int32) -> Int32:
    """Column blocks of one row: the tiles of a task (the last block may be short)."""
    return _ceil_div_i32(hidden_size, Int32(shape.columns))


@cute.jit
def no_tile() -> InFlightTile:
    return InFlightTile(slot=Int32(-1), task=Int32(0), batch=Int32(0), row=Int32(0), col_block=Int32(0))


@dataclasses.dataclass(frozen=True)
class ArriveBatch:
    """Finished tiles not yet released: ``count`` arrivals for the counter word at ``address``
    (0: nobody waits).  Warp-uniform; every method is called by the whole warp."""

    address: Int64
    count: Int32

    @cute.jit
    def fire(self) -> None:
        """arrive: the leader releases the batched count (atom.release.gpu)."""
        if self.address != Int64(0):
            with cute.arch.elect_one():
                counter = cute.make_ptr(Int32, self.address, cute.AddressSpace.gmem, assumed_align=4)
                cute.arch.atomic_add(counter, self.count, sem="release", scope="gpu")
        cute.arch.sync_warp()

    @cute.jit
    def add(self, address: Int64) -> "ArriveBatch":
        """Count one finished tile toward the counter at ``address``; the pending count is released
        first when the counter changes."""
        pending_address = self.address
        pending_count = self.count
        if address != self.address:
            self.fire()
            pending_address = address
            pending_count = Int32(0)
        return ArriveBatch(address=pending_address, count=pending_count + Int32(1))

    @cute.jit
    def flush(self) -> "ArriveBatch":
        """Release the pending count (before blocking on a wait, and at the end)."""
        self.fire()
        return no_arrivals()


@cute.jit
def no_arrivals() -> ArriveBatch:
    return ArriveBatch(address=Int64(0), count=Int32(0))


@cute.jit
def counter_address(counter: cute.Tensor, index: Int32) -> Int64:
    """Address of ``counter[index]`` (the identity of an arrive unit), 0 for index -1 (nobody waits)."""
    address = Int64(0)
    if index >= Int32(0):
        address = (counter.iterator + index).toint()
    return address


@cute.jit
def init(slot_pool: SlotPool) -> None:
    """Whole CTA: initialise the slot barriers, mark every slot free, zero the phases."""
    thread_idx = cute.arch.thread_idx()[0]
    if thread_idx < Int32(NUM_SLOTS):
        cute.arch.mbarrier_init(slot_pool.mbarriers + thread_idx, 1)
    if thread_idx < Int32(SCRATCH_WORDS):
        slot_pool.scratch[thread_idx] = Int32(0)
    if thread_idx == Int32(0):
        slot_pool.scratch[_POOL_WORD] = Int32((1 << NUM_SLOTS) - 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()


@cute.jit
def reached(counter: cute.Tensor, index: Int32, required: Int32) -> Int32:
    """Warp-uniform: 1 if counter[index] >= required already (lane 0 reads, the value is broadcast;
    elect_one does not fix the lane a shuffle can read from)."""
    seen = Int32(0)
    if cute.arch.lane_idx() == Int32(0):
        if cute.arch.load(counter.iterator + index, Int32, sem="acquire", scope="gpu") >= required:
            seen = Int32(1)
    return cute.arch.shuffle_sync(seen, 0)


@cute.jit
def wait(counter: cute.Tensor, index: Int32, required: Int32) -> None:
    """Leader blocks until counter[index] >= required; the warp syncs behind it."""
    with cute.arch.elect_one():
        _counter_wait_gpu(counter, index, required)
    cute.arch.sync_warp()


@cute.jit
def _try_acquire(scratch: cute.Tensor, block: Int32) -> Int32:
    """Leader: take the lowest free slot, or -1 (only when ``block`` is 0)."""
    slot = Int32(-1)
    trying = Boolean(True)
    while trying:
        state = cute.arch.load(scratch.iterator + Int32(_POOL_WORD), Int32, sem="relaxed", scope="cta")
        free = state & Int32((1 << NUM_SLOTS) - 1)
        if free != Int32(0):
            candidate = Int32(0)
            found = Int32(0)
            for s in cutlass.range_constexpr(NUM_SLOTS):
                if found == Int32(0):
                    if (free >> Int32(s)) & Int32(1) == Int32(1):
                        candidate = Int32(s)
                        found = Int32(1)
            old = cute.arch.atomic_cas(scratch.iterator + Int32(_POOL_WORD), cmp=state, val=state & ~(Int32(1) << candidate), sem="acquire", scope="cta")
            if old == state:
                slot = candidate
                trying = Boolean(False)
        else:
            if block == Int32(0):
                trying = Boolean(False)
            else:
                _nanosleep_compat(32)
    return slot


@cute.jit
def acquire(slot_pool: SlotPool, block: Int32) -> Int32:
    """Warp-uniform slot (-1 only when ``block`` is 0: the caller holds a tile and must finish it first).
    Lane 0 takes the slot and broadcasts it (a shuffle needs a fixed source lane)."""
    slot = Int32(-1)
    if cute.arch.lane_idx() == Int32(0):
        slot = _try_acquire(slot_pool.scratch, block)
    return cute.arch.shuffle_sync(slot, 0)


@cute.jit
def phase(slot_pool: SlotPool, slot: Int32) -> Int32:
    """Current mbarrier phase of ``slot`` (kept in shared memory: a slot changes owner)."""
    return slot_pool.scratch[_PHASE_WORD0 + slot]


@cute.jit
def release(slot_pool: SlotPool, slot: Int32, slot_phase: Int32) -> None:
    """Whole warp, after the slot's stores were waited: flip its phase and return it to the pool."""
    with cute.arch.elect_one():
        slot_pool.scratch[_PHASE_WORD0 + slot] = slot_phase ^ Int32(1)
        cute.arch.atomic_or(slot_pool.scratch.iterator + Int32(_POOL_WORD), Int32(1) << slot, sem="release", scope="cta")
    cute.arch.sync_warp()
