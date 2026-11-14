# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chain-relay conflict-rank dKV storage ("chain") for the deterministic SM100 M64 DSA backward kernel.

Selected with ``deterministic_dkv_mode="chain"`` where ``_chain_dkv_selected`` holds
(``dsa_bwd_sm100_deterministic``); every other shape keeps the records storage of
``dsa_bwd_sm100_rank_shard_m64`` or the shard layout.

Waves and tickets. A wave holds ``RANK_M64_CHAIN_WAVE`` query positions (``rank_wave_query`` spreads the
queries over the waves) and runs every head block in one main-kernel launch of one CTA per (position, head
block). Each CTA claims a ticket with one atomic in start order; ticket ``t`` runs position
``t // head_blocks`` and head block ``t % head_blocks``. A CTA claims before its ``griddepcontrol.wait``;
the wave's ``rank_setup_a`` clears the counter after its own wait, before ``rank_setup_b`` lets any CTA of
the main kernel launch.

Chains. ``rank_setup_a`` sorts each position's active rows (one CTA per position) and marks each row's
writer positions; ``rank_setup_b`` gives every slot its conflict rank ``c`` among the wave's positions that
select the row. A row is spread over ``stride`` planes (``rank_chain_spread``): rank ``c`` adds into plane
``c mod stride`` at chain step ``c // stride``, so each plane row has one chain of writers in ticket
order. Step 0 of a plane row that holds no data yet stores; later steps add. A slot repeated within a
query's sorted rows is skipped, and the first one scales its FP32 fragment by the repeat count (code bits
at ``RANK_M64_MULT_SHIFT``): repeated slots contribute identical values. A row written by every active
position, with more writers than planes, uses the ``RANK_M64_HOT_PLANES`` hot plane rows of its slot among
position 0's first ``RANK_M64_HOT_ROWS`` sorted rows instead; the next ``rank_setup_b`` folds them into plane
0 in plane order.

Hand-off. A chain member waits for its predecessor before it adds. The CTA's idle warp
(``rank_chain_acquire_warp``) runs one waiting tile ahead of the reducers: it polls each waiting row's
semaphore until ``ld.acquire.gpu`` observes the predecessor's value, then passes the tile to the reducer
warps through named barriers. After a tile's adds the reducers pass it through named barriers to the load
warp (``rank_chain_release_warp``), which writes each successor's value with ``st.release.gpu``. Member
``m`` of a chain (counting head blocks) waits for ``tag + m`` and publishes ``tag + m + 1``; the wave's
``tag = (2 * w + 1) * 256`` (ticket word 4 holds ``2 * w + 1``), so values of earlier waves never match and
the semaphores are cleared once per execution, by ``rank_init``. A CTA waits only for a lower ticket, held
by a CTA that already runs, so the relay progresses whatever the residency.

Launch sequence of one execution (an ordinary launch starts after the preceding grid completed; a
programmatic one may start once every CTA of the preceding grid triggered or exited):

- ``rank_init`` (ordinary): no trigger, no wait. Clears the row state, initialized-plane counts,
  semaphores, ticket words and hot-row words.
- For each wave ``w``:

  - ``rank_setup_a`` (programmatic): triggers at entry. In the first wave every CTA waits at entry, since
    ``rank_init`` neither waits nor triggers. Otherwise its first ``RANK_M64_SETUP_WAITERS`` item CTAs wait
    after their items and the other CTAs exit. Before that wait it sorts wave ``w``, sums its O.dO and
    scaled LSE, finishes wave ``w - 1`` (reads its sorted rows, codes, chain words and row state; raises
    initialized-plane counts; marks hot rows and cold flags; clears its row state) and marks wave ``w``'s
    row state: data the main kernel ``w - 1`` does not access. After it, the waiters clear the ticket
    counter, tile flags and depths and write the wave's tag.
  - ``rank_scan_first`` (programmatic, first wave only): waits, then triggers.
  - ``rank_setup_b`` (programmatic): waits, then triggers. Folds wave ``w - 1``'s hot rows, then writes
    wave ``w``'s codes, chain words, tile flags and hot depths.
  - main kernel (programmatic): every CTA waits before it triggers or exits, and only the load warp
    triggers, after its wait. Before waiting a CTA claims its ticket and reads the inputs and wave ``w``'s
    sorted rows.

- ``rank_setup_a`` (no current wave) and ``rank_setup_b`` (hot fold only) finish the last wave.
- ``rank_fold_convert`` (programmatic): waits, then triggers.
- ``sum_dSink`` (programmatic, ``_ConflictRankPlan``): CTA (0, 0, 0) waits at its end.

Invariants. Every programmatic grid has a CTA that waits before the grid completes, so the completion of
``sum_dSink`` implies the whole sequence's: the next ordinary launch on the stream (the next execution's
``rank_init``, or a consumer of the outputs) starts after every write of this execution. A grid triggers
before waiting only where its dependents touch nothing before their own waits (``rank_setup_a``), so a grid
that starts early finds every grid before its primary complete. The kernels rely on the visibility of
completed grids: a write of a grid whose completion a wait has observed is visible to later grids before
their own waits (the main kernel's tickets and sorted rows, ``rank_setup_b``'s hot fold of the preceding
main kernel's hot planes, ``sum_dSink``'s statistics).

Workspace (32-bit words in this order, ``pitch = round_up(S_kv, 8)``, ``R = round_up(D, 8)``,
``S = RANK_M64_CHAIN_WAVE * K``, ``H = min(K, RANK_M64_HOT_ROWS) * RANK_M64_HOT_PLANES``): dense planes
``P * pitch * R``; hot plane rows ``H * R``; row state, two parities ``2 * pitch * rank_state_words()``;
``pitch`` words no kernel accesses; destination codes ``S``; ``RANK_M64_CHAIN_COUNTER_WORDS`` words no kernel
accesses; initialized planes per row ``pitch``; sorted rows, two parities ``2 * S``; chain words ``S``;
semaphores ``P * pitch + H``; ticket words ``RANK_M64_TICKET_WORDS``; ``2 * RANK_M64_CHAIN_WAVE`` words no
kernel accesses; tile flags and hot depths ``(K // 64 + 1) * RANK_M64_CHAIN_WAVE`` each; hot-row words,
two parities. The unaccessed regions stay because the plane count ``P`` (``rank_chain_planes(...,
records=False)``, repeated on the device by ``rank_geometry``) is the budget left after the fixed
regions: dropping them could change ``P`` for some KV extents, and with it the summation order.
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
import cutlass.pipeline as pipeline
from cudnn._cutlass_compat import SmemAllocator

from ._rank_common_sm100 import (
    RANK_M64_COMPACT_BYTES,
    RANK_M64_DENSE_BUDGET_BYTES,
    RANK_M64_FRESH_FLAG,
    RANK_M64_PAIR_BUDGET_BYTES,
    RANK_M64_SETUP_CTAS,
    RANK_M64_STATE_WORDS,
    RANK_M64_THREADS,
    RANK_M64_WAVE,
    ConflictRankReducerMixin,
    _partial_cta_sync,
    _rank_load_u32x4,
    _rank_prefetch_l2,
    _rank_store_f32x2,
    _rank_store_f32x4,
    _rank_store_zero_x4,
    _rank_topk,
    rank_m64_pitch,
    rank_m64_record,
)

# The records rule's wave (``dsa_bwd_sm100_rank_shard_m64``), which the chain plane count and the selection
# pre-screen start from; a chain wave holds two of them.
RANK_M64_CHAIN_WAVES = 2
RANK_M64_CHAIN_WAVE = RANK_M64_CHAIN_WAVES * RANK_M64_WAVE
# Plane policy of the records rule (see ``dsa_bwd_sm100_rank_shard_m64``).
# Row-state words per row of the records rule's sizing (chain plans keep ``rank_state_words()`` words).
# Words between the destination codes and the initialized-plane counts (budgeted, not accessed).
RANK_M64_CHAIN_COUNTER_WORDS = 8
# Grid of rank_init, rank_scan_first and the final rank_setup_b.
# Grid of the per-wave setups: one sort CTA per position, then the item CTAs (eight resident CTAs per SM).
RANK_M64_WAVE_SETUP_CTAS = 1184
# Item CTAs of rank_setup_a that wait for the preceding grid (the others exit before waiting).
RANK_M64_SETUP_WAITERS = 8
# Destination-code flag: the slot stores (first writer of its plane row).
# Chain word of a slot: chain step (bits 0..6), member flag, successor flag, plane row at RANK_M64_CHAIN_ROW_SHIFT.
RANK_M64_CHAIN_SUCCESSOR = 1 << 8
RANK_M64_CHAIN_MEMBER = 1 << 7
RANK_M64_CHAIN_POSITION_MASK = RANK_M64_CHAIN_MEMBER - 1
RANK_M64_CHAIN_ROW_SHIFT = 9
# A plan with fewer records-rule planes than this takes the whole 1 GiB plane budget (``rank_chain_planes``) and
# runs chain-only only if that does not make its workspace both larger and above 1 GiB (``rank_chain_only``).
RANK_M64_CHAIN_MIN_PLANES = 4
# Planes a row that not every active position writes spreads over in a cold wave (``rank_chain_cold``, ``rank_chain_spread``).
RANK_M64_CHAIN_SPREAD = 3
# Fewest dense planes of a chain plan.
RANK_M64_CHAIN_FLOOR_PLANES = 3
# Per (position, tile) flags: a slot waits for a predecessor, releases a successor, is a chain member.
RANK_M64_TILE_WAITS = 1
RANK_M64_TILE_RELEASES = 2
RANK_M64_TILE_MEMBERS = 4
# Ticket words: 0 the ticket counter, 4 the wave tag, 5 the active positions, 6 the deepest hot chain, and
# 2, 3, 7 the cold flags of three rotating waves (``rank_chain_flag_word``).
RANK_M64_ACTIVE_WORD = 5
RANK_M64_DEPTH_WORD = 6
RANK_M64_TICKET_WORDS = 8
# Plane rows a code or chain word can address (dense planes and hot plane rows).
RANK_M64_CHAIN_MAX_PLANE_ROWS = 1 << 19
RANK_M64_CHAIN_ROW_MASK = RANK_M64_CHAIN_MAX_PLANE_ROWS - 1
# The reducers carry a tile's relay flags at this bit of their per-tile word.
RANK_M64_RELAY_FLAG_SHIFT = 28
# Code bits of a slot's repeat count minus one; top-k widths up to RANK_M64_MULT_LIMIT keep it in range.
RANK_M64_MULT_SHIFT = 19
RANK_M64_MULT_LIMIT = 1 << 11
# Code of a repeated slot whose first slot carries the repeat count.
RANK_M64_SKIP_CODE = -1
RANK_M64_SORT_SENTINEL = 0x7FFFFFFF
# Hot plane rows per hot slot, and position 0's sorted slots that may take them.
RANK_M64_HOT_PLANES = 32
RANK_M64_HOT_SHIFT = 5
assert RANK_M64_HOT_PLANES == 1 << RANK_M64_HOT_SHIFT
RANK_M64_HOT_ROWS = 64
# Hot-row word flag: plane 0 of the row already holds data (the fold adds instead of storing).
RANK_M64_HOT_INIT = 1 << 30
# KV rows per rank_fold_convert CTA.
RANK_M64_FOLD_ROWS = 4


def rank_chain_wave(*, records: bool) -> int:
    """Return a plan's wave positions."""
    return RANK_M64_WAVE if records else RANK_M64_CHAIN_WAVE


def rank_chain_state_words(*, records: bool) -> int:
    """Return the row-state words per row and parity."""
    return RANK_M64_STATE_WORDS if records else (rank_chain_wave(records=False) + 127) // 128 * 4


def rank_chain_metadata_words(pitch: int, record: int, slots: int, *, records: bool, state_words: int = RANK_M64_STATE_WORDS) -> int:
    """Return the 32-bit words of everything but the dense planes."""
    record_slots = slots if records else 0
    return (
        record_slots * record
        + 2 * pitch * state_words
        + pitch
        + record_slots
        + 8 * record_slots
        + 4 * record_slots
        + 2 * record_slots
        + slots
        + RANK_M64_CHAIN_COUNTER_WORDS
        + pitch
    )


def rank_chain_fixed_words(slots: int, wave: int = RANK_M64_WAVE) -> int:
    """Return the chain-relay words that do not scale with the planes."""
    tile_words = slots // wave // 64 * wave + wave
    return 3 * slots + RANK_M64_TICKET_WORDS + 2 * wave + 2 * tile_words


def rank_chain_hot_slots(max_topk: int) -> int:
    """Return a chain plan's hot plane rows."""
    return min(int(max_topk), RANK_M64_HOT_ROWS) * RANK_M64_HOT_PLANES


def rank_chain_planes(total_s_kv: int, head_dim: int, max_topk: int, *, records: bool) -> int:
    """Return the dense plane count of one plan.

    ``records=True``: the records rule for a full ``RANK_M64_WAVE`` wave. ``records=False``: the chain plans' count,
    at least the records-rule count and ``RANK_M64_CHAIN_FLOOR_PLANES``, filling the plane budget."""
    if not records:
        planes = rank_chain_planes(total_s_kv, head_dim, max_topk, records=True)
        pitch = rank_m64_pitch(total_s_kv)
        record = rank_m64_record(head_dim)
        slots = RANK_M64_WAVE * int(max_topk)
        room = min(RANK_M64_DENSE_BUDGET_BYTES // 4, rank_chain_workspace_words(total_s_kv, head_dim, max_topk, records=True))
        if planes < RANK_M64_CHAIN_MIN_PLANES:
            room = RANK_M64_DENSE_BUDGET_BYTES // 4
        fit = (room - rank_chain_metadata_words(pitch, record, slots, records=False) - rank_chain_fixed_words(slots)) // (pitch * (record + 1))
        budget = RANK_M64_DENSE_BUDGET_BYTES // (pitch * record * 4)
        return max(RANK_M64_CHAIN_FLOOR_PLANES, planes, min(RANK_M64_WAVE - 1, budget, fit))
    pitch = rank_m64_pitch(total_s_kv)
    record = rank_m64_record(head_dim)
    slots = RANK_M64_WAVE * int(max_topk)
    density = (slots + pitch - 1) // pitch
    budget = RANK_M64_DENSE_BUDGET_BYTES // (pitch * record * 4)
    if budget < 2 and 2 * pitch * record * 4 <= RANK_M64_PAIR_BUDGET_BYTES and 2 * slots >= pitch:
        budget = 2
    margin = 1 + (1 if slots > 2 * pitch else 0)
    planes = max(1, min(RANK_M64_WAVE - 1, density + margin, budget))
    if (planes * pitch * record + rank_chain_metadata_words(pitch, record, slots, records=True)) * 4 > RANK_M64_COMPACT_BYTES:
        planes = max(1, min(RANK_M64_WAVE - 1, density, budget))
    return planes


def rank_chain_region_words(pitch: int, planes: int, slots: int, wave: int = RANK_M64_WAVE) -> int:
    """Return the 32-bit words of the chain-relay regions."""
    return planes * pitch + rank_chain_fixed_words(slots, wave)


def rank_chain_workspace_words(total_s_kv: int, head_dim: int, max_topk: int, *, records: bool) -> int:
    """Return the 32-bit word count of the conflict-rank dKV workspace."""
    pitch = rank_m64_pitch(total_s_kv)
    record = rank_m64_record(head_dim)
    wave = rank_chain_wave(records=records)
    slots = wave * int(max_topk)
    planes = rank_chain_planes(total_s_kv, head_dim, max_topk, records=records)
    metadata = rank_chain_metadata_words(pitch, record, slots, records=records, state_words=rank_chain_state_words(records=records))
    hot = 0 if records else rank_chain_hot_slots(max_topk)
    return planes * pitch * record + metadata + rank_chain_region_words(pitch, planes, slots, wave) + hot * (record + 1) + hot // RANK_M64_HOT_PLANES * 2


def rank_chain_workspace_rows(total_s_kv: int, head_dim: int, max_topk: int, *, records: bool) -> int:
    """Return the workspace size in whole records."""
    record = rank_m64_record(head_dim)
    return -(-rank_chain_workspace_words(total_s_kv, head_dim, max_topk, records=records) // record)


def rank_chain_workspace_bytes(total_s_kv: int, head_dim: int, max_topk: int, *, records: bool) -> int:
    """Return the conflict-rank dKV workspace in bytes."""
    return rank_chain_workspace_rows(total_s_kv, head_dim, max_topk, records=records) * rank_m64_record(head_dim) * 4


def rank_chain_only(total_s_kv: int, head_dim: int, max_topk: int) -> bool:
    """Return whether a generic M64 plan runs chain-only (no records)."""
    if int(max_topk) > RANK_M64_MULT_LIMIT:
        return False
    if rank_chain_planes(total_s_kv, head_dim, max_topk, records=True) < RANK_M64_CHAIN_MIN_PLANES:
        record_bytes = rank_chain_workspace_bytes(total_s_kv, head_dim, max_topk, records=True)
        chain_bytes = rank_chain_workspace_bytes(total_s_kv, head_dim, max_topk, records=False)
        if chain_bytes >= record_bytes and chain_bytes > RANK_M64_DENSE_BUDGET_BYTES:
            return False
    return (
        rank_chain_planes(total_s_kv, head_dim, max_topk, records=False) * rank_m64_pitch(total_s_kv) + rank_chain_hot_slots(max_topk)
        <= RANK_M64_CHAIN_MAX_PLANE_ROWS
    )


@cute.jit
def _rank_wait_value(pointer: cute.Pointer, expected: Int32):
    """Wait until ``*pointer == expected`` (relaxed polls, then an acquire load)."""
    current = cute.arch.load(pointer, Int32, sem="relaxed", scope="gpu")
    loop = cutlass.while_generate([current], lambda value: value != expected)
    with loop as [value]:
        value = cute.arch.load(pointer, Int32, sem="relaxed", scope="gpu")
        cutlass.yield_out([value])
    acquired = cute.arch.load(pointer, Int32, sem="acquire", scope="gpu")
    acquire_loop = cutlass.while_generate([acquired], lambda value: value != expected)
    with acquire_loop as [value]:
        value = cute.arch.load(pointer, Int32, sem="acquire", scope="gpu")
        cutlass.yield_out([value])


class RankChainM64Reduction(ConflictRankReducerMixin):
    """Conflict-rank dKV storage: geometry, wave setup, codes and fold."""

    rank_shard = True
    rank_chain = True

    def rank_wave(self) -> int:
        """Return the positions of one wave (``RANK_M64_CHAIN_WAVE``)."""
        return rank_chain_wave(records=False)

    def rank_state_words(self) -> int:
        """Return the row-state words per row and parity (``rank_chain_state_words``)."""
        return rank_chain_state_words(records=False)

    @cute.jit
    def _rank_state_count(self, entry: cute.Pointer, ordinal: Optional[Int32]):
        """Return a row's writers in the wave and those below ``ordinal``."""
        ordinal_words = (self.rank_wave() + 31) // 32
        chunks = (ordinal_words + 3) // 4
        bits = cute.make_rmem_tensor((4 * chunks,), Uint32)
        for chunk in cutlass.range_constexpr(chunks):
            b0, b1, b2, b3 = _rank_load_u32x4(entry + Int64(4 * chunk))
            bits[4 * chunk] = b0
            bits[4 * chunk + 1] = b1
            bits[4 * chunk + 2] = b2
            bits[4 * chunk + 3] = b3
        count = Int32(0)
        conflict = Int32(0)
        full = Int32(0)
        partial = Uint32(0)
        if cutlass.const_expr(ordinal is not None):
            full = ordinal >> Int32(5)
            partial = (Uint32(1) << Uint32(ordinal & Int32(31))) - Uint32(1)
        for word in cutlass.range_constexpr(ordinal_words):
            ones = Int32(cute.arch.popc(bits[word]))
            count += ones
            if cutlass.const_expr(ordinal is not None):
                if Int32(word) < full:
                    conflict += ones
                elif Int32(word) == full:
                    conflict += Int32(cute.arch.popc(bits[word] & partial))
        return count, conflict

    @cute.jit
    def rank_geometry(self, seqlen_kv: Int32):
        """Return the plane pitch and the dense plane count ``(pitch, planes)`` of one KV extent (with range-checked
        32-bit quotients; ``rank_chain_planes(..., records=False)`` on the host). The launchers compute it once and
        pass it to every kernel, which hands the pair to the helpers that need it (their ``geometry`` argument)."""
        record = rank_m64_record(self.head_dim)
        slots = RANK_M64_WAVE * self.max_topk
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        if cutlass.const_expr(slots < (1 << 30)):
            density = Int64(1)
            if pitch <= Int64(1 << 30):
                density = Int64((Int32(slots - 1) + Int32(pitch)) // Int32(pitch))
        else:
            density = (Int64(slots) + pitch - Int64(1)) // pitch
        margin = Int64(1)
        if Int64(slots) > Int64(2) * pitch:
            margin = Int64(2)
        dense = Int64(0)
        if pitch <= Int64(RANK_M64_DENSE_BUDGET_BYTES // (record * 4)):
            dense = Int64(Int32(RANK_M64_DENSE_BUDGET_BYTES) // (Int32(pitch) * Int32(record * 4)))
        budget = dense
        if budget < Int64(2):
            if Int64(2) * pitch * Int64(record * 4) <= Int64(RANK_M64_PAIR_BUDGET_BYTES):
                if Int64(2 * slots) >= pitch:
                    budget = Int64(2)
        planes = min(density + margin, budget)
        planes = min(planes, Int64(RANK_M64_WAVE - 1))
        planes = max(planes, Int64(1))
        metadata = Int64(slots * record) + Int64(2 * RANK_M64_STATE_WORDS + 2) * pitch + Int64(slots * 16 + RANK_M64_CHAIN_COUNTER_WORDS)
        if (planes * pitch * Int64(record) + metadata) * Int64(4) > Int64(RANK_M64_COMPACT_BYTES):
            planes = min(density, budget)
            planes = min(planes, Int64(RANK_M64_WAVE - 1))
            planes = max(planes, Int64(1))
        chain_fixed = Int64(rank_chain_fixed_words(slots))
        room = planes * pitch * Int64(record + 1) + metadata + chain_fixed
        room = min(room, Int64(RANK_M64_DENSE_BUDGET_BYTES // 4))
        if planes < Int64(RANK_M64_CHAIN_MIN_PLANES):
            room = Int64(RANK_M64_DENSE_BUDGET_BYTES // 4)
        chain_metadata = Int64(2 * RANK_M64_STATE_WORDS + 2) * pitch + Int64(slots + RANK_M64_CHAIN_COUNTER_WORDS)
        # room <= 2**28 words; no fit keeps planes
        remainder = room - chain_metadata - chain_fixed
        row_words = pitch * Int64(record + 1)
        fit = Int64(0)
        if remainder > Int64(0) and row_words <= remainder:
            fit = Int64(Int32(remainder) // Int32(row_words))
        fit = min(fit, dense)
        fit = min(fit, Int64(RANK_M64_WAVE - 1))
        planes = max(planes, fit)
        planes = max(planes, Int64(RANK_M64_CHAIN_FLOOR_PLANES))
        return pitch, Int32(planes)

    @cute.jit
    def rank_offsets(self, pitch: Int64, planes: Int32):
        """Return the 32-bit word offsets of the row state and the destination codes (see the module docstring)."""
        record = Int64(rank_m64_record(self.head_dim))
        state = Int64(planes) * pitch * record + Int64(self.rank_hot_slots()) * record
        # The row-state parities are followed by one row of ``pitch`` words that the plane-count rule budgets
        # (``rank_chain_metadata_words``) and no kernel accesses.
        codes = state + Int64(2 * self.rank_state_words()) * pitch + pitch
        return state, codes

    @cute.jit
    def rank_high(self, pitch: Int64, planes: Int32) -> Int64:
        """Return the word offset of the per-row initialized-plane counts (behind ``RANK_M64_CHAIN_COUNTER_WORDS`` unused words)."""
        _, codes = self.rank_offsets(pitch, planes)
        return codes + Int64(self.rank_wave() * self.max_topk) + Int64(RANK_M64_CHAIN_COUNTER_WORDS)

    @cute.jit
    def rank_chain_offsets(self, pitch: Int64, planes: Int32):
        """Return the word offsets of the chain-relay regions."""
        slots = Int64(self.rank_wave() * self.max_topk)
        sorted_rows = self.rank_high(pitch, planes) + pitch
        chain = sorted_rows + Int64(2) * slots
        semaphores = chain + slots
        tickets = semaphores + Int64(planes) * pitch + Int64(self.rank_hot_slots())
        ticket_slots = tickets + Int64(RANK_M64_TICKET_WORDS)
        return sorted_rows, chain, semaphores, tickets, ticket_slots

    @cute.jit
    def rank_tile_flags(self, pitch: Int64, planes: Int32) -> Int64:
        """Return the word offset of the per (position, tile) relay flags."""
        _, _, _, _, ticket_slots = self.rank_chain_offsets(pitch, planes)
        return ticket_slots + Int64(2 * self.rank_wave())

    @cute.jit
    def rank_hot_depths(self, pitch: Int64, planes: Int32) -> Int64:
        """Return the word offset of the per (position, tile) hot-row depths."""
        return self.rank_tile_flags(pitch, planes) + Int64(self.max_topk // 64 * self.rank_wave() + self.rank_wave())

    @cute.jit
    def rank_hot_list(self, pitch: Int64, planes: Int32) -> Int64:
        """Return the word offset of the two parities' hot-row words."""
        return self.rank_hot_depths(pitch, planes) + Int64(self.max_topk // 64 * self.rank_wave() + self.rank_wave())

    def rank_hot_rows(self) -> int:
        """Return position 0's sorted slots that may take hot planes."""
        return min(int(self.max_topk), RANK_M64_HOT_ROWS)

    def rank_hot_slots(self) -> int:
        """Return the hot plane rows (``rank_chain_hot_slots``)."""
        return self.rank_hot_rows() * RANK_M64_HOT_PLANES

    @cute.jit
    def rank_tile_order(self, mdKV_acc: cute.Tensor, geometry, ordinal: Int32, tile_count: Int32, order: cute.Tensor, lane: Int32, head_blocks: Int32):
        """One warp: write ``ordinal``'s tile order."""
        assert self.rank_tiles() <= 32, "chain plans order at most 32 tiles (top-k width <= 2048) in one warp"
        pitch, planes = geometry
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Int32)
        depths = words + self.rank_hot_depths(pitch, planes) + Int64(ordinal) * Int64(self.rank_tiles())
        _, _, _, tickets, _ = self.rank_chain_offsets(pitch, planes)
        deepest = cute.arch.load(words + tickets + Int64(RANK_M64_DEPTH_WORD), Int32) - Int32(1)
        span = max(deepest, tile_count - Int32(1))
        key = Int32(0x7FFFFFFF)
        if lane < tile_count:
            depth = cute.arch.load(depths + Int64(lane), Int32)
            natural = tile_count - Int32(1) - lane
            key = Int32(2) * natural + Int32(1)
            if depth > Int32(0):
                spread = Int32(self.rank_hot_stagger()) * (depth - Int32(1)) * (tile_count - Int32(1)) // max(span, Int32(1))
                if cutlass.const_expr(self.rank_tiles() < 16):
                    ticket = ordinal * head_blocks
                    spread = ticket % Int32(RANK_M64_WAVE) * ((tile_count - Int32(1)) * Int32(3) // Int32(4)) // Int32(RANK_M64_WAVE - 1)
                    if ticket >= Int32(RANK_M64_WAVE):
                        spread += (tile_count - Int32(1)) // Int32(4)
                key = Int32(2) * min(natural, spread)
        slot = Int32(0)
        for other in cutlass.range_constexpr(32):
            other_key = cute.arch.shuffle_sync(key, other)
            if other_key < key:
                slot += Int32(1)
            if other_key == key and Int32(other) > lane:
                slot += Int32(1)
        if lane < tile_count:
            order[slot] = lane

    def rank_tiles(self) -> int:
        """Return the reducer tiles of one query's top-k row."""
        return (int(self.max_topk) + int(self.block_tile) - 1) // int(self.block_tile)

    def rank_hot_stagger(self) -> int:
        """Return the tile steps between successive hot-chain depths."""
        return 2 if self.rank_tiles() >= 16 else 1

    @cute.jit
    def rank_wave_query(self, wave_begin: Int32, ordinal: Int32, total_q: Int32) -> Int32:
        """Return the query at wave position ``ordinal`` (position ``p`` of wave ``w`` of ``n`` holds query ``p * n + (w + p) % n``)."""
        waves = (total_q + Int32(self.rank_wave() - 1)) // Int32(self.rank_wave())
        wave = wave_begin // Int32(self.rank_wave())
        return ordinal * waves + Int32(Uint32(wave + ordinal) % Uint32(waves))

    @cute.jit
    def rank_wave_active(self, wave_begin: Int32, total_q: Int32) -> Int32:
        """Return the number of wave positions whose query exists."""
        waves = (total_q + Int32(self.rank_wave() - 1)) // Int32(self.rank_wave())
        wave = wave_begin // Int32(self.rank_wave())
        full = total_q // waves
        active = min(Int32(self.rank_wave()), full)
        if full < Int32(self.rank_wave()):
            if (wave + full) % waves < total_q - full * waves:
                active += Int32(1)
        return active

    @cute.jit
    def rank_wave_size(self, wave_begin: Int32, total_q: Int32) -> Int32:
        """Return the wave positions to scan (every position; those past ``total_q`` hold no query)."""
        return Int32(self.rank_wave())

    @cute.jit
    def rank_sorted_view(self, mdKV_acc: cute.Tensor, geometry, parity: Int32) -> cute.Tensor:
        """Return one wave parity's sorted rows by wave position."""
        pitch, planes = geometry
        sorted_rows, _, _, _, _ = self.rank_chain_offsets(pitch, planes)
        slots = Int64(self.rank_wave() * self.max_topk)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Int32)
        return cute.make_tensor(
            words + sorted_rows + Int64(parity) * slots,
            cute.make_layout((self.max_topk, (self.rank_wave(), 1)), stride=(1, (self.max_topk, 0))),
        )

    @cute.jit
    def _rank_sort_query(
        self, mTopkIdxs: cute.Tensor, mTopkLength: Optional[cute.Tensor], seqlen_kv: Int32, query: Int32, destination: cute.Pointer, tidx: Int32
    ):
        """Write one query's active rows ascending, -1 after (one CTA)."""
        cap = self.rank_sort_cap()
        smem = SmemAllocator()
        keys = smem.allocate_tensor(Int32, cute.make_layout((cap,)), byte_alignment=16)
        unsorted = smem.allocate_tensor(Int32, cute.make_layout((RANK_M64_THREADS // 32,)), byte_alignment=16)
        topk = _rank_topk(mTopkIdxs, mTopkLength, query)
        for position in cutlass.range(tidx, cap, RANK_M64_THREADS, unroll=1):
            key = Int32(RANK_M64_SORT_SENTINEL)
            if position < topk:
                row = mTopkIdxs[position, (query, 0)]
                if row >= Int32(0) and row < seqlen_kv:
                    key = row
            keys[position] = key
        cute.arch.barrier()
        inversion = cutlass.Boolean(False)
        for position in cutlass.range(tidx + Int32(1), cap, RANK_M64_THREADS, unroll=1):
            if keys[position - Int32(1)] > keys[position]:
                inversion = cutlass.Boolean(True)
        warp_inversions = Uint32(cute.arch.vote_ballot_sync(inversion))
        if tidx % Int32(32) == Int32(0):
            unsorted[tidx // Int32(32)] = Int32(0)
            if warp_inversions != Uint32(0):
                unsorted[tidx // Int32(32)] = Int32(1)
        cute.arch.barrier()
        any_unsorted = Int32(0)
        for warp in cutlass.range_constexpr(RANK_M64_THREADS // 32):
            any_unsorted = any_unsorted | unsorted[warp]
        if any_unsorted != Int32(0):
            self._rank_bitonic_sort(keys, tidx)
        for position in cutlass.range(tidx, self.max_topk, RANK_M64_THREADS, unroll=1):
            key = keys[position]
            if key == Int32(RANK_M64_SORT_SENTINEL):
                key = Int32(-1)
            cute.arch.store(destination + Int64(position), key)

    @cute.jit
    def _rank_bitonic_sort(self, keys: cute.Tensor, tidx: Int32):
        """Bitonic sort of ``rank_sort_cap()`` shared-memory keys (one CTA)."""
        cap = self.rank_sort_cap()
        per = cap // RANK_M64_THREADS
        stages = cap.bit_length()
        if cutlass.const_expr(per < 2):
            for size_log in cutlass.range_constexpr(1, stages):
                for stride_log in cutlass.range_constexpr(size_log - 1, -1, -1):
                    stride = 1 << stride_log
                    for pair in cutlass.range(tidx, cap // 2, RANK_M64_THREADS, unroll=1):
                        low = (pair >> Int32(stride_log)) * Int32(2 * stride) + (pair & Int32(stride - 1))
                        high = low + Int32(stride)
                        a = keys[low]
                        b = keys[high]
                        if (low & Int32(1 << size_log)) == Int32(0):
                            if a > b:
                                keys[low] = b
                                keys[high] = a
                        else:
                            if a < b:
                                keys[low] = b
                                keys[high] = a
                    cute.arch.barrier()
        else:
            vals = cute.make_rmem_tensor((per,), Int32)
            base = tidx * Int32(per)
            for e in cutlass.range_constexpr(per):
                vals[e] = keys[base + Int32(e)]
            for size_log in cutlass.range_constexpr(1, stages):
                size = 1 << size_log
                for stride_log in cutlass.range_constexpr(size_log - 1, -1, -1):
                    stride = 1 << stride_log
                    if cutlass.const_expr(stride < per):
                        for e in cutlass.range_constexpr(per):
                            if cutlass.const_expr((e & stride) == 0):
                                low = min(vals[e], vals[e + stride])
                                high = max(vals[e], vals[e + stride])
                                descending = (((base + Int32(e)) // Int32(size)) & Int32(1)) != Int32(0)
                                vals[e] = high if descending else low
                                vals[e + stride] = low if descending else high
                    else:
                        take_low = (((tidx // Int32(stride // per)) ^ (tidx // Int32(size // per))) & Int32(1)) == Int32(0)
                        if cutlass.const_expr(stride < 32 * per):
                            for e in cutlass.range_constexpr(per):
                                other = cute.arch.shuffle_sync_bfly(vals[e], offset=stride // per)
                                vals[e] = min(vals[e], other) if take_low else max(vals[e], other)
                        else:
                            cute.arch.barrier()
                            for e in cutlass.range_constexpr(per):
                                keys[base + Int32(e)] = vals[e]
                            cute.arch.barrier()
                            for e in cutlass.range_constexpr(per):
                                other = keys[(base ^ Int32(stride)) + Int32(e)]
                                vals[e] = min(vals[e], other) if take_low else max(vals[e], other)
            cute.arch.barrier()
            for e in cutlass.range_constexpr(per):
                keys[base + Int32(e)] = vals[e]
            cute.arch.barrier()

    @cute.jit
    def rank_claim_ordinal(self, mdKV_acc: cute.Tensor, geometry, own: cute.Tensor, tidx: Int32) -> Int32:
        """Return this CTA's ticket, claimed in start order (one atomic, shared-memory broadcast)."""
        pitch, planes = geometry
        _, _, _, tickets, _ = self.rank_chain_offsets(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Int32)
        if tidx == Int32(0):
            own[0] = cute.arch.atomic_add((words + tickets).llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
        cute.arch.barrier()
        return own[0]

    @cute.jit
    def rank_chain_relay_tables(self, mdKV_acc: cute.Tensor, geometry, ordinal: Int32):
        """Return a position's chain words, tile flags, semaphores and tag."""
        pitch, planes = geometry
        _, chain, semaphores, tickets, _ = self.rank_chain_offsets(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Int32)
        relay_row = words + chain + Int64(ordinal) * Int64(self.max_topk)
        flag_row = words + self.rank_tile_flags(pitch, planes) + Int64(ordinal) * Int64(self.rank_tiles())
        tag = cute.arch.load(words + tickets + Int64(4), Int32) * Int32(256)
        return relay_row, flag_row, words + semaphores, tag

    def __init__(self, *args, **kwargs):
        """Create the hand-off barriers (ids 11-15, above the main kernel's 1-10) after the kernel's own setup."""
        super().__init__(*args, **kwargs)
        self.rank_acquire_free_barrier = pipeline.NamedBarrier(
            barrier_id=11,
            num_threads=(1 + self.num_reduce_warps) * self.threads_per_warp,
        )
        self.rank_acquire_full_barrier = pipeline.NamedBarrier(
            barrier_id=15,
            num_threads=(1 + self.num_reduce_warps) * self.threads_per_warp,
        )
        self.rank_order_barrier = pipeline.NamedBarrier(
            barrier_id=12,
            num_threads=(2 + self.num_load_KV_warps + self.num_reduce_warps) * self.threads_per_warp,
        )
        self.rank_release_done_barrier = pipeline.NamedBarrier(
            barrier_id=13,
            num_threads=(1 + self.num_reduce_warps) * self.threads_per_warp,
        )
        self.rank_release_ready_barrier = pipeline.NamedBarrier(
            barrier_id=14,
            num_threads=(1 + self.num_reduce_warps) * self.threads_per_warp,
        )

    @cute.jit
    def rank_chain_tile_acquire(self, relay: Int32):
        """Chain plans: take the acquire warp's hand-off before a waiting tile."""
        if (relay & Int32(RANK_M64_TILE_WAITS << RANK_M64_RELAY_FLAG_SHIFT)) != Int32(0):
            _partial_cta_sync(self.rank_acquire_full_barrier.barrier_id, self.rank_acquire_full_barrier.num_threads)
            self.rank_acquire_free_barrier.arrive_unaligned()

    @cute.jit
    def rank_chain_acquire_warp(
        self, mdKV_acc: cute.Tensor, geometry, ordinal: Int32, head_block_idx: Int32, head_blocks: Int32, tile_count: Int32, order: cute.Tensor
    ):
        """Chain plans, idle warp: one waiting tile ahead of the reducers, wait for every waiting row's chain predecessor."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % Int32(32)
        relay_row, flag_row, semaphores, tag = self.rank_chain_relay_tables(mdKV_acc, geometry, ordinal)
        handed = Int32(0)
        step = Int32(0)
        while step < tile_count:
            tile = order[step]
            flags = self.rank_chain_head_flags(cute.arch.load(flag_row + Int64(tile), Int32), head_block_idx, head_blocks)
            if (flags & Int32(RANK_M64_TILE_WAITS)) != Int32(0):
                word_0 = Int32(0)
                word_1 = Int32(0)
                slot = tile * Int32(self.block_tile) + lane
                if slot < Int32(self.max_topk):
                    word_0 = cute.arch.load(relay_row + Int64(slot), Int32)
                if slot + Int32(32) < Int32(self.max_topk):
                    word_1 = cute.arch.load(relay_row + Int64(slot + Int32(32)), Int32)
                if handed != Int32(0):
                    _partial_cta_sync(self.rank_acquire_free_barrier.barrier_id, self.rank_acquire_free_barrier.num_threads)
                self.rank_chain_wait(semaphores, tag, word_0, head_block_idx, head_blocks)
                self.rank_chain_wait(semaphores, tag, word_1, head_block_idx, head_blocks)
                self.rank_acquire_full_barrier.arrive_unaligned()
                handed = Int32(1)
            step += Int32(1)
        if handed != Int32(0):
            _partial_cta_sync(self.rank_acquire_free_barrier.barrier_id, self.rank_acquire_free_barrier.num_threads)

    @cute.jit
    def rank_chain_tile_release(self, relay: Int32):
        """Chain plans: hand a releasing tile to the release warp after this thread's updates."""
        if (relay & Int32(RANK_M64_TILE_RELEASES << RANK_M64_RELAY_FLAG_SHIFT)) != Int32(0):
            _partial_cta_sync(self.rank_release_ready_barrier.barrier_id, self.rank_release_ready_barrier.num_threads)
            self.rank_release_done_barrier.arrive_unaligned()

    @cute.jit
    def rank_chain_release_warp(
        self, mdKV_acc: cute.Tensor, geometry, ordinal: Int32, head_block_idx: Int32, head_blocks: Int32, tile_count: Int32, order: cute.Tensor
    ):
        """Chain plans, load warp: after each releasing tile's updates, release its chain successors."""
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % Int32(32)
        cute.arch.griddepcontrol_wait()
        relay_row, flag_row, semaphores, tag = self.rank_chain_relay_tables(mdKV_acc, geometry, ordinal)
        _partial_cta_sync(self.rank_order_barrier.barrier_id, self.rank_order_barrier.num_threads)
        step = Int32(0)
        while step < tile_count:
            tile = order[step]
            flags = self.rank_chain_head_flags(cute.arch.load(flag_row + Int64(tile), Int32), head_block_idx, head_blocks)
            if (flags & Int32(RANK_M64_TILE_RELEASES)) != Int32(0):
                word_0 = Int32(0)
                word_1 = Int32(0)
                slot = tile * Int32(self.block_tile) + lane
                if slot < Int32(self.max_topk):
                    word_0 = cute.arch.load(relay_row + Int64(slot), Int32)
                if slot + Int32(32) < Int32(self.max_topk):
                    word_1 = cute.arch.load(relay_row + Int64(slot + Int32(32)), Int32)
                self.rank_release_ready_barrier.arrive_unaligned()
                _partial_cta_sync(self.rank_release_done_barrier.barrier_id, self.rank_release_done_barrier.num_threads)
                self.rank_chain_signal(semaphores, tag, word_0, head_block_idx, head_blocks)
                self.rank_chain_signal(semaphores, tag, word_1, head_block_idx, head_blocks)
            step += Int32(1)

    @cute.jit
    def rank_chain_head_flags(self, flags: Int32, head_block: Int32, head_blocks: Int32) -> Int32:
        """Return a tile's relay flags for one head block."""
        members = Int32(0)
        if (flags & Int32(RANK_M64_TILE_MEMBERS)) != Int32(0):
            members = Int32(1)
        result = flags & Int32(RANK_M64_TILE_WAITS | RANK_M64_TILE_RELEASES)
        if head_block > Int32(0):
            result = (result & Int32(RANK_M64_TILE_RELEASES)) | (members * Int32(RANK_M64_TILE_WAITS))
        if head_block < head_blocks - Int32(1):
            result = (result & Int32(RANK_M64_TILE_WAITS)) | (members * Int32(RANK_M64_TILE_RELEASES))
        return result

    @cute.jit
    def rank_chain_spread(self, count: Int32, active: Int32, planes: Int32, cold: Int32) -> Int32:
        """Return a row's chain planes this wave."""
        spread = planes
        if cold != Int32(0) and count < active and count >= planes:
            spread = min(planes, Int32(RANK_M64_CHAIN_SPREAD))
        return spread

    @cute.jit
    def rank_chain_flag_word(self, tickets: cute.Pointer, wave_index: Int32) -> cute.Pointer:
        """Return the word flagging that every position of a wave wrote a row."""
        n = wave_index % Int32(3)
        return tickets + Int64(Int32(2) + n + Int32(3) * (n // Int32(2)))

    @cute.jit
    def rank_chain_cold(self, tickets: cute.Pointer, wave_index: Int32, pitch: Int64, planes: Int32, num_heads: Int32) -> Int32:
        """Return 1 when wave ``wave_index`` may share planes."""
        cold = Int32(0)
        if num_heads <= Int32(self.block_tile) and Int64(self.rank_wave() * self.max_topk) >= Int64(planes) * pitch:
            if cute.arch.load(self.rank_chain_flag_word(tickets, wave_index + Int32(2)), Int32) == Int32(0):
                cold = Int32(1)
        return cold

    @cute.jit
    def rank_chain_wait(self, semaphores: cute.Pointer, tag: Int32, relay: Int32, head_block: Int32, head_blocks: Int32):
        """Slot lane: wait for the row's chain predecessor (positions >= 1)."""
        if (relay & Int32(RANK_M64_CHAIN_MEMBER)) != Int32(0):
            position = (relay & Int32(RANK_M64_CHAIN_POSITION_MASK)) * head_blocks + head_block
            if position > Int32(0):
                plane_row = (relay >> Int32(RANK_M64_CHAIN_ROW_SHIFT)) & Int32(RANK_M64_CHAIN_MAX_PLANE_ROWS - 1)
                _rank_wait_value(semaphores + Int64(plane_row), tag + position)

    @cute.jit
    def rank_chain_signal(self, semaphores: cute.Pointer, tag: Int32, relay: Int32, head_block: Int32, head_blocks: Int32):
        """Slot lane: let the row's chain successor add (after the reducers' barrier)."""
        if (relay & Int32(RANK_M64_CHAIN_MEMBER)) != Int32(0):
            successor = relay & Int32(RANK_M64_CHAIN_SUCCESSOR)
            if head_block < head_blocks - Int32(1):
                successor = Int32(1)
            if successor != Int32(0):
                position = (relay & Int32(RANK_M64_CHAIN_POSITION_MASK)) * head_blocks + head_block
                plane_row = (relay >> Int32(RANK_M64_CHAIN_ROW_SHIFT)) & Int32(RANK_M64_CHAIN_MAX_PLANE_ROWS - 1)
                cute.arch.store(semaphores + Int64(plane_row), tag + position + Int32(1), sem="release", scope="gpu")

    def rank_sort_cap(self) -> int:
        """Return the sort width (top-k width rounded up to a power of two)."""
        return 1 << (int(self.max_topk) - 1).bit_length()

    def rank_setup_ctas(self) -> int:
        """Return a wave setup's CTAs."""
        return RANK_M64_WAVE_SETUP_CTAS

    def rank_setup_bounds(self) -> dict:
        """Return the wave setups' occupancy bound (eight resident CTAs per SM)."""
        return {"min_blocks_per_mp": 8}

    def rank_setup_smem_bytes(self) -> int:
        """Return ``rank_setup_a``'s dynamic shared memory (the top-k sort)."""
        return 4 * self.rank_sort_cap() + 128

    @cute.jit
    def rank_parity(self, q_offset: Int32) -> Int32:
        """Return the state parity of the wave that starts at ``q_offset``."""
        return (q_offset // Int32(self.rank_wave())) % Int32(2)

    @cute.jit
    def rank_query_codes(self, mdKV_acc: cute.Tensor, geometry, token_idx: Int32):
        """Return the destination-code row of wave position ``token_idx``."""
        pitch, planes = geometry
        _, codes = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        ordinal = token_idx % Int32(self.rank_wave())
        return cute.recast_ptr(words + codes + Int64(ordinal) * Int64(self.max_topk), dtype=Int32)

    @cute.jit
    def _rank_mark_slot(self, words: cute.Pointer, state: Int64, pitch: Int64, parity: Int32, ordinal: Int32, row: Int32):
        """Set ``row``'s ordinal bit in the row state of parity ``parity``."""
        entry = words + state + (Int64(parity) * pitch + Int64(row)) * Int64(self.rank_state_words())
        bit = Uint32(1) << Uint32(ordinal % Int32(32))
        cute.arch.atomic_or((entry + Int64(ordinal // Int32(32))).llvm_ptr, bit, sem="relaxed", scope="gpu")

    @cute.jit
    def _rank_sum_OdO(self, odo, cur_begin: Int32, total_q: Int32, thread: Int32):
        """Chain plans: one wave's O.dO sums and sink-scaled LSE."""
        mO, mdO, mSum_OdO, mLSE, mAttnSink, mScaledLSE = odo
        heads = mLSE.shape[0]
        lane = thread % Int32(8)
        group = thread // Int32(8)
        position = group // heads
        head = group - position * heads
        query = self.rank_wave_query(cur_begin, min(position, Int32(self.rank_wave() - 1)), total_q)
        valid = Int32(0)
        if position < Int32(self.rank_wave()):
            if query < total_q:
                valid = Int32(1)
        acc = Float32(0.0)
        if valid != Int32(0):
            O_bhq = cute.logical_divide(mO[head, None, (query, 0)], cute.make_layout(self.sum_OdO_elem_per_load))
            dO_bhq = cute.logical_divide(mdO[head, None, (query, 0)], cute.make_layout(self.sum_OdO_elem_per_load))
            # Chunks lane, lane + 8, ... summed in order; four chunks' loads issue before their sums.
            for g in cutlass.range_constexpr(self.head_dim_v // self.sum_OdO_elem_per_load // 32):
                O_frags = [O_bhq[None, lane + Int32(8 * (4 * g + j))].load() for j in range(4)]
                dO_frags = [dO_bhq[None, lane + Int32(8 * (4 * g + j))].load() for j in range(4)]
                for j in cutlass.range_constexpr(4):
                    prod_frag = O_frags[j].to(self.acc_dtype) * dO_frags[j].to(self.acc_dtype)
                    acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)
        acc = cute.arch.warp_reduction_sum(acc, threads_in_group=8)
        if valid != Int32(0) and lane == Int32(0):
            lse_bhq = mLSE[head, (query, 0)]
            attn_sink_bh = mAttnSink[head, (0, 0)]
            log2_e = -Float32(-math.log2(math.e))
            lse_log2 = lse_bhq * log2_e
            sink_log2 = attn_sink_bh * log2_e
            lse_max_log2 = cute.arch.fmax(lse_log2, sink_log2)
            sum_exp2 = Float32(cute.math.exp2(lse_log2 - lse_max_log2) + cute.math.exp2(sink_log2 - lse_max_log2))
            scaled_lse_bhq = -(lse_max_log2 + cute.math.log2(sum_exp2))
            if (
                (lse_bhq == Float32(float("-inf")) and attn_sink_bh == Float32(float("-inf")))
                or lse_bhq == Float32(float("inf"))
                or attn_sink_bh == Float32(float("inf"))
            ):
                scaled_lse_bhq = Float32(float("-inf"))
            mSum_OdO[head, (query, 0)] = Float32(-1.0) * acc
            mScaledLSE[head, (query, 0)] = scaled_lse_bhq

    @cute.jit
    def _rank_lower_bound(self, wave_rows: cute.Tensor, row: Int32) -> Int32:
        """Return the first slot of position 0's sorted rows that holds ``row``."""
        low = Int32(0)
        top = Int32(min(int(self.max_topk), max(self.rank_hot_rows(), 1)))
        while low < top:
            middle = (low + top) // Int32(2)
            value = wave_rows[middle, (0, 0)]
            if value >= Int32(0) and value < row:
                low = middle + Int32(1)
            else:
                top = middle
        return low

    @cute.jit
    def _rank_fold_hot(self, floats: cute.Pointer, hot_list: cute.Pointer, hot_base: Int32, thread: Int32, nthreads: Int32):
        """Add each hot row's planes in ascending plane order into its plane-0 row."""
        item = thread
        while item < Int32(self.rank_hot_rows() * self.head_dim):
            hot_slot = item // Int32(self.head_dim)
            column = item - hot_slot * Int32(self.head_dim)
            word = cute.arch.load(hot_list + Int64(hot_slot), Int32)
            if word != Int32(0):
                acc = cute.make_rmem_tensor((1,), Float32)
                acc[0] = Float32(0.0)
                source = floats + (Int64(hot_base + hot_slot * Int32(RANK_M64_HOT_PLANES)) * Int64(self.head_dim) + Int64(column))
                for plane in cutlass.range((word >> Int32(24)) & Int32(63), unroll=32):
                    acc[0] = acc[0] + cute.arch.load(source + Int64(plane) * Int64(self.head_dim), Float32)
                target = floats + (Int64(word & Int32(0xFFFFFF)) * Int64(self.head_dim) + Int64(column))
                if (word & Int32(RANK_M64_HOT_INIT)) != Int32(0):
                    acc[0] = cute.arch.load(target, Float32) + acc[0]
                cute.arch.store(target, acc[0])
            item += nthreads

    @cute.kernel
    def rank_init(self, mdKV_acc: cute.Tensor, seqlen_kv: Int32, rank_pitch: Int64, rank_planes: Int32):
        """Clear both row-state parities, the initialized-plane counts, the semaphores, the tickets and the hot-row words.

        The planes, codes, chain words, sorted rows and tile flags are written before they are read."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = rank_pitch, rank_planes
        state, _ = self.rank_offsets(pitch, planes)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        stride = Int64(nblocks) * Int64(RANK_M64_THREADS)
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        state_chunks = Int64(2 * self.rank_state_words() // 4) * pitch
        while chunk < state_chunks:
            _rank_store_zero_x4(words + state + chunk * Int64(4))
            chunk += stride
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        while chunk < pitch // Int64(4):
            _rank_store_zero_x4(words + high + chunk * Int64(4))
            chunk += stride
        _, _, semaphores, tickets, _ = self.rank_chain_offsets(pitch, planes)
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        span = (Int64(planes) * pitch + Int64(self.rank_hot_slots() + RANK_M64_TICKET_WORDS) + Int64(3)) // Int64(4)
        while chunk < span:
            _rank_store_zero_x4(words + semaphores + chunk * Int64(4))
            chunk += stride
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        if chunk < Int64(2 * self.rank_hot_rows()):
            cute.arch.store(words + self.rank_hot_list(pitch, planes) + chunk, Uint32(0))

    @cute.kernel
    def rank_setup_a(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        cur_parity: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        total_q: Int32,
        num_heads: Int32,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mSum_OdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mScaledLSE: cute.Tensor,
        rank_pitch: Int64,
        rank_planes: Int32,
    ):
        """Sort the current wave's rows, finish the previous wave and record the current wave's row ordinals.

        CTA ``p < RANK_M64_CHAIN_WAVE`` sorts wave position ``p``'s active rows; the other CTAs share one item list
        (this wave's O.dO sums, the previous wave's finish slots, this wave's row marks). ``cur_size == 0`` (the
        final setup) only finishes the previous wave; ``prev_size == 0`` (the first wave) has nothing to finish."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = rank_pitch, rank_planes
        state, codes = self.rank_offsets(pitch, planes)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        topk_width = Int32(self.max_topk)
        slots = Int64(self.rank_wave() * self.max_topk)
        prev_parity = Int32(1) - cur_parity
        # Item threads are numbered from the first CTA after the sort CTAs.
        thread = (bidx - Int32(self.rank_wave())) * Int32(RANK_M64_THREADS) + tidx
        nthreads = (nblocks - Int32(self.rank_wave())) * Int32(RANK_M64_THREADS)
        if bidx < Int32(self.rank_wave()):
            thread = Int32(0x7FFFFFFF)
        cute.arch.griddepcontrol_launch_dependents()
        # The first wave's setup is a programmatic dependent of ``rank_init``, which never waits or triggers: it
        # can start once those CTAs exit, before their stores are guaranteed visible. Every CTA of the first wave
        # therefore waits before touching the workspace (its row marks update the state words ``rank_init``
        # clears). A later wave follows a main kernel whose CTAs all waited before they triggered or exited.
        if prev_size == Int32(0):
            cute.arch.griddepcontrol_wait()
        prev_wave = prev_begin
        if cur_size > Int32(0):
            if bidx < Int32(self.rank_wave()):
                query = self.rank_wave_query(cur_begin, bidx, total_q)
                if query < total_q:
                    sorted_rows, _, _, _, _ = self.rank_chain_offsets(pitch, planes)
                    destination = cute.recast_ptr(words + sorted_rows + Int64(cur_parity) * slots + Int64(bidx) * Int64(self.max_topk), dtype=Int32)
                    self._rank_sort_query(mTopkIdxs, mTopkLength, seqlen_kv, query, destination, tidx)
        prev_rows = self.rank_sorted_view(mdKV_acc, (pitch, planes), prev_parity)
        prev_size = min(prev_size, Int32(1)) * Int32(self.rank_wave())
        chain_tickets = cute.recast_ptr(words + self.rank_chain_offsets(pitch, planes)[3], dtype=Int32)
        prev_active = self.rank_wave_active(prev_wave, total_q)
        prev_cold = self.rank_chain_cold(chain_tickets, prev_wave // Int32(self.rank_wave()), pitch, planes, num_heads)
        # One flat item list over the non-sort threads: this wave's O.dO groups, the previous wave's finish slots,
        # then this wave's ordinal marks (each item unchanged), so no thread runs all three phases.
        odo_items = Int32(0)
        if cur_size > Int32(0):
            odo_items = Int32(self.rank_wave() * 8) * mLSE.shape[0]
        fin_items = Int32(0)
        if prev_size > Int32(0):
            fin_items = prev_size * topk_width
        mark_items = Int32(0)
        if cur_size > Int32(0):
            mark_items = self.rank_wave_size(cur_begin, total_q) * topk_width
        item = thread
        while item < odo_items + fin_items + mark_items:
            if item < odo_items:
                self._rank_sum_OdO((mO, mdO, mSum_OdO, mLSE, mAttnSink, mScaledLSE), cur_begin, total_q, item)
            elif item < odo_items + fin_items:
                slot_index = item - odo_items
                ordinal = slot_index // topk_width
                slot = slot_index - ordinal * topk_width
                query = self.rank_wave_query(prev_wave, ordinal, total_q)
                row = Int32(-1)
                slot_code = Int32(0)
                slot_chain = Int32(0)
                if query < total_q:
                    # Sorted rows hold -1 past the active extent.
                    _, chain_words, _, _, _ = self.rank_chain_offsets(pitch, planes)
                    row = prev_rows[slot, (ordinal, 0)]
                    slot_code = cute.arch.load(cute.recast_ptr(words + codes + Int64(slot_index), dtype=Int32), Int32)
                    slot_chain = cute.arch.load(cute.recast_ptr(words + chain_words + Int64(slot_index), dtype=Int32), Int32)
                if row >= Int32(0) and row < seqlen_kv:
                    code = slot_code
                    rank_zero = Int32(1)
                    if (slot_chain & Int32(RANK_M64_CHAIN_POSITION_MASK)) != Int32(0):
                        rank_zero = Int32(0)
                    plane_row = code & Int32(RANK_M64_CHAIN_ROW_MASK)
                    hot_entry = plane_row - planes * Int32(pitch)
                    if hot_entry % Int32(RANK_M64_HOT_PLANES) != Int32(0):
                        hot_entry = Int32(-1)
                    if code >= Int32(0) and (plane_row < Int32(pitch) or hot_entry >= Int32(0)) and rank_zero != Int32(0):
                        entry = words + state + (Int64(prev_parity) * pitch + Int64(row)) * Int64(self.rank_state_words())
                        count, _ = self._rank_state_count(entry, None)
                        high_ptr = cute.recast_ptr(words + high + Int64(row), dtype=Int32)
                        written = min(count, self.rank_chain_spread(count, prev_active, planes, prev_cold))
                        if count >= prev_active:
                            cute.arch.atomic_or(
                                self.rank_chain_flag_word(chain_tickets, prev_wave // Int32(self.rank_wave())).llvm_ptr,
                                Int32(1),
                                sem="relaxed",
                                scope="gpu",
                            )
                        if hot_entry >= Int32(0):
                            # The hot fold adds into plane 0 when it already holds data, else stores.
                            written = Int32(1)
                            if cute.arch.load(high_ptr, Int32) > Int32(0):
                                hot_word = cute.recast_ptr(
                                    words
                                    + self.rank_hot_list(pitch, planes)
                                    + Int64(prev_parity * Int32(self.rank_hot_rows()) + hot_entry // Int32(RANK_M64_HOT_PLANES)),
                                    dtype=Int32,
                                )
                                cute.arch.store(hot_word, cute.arch.load(hot_word, Int32) | Int32(RANK_M64_HOT_INIT))
                        if written > cute.arch.load(high_ptr, Int32):
                            cute.arch.store(high_ptr, written)
                        for chunk in cutlass.range_constexpr(self.rank_state_words() // 4):
                            _rank_store_zero_x4(entry + Int64(4 * chunk))
            else:
                slot_index = item - odo_items - fin_items
                ordinal = slot_index // topk_width
                slot = slot_index - ordinal * topk_width
                query = self.rank_wave_query(cur_begin, ordinal, total_q)
                if query < total_q:
                    if slot < _rank_topk(mTopkIdxs, mTopkLength, query):
                        row = mTopkIdxs[slot, (query, 0)]
                        if row >= Int32(0) and row < seqlen_kv:
                            self._rank_mark_slot(words, state, pitch, cur_parity, ordinal, row)
            item += nthreads
        # Only the first RANK_M64_SETUP_WAITERS non-sort CTAs wait for the preceding grid (the previous wave's main
        # kernel; rank_init for the first wave) and then clear the tables that main kernel reads; their exit after
        # that wait makes this grid's completion (which rank_setup_b and then the next main kernel wait for) imply
        # the preceding grid's. rank_scan_first and rank_setup_b wait before any access, so every other CTA exits
        # after its pre-wait items.
        if thread >= Int32(RANK_M64_SETUP_WAITERS * RANK_M64_THREADS):
            cute.arch.nvvm.exit()
        nthreads = Int32(RANK_M64_SETUP_WAITERS * RANK_M64_THREADS)
        cute.arch.griddepcontrol_wait()
        if thread == Int32(0):
            _, _, _, tickets, _ = self.rank_chain_offsets(pitch, planes)
            ticket_words = cute.recast_ptr(words + tickets, dtype=Int32)
            cute.arch.store(ticket_words, Int32(0))
            cute.arch.store(ticket_words + 1, Int32(0))
            cute.arch.store(ticket_words + 4, (cur_begin // Int32(self.rank_wave())) * Int32(2) + Int32(1))
            cute.arch.store(ticket_words + RANK_M64_ACTIVE_WORD, self.rank_wave_active(cur_begin, total_q))
            cute.arch.store(ticket_words + RANK_M64_DEPTH_WORD, Int32(0))
            cute.arch.store(self.rank_chain_flag_word(ticket_words, cur_begin // Int32(self.rank_wave())), Int32(0))
        flags_base = words + self.rank_tile_flags(pitch, planes)
        depths_base = words + self.rank_hot_depths(pitch, planes)
        flag = thread
        while flag < Int32(self.rank_wave() * self.rank_tiles()):
            cute.arch.store(cute.recast_ptr(flags_base + Int64(flag), dtype=Int32), Int32(0))
            cute.arch.store(cute.recast_ptr(depths_base + Int64(flag), dtype=Int32), Int32(0))
            flag += nthreads

    @cute.kernel
    def rank_scan_first(self, mdKV_acc: cute.Tensor, seqlen_kv: Int32, total_q: Int32, rank_pitch: Int64, rank_planes: Int32):
        """First wave: flag whether some row is written by every active position (``rank_chain_cold`` reads it)."""
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = rank_pitch, rank_planes
        state, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        tickets = cute.recast_ptr(words + self.rank_chain_offsets(pitch, planes)[3], dtype=Int32)
        active = self.rank_wave_active(Int32(0), total_q)
        row = bidx * Int32(RANK_M64_THREADS) + tidx
        while row < seqlen_kv:
            count, _ = self._rank_state_count(words + state + Int64(row) * Int64(self.rank_state_words()), None)
            if count >= active:
                cute.arch.atomic_or(self.rank_chain_flag_word(tickets, Int32(2)).llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
            row += nblocks * Int32(RANK_M64_THREADS)

    @cute.kernel
    def rank_setup_b(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        cur_parity: Int32,
        total_q: Int32,
        num_heads: Int32,
        rank_pitch: Int64,
        rank_planes: Int32,
    ):
        """Fold the previous wave's hot rows into plane 0, then write the current wave's destination codes, chain words
        and tile flags. ``cur_size == 0`` (the final setup) only folds the hot rows."""
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = rank_pitch, rank_planes
        state, codes = self.rank_offsets(pitch, planes)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        topk_width = Int32(self.max_topk)
        thread = bidx * Int32(RANK_M64_THREADS) + tidx
        nthreads = nblocks * Int32(RANK_M64_THREADS)
        lane = tidx % Int32(32)
        total_slots = self.rank_wave_size(cur_begin, total_q) * topk_width
        wave_rows = self.rank_sorted_view(mdKV_acc, (pitch, planes), cur_parity)
        _, chain_words, _, _, _ = self.rank_chain_offsets(pitch, planes)
        flags_base = words + self.rank_tile_flags(pitch, planes)
        depths_base = words + self.rank_hot_depths(pitch, planes)
        _, _, _, tickets, _ = self.rank_chain_offsets(pitch, planes)
        active = cute.arch.load(cute.recast_ptr(words + tickets + Int64(RANK_M64_ACTIVE_WORD), dtype=Int32), Int32)
        cold = self.rank_chain_cold(cute.recast_ptr(words + tickets, dtype=Int32), cur_begin // Int32(self.rank_wave()), pitch, planes, num_heads)
        hot_rows = Int32(self.rank_hot_rows())
        hot_base = planes * Int32(pitch)
        hot_list = cute.recast_ptr(words + self.rank_hot_list(pitch, planes), dtype=Int32)
        # Last threads fold (fewest slots below 2 per thread); the fold reads only the other parity's list.
        self._rank_fold_hot(mdKV_acc.iterator, hot_list + Int64((Int32(1) - cur_parity) * hot_rows), hot_base, nthreads - Int32(1) - thread, nthreads)
        if cur_size == Int32(0):
            total_slots = Int32(0)
        base = thread - lane
        while base < total_slots:
            slot_index = base + lane
            ordinal = slot_index // topk_width
            slot = slot_index - ordinal * topk_width
            query = self.rank_wave_query(cur_begin, ordinal, total_q)
            candidate = Int32(-1)
            previous = Int32(-1)
            following = Int32(-1)
            if slot_index < total_slots and query < total_q:
                # Sorted rows hold -1 past the active extent.
                candidate = wave_rows[slot, (ordinal, 0)]
                previous = wave_rows[max(slot - Int32(1), Int32(0)), (ordinal, 0)]
                following = wave_rows[min(slot + Int32(1), topk_width - Int32(1)), (ordinal, 0)]
            chain_word = Int32(0)
            lane_flags = Int32(0)
            hot_word = Int32(0)
            if candidate >= Int32(0) and candidate < seqlen_kv:
                code = Int32(RANK_M64_SKIP_CODE)
                follower = Int32(0)
                if slot > Int32(0):
                    if previous == candidate:
                        follower = Int32(1)
                if follower == Int32(0):
                    probe = slot + Int32(1)
                    same = Int32(0)
                    if probe < topk_width:
                        if following == candidate:
                            same = Int32(1)
                            probe += Int32(1)
                    while same != Int32(0):
                        same = Int32(0)
                        if probe < topk_width:
                            if wave_rows[probe, (ordinal, 0)] == candidate:
                                same = Int32(1)
                                probe += Int32(1)
                    entry = words + state + (Int64(cur_parity) * pitch + Int64(candidate)) * Int64(self.rank_state_words())
                    initialized = cute.arch.load(cute.recast_ptr(words + high + Int64(candidate), dtype=Int32), Int32)
                    count, conflict = self._rank_state_count(entry, ordinal)
                    stride = self.rank_chain_spread(count, active, planes, cold)
                    # One unsigned division gives the chain position and the plane (conflict >= 0, stride >= 1).
                    position = Int32(Uint32(conflict) // Uint32(stride))
                    code = (conflict - position * stride) * Int32(pitch) + candidate
                    fresh = Int32(0)
                    if position == Int32(0) and conflict >= initialized:
                        fresh = Int32(1)
                    if count >= active and count > planes and planes < Int32(RANK_M64_HOT_PLANES):
                        # A row every query of the wave writes relays along the hot planes of its slot in position 0's rows.
                        hot_slot = self._rank_lower_bound(wave_rows, candidate)
                        if hot_slot < hot_rows:
                            stride = Int32(RANK_M64_HOT_PLANES)
                            position = conflict >> Int32(RANK_M64_HOT_SHIFT)
                            code = hot_base + hot_slot * stride + (conflict & Int32(RANK_M64_HOT_PLANES - 1))
                            fresh = Int32(0)
                            if position == Int32(0):
                                fresh = Int32(1)
                            if conflict == Int32(0):
                                hot_word = (min(count, stride) << Int32(24)) | candidate
                    chain_word = position | Int32(RANK_M64_CHAIN_MEMBER) | (code << Int32(RANK_M64_CHAIN_ROW_SHIFT))
                    if fresh != Int32(0):
                        code = code | Int32(RANK_M64_FRESH_FLAG)
                    code = code | ((probe - slot - Int32(1)) << Int32(RANK_M64_MULT_SHIFT))
                    if conflict + stride < count:
                        chain_word = chain_word | Int32(RANK_M64_CHAIN_SUCCESSOR)
                    flags = Int32(RANK_M64_TILE_MEMBERS)
                    if position > Int32(0):
                        flags = flags | Int32(RANK_M64_TILE_WAITS)
                    if conflict + stride < count:
                        flags = flags | Int32(RANK_M64_TILE_RELEASES)
                    lane_flags = flags
                    if count >= active and count > planes:
                        depth_word = depths_base + Int64(ordinal * Int32(self.rank_tiles()) + slot // Int32(self.block_tile))
                        cute.arch.atomic_max(cute.recast_ptr(depth_word, dtype=Int32).llvm_ptr, position + Int32(1), sem="relaxed", scope="gpu")
                        if conflict + Int32(1) == count:
                            cute.arch.atomic_max(
                                cute.recast_ptr(words + tickets + Int64(RANK_M64_DEPTH_WORD), dtype=Int32).llvm_ptr,
                                position + Int32(1),
                                sem="relaxed",
                                scope="gpu",
                            )
                cute.arch.store(cute.recast_ptr(words + codes + Int64(slot_index), dtype=Int32), code)
            if slot_index < total_slots:
                cute.arch.store(cute.recast_ptr(words + chain_words + Int64(slot_index), dtype=Int32), chain_word)
                if ordinal == Int32(0) and slot < hot_rows:
                    cute.arch.store(hot_list + Int64(cur_parity * hot_rows + slot), hot_word)
            # One flag update per warp when its 32 consecutive slots share a tile word.
            key = ordinal * Int32(self.rank_tiles()) + slot // Int32(self.block_tile)
            if cute.arch.shuffle_sync(key, 0) == cute.arch.shuffle_sync(key, 31):
                for offset in cutlass.range_constexpr(5):
                    lane_flags = lane_flags | cute.arch.shuffle_sync_bfly(lane_flags, offset=16 >> offset)
                if lane != Int32(0):
                    lane_flags = Int32(0)
            if lane_flags != Int32(0):
                cute.arch.atomic_or(cute.recast_ptr(flags_base + Int64(key), dtype=Int32).llvm_ptr, lane_flags, sem="relaxed", scope="gpu")
            base += nthreads

    @cute.kernel
    def rank_fold_convert(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen_kv: Int32, rank_pitch: Int64, rank_planes: Int32):
        """Fold each row's initialized planes in plane order and convert."""
        # PDL: wait first; the dSink dependent reads only final O.dO sums and scaled LSE.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = Int32(RANK_M64_FOLD_ROWS) * seq_block_idx + tidy
        pitch, planes = rank_pitch, rank_planes
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        num_128_tiles = self.head_dim_main // 64
        num_terms = 2 * num_128_tiles + (0 if self.same_hdim_kv else 2)
        if seq_id < seqlen_kv:
            written = cute.arch.load(cute.recast_ptr(words + high + Int64(seq_id), dtype=Int32), Int32)
            acc = cute.make_rmem_tensor((num_terms,), Float32)
            acc.fill(0.0)
            first_row = mdKV_acc[None, seq_id, (0, batch_idx)]
            first_tile = cute.flat_divide(cute.flat_divide(first_row, (64,)), (32,))
            if written > Int32(0):
                for i in cutlass.range_constexpr(num_128_tiles):
                    for j in cutlass.range_constexpr(2):
                        acc[2 * i + j] = first_tile[tidx, j, i]
                if cutlass.const_expr(not self.same_hdim_kv):
                    for j in cutlass.range_constexpr(2):
                        acc[2 * num_128_tiles + j] = first_tile[tidx, j, num_128_tiles]
            for plane in cutlass.range(1, written, unroll=1):
                source_row = mdKV_acc[None, plane * Int32(pitch) + seq_id, (0, batch_idx)]
                tile = cute.flat_divide(cute.flat_divide(source_row, (64,)), (32,))
                for i in cutlass.range_constexpr(num_128_tiles):
                    for j in cutlass.range_constexpr(2):
                        acc[2 * i + j] = acc[2 * i + j] + tile[tidx, j, i]
                if cutlass.const_expr(not self.same_hdim_kv):
                    for j in cutlass.range_constexpr(2):
                        acc[2 * num_128_tiles + j] = acc[2 * num_128_tiles + j] + tile[tidx, j, num_128_tiles]
            out_row = mdKV[None, seq_id, (0, batch_idx)]
            for i in cutlass.range_constexpr(num_128_tiles):
                for j in cutlass.range_constexpr(2):
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    out_row[dim_idx] = self.element_dtype(acc[2 * i + j])
            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range_constexpr(2):
                    k = tidx // 2 + j * 16
                    dim_idx = self.head_dim_main + (k // 8) * 16 + k % 8 + (tidx % 2) * 8
                    out_row[dim_idx] = self.element_dtype(acc[2 * num_128_tiles + j])

    @cute.jit
    def launch_rank_fold_convert(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen_kv: Int32, stream: cuda.CUstream):
        """Fold the initialized planes and convert to the public dKV."""
        pitch, planes = self.rank_geometry(seqlen_kv)
        self.rank_fold_convert(mdKV_acc, mdKV, seqlen_kv, pitch, planes).launch(
            grid=[cute.ceil_div(seqlen_kv, RANK_M64_FOLD_ROWS), 1, 1],
            block=[self.num_threads_D_convert, RANK_M64_FOLD_ROWS, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def launch_rank_init(self, mdKV_acc: cute.Tensor, seqlen_kv: Int32, stream: cuda.CUstream):
        """Clear the conflict-rank workspace once per execution."""
        pitch, planes = self.rank_geometry(seqlen_kv)
        self.rank_init(mdKV_acc, seqlen_kv, pitch, planes).launch(grid=[RANK_M64_SETUP_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream)

    def rank_odo_args(self, mdKV_acc: cute.Tensor, odo):
        """Return ``rank_setup_a``'s O.dO arguments (the wave setups compute O.dO and the scaled LSE)."""
        assert odo is not None, "chain plans compute O.dO in their wave setups"
        return tuple(odo)

    @cute.jit
    def launch_rank_wave(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        q_offset: Int32,
        total_q: Int32,
        num_heads: Int32,
        stream: cuda.CUstream,
        odo=None,
    ):
        """Finish the previous wave, then assign the wave that starts at ``q_offset`` (programmatic launches)."""
        cur_size = min(Int32(self.rank_wave()), total_q - q_offset)
        cur_parity = self.rank_parity(q_offset)
        prev_size = min(q_offset, Int32(self.rank_wave()))
        pitch, planes = self.rank_geometry(seqlen_kv)
        self.rank_setup_a(
            mdKV_acc,
            mTopkIdxs,
            mTopkLength,
            seqlen_kv,
            q_offset,
            cur_size,
            cur_parity,
            q_offset - prev_size,
            prev_size,
            total_q,
            num_heads,
            *self.rank_odo_args(mdKV_acc, odo),
            pitch,
            planes,
        ).launch(
            grid=[self.rank_setup_ctas(), 1, 1],
            block=[RANK_M64_THREADS, 1, 1],
            smem=self.rank_setup_smem_bytes(),
            stream=stream,
            use_pdl=True,
            **self.rank_setup_bounds(),
        )
        # The first wave only: rank_scan_first (between rank_setup_a and rank_setup_b).
        for _ in cutlass.range(Int32(1) - min(q_offset, Int32(1)), unroll=1):
            self.rank_scan_first(mdKV_acc, seqlen_kv, total_q, pitch, planes).launch(
                grid=[RANK_M64_SETUP_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream, use_pdl=True
            )
        self.rank_setup_b(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, q_offset, cur_size, cur_parity, total_q, num_heads, pitch, planes).launch(
            grid=[self.rank_setup_ctas(), 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream, use_pdl=True, **self.rank_setup_bounds()
        )

    @cute.jit
    def launch_rank_final(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        total_q: Int32,
        num_heads: Int32,
        stream: cuda.CUstream,
        odo=None,
    ):
        """Finish the last wave: raise its rows' initialized-plane counts (``rank_setup_a`` without a current wave),
        then fold its hot rows into plane 0 (``rank_setup_b`` without slots). Both are programmatic launches."""
        last_begin = (total_q - Int32(1)) // Int32(self.rank_wave()) * Int32(self.rank_wave())
        last_parity = self.rank_parity(last_begin)
        pitch, planes = self.rank_geometry(seqlen_kv)
        self.rank_setup_a(
            mdKV_acc,
            mTopkIdxs,
            mTopkLength,
            seqlen_kv,
            last_begin,
            Int32(0),
            Int32(1) - last_parity,
            last_begin,
            total_q - last_begin,
            total_q,
            num_heads,
            *self.rank_odo_args(mdKV_acc, odo),
            pitch,
            planes,
        ).launch(
            grid=[self.rank_setup_ctas(), 1, 1],
            block=[RANK_M64_THREADS, 1, 1],
            smem=self.rank_setup_smem_bytes(),
            stream=stream,
            use_pdl=True,
            **self.rank_setup_bounds(),
        )
        # Only the hot-row fold runs here (no slots): RANK_M64_SETUP_CTAS threads cover its rows x head_dim items.
        self.rank_setup_b(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, last_begin, Int32(0), Int32(1) - last_parity, total_q, num_heads, pitch, planes).launch(
            grid=[RANK_M64_SETUP_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream, use_pdl=True, **self.rank_setup_bounds()
        )

    def rank_order_tiles(self) -> int:
        """Return the shared-memory tile-order length (chain-relay plans)."""
        return (int(self.max_topk) + int(self.block_tile) - 1) // int(self.block_tile) if self.rank_chain else 1

    @cute.jit
    def load_KV_ordered(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        sK: cute.Tensor,
        tile_count: Int32,
        topk: Int32,
        token_idx: Int32,
        load_mma_K_pipeline,
        sValid: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        sTileOrder: cute.Tensor,
    ):
        """Chain plans: gather one query's KV rows in the CTA's tile order."""
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_idx = cute.arch.block_idx()
        local_tidx = tidx % self.threads_per_warp
        local_warp_idx = tidx // self.threads_per_warp
        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, cute.make_layout((8,)), cute.make_layout((8,)))
        async_thr_copy = async_tiled_copy.get_slice(local_tidx % 8)
        load_mma_K_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_K_stage)
        # Gather the usual first tile while the tile order is formed; redo it if the order starts elsewhere.
        tile_index = tile_count - Int32(1)
        idx_reg = Int32(-1)
        load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
        sK_slice = sK[(None, None), 0, (None, None), load_mma_K_producer_state.index]
        sK_slice = cute.composition(sK_slice, cute.make_layout((self.block_tile, self.head_dim)))
        step = Int32(0)
        while step < Int32(2):
            idx_reg = self._load_tile_topk_idx(mTopkIdxs, tile_index, local_warp_idx, local_tidx, token_idx, batch_idx)
            self._load_kv_rows(
                mKV,
                sK_slice,
                sValid[None, 0],
                idx_reg,
                tile_index,
                topk,
                mTopkLength,
                is_first=True,
                local_tidx=local_tidx,
                local_warp_idx=local_warp_idx,
                async_copy_atom=async_copy_atom,
                async_thr_copy=async_thr_copy,
                token_idx=token_idx,
            )
            if step == Int32(0):
                _partial_cta_sync(self.rank_order_barrier.barrier_id, self.rank_order_barrier.num_threads)
            step += Int32(1)
            if sTileOrder[0] != tile_index:
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                tile_index = sTileOrder[0]
            else:
                step = Int32(2)
        step = Int32(0)
        while step < tile_count:
            if step > Int32(0):
                load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
                sK_slice = sK[(None, None), 0, (None, None), load_mma_K_producer_state.index]
                sK_slice = cute.composition(sK_slice, cute.make_layout((self.block_tile, self.head_dim)))
                self._load_kv_rows(
                    mKV,
                    sK_slice,
                    sValid[None, 0],
                    idx_reg,
                    tile_index,
                    topk,
                    mTopkLength,
                    is_first=True,
                    local_tidx=local_tidx,
                    local_warp_idx=local_warp_idx,
                    async_copy_atom=async_copy_atom,
                    async_thr_copy=async_thr_copy,
                    token_idx=token_idx,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            _partial_cta_sync(self.load_KV_sync_barrier.barrier_id, self.load_KV_sync_barrier.num_threads)
            load_mma_K_pipeline.producer_commit(load_mma_K_producer_state)
            load_mma_K_producer_state.advance()
            step += 1
            if step < tile_count:
                tile_index = sTileOrder[step]
                idx_reg = self._load_tile_topk_idx(mTopkIdxs, tile_index, local_warp_idx, local_tidx, token_idx, batch_idx)
                if cutlass.const_expr(self.max_topk < 1024):
                    # Short queries (fewer than 16 tiles): pull the next tile's rows into L2 while this tile computes.
                    if idx_reg >= Int32(0) and idx_reg < mKV.shape[0]:
                        _rank_prefetch_l2(mKV[idx_reg, None, (0, batch_idx)].iterator, self.head_dim * 2)

    @cute.jit
    def reduce_dKV_chain(
        self,
        tdKVtdKV: Tuple,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        max_seqlen_kv: Int32,
        tile_count: Int32,
        topk: Int32,
        token_idx: Int32,
        head_block_idx: Int32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mma_reduce_dKV_pipeline,
        sTileOrder: cute.Tensor,
        chain_geometry,
    ):
        """Chain-relay reducer: add each tile into its plane rows in the published tile order (``chain_geometry``:
        the plane pitch and count).

        ``mTopkIdxs`` is the wave's sorted-row view and ``token_idx`` the CTA's wave position. The TMEM/T2R
        pipeline is ``reduce_dKV``'s; the reducers read a tile's dKV0/dKV1 before they wait for the tile's chain
        predecessors (``rank_chain_tile_acquire``) and release its chain successors after the last add
        (``rank_chain_tile_release``). TMEM reuse is ordered by the T2R barriers; the hand-offs use their own.
        """
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4
        mdKV_acc_full = mdKV_acc
        mdKV_acc = self.select_dKV_accumulator(mdKV_acc_full, max_seqlen_kv, token_idx)
        tdKVtdKV0 = tdKVtdKV0[(None, None), 0, 0]
        tdKVtdKV1 = tdKVtdKV1[(None, None), 0, 0]
        tdKVtdKV2 = tdKVtdKV2[(None, None), 0, 0]
        tdKVtdKV3 = tdKVtdKV3[(None, None), 0, 0]
        if cutlass.const_expr(not self.same_hdim_kv):
            tdKVtdKV4 = tdKVtdKV4[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV0)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        mma_reduce_dKV_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_reduce_dKV_stage)
        rTopkIdx = cute.make_rmem_tensor((8,), cutlass.Int32)
        multiplicity = Int32(0)
        full_tiles = (topk % self.block_tile) == 0
        cute.arch.griddepcontrol_wait()
        query_codes = self.rank_query_codes(mdKV_acc_full, chain_geometry, token_idx)
        step = Int32(0)
        _partial_cta_sync(self.rank_order_barrier.barrier_id, self.rank_order_barrier.num_threads)
        tile_index = sTileOrder[0]
        # Each tile's indices and destination codes are loaded one tile ahead.
        rNextIdx = cute.make_rmem_tensor((8,), cutlass.Int32)
        rNextCode = cute.make_rmem_tensor((8,), cutlass.Int32)
        self._prefetch_dKV_destinations(mTopkIdxs, query_codes, tTR_cdKV, tile_index, topk, full_tiles, token_idx, batch_idx, rNextIdx, rNextCode)
        rRelayFlags = Int32(0)
        head_blocks = cute.ceil_div(problem_shape[3][0], self.block_tile)
        relay_row, relay_flags, relay_semaphores, relay_tag = self.rank_chain_relay_tables(mdKV_acc_full, chain_geometry, token_idx)
        # Lane t holds tile t's relay flags (<= 32 tiles).
        relay_lane = tidx % Int32(32)
        if relay_lane < tile_count:
            rRelayFlags = cute.arch.load(relay_flags + Int64(relay_lane), Int32)
        rRelayNext = self.rank_chain_head_flags(cute.arch.shuffle_sync(rRelayFlags, tile_index), head_block_idx, head_blocks) << Int32(
            RANK_M64_RELAY_FLAG_SHIFT
        )
        while step < tile_count:
            # dKV0/dKV1 are read before the chain hand-off: the next tile's MMAs need not wait for the acquire.
            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)
            rdKV0 = self.t2r_dKV(tdKVtdKV0)
            rdKV1 = self.t2r_dKV(tdKVtdKV1)
            cute.arch.fence_view_async_tmem_load()
            _partial_cta_sync(self.t2r_dKV01_done_barrier.barrier_id, self.t2r_dKV01_done_barrier.num_threads)
            mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
            mma_reduce_dKV_consumer_state.advance()
            next_tile = sTileOrder[min(step + Int32(1), tile_count - Int32(1))]
            relay = rRelayNext
            rRelayNext = self.rank_chain_head_flags(cute.arch.shuffle_sync(rRelayFlags, next_tile), head_block_idx, head_blocks) << Int32(
                RANK_M64_RELAY_FLAG_SHIFT
            )
            for i in cutlass.range_constexpr(8):
                topk_idx = rNextIdx[i]
                rTopkIdx[i] = Int32(-1)
                if topk_idx >= Int32(0) and topk_idx < max_seqlen_kv:
                    code = rNextCode[i]
                    if code >= Int32(0):
                        destination = code
                        if head_block_idx != Int32(0):
                            destination = destination & Int32(RANK_M64_FRESH_FLAG - 1)
                        rTopkIdx[i] = destination
                        # Warp i // 2 of each warpgroup prefetches row slot i (spreads the prefetches over four warps).
                        if cute.get(tTR_cdKV[i * 2 - i % 2], mode=[0]) == (i // 2) * 32 and head_block_idx == Int32(0):
                            if destination < Int32(RANK_M64_FRESH_FLAG):
                                plane_row = destination & Int32(RANK_M64_CHAIN_ROW_MASK)
                                _rank_prefetch_l2(mdKV_acc.iterator + Int64(plane_row) * Int64(self.head_dim), self.head_dim * 4)
            self._prefetch_dKV_destinations(mTopkIdxs, query_codes, tTR_cdKV, next_tile, topk, full_tiles, token_idx, batch_idx, rNextIdx, rNextCode)
            # One multiplicity check per tile.
            multiplicity = Int32(0)
            for i in cutlass.range_constexpr(8):
                if rTopkIdx[i] >= Int32(0):
                    multiplicity = multiplicity | (rTopkIdx[i] & Int32(RANK_M64_FRESH_FLAG - (1 << RANK_M64_MULT_SHIFT)))
            if cutlass.const_expr(self.same_hdim_kv):
                self.rank_chain_tile_acquire(relay)
                self.reduce_dKV_from_reg_chain(mdKV_acc, rdKV0, rTopkIdx, 0, multiplicity)
                self.reduce_dKV_from_reg_chain(mdKV_acc, rdKV1, rTopkIdx, 1, multiplicity)
            else:
                mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)
                rdKV4 = self.t2r_dKV_64(tdKVtdKV4)
                cute.arch.fence_view_async_tmem_load()
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                self.rank_chain_tile_acquire(relay)
                self.reduce_dKV_0_tail_1_from_reg_chain(mdKV_acc, rdKV0, rdKV4, rdKV1, rTopkIdx, multiplicity)
            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)
            if cutlass.const_expr(self.same_hdim_kv):
                rdKV2 = self.t2r_dKV(tdKVtdKV2)
                rdKV3 = self.t2r_dKV(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                _partial_cta_sync(self.t2r_dKV4_done_barrier.barrier_id, self.t2r_dKV4_done_barrier.num_threads)
            else:
                rdKV2 = self.t2r_dKV(tdKVtdKV2)
                rdKV3 = self.t2r_dKV(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                _partial_cta_sync(self.t2r_dKV23_done_barrier.barrier_id, self.t2r_dKV23_done_barrier.num_threads)
            if cutlass.const_expr(self.same_hdim_kv):
                self.reduce_dKV_from_reg_chain(mdKV_acc, rdKV2, rTopkIdx, 2, multiplicity)
                self.reduce_dKV_from_reg_chain(mdKV_acc, rdKV3, rTopkIdx, 3, multiplicity)
            else:
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                self.reduce_dKV_pair_from_reg_chain(mdKV_acc, rdKV2, rdKV3, rTopkIdx, 2, 3, multiplicity)
            if cutlass.const_expr(self.same_hdim_kv):
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
            self.rank_chain_tile_release(relay)
            step += 1

    @cute.jit
    def _rank_scale_by_multiplicity(self, destination: Int32, fragment: cute.Tensor):
        """Chain-only plans: scale a slot's FP32 dKV fragment by its slot multiplicity."""
        extra = (destination >> Int32(RANK_M64_MULT_SHIFT)) & Int32((RANK_M64_FRESH_FLAG >> RANK_M64_MULT_SHIFT) - 1)
        if extra != Int32(0):
            scale = Float32(extra + Int32(1))
            for k in cutlass.range_constexpr(cute.size(fragment)):
                fragment[k] = fragment[k] * scale

    @cute.jit
    def reduce_dKV_from_reg_chain(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx: int,
        multiplicity: Int32,
    ):
        """Reduce dKV from registers via atomic_add (stores for fresh rows)."""
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            rdKV_frg = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]
            rdKV_frg[2] = tTR_rdKV[coord_base + 16]
            rdKV_frg[3] = tTR_rdKV[coord_base + 18]
            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                row_idx = topk_idx & Int32(RANK_M64_CHAIN_ROW_MASK)
                if multiplicity != Int32(0):
                    self._rank_scale_by_multiplicity(topk_idx, rdKV_frg)
                dKV_row = dKV_acc[None, row_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (128,))
                tile_dKV_row = tile_dKV_row[None, sub_tile_idx]
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (4,))
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                if topk_idx >= Int32(RANK_M64_FRESH_FLAG):
                    _rank_store_f32x4(cur_dKV_frg.iterator, rdKV_frg[0], rdKV_frg[1], rdKV_frg[2], rdKV_frg[3])
                else:
                    cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load())

    @cute.jit
    def reduce_dKV_pair_from_reg_chain(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV0: cute.Tensor,
        tTR_rdKV1: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx0: int,
        sub_tile_idx1: int,
        multiplicity: Int32,
    ):
        """Reduce two resident 128-wide fragments with one index/address loop."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2
            rdKV_frg0 = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg0[0] = tTR_rdKV0[coord_base]
            rdKV_frg0[1] = tTR_rdKV0[coord_base + 2]
            rdKV_frg0[2] = tTR_rdKV0[coord_base + 16]
            rdKV_frg0[3] = tTR_rdKV0[coord_base + 18]
            rdKV_frg1 = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg1[0] = tTR_rdKV1[coord_base]
            rdKV_frg1[1] = tTR_rdKV1[coord_base + 2]
            rdKV_frg1[2] = tTR_rdKV1[coord_base + 16]
            rdKV_frg1[3] = tTR_rdKV1[coord_base + 18]
            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                row_idx = topk_idx & Int32(RANK_M64_CHAIN_ROW_MASK)
                if multiplicity != Int32(0):
                    self._rank_scale_by_multiplicity(topk_idx, rdKV_frg0)
                    self._rank_scale_by_multiplicity(topk_idx, rdKV_frg1)
                row_ptr = dKV_acc.iterator + Int64(row_idx) * Int64(self.head_dim)
                lane_offset = (dp_idx // 4) * 4
                ptr0 = row_ptr + sub_tile_idx0 * 128 + lane_offset
                ptr1 = row_ptr + sub_tile_idx1 * 128 + lane_offset
                if topk_idx >= Int32(RANK_M64_FRESH_FLAG):
                    _rank_store_f32x4(ptr0, rdKV_frg0[0], rdKV_frg0[1], rdKV_frg0[2], rdKV_frg0[3])
                    _rank_store_f32x4(ptr1, rdKV_frg1[0], rdKV_frg1[1], rdKV_frg1[2], rdKV_frg1[3])
                else:
                    cute.arch.atomic_add(ptr0.llvm_ptr, rdKV_frg0.load())
                    cute.arch.atomic_add(ptr1.llvm_ptr, rdKV_frg1.load())

    @cute.jit
    def reduce_dKV_0_tail_1_from_reg_chain(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV0: cute.Tensor,
        tTR_rdKV4: cute.Tensor,
        tTR_rdKV1: cute.Tensor,
        rTopkIdx: cute.Tensor,
        multiplicity: Int32,
    ):
        """Reduce resident dKV0/dKV1/tail fragments, preparing one KV row before REDG."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        for i0 in cutlass.range_constexpr(8):
            coord0 = i0 * 2 - i0 % 2
            frg00 = cute.make_rmem_tensor((4,), self.acc_dtype)
            frg00[0] = tTR_rdKV0[coord0]
            frg00[1] = tTR_rdKV0[coord0 + 2]
            frg00[2] = tTR_rdKV0[coord0 + 16]
            frg00[3] = tTR_rdKV0[coord0 + 18]
            frg40 = cute.make_rmem_tensor((2,), self.acc_dtype)
            frg40[0] = tTR_rdKV4[coord0]
            frg40[1] = tTR_rdKV4[coord0 + 2]
            frg10 = cute.make_rmem_tensor((4,), self.acc_dtype)
            frg10[0] = tTR_rdKV1[coord0]
            frg10[1] = tTR_rdKV1[coord0 + 2]
            frg10[2] = tTR_rdKV1[coord0 + 16]
            frg10[3] = tTR_rdKV1[coord0 + 18]
            lane_offset4 = (dp_idx // 4) * 4
            lane_offset2 = (dp_idx // 4) * 2
            topk0 = rTopkIdx[i0]
            if topk0 >= 0:
                row_idx0 = topk0 & Int32(RANK_M64_CHAIN_ROW_MASK)
                if multiplicity != Int32(0):
                    self._rank_scale_by_multiplicity(topk0, frg00)
                    self._rank_scale_by_multiplicity(topk0, frg10)
                    self._rank_scale_by_multiplicity(topk0, frg40)
                row0 = dKV_acc.iterator + Int64(row_idx0) * Int64(self.head_dim)
                if topk0 >= Int32(RANK_M64_FRESH_FLAG):
                    _rank_store_f32x4(row0 + lane_offset4, frg00[0], frg00[1], frg00[2], frg00[3])
                    _rank_store_f32x4(row0 + 128 + lane_offset4, frg10[0], frg10[1], frg10[2], frg10[3])
                    _rank_store_f32x2(row0 + self.head_dim_main + lane_offset2, frg40[0], frg40[1])
                else:
                    cute.arch.atomic_add((row0 + lane_offset4).llvm_ptr, frg00.load())
                    cute.arch.atomic_add((row0 + 128 + lane_offset4).llvm_ptr, frg10.load())
                    cute.arch.atomic_add((row0 + self.head_dim_main + lane_offset2).llvm_ptr, frg40.load())
