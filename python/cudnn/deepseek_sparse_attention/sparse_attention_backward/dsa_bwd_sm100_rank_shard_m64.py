# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conflict-rank dKV storage ("records") for the deterministic SM100 M64 DSA backward kernel.

Waves. A wave holds ``W`` queries (``rank_m64_wave``: 148, or 74 for paired head blocks with a large
record pool and for split pairs); the last wave may be partial, and ``rank_m64_strided`` plans interleave
the waves' queries (wave ``w`` of ``n`` holds queries ``w, w + n, ...``). One main-kernel launch runs a
wave: one CTA per query for at most 64 heads; a two-CTA cluster per query for 64 < H <= 128, CTA ``r``
running head block ``r`` (``rank_pair_heads``); a two-CTA cluster per query for split pairs, CTA ``r``
running the ``r``-th part of the query's top-k tiles (``rank_split_slots``, ``rank_m64_split``).

Destinations. Within a wave, the queries that select a KV row are ranked by wave ordinal. Rank
``k < P`` (``rank_m64_planes``) adds into dense plane ``k`` of the row, higher ranks into slot-private
overflow records, so every (plane, row) and every record has one writer per wave and head block. Rows
selected more than once by one query take per-slot records in every slot of the wave that selects them.
Destination codes are computed once per wave and reused by every head block. A slot whose plane row
holds no data yet, and every record's first writer, stores (the code carries ``RANK_M64_FRESH_FLAG``);
later writers add. No plane is cleared: row ``r`` of plane ``k`` is first stored by the rank-``k``
slot of the first wave in which ``r`` has more than ``k`` writers, and the next wave's ``rank_setup_b``
raises the per-row count of initialized planes, which the fold reads.

Writer order inside a launch:

- Paired head blocks: head block 0 adds a tile's rows; then each of its reducer threads publishes the
  tile in its completion word for (wave ordinal, thread) with ``st.release.gpu``, and the same thread of
  head block 1 spins with ``ld.acquire.gpu`` until it observes the tile before adding the same rows.
- Split pairs: CTA 1 stores its FP32 dQ partial into the workspace and publishes its completion word;
  CTA 0 acquires it and adds the partial before storing dQ. Their dKV slots are disjoint.

The two CTAs of a cluster are co-resident; published values grow with the wave index, so the completion
words are cleared only by ``rank_init``.

Record tail. The next wave's ``rank_setup_a`` sums each row's records in segments of ``rank_segment``
records (one warp per segment; twice ``RANK_M64_SEGMENT`` while a wave has at most
``RANK_M64_LONG_SEGMENT_SLOTS`` slots): a single-segment row goes straight to plane 0, a multi-segment
row leaves each sum in its segment's first record, and ``rank_setup_b`` adds those to plane 0 in segment
order. Per-slot records of repeated rows are added in ascending (query, slot) order. ``rank_fold_convert``
folds each row's initialized planes in plane order.

Launch sequence of one execution (ordinary launches start after the preceding grid completed;
programmatic ones may start once every CTA of the preceding grid triggered or exited):

- ``rank_init`` (ordinary): triggers at entry, never waits; clears the row state, the counters, the
  initialized-plane counts and the completion words.
- ``rank_mark_first`` (programmatic): triggers and waits at entry in every CTA; records wave 0's ordinals.
- ``sum_OdO`` (programmatic): no explicit trigger; CTA 0 waits at its end. Before that wait it touches only
  the inputs and the O.dO / scaled-LSE statistics, which no conflict-rank grid accesses.
- For each wave ``w``:

  - ``rank_setup_a`` (ordinary): triggers at entry, never waits. It finishes wave ``w - 1``'s records and
    reads only wave ``w - 1``'s lists, counters and records (row-state and counter slot ``w - 1 mod 3``,
    list parity ``w - 1 mod 2``).
  - ``rank_setup_b`` (programmatic): triggers at entry; CTA 0 and the CTAs with segment-merge items wait
    after their assignment work. Before that wait it touches row-state slots ``w`` and ``w - 1`` (reads),
    ``w + 1`` (clears), wave ``w``'s lists, codes and offsets, and the initialized-plane counts, which it
    raises for wave ``w - 1``'s rows and reads for freshness while ``rank_setup_a`` may still store 1 into
    them for repeated rows (``rank_setup_b`` takes the larger of the stored count and the one wave
    ``w - 1``'s row state implies). ``rank_setup_a`` touches none of the rest.
  - main kernel (programmatic): in-range CTAs trigger after the range check (no effect: the next launch is
    an ordinary ``rank_setup_a``); the reducer warps wait before reading destination codes, and the empty
    warp of head block 0 waits before recording wave ``w + 1``'s ordinals into the row-state slot
    ``rank_setup_b`` cleared. Before their waits the CTAs touch the inputs, the statistics, dQ and the
    split pairs' dQ partials and completion words, which the setup grids do not access. Out-of-range CTAs
    exit without waiting; every wave has an in-range query, and its head block 0 waits.

- ``rank_setup_a`` and ``rank_setup_b`` once more (no current wave) finish the last wave.
- ``rank_fold_convert`` (ordinary): triggers at entry, never waits; folds the planes into dKV.
- ``sum_dSink`` (programmatic, ``_ConflictRankPlan``): no explicit trigger; CTA (0, 0, 0) waits at its end.
  Before that wait it reads the statistics and writes dSink, which the fold does not access.

Invariants. Every programmatic grid has at least one CTA that executes ``griddepcontrol.wait`` before the
grid completes, so a grid's completion implies its primary's, and the completion of ``sum_dSink`` implies
the whole sequence's: the next ordinary launch on the stream (the next execution's ``rank_init``, or a
consumer of dQ, dKV or dSink) starts after every write of this execution. Programmatic grids that trigger
before waiting (``rank_mark_first``, ``rank_setup_b``) have dependents that read nothing of the preceding
grids before their own waits. The kernels rely on the usual visibility of completed grids: a write of a
grid whose completion a wait has observed (``rank_setup_a``'s writes, observed by ``rank_setup_b``'s CTA
0) is visible to later grids (the main kernel after its own wait).

Workspace (32-bit words in this order, ``pitch = round_up(S_kv, 8)``, ``R = round_up(D, 8)``, ``S`` the
wave's ``W * K`` slots rounded up to 32, ``rank_offsets``): dense planes ``P * pitch * R``; records
``S * R``; row state, three rotating wave slots ``3 * pitch * 8`` (five ordinal words, a duplicate flag);
overflow offsets ``pitch``; per-slot duplicate records, two parities ``2 * S``; overflow row lists and
segment lists, two parities ``2 * 4 * S`` each; duplicate row lists, three slots ``3 * S``; destination
codes ``S``; counters ``12``; initialized planes per row ``pitch``; completion words
``RANK_M64_FLAG_WORDS``; for split pairs, then the FP32 dQ partials ``W * 64 * R`` (aligned up to 4 words).
"""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import dsl_user_op

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
    _rank_flag_acquire,
    _rank_flag_release,
    _rank_load_f32x4,
    _rank_load_once_f32x4,
    _rank_load_u32x4,
    _rank_prefetch_l2,
    _rank_store_f32x4,
    _rank_store_zero_x4,
    _rank_topk,
    rank_m64_pitch,
    rank_m64_record,
)

RANK_M64_ORDINAL_WORDS = 5
# Allocation counters of three rotating wave slots: packed (records, rows, segments), duplicate rows, padding.
RANK_M64_COUNTER_WORDS = 12
# Paired head blocks (H96/H128): one completion word per (wave ordinal, reducer thread) that the
# head block 0 reducer publishes after its tile's adds and the head block 1 reducer waits for.
RANK_M64_PAIR_THREADS = 256
RANK_M64_FLAG_WORDS = RANK_M64_WAVE * RANK_M64_PAIR_THREADS
# ``rank_setup_a`` keeps four records per lane in registers (two resident CTAs per SM);
# ``rank_setup_b`` CTAs take the SMs its CTAs free.
RANK_M64_TAIL_CTAS = 296
RANK_M64_ASSIGN_CTAS = 444
# Top-k widths up to this use ``RANK_M64_TAIL_CTAS`` assignment CTAs (see ``rank_assign_ctas``).
RANK_M64_ASSIGN_COMPACT_TOPK = 1024
# Bits per field of the packed (records, overflow rows, segments) allocation counter;
# conflict-rank storage is selected only while ``W * K`` fits one field.
RANK_M64_FIELD_BITS = 21
# Records whose loads one tail warp keeps in flight before adding them in order.
RANK_M64_TAIL_BATCH = 4
# Records one tail warp sums per overflow segment.
RANK_M64_SEGMENT = 16
# Waves of at most this many slots sum records in segments of twice that length (fewer, shorter merges).
RANK_M64_LONG_SEGMENT_SLOTS = 1 << 17
# Paired head blocks (64 < H <= 128) run half waves (one resident round of CTA pairs) when a full
# wave's record pool would exceed this, halving the record pool and the planes a wave needs.
RANK_M64_PAIR_RECORD_BYTES = (2 << 30) // 5  # 0.4 GiB
# Strided waves (wave ``w`` of ``n`` holds queries ``w, w + n, ...``) spread neighbouring queries' shared
# rows over the waves; used with at least this many waves while the KV rows stay L2-resident. The wave order
# is a compile-time choice, so a records plan has two compiled variants over the query count: up to
# ``(RANK_M64_STRIDED_MIN_WAVES - 1) * W`` queries in order, more queries strided (L2-resident KV only).
RANK_M64_STRIDED_MIN_WAVES = 8
RANK_M64_STRIDED_KV_BYTES = 48 << 20
# A split pair spends each CTA's fixed cost on half the tiles (about 20% more time per query).
RANK_M64_SPLIT_GAIN = 1.65


def rank_m64_split(num_heads: int, head_dim: int, max_topk: int, total_s_q: int = 0, total_s_kv: int = 0) -> bool:
    """Return whether a plan of at most 64 heads runs each query as a CTA pair that splits its top-k tiles: half waves
    (one resident round of pairs) halve the record pool, used when that cuts the workspace (at least the 1 GiB floor) by
    more than ``RANK_M64_SPLIT_GAIN``.

    The rule compares the fixed full and half waves (``RANK_M64_WAVE`` and half of it), so it depends on the head
    count, head dimension, top-k width and KV extent but not on the query count."""
    if not 0 < int(num_heads) <= 64 or int(total_s_q) <= 0 or int(total_s_kv) <= 0:
        return False
    full_bytes = rank_m64_workspace_words(total_s_kv, head_dim, max_topk, RANK_M64_WAVE) * 4
    half_bytes = rank_m64_workspace_words(total_s_kv, head_dim, max_topk, RANK_M64_WAVE // 2, True) * 4
    return full_bytes > RANK_M64_SPLIT_GAIN * max(half_bytes, RANK_M64_DENSE_BUDGET_BYTES)


def rank_m64_wave(num_heads: int, head_dim: int, max_topk: int, total_s_q: int = 0, total_s_kv: int = 0) -> int:
    """Return the query CTAs per wave of a plan (see ``RANK_M64_PAIR_RECORD_BYTES`` and ``rank_m64_split``).

    The wave is a compile-time constant of the plan; it does not follow the query count, so one
    compiled plan serves every query extent (the last wave may be partial)."""
    wave = RANK_M64_WAVE
    if 64 < int(num_heads) <= 128 and RANK_M64_WAVE * int(max_topk) * rank_m64_record(head_dim) * 4 > RANK_M64_PAIR_RECORD_BYTES:
        wave = RANK_M64_WAVE // 2
    if rank_m64_split(num_heads, head_dim, max_topk, total_s_q, total_s_kv):
        wave = RANK_M64_WAVE // 2
    return wave


def rank_m64_strided(total_s_q: int, total_s_kv: int, head_dim: int, wave: int) -> bool:
    """Return whether a plan uses strided waves (see ``RANK_M64_STRIDED_KV_BYTES``; the one rule of a records plan
    that follows the query count, giving two compiled variants per shape)."""
    return -(-int(total_s_q) // int(wave)) >= RANK_M64_STRIDED_MIN_WAVES and int(total_s_kv) * int(head_dim) * 2 <= RANK_M64_STRIDED_KV_BYTES


def rank_m64_mode(num_heads: int, head_dim: int, max_topk: int, total_s_q: int, total_s_kv: int, wave: int) -> int:
    """Return the wave-mode bits of a plan of ``wave`` query CTAs per wave: 1 strided, 2 split pairs."""
    if int(wave) <= 0:
        return 0
    return int(rank_m64_strided(total_s_q, total_s_kv, head_dim, wave)) | (2 if rank_m64_split(num_heads, head_dim, max_topk, total_s_q, total_s_kv) else 0)


def rank_m64_plan_config(num_heads: int, head_dim: int, max_topk: int, total_s_q: int, total_s_kv: int):
    """Return a records plan's compile-time wave settings ``(wave, mode)``: ``rank_m64_wave`` and its ``rank_m64_mode`` bits."""
    wave = rank_m64_wave(num_heads, head_dim, max_topk, total_s_q, total_s_kv)
    return wave, rank_m64_mode(num_heads, head_dim, max_topk, total_s_q, total_s_kv, wave)


# Words per slot of the slot-indexed metadata arrays, in the order ``rank_offsets`` lays them out: duplicate
# records (2), overflow row lists (8), segment lists (8), duplicate row lists (3) and destination codes (1).
RANK_M64_SLOT_ARRAY_WORDS = 22


def rank_m64_slot_stride(slots: int) -> int:
    """Return the words reserved per slot-sized array: the slot count rounded up to 128 bytes, so every region stays aligned."""
    return (int(slots) + 31) // 32 * 32


def rank_m64_metadata_words(pitch: int, record: int, slots: int) -> int:
    """Return the 32-bit words of everything but the dense planes: records, row state, offsets, slot lists, counters."""
    stride = rank_m64_slot_stride(slots)
    return stride * record + 3 * pitch * RANK_M64_STATE_WORDS + 2 * pitch + RANK_M64_SLOT_ARRAY_WORDS * stride + RANK_M64_COUNTER_WORDS + RANK_M64_FLAG_WORDS


def rank_m64_planes(total_s_kv: int, head_dim: int, max_topk: int, wave: int = RANK_M64_WAVE) -> int:
    """Return the dense plane count of one plan (see the plane policy at ``RANK_M64_DENSE_BUDGET_BYTES``)."""
    pitch = rank_m64_pitch(total_s_kv)
    record = rank_m64_record(head_dim)
    slots = wave * int(max_topk)
    density = (slots + pitch - 1) // pitch
    budget = RANK_M64_DENSE_BUDGET_BYTES // (pitch * record * 4)
    if budget < 2 and 2 * pitch * record * 4 <= RANK_M64_PAIR_BUDGET_BYTES and 2 * slots >= pitch:
        budget = 2
    margin = 1 + (1 if slots > 2 * pitch else 0)
    planes = max(1, min(wave - 1, density + margin, budget))
    if (planes * pitch * record + rank_m64_metadata_words(pitch, record, slots)) * 4 > RANK_M64_COMPACT_BYTES:
        planes = max(1, min(wave - 1, density, budget))
    return planes


def rank_m64_workspace_words(total_s_kv: int, head_dim: int, max_topk: int, wave: int = RANK_M64_WAVE, split: bool = False) -> int:
    """Return the 32-bit word count of the conflict-rank dKV workspace (``split``: plus one FP32 dQ partial of 64 heads per wave ordinal)."""
    pitch = rank_m64_pitch(total_s_kv)
    record = rank_m64_record(head_dim)
    slots = wave * int(max_topk)
    planes = rank_m64_planes(total_s_kv, head_dim, max_topk, wave)
    return planes * pitch * record + rank_m64_metadata_words(pitch, record, slots) + (wave * 64 * record + 4 if split else 0)


def rank_m64_workspace_rows(total_s_kv: int, head_dim: int, max_topk: int, wave: int = RANK_M64_WAVE, split: bool = False) -> int:
    """Return the workspace size in whole records."""
    record = rank_m64_record(head_dim)
    return -(-rank_m64_workspace_words(total_s_kv, head_dim, max_topk, wave, split) // record)


def rank_m64_workspace_bytes(total_s_kv: int, head_dim: int, max_topk: int, wave: int = RANK_M64_WAVE, split: bool = False) -> int:
    """Return the conflict-rank dKV workspace in bytes."""
    return rank_m64_workspace_rows(total_s_kv, head_dim, max_topk, wave, split) * rank_m64_record(head_dim) * 4


@dsl_user_op
def _rank_discard_l2(source: cute.Pointer, *, loc=None, ip=None) -> None:
    """Drop one consumed 128-byte line from L2 without writing it back (after the warp has read it)."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [source_i64],
        # Only a whole 128-byte line may be dropped: skip the hint when the workspace view is not 128-byte aligned.
        "{\n\t.reg .pred p;\n\t.reg .b64 a;\n\tand.b64 a, $0, 127;\n\tsetp.eq.b64 p, a, 0;\n\t@p discard.global.L2 [$0], 128;\n\t}",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_sync_warp(*, loc=None, ip=None) -> None:
    """Warp barrier (every lane calls it)."""
    llvm.inline_asm(None, [], "bar.warp.sync -1;", "", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def _rank_warp_allocate3(counter: cute.Pointer, records: Int32, rows: Int32, segments: Int32, lane: Int32):
    """Reserve consecutive records, overflow rows and segments for each lane with one 64-bit atomic per warp.

    Every lane of the warp must call this together. Returns the lane's first entry of each.
    """
    inc_records = records
    inc_rows = rows
    inc_segments = segments
    for step in cutlass.range_constexpr(5):
        offset = 1 << step
        part_records = cute.arch.shuffle_sync_up(inc_records, offset=offset, mask_and_clamp=0)
        part_rows = cute.arch.shuffle_sync_up(inc_rows, offset=offset, mask_and_clamp=0)
        part_segments = cute.arch.shuffle_sync_up(inc_segments, offset=offset, mask_and_clamp=0)
        if lane >= Int32(offset):
            inc_records += part_records
            inc_rows += part_rows
            inc_segments += part_segments
    total_records = cute.arch.shuffle_sync(inc_records, 31)
    total_rows = cute.arch.shuffle_sync(inc_rows, 31)
    total_segments = cute.arch.shuffle_sync(inc_segments, 31)
    low = Int32(0)
    high = Int32(0)
    if lane == Int32(31):
        if total_records + total_rows + total_segments > Int32(0):
            packed = Int64(total_records) + (Int64(total_rows) << Int64(RANK_M64_FIELD_BITS)) + (Int64(total_segments) << Int64(2 * RANK_M64_FIELD_BITS))
            old = cute.arch.atomic_add(cute.recast_ptr(counter, dtype=Int64).llvm_ptr, packed, sem="relaxed", scope="gpu")
            low = Int32(old & Int64(0xFFFFFFFF))
            high = Int32(old >> Int64(32))
    old = (Int64(cute.arch.shuffle_sync(high, 31)) << Int64(32)) | (Int64(cute.arch.shuffle_sync(low, 31)) & Int64(0xFFFFFFFF))
    mask = Int64((1 << RANK_M64_FIELD_BITS) - 1)
    first_record = Int32(old & mask) + inc_records - records
    first_row = Int32((old >> Int64(RANK_M64_FIELD_BITS)) & mask) + inc_rows - rows
    first_segment = Int32(old >> Int64(2 * RANK_M64_FIELD_BITS)) + inc_segments - segments
    return first_record, first_row, first_segment


@cute.jit
def _rank_counter_field(words: cute.Pointer, counters: Int64, parity: Int32, field: cutlass.Constexpr[int]) -> Int32:
    """Read field ``field`` (0 records, 1 overflow rows, 2 segments) of a wave parity's packed allocation counter."""
    packed = cute.arch.load(cute.recast_ptr(words + counters + Int64(parity * Int32(4)), dtype=Int64), Int64)
    return Int32((packed >> Int64(field * RANK_M64_FIELD_BITS)) & Int64((1 << RANK_M64_FIELD_BITS) - 1))


@cute.jit
def _rank_lower_mask(ordinal: Int32, word: cutlass.Constexpr[int]) -> Uint32:
    """Bits of ordinal word ``word`` that belong to lower wave ordinals."""
    shift = ordinal - Int32(word * 32)
    mask = Uint32(0)
    if shift >= Int32(32):
        mask = Uint32(0xFFFFFFFF)
    elif shift > Int32(0):
        mask = (Uint32(1) << Uint32(shift)) - Uint32(1)
    return mask


class RankShardM64Reduction(ConflictRankReducerMixin):
    """Workspace geometry, wave setup, destination codes and fold for conflict-rank dKV storage.

    The host class provides ``head_dim``, ``max_topk`` and ``q_wave_ctas`` (``rank_m64_wave``).
    """

    rank_shard = True
    # The main kernel of wave ``w`` records the row ordinals of wave ``w + 1`` (its idle warp);
    # ``rank_mark_first`` records the first wave.
    rank_bits_in_main = False
    rank_strided = False
    # See ``rank_m64_split``: CTA ``r`` of a query's pair runs top-k tiles [r * T0, T0 + r * (T - T0)), T0 = ceil(T / 2).
    rank_split_slots = False

    @cute.jit
    def rank_query(self, wave_begin: Int32, ordinal: Int32, total_q: Int32) -> Int32:
        """Return the query of wave ordinal ``ordinal`` in the wave that starts at ``wave_begin``."""
        query = wave_begin + ordinal
        if cutlass.const_expr(self.rank_strided):
            query = ordinal * ((total_q + Int32(self.q_wave_ctas - 1)) // Int32(self.q_wave_ctas)) + wave_begin // Int32(self.q_wave_ctas)
        return query

    @cute.jit
    def rank_wave_size(self, wave_begin: Int32, total_q: Int32) -> Int32:
        """Return the query count of the wave that starts at ``wave_begin``."""
        size = min(Int32(self.q_wave_ctas), total_q - wave_begin)
        if cutlass.const_expr(self.rank_strided):
            waves = (total_q + Int32(self.q_wave_ctas - 1)) // Int32(self.q_wave_ctas)
            size = (total_q - wave_begin // Int32(self.q_wave_ctas) + waves - Int32(1)) // waves
        return size

    @cute.jit
    def rank_ordinal(self, query: Int32, total_q: Int32):
        """Return the wave ordinal and the wave index of ``query``."""
        ordinal = query % Int32(self.q_wave_ctas)
        wave = query // Int32(self.q_wave_ctas)
        if cutlass.const_expr(self.rank_strided):
            waves = (total_q + Int32(self.q_wave_ctas - 1)) // Int32(self.q_wave_ctas)
            ordinal = query // waves
            wave = query % waves
        return ordinal, wave

    @cute.jit
    def rank_geometry(self, seqlen_kv: Int32):
        """Return the padded plane pitch and the dense plane count for one KV extent."""
        record = rank_m64_record(self.head_dim)
        slots = self.q_wave_ctas * self.max_topk
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        # Same arithmetic as ``rank_m64_planes``.
        density = (Int64(slots) + pitch - Int64(1)) // pitch
        margin = Int64(1)
        if Int64(slots) > Int64(2) * pitch:
            margin = Int64(2)
        budget = Int64(RANK_M64_DENSE_BUDGET_BYTES) // (pitch * Int64(record * 4))
        if budget < Int64(2):
            if Int64(2) * pitch * Int64(record * 4) <= Int64(RANK_M64_PAIR_BUDGET_BYTES):
                if Int64(2 * slots) >= pitch:
                    budget = Int64(2)
        planes = min(density + margin, budget)
        planes = min(planes, Int64(self.q_wave_ctas - 1))
        planes = max(planes, Int64(1))
        stride = self.rank_slot_stride()
        metadata = (
            Int64(stride * record)
            + Int64(3 * RANK_M64_STATE_WORDS + 2) * pitch
            + Int64(stride * RANK_M64_SLOT_ARRAY_WORDS + RANK_M64_COUNTER_WORDS + RANK_M64_FLAG_WORDS)
        )
        if (planes * pitch * Int64(record) + metadata) * Int64(4) > Int64(RANK_M64_COMPACT_BYTES):
            planes = min(density, budget)
            planes = min(planes, Int64(self.q_wave_ctas - 1))
            planes = max(planes, Int64(1))
        return pitch, Int32(planes)

    @cute.jit
    def rank_slot_stride(self) -> int:
        """Return the words reserved per slot-sized array (``rank_m64_slot_stride`` of one wave's slots)."""
        return rank_m64_slot_stride(self.q_wave_ctas * self.max_topk)

    @cute.jit
    def rank_offsets(self, pitch: Int64, planes: Int32):
        """Return the 32-bit word offsets of every workspace region."""
        record = Int64(rank_m64_record(self.head_dim))
        slots = Int64(self.rank_slot_stride())
        records = Int64(planes) * pitch * record
        state = records + slots * record
        offsets = state + Int64(3 * RANK_M64_STATE_WORDS) * pitch
        dup_records = offsets + pitch
        row_list = dup_records + Int64(2) * slots
        segment_list = row_list + Int64(8) * slots
        dup_list = segment_list + Int64(8) * slots
        codes = dup_list + Int64(3) * slots
        counters = codes + slots
        return records, state, offsets, dup_records, row_list, dup_list, counters, codes, segment_list

    @cute.jit
    def rank_high(self, pitch: Int64, planes: Int32) -> Int64:
        """Return the word offset of the per-row count of initialized planes."""
        _, _, _, _, _, _, counters, _, _ = self.rank_offsets(pitch, planes)
        return counters + Int64(RANK_M64_COUNTER_WORDS)

    @cute.jit
    def rank_flags(self, pitch: Int64, planes: Int32) -> Int64:
        """Return the word offset of the paired head-block completion words."""
        return self.rank_high(pitch, planes) + pitch

    @cute.jit
    def rank_split_dq(self, mdKV_acc: cute.Tensor, seqlen_kv: Int32, token_idx: Int32, total_q: Int32, thread: Int32):
        """Split pairs: one compute thread's completion word and FP32 dQ partial (behind the completion words; 16-byte
        chunks of the 128 threads interleaved) of the query's wave ordinal, and the wave's completion value."""
        pitch, planes = self.rank_geometry(seqlen_kv)
        flags = self.rank_flags(pitch, planes)
        ordinal, wave = self.rank_ordinal(token_idx, total_q)
        flag = cute.recast_ptr(cute.recast_ptr(mdKV_acc.iterator, dtype=Int32) + flags + Int64(ordinal * Int32(128) + thread), dtype=Int32)
        base = (flags + Int64(RANK_M64_FLAG_WORDS + 3)) // Int64(4) * Int64(4)
        partial = mdKV_acc.iterator + base + Int64(ordinal) * Int64(64 * rank_m64_record(self.head_dim)) + Int64(thread * Int32(4))
        return flag, partial, wave + Int32(1)

    @property
    def rank_segment(self) -> int:
        """Records per tail segment of this plan."""
        return 2 * RANK_M64_SEGMENT if self.q_wave_ctas * self.max_topk <= RANK_M64_LONG_SEGMENT_SLOTS else RANK_M64_SEGMENT

    def rank_assign_ctas(self) -> int:
        """Grid of ``rank_setup_b``: with at most ``RANK_M64_ASSIGN_COMPACT_TOPK`` slots per query the
        assignment fits in one round of resident CTAs, so the merge after ``rank_setup_a`` is not
        held behind a second round of CTAs that only wait for it."""
        if self.q_wave_ctas * self.max_topk <= RANK_M64_WAVE * RANK_M64_ASSIGN_COMPACT_TOPK:
            return RANK_M64_TAIL_CTAS
        return RANK_M64_ASSIGN_CTAS

    @cute.jit
    def rank_parity(self, q_offset: Int32) -> Int32:
        """Return the list parity (wave index modulo two) of the wave that starts at ``q_offset``."""
        return (q_offset // Int32(self.q_wave_ctas)) % Int32(2)

    @cute.jit
    def rank_slot3(self, q_offset: Int32) -> Int32:
        """Return the row-state, counter and duplicate-list slot (wave index modulo three) of the wave that starts at ``q_offset``."""
        return (q_offset // Int32(self.q_wave_ctas)) % Int32(3)

    @cute.jit
    def rank_query_tables(self, mdKV_acc: cute.Tensor, max_seqlen_kv: Int32, token_idx: Int32, total_q: Int32):
        """Return one query CTA's destination-code row, the per-row record offsets and the first record row."""
        pitch, planes = self.rank_geometry(max_seqlen_kv)
        _, _, offsets, _, _, _, _, codes, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        ordinal, _ = self.rank_ordinal(token_idx, total_q)
        query_codes = cute.recast_ptr(words + codes + Int64(ordinal) * Int64(self.max_topk), dtype=Int32)
        row_offsets = cute.recast_ptr(words + offsets, dtype=Int32)
        return query_codes, row_offsets, planes * Int32(pitch)

    @cute.jit
    def _rank_add_records(
        self,
        floats: cute.Pointer,
        destination: Int64,
        first: Int64,
        count: Int32,
        stride: Int64,
        to_plane: Int32,
        lane: Int32,
        batch: cutlass.Constexpr[int],
    ):
        """Sum records ``first + j * stride`` for ``j < count`` in order.

        With ``to_plane`` the sum is added to plane-0 row ``destination``; otherwise
        it replaces record ``first``. ``batch`` records are loaded before any of
        them is added, so their loads overlap.
        """
        record = Int64(rank_m64_record(self.head_dim))
        quads = rank_m64_record(self.head_dim) // 4
        chunks = (quads + 31) // 32
        if to_plane != Int32(0):
            # The plane-0 row is read back after the records are summed: fetch it meanwhile.
            if lane == Int32(0):
                _rank_prefetch_l2(floats + destination * record, rank_m64_record(self.head_dim) * 4)
        acc = cute.make_rmem_tensor((4 * chunks,), Float32)
        acc.fill(0.0)
        loaded = cute.make_rmem_tensor((batch * 4 * chunks,), Float32)
        loaded.fill(0.0)
        item = Int32(0)
        while item < count:
            for b in cutlass.range_constexpr(batch):
                if item + Int32(b) < count:
                    for chunk in cutlass.range_constexpr(chunks):
                        quad = lane + Int32(32 * chunk)
                        if quad < Int32(quads):
                            source = floats + (first + Int64(item + Int32(b)) * stride) * record + Int64(quad * Int32(4))
                            v0, v1, v2, v3 = _rank_load_once_f32x4(source)
                            loaded[(b * chunks + chunk) * 4] = v0
                            loaded[(b * chunks + chunk) * 4 + 1] = v1
                            loaded[(b * chunks + chunk) * 4 + 2] = v2
                            loaded[(b * chunks + chunk) * 4 + 3] = v3
            for b in cutlass.range_constexpr(batch):
                if item + Int32(b) < count:
                    for value in cutlass.range_constexpr(4 * chunks):
                        acc[value] = acc[value] + loaded[b * 4 * chunks + value]
            # The batch is in registers: drop its dead records' L2 lines unwritten (a segment's first record is rewritten below).
            if cutlass.const_expr(rank_m64_record(self.head_dim) % 32 == 0):
                _rank_sync_warp()
                for b in cutlass.range_constexpr(batch):
                    if item + Int32(b) < count:
                        if to_plane != Int32(0) or item + Int32(b) > Int32(0):
                            if lane < Int32(rank_m64_record(self.head_dim) // 32):
                                _rank_discard_l2(floats + (first + Int64(item + Int32(b)) * stride) * record + Int64(lane * Int32(32)))
            item += Int32(batch)
        for chunk in cutlass.range_constexpr(chunks):
            quad = lane + Int32(32 * chunk)
            if quad < Int32(quads):
                if to_plane != Int32(0):
                    target = floats + destination * record + Int64(quad * Int32(4))
                    p0, p1, p2, p3 = _rank_load_f32x4(target)
                    _rank_store_f32x4(target, p0 + acc[4 * chunk], p1 + acc[4 * chunk + 1], p2 + acc[4 * chunk + 2], p3 + acc[4 * chunk + 3])
                else:
                    _rank_store_f32x4(
                        floats + first * record + Int64(quad * Int32(4)), acc[4 * chunk], acc[4 * chunk + 1], acc[4 * chunk + 2], acc[4 * chunk + 3]
                    )

    @cute.jit
    def _rank_add_duplicate_row(
        self,
        floats: cute.Pointer,
        words: cute.Pointer,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        records: Int64,
        dup_records: Int64,
        high: Int64,
        row: Int32,
        wave_begin: Int32,
        wave_size: Int32,
        lane: Int32,
        total_q: Int32,
    ):
        """Add a duplicated row's per-slot records to plane 0 in ascending (query, slot) order.

        A row with no initialized plane (never written) stores the sum and marks plane 0 initialized.
        """
        record = Int64(rank_m64_record(self.head_dim))
        quads = rank_m64_record(self.head_dim) // 4
        chunks = (quads + 31) // 32
        if lane == Int32(0):
            _rank_prefetch_l2(floats + Int64(row) * record, rank_m64_record(self.head_dim) * 4)
        acc = cute.make_rmem_tensor((4 * chunks,), Float32)
        acc.fill(0.0)
        for ordinal in cutlass.range(wave_size, unroll=1):
            query = self.rank_query(wave_begin, ordinal, total_q)
            topk = _rank_topk(mTopkIdxs, mTopkLength, query)
            for slot_base in cutlass.range(0, topk, 32, unroll=1):
                slot = slot_base + lane
                match = cutlass.Boolean(False)
                own_record = Int32(0)
                if slot < topk:
                    if mTopkIdxs[slot, (query, 0)] == row:
                        match = cutlass.Boolean(True)
                        own_record = cute.arch.load(cute.recast_ptr(words + dup_records + Int64(ordinal * Int32(self.max_topk) + slot), dtype=Int32), Int32)
                pending = Uint32(cute.arch.vote_ballot_sync(match))
                while pending != Uint32(0):
                    lowest = pending & (~pending + Uint32(1))
                    source_lane = Int32(cute.arch.popc(lowest - Uint32(1)))
                    slot_record = cute.arch.shuffle_sync(own_record, source_lane)
                    for chunk in cutlass.range_constexpr(chunks):
                        quad = lane + Int32(32 * chunk)
                        if quad < Int32(quads):
                            source = floats + (records // record + Int64(slot_record)) * record + Int64(quad * Int32(4))
                            v0, v1, v2, v3 = _rank_load_once_f32x4(source)
                            acc[4 * chunk] = acc[4 * chunk] + v0
                            acc[4 * chunk + 1] = acc[4 * chunk + 1] + v1
                            acc[4 * chunk + 2] = acc[4 * chunk + 2] + v2
                            acc[4 * chunk + 3] = acc[4 * chunk + 3] + v3
                    pending = pending ^ lowest
        high_ptr = cute.recast_ptr(words + high + Int64(row), dtype=Int32)
        initialized = cute.arch.load(high_ptr, Int32)
        for chunk in cutlass.range_constexpr(chunks):
            quad = lane + Int32(32 * chunk)
            if quad < Int32(quads):
                target = floats + Int64(row) * record + Int64(quad * Int32(4))
                if initialized == Int32(0):
                    _rank_store_f32x4(target, acc[4 * chunk], acc[4 * chunk + 1], acc[4 * chunk + 2], acc[4 * chunk + 3])
                else:
                    p0, p1, p2, p3 = _rank_load_f32x4(target)
                    _rank_store_f32x4(target, p0 + acc[4 * chunk], p1 + acc[4 * chunk + 1], p2 + acc[4 * chunk + 2], p3 + acc[4 * chunk + 3])
        if initialized == Int32(0):
            if lane == Int32(0):
                cute.arch.store(high_ptr, Int32(1))

    @cute.jit
    def _rank_mark_slot(self, words: cute.Pointer, state: Int64, dup_list: Int64, counters: Int64, pitch: Int64, parity: Int32, ordinal: Int32, row: Int32):
        """Set ordinal bit ``ordinal`` of ``row`` in the row state of wave slot ``parity``; list the row once if the query repeats it."""
        bit = Uint32(1) << Uint32(ordinal % Int32(32))
        entry = words + state + (Int64(parity) * pitch + Int64(row)) * Int64(RANK_M64_STATE_WORDS)
        previous = cute.arch.atomic_or((entry + Int64(ordinal // Int32(32))).llvm_ptr, bit, sem="relaxed", scope="gpu")
        if (previous & bit) != Uint32(0):
            self._rank_mark_duplicate(words, state, dup_list, counters, pitch, parity, row)

    @cute.jit
    def _rank_mark_duplicate(self, words: cute.Pointer, state: Int64, dup_list: Int64, counters: Int64, pitch: Int64, parity: Int32, row: Int32):
        """Flag ``row`` as selected twice by one query and list it once in the duplicate list of wave slot ``parity``."""
        entry = words + state + (Int64(parity) * pitch + Int64(row)) * Int64(RANK_M64_STATE_WORDS)
        flagged = cute.arch.atomic_or((entry + Int64(RANK_M64_ORDINAL_WORDS)).llvm_ptr, Uint32(1), sem="relaxed", scope="gpu")
        if flagged == Uint32(0):
            counter = cute.recast_ptr(words + counters + Int64(parity * Int32(4) + Int32(2)), dtype=Int32)
            position = cute.arch.atomic_add(counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
            cute.arch.store(cute.recast_ptr(words + dup_list + Int64(parity) * Int64(self.rank_slot_stride()) + Int64(position), dtype=Int32), row)

    @cute.jit
    def rank_mark_next_wave(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        total_q: Int32,
        q_offset: Int32,
        ordinal: Int32,
        lane: Int32,
        part: Int32 = 0,
    ):
        """Record the row ordinals of query ``q_offset + W + ordinal`` for the next wave (one warp; split pairs take alternate 128-slot chunks by ``part``).

        Runs in the idle warp of the wave starting at ``q_offset``, after this wave's
        ``rank_setup_b`` cleared the next wave's row-state slot and reset its duplicate
        list; the next wave's ``rank_setup_b`` reads the bits after this kernel.
        """
        query = self.rank_query(q_offset + Int32(self.q_wave_ctas), ordinal, total_q)
        if query < total_q and q_offset + Int32(self.q_wave_ctas) < total_q:
            pitch, planes = self.rank_geometry(seqlen_kv)
            _, state, _, _, _, dup_list, counters, _, _ = self.rank_offsets(pitch, planes)
            words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
            parity = self.rank_slot3(q_offset + Int32(self.q_wave_ctas))
            topk = _rank_topk(mTopkIdxs, mTopkLength, query)
            # Four slots per lane in flight: independent index loads, then independent atomics.
            bit = Uint32(1) << Uint32(ordinal % Int32(32))
            rows = cute.make_rmem_tensor((4,), Int32)
            previous = cute.make_rmem_tensor((4,), Uint32)
            step = 128
            if cutlass.const_expr(self.rank_split_slots):
                step = 256
            for base in cutlass.range(lane + part * Int32(128), topk, step, unroll=1):
                for b in cutlass.range_constexpr(4):
                    rows[b] = Int32(-1)
                    if base + Int32(32 * b) < topk:
                        row = mTopkIdxs[base + Int32(32 * b), (query, 0)]
                        if row >= Int32(0) and row < seqlen_kv:
                            rows[b] = row
                for b in cutlass.range_constexpr(4):
                    previous[b] = Uint32(0)
                    if rows[b] >= Int32(0):
                        entry = words + state + (Int64(parity) * pitch + Int64(rows[b])) * Int64(RANK_M64_STATE_WORDS) + Int64(ordinal // Int32(32))
                        previous[b] = cute.arch.atomic_or(entry.llvm_ptr, bit, sem="relaxed", scope="gpu")
                for b in cutlass.range_constexpr(4):
                    if rows[b] >= Int32(0):
                        if (previous[b] & bit) != Uint32(0):
                            self._rank_mark_duplicate(words, state, dup_list, counters, pitch, parity, rows[b])

    @cute.kernel
    def rank_init(self, mdKV_acc: cute.Tensor, seqlen_kv: Int32):
        """Clear the three row-state slots, the counters and the initialized-plane counts.

        The planes and the records are written before they are read."""
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = self.rank_geometry(seqlen_kv)
        _, state, _, _, _, _, counters, _, _ = self.rank_offsets(pitch, planes)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        stride = Int64(nblocks) * Int64(RANK_M64_THREADS)
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        state_chunks = Int64(3 * RANK_M64_STATE_WORDS // 4) * pitch
        while chunk < state_chunks:
            _rank_store_zero_x4(words + state + chunk * Int64(4))
            chunk += stride
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        while chunk < pitch // Int64(4):
            _rank_store_zero_x4(words + high + chunk * Int64(4))
            chunk += stride
        if bidx == Int32(0) and tidx < Int32(RANK_M64_COUNTER_WORDS // 4):
            _rank_store_zero_x4(words + counters + Int64(tidx) * Int64(4))
        flags = self.rank_flags(pitch, planes)
        chunk = Int64(bidx) * Int64(RANK_M64_THREADS) + Int64(tidx)
        while chunk < Int64(RANK_M64_FLAG_WORDS):
            cute.arch.store(cute.recast_ptr(words + flags + chunk, dtype=Int32), Int32(0))
            chunk += stride

    @cute.kernel
    def rank_mark_first(self, mdKV_acc: cute.Tensor, mTopkIdxs: cute.Tensor, mTopkLength: Optional[cute.Tensor], seqlen_kv: Int32, total_q: Int32):
        """Record the row ordinals of the first wave (each main kernel records the next wave's).

        A programmatic dependent of ``rank_init`` (waits for its clears); ``sum_OdO`` runs beside it."""
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = self.rank_geometry(seqlen_kv)
        _, state, _, _, _, dup_list, counters, _, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        topk_width = Int32(self.max_topk)
        slot_index = bidx * Int32(RANK_M64_THREADS) + tidx
        while slot_index < self.rank_wave_size(Int32(0), total_q) * topk_width:
            ordinal = slot_index // topk_width
            slot = slot_index - ordinal * topk_width
            query = self.rank_query(Int32(0), ordinal, total_q)
            row = mTopkIdxs[slot, (query, 0)]
            if slot < _rank_topk(mTopkIdxs, mTopkLength, query):
                if row >= Int32(0) and row < seqlen_kv:
                    self._rank_mark_slot(words, state, dup_list, counters, pitch, Int32(0), ordinal, row)
            slot_index += nblocks * Int32(RANK_M64_THREADS)

    @cute.kernel
    def rank_setup_a(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        cur_begin: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        total_q: Int32,
    ):
        """Finish the previous wave's records (``rank_setup_b`` runs beside it).

        Sums the previous wave's overflow segments: single-segment rows go straight to
        plane 0; a multi-segment row leaves each segment sum in the segment's first
        record (``rank_setup_b`` merges them after this grid). Adds the previous wave's
        duplicate records to plane 0.
        Reads only the previous wave's lists, counters and records, which
        ``rank_setup_b`` (running concurrently) does not write.
        """
        # ``rank_setup_b`` is a programmatic dependent that runs beside this grid.
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = self.rank_geometry(seqlen_kv)
        records, state, offsets, dup_records, row_list, dup_list, counters, codes, segment_list = self.rank_offsets(pitch, planes)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        floats = mdKV_acc.iterator
        record = Int64(rank_m64_record(self.head_dim))
        slots = Int64(self.rank_slot_stride())
        cur_parity = self.rank_parity(cur_begin)
        prev_parity = Int32(1) - cur_parity
        cur_slot3 = self.rank_slot3(cur_begin)
        prev_slot3 = (cur_slot3 + Int32(2)) % Int32(3)
        thread = bidx * Int32(RANK_M64_THREADS) + tidx
        nthreads = nblocks * Int32(RANK_M64_THREADS)
        warp = thread // Int32(32)
        nwarps = nthreads // Int32(32)
        lane = tidx % Int32(32)
        if prev_size > Int32(0):
            prev_segments = words + segment_list + Int64(prev_parity) * Int64(4) * slots
            prev_segment_count = _rank_counter_field(words, counters, prev_slot3, 2)
            item = warp
            while item < prev_segment_count:
                entry = cute.recast_ptr(prev_segments + Int64(item) * Int64(4), dtype=Int32)
                row = cute.arch.load(entry, Int32)
                first = cute.arch.load(entry + 1, Int32)
                count = cute.arch.load(entry + 2, Int32)
                link = cute.arch.load(entry + 3, Int32)
                single = Int32(0)
                if link < Int32(0):
                    single = Int32(1)
                self._rank_add_records(floats, Int64(row), records // record + Int64(first), count, Int64(1), single, lane, RANK_M64_TAIL_BATCH)
                item += nwarps
            duplicate_rows = cute.arch.load(cute.recast_ptr(words + counters + Int64(prev_slot3 * Int32(4) + Int32(2)), dtype=Int32), Int32)
            item = warp
            while item < duplicate_rows:
                row = cute.arch.load(cute.recast_ptr(words + dup_list + Int64(prev_slot3) * slots + Int64(item), dtype=Int32), Int32)
                self._rank_add_duplicate_row(
                    floats, words, mTopkIdxs, mTopkLength, records, dup_records + Int64(prev_parity) * slots, high, row, prev_begin, prev_size, lane, total_q
                )
                item += nwarps

    @cute.kernel
    def rank_setup_b(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        prev_size: Int32,
        total_q: Int32,
    ):
        """Assign the current wave while ``rank_setup_a`` finishes the previous one.

        - Raises the initialized-plane counts with the previous wave's writers (its
          duplicated rows only initialize plane 0, in ``rank_setup_a``) and clears the
          row state of the wave before it, whose slot the next wave reuses.
        - Allocates the current wave's records and segments and writes its destination
          codes. A slot of rank ``k`` is fresh while plane ``k`` of its row is
          uninitialized, counting the previous wave's writers (whose counts may be
          raised concurrently).

        - After ``rank_setup_a`` (its programmatic primary) finished, adds each
          multi-segment row's segment sums to plane 0 in segment order.

        Row state and counters rotate over three wave slots and the record lists over
        two parities, so until that wait nothing here touches data ``rank_setup_a``
        reads or writes; the main kernel's wait for this grid covers both.
        """
        # The wave's main kernel is a programmatic dependent: let it be scheduled early;
        # its reducers wait for this grid before reading destination codes.
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = self.rank_geometry(seqlen_kv)
        records, state, offsets, dup_records, row_list, dup_list, counters, codes, segment_list = self.rank_offsets(pitch, planes)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        floats = mdKV_acc.iterator
        record = Int64(rank_m64_record(self.head_dim))
        topk_width = Int32(self.max_topk)
        slots = Int64(self.rank_slot_stride())
        cur_parity = self.rank_parity(cur_begin)
        prev_parity = Int32(1) - cur_parity
        cur_slot3 = self.rank_slot3(cur_begin)
        prev_slot3 = (cur_slot3 + Int32(2)) % Int32(3)
        old_slot3 = (cur_slot3 + Int32(1)) % Int32(3)
        thread = bidx * Int32(RANK_M64_THREADS) + tidx
        nthreads = nblocks * Int32(RANK_M64_THREADS)
        warp = thread // Int32(32)
        nwarps = nthreads // Int32(32)
        lane = tidx % Int32(32)
        prev_state = words + state + Int64(prev_slot3) * pitch * Int64(RANK_M64_STATE_WORDS)
        cur_state = words + state + Int64(cur_slot3) * pitch * Int64(RANK_M64_STATE_WORDS)
        # Each lane loads its first slot's index before the row pass, then one iteration ahead.
        total_slots = cur_size * topk_width
        upcoming = Int32(-1)
        if thread < total_slots:
            upcoming = mTopkIdxs[thread % topk_width, (self.rank_query(cur_begin, thread // topk_width, total_q), 0)]
        # Rows: raise the initialized-plane counts with the previous wave's writers (its
        # duplicated rows only initialize plane 0, above) and clear the older slot's state.
        old_state = words + state + Int64(old_slot3) * pitch * Int64(RANK_M64_STATE_WORDS)
        row = thread
        while row < seqlen_kv:
            if prev_size > Int32(0):
                entry = prev_state + Int64(row) * Int64(RANK_M64_STATE_WORDS)
                high_ptr = cute.recast_ptr(words + high + Int64(row), dtype=Int32)
                state_0, state_1, state_2, state_3 = _rank_load_u32x4(entry)
                state_4, duplicate, _, _ = _rank_load_u32x4(entry + Int64(4))
                initialized = cute.arch.load(high_ptr, Int32)
                count = Int32(cute.arch.popc(state_0) + cute.arch.popc(state_1) + cute.arch.popc(state_2) + cute.arch.popc(state_3) + cute.arch.popc(state_4))
                if count > Int32(0) and duplicate == Uint32(0):
                    if min(count, planes) > initialized:
                        cute.arch.store(high_ptr, min(count, planes))
            _rank_store_zero_x4(old_state + Int64(row) * Int64(RANK_M64_STATE_WORDS))
            _rank_store_zero_x4(old_state + Int64(row) * Int64(RANK_M64_STATE_WORDS) + Int64(4))
            row += nthreads
        if thread == Int32(0):
            cute.arch.store(cute.recast_ptr(words + counters + Int64(old_slot3 * Int32(4)), dtype=Int64), Int64(0))
            cute.arch.store(cute.recast_ptr(words + counters + Int64(old_slot3 * Int32(4) + Int32(2)), dtype=Int64), Int64(0))
        packed_counter = words + counters + Int64(cur_slot3 * Int32(4))
        multi_counter = cute.recast_ptr(packed_counter + Int64(3), dtype=Int32)
        cur_rows = words + row_list + Int64(cur_parity) * Int64(4) * slots
        cur_segments = words + segment_list + Int64(cur_parity) * Int64(4) * slots
        cur_dup_records = words + dup_records + Int64(cur_parity) * slots
        # Warp-uniform slot loop: every lane of a warp runs the same iterations so the
        # record, row and segment reservations can be aggregated with one atomic per warp.
        base = thread - lane
        while base < total_slots:
            slot_index = base + lane
            row = Int32(-1)
            duplicate_slot = Int32(0)
            overflow = Int32(0)
            candidate = upcoming
            following = slot_index + nthreads
            if following < total_slots:
                upcoming = mTopkIdxs[following % topk_width, (self.rank_query(cur_begin, following // topk_width, total_q), 0)]
            if slot_index < total_slots:
                ordinal = slot_index // topk_width
                slot = slot_index - ordinal * topk_width
                query = self.rank_query(cur_begin, ordinal, total_q)
                # The row-state loads do not wait for the length check.
                state_0, state_1, state_2, state_3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                state_4, duplicate = Uint32(0), Uint32(0)
                prev_0, prev_1, prev_2, prev_3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                prev_4, prev_duplicate = Uint32(0), Uint32(0)
                initialized = Int32(0)
                if candidate >= Int32(0) and candidate < seqlen_kv:
                    entry = cur_state + Int64(candidate) * Int64(RANK_M64_STATE_WORDS)
                    state_0, state_1, state_2, state_3 = _rank_load_u32x4(entry)
                    state_4, duplicate, _, _ = _rank_load_u32x4(entry + Int64(4))
                    prev_entry = prev_state + Int64(candidate) * Int64(RANK_M64_STATE_WORDS)
                    prev_0, prev_1, prev_2, prev_3 = _rank_load_u32x4(prev_entry)
                    prev_4, prev_duplicate, _, _ = _rank_load_u32x4(prev_entry + Int64(4))
                    initialized = cute.arch.load(cute.recast_ptr(words + high + Int64(candidate), dtype=Int32), Int32)
                if slot < _rank_topk(mTopkIdxs, mTopkLength, query):
                    if candidate >= Int32(0) and candidate < seqlen_kv:
                        row = candidate
                        if duplicate != Uint32(0):
                            duplicate_slot = Int32(1)
                        else:
                            prev_count = Int32(
                                cute.arch.popc(prev_0) + cute.arch.popc(prev_1) + cute.arch.popc(prev_2) + cute.arch.popc(prev_3) + cute.arch.popc(prev_4)
                            )
                            if prev_count > Int32(0):
                                if prev_duplicate != Uint32(0):
                                    initialized = max(initialized, Int32(1))
                                else:
                                    initialized = max(initialized, min(prev_count, planes))
                            count = Int32(
                                cute.arch.popc(state_0) + cute.arch.popc(state_1) + cute.arch.popc(state_2) + cute.arch.popc(state_3) + cute.arch.popc(state_4)
                            )
                            conflict = Int32(
                                cute.arch.popc(state_0 & _rank_lower_mask(ordinal, 0))
                                + cute.arch.popc(state_1 & _rank_lower_mask(ordinal, 1))
                                + cute.arch.popc(state_2 & _rank_lower_mask(ordinal, 2))
                                + cute.arch.popc(state_3 & _rank_lower_mask(ordinal, 3))
                                + cute.arch.popc(state_4 & _rank_lower_mask(ordinal, 4))
                            )
                            code_ptr = cute.recast_ptr(words + codes + Int64(slot_index), dtype=Int32)
                            if conflict < planes:
                                code = conflict * Int32(pitch) + row
                                if conflict >= initialized:
                                    code = code | Int32(RANK_M64_FRESH_FLAG)
                                cute.arch.store(code_ptr, code)
                            else:
                                cute.arch.store(code_ptr, Int32(-2) - (conflict - planes))
                            if count > planes and conflict == planes:
                                overflow = count - planes
            segments = (overflow + Int32(self.rank_segment - 1)) // Int32(self.rank_segment)
            row_needed = Int32(0)
            if segments == Int32(1):
                row_needed = Int32(1)
            first = Int32(0)
            position = Int32(0)
            segment = Int32(0)
            # Most warps reserve nothing: skip the warp-wide scan then (the vote is warp-uniform).
            if Uint32(cute.arch.vote_ballot_sync(overflow + duplicate_slot > Int32(0))) != Uint32(0):
                first, position, segment = _rank_warp_allocate3(packed_counter, overflow + duplicate_slot, row_needed, segments, lane)
            if segments > Int32(1):
                # Multi-segment rows fill the row list from its end; the merge visits only them.
                position = Int32(slots) - Int32(1) - cute.arch.atomic_add(multi_counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
            if duplicate_slot != Int32(0):
                cute.arch.store(cute.recast_ptr(cur_dup_records + Int64(slot_index), dtype=Int32), first)
                cute.arch.store(cute.recast_ptr(words + codes + Int64(slot_index), dtype=Int32), (planes * Int32(pitch) + first) | Int32(RANK_M64_FRESH_FLAG))
            if overflow > Int32(0):
                list_entry = cute.recast_ptr(cur_rows + Int64(position) * Int64(4), dtype=Int32)
                cute.arch.store(list_entry, row)
                cute.arch.store(list_entry + 1, first)
                cute.arch.store(list_entry + 2, overflow)
                cute.arch.store(cute.recast_ptr(words + offsets + Int64(row), dtype=Int32), first)
                link = Int32(0)
                if segments == Int32(1):
                    link = Int32(-1)
                for index in cutlass.range(segments, unroll=1):
                    segment_entry = cute.recast_ptr(cur_segments + Int64(segment + index) * Int64(4), dtype=Int32)
                    cute.arch.store(segment_entry, row)
                    cute.arch.store(segment_entry + 1, first + index * Int32(self.rank_segment))
                    cute.arch.store(segment_entry + 2, min(Int32(self.rank_segment), overflow - index * Int32(self.rank_segment)))
                    cute.arch.store(segment_entry + 3, link)
            base += nthreads
        # After ``rank_setup_a``, add each multi-segment row's segment sums to plane 0 in segment order,
        # one warp per (row, 32-quad column slice). A warp's first item (row-list entry, plane-0 slice;
        # ``rank_setup_a`` writes neither) is fetched before the wait.
        prev_rows = words + row_list + Int64(prev_parity) * Int64(4) * slots
        quads = rank_m64_record(self.head_dim) // 4
        column_slices = (quads + 31) // 32
        items = Int32(0)
        if prev_size > Int32(0):
            items = cute.arch.load(cute.recast_ptr(words + counters + Int64(prev_slot3 * Int32(4) + Int32(3)), dtype=Int32), Int32) * Int32(column_slices)
        row = Int32(0)
        row_first = Int32(0)
        row_segments = Int32(0)
        p0 = Float32(0.0)
        p1 = Float32(0.0)
        p2 = Float32(0.0)
        p3 = Float32(0.0)
        item = warp
        if item < items:
            multi = item // Int32(column_slices)
            quad = (item - multi * Int32(column_slices)) * Int32(32) + lane
            row_entry = cute.recast_ptr(prev_rows + (slots - Int64(1) - Int64(multi)) * Int64(4), dtype=Int32)
            row = cute.arch.load(row_entry, Int32)
            row_first = cute.arch.load(row_entry + 1, Int32)
            row_segments = (cute.arch.load(row_entry + 2, Int32) + Int32(self.rank_segment - 1)) // Int32(self.rank_segment)
            if quad < Int32(quads):
                p0, p1, p2, p3 = _rank_load_f32x4(floats + Int64(row) * record + Int64(quad * Int32(4)))
        # Only CTAs with merge items and CTA 0 (this grid ends after ``rank_setup_a``) wait.
        if bidx * Int32(RANK_M64_THREADS // 32) < items or bidx == Int32(0):
            cute.arch.griddepcontrol_wait()
        sums = cute.make_rmem_tensor((32,), Float32)
        while item < items:
            multi = item // Int32(column_slices)
            quad = (item - multi * Int32(column_slices)) * Int32(32) + lane
            if item != warp:
                row_entry = cute.recast_ptr(prev_rows + (slots - Int64(1) - Int64(multi)) * Int64(4), dtype=Int32)
                row = cute.arch.load(row_entry, Int32)
                row_first = cute.arch.load(row_entry + 1, Int32)
                row_segments = (cute.arch.load(row_entry + 2, Int32) + Int32(self.rank_segment - 1)) // Int32(self.rank_segment)
            if quad < Int32(quads):
                target = floats + Int64(row) * record + Int64(quad * Int32(4))
                acc_0 = Float32(0.0)
                acc_1 = Float32(0.0)
                acc_2 = Float32(0.0)
                acc_3 = Float32(0.0)
                base = Int32(0)
                while base < row_segments:
                    for b in cutlass.range_constexpr(8):
                        if base + Int32(b) < row_segments:
                            source = (
                                floats + (records // record + Int64(row_first + (base + Int32(b)) * Int32(self.rank_segment))) * record + Int64(quad * Int32(4))
                            )
                            v0, v1, v2, v3 = _rank_load_once_f32x4(source)
                            sums[4 * b] = v0
                            sums[4 * b + 1] = v1
                            sums[4 * b + 2] = v2
                            sums[4 * b + 3] = v3
                    for b in cutlass.range_constexpr(8):
                        if base + Int32(b) < row_segments:
                            acc_0 = acc_0 + sums[4 * b]
                            acc_1 = acc_1 + sums[4 * b + 1]
                            acc_2 = acc_2 + sums[4 * b + 2]
                            acc_3 = acc_3 + sums[4 * b + 3]
                    base += Int32(8)
                if item != warp:
                    p0, p1, p2, p3 = _rank_load_f32x4(target)
                _rank_store_f32x4(target, p0 + acc_0, p1 + acc_1, p2 + acc_2, p3 + acc_3)
            item += nwarps

    @cute.kernel
    def rank_fold_convert(self, mdKV_acc: cute.Tensor, mdKV: cute.Tensor, seqlen_kv: Int32):
        """Fold each row's initialized planes in ascending plane order and store the output dtype.

        Plane 0 first, then planes ``1 .. initialized - 1``; the folded row is
        converted in registers instead of being written back and read again. The
        column permutation matches the reducers' FP32 layout (see ``convert``).
        """
        cute.arch.griddepcontrol_launch_dependents()
        tidx, tidy, _ = cute.arch.thread_idx()
        seq_block_idx, _, batch_idx = cute.arch.block_idx()
        seq_id = self.block_seq * seq_block_idx + tidy
        pitch, planes = self.rank_geometry(seqlen_kv)
        high = self.rank_high(pitch, planes)
        words = cute.recast_ptr(mdKV_acc.iterator, dtype=Uint32)
        num_128_tiles = self.head_dim_main // 64
        num_terms = 2 * num_128_tiles + (0 if self.same_hdim_kv else 2)
        if seq_id < seqlen_kv:
            written = cute.arch.load(cute.recast_ptr(words + high + Int64(seq_id), dtype=Int32), Int32)
            acc = cute.make_rmem_tensor((num_terms,), Float32)
            acc.fill(0.0)
            # A row no slot ever wrote has no initialized plane (plane 0 is not cleared): it folds to zero.
            if written > Int32(0):
                first_row = mdKV_acc[None, seq_id, (0, batch_idx)]
                first_tile = cute.flat_divide(cute.flat_divide(first_row, (64,)), (32,))
                for i in cutlass.range_constexpr(num_128_tiles):
                    for j in cutlass.range_constexpr(2):
                        acc[2 * i + j] = first_tile[tidx, j, i]
                if cutlass.const_expr(not self.same_hdim_kv):
                    for j in cutlass.range_constexpr(2):
                        acc[2 * num_128_tiles + j] = first_tile[tidx, j, num_128_tiles]
            # Two planes' loads are in flight per step; they are still added in plane order.
            pair = cute.make_rmem_tensor((2 * num_terms,), Float32)
            plane = Int32(1)
            while plane + Int32(1) < written:
                for half in cutlass.range_constexpr(2):
                    source_row = mdKV_acc[None, (plane + Int32(half)) * Int32(pitch) + seq_id, (0, batch_idx)]
                    tile = cute.flat_divide(cute.flat_divide(source_row, (64,)), (32,))
                    for i in cutlass.range_constexpr(num_128_tiles):
                        for j in cutlass.range_constexpr(2):
                            pair[half * num_terms + 2 * i + j] = tile[tidx, j, i]
                    if cutlass.const_expr(not self.same_hdim_kv):
                        for j in cutlass.range_constexpr(2):
                            pair[half * num_terms + 2 * num_128_tiles + j] = tile[tidx, j, num_128_tiles]
                for half in cutlass.range_constexpr(2):
                    for t in cutlass.range_constexpr(num_terms):
                        acc[t] = acc[t] + pair[half * num_terms + t]
                plane += Int32(2)
            if plane < written:
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
        self.rank_fold_convert(mdKV_acc, mdKV, seqlen_kv).launch(
            grid=[cute.ceil_div(seqlen_kv, self.block_seq), 1, 1],
            block=[self.num_threads_D_convert, self.num_threads_seq, 1],
            stream=stream,
        )

    @cute.jit
    def launch_rank_init(self, mdKV_acc: cute.Tensor, seqlen_kv: Int32, stream: cuda.CUstream):
        """Clear the conflict-rank workspace once per execution."""
        self.rank_init(mdKV_acc, seqlen_kv).launch(grid=[RANK_M64_SETUP_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream)

    @cute.jit
    def launch_rank_wave(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        q_offset: Int32,
        total_q: Int32,
        stream: cuda.CUstream,
    ):
        """Finish the previous wave, then assign the wave that starts at ``q_offset`` (its ordinals are recorded)."""
        prev_size = min(q_offset, Int32(self.q_wave_ctas))
        prev_begin = q_offset - prev_size
        prev_size = min(prev_size, self.rank_wave_size(prev_begin, total_q))
        self.rank_setup_a(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, q_offset, prev_begin, prev_size, total_q).launch(
            grid=[RANK_M64_TAIL_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream
        )
        self.rank_setup_b(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, q_offset, self.rank_wave_size(q_offset, total_q), prev_size, total_q).launch(
            grid=[self.rank_assign_ctas(), 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.jit
    def launch_rank_mark_first(
        self, mdKV_acc: cute.Tensor, mTopkIdxs: cute.Tensor, mTopkLength: Optional[cute.Tensor], seqlen_kv: Int32, total_q: Int32, stream: cuda.CUstream
    ):
        """Record the first wave's row ordinals after the initialization."""
        self.rank_mark_first(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, total_q).launch(
            grid=[RANK_M64_SETUP_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.jit
    def launch_rank_final(
        self,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        total_q: Int32,
        stream: cuda.CUstream,
    ):
        """Add the last wave's records to plane 0 and raise its initialized-plane counts (``finalize_dKV``
        folds the planes while converting)."""
        last_begin = (total_q - Int32(1)) // Int32(self.q_wave_ctas) * Int32(self.q_wave_ctas)
        last_size = self.rank_wave_size(last_begin, total_q)
        self.rank_setup_a(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, last_begin + Int32(self.q_wave_ctas), last_begin, last_size, total_q).launch(
            grid=[RANK_M64_TAIL_CTAS, 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream
        )
        self.rank_setup_b(mdKV_acc, mTopkIdxs, mTopkLength, seqlen_kv, last_begin + Int32(self.q_wave_ctas), Int32(0), last_size, total_q).launch(
            grid=[self.rank_assign_ctas(), 1, 1], block=[RANK_M64_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.jit
    def reduce_dKV_records(
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
        slot_begin: Int32 = 0,
    ):
        """Conflict-rank records plans: reduce dKV fragments into planes and records (``slot_begin``: first top-k slot of ``mTopkIdxs``)."""
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

        tile_index = tile_count - 1
        rTopkIdx = cute.make_rmem_tensor((8,), cutlass.Int32)
        demote = Int32(0)
        if head_block_idx == cute.ceil_div(problem_shape[3][0], self.block_tile) - 1:
            demote = Int32(1)
        blocking = problem_shape[3][0] < Int32(64)
        full_tiles = (topk % self.block_tile) == 0
        cute.arch.griddepcontrol_wait()
        query_codes, row_offsets, record_base = self.rank_query_tables(mdKV_acc_full, max_seqlen_kv, token_idx, problem_shape[0])
        if cutlass.const_expr(self.rank_split_slots):
            query_codes = query_codes + slot_begin
        if cutlass.const_expr(self.rank_pair_heads):
            pair_pitch, pair_planes = self.rank_geometry(max_seqlen_kv)
            pair_ordinal, pair_wave = self.rank_ordinal(token_idx, problem_shape[0])
            pair_flag = cute.recast_ptr(
                cute.recast_ptr(mdKV_acc_full.iterator, dtype=Int32)
                + self.rank_flags(pair_pitch, pair_planes)
                + Int64(pair_ordinal * Int32(RANK_M64_PAIR_THREADS) + tidx_in_wg),
                dtype=Int32,
            )
            pair_base = pair_wave * Int32((self.max_topk + self.block_tile - 1) // self.block_tile + 1) + tile_count
        # Each tile's indices, destination codes and record offsets are loaded one tile ahead.
        rNextIdx = cute.make_rmem_tensor((8,), cutlass.Int32)
        rNextCode = cute.make_rmem_tensor((8,), cutlass.Int32)
        rNextFirst = cute.make_rmem_tensor((8,), cutlass.Int32)
        self._prefetch_dKV_destinations(mTopkIdxs, query_codes, tTR_cdKV, tile_index, topk, full_tiles, token_idx, batch_idx, rNextIdx, rNextCode)
        self._prefetch_record_offsets(row_offsets, max_seqlen_kv, rNextIdx, rNextCode, rNextFirst)
        early_release = not (self.same_hdim_kv and self.rank_pair_heads)
        while tile_index >= 0:
            for i in cutlass.range_constexpr(8):
                topk_idx = rNextIdx[i]
                rTopkIdx[i] = Int32(-1)
                if topk_idx >= Int32(0) and topk_idx < max_seqlen_kv:
                    code = rNextCode[i]
                    destination = code
                    if code < Int32(0):
                        destination = (record_base + rNextFirst[i] + (Int32(-2) - code)) | Int32(RANK_M64_FRESH_FLAG)
                    if head_block_idx != Int32(0):
                        destination = destination & Int32(RANK_M64_FRESH_FLAG - 1)
                    rTopkIdx[i] = destination
                    if cute.get(tTR_cdKV[i * 2 - i % 2], mode=[0]) == 0:
                        if destination < Int32(RANK_M64_FRESH_FLAG) and (not self.rank_pair_heads or head_block_idx == Int32(0)):
                            _rank_prefetch_l2(mdKV_acc.iterator + Int64(destination) * Int64(self.head_dim), self.head_dim * 4)
            next_tile = tile_index - 1
            if next_tile < Int32(0):
                next_tile = Int32(0)
            self._prefetch_dKV_destinations(mTopkIdxs, query_codes, tTR_cdKV, next_tile, topk, full_tiles, token_idx, batch_idx, rNextIdx, rNextCode)

            self.acquire_dKV_tile_relay(
                mdKV_acc_full,
                mTopkIdxs,
                max_seqlen_kv,
                tile_index,
                topk,
                token_idx,
                head_block_idx,
                problem_shape,
            )

            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

            rdKV0 = self.t2r_dKV(tdKVtdKV0)
            rdKV1 = self.t2r_dKV(tdKVtdKV1)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.same_hdim_kv and not self.rank_pair_heads):
                _partial_cta_sync(self.t2r_dKV01_done_barrier.barrier_id, self.t2r_dKV01_done_barrier.num_threads)
            elif blocking:
                _partial_cta_sync(self.t2r_dKV01_done_barrier.barrier_id, self.t2r_dKV01_done_barrier.num_threads)
            else:
                self.t2r_dKV01_done_barrier.arrive_unaligned()
            # Conflict-rank plans free the dKV0/dKV1 stage as soon as it is in registers (paired D512 keeps the
            # release after its adds), ahead of the destination loads and release stores below.
            if cutlass.const_expr(early_release):
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)

            if cutlass.const_expr(self.same_hdim_kv):
                self._prefetch_record_offsets(row_offsets, max_seqlen_kv, rNextIdx, rNextCode, rNextFirst)
                if cutlass.const_expr(self.rank_pair_heads):
                    if head_block_idx == Int32(0) and tile_index < tile_count - Int32(1):
                        _rank_flag_release(pair_flag, pair_base - tile_index - Int32(1))
                if cutlass.const_expr(self.rank_pair_heads):
                    self._rank_pair_wait(pair_flag, pair_base - tile_index, head_block_idx)
                self._reduce_dKV_d512_pair(mdKV_acc, rdKV0, rdKV1, rTopkIdx, 0, demote)

            if cutlass.const_expr(not early_release):
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
            mma_reduce_dKV_consumer_state.advance()

            if cutlass.const_expr(not self.same_hdim_kv):
                mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

                rdKV4 = self.t2r_dKV_64(tdKVtdKV4)
                cute.arch.fence_view_async_tmem_load()
                if blocking:
                    _partial_cta_sync(self.t2r_dKV4_done_barrier.barrier_id, self.t2r_dKV4_done_barrier.num_threads)
                else:
                    self.t2r_dKV4_done_barrier.arrive_unaligned()

                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                # After the dKV4 handoff (it gates the next tile's S MMA): the next tile's record offsets and
                # head block 0 of a pair publishing the previous tile's adds (the release waits for them).
                self._prefetch_record_offsets(row_offsets, max_seqlen_kv, rNextIdx, rNextCode, rNextFirst)
                if cutlass.const_expr(self.rank_pair_heads):
                    if head_block_idx == Int32(0) and tile_index < tile_count - Int32(1):
                        _rank_flag_release(pair_flag, pair_base - tile_index - Int32(1))

                if cutlass.const_expr(self.rank_pair_heads):
                    self._rank_pair_wait(pair_flag, pair_base - tile_index, head_block_idx)
                self.reduce_dKV_0_tail_1_from_reg(mdKV_acc, rdKV0, rdKV4, rdKV1, rTopkIdx, demote)

            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

            if cutlass.const_expr(self.same_hdim_kv):
                rdKV2 = self.t2r_dKV(tdKVtdKV2)
                rdKV3 = self.t2r_dKV(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(self.rank_pair_heads):
                    self.t2r_dKV4_done_barrier.arrive_unaligned()
                else:
                    _partial_cta_sync(self.t2r_dKV4_done_barrier.barrier_id, self.t2r_dKV4_done_barrier.num_threads)
                if cutlass.const_expr(early_release):
                    mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
            else:
                rdKV2 = self.t2r_dKV(tdKVtdKV2)
                rdKV3 = self.t2r_dKV(tdKVtdKV3)
                cute.arch.fence_view_async_tmem_load()
                if blocking:
                    _partial_cta_sync(self.t2r_dKV23_done_barrier.barrier_id, self.t2r_dKV23_done_barrier.num_threads)
                else:
                    self.t2r_dKV23_done_barrier.arrive_unaligned()

            if cutlass.const_expr(self.same_hdim_kv):
                self._reduce_dKV_d512_pair(mdKV_acc, rdKV2, rdKV3, rTopkIdx, 2, demote)
            else:
                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()
                self.reduce_dKV_pair_from_reg(mdKV_acc, rdKV2, rdKV3, rTopkIdx, 2, 3, demote)

            if cutlass.const_expr(self.same_hdim_kv):
                if cutlass.const_expr(not early_release):
                    mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()

            self.release_dKV_tile_relay(
                mdKV_acc_full,
                mTopkIdxs,
                max_seqlen_kv,
                tile_index,
                topk,
                token_idx,
                head_block_idx,
                problem_shape,
            )

            tile_index -= 1
        if cutlass.const_expr(self.rank_pair_heads):
            if head_block_idx == Int32(0):
                _rank_flag_release(pair_flag, pair_base)

    @cute.jit
    def _rank_pair_wait(self, pair_flag: cute.Pointer, target: Int32, head_block_idx: Int32):
        """Head block 1 of a pair: wait until head block 0 published this tile's adds (same reducer thread, same rows)."""
        if head_block_idx != Int32(0):
            observed = _rank_flag_acquire(pair_flag)
            while observed < target:
                observed = _rank_flag_acquire(pair_flag)

    @cute.jit
    def _prefetch_record_offsets(
        self,
        row_offsets: cute.Pointer,
        max_seqlen_kv: Int32,
        rNextIdx: cute.Tensor,
        rNextCode: cute.Tensor,
        rNextFirst: cute.Tensor,
    ):
        """Load the first-record offset of every record slot among the next tile's eight reducer rows (conflict-rank plans)."""
        for i in cutlass.range_constexpr(8):
            topk_idx = rNextIdx[i]
            rNextFirst[i] = Int32(0)
            if topk_idx >= Int32(0) and topk_idx < max_seqlen_kv:
                if rNextCode[i] < Int32(0):
                    rNextFirst[i] = cute.arch.load(row_offsets + Int64(topk_idx), Int32)

    @cute.jit
    def _reduce_dKV_d512_pair(self, dKV_acc: cute.Tensor, rdKV0: cute.Tensor, rdKV1: cute.Tensor, rTopkIdx: cute.Tensor, first: int, demote: Int32):
        """D512: reduce sub-tiles ``first`` and ``first + 1``; unpaired conflict-rank plans use one address loop
        and one reducer-group barrier for both."""
        if cutlass.const_expr(self.rank_shard and not self.rank_pair_heads):
            self.reduce_dKV_pair_from_reg(dKV_acc, rdKV0, rdKV1, rTopkIdx, first, first + 1, demote)
            _partial_cta_sync(self.reduce_sync_barrier.barrier_id, self.reduce_sync_barrier.num_threads)
        else:
            self.reduce_dKV_from_reg(dKV_acc, rdKV0, rTopkIdx, first, demote)
            self.reduce_dKV_from_reg(dKV_acc, rdKV1, rTopkIdx, first + 1, demote)

    @cute.jit
    def _split_dq_exchange(self, values: cute.Tensor, dq_exchange, chunk_base: int):
        """Split pairs: rank 1 saves its FP32 dQ partial; rank 0 adds it to its own (lower tiles) before storing dQ."""
        if dq_exchange[3] != Int32(0):
            for c in cutlass.range_constexpr(cute.size(values) // 4):
                _rank_store_f32x4(dq_exchange[1] + Int64((chunk_base + c) * 512), values[4 * c], values[4 * c + 1], values[4 * c + 2], values[4 * c + 3])
        elif dq_exchange[4] > Int32(0):
            peer = cute.make_rmem_tensor((32,), Float32)
            for b in cutlass.range_constexpr(cute.size(values) // 32):
                for c in cutlass.range_constexpr(8):
                    peer[4 * c], peer[4 * c + 1], peer[4 * c + 2], peer[4 * c + 3] = _rank_load_once_f32x4(
                        dq_exchange[1] + Int64((chunk_base + 8 * b + c) * 512)
                    )
                for j in cutlass.range_constexpr(32):
                    values[32 * b + j] = values[32 * b + j] + peer[j]
