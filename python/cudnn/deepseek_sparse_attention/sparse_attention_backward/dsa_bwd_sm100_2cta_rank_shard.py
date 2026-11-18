# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conflict-rank dKV storage for deterministic two-CTA DSA backward (H128 D512).

A query wave of ``W`` query pairs accumulates FP32 dKV records. Within one
wave, the queries that select a KV row are ranked by wave-local ordinal. The
query with conflict rank ``k < P`` writes dense plane ``k``; higher ranks
store their record into a compact overflow area. Each (plane, row) and each
overflow record therefore has one writer per wave, so the main kernel needs no
writer ordering. A per-wave tail adds overflow records into plane 0 in rank
order, and a fixed-order fold combines the planes each row used.

Plane 0 is zero-filled once per execution. A per-row plane-use count records
how many dense planes a row has initialized so far; a writer whose plane is not
yet initialized stores its record instead of adding it, so planes above 0 need
no clearing and the fold reads only the planes a row used. The count lives in
the first 64-bit word of the row's plane-use entry: the low half holds the step
that last raised it and the high half ``(count before that step << 16) | count
after it``, so every reader of one setup pass sees the same pre-wave count.

Rows selected more than once by one query use per-slot overflow records (every
slot of the wave that selects such a row). The tail adds them in ascending
(query, slot) order.

Plans. Wave ``w``'s setup clears wave ``w - 1``'s row state, assigns wave
``w``'s destination codes and records wave ``w + 1``'s ordinal bits (the slot
phases, ``_rank_setup_slots``), and adds wave ``w - 1``'s overflow and
duplicate records to plane 0 (the tail, ``_rank_tail``).

- Per-wave-setup plans run one setup launch (``rank_wave_setup``) before each
  main kernel, after a leading launch that records wave 0's ordinals.
- Fused plans (``rank_merged_fused``: ``K <= RANK_FUSED_MAX_TOPK`` with at
  least ``RANK_FUSED_MIN_PLANES`` planes) run two leading setup launches (wave
  0's ordinals, then wave 0's codes and wave 1's ordinals). Main kernel ``w``
  then runs wave ``w - 1``'s tail in its reducer warps before they drain, and
  its setup warp runs the slot phases that prepare wave ``w + 1`` (clearing wave
  ``w``'s row state and recording wave ``w + 2``'s ordinals). The tail of wave
  ``w - 1`` and the drains of wave ``w`` touch plane 0 concurrently, so a row
  whose wave ``w - 1`` left a tail shifts wave ``w``'s ranks up by one (off
  plane 0); the plane-use entry's mark word ``(step << 1) | shift`` records
  that, and overflow records keep two wave parities.
- ``rank_final`` adds the last wave's tail, and ``fold_rank_convert_dkv``
  folds and converts the planes.

Row state is triple-buffered by wave. Codes and overflow offsets use two wave
parities; duplicate records, overflow row lists and duplicate row lists use
``RANK_LIST_PARITIES``; counters rotate over ``RANK_COUNTER_BLOCKS`` blocks.

Launch ordering (programmatic dependent launch). Every launch after
``rank_init`` (setups, main kernels, ``rank_final``, the fold) is a
programmatic dependent of the previous one.

- A setup launch triggers its dependents at entry. Early launches (the leading
  setups and wave 0's setup) wait before their slot phases, which read state
  that the launch right before them wrote (``rank_init`` clears it, a leading
  setup records ordinals into it). A later setup runs its slot phases before the
  wait: they read only buffers that earlier setup launches wrote, the latest of
  which completed before the preceding main kernel triggered this launch, and
  they write no buffer that main kernel reads (it reads the other parity of the
  codes and offsets). The tail runs after the wait. CTAs other than CTA 0 skip
  the wait and exit when wave ``w - 1`` left no tail; CTA 0 always waits, so the
  launch completes only after the preceding main kernel.
- In a main kernel, the reducer warps wait before they read destination codes
  (or run a fused tail) and then trigger the next launch; a fused plan's setup
  warp waits before its slot phases. No other role touches the conflict-rank
  workspace, so none of them waits.
- ``rank_final`` triggers, then waits; the fold waits before it reads a plane.

``P`` uses the caller-workspace budget (1 GiB less an LSE reserve, and never
more than the ``RANK_PRIVATE_PLANES`` private-plane layout):
``P = clamp((budget - fixed) // plane, 1, W)``. Per-wave-setup plans for which
fewer than ``RANK_FLOOR_PLANES`` planes fit take at least the expected per-row
conflict density (``rank_merged_planes``).

Workspace (32-bit words, ``pitch = round_up(S_kv, 8)``, ``R`` = record width,
``S = W * K``, regions in this order, see ``rank_offsets``):

- dense planes: ``P * pitch * R``
- overflow records: ``S * R`` (fused plans: two wave parities, ``2 * S * R``)
- row state, three wave buffers: ``3 * pitch * 4`` (three ordinal words, one duplicate flag)
- plane-use entries: ``4 * pitch`` (64-bit plane-use count, fused-plan mark word, padding)
- overflow offsets, two parities: ``2 * pitch``
- per-slot duplicate records, four parities: ``4 * S``
- overflow row lists, four parities: ``4 * 4 * S`` (row, first record, record count, padding)
- duplicate row lists, four parities: ``4 * S``
- per-slot destination codes, two parities: ``2 * S``
- counters, eight wave blocks plus one padding vec4: ``36`` (overflow rows, overflow records,
  duplicate rows, long-chain rows)

Invariants: a wave's ordinal bits fit the row state (``W <= RANK_MAX_WAVE``);
every record index fits the destination-code value field and every plane count
the 16-bit plane-use fields (``rank_merged_codes_fit``, checked by the selection
and asserted by the workspace size); ``rank_geometry`` repeats the host plane
count ``rank_merged_planes`` exactly.
"""

import math
from typing import Optional
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.typing import Float32, Int32, Int64, Uint32, Uint64
from cudnn._cutlass_compat import SmemAllocator

RANK_DENSE_BUDGET_BYTES = 1 << 30
# Reserve for the LSE/OdO statistics segment inside the 1 GiB target (S_q <= 32768).
RANK_LSE_RESERVE_BYTES = 32 << 20
# Plane capacity of the private-plane layout (deterministic H128 D512 without conflict-rank storage);
# conflict-rank workspaces never exceed it.
RANK_PRIVATE_PLANES = 128
RANK_STATE_WORDS = 4
RANK_ORDINAL_WORDS = 3
RANK_MAX_WAVE = RANK_ORDINAL_WORDS * 32
# Row-state buffers: one setup pass clears wave w - 1's, reads wave w's and records wave w + 1's.
RANK_STATE_BUFFERS = 3
# Counter blocks rotated by wave step, four words each: overflow rows, overflow records, duplicate rows
# and long-chain rows. At most five consecutive blocks are live at once (``_rank_setup_slots``), so eight
# rotating blocks never reset one in use.
RANK_COUNTER_BLOCKS = 8
RANK_COUNTER_BLOCK_WORDS = 4 * RANK_COUNTER_BLOCKS
# The counter region ends with one vec4 that nothing reads. It stays because the dense plane count is
# the budget left after the fixed regions: dropping 16 bytes could move ``P`` for a KV extent at a
# budget boundary, and with it the order in which that extent's records are summed.
RANK_MERGED_COUNTER_WORDS = RANK_COUNTER_BLOCK_WORDS + 4
# Parities of the per-slot row lists, duplicate lists and duplicate records.
RANK_LIST_PARITIES = 4
# Fused plans: widest top-k and fewest dense planes (see ``rank_merged_fused``). From K = 1024 on, the
# setup warp's slot passes (16+ per wave) slow the main kernel more than a per-wave setup launch costs.
RANK_FUSED_MAX_TOPK = 512
RANK_FUSED_MIN_PLANES = 3
# Plans without per-query lengths interleave their waves (wave ``w`` holds queries ``w + j * waves``) once
# this many dense planes absorb the rows that queries ``waves`` apart still share.
RANK_INTERLEAVE_MIN_PLANES = 3
# Non-fused plans whose budget fits fewer planes take the expected per-row conflict density.
RANK_FLOOR_PLANES = 3
# Overflow tail: items per row (128 columns each) and records loaded per step.
RANK_TAIL_BATCH = 8
# Plane-use entry words per row: the 64-bit plane-use count, the fused-plan mark word and one padding
# word (16-byte entries, cleared with the row state by ``rank_init``).
RANK_USE_WORDS = 4
# Keeps every rotating parity non-negative: a multiple of 2, 3 and 4.
RANK_STEP_BIAS = 12
# SM count of the device the setup, init and fold grid sizes are tuned for (B200). They are
# tuning constants: these kernels walk their work in grid-stride loops, so every grid size is
# correct on every device. A setup grid also decides whether whole CTAs or warps sum the long
# overflow chains (``_rank_add_long_chains``), which fixes their summation order: a given plan
# stays bitwise reproducible, and device-independent constants keep that order off the SM count.
RANK_TUNED_SM_COUNT = 148
# Setup launches: at most four resident CTAs per SM (also the register cap of the launch).
RANK_SETUP_MIN_BLOCKS = 4
# Widest setup and ``rank_final`` grid: four CTAs per SM of the tuned device.
RANK_SETUP_CTAS = RANK_SETUP_MIN_BLOCKS * RANK_TUNED_SM_COUNT
RANK_SETUP_THREADS = 256
# Per-wave setup plans: overflow chains with more records than this are listed from the end of the
# row list and summed by whole setup CTAs (one CTA per row and 32-column group, one warp per contiguous
# rank segment, segment sums added in segment order); shorter chains keep one warp per 128-column part.
RANK_TAIL_LONG = 16
RANK_TAIL_GROUP_COLUMNS = 32
# ``long_ctas`` value telling ``_rank_tail`` that whole setup CTAs already summed every long chain.
RANK_TAIL_ALL_LONG_CTAS = 1 << 30
# Shared memory of one wave setup CTA: the broadcast tail flag (one 16-byte slot), the
# long-chain segment sums (four FP32 values per lane and warp) and 128 bytes of slack.
RANK_SETUP_SMEM_BYTES = 16 + 4 * 4 * RANK_SETUP_THREADS + 128
# Init grid: two CTAs per SM (tuning constant, see ``RANK_TUNED_SM_COUNT``).
RANK_INIT_CTAS = 2 * RANK_TUNED_SM_COUNT
RANK_INIT_THREADS = 256
# Destination code fields: record value, indirect (overflow offset of the row) and store flags.
RANK_CODE_VALUE = (1 << 29) - 1
RANK_CODE_INDIRECT = 1 << 29
RANK_CODE_STORE = 1 << 30
# Rows per fold CTA: one warp per row (``fold_rank_convert_dkv``).
RANK_FOLD_ROWS = 4
# Fold grid cap (16 four-warp CTAs per SM, see ``RANK_TUNED_SM_COUNT``): wider KV extents loop each warp
# over rows instead of launching one short-lived CTA per four rows.
RANK_FOLD_MAX_CTAS = 16 * RANK_TUNED_SM_COUNT


def rank_pitch(total_s_kv: int) -> int:
    """Return the padded KV row count of one plane."""
    return (int(total_s_kv) + 7) // 8 * 8


def rank_merged_fixed_words(total_s_kv: int, max_topk: int, record: int, wave: int, fused: bool) -> int:
    """Return the 32-bit word count of every D512 region except the dense planes.

    Fused plans keep two overflow-record parities.
    """
    pitch = rank_pitch(total_s_kv)
    slots = wave * max_topk
    record_buffers = 2 if fused else 1
    return (
        record_buffers * slots * record
        + RANK_STATE_BUFFERS * pitch * RANK_STATE_WORDS
        + RANK_USE_WORDS * pitch
        + 2 * pitch
        + RANK_LIST_PARITIES * slots
        + RANK_LIST_PARITIES * 4 * slots
        + RANK_LIST_PARITIES * slots
        + 2 * slots
        + RANK_MERGED_COUNTER_WORDS
    )


def _rank_budget_words(total_s_kv: int, record: int) -> int:
    """Return the dense-plane plus fixed-region budget: 1 GiB less the LSE reserve, at most the private-plane layout."""
    plane_words = rank_pitch(total_s_kv) * record
    return min((RANK_DENSE_BUDGET_BYTES - RANK_LSE_RESERVE_BYTES) // 4, RANK_PRIVATE_PLANES * plane_words)


def rank_merged_fused(total_s_kv: int, max_topk: int, record: int, wave: int) -> bool:
    """Fuse each wave's tail into the next main kernel for ``K <= RANK_FUSED_MAX_TOPK`` when the doubled
    overflow area still leaves ``RANK_FUSED_MIN_PLANES`` dense planes.

    Fused waves shift a tail row's ranks off plane 0, so too few planes would push
    shared rows into overflow wave after wave; wide top-k keeps the per-wave setup.
    """
    if max_topk > RANK_FUSED_MAX_TOPK:
        return False
    plane_words = rank_pitch(total_s_kv) * record
    planes = (_rank_budget_words(total_s_kv, record) - rank_merged_fixed_words(total_s_kv, max_topk, record, wave, True)) // plane_words
    return planes >= RANK_FUSED_MIN_PLANES


def rank_merged_planes(total_s_kv: int, max_topk: int, record: int, wave: int, fused: bool) -> int:
    """Return the D512 dense plane count, within ``[1, wave]``.

    Plans take the planes that fit the workspace budget. When fewer than
    ``RANK_FLOOR_PLANES`` fit, they take at least the expected per-row conflict
    density of a wave (``ceil(W * K / pitch)`` planes, plus one spare once rows
    are expected to be shared) as long as the private-plane layout still bounds
    the workspace; with more planes the budget fit always covers that density.
    Fused plans are selected with at least ``RANK_FUSED_MIN_PLANES`` planes, so
    the floor only ever applies to per-wave-setup plans.
    Must match ``RankShardReduction.rank_geometry`` exactly (integer arithmetic only).
    """
    pitch = rank_pitch(total_s_kv)
    plane_words = pitch * record
    fixed = rank_merged_fixed_words(total_s_kv, max_topk, record, wave, fused)
    planes = (_rank_budget_words(total_s_kv, record) - fixed) // plane_words
    if planes < RANK_FLOOR_PLANES:
        slots = wave * max_topk
        demand = (slots + pitch - 1) // pitch + min(slots // pitch, 1)
        demand = min(demand, (RANK_PRIVATE_PLANES * plane_words - fixed) // plane_words)
        planes = max(planes, demand)
    return max(1, min(wave, planes))


# Per-wave-setup plans with at most this many dense planes overflow most shared rows: their drains
# overlap the overflow-offset lookups of a tile's slots (see ``rank_merged_few_planes``).
RANK_FEW_PLANES = 3


def rank_merged_few_planes(total_s_kv: int, max_topk: int, record: int, wave: int) -> bool:
    """Return whether a per-wave-setup D512 plan keeps at most ``RANK_FEW_PLANES`` dense planes."""
    if rank_merged_fused(total_s_kv, max_topk, record, wave):
        return False
    return rank_merged_planes(total_s_kv, max_topk, record, wave, False) <= RANK_FEW_PLANES


def rank_merged_codes_fit(total_s_kv: int, max_topk: int, record: int, wave: int) -> bool:
    """Return whether every record of the D512 conflict-rank layout fits the destination-code fields.

    A code's value field (``RANK_CODE_VALUE``) holds record indices below ``P * pitch`` plus the
    overflow-record parities' ``S`` records, and the plane-use entries keep plane counts in 16-bit
    fields. Only KV extents far beyond any allocatable private-plane layout fail this.
    """
    fused = rank_merged_fused(total_s_kv, max_topk, record, wave)
    planes = rank_merged_planes(total_s_kv, max_topk, record, wave, fused)
    slots = wave * max_topk
    return planes * rank_pitch(total_s_kv) + (2 if fused else 1) * slots <= RANK_CODE_VALUE and planes < 1 << 16


def rank_merged_workspace_words(total_s_kv: int, max_topk: int, record: int, wave: int) -> int:
    """Return the 32-bit word count of the D512 pipelined conflict-rank workspace."""
    assert rank_merged_codes_fit(total_s_kv, max_topk, record, wave)
    fused = rank_merged_fused(total_s_kv, max_topk, record, wave)
    planes = rank_merged_planes(total_s_kv, max_topk, record, wave, fused)
    return planes * rank_pitch(total_s_kv) * record + rank_merged_fixed_words(total_s_kv, max_topk, record, wave, fused)


def rank_merged_workspace_rows(total_s_kv: int, max_topk: int, record: int, wave: int) -> int:
    """Return the D512 pipelined workspace size in whole records."""
    return -(-rank_merged_workspace_words(total_s_kv, max_topk, record, wave) // record)


def rank_merged_selected(total_s_kv: int, max_topk: int, record: int, wave: int, private_bytes: int) -> bool:
    """Prefer D512 conflict-rank storage whenever it fits the private-plane layout (``private_bytes``).

    The wave's ordinal bits must fit the row state (``RANK_MAX_WAVE``) and its records the codes
    (``rank_merged_codes_fit``).
    """
    if wave < 2 or wave > RANK_MAX_WAVE:
        return False
    if not rank_merged_codes_fit(total_s_kv, max_topk, record, wave):
        return False
    return rank_merged_workspace_rows(total_s_kv, max_topk, record, wave) * record * 4 <= private_bytes


@dsl_user_op
def _rank_load_u32x4(source: cute.Pointer, *, loc=None, ip=None):
    """Load one 16-byte row-state entry."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _rank_load_u32x4_cg(source: cute.Pointer, *, loc=None, ip=None):
    """Load one 16-byte row-state entry at the L2 coherence point."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _rank_load_s32_cg(source: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Load one 32-bit word at the L2 coherence point."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(), [source_i64], "ld.global.cg.s32 $0, [$1];", "=r,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
        )
    )


@dsl_user_op
def _rank_load_u64_cg(source: cute.Pointer, *, loc=None, ip=None) -> Uint64:
    """Load one aligned 64-bit plane-use entry at the L2 coherence point."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    return Uint64(
        llvm.inline_asm(
            T.i64(), [source_i64], "ld.global.cg.u64 $0, [$1];", "=l,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
        )
    )


@dsl_user_op
def _rank_store_u64(destination: cute.Pointer, value: Uint64, *, loc=None, ip=None) -> None:
    """Store one aligned 64-bit plane-use entry with a single access."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Uint64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _rank_store_zero_x4(destination: cute.Pointer, *, loc=None, ip=None) -> None:
    """Clear four aligned 32-bit words with one 128-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None, [destination_i64], "st.global.v4.b32 [$0], {0, 0, 0, 0};", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
    )


@dsl_user_op
def _rank_load_f32x4(source: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned FP32 values with one 128-bit load."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), values, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), values, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), values, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), values, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _rank_load_f32x4_stream(source: cute.Pointer, *, loc=None, ip=None):
    """Load four FP32 values of a read-once overflow record (cache-streaming: first eviction candidate)."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 4)
    value = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.cs.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), value, [index], loc=loc, ip=ip)) for index in range(4))


@dsl_user_op
def rank_store_f32x4(destination: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Store four aligned FP32 values with one 128-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def rank_reduce_f32x4(target: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Atomically add four FP32 values without returning the old values."""
    target_i64 = target.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            target_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "red.global.v4.f32.add [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def rank_store_f32x4_stream(destination: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Store four aligned FP32 values with one 128-bit store marked L2 evict-first."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            destination_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .b64 pol;\ncreatepolicy.fractional.L2::evict_first.b64 pol, 1.0;\nst.global.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, pol;\n}",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def rank_reduce_f32x4_stream(target: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Atomically add four FP32 values (no return) with the line marked L2 evict-first."""
    target_i64 = target.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            target_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .b64 pol;\ncreatepolicy.fractional.L2::evict_first.b64 pol, 1.0;\nred.global.add.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, pol;\n}",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


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


@cute.jit
def _rank_topk(mTopkIdxs: cute.Tensor, mTopkLength: Optional[cute.Tensor], query: Int32) -> Int32:
    """Return the clamped active top-k extent of one query."""
    topk = Int32(mTopkIdxs.shape[0])
    if cutlass.const_expr(mTopkLength is not None):
        raw = Int32(mTopkLength[query])
        if raw < topk:
            topk = raw
        if topk < Int32(0):
            topk = Int32(0)
    return topk


@cute.jit
def _rank_warp_inclusive_sum(value: Int32, lane: Int32) -> Int32:
    """Inclusive warp prefix sum; every lane of the warp must participate.

    ``shfl.sync.up`` keeps a lane's own value when its source lane is below the clamp operand, so the packed
    segment mask and clamp must be 0 (one 32-lane segment). It is passed explicitly because
    ``shuffle_sync_up`` defaults to a clamp of 31 before nvidia-cutlass-dsl 4.7.0, which makes every
    shuffle return the lane's own value.
    """
    total = value
    for shift in cutlass.range_constexpr(5):
        moved = cute.arch.shuffle_sync_up(total, 1 << shift, mask_and_clamp=0)
        if lane >= Int32(1 << shift):
            total = total + moved
    return total


class RankShardReduction:
    """Workspace geometry, pipelined wave setup, and destination records for conflict-rank dKV storage.

    The host class provides ``rank_record``, ``max_query_pairs_per_wave`` and ``max_topk``.
    """

    rank_shard = True
    rank_record = 512
    # Fused plans run wave ``w``'s tail inside main kernel ``w + 1`` (see ``rank_record_buffers``).
    rank_fused = False

    def rank_record_buffers(self) -> int:
        """Overflow-record parities: fused plans read wave ``w - 1``'s records while wave ``w`` writes."""
        return 2 if self.rank_fused else 1

    @cute.jit
    def rank_record_base(self, parity: Int32) -> Int32:
        """Return the first overflow record of a wave parity, relative to the overflow area."""
        base = Int32(0)
        if cutlass.const_expr(self.rank_fused):
            base = parity * Int32(self.max_query_pairs_per_wave * self.max_topk)
        return base

    @cute.jit
    def rank_geometry(self, seqlen_kv: Int32):
        """Return the padded plane pitch and the dense plane count for one KV extent.

        Mirrors ``rank_merged_planes`` in 64-bit arithmetic (a negative budget clamps to one plane).
        """
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        plane_words = pitch * Int64(self.rank_record)
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        fixed = (
            Int64(self.rank_record_buffers()) * slots * Int64(self.rank_record)
            + Int64(RANK_STATE_BUFFERS * RANK_STATE_WORDS + RANK_USE_WORDS + 2) * pitch
            + Int64(RANK_LIST_PARITIES * (1 + 4 + 1) + 2) * slots
            + Int64(RANK_MERGED_COUNTER_WORDS)
        )
        budget = min(Int64((RANK_DENSE_BUDGET_BYTES - RANK_LSE_RESERVE_BYTES) // 4), Int64(RANK_PRIVATE_PLANES) * plane_words)
        planes = (budget - fixed) // plane_words
        if cutlass.const_expr(not self.rank_fused):
            # Fused plans are selected with at least RANK_FUSED_MIN_PLANES >= RANK_FLOOR_PLANES planes.
            if planes < Int64(RANK_FLOOR_PLANES):
                demand = (slots + pitch - Int64(1)) // pitch + min(slots // pitch, Int64(1))
                demand = min(demand, (Int64(RANK_PRIVATE_PLANES) * plane_words - fixed) // plane_words)
                planes = max(planes, demand)
        planes = min(planes, Int64(self.max_query_pairs_per_wave))
        planes = max(planes, Int64(1))
        return pitch, Int32(planes)

    @cute.jit
    def rank_offsets(self, pitch: Int64, planes: Int32):
        """Return the 32-bit word offsets of every workspace region (see the module docstring).

        Rotating regions start at the returned offset; buffer ``b`` of the row
        state is at ``state + b * pitch * 4``, of the offsets at ``offsets + b * pitch``,
        of the per-slot regions at ``region + b * slots`` (row lists:
        ``row_list + b * 4 * slots``), and counter block ``b`` at ``counters + 4 * b``.
        """
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        record = Int64(self.rank_record)
        records = Int64(planes) * pitch * record
        state = records + Int64(self.rank_record_buffers()) * slots * record
        use = state + Int64(RANK_STATE_BUFFERS * RANK_STATE_WORDS) * pitch
        offsets = use + Int64(RANK_USE_WORDS) * pitch
        dup_records = offsets + Int64(2) * pitch
        row_list = dup_records + Int64(RANK_LIST_PARITIES) * slots
        dup_list = row_list + Int64(RANK_LIST_PARITIES * 4) * slots
        codes = dup_list + Int64(RANK_LIST_PARITIES) * slots
        counters = codes + Int64(2) * slots
        return records, state, use, offsets, dup_records, row_list, dup_list, codes, counters

    @cute.jit
    def make_rank_rows(self, workspace_dKV: cute.Tensor, seqlen_kv: Int32):
        """View the workspace as records; dense plane ``k`` row ``r`` is record ``k * pitch + r``."""
        pitch, planes = self.rank_geometry(seqlen_kv)
        rows = planes * Int32(pitch) + Int32(self.rank_record_buffers() * self.max_query_pairs_per_wave * self.max_topk)
        return cute.make_tensor(
            cute.recast_ptr(workspace_dKV.iterator, dtype=Float32),
            cute.make_layout((self.rank_record, rows, (1, 1)), stride=(1, Int64(self.rank_record), (0, 0))),
        )

    @cute.jit
    def rank_code(self, mRows: cute.Tensor, pitch: Int64, planes: Int32, parity: Int32, ordinal: Int32, slot: Int32) -> Int32:
        """Load the destination code of slot ``slot`` (below the top-k width) of wave ordinal ``ordinal``."""
        _, _, _, _, _, _, _, codes, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        return cute.arch.load(cute.recast_ptr(words + codes + Int64(parity) * slots + Int64(ordinal * Int32(self.max_topk) + slot), dtype=Int32), Int32)

    @cute.jit
    def rank_offsets_parity_base(self, mRows: cute.Tensor, pitch: Int64, planes: Int32, parity: Int32) -> cute.Pointer:
        """Return the Int32 pointer to row 0 of the overflow offsets of wave parity ``parity``."""
        _, _, _, offsets, _, _, _, _, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        return cute.recast_ptr(words + offsets + Int64(parity) * pitch, dtype=Int32)

    @cute.jit
    def rank_code_destination(self, mRows: cute.Tensor, pitch: Int64, planes: Int32, parity: Int32, code: Int32, row: Int32):
        """Return the destination record of an active slot of ``row`` from its loaded code, and whether the slot adds into it.

        Codes are written by the setup launch of the slot's wave into its parity buffer
        (``rank_code``): the value field is a record, or with the indirect flag the ``j``-th
        overflow record of ``row``; the store flag marks a first write. ``pitch`` and ``planes``
        come from ``rank_geometry``.
        """
        _, _, _, offsets, _, _, _, _, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        destination = code & Int32(RANK_CODE_VALUE)
        if (code & Int32(RANK_CODE_INDIRECT)) != Int32(0):
            first = cute.arch.load(cute.recast_ptr(words + offsets + Int64(parity) * pitch + Int64(row), dtype=Int32), Int32)
            destination = planes * Int32(pitch) + self.rank_record_base(parity) + first + destination
        return destination, (code & Int32(RANK_CODE_STORE)) == Int32(0)

    @cute.jit
    def _rank_add_records(self, floats: cute.Pointer, destination: Int64, first: Int64, count: Int32, part: Int32, lane: Int32):
        """Add ``count`` consecutive overflow records to one 128-column part of a plane-0 row in record order.

        Each lane owns one FP32x4 column; ``RANK_TAIL_BATCH`` records are loaded
        per step (the last step predicated) so many independent loads stay in
        flight, and the additions keep ascending record order.
        """
        record = Int64(self.rank_record)
        column = Int64((lane + Int32(32) * part) * Int32(4))
        early_target = not self.rank_fused
        if cutlass.const_expr(early_target):
            # Setup-launch tails: the plane-0 value does not depend on the records, so load it first and
            # overlap both latencies. Fused tails run on the main kernel's reducers and keep the late load.
            target = floats + destination * record + column
            p0, p1, p2, p3 = _rank_load_f32x4(target)
        acc = cute.make_rmem_tensor((4,), Float32)
        acc.fill(0.0)
        loaded = cute.make_rmem_tensor((RANK_TAIL_BATCH, 4), Float32)
        item = Int32(0)
        while item < count:
            for step in cutlass.range_constexpr(RANK_TAIL_BATCH):
                if item + Int32(step) < count:
                    a0, a1, a2, a3 = _rank_load_f32x4_stream(floats + (first + Int64(item + Int32(step))) * record + column)
                    loaded[step, 0] = a0
                    loaded[step, 1] = a1
                    loaded[step, 2] = a2
                    loaded[step, 3] = a3
            for step in cutlass.range_constexpr(RANK_TAIL_BATCH):
                if item + Int32(step) < count:
                    for value in cutlass.range_constexpr(4):
                        acc[value] = acc[value] + loaded[step, value]
            item += Int32(RANK_TAIL_BATCH)
        if cutlass.const_expr(not early_target):
            target = floats + destination * record + column
            p0, p1, p2, p3 = _rank_load_f32x4(target)
        rank_store_f32x4(target, p0 + acc[0], p1 + acc[1], p2 + acc[2], p3 + acc[3])

    @cute.jit
    def _rank_sum_records_column(self, floats: cute.Pointer, first: Int64, count: Int32, column: Int32):
        """Return the ascending-order FP32 sum of one 16-byte column of ``count`` consecutive records.

        ``RANK_TAIL_BATCH`` records are loaded before their additions, so one
        thread keeps that many independent loads in flight.
        """
        record = Int64(self.rank_record)
        source = floats + first * record + Int64(column * Int32(4))
        acc = cute.make_rmem_tensor((4,), Float32)
        acc.fill(0.0)
        loaded = cute.make_rmem_tensor((4 * RANK_TAIL_BATCH,), Float32)
        batches = count // Int32(RANK_TAIL_BATCH)
        for batch in cutlass.range(batches, unroll=1):
            batch_source = source + Int64(batch * Int32(RANK_TAIL_BATCH)) * record
            for item in cutlass.range_constexpr(RANK_TAIL_BATCH):
                v0, v1, v2, v3 = _rank_load_f32x4_stream(batch_source + Int64(item) * record)
                loaded[4 * item] = v0
                loaded[4 * item + 1] = v1
                loaded[4 * item + 2] = v2
                loaded[4 * item + 3] = v3
            for item in cutlass.range_constexpr(RANK_TAIL_BATCH):
                for value in cutlass.range_constexpr(4):
                    acc[value] = acc[value] + loaded[4 * item + value]
        remainder = count - batches * Int32(RANK_TAIL_BATCH)
        remainder_source = source + Int64(batches * Int32(RANK_TAIL_BATCH)) * record
        for item in cutlass.range_constexpr(RANK_TAIL_BATCH - 1):
            if Int32(item) < remainder:
                v0, v1, v2, v3 = _rank_load_f32x4_stream(remainder_source + Int64(item) * record)
                loaded[4 * item] = v0
                loaded[4 * item + 1] = v1
                loaded[4 * item + 2] = v2
                loaded[4 * item + 3] = v3
        for item in cutlass.range_constexpr(RANK_TAIL_BATCH - 1):
            if Int32(item) < remainder:
                for value in cutlass.range_constexpr(4):
                    acc[value] = acc[value] + loaded[4 * item + value]
        return acc[0], acc[1], acc[2], acc[3]

    @cute.jit
    def _rank_add_long_chains(self, mRows: cute.Tensor, seqlen_kv: Int32, tail_step: Int32, partials: cute.Tensor, bidx: Int32, nblocks: Int32, tidx: Int32):
        """Add wave ``v``'s long overflow chains to plane 0 with whole CTAs (``tail_step = v + RANK_STEP_BIAS``).

        When all (long row, 32-column group) items fit in one pass of the grid, one CTA
        takes one item: warp ``s`` sums the records ``[s * L, (s + 1) * L)`` of the chain
        (``L`` = chain length over the warp count) in rank order; warp 0 then adds the
        segment sums in segment order and adds the total to the plane-0 value. Otherwise
        one CTA takes one whole long row per pass: warp ``g + 4 * h`` sums half ``h`` of the
        chain on column group ``g``, and warp ``g`` adds the two halves in order to the
        plane-0 value, so no warp sums two whole chains back to back. Every thread of the
        CTA must call this.
        """
        pitch, planes = self.rank_geometry(seqlen_kv)
        records, _, _, _, _, row_list, _, _, counters = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        floats = mRows.iterator
        record = Int64(self.rank_record)
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        record_base = records // record + Int64(self.rank_record_base(tail_step % Int32(2)))
        tail_row_list = row_list + Int64(tail_step % Int32(RANK_LIST_PARITIES)) * Int64(4) * slots
        tail_counters = counters + Int64(Int32(4) * (tail_step % Int32(RANK_COUNTER_BLOCKS)))
        segments = RANK_SETUP_THREADS // 32
        groups = (self.rank_record // 4) // RANK_TAIL_GROUP_COLUMNS
        long_rows = cute.arch.load(cute.recast_ptr(words + tail_counters + Int64(3), dtype=Int32), Int32)
        items = long_rows * Int32(groups)
        segment = tidx // Int32(32)
        lane = tidx % Int32(32)
        if items > nblocks:
            # Whole-row items over several passes: two rank-ordered halves per 32-column group.
            halves = segments // groups
            row_group = segment % Int32(groups)
            half = segment // Int32(groups)
            row_item = bidx
            while row_item < long_rows:
                entry_row, entry_first, entry_count, _ = _rank_load_u32x4(words + tail_row_list + (slots - Int64(1) - Int64(row_item)) * Int64(4))
                count = Int32(entry_count)
                per = (count + Int32(halves - 1)) // Int32(halves)
                begin = min(count, half * per)
                end = min(count, begin + per)
                column = row_group * Int32(RANK_TAIL_GROUP_COLUMNS) + lane
                s0, s1, s2, s3 = self._rank_sum_records_column(floats, record_base + Int64(Int32(entry_first)) + Int64(begin), end - begin, column)
                partials[0, lane, segment] = s0
                partials[1, lane, segment] = s1
                partials[2, lane, segment] = s2
                partials[3, lane, segment] = s3
                cute.arch.barrier()
                if half == Int32(0):
                    target = floats + Int64(Int32(entry_row)) * record + Int64(column * Int32(4))
                    p0, p1, p2, p3 = _rank_load_f32x4(target)
                    a0 = partials[0, lane, row_group]
                    a1 = partials[1, lane, row_group]
                    a2 = partials[2, lane, row_group]
                    a3 = partials[3, lane, row_group]
                    for other in cutlass.range_constexpr(1, halves):
                        a0 = a0 + partials[0, lane, row_group + Int32(other * groups)]
                        a1 = a1 + partials[1, lane, row_group + Int32(other * groups)]
                        a2 = a2 + partials[2, lane, row_group + Int32(other * groups)]
                        a3 = a3 + partials[3, lane, row_group + Int32(other * groups)]
                    rank_store_f32x4(target, p0 + a0, p1 + a1, p2 + a2, p3 + a3)
                cute.arch.barrier()
                row_item += nblocks
            items = Int32(0)
        item = bidx
        while item < items:
            long_index = item // Int32(groups)
            group = item - long_index * Int32(groups)
            entry_row, entry_first, entry_count, _ = _rank_load_u32x4(words + tail_row_list + (slots - Int64(1) - Int64(long_index)) * Int64(4))
            count = Int32(entry_count)
            per = (count + Int32(segments - 1)) // Int32(segments)
            begin = min(count, segment * per)
            end = min(count, begin + per)
            column = group * Int32(RANK_TAIL_GROUP_COLUMNS) + lane
            s0, s1, s2, s3 = self._rank_sum_records_column(floats, record_base + Int64(Int32(entry_first)) + Int64(begin), end - begin, column)
            partials[0, lane, segment] = s0
            partials[1, lane, segment] = s1
            partials[2, lane, segment] = s2
            partials[3, lane, segment] = s3
            cute.arch.barrier()
            if segment == Int32(0):
                target = floats + Int64(Int32(entry_row)) * record + Int64(column * Int32(4))
                p0, p1, p2, p3 = _rank_load_f32x4(target)
                a0 = partials[0, lane, 0]
                a1 = partials[1, lane, 0]
                a2 = partials[2, lane, 0]
                a3 = partials[3, lane, 0]
                for other in cutlass.range_constexpr(1, segments):
                    a0 = a0 + partials[0, lane, other]
                    a1 = a1 + partials[1, lane, other]
                    a2 = a2 + partials[2, lane, other]
                    a3 = a3 + partials[3, lane, other]
                rank_store_f32x4(target, p0 + a0, p1 + a1, p2 + a2, p3 + a3)
            cute.arch.barrier()
            item += nblocks

    @cute.jit
    def _rank_add_duplicate_row(
        self,
        floats: cute.Pointer,
        words: cute.Pointer,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        records: Int64,
        dup_records: Int64,
        row: Int32,
        wave_begin: Int32,
        wave_size: Int32,
        wave_stride: Int32,
        lane: Int32,
    ):
        """Add a duplicated row's per-slot records to plane 0 in ascending (query, slot) order."""
        record = Int64(self.rank_record)
        chunks = self.rank_record // 128
        acc = cute.make_rmem_tensor((4 * chunks,), Float32)
        acc.fill(0.0)
        for ordinal in cutlass.range(wave_size, unroll=1):
            query = wave_begin + ordinal * wave_stride
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
                    source = Int32(cute.arch.popc(lowest - Uint32(1)))
                    slot_record = cute.arch.shuffle_sync(own_record, source)
                    for chunk in cutlass.range_constexpr(chunks):
                        column = Int64((lane + Int32(32 * chunk)) * Int32(4))
                        v0, v1, v2, v3 = _rank_load_f32x4(floats + (records // record + Int64(slot_record)) * record + column)
                        acc[4 * chunk] = acc[4 * chunk] + v0
                        acc[4 * chunk + 1] = acc[4 * chunk + 1] + v1
                        acc[4 * chunk + 2] = acc[4 * chunk + 2] + v2
                        acc[4 * chunk + 3] = acc[4 * chunk + 3] + v3
                    pending = pending ^ lowest
        for chunk in cutlass.range_constexpr(chunks):
            column = Int64((lane + Int32(32 * chunk)) * Int32(4))
            target = floats + Int64(row) * record + column
            p0, p1, p2, p3 = _rank_load_f32x4(target)
            rank_store_f32x4(target, p0 + acc[4 * chunk], p1 + acc[4 * chunk + 1], p2 + acc[4 * chunk + 2], p3 + acc[4 * chunk + 3])

    @cute.kernel
    def rank_init(self, mRows: cute.Tensor, seqlen_kv: Int32):
        """Clear dense plane 0, every row-state buffer, the plane-use entries and the counters."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch, planes = self.rank_geometry(seqlen_kv)
        _, state, _, _, _, _, _, _, counters = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        stride = Int64(nblocks) * Int64(RANK_INIT_THREADS)
        chunk = Int64(bidx) * Int64(RANK_INIT_THREADS) + Int64(tidx)
        dense_chunks = pitch * Int64(self.rank_record // 4)
        while chunk < dense_chunks:
            _rank_store_zero_x4(words + chunk * Int64(4))
            chunk += stride
        # Row state and plane-use entries are contiguous: RANK_STATE_BUFFERS * RANK_STATE_WORDS + RANK_USE_WORDS words per row.
        chunk = Int64(bidx) * Int64(RANK_INIT_THREADS) + Int64(tidx)
        state_chunks = (Int64(RANK_STATE_BUFFERS * RANK_STATE_WORDS + RANK_USE_WORDS) * pitch) // Int64(4)
        while chunk < state_chunks:
            _rank_store_zero_x4(words + state + chunk * Int64(4))
            chunk += stride
        if bidx == Int32(0) and tidx < Int32(RANK_MERGED_COUNTER_WORDS // 4):
            _rank_store_zero_x4(words + counters + Int64(tidx) * Int64(4))

    @cute.jit
    def _rank_setup_slots(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        step: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        next_begin: Int32,
        next_size: Int32,
        thread: Int32,
        nthreads: Int32,
        stride: Int32,
        cta_scratch: Optional[cute.Tensor] = None,
    ):
        """Clear wave ``w - 1``'s row state, record wave ``w + 1``'s ordinal bits and assign wave ``w``'s codes.

        With ``cta_scratch`` (the setup launch, whose threads cover every slot of a wave in one pass)
        each CTA allocates its overflow records and row-list positions with one atomic per counter:
        every warp of a wave would otherwise add to the same counter words, which serializes at one
        L2 slice when most rows of a wave overflow. Allocation order never changes a sum.

        ``step`` is ``w + RANK_STEP_BIAS``; ``thread``/``nthreads`` index the
        participating threads, whole warps only. These phases touch only
        setup-owned buffers and wave ``w``'s parity of codes and offsets, and
        read row state recorded by the previous phase pass, which must have
        completed (with the tail of wave ``w - 2``).
        """
        pitch, planes = self.rank_geometry(seqlen_kv)
        _, state, use, offsets, dup_records, row_list, dup_list, codes, counters = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        topk_width = Int32(self.max_topk)
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        prev_state = state + Int64((step + Int32(2)) % Int32(3)) * pitch * Int64(RANK_STATE_WORDS)
        cur_state = state + Int64(step % Int32(3)) * pitch * Int64(RANK_STATE_WORDS)
        next_state = state + Int64((step + Int32(1)) % Int32(3)) * pitch * Int64(RANK_STATE_WORDS)
        next_dup_list = dup_list + Int64((step + Int32(1)) % Int32(RANK_LIST_PARITIES)) * slots
        cur_half = Int64(step % Int32(2))
        cur_list = Int64(step % Int32(RANK_LIST_PARITIES))
        cur_dup_records = dup_records + cur_list * slots
        cur_row_list = row_list + cur_list * Int64(4) * slots
        cur_codes = codes + cur_half * slots
        cur_offsets = offsets + cur_half * pitch
        cur_counters = counters + Int64(Int32(4) * (step % Int32(RANK_COUNTER_BLOCKS)))
        next_counters = counters + Int64(Int32(4) * ((step + Int32(1)) % Int32(RANK_COUNTER_BLOCKS)))
        reset_counters = counters + Int64(Int32(4) * ((step + Int32(2)) % Int32(RANK_COUNTER_BLOCKS)))
        lane = thread % Int32(32)
        rows_counter = cute.recast_ptr(words + cur_counters, dtype=Int32)
        records_counter = cute.recast_ptr(words + cur_counters + Int64(1), dtype=Int32)
        long_counter = cute.recast_ptr(words + cur_counters + Int64(3), dtype=Int32)
        duplicates_counter = cute.recast_ptr(words + next_counters + Int64(2), dtype=Int32)
        prev_slots = prev_size * topk_width
        cur_slots = cur_size * topk_width
        next_slots = next_size * topk_width
        span = max(prev_slots, max(cur_slots, next_slots))
        warp_base = thread - lane
        # Aggregated plans step whole CTAs, so every warp of a CTA reaches the same barriers.
        loop_base = warp_base
        if cutlass.const_expr(cta_scratch is not None):
            loop_base = thread - thread % Int32(RANK_SETUP_THREADS)
        while loop_base < span:
            slot_index = warp_base + lane
            # Gather this slot's rows of waves w - 1, w + 1 and w first. Out-of-wave slots read
            # query 0 (always in bounds) and are masked, so the six loads issue back to back.
            prev_ordinal = slot_index // topk_width
            prev_slot = slot_index - prev_ordinal * topk_width
            prev_query = prev_begin + prev_ordinal * stride
            if slot_index >= prev_slots:
                prev_query = Int32(0)
            next_ordinal = slot_index // topk_width
            next_slot = slot_index - next_ordinal * topk_width
            next_query = next_begin + next_ordinal * stride
            if slot_index >= next_slots:
                next_query = Int32(0)
            ordinal = slot_index // topk_width
            slot = slot_index - ordinal * topk_width
            query = cur_begin + ordinal * stride
            if slot_index >= cur_slots:
                query = Int32(0)
            prev_extent = _rank_topk(mTopkIdxs, mTopkLength, prev_query)
            next_extent = _rank_topk(mTopkIdxs, mTopkLength, next_query)
            cur_extent = _rank_topk(mTopkIdxs, mTopkLength, query)
            prev_index = mTopkIdxs[prev_slot, (prev_query, 0)]
            next_index = mTopkIdxs[next_slot, (next_query, 0)]
            cur_index = mTopkIdxs[slot, (query, 0)]
            prev_row = Int32(-1)
            if slot_index < prev_slots and prev_slot < prev_extent:
                if prev_index >= Int32(0) and prev_index < seqlen_kv:
                    prev_row = prev_index
            next_row = Int32(-1)
            if slot_index < next_slots and next_slot < next_extent:
                if next_index >= Int32(0) and next_index < seqlen_kv:
                    next_row = next_index
            row = Int32(-1)
            if slot_index < cur_slots and slot < cur_extent:
                if cur_index >= Int32(0) and cur_index < seqlen_kv:
                    row = cur_index
            # Clear wave w - 1's row state and record wave w + 1's ordinal bit, then issue wave w's
            # row-state, mark and plane-use loads; the three touch disjoint buffers, so their latencies overlap.
            if prev_row >= Int32(0):
                _rank_store_zero_x4(words + prev_state + Int64(prev_row) * Int64(RANK_STATE_WORDS))
            next_entry = words + next_state + Int64(max(next_row, Int32(0))) * Int64(RANK_STATE_WORDS)
            next_bit = Uint32(1) << Uint32(next_ordinal % Int32(32))
            previous = Uint32(0)
            if next_row >= Int32(0):
                previous = Uint32(cute.arch.atomic_or((next_entry + Int64(next_ordinal // Int32(32))).llvm_ptr, next_bit, sem="relaxed", scope="gpu"))
            state_0 = Uint32(0)
            state_1 = Uint32(0)
            state_2 = Uint32(0)
            duplicate = Uint32(0)
            mark = Int32(0)
            packed = Uint64(0)
            mark_entry = words + use + Int64(max(row, Int32(0))) * Int64(RANK_USE_WORDS) + Int64(2)
            use_entry = words + use + Int64(max(row, Int32(0))) * Int64(RANK_USE_WORDS)
            if row >= Int32(0):
                state_0, state_1, state_2, duplicate = _rank_load_u32x4_cg(words + cur_state + Int64(row) * Int64(RANK_STATE_WORDS))
                if cutlass.const_expr(self.rank_fused):
                    mark = _rank_load_s32_cg(mark_entry)
                packed = _rank_load_u64_cg(use_entry)
            # Assign wave w's destinations; each warp allocates its overflow records with one atomic per counter.
            code = Int32(0)
            has_code = Int32(0)
            is_duplicate = Int32(0)
            need_records = Int32(0)
            need_row = Int32(0)
            need_long = Int32(0)
            if row >= Int32(0):
                has_code = Int32(1)
                # Fused plans: a row whose previous wave needs a tail (added to plane 0 by the
                # next main kernel) shifts this wave's ranks off plane 0.
                shift = Int32(0)
                if cutlass.const_expr(self.rank_fused):
                    if (mark >> Int32(1)) == step:
                        shift = mark & Int32(1)
                    elif (mark >> Int32(1)) == step - Int32(1):
                        shift = Int32(1)
                if duplicate != Uint32(0):
                    is_duplicate = Int32(1)
                    need_records = Int32(1)
                    if cutlass.const_expr(self.rank_fused):
                        cute.arch.store(cute.recast_ptr(mark_entry, dtype=Int32), (step << Int32(1)) | shift)
                else:
                    count = Int32(cute.arch.popc(state_0) + cute.arch.popc(state_1) + cute.arch.popc(state_2))
                    conflict = Int32(
                        cute.arch.popc(state_0 & _rank_lower_mask(ordinal, 0))
                        + cute.arch.popc(state_1 & _rank_lower_mask(ordinal, 1))
                        + cute.arch.popc(state_2 & _rank_lower_mask(ordinal, 2))
                    )
                    first_writer = conflict == Int32(0)
                    conflict = conflict + shift
                    count = count + shift
                    # Planes this row initialized before this wave (plane 0 is zero-filled).
                    tag = Int32(packed & Uint64(0xFFFFFFFF))
                    counts = Int32(packed >> Uint64(32))
                    used = counts & Int32(0xFFFF)
                    if tag == step:
                        used = counts >> Int32(16)
                    used = max(used, Int32(1))
                    if conflict < planes:
                        code = conflict * Int32(pitch) + row
                        if conflict >= used:
                            code = code | Int32(RANK_CODE_STORE)
                    else:
                        code = (conflict - planes) | Int32(RANK_CODE_INDIRECT | RANK_CODE_STORE)
                    if first_writer:
                        now_used = max(used, min(count, planes))
                        if now_used > used:
                            entry_value = (Uint64(Uint32((used << Int32(16)) | now_used)) << Uint64(32)) | Uint64(Uint32(step))
                            _rank_store_u64(use_entry, entry_value)
                    if count > planes and conflict == planes:
                        need_records = count - planes
                        need_row = Int32(1)
                        if cutlass.const_expr(not self.rank_fused):
                            # Long chains go to the end of the row list for the whole-CTA tail.
                            if need_records > Int32(RANK_TAIL_LONG):
                                need_row = Int32(0)
                                need_long = Int32(1)
                        if cutlass.const_expr(self.rank_fused):
                            cute.arch.store(cute.recast_ptr(mark_entry, dtype=Int32), (step << Int32(1)) | shift)
            # Wave w + 1's duplicated rows: a query that set its own bit before selects the row twice.
            if next_row >= Int32(0):
                if (previous & next_bit) != Uint32(0):
                    flagged = cute.arch.atomic_or((next_entry + Int64(3)).llvm_ptr, Uint32(1), sem="relaxed", scope="gpu")
                    if flagged == Uint32(0):
                        position = cute.arch.atomic_add(duplicates_counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                        cute.arch.store(cute.recast_ptr(words + next_dup_list + Int64(position), dtype=Int32), next_row)
            take_bases = warp_base < cur_slots
            if cutlass.const_expr(cta_scratch is not None):
                take_bases = loop_base < cur_slots
            if take_bases:
                records_inclusive = _rank_warp_inclusive_sum(need_records, lane)
                rows_inclusive = _rank_warp_inclusive_sum(need_row, lane)
                records_total = cute.arch.shuffle_sync(records_inclusive, 31)
                rows_total = cute.arch.shuffle_sync(rows_inclusive, 31)
                long_inclusive = Int32(0)
                long_total = Int32(0)
                if cutlass.const_expr(not self.rank_fused):
                    long_inclusive = _rank_warp_inclusive_sum(need_long, lane)
                    long_total = cute.arch.shuffle_sync(long_inclusive, 31)
                records_base = Int32(0)
                rows_base = Int32(0)
                long_base = Int32(0)
                if cutlass.const_expr(cta_scratch is None):
                    if lane == Int32(31):
                        if records_total > Int32(0):
                            records_base = Int32(cute.arch.atomic_add(records_counter.llvm_ptr, records_total, sem="relaxed", scope="gpu"))
                        if rows_total > Int32(0):
                            rows_base = Int32(cute.arch.atomic_add(rows_counter.llvm_ptr, rows_total, sem="relaxed", scope="gpu"))
                        if cutlass.const_expr(not self.rank_fused):
                            if long_total > Int32(0):
                                long_base = Int32(cute.arch.atomic_add(long_counter.llvm_ptr, long_total, sem="relaxed", scope="gpu"))
                    records_base = cute.arch.shuffle_sync(records_base, 31)
                    rows_base = cute.arch.shuffle_sync(rows_base, 31)
                    if cutlass.const_expr(not self.rank_fused):
                        long_base = cute.arch.shuffle_sync(long_base, 31)
                else:
                    # Row w of the scratch holds warp w's (records, rows, long) totals in columns 0-2 and
                    # their exclusive prefixes over the CTA's warps in columns 3-5; row 8 holds the CTA bases.
                    cta_warp = (thread % Int32(RANK_SETUP_THREADS)) // Int32(32)
                    if lane == Int32(31):
                        cta_scratch[cta_warp, 0] = records_total
                        cta_scratch[cta_warp, 1] = rows_total
                        cta_scratch[cta_warp, 2] = long_total
                    cute.arch.barrier()
                    counter_index = thread % Int32(RANK_SETUP_THREADS)
                    if counter_index < Int32(3):
                        running = Int32(0)
                        for scan_warp in cutlass.range_constexpr(RANK_SETUP_THREADS // 32):
                            warp_total = cta_scratch[scan_warp, counter_index]
                            cta_scratch[scan_warp, counter_index + Int32(3)] = running
                            running = running + warp_total
                        cta_base = Int32(0)
                        if running > Int32(0):
                            counter_word = Int64(1)
                            if counter_index == Int32(1):
                                counter_word = Int64(0)
                            elif counter_index == Int32(2):
                                counter_word = Int64(3)
                            counter = cute.recast_ptr(words + cur_counters + counter_word, dtype=Int32)
                            cta_base = Int32(cute.arch.atomic_add(counter.llvm_ptr, running, sem="relaxed", scope="gpu"))
                        cta_scratch[RANK_SETUP_THREADS // 32, counter_index] = cta_base
                    cute.arch.barrier()
                    records_base = cta_scratch[RANK_SETUP_THREADS // 32, 0] + cta_scratch[cta_warp, 3]
                    rows_base = cta_scratch[RANK_SETUP_THREADS // 32, 1] + cta_scratch[cta_warp, 4]
                    if cutlass.const_expr(not self.rank_fused):
                        long_base = cta_scratch[RANK_SETUP_THREADS // 32, 2] + cta_scratch[cta_warp, 5]
                    # The next pass of this CTA rewrites the scratch only after every warp read its bases.
                    cute.arch.barrier()
                first = records_base + records_inclusive - need_records
                if has_code != Int32(0):
                    if is_duplicate != Int32(0):
                        cute.arch.store(cute.recast_ptr(words + cur_dup_records + Int64(slot_index), dtype=Int32), first)
                        code = (planes * Int32(pitch) + self.rank_record_base(step % Int32(2)) + first) | Int32(RANK_CODE_STORE)
                    cute.arch.store(cute.recast_ptr(words + cur_codes + Int64(slot_index), dtype=Int32), code)
                    if cutlass.const_expr(self.rank_fused):
                        if need_row != Int32(0):
                            row_position = rows_base + rows_inclusive - Int32(1)
                            row_entry = cute.recast_ptr(words + cur_row_list + Int64(row_position) * Int64(4), dtype=Int32)
                            cute.arch.store(row_entry, row)
                            cute.arch.store(row_entry + 1, first)
                            cute.arch.store(row_entry + 2, need_records)
                            cute.arch.store(cute.recast_ptr(words + cur_offsets + Int64(row), dtype=Int32), first)
                    else:
                        # Short rows fill the row list from the front, long rows from the end.
                        listed = need_row + need_long
                        row_position = Int64(rows_base + rows_inclusive - Int32(1))
                        if need_long != Int32(0):
                            row_position = slots - Int64(1) - Int64(long_base + long_inclusive - Int32(1))
                        if listed != Int32(0):
                            row_entry = cute.recast_ptr(words + cur_row_list + row_position * Int64(4), dtype=Int32)
                            cute.arch.store(row_entry, row)
                            cute.arch.store(row_entry + 1, first)
                            cute.arch.store(row_entry + 2, need_records)
                            cute.arch.store(cute.recast_ptr(words + cur_offsets + Int64(row), dtype=Int32), first)
            warp_base += nthreads
            loop_base += nthreads
        if thread < Int32(4):
            # Reset the block two steps ahead for that wave's slot phases. At most five consecutive blocks are
            # live at once (a fused main kernel runs step - 2's tail while its setup warp counts into blocks
            # step and step + 1 and resets step + 2), so this block, last used by step - 6, is free.
            cute.arch.store(cute.recast_ptr(words + reset_counters + Int64(thread), dtype=Int32), Int32(0))

    @cute.jit
    def _rank_tail(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        tail_step: Int32,
        tail_begin: Int32,
        tail_size: Int32,
        warp: Int32,
        nwarps: Int32,
        lane: Int32,
        tail_stride: Int32,
        long_ctas: Int32,
    ):
        """Add wave ``v``'s overflow and duplicate records to plane 0 (``tail_step = v + RANK_STEP_BIAS``).

        Wave ``v``'s main kernel must have completed. Long chains (listed from the
        end of the row list) join the per-warp items unless ``long_ctas`` CTAs take
        them all in one pass through ``_rank_add_long_chains``.
        """
        if tail_size > Int32(0):
            pitch, planes = self.rank_geometry(seqlen_kv)
            records, _, _, _, dup_records, row_list, dup_list, _, counters = self.rank_offsets(pitch, planes)
            words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
            floats = mRows.iterator
            slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
            tail_list = Int64(tail_step % Int32(RANK_LIST_PARITIES))
            records = records + Int64(self.rank_record_base(tail_step % Int32(2))) * Int64(self.rank_record)
            tail_counters = counters + Int64(Int32(4) * (tail_step % Int32(RANK_COUNTER_BLOCKS)))
            tail_row_list = row_list + tail_list * Int64(4) * slots
            tail_dup_list = dup_list + tail_list * slots
            tail_dup_records = dup_records + tail_list * slots
            # One warp per 128 record columns.
            tail_split = self.rank_record // 128
            if cutlass.const_expr(self.rank_fused):
                # Fused plans list every overflow row from the front (no long-chain list).
                # One warp per quarter row: each work item adds a row's records to 128 plane-0 columns.
                overflow_items = Int32(tail_split) * cute.arch.load(cute.recast_ptr(words + tail_counters, dtype=Int32), Int32)
                overflow_item = warp
                while overflow_item < overflow_items:
                    entry_index = overflow_item // Int32(tail_split)
                    part = overflow_item - entry_index * Int32(tail_split)
                    tail_entry = cute.recast_ptr(words + tail_row_list + Int64(entry_index) * Int64(4), dtype=Int32)
                    tail_row = cute.arch.load(tail_entry, Int32)
                    tail_first = cute.arch.load(tail_entry + 1, Int32)
                    tail_count = cute.arch.load(tail_entry + 2, Int32)
                    self._rank_add_records(floats, Int64(tail_row), records // Int64(self.rank_record) + Int64(tail_first), tail_count, part, lane)
                    overflow_item += nwarps
            else:
                short_rows = cute.arch.load(cute.recast_ptr(words + tail_counters, dtype=Int32), Int32)
                long_rows = cute.arch.load(cute.recast_ptr(words + tail_counters + Int64(3), dtype=Int32), Int32)
                warp_rows = short_rows
                warp_long_rows = Int32(0)
                if long_rows * Int32((self.rank_record // 4) // RANK_TAIL_GROUP_COLUMNS) > long_ctas:
                    warp_rows = short_rows + long_rows
                    warp_long_rows = long_rows
                # One warp per quarter row: each work item adds a row's records to 128 plane-0 columns.
                # Long chains (listed from the end of the row list) are dealt out first, so the warps that
                # take them start early and the short chains fill the remaining warps around them.
                overflow_items = Int32(tail_split) * warp_rows
                overflow_item = warp
                while overflow_item < overflow_items:
                    entry_index = overflow_item // Int32(tail_split)
                    part = overflow_item - entry_index * Int32(tail_split)
                    list_index = Int64(entry_index - warp_long_rows)
                    if entry_index < warp_long_rows:
                        list_index = slots - Int64(1) - Int64(entry_index)
                    tail_entry = cute.recast_ptr(words + tail_row_list + list_index * Int64(4), dtype=Int32)
                    tail_row = cute.arch.load(tail_entry, Int32)
                    tail_first = cute.arch.load(tail_entry + 1, Int32)
                    tail_count = cute.arch.load(tail_entry + 2, Int32)
                    self._rank_add_records(floats, Int64(tail_row), records // Int64(self.rank_record) + Int64(tail_first), tail_count, part, lane)
                    overflow_item += nwarps
            duplicate_rows = cute.arch.load(cute.recast_ptr(words + tail_counters + Int64(2), dtype=Int32), Int32)
            duplicate_item = warp
            while duplicate_item < duplicate_rows:
                duplicate_row = cute.arch.load(cute.recast_ptr(words + tail_dup_list + Int64(duplicate_item), dtype=Int32), Int32)
                self._rank_add_duplicate_row(
                    floats, words, mTopkIdxs, mTopkLength, records, tail_dup_records, duplicate_row, tail_begin, tail_size, tail_stride, lane
                )
                duplicate_item += nwarps

    @cute.kernel
    def rank_wave_setup(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        step: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        next_begin: Int32,
        next_size: Int32,
        early: Int32,
        stride: Int32,
    ):
        """Assign wave ``w``'s destinations and record wave ``w + 1``, then finish wave ``w - 1``.

        ``step`` is ``w + RANK_STEP_BIAS``. ``early != 0`` waits before every
        phase (the first launches of an execution). After the grid dependency
        wait, wave ``w - 1``'s overflow and duplicate records are added to plane 0.
        """
        cute.arch.griddepcontrol_launch_dependents()
        if early != Int32(0):
            cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        thread = bidx * Int32(RANK_SETUP_THREADS) + tidx
        nthreads = nblocks * Int32(RANK_SETUP_THREADS)
        smem = SmemAllocator()
        tail_flag = smem.allocate_tensor(Int32, cute.make_layout((4,), stride=(1,)), byte_alignment=16)
        partials = smem.allocate_tensor(Float32, cute.make_layout((4, 32, RANK_SETUP_THREADS // 32), stride=(1, 4, 128)), byte_alignment=16)
        # The long-chain partials are only used after the slot phases: lend them as allocation scratch.
        cta_scratch = cute.make_tensor(cute.recast_ptr(partials.iterator, dtype=Int32), cute.make_layout((RANK_SETUP_THREADS // 32 + 1, 8), stride=(8, 1)))
        self._rank_setup_slots(
            mRows,
            mTopkIdxs,
            mTopkLength,
            seqlen_kv,
            step,
            prev_begin,
            prev_size,
            cur_begin,
            cur_size,
            next_begin,
            next_size,
            thread,
            nthreads,
            stride,
            cta_scratch,
        )
        # Wave w - 1's tail counters were final before its main kernel started. When that wave left
        # no overflow or duplicate records, CTAs other than CTA 0 exit here and free their SMs for the
        # next main kernel; CTA 0 always waits, so the launch still completes after the preceding kernel.
        # A wave with tail work keeps every CTA resident until the tail is done, so the next main
        # kernel does not start early and compete with the tail for memory bandwidth.
        has_tail = bidx == Int32(0)
        if prev_size > Int32(0):
            # One thread per CTA reads the counter block at L2 and broadcasts it: every thread of the
            # grid reading the same line would serialize thousands of requests on one L2 slice.
            if tidx == Int32(0):
                pitch, planes = self.rank_geometry(seqlen_kv)
                _, _, _, _, _, _, _, _, counters = self.rank_offsets(pitch, planes)
                words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
                tail_counters = counters + Int64(Int32(4) * ((step - Int32(1)) % Int32(RANK_COUNTER_BLOCKS)))
                overflow_rows, _, duplicate_rows, long_rows = _rank_load_u32x4_cg(words + tail_counters)
                tail_flag[0] = Int32(0)
                if (overflow_rows | duplicate_rows | long_rows) != Uint32(0):
                    tail_flag[0] = Int32(1)
            cute.arch.barrier()
            if tail_flag[0] != Int32(0):
                has_tail = cutlass.Boolean(True)
        if has_tail:
            cute.arch.griddepcontrol_wait()
            if prev_size > Int32(0):
                self._rank_add_long_chains(mRows, seqlen_kv, step - Int32(1), partials, bidx, nblocks, tidx)
            # ``_rank_add_long_chains`` took every long chain (one pass or whole-row passes).
            self._rank_tail(
                mRows,
                mTopkIdxs,
                mTopkLength,
                seqlen_kv,
                step - Int32(1),
                prev_begin,
                prev_size,
                thread // Int32(32),
                nthreads // Int32(32),
                tidx % Int32(32),
                stride,
                Int32(RANK_TAIL_ALL_LONG_CTAS),
            )

    @cute.kernel
    def rank_final(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        tail_step: Int32,
        tail_begin: Int32,
        tail_size: Int32,
        tail_stride: Int32,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        problem_shape,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Finish the last wave's overflow and duplicate records and form the ordered dSink partials.

        Every main kernel has completed, so each thread reduces one (query chunk,
        head) into consumed O*dO storage exactly as the standalone partial kernel.
        """
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        thread = bidx * Int32(RANK_SETUP_THREADS) + tidx
        nthreads = nblocks * Int32(RANK_SETUP_THREADS)
        self._rank_tail(
            mRows,
            mTopkIdxs,
            mTopkLength,
            seqlen_kv,
            tail_step,
            tail_begin,
            tail_size,
            thread // Int32(32),
            nthreads // Int32(32),
            tidx % Int32(32),
            tail_stride,
            Int32(0),
        )
        heads = problem_shape[3][0]
        items = cute.ceil_div(problem_shape[0], chunk_size) * heads
        item = thread
        while item < items:
            q_block_idx = item // heads
            self._sum_dSink_partial_head_unrolled(sum_odo, scaled_lse, attn_sink, problem_shape, q_block_idx, item - q_block_idx * heads, chunk_size)
            item += nthreads

    @cute.kernel
    def fold_rank_convert_dkv(
        self, mRows: cute.Tensor, mdKV: cute.Tensor, seqlen: Int32, sum_odo: cute.Tensor, d_sink: cute.Tensor, problem_shape, chunk_size: cutlass.Constexpr[int]
    ):
        """Add each row's used planes in plane order, emit BF16, and publish the stream-ordered dSink fold.

        Each lane folds 16 record columns.
        """
        cute.arch.griddepcontrol_wait()
        seq_block, _, _ = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()
        fold_grid, _, _ = cute.arch.grid_dim()
        seq_id = seq_block * Int32(RANK_FOLD_ROWS) + tidy
        # Rows of this warp: seq_id, seq_id + 4 * grid, ...
        row_stop = seqlen
        while seq_id < row_stop:
            pitch, planes = self.rank_geometry(seqlen)
            _, _, use, _, _, _, _, _, _ = self.rank_offsets(pitch, planes)
            words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
            packed = _rank_load_u64_cg(words + use + Int64(seq_id) * Int64(RANK_USE_WORDS))
            used = max(Int32(packed >> Uint64(32)) & Int32(0xFFFF), Int32(1))
            source_base = Int64(tidx * Int32(16))
            final_values = cute.make_rmem_tensor((16,), Float32)
            for value_idx in cutlass.range_constexpr(16):
                final_values[value_idx] = Float32(0.0)
            plane_stride = pitch * Int64(self.rank_record)
            source_row = mRows.iterator + Int64(seq_id) * Int64(self.rank_record) + source_base
            loaded = cute.make_rmem_tensor((4, 16), Float32)
            plane = Int32(0)
            while plane + Int32(3) < used:
                for step in cutlass.range_constexpr(4):
                    source = source_row + Int64(plane + Int32(step)) * plane_stride
                    for quad in cutlass.range_constexpr(4):
                        q0, q1, q2, q3 = _rank_load_f32x4(source + Int64(4 * quad))
                        loaded[step, 4 * quad] = q0
                        loaded[step, 4 * quad + 1] = q1
                        loaded[step, 4 * quad + 2] = q2
                        loaded[step, 4 * quad + 3] = q3
                for step in cutlass.range_constexpr(4):
                    for value_idx in cutlass.range_constexpr(16):
                        final_values[value_idx] = final_values[value_idx] + loaded[step, value_idx]
                plane += Int32(4)
            while plane < used:
                source = source_row + Int64(plane) * plane_stride
                for quad in cutlass.range_constexpr(4):
                    q0, q1, q2, q3 = _rank_load_f32x4(source + Int64(4 * quad))
                    final_values[4 * quad] = final_values[4 * quad] + q0
                    final_values[4 * quad + 1] = final_values[4 * quad + 1] + q1
                    final_values[4 * quad + 2] = final_values[4 * quad + 2] + q2
                    final_values[4 * quad + 3] = final_values[4 * quad + 3] + q3
                plane += Int32(1)
            out_row = mdKV[None, seq_id, (0, 0)]
            out_column = Int32(tidx * Int32(16))
            chunk = out_column // Int32(32) * Int32(32)
            a0 = out_column % Int32(32) // Int32(4)
            for b in cutlass.range_constexpr(4):
                output_base = chunk + a0 + b * Int32(8)
                packed01 = rank_pack_bf16x2(final_values[b], final_values[b + 4])
                packed23 = rank_pack_bf16x2(final_values[b + 8], final_values[b + 12])
                rank_store_bf16x4(out_row.iterator + output_base, packed01, packed23)
            seq_id += fold_grid * Int32(RANK_FOLD_ROWS)
        # dSink: warp 0 of fold CTA b folds heads b, b + grid, ... (one warp per head; CTA 0's serial
        # per-head loops outlasted the plane fold of every other CTA).
        if tidy == Int32(0):
            head = seq_block
            while head < problem_shape[3][0]:
                self._finalize_dSink_head_warp(sum_odo, d_sink, problem_shape, head, tidx, chunk_size)
                head += fold_grid

    @cute.jit
    def _finalize_dSink_head_warp(self, sum_odo: cute.Tensor, d_sink: cute.Tensor, problem_shape, head: Int32, lane: Int32, chunk_size: cutlass.Constexpr[int]):
        """Fold one head's query-chunk partials with one warp in a fixed order.

        Lane ``l`` sums chunks ``l, l + 32, ...`` over four interleaved accumulators (combined as
        ``(a0 + a1) + (a2 + a3)``); a fixed XOR butterfly then adds the lane sums, so every lane
        holds the same total and lane 0 stores it.
        """
        chunk_count = cute.ceil_div(problem_shape[0], chunk_size)
        acc_0 = Float32(0.0)
        acc_1 = Float32(0.0)
        acc_2 = Float32(0.0)
        acc_3 = Float32(0.0)
        chunk = lane
        while chunk + Int32(96) < chunk_count:
            acc_0 += sum_odo[head, (chunk * Int32(chunk_size), 0)]
            acc_1 += sum_odo[head, ((chunk + Int32(32)) * Int32(chunk_size), 0)]
            acc_2 += sum_odo[head, ((chunk + Int32(64)) * Int32(chunk_size), 0)]
            acc_3 += sum_odo[head, ((chunk + Int32(96)) * Int32(chunk_size), 0)]
            chunk += Int32(128)
        while chunk < chunk_count:
            acc_0 += sum_odo[head, (chunk * Int32(chunk_size), 0)]
            chunk += Int32(32)
        total = acc_0 + acc_1 + (acc_2 + acc_3)
        for shift in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync(total, lane ^ Int32(16 >> shift))
        if lane == Int32(0):
            d_sink[head, (0, 0)] = total

    @cute.jit
    def _sum_dSink_partial_head_unrolled(
        self,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        problem_shape,
        q_block_idx: Int32,
        head: Int32,
        chunk_size: cutlass.Constexpr[int],
    ):
        """``_sum_dSink_partial_head_deterministic`` with every load of the chunk issued up front.

        The FP32 order is unchanged: query ``q_begin + i`` adds into accumulator ``i % 4`` while it
        belongs to a full group of four and into accumulator 0 after that.
        """
        sink_log2 = Float32(attn_sink[head, (0, 0)]) * Float32(math.log2(math.e))
        q_begin = q_block_idx * Int32(chunk_size)
        count = min(problem_shape[0], q_begin + Int32(chunk_size)) - q_begin
        quad_end = count // Int32(4) * Int32(4)
        lse_values = cute.make_rmem_tensor((chunk_size,), Float32)
        odo_values = cute.make_rmem_tensor((chunk_size,), Float32)
        for qi in cutlass.range_constexpr(chunk_size):
            lse_values[qi] = Float32(0.0)
            odo_values[qi] = Float32(0.0)
            if Int32(qi) < count:
                lse_values[qi] = Float32(scaled_lse[head, (q_begin + Int32(qi), 0)])
                odo_values[qi] = Float32(sum_odo[head, (q_begin + Int32(qi), 0)])
        acc_0 = Float32(0.0)
        acc_1 = Float32(0.0)
        acc_2 = Float32(0.0)
        acc_3 = Float32(0.0)
        for qi in cutlass.range_constexpr(chunk_size):
            p = cute.math.exp2(sink_log2 + lse_values[qi])
            if sink_log2 == Float32(float("inf")):
                p = Float32(1.0)
            elif sink_log2 == Float32(float("-inf")):
                p = Float32(0.0)
            if Int32(qi) < quad_end:
                if cutlass.const_expr(qi % 4 == 0):
                    acc_0 += p * odo_values[qi]
                elif cutlass.const_expr(qi % 4 == 1):
                    acc_1 += p * odo_values[qi]
                elif cutlass.const_expr(qi % 4 == 2):
                    acc_2 += p * odo_values[qi]
                else:
                    acc_3 += p * odo_values[qi]
            elif Int32(qi) < count:
                acc_0 += p * odo_values[qi]
        # Main-kernel waves have consumed these O*dO entries before this launch.
        # Their storage can now hold one partial for the terminal dSink fold.
        sum_odo[head, (q_begin, 0)] = acc_0 + acc_1 + (acc_2 + acc_3)

    @cute.jit
    def launch_rank_init(self, mRows: cute.Tensor, seqlen_kv: Int32, stream: cuda.CUstream):
        """Clear plane 0 and the setup state once per execution."""
        self.rank_init(mRows, seqlen_kv).launch(grid=[RANK_INIT_CTAS, 1, 1], block=[RANK_INIT_THREADS, 1, 1], stream=stream)

    @cute.jit
    def launch_rank_wave_setup(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        step: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        next_begin: Int32,
        next_size: Int32,
        early: Int32,
        stride: Int32,
        stream: cuda.CUstream,
    ):
        """Launch one pipelined wave setup as a programmatic dependent of the previous launch."""
        # One slot per thread: narrower top-k widths launch fewer CTAs, which leave the SMs sooner.
        setup_ctas = min(RANK_SETUP_CTAS, -(-self.max_query_pairs_per_wave * self.max_topk // RANK_SETUP_THREADS))
        self.rank_wave_setup(
            mRows, mTopkIdxs, mTopkLength, seqlen_kv, step, prev_begin, prev_size, cur_begin, cur_size, next_begin, next_size, early, stride
        ).launch(
            grid=[setup_ctas, 1, 1],
            block=[RANK_SETUP_THREADS, 1, 1],
            smem=RANK_SETUP_SMEM_BYTES,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=RANK_SETUP_MIN_BLOCKS,
        )

    @cute.jit
    def launch_rank_final(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        tail_step: Int32,
        tail_begin: Int32,
        tail_size: Int32,
        tail_stride: Int32,
        sum_odo: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        problem_shape,
        stream: cuda.CUstream,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Launch the last wave's tail with the dSink partials as a programmatic dependent."""
        self.rank_final(
            mRows, mTopkIdxs, mTopkLength, seqlen_kv, tail_step, tail_begin, tail_size, tail_stride, sum_odo, scaled_lse, attn_sink, problem_shape, chunk_size
        ).launch(
            grid=[RANK_SETUP_CTAS, 1, 1],
            block=[RANK_SETUP_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=RANK_SETUP_MIN_BLOCKS,
        )

    @cute.jit
    def launch_rank_fold(
        self,
        mRows: cute.Tensor,
        mdKV: cute.Tensor,
        seqlen: Int32,
        sum_odo: cute.Tensor,
        d_sink: cute.Tensor,
        problem_shape,
        stream: cuda.CUstream,
        chunk_size: cutlass.Constexpr[int],
    ):
        """Launch the terminal used-plane fold and dSink finalization."""
        self.fold_rank_convert_dkv(mRows, mdKV, seqlen, sum_odo, d_sink, problem_shape, chunk_size).launch(
            grid=[min(cute.ceil_div(seqlen, RANK_FOLD_ROWS), Int32(RANK_FOLD_MAX_CTAS)), 1, 1], block=[32, RANK_FOLD_ROWS, 1], stream=stream, use_pdl=True
        )


@dsl_user_op
def rank_pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Round two FP32 values into one BF16x2 word, low logical value first."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def rank_store_bf16x4(destination: cute.Pointer, packed01: Uint32, packed23: Uint32, *, loc=None, ip=None) -> None:
    """Store four adjacent BF16 outputs with one ordinary 64-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Uint32(packed01).ir_value(loc=loc, ip=ip), Uint32(packed23).ir_value(loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
