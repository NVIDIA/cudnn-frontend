# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conflict-rank dKV storage for deterministic two-CTA DSA backward (H128 D576).

A query wave of ``W`` query pairs (``rounds`` rounds of the cluster pairs the
device runs at once) accumulates FP32 dKV records. Within one wave, the
queries that select a KV row are ranked by wave-local ordinal. The query with
conflict rank ``k < P`` adds into dense plane ``k``; higher ranks store their
record into a compact overflow area. Each (plane, row) and each overflow
record therefore has one writer per wave, so the main kernel needs no writer
ordering. A per-wave tail adds overflow records into plane 0 in rank order,
and a fixed-order fold combines the ``P`` planes.

Rows selected more than once by one query use per-slot overflow records (every
slot of the wave that selects such a row). The tail adds them in ascending
(query, slot) order; sorting plans order a query's slots by sorted position.

Dense planes. ``P`` is the larger of the expected per-row load of one wave,
``floor(1.25 * rounds * RANK_LOAD_WAVE * K / S_kv + 1.5)`` (within a 1 GiB dense
budget), and the planes that fit in the workspace budget ``min(1 GiB - 32 MiB,
private-plane bytes - 64 KiB)`` next to the fixed regions of the host-sized wave,
clamped to ``[1, W]`` (``conflict_rank_planes``; ``RankShardReduction.rank_geometry``
evaluates the same formula). Ranks beyond ``P`` remain exact through the
overflow records, so ``P`` only trades memory for tail work.

Plane 0 is cleared once per execution. Planes ``k >= 1`` are never cleared:
each row keeps a high-water mark ``h`` of the planes written so far, the first
writer of ``(k, row)`` (``k >= h``) stores its record instead of adding it, and
the fold reads only the row's first ``max(h, 1)`` planes. The mark is one
64-bit entry per row, ``(h before the stamped wave << 16 | h after it, wave
stamp)``, updated by the row's rank-0 writer while the other writers of the same
wave read the ``before`` value, so every writer of one wave sees the same mark.

Destination codes (one per wave slot): ``r < P * pitch`` adds into dense record
``r``; ``P * pitch + j`` (below ``RANK_FIRST_WRITE``) stores duplicate record
``j``; ``RANK_FIRST_WRITE + r`` stores the first write of dense record ``r``;
``-2 - j`` stores the ``j``-th overflow record of the slot's row.

Overflow tail. Every overflow row is split into 16-byte columns. Chains of at
most ``RANK_TAIL_LONG`` records: one thread sums one column of the row's records
in ascending rank order and adds the sum to plane 0. Longer chains (rows shared
by most of a wave, such as sink rows) are listed from the end of the row list;
when their (row, 32-column group) items fit in one pass of the setup grid, one
CTA takes an item, each warp sums one contiguous rank segment, and the segment
sums are added in segment order; otherwise they join the per-thread units. A
plan's setup grid follows only its wave and top-k width (thin grids launch one
CTA per cluster pair of a round, sorting plans one sort CTA per wave query), so
each plan's order is fixed.

Launches of one execution: the workspace init (``rank_init``) and, for plans that
would stride their waves, the wave-order probe (``rank_probe``) are ordinary launches; then a
leading setup launch records wave 0's ordinal bits (and sorts wave 0's rows),
each wave ``w`` runs one setup launch and one main kernel, and a trailing setup
launch finishes the last wave before the fold. Setup launch ``w`` (``rank_setup``)
clears wave ``w - 1``'s row state, writes wave ``w``'s destination codes and
allocates its overflow records, records wave ``w + 1``'s ordinal bits, sorts wave
``w + 1``'s rows, resets the counters two waves ahead, and after its grid
dependency wait adds wave ``w - 1``'s overflow and duplicate records to plane 0.
Row state and sorted rows rotate through three parities, counters and
duplicate lists through four; codes, overflow offsets, duplicate records and
overflow row lists through two.

Wave orders. Wave ``w`` holds queries ``begin(w) + c * stride`` for its clusters
``c``: consecutive blocks of ``W`` queries, or (``3 <= P`` and ``8 P < 3 W``; top-k
128 one-round plans from ``P >= 3``) ``W`` queries strided by the wave count, which
spreads rows that nearby queries share over many waves. Strided orders with an
even wave count use the skewed order (negative stride). One-round plans wider
than top-k 128 that would stride first probe one consecutive wave and take
consecutive waves when fewer than ``RANK_PROBE_EXCESS_NUM / RANK_PROBE_EXCESS_DEN``
of its active slots land beyond the dense planes.

Sorted rows. Plans with top-k ``>= RANK_SORT_MIN_TOPK`` sort every query's
top-k rows by KV range: the leading CTAs of setup launch ``w`` (one per query of
wave ``w + 1``) write wave ``w + 1``'s rows, and codes, tiles and the duplicate
tail index slots by sorted position. Plans with top-k lengths and top-k
``>= RANK_SORT_ROTATE_MIN_TOPK`` rotate the bucket order per wave
(``RANK_SORT_ROTATE_TILES``); narrower sorting plans alternate the walk direction
of consecutive waves instead (the kernel's ``_walk_alternates``).

Multi-round waves. ``rank_wave_rounds`` widens top-k 512 waves to three rounds
of cluster pairs, and top-k 128 waves whose one-round layout keeps fewer than
``RANK_NARROW_STRIDED_PLANES`` planes to two. (Its three-round top-k 128 case
needs the three-round layout to keep a round's worth of planes, which no KV
extent long enough to leave the one-round layout below three planes reaches.)
The ordinal bits of a wave must fit ``RANK_MAX_ORDINAL_WORDS`` words, so the
interface clamps the pairs per round (``rank_max_round_pairs``).

Setup grids. Top-k 128 plans launch thin setup CTAs (``RANK_THIN_*``) that fit
beside resident main CTAs. Wider plans launch ``RANK_SETUP_WIDE_CTAS`` CTAs: the
sort CTAs of sorting plans, ``RANK_SETUP_WORK_CTAS`` slot-phase CTAs, and
tail-only CTAs for the rest, which join a non-empty overflow tail and otherwise
leave at once. Below top-k ``RANK_SETUP_TAIL_TOPK``, plans with ``2 P >= W`` bound
each row's overflow chain by half a wave and launch only the sort CTAs and
``rank_setup_slot_ctas`` slot-phase CTAs.

Launch ordering (programmatic dependent launch). Every setup launch and main
kernel is a programmatic dependent of the launch before it.

- Setup launches with ``early == 0`` (the leading setup and wave 0's) wait for
  the preceding launch before their phases and before they trigger their
  dependents. Wave 0's slot phases read the ordinal bits the leading setup
  records, and main kernel 0, which may start as soon as wave 0's setup
  triggered, reads wave 0's sorted rows (written by the leading setup) without
  a wait: triggering before the wait would let main kernel 0 read sorted rows
  the leading setup has not finished writing.
- Later setup launches trigger at entry and run their slot phases and sorts
  before the wait: these read only buffers that the setup before the preceding
  main kernel wrote, which completed before that main kernel triggered this
  launch, and write only parities the preceding main kernel does not read.
  Their tail runs after the wait. CTAs without tail work skip the wait; CTA 0
  always waits, so a setup launch completes only after the preceding main kernel.
- In a main kernel the reducer warps wait, then trigger the next launch; the
  gather warps wait inside the dK tail, before they read destination codes or
  write records. The math, MMA and load warps never wait. Of the workspace,
  they, the reducer warps before their wait and the gather warps before the dK
  tail read only the sorted rows, the probe words and the precomputed O*dO
  statistics, written by launches that completed before this kernel started
  (for wave ``w > 0``, setup ``w - 1`` completed before main kernel ``w - 1``
  triggered setup ``w``; the probe and the O*dO precompute are ordinary
  launches before the first setup).

D576 workspace (32-bit words, ``pitch = round_up(S_kv, 8)``, ``R`` = record width,
``S = W * K``, regions in this order, see ``rank_offsets``):

- dense planes: ``P * pitch * R``
- overflow records: ``S * R`` (the probe counts writers per row in its first ``pitch``
  words before any wave writes a record)
- plane high-water marks: ``2 * pitch`` (one 64-bit entry per row)
- row state, three wave parities: ``3 * pitch * rank_state_words(W)`` (``ceil(W / 32)``
  ordinal words and one duplicate flag in the last word: 4 words for waves up to
  96 pairs, 8 up to 224)
- overflow offsets, two parities: ``2 * pitch``
- per-slot duplicate records, two parities: ``2 * S``
- overflow row lists, two parities: ``2 * 4 * S`` (row, first record, record count, padding)
- duplicate row lists, four parities: ``4 * S``
- per-slot destination codes, two parities: ``2 * S``
- counter region: ``RANK_COUNTER_REGION_WORDS`` (four parities of overflow rows,
  overflow records, duplicate rows and long-chain rows, then the probe's
  beyond-plane and active slot counts and two padding words)
- sorted top-k rows (sorting plans), three parities: ``3 * S``

Invariants (asserted by ``conflict_rank_words``): duplicate codes stay below
``RANK_FIRST_WRITE`` (``P * pitch + S <= RANK_FIRST_WRITE``), plane counts fit the
16-bit mark fields, and a probe needs an overflow area of at least ``pitch`` words.
"""

from typing import Optional
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.typing import Float32, Int32, Int64, Uint32
from cudnn._cutlass_compat import SmemAllocator

RANK_DENSE_BUDGET_BYTES = 1 << 30
RANK_STATE_WORDS = 4
RANK_ORDINAL_WORDS = 3
RANK_MAX_WAVE = RANK_ORDINAL_WORDS * 32
# SM count of the device the setup, probe and init grid sizes below are tuned for
# (B200). They are tuning constants: these kernels walk their work in grid-stride
# loops, so every grid size is correct on every device. The grid size also picks how
# a setup splits long overflow chains (whole CTAs or per-thread units), which fixes
# their summation order: a given plan stays bitwise reproducible, and keeping these
# constants device-independent keeps the setup grids of equal waves equal.
RANK_TUNED_SM_COUNT = 148
# CTAs that run the slot phases (two per SM).
RANK_SETUP_WORK_CTAS = 2 * RANK_TUNED_SM_COUNT
# Fewest slot-phase CTAs of a wide setup sized at about two wave slots per thread
# (``rank_setup_slot_ctas``): at least one CTA per SM pair, which leaves SMs free for
# the next main kernel.
RANK_SETUP_NARROW_MIN_CTAS = RANK_TUNED_SM_COUNT // 2
# Top-k widths of 512 and more launch RANK_SETUP_WIDE_CTAS setup CTAs: the sort
# CTAs of sorting plans, RANK_SETUP_WORK_CTAS slot-phase CTAs, and tail-only CTAs
# for the rest, which join a non-empty overflow tail and otherwise leave at once
# (see ``launch_rank_setup`` for the plans without tail-only CTAs); the launch caps
# registers so four setup CTAs stay resident per SM.
RANK_SETUP_WIDE_TOPK = 512
RANK_SETUP_WIDE_MIN_BLOCKS = 4
RANK_SETUP_WIDE_CTAS = RANK_SETUP_WIDE_MIN_BLOCKS * RANK_TUNED_SM_COUNT
# From this top-k width on, the wide launch always keeps its tail-only CTAs.
RANK_SETUP_TAIL_TOPK = 2048
RANK_SETUP_THREADS = 256
# Long-chain segment sums: four FP32 values per lane and warp.
RANK_SETUP_SMEM_BYTES = 4 * 4 * RANK_SETUP_THREADS
# Thin setup launches (top-k widths up to RANK_THIN_MAX_TOPK): CTAs small enough to stay
# resident beside a main-kernel CTA (whose 20 warps of 96 registers leave 1024
# registers per sub-partition and ~4 KB of shared memory), one warp of at most 32
# registers per sub-partition. Wave i + 1's setup runs while main kernel i still
# occupies every SM, so main kernel i + 1's CTAs start as soon as i's exit.
RANK_THIN_THREADS = 128
# Caps the thin setup at 32 registers per thread (four warps of 1024 registers).
RANK_THIN_MIN_BLOCKS = 16
# The thin setup prefers the maximum shared-memory carveout (the main kernel's),
# so its CTAs join SMs that run main CTAs without a carveout change.
RANK_THIN_CARVEOUT = 100
# Widest top-k that launches thin setups. Wider plans keep full setup CTAs: their
# overflow tails (after the previous main kernel) need the wider grid.
RANK_THIN_MAX_TOPK = 128
RANK_INIT_THREADS = 256
# Init grid: four CTAs per SM (tuning constant, see ``RANK_TUNED_SM_COUNT``).
RANK_INIT_CTAS = 4 * RANK_TUNED_SM_COUNT
# D576 conflict-rank layout: parity counts of the rotating setup buffers.
RANK_STATE_PARITIES = 3
RANK_LIST_PARITIES = 4
RANK_COUNTER_STRIDE = 4
RANK_SETUP_COUNTER_WORDS = RANK_LIST_PARITIES * RANK_COUNTER_STRIDE
# Wave-order probe: the words after the per-parity counters hold one probe
# wave's writers beyond the dense planes and its active slots (consecutive
# order); the whole counter region is cleared once per execution.
RANK_PROBE_WORDS = 4
RANK_COUNTER_REGION_WORDS = RANK_SETUP_COUNTER_WORDS + RANK_PROBE_WORDS
# Plans that may stride their waves take consecutive waves instead when fewer
# than this fraction of the probe wave's active slots land beyond the dense
# planes: consecutive waves keep the KV rows and dense records that nearby
# queries share close in time, while heavily overlapping rows (sliding
# windows) would overflow.
RANK_PROBE_EXCESS_NUM = 2
RANK_PROBE_EXCESS_DEN = 5
# Probe grid: one CTA per SM (tuning constant, see ``RANK_TUNED_SM_COUNT``).
RANK_PROBE_CTAS = RANK_TUNED_SM_COUNT
# Keeps every rotating parity non-negative: a multiple of 2, 3 and 4.
RANK_STEP_BIAS = 12
# Wave width of the D576 plane-count load estimate. Host sizing reserves the
# maximum wave while devices launch at most that many pairs, so a fixed load
# wave keeps the host and device plane counts identical below the clamp.
RANK_LOAD_WAVE = 74
# Host sizing reserves the largest wave; devices launch at most that many pairs.
RANK_HOST_WAVE = 76
# Total D576 dKV workspace the plane budget fills (1 GiB less a 32 MiB margin).
RANK_TOTAL_BUDGET_BYTES = (1 << 30) - (32 << 20)
# Bytes per KV row of the private-plane layout (96 D576 FP32 planes); the plane
# budget stays RANK_PRIVATE_MARGIN_BYTES below it, so the workspace never exceeds it.
RANK_PRIVATE_ROW_BYTES = 96 * 576 * 4
RANK_PRIVATE_MARGIN_BYTES = 64 << 10
# Words of one row's high-water-mark entry (a 64-bit mark and wave stamp).
RANK_HWM_WORDS = 2
# Sorted top-k rows: each wave's setup sorts the next wave's queries (one CTA per
# query) into one of three parities of ``W * K`` words; codes, tiles and the
# duplicate tail index slots by sorted position, so a tile's dKV records sit
# close together in every plane whatever order the caller's top-k uses.
RANK_SORT_PARITIES = 3
# Top-k widths that sort (many tiles per query). One sort CTA per wave query
# precedes the slot-phase CTAs, so the scheduler places the sorts first, on the
# SMs the previous main kernel frees first, overlapping its last CTAs.
RANK_SORT_MIN_TOPK = 1024
# KV-range buckets of the per-query sort.
RANK_SORT_BUCKETS = 32
# Wave v's sort rotates the bucket order by v * RANK_SORT_ROTATE_TILES tiles, so each
# walk (plans with top-k lengths) starts on the bands the previous wave walked last,
# still in L2. Top-k 1024/1152 plans alternate their walk direction instead
# (``WALK_ALTERNATE_MAX_TOPK`` in the kernel module).
RANK_SORT_ROTATE_TILES = 4
RANK_SORT_ROTATE_MIN_TOPK = 2048
# Destination codes at or above this bias are first writes of a dense record (store, not add).
RANK_FIRST_WRITE = 1 << 30
# Overflow records loaded ahead of their in-order additions in the tail.
RANK_TAIL_BATCH = 8
# Longer overflow chains are summed by whole setup CTAs (each warp one contiguous
# segment for 32 sixteen-byte columns, segment sums added to plane 0 in segment
# order); shorter chains keep one thread per (row, column).
RANK_TAIL_LONG = 16
RANK_TAIL_GROUP_COLUMNS = 32
# D576 top-k widths in [RANK_ROUNDS_MIN_TOPK, RANK_ROUNDS_MAX_TOPK] serve
# RANK_WIDE_ROUNDS rounds of cluster pairs per wave (one setup and one main launch
# per round group; ordinal masks widen to the larger wave). Wider top-k keeps one
# round: the doubled overflow area costs dense planes and wave-shared rows (sink
# prefixes) grow longer overflow chains than the setup saves.
RANK_ROUNDS_MIN_TOPK = 512
RANK_ROUNDS_MAX_TOPK = 512
RANK_WIDE_ROUNDS = 3
# Top-k widths up to this one serve one round per wave when the one-round layout
# keeps at least RANK_NARROW_STRIDED_PLANES dense planes, with the strided wave
# order (rows nearby queries share spread over many waves, so short waves avoid
# overflow chains); fewer planes serve RANK_NARROW_ROUNDS rounds per wave.
RANK_NARROW_ROUNDS_TOPK = 128
RANK_NARROW_ROUNDS = 2
RANK_NARROW_STRIDED_PLANES = 3
# Narrow waves would widen further to this many rounds while the dense planes of
# the wider layout still cover one round of cluster pairs; the KV extents that
# leave the one-round layout below RANK_NARROW_STRIDED_PLANES planes are too long
# for that, so top-k 128 plans take one or two rounds.
RANK_NARROW_WIDE_ROUNDS = 3
# D576 conflict-rank record width in FP32 values: 512 core values and two
# rank-private 64-value tails.
RANK_D576_RECORD = 640
# Top-k widths above RANK_TWO_TAIL_MAX_TOPK use single-tail records: the two
# CTAs of a pair combine their D64 tail partials before writing (their tails
# overlap the next tile's score and dP MMAs). Top-k 128 runs each tile's tail
# inline, where the pair exchange would add its latency, and keeps two tails.
RANK_D576_SINGLE_TAIL_RECORD = 576
# Widest top-k whose records keep two rank-private tails (``RANK_D576_RECORD``).
RANK_TWO_TAIL_MAX_TOPK = 128


def rank_d576_record(max_topk: int) -> int:
    """Return the D576 conflict-rank record width for one top-k width."""
    return RANK_D576_SINGLE_TAIL_RECORD if int(max_topk) > RANK_TWO_TAIL_MAX_TOPK else RANK_D576_RECORD


def rank_wave_rounds(max_topk: int, total_s_kv: Optional[int] = None) -> int:
    """Return the rounds of cluster pairs one D576 wave serves for a top-k width and KV extent."""
    if RANK_ROUNDS_MIN_TOPK <= int(max_topk) <= RANK_ROUNDS_MAX_TOPK:
        return RANK_WIDE_ROUNDS
    if int(max_topk) <= RANK_NARROW_ROUNDS_TOPK:
        if total_s_kv is not None:
            if _conflict_rank_planes_for(int(total_s_kv), int(max_topk), RANK_D576_RECORD, RANK_HOST_WAVE, 1) >= RANK_NARROW_STRIDED_PLANES:
                return 1
            wide_planes = _conflict_rank_planes_for(int(total_s_kv), int(max_topk), RANK_D576_RECORD, RANK_HOST_WAVE, RANK_NARROW_WIDE_ROUNDS)
            if wide_planes >= RANK_LOAD_WAVE:
                return RANK_NARROW_WIDE_ROUNDS
        return RANK_NARROW_ROUNDS
    return 1


# Ordinal-bit words of one row-state entry that the setup kernel reads (``state_0``
# .. ``state_6``): a wave holds at most 7 * 32 = 224 cluster pairs.
RANK_MAX_ORDINAL_WORDS = 7


def rank_ordinal_words(wave: int) -> int:
    """Return the 32-bit ordinal-bit words of one D576 row-state entry."""
    return (int(wave) + 31) // 32


def rank_max_round_pairs(rounds: int) -> int:
    """Return the most cluster pairs one round of a ``rounds``-round wave may launch.

    The wave's ordinal bits must fit ``RANK_MAX_ORDINAL_WORDS`` words: three-round
    waves take at most 74 pairs per round (the pairs of a 148-SM device), two-round
    waves at most 112, one-round waves 224.
    """
    return RANK_MAX_ORDINAL_WORDS * 32 // max(1, int(rounds))


def rank_state_words(wave: int) -> int:
    """Return the D576 row-state entry width: ordinal words and the duplicate flag, padded to 16 bytes."""
    return (rank_ordinal_words(wave) + 4) // 4 * 4


def rank_pitch(total_s_kv: int) -> int:
    """Return the padded KV row count of one plane."""
    return (int(total_s_kv) + 7) // 8 * 8


def rank_load_planes(total_s_kv: int, max_topk: int, record: int, wave: int) -> int:
    """Return the load-sized D576 dense plane count of the selection layout (``_load_sized_conflict_rank_words``).

    The expected per-row load of one wave, ``floor(1.25 * RANK_LOAD_WAVE * K / S_kv + 1.5)``,
    within ``[1, wave - 1]`` and the dense budget.
    """
    total_s_kv = int(total_s_kv)
    plane_bytes = rank_pitch(total_s_kv) * record * 4
    load = (5 * RANK_LOAD_WAVE * int(max_topk) + 6 * total_s_kv) // (4 * total_s_kv)
    return max(1, min(wave - 1, RANK_DENSE_BUDGET_BYTES // plane_bytes, load))


def _conflict_rank_fixed_words(pitch: int, max_topk: int, record: int, wave: int) -> int:
    """Return the 32-bit words of every D576 region except the dense planes (see the module docstring).

    ``RankShardReduction.rank_geometry`` repeats this sum in device arithmetic; keep both in sync.
    """
    slots = wave * max_topk
    return (
        slots * record
        + RANK_HWM_WORDS * pitch
        + RANK_STATE_PARITIES * pitch * rank_state_words(wave)
        + 2 * pitch
        + 2 * slots
        + 2 * 4 * slots
        + RANK_LIST_PARITIES * slots
        + 2 * slots
        + RANK_COUNTER_REGION_WORDS
        + (RANK_SORT_PARITIES * slots if max_topk >= RANK_SORT_MIN_TOPK else 0)
    )


def conflict_rank_planes(total_s_kv: int, max_topk: int, record: int, wave: int) -> int:
    """Return the D576 dense plane count: the larger of the load estimate and the workspace budget, within ``[1, wave]``.

    The budget fills ``min(RANK_TOTAL_BUDGET_BYTES, private-plane bytes - margin)``
    with the host-wave layout, so the device plane count (same formula, clamped
    to the launched wave) never exceeds the host allocation.
    """
    return _conflict_rank_planes_for(total_s_kv, max_topk, record, wave, rank_wave_rounds(max_topk, total_s_kv))


def _conflict_rank_planes_for(total_s_kv: int, max_topk: int, record: int, wave: int, rounds: int) -> int:
    """Return the D576 dense plane count of a layout whose waves serve ``rounds`` rounds.

    ``RankShardReduction.rank_geometry`` evaluates the same formula on the device
    (64-bit integer arithmetic); keep both in sync.
    """
    total_s_kv = int(total_s_kv)
    pitch = rank_pitch(total_s_kv)
    plane_bytes = pitch * record * 4
    load = (5 * rounds * RANK_LOAD_WAVE * int(max_topk) + 6 * total_s_kv) // (4 * total_s_kv)
    load = min(load, RANK_DENSE_BUDGET_BYTES // plane_bytes)
    budget_bytes = min(RANK_TOTAL_BUDGET_BYTES, pitch * RANK_PRIVATE_ROW_BYTES - RANK_PRIVATE_MARGIN_BYTES)
    budget = (budget_bytes - 4 * _conflict_rank_fixed_words(pitch, int(max_topk), record, rounds * RANK_HOST_WAVE)) // plane_bytes
    return max(1, min(rounds * wave, max(load, budget)))


def conflict_rank_words(total_s_kv: int, max_topk: int, record: int, wave: int, launch_pairs: Optional[int] = None) -> int:
    """Return the 32-bit word count of the D576 conflict-rank dKV workspace (``wave`` counts cluster pairs).

    The dense plane count keeps the ``wave`` (host-wave) budget. The per-slot
    regions (overflow records, row state, lists, codes, sorted rows) are sized
    for ``launch_pairs`` cluster pairs per round when given: the pairs the
    plan's device launches (``min(SMs / 2, wave)``, at most
    ``rank_max_round_pairs(rounds)``), whose layout the kernels
    address. The device clamps its plane count to the same launched wave, so
    every region fits.

    The asserts check the layout invariants of the module docstring for this
    (largest) plane count and slot count, which bound the device's.
    """
    pitch = rank_pitch(total_s_kv)
    planes = conflict_rank_planes(total_s_kv, max_topk, record, wave)
    slot_wave = int(wave) if launch_pairs is None else max(1, min(int(wave), int(launch_pairs)))
    rounds = rank_wave_rounds(max_topk, total_s_kv)
    slots = rounds * slot_wave * int(max_topk)
    assert planes * pitch + slots <= RANK_FIRST_WRITE, "duplicate destination codes must stay below RANK_FIRST_WRITE"
    assert planes < 1 << 16, "plane counts must fit the 16-bit high-water-mark fields"
    if rounds == 1 and int(max_topk) > RANK_NARROW_ROUNDS_TOPK and planes > 2:
        # Plans that may stride their waves probe one wave, counting writers per row in the overflow area.
        assert slots * record >= pitch, "the wave-order probe needs an overflow area of at least pitch words"
    return planes * pitch * record + _conflict_rank_fixed_words(pitch, int(max_topk), record, rounds * slot_wave)


def conflict_rank_rows(total_s_kv: int, max_topk: int, record: int, wave: int, launch_pairs: Optional[int] = None) -> int:
    """Return the D576 workspace size in whole records."""
    return -(-conflict_rank_words(total_s_kv, max_topk, record, wave, launch_pairs) // record)


def _load_sized_conflict_rank_words(total_s_kv: int, max_topk: int, record: int, wave: int) -> int:
    """Return the word count of the load-sized selection layout that decides the storage selection.

    ``rank_load_planes`` dense planes, four-word row state, and no marks, probe
    words or sorted rows: a fixed reference size, not the allocated layout.
    """
    pitch = rank_pitch(total_s_kv)
    planes = rank_load_planes(total_s_kv, max_topk, record, wave)
    slots = wave * max_topk
    return (
        planes * pitch * record
        + slots * record
        + RANK_STATE_PARITIES * pitch * RANK_STATE_WORDS
        + 2 * pitch
        + 2 * slots
        + 2 * 4 * slots
        + RANK_LIST_PARITIES * slots
        + 2 * slots
        + RANK_SETUP_COUNTER_WORDS
    )


def conflict_rank_selected(total_s_kv: int, max_topk: int, record: int, wave: int, private_bytes: int) -> bool:
    """Whether D576 conflict-rank storage replaces the ``private_bytes`` private-plane layout.

    Two-tail records (top-k 128) require the load-sized layout to need at most
    3/4 of the private-plane layout; single-tail records (top-k above 128) skip
    that rule. Either way the allocated layout (``conflict_rank_rows``), which
    grows its plane count up to the workspace budget, must fit the private-plane
    layout.
    """
    if wave < 2 or wave > RANK_MAX_WAVE:
        return False
    load_sized_rows = -(-_load_sized_conflict_rank_words(total_s_kv, max_topk, record, wave) // record)
    # Single-tail records (top-k above 128) take conflict-rank storage wherever
    # it fits the private-plane layout: they move a tenth less dKV data per slot.
    if record != RANK_D576_SINGLE_TAIL_RECORD and 4 * load_sized_rows * record * 4 > 3 * private_bytes:
        return False
    # The multi-round layout sizes its dense planes from the larger of the load
    # estimate and the budget; keep the private-plane layout wherever that would
    # exceed it, so the workspace never grows past the private-plane layout.
    return conflict_rank_rows(total_s_kv, max_topk, record, wave) * record * 4 <= private_bytes


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
def _rank_load_mark(source: cute.Pointer, *, loc=None, ip=None):
    """Load one 64-bit high-water-mark entry with a single relaxed GPU-scope access; return (low, high) words."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.i32()] * 2)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "{ .reg .b64 t; ld.relaxed.gpu.global.b64 t, [$2]; mov.b64 {$0, $1}, t; }",
        "=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), values, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), values, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _rank_store_mark(destination: cute.Pointer, low: Int32, high: Int32, *, loc=None, ip=None) -> None:
    """Store one 64-bit high-water-mark entry with a single relaxed GPU-scope access."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Int32(low).ir_value(loc=loc, ip=ip), Int32(high).ir_value(loc=loc, ip=ip)],
        "{ .reg .b64 t; mov.b64 t, {$1, $2}; st.relaxed.gpu.global.b64 [$0], t; }",
        "l,r,r",
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
def _rank_load_f32x4_streaming(source: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned FP32 values of a read-once overflow record with the streaming cache policy."""
    source_i64 = source.toint(loc=loc, ip=ip).ir_value()
    result_type = llvm.StructType.get_literal([T.f32()] * 4)
    values = llvm.inline_asm(
        result_type,
        [source_i64],
        "ld.global.cs.v4.f32 {$0, $1, $2, $3}, [$4];",
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


# L2 eviction policy of dense-record reductions from the main kernel: each
# (plane, row) receives at most one add per wave, so keep its lines from
# displacing the operands that every tile re-reads. RANK_L2_EVICT_FIRST and
# RANK_L2_EVICT_LAST are the 64-bit ``.L2::cache_hint`` policy operands of a
# full-fraction evict-first / evict-last policy (CUTLASS ``CacheHintSm90``).
RANK_L2_EVICT_FIRST = 0x12F0000000000000


@dsl_user_op
def rank_reduce_f32x4_evict_first(target: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Atomically add four FP32 values without returning them, with an L2 evict-first hint."""
    target_i64 = target.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            target_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
            Int64(RANK_L2_EVICT_FIRST).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, $5;",
        "l,f,f,f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# L2 policy of dense reductions into the leading planes when they fit
# RANK_RESIDENT_L2_BYTES: every row a wave touches takes its rank-0 add in
# plane 0 in every wave, so those lines are revisited wave after wave.
RANK_L2_EVICT_LAST = 0x14F0000000000000
RANK_RESIDENT_L2_BYTES = 48 << 20
# KV extents whose BF16 D576 rows fit this much L2 keep their gathered rows L2-resident
# across tiles and waves (tuning constant sized for a B200 L2; any value is correct).
RANK_KV_RESIDENT_L2_BYTES = 64 << 20
# Widest top-k whose dense reductions keep the leading planes L2-resident.
RANK_RESIDENT_MAX_TOPK = 128


@dsl_user_op
def rank_reduce_f32x4_evict_last(target: cute.Pointer, value_0: Float32, value_1: Float32, value_2: Float32, value_3: Float32, *, loc=None, ip=None) -> None:
    """Atomically add four FP32 values without returning them, with an L2 evict-last hint."""
    target_i64 = target.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            target_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Float32(value_2).ir_value(loc=loc, ip=ip),
            Float32(value_3).ir_value(loc=loc, ip=ip),
            Int64(RANK_L2_EVICT_LAST).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, $5;",
        "l,f,f,f,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def rank_reduce_f32x2_evict_first(target: cute.Pointer, value_0: Float32, value_1: Float32, *, loc=None, ip=None) -> None:
    """Atomically add two FP32 values without returning them, with an L2 evict-first hint."""
    target_i64 = target.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            target_i64,
            Float32(value_0).ir_value(loc=loc, ip=ip),
            Float32(value_1).ir_value(loc=loc, ip=ip),
            Int64(RANK_L2_EVICT_FIRST).ir_value(loc=loc, ip=ip),
        ],
        "red.global.add.L2::cache_hint.v2.f32 [$0], {$1, $2}, $3;",
        "l,f,f,l",
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
def rank_wave_query(begin: Int32, ordinal: Int32, stride: Int32) -> Int32:
    """Return the query of wave position ``ordinal``: ``begin + ordinal * stride``.

    A negative ``stride`` selects skewed strided waves of period ``-stride``:
    position ``c`` of wave ``begin`` takes query ``c * period + (begin + c) % period``.
    Every block of ``period`` queries then holds one query of each wave, and
    queries of one wave sit ``period + 1`` apart (odd for an even period), so
    index patterns with a power-of-two period do not align across a wave.
    """
    query = begin + ordinal * stride
    if stride < Int32(0):
        period = Int32(0) - stride
        query = ordinal * period + (begin + ordinal) % period
    return query


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


class RankShardReduction:
    """Workspace geometry, wave setup, and destination records for conflict-rank dKV storage.

    The host class provides ``rank_record``, ``rank_rounds``, ``max_query_pairs_per_wave``
    (the wave ``W``: ``rank_rounds`` rounds of cluster pairs) and ``max_topk``.
    """

    rank_shard = True
    # Launch setup and main kernels with programmatic dependent launch.
    rank_pdl = True

    @cute.jit
    def rank_geometry(self, seqlen_kv: Int32):
        """Return the padded plane pitch and the dense plane count for one KV extent.

        Mirrors ``conflict_rank_planes`` and ``_conflict_rank_fixed_words``
        (host-wave budget, device-wave clamp) in 64-bit arithmetic; keep them in
        sync. Evaluated once per execution on the host side of the launch;
        device code receives the plane count as a kernel argument.
        """
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        record = Int64(self.rank_record)
        plane_bytes = pitch * record * Int64(4)
        rounds = self.rank_rounds
        load = (Int64(5 * rounds * RANK_LOAD_WAVE * self.max_topk) + Int64(6) * Int64(seqlen_kv)) // (Int64(4) * Int64(seqlen_kv))
        load = min(load, Int64(RANK_DENSE_BUDGET_BYTES) // plane_bytes)
        host_wave = rounds * RANK_HOST_WAVE
        host_slots = Int64(host_wave * self.max_topk)
        fixed_words = (
            host_slots * record
            + Int64(RANK_HWM_WORDS + RANK_STATE_PARITIES * rank_state_words(host_wave) + 2) * pitch
            + Int64(2 + 2 * 4 + RANK_LIST_PARITIES + 2 + (RANK_SORT_PARITIES if self.max_topk >= RANK_SORT_MIN_TOPK else 0)) * host_slots
            + Int64(RANK_COUNTER_REGION_WORDS)
        )
        budget_bytes = min(Int64(RANK_TOTAL_BUDGET_BYTES), pitch * Int64(RANK_PRIVATE_ROW_BYTES) - Int64(RANK_PRIVATE_MARGIN_BYTES))
        budget = (budget_bytes - Int64(4) * fixed_words) // plane_bytes
        planes = max(load, budget)
        planes = min(planes, Int64(self.max_query_pairs_per_wave))
        planes = max(planes, Int64(1))
        return pitch, Int32(planes)

    @cute.jit
    def rank_offsets(self, pitch: Int64, planes: Int32):
        """Return the 32-bit word offsets of every workspace region (parity 0 of each rotating buffer)."""
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        record = Int64(self.rank_record)
        records = Int64(planes) * pitch * record
        marks = records + slots * record
        state = marks + Int64(RANK_HWM_WORDS) * pitch
        offsets = state + Int64(RANK_STATE_PARITIES * rank_state_words(self.max_query_pairs_per_wave)) * pitch
        dup_records = offsets + Int64(2) * pitch
        row_list = dup_records + Int64(2) * slots
        dup_list = row_list + Int64(2 * 4) * slots
        codes = dup_list + Int64(RANK_LIST_PARITIES) * slots
        counters = codes + Int64(2) * slots
        return records, state, offsets, dup_records, row_list, dup_list, counters, codes, marks

    @cute.jit
    def rank_sorted_base(self, pitch: Int64, planes: Int32, wave_step: Int32) -> Int64:
        """Return the word offset of the sorted top-k rows of the wave with setup step ``wave_step`` (its parity)."""
        counters = self.rank_offsets(pitch, planes)[6]
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        return counters + Int64(RANK_COUNTER_REGION_WORDS) + Int64(wave_step % Int32(RANK_SORT_PARITIES)) * slots

    @cute.jit
    def rank_sorted_view(self, mRows: cute.Tensor, mTopkIdxs: cute.Tensor, seqlen_kv: Int32, planes: Int32, wave: Int32, ordinal: Int32):
        """View one wave position's sorted top-k rows with ``mTopkIdxs``'s (slot, (token, batch)) coordinates.

        The token mode has stride zero: every token coordinate reads the rows of
        wave ``wave`` position ``ordinal``, which the setup launch of wave
        ``wave - 1`` sorted (for wave 0, the leading setup launch). Main kernels
        read them without a grid dependency wait: for ``wave > 0`` that setup
        launch completed before main kernel ``wave - 1`` triggered wave ``wave``'s
        setup, and wave 0's setup waits for the leading setup before it triggers
        (``early == 0`` in ``rank_setup``).
        """
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        base = self.rank_sorted_base(pitch, planes, wave + Int32(RANK_STEP_BIAS)) + Int64(ordinal) * Int64(self.max_topk)
        rows = cute.recast_ptr(mRows.iterator, dtype=Int32) + base
        return cute.make_tensor(rows, cute.make_layout((mTopkIdxs.shape[0], mTopkIdxs.shape[1]), stride=(1, (0, 0))))

    @cute.jit
    def _rank_sort_query(
        self,
        counts: cute.Tensor,
        sums: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        query: Int32,
        seqlen_kv: Int32,
        destination: cute.Pointer,
        tidx: Int32,
        rotation: Int32,
    ):
        """Order one query's active top-k rows by KV range with a stable counting sort; later slots hold -1.

        Rows already in bucket order are stored in place without the sort.

        Valid rows go to bucket ``row * RANK_SORT_BUCKETS // S_kv``, invalid
        active ids to bucket 0 and inactive slots to the last bucket. Thread
        ``t`` owns slots ``[t * n, (t + 1) * n)``; per-thread bucket counts are
        scanned in bucket-major order, so equal buckets keep slot order and
        the result does not depend on scheduling. ``rotation`` cyclically shifts
        the KV bucket order (bucket ``b`` sorts as ``(b + rotation) % RANK_SORT_BUCKETS``).
        """
        per = self.rank_sort_width() // RANK_SETUP_THREADS
        buckets_total = RANK_SORT_BUCKETS + 1
        topk = _rank_topk(mTopkIdxs, mTopkLength, query)
        rows = cute.make_rmem_tensor((per,), Int32)
        buckets = cute.make_rmem_tensor((per,), Int32)
        for item in cutlass.range_constexpr(per):
            position = tidx * Int32(per) + Int32(item)
            row = Int32(-1)
            bucket = Int32(RANK_SORT_BUCKETS)
            if position < topk:
                row = mTopkIdxs[position, (query, 0)]
                bucket = Int32(0)
                if row >= Int32(0) and row < seqlen_kv:
                    bucket = Int32((Int64(row) * Int64(RANK_SORT_BUCKETS)) // Int64(seqlen_kv))
                    bucket = (bucket + rotation) % Int32(RANK_SORT_BUCKETS)
            rows[item] = row
            buckets[item] = bucket
        # Rows already in bucket order (for example sorted top-k) keep their slots.
        ordered = Int32(1)
        for item in cutlass.range_constexpr(per - 1):
            if buckets[item] > buckets[item + 1]:
                ordered = Int32(0)
        if tidx == Int32(0):
            sums[RANK_SETUP_THREADS] = Int32(1)
        sums[tidx] = buckets[0]
        cute.arch.barrier()
        if tidx + Int32(1) < Int32(RANK_SETUP_THREADS):
            if buckets[per - 1] > sums[tidx + Int32(1)]:
                ordered = Int32(0)
        if ordered == Int32(0):
            sums[RANK_SETUP_THREADS] = Int32(0)
        cute.arch.barrier()
        if sums[RANK_SETUP_THREADS] != Int32(0):
            for item in cutlass.range_constexpr(per):
                position = tidx * Int32(per) + Int32(item)
                if position < Int32(self.max_topk):
                    cute.arch.store(destination + position, rows[item])
            cute.arch.barrier()
        else:
            for bucket in cutlass.range_constexpr(buckets_total):
                counts[bucket * RANK_SETUP_THREADS + tidx] = cutlass.Int16(0)
            for item in cutlass.range_constexpr(per):
                entry = buckets[item] * Int32(RANK_SETUP_THREADS) + tidx
                counts[entry] = cutlass.Int16(Int32(counts[entry]) + Int32(1))
            cute.arch.barrier()
            # Thread t scans entries [t * B, (t + 1) * B) of the bucket-major count array.
            chunk = tidx * Int32(buckets_total)
            local = Int32(0)
            for entry in cutlass.range_constexpr(buckets_total):
                local = local + Int32(counts[chunk + Int32(entry)])
            sums[tidx] = local
            cute.arch.barrier()
            distance = 1
            while distance < RANK_SETUP_THREADS:
                value = sums[tidx]
                if tidx >= Int32(distance):
                    value = value + sums[tidx - Int32(distance)]
                cute.arch.barrier()
                sums[tidx] = value
                cute.arch.barrier()
                distance = distance * 2
            running = sums[tidx] - local
            for entry in cutlass.range_constexpr(buckets_total):
                count = Int32(counts[chunk + Int32(entry)])
                counts[chunk + Int32(entry)] = cutlass.Int16(running)
                running = running + count
            cute.arch.barrier()
            for item in cutlass.range_constexpr(per):
                entry = buckets[item] * Int32(RANK_SETUP_THREADS) + tidx
                target = Int32(counts[entry])
                counts[entry] = cutlass.Int16(target + Int32(1))
                if target < Int32(self.max_topk):
                    cute.arch.store(destination + target, rows[item])
            cute.arch.barrier()

    def rank_sorts(self) -> bool:
        """Whether this plan sorts each query's top-k rows (``RANK_SORT_MIN_TOPK``)."""
        return self.max_topk >= RANK_SORT_MIN_TOPK

    def rank_sort_rotates(self, has_length: bool) -> bool:
        """Whether consecutive waves rotate their KV-bucket sort (``RANK_SORT_ROTATE_TILES``)."""
        return self.rank_sorts() and has_length and self.max_topk >= RANK_SORT_ROTATE_MIN_TOPK

    def rank_thin(self) -> bool:
        """Whether the per-wave setup launches thin CTAs that fit beside a resident main CTA (``RANK_THIN_MAX_TOPK``).

        Thin plans never sort (``RANK_THIN_MAX_TOPK < RANK_SORT_MIN_TOPK``).
        """
        return self.max_topk <= RANK_THIN_MAX_TOPK

    def rank_threads(self) -> int:
        """Return the setup launch's threads per CTA."""
        return RANK_THIN_THREADS if self.rank_thin() else RANK_SETUP_THREADS

    def rank_tail_groups(self) -> int:
        """Return the 32-column groups of one record in the long-chain tail (the last one may be partial)."""
        return (self.rank_record // 4 + RANK_TAIL_GROUP_COLUMNS - 1) // RANK_TAIL_GROUP_COLUMNS

    def rank_tail_batch(self) -> int:
        """Return the overflow records a tail thread loads ahead of their additions (thin CTAs keep fewer registers)."""
        return RANK_TAIL_BATCH // 2 if self.rank_thin() else RANK_TAIL_BATCH

    def rank_work_ctas(self) -> int:
        """Return the leading setup CTAs that run the slot phases and duplicate rows (every CTA of a thin launch)."""
        return (1 << 24) if self.rank_thin() else RANK_SETUP_WORK_CTAS

    def rank_sort_ctas(self) -> int:
        """Return the setup grid's leading sort CTAs: one per wave query, or none."""
        return self.max_query_pairs_per_wave if self.rank_sorts() else 0

    def rank_sort_width(self) -> int:
        """Return the sorted slot span: the top-k width rounded up to a power of two (at least the CTA)."""
        width = RANK_SETUP_THREADS
        while width < self.max_topk:
            width *= 2
        return width

    def rank_setup_smem_bytes(self) -> int:
        """Return the setup launch's shared memory: the sort keys (sorting plans), then the long-chain partial sums.

        Sorting plans add the per-thread bucket counts (Int16) and the scan sums
        (Int32) of ``_rank_sort_query`` to the partial sums, each allocation
        16-byte aligned; the 256 extra bytes cover the alignment padding.
        """
        if self.rank_sorts():
            return 2 * (RANK_SORT_BUCKETS + 1) * RANK_SETUP_THREADS + 4 * (RANK_SETUP_THREADS + 4) + RANK_SETUP_SMEM_BYTES + 256
        return 4 * 4 * self.rank_threads()

    @cute.jit
    def make_rank_rows(self, workspace_dKV: cute.Tensor, seqlen_kv: Int32):
        """View the workspace as records; dense plane ``k`` row ``r`` is record ``k * pitch + r``."""
        pitch, planes = self.rank_geometry(seqlen_kv)
        rows = planes * Int32(pitch) + Int32(self.max_query_pairs_per_wave * self.max_topk)
        return cute.make_tensor(
            cute.recast_ptr(workspace_dKV.iterator, dtype=Float32),
            cute.make_layout((self.rank_record, rows, (1, 1)), stride=(1, Int64(self.rank_record), (0, 0))),
        )

    @cute.jit
    def rank_code_bases(self, seqlen_kv: Int32, planes: Int32, wave: Int32, ordinal: Int32):
        """Return the word offsets of one query's destination codes and of its wave's overflow offsets, and the overflow base.

        Batched lookups load every slot's code (its address depends only on
        the slot) before any of them is consumed, then decode with
        ``rank_decode``.
        """
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        _, _, offsets, _, _, _, _, codes, _ = self.rank_offsets(pitch, planes)
        half = Int64(wave % Int32(2))
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        code_offset = codes + half * slots + Int64(ordinal * Int32(self.max_topk))
        offset_offset = offsets + half * pitch
        return code_offset, offset_offset, planes * Int32(pitch)

    @cute.jit
    def rank_load_code(self, words: cute.Pointer, code_offset: Int64, slot: Int32) -> Int32:
        """Load one slot's destination code; the slot is clamped into the query's code row."""
        clamped = min(slot, Int32(self.max_topk - 1))
        return cute.arch.load(cute.recast_ptr(words + code_offset + Int64(clamped), dtype=Int32), Int32)

    @cute.jit
    def rank_load_first(self, words: cute.Pointer, offset_offset: Int64, row: Int32) -> Int32:
        """Load the first overflow record of one row in the query's wave."""
        return cute.arch.load(cute.recast_ptr(words + offset_offset + Int64(row), dtype=Int32), Int32)

    @cute.jit
    def rank_decode(self, code: Int32, first: Int32, overflow_base: Int32):
        """Return the destination record of one active slot and whether it adds into an initialized dense record.

        Codes are written by the setup launch of the query's wave into the
        wave-parity buffer (``rank_code_bases``): a code below the overflow base
        ``P * pitch`` is a dense record that already holds data (add); codes from
        the overflow base up to ``RANK_FIRST_WRITE`` are per-slot duplicate
        records (store); ``RANK_FIRST_WRITE + r`` is the first write of dense
        record ``r`` (store); ``-2 - j`` is the ``j``-th overflow record of the
        slot's row, whose first overflow record ``first`` (``rank_load_first``)
        is only read for these codes (store).
        """
        destination = code
        if code >= Int32(RANK_FIRST_WRITE):
            destination = code - Int32(RANK_FIRST_WRITE)
        if code < Int32(0):
            destination = overflow_base + first + (Int32(-2) - code)
        accumulate = destination < overflow_base
        if code >= Int32(RANK_FIRST_WRITE):
            accumulate = cutlass.Boolean(False)
        return destination, accumulate

    @cute.jit
    def _rank_add_records_column(self, floats: cute.Pointer, destination: Int64, first: Int64, count: Int32, column: Int32):
        """Add one 16-byte column of ``count`` consecutive overflow records to a plane-0 row in record order.

        The plane-0 value and ``tail_batch`` records are loaded before
        their additions so one thread keeps that many independent loads in
        flight; the FP32 sum keeps ascending record order and is added to the
        plane-0 value last.
        """
        record = Int64(self.rank_record)
        tail_batch = self.rank_tail_batch()
        source = floats + first * record + Int64(column * Int32(4))
        target = floats + destination * record + Int64(column * Int32(4))
        p0, p1, p2, p3 = _rank_load_f32x4(target)
        acc = cute.make_rmem_tensor((4,), Float32)
        acc.fill(0.0)
        loaded = cute.make_rmem_tensor((4 * tail_batch,), Float32)
        batches = count // Int32(tail_batch)
        for batch in cutlass.range(batches, unroll=1):
            batch_source = source + Int64(batch * Int32(tail_batch)) * record
            for item in cutlass.range_constexpr(tail_batch):
                v0, v1, v2, v3 = _rank_load_f32x4_streaming(batch_source + Int64(item) * record)
                loaded[4 * item] = v0
                loaded[4 * item + 1] = v1
                loaded[4 * item + 2] = v2
                loaded[4 * item + 3] = v3
            for item in cutlass.range_constexpr(tail_batch):
                for value in cutlass.range_constexpr(4):
                    acc[value] = acc[value] + loaded[4 * item + value]
        remainder = count - batches * Int32(tail_batch)
        remainder_source = source + Int64(batches * Int32(tail_batch)) * record
        for item in cutlass.range_constexpr(tail_batch - 1):
            if Int32(item) < remainder:
                v0, v1, v2, v3 = _rank_load_f32x4_streaming(remainder_source + Int64(item) * record)
                loaded[4 * item] = v0
                loaded[4 * item + 1] = v1
                loaded[4 * item + 2] = v2
                loaded[4 * item + 3] = v3
        for item in cutlass.range_constexpr(tail_batch - 1):
            if Int32(item) < remainder:
                for value in cutlass.range_constexpr(4):
                    acc[value] = acc[value] + loaded[4 * item + value]
        rank_store_f32x4(target, p0 + acc[0], p1 + acc[1], p2 + acc[2], p3 + acc[3])

    @cute.jit
    def _rank_sum_records_column(self, floats: cute.Pointer, first: Int64, count: Int32, column: Int32):
        """Return the ascending-order FP32 sum of one 16-byte column of ``count`` consecutive records."""
        record = Int64(self.rank_record)
        tail_batch = self.rank_tail_batch()
        source = floats + first * record + Int64(column * Int32(4))
        acc = cute.make_rmem_tensor((4,), Float32)
        acc.fill(0.0)
        loaded = cute.make_rmem_tensor((4 * tail_batch,), Float32)
        batches = count // Int32(tail_batch)
        for batch in cutlass.range(batches, unroll=1):
            batch_source = source + Int64(batch * Int32(tail_batch)) * record
            for item in cutlass.range_constexpr(tail_batch):
                v0, v1, v2, v3 = _rank_load_f32x4_streaming(batch_source + Int64(item) * record)
                loaded[4 * item] = v0
                loaded[4 * item + 1] = v1
                loaded[4 * item + 2] = v2
                loaded[4 * item + 3] = v3
            for item in cutlass.range_constexpr(tail_batch):
                for value in cutlass.range_constexpr(4):
                    acc[value] = acc[value] + loaded[4 * item + value]
        remainder = count - batches * Int32(tail_batch)
        remainder_source = source + Int64(batches * Int32(tail_batch)) * record
        for item in cutlass.range_constexpr(tail_batch - 1):
            if Int32(item) < remainder:
                v0, v1, v2, v3 = _rank_load_f32x4_streaming(remainder_source + Int64(item) * record)
                loaded[4 * item] = v0
                loaded[4 * item + 1] = v1
                loaded[4 * item + 2] = v2
                loaded[4 * item + 3] = v3
        for item in cutlass.range_constexpr(tail_batch - 1):
            if Int32(item) < remainder:
                for value in cutlass.range_constexpr(4):
                    acc[value] = acc[value] + loaded[4 * item + value]
        return acc[0], acc[1], acc[2], acc[3]

    @cute.jit
    def _rank_list_index(self, entry: Int32, short_rows: Int32, slots: Int64) -> Int64:
        """Return the row-list position of the ``entry``-th overflow row: short rows from the front, then long rows from the end."""
        index = Int64(entry)
        if entry >= short_rows:
            index = slots - Int64(1) - Int64(entry - short_rows)
        return index

    @cute.jit
    def _rank_add_long_chains(
        self,
        floats: cute.Pointer,
        record_base: Int64,
        prev_row_list: cute.Pointer,
        items: Int32,
        slots: Int64,
        bidx: Int32,
        nblocks: Int32,
        tidx: Int32,
        first_entry_row: Uint32,
        first_entry_first: Uint32,
        first_entry_count: Uint32,
    ):
        """Add long overflow chains to plane 0: one CTA per (row, 32-column group), one warp per rank segment.

        Long rows are listed from the end of the row list. Warp ``s`` sums the
        records ``[s * L, (s + 1) * L)`` of the chain (``L`` = chain length over
        the warp count) in rank order; warp 0 then adds the segment sums in
        segment order and adds the total to the plane-0 value. The CTA's first
        item entry is loaded by the caller before its grid dependency wait, and
        warp 0 issues the plane-0 load together with its segment's record loads.
        """
        segments = self.rank_threads() // 32
        groups = self.rank_tail_groups()
        smem = SmemAllocator()
        partials = smem.allocate_tensor(Float32, cute.make_layout((4, 32, segments), stride=(1, 4, 128)), byte_alignment=16)
        segment = tidx // Int32(32)
        lane = tidx % Int32(32)
        record = Int64(self.rank_record)
        item = bidx
        entry_row = first_entry_row
        entry_first = first_entry_first
        entry_count = first_entry_count
        while item < items:
            long_index = item // Int32(groups)
            group = item - long_index * Int32(groups)
            if item != bidx:
                entry_row, entry_first, entry_count, _ = _rank_load_u32x4(prev_row_list + (slots - Int64(1) - Int64(long_index)) * Int64(4))
            count = Int32(entry_count)
            per = (count + Int32(segments - 1)) // Int32(segments)
            begin = min(count, segment * per)
            end = min(count, begin + per)
            column = group * Int32(RANK_TAIL_GROUP_COLUMNS) + lane
            # The last group of a record whose width is not a multiple of the
            # group covers fewer columns; its other lanes sum nothing.
            column_ok = column < Int32(self.rank_record // 4)
            if not column_ok:
                end = begin
                column = Int32(0)
            target = floats + Int64(Int32(entry_row)) * record + Int64(column * Int32(4))
            p0 = Float32(0.0)
            p1 = Float32(0.0)
            p2 = Float32(0.0)
            p3 = Float32(0.0)
            if segment == Int32(0) and column_ok:
                p0, p1, p2, p3 = _rank_load_f32x4(target)
            s0, s1, s2, s3 = self._rank_sum_records_column(floats, record_base + Int64(Int32(entry_first)) + Int64(begin), end - begin, column)
            partials[0, lane, segment] = s0
            partials[1, lane, segment] = s1
            partials[2, lane, segment] = s2
            partials[3, lane, segment] = s3
            cute.arch.barrier()
            if segment == Int32(0) and column_ok:
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
        stride: Int32,
        lane: Int32,
        sorted_prev: Int64,
    ):
        """Add a duplicated row's per-slot records to plane 0 in ascending (query, sorted slot) order."""
        record = Int64(self.rank_record)
        chunks = (self.rank_record + 127) // 128
        acc = cute.make_rmem_tensor((4 * chunks,), Float32)
        acc.fill(0.0)
        for ordinal in cutlass.range(wave_size, unroll=1):
            query = rank_wave_query(wave_begin, ordinal, stride)
            topk = _rank_topk(mTopkIdxs, mTopkLength, query)
            for slot_base in cutlass.range(0, topk, 32, unroll=1):
                slot = slot_base + lane
                match = cutlass.Boolean(False)
                own_record = Int32(0)
                if slot < topk:
                    slot_row = Int32(-1)
                    if cutlass.const_expr(self.rank_sorts()):
                        slot_row = cute.arch.load(cute.recast_ptr(words, dtype=Int32) + sorted_prev + Int64(ordinal * Int32(self.max_topk) + slot), Int32)
                    else:
                        slot_row = mTopkIdxs[slot, (query, 0)]
                    if slot_row == row:
                        match = cutlass.Boolean(True)
                        own_record = cute.arch.load(cute.recast_ptr(words + dup_records + Int64(ordinal * Int32(self.max_topk) + slot), dtype=Int32), Int32)
                pending = Uint32(cute.arch.vote_ballot_sync(match))
                while pending != Uint32(0):
                    lowest = pending & (~pending + Uint32(1))
                    source = Int32(cute.arch.popc(lowest - Uint32(1)))
                    slot_record = cute.arch.shuffle_sync(own_record, source)
                    for chunk in cutlass.range_constexpr(chunks):
                        if lane + Int32(32 * chunk) < Int32(self.rank_record // 4):
                            column = Int64((lane + Int32(32 * chunk)) * Int32(4))
                            v0, v1, v2, v3 = _rank_load_f32x4(floats + (records // record + Int64(slot_record)) * record + column)
                            acc[4 * chunk] = acc[4 * chunk] + v0
                            acc[4 * chunk + 1] = acc[4 * chunk + 1] + v1
                            acc[4 * chunk + 2] = acc[4 * chunk + 2] + v2
                            acc[4 * chunk + 3] = acc[4 * chunk + 3] + v3
                    pending = pending ^ lowest
        for chunk in cutlass.range_constexpr(chunks):
            if lane + Int32(32 * chunk) < Int32(self.rank_record // 4):
                column = Int64((lane + Int32(32 * chunk)) * Int32(4))
                target = floats + Int64(row) * record + column
                p0, p1, p2, p3 = _rank_load_f32x4(target)
                rank_store_f32x4(target, p0 + acc[4 * chunk], p1 + acc[4 * chunk + 1], p2 + acc[4 * chunk + 2], p3 + acc[4 * chunk + 3])

    def rank_order_probed(self) -> bool:
        """Whether this plan may take consecutive waves after the wave-order probe (one-round waves wider than top-k 128)."""
        return self.rank_rounds == 1 and self.max_topk > RANK_NARROW_ROUNDS_TOPK

    @cute.jit
    def rank_alt_order(self, words: cute.Pointer, counters: Int64) -> Int32:
        """Return 1 when the wave-order probe chose consecutive waves (the probe words stay zero without a probe)."""
        probe = cute.recast_ptr(words + counters + Int64(RANK_SETUP_COUNTER_WORDS), dtype=Int32)
        excess = cute.arch.load(probe, Int32)
        active = cute.arch.load(probe + 1, Int32)
        alt = Int32(0)
        if excess * Int32(RANK_PROBE_EXCESS_DEN) < active * Int32(RANK_PROBE_EXCESS_NUM):
            alt = Int32(1)
        return alt

    @cute.kernel
    def rank_probe(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        probe_begin: Int32,
        probe_size: Int32,
        planes: Int32,
    ):
        """Count one consecutive probe wave's writers beyond the dense planes, and its active slots.

        Every valid active slot adds one to its row's counter; a slot whose row
        already had ``planes`` writers lands beyond the dense planes. Both
        totals are integer sums, so the wave order chosen from them does not
        depend on scheduling.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        records, _, _, _, _, _, counters, _, _ = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        row_counts = cute.recast_ptr(words + records, dtype=Int32)
        probe = cute.recast_ptr(words + counters + Int64(RANK_SETUP_COUNTER_WORDS), dtype=Int32)
        topk_width = Int32(self.max_topk)
        span = probe_size * topk_width
        lane = tidx % Int32(32)
        warp_base = bidx * Int32(RANK_SETUP_THREADS) + (tidx - lane)
        nthreads = nblocks * Int32(RANK_SETUP_THREADS)
        excess = Int32(0)
        active = Int32(0)
        for base in cutlass.range(warp_base, span, nthreads, unroll=1):
            slot_index = base + lane
            beyond = cutlass.Boolean(False)
            counted = cutlass.Boolean(False)
            if slot_index < span:
                ordinal = slot_index // topk_width
                slot = slot_index - ordinal * topk_width
                query = probe_begin + ordinal
                if slot < _rank_topk(mTopkIdxs, mTopkLength, query):
                    row = mTopkIdxs[slot, (query, 0)]
                    if row >= Int32(0) and row < seqlen_kv:
                        counted = cutlass.Boolean(True)
                        previous = cute.arch.atomic_add((row_counts + row).llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                        if previous >= planes:
                            beyond = cutlass.Boolean(True)
            excess = excess + Int32(cute.arch.popc(Uint32(cute.arch.vote_ballot_sync(beyond))))
            active = active + Int32(cute.arch.popc(Uint32(cute.arch.vote_ballot_sync(counted))))
        if lane == Int32(0):
            if excess > Int32(0):
                cute.arch.atomic_add(probe.llvm_ptr, excess, sem="relaxed", scope="gpu")
            if active > Int32(0):
                cute.arch.atomic_add((probe + 1).llvm_ptr, active, sem="relaxed", scope="gpu")

    @cute.jit
    def launch_rank_probe(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        probe_begin: Int32,
        probe_size: Int32,
        planes: Int32,
        stream: cuda.CUstream,
    ):
        """Probe one consecutive wave once per execution; an ordinary launch, so every later launch starts after it completed."""
        self.rank_probe(mRows, mTopkIdxs, mTopkLength, seqlen_kv, probe_begin, probe_size, planes).launch(
            grid=[RANK_PROBE_CTAS, 1, 1], block=[RANK_SETUP_THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def rank_init(self, mRows: cute.Tensor, seqlen_kv: Int32, planes: Int32):
        """Clear dense plane 0, the high-water marks, every row-state parity and the counters.

        Planes ``k >= 1`` are not cleared: their first writer stores its record.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        records, _, _, _, _, _, counters, _, marks = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        stride = Int64(nblocks) * Int64(RANK_INIT_THREADS)
        chunk = Int64(bidx) * Int64(RANK_INIT_THREADS) + Int64(tidx)
        dense_chunks = pitch * Int64(self.rank_record // 4)
        while chunk < dense_chunks:
            _rank_store_zero_x4(words + chunk * Int64(4))
            chunk += stride
        # Marks and the row-state parities are contiguous.
        chunk = Int64(bidx) * Int64(RANK_INIT_THREADS) + Int64(tidx)
        state_chunks = Int64(RANK_HWM_WORDS + RANK_STATE_PARITIES * rank_state_words(self.max_query_pairs_per_wave)) * pitch // Int64(4)
        while chunk < state_chunks:
            _rank_store_zero_x4(words + marks + chunk * Int64(4))
            chunk += stride
        if bidx == Int32(0) and tidx < Int32(RANK_COUNTER_REGION_WORDS // 4):
            _rank_store_zero_x4(words + counters + Int64(tidx) * Int64(4))
        if cutlass.const_expr(self.rank_order_probed()):
            # The wave-order probe counts writers per row in the first ``pitch``
            # words of the overflow-record area, which no wave has written yet.
            chunk = Int64(bidx) * Int64(RANK_INIT_THREADS) + Int64(tidx)
            while chunk < pitch // Int64(4):
                _rank_store_zero_x4(words + records + chunk * Int64(4))
                chunk += stride

    @cute.kernel
    def rank_setup(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        next_begin: Int32,
        next_size: Int32,
        step: Int32,
        stride: Int32,
        planes: Int32,
        early: Int32,
        prev_begin_alt: Int32,
        cur_begin_alt: Int32,
        next_begin_alt: Int32,
        stride_alt: Int32,
    ):
        """Finish the previous wave, write the current wave's codes and record the next wave's ordinals.

        ``step`` is the current wave index plus ``RANK_STEP_BIAS``. The previous
        wave's row state is cleared; the current wave's row state yields one
        destination code per slot and its overflow records are allocated; the
        next wave's slots set one ordinal bit each; the counters two waves ahead
        are reset; sorting plans also sort the next wave's rows. These phases
        touch disjoint parities and only depend on the setup launch before the
        preceding main kernel, which that kernel waited for before it released
        this launch. After the grid dependency wait the previous wave's overflow
        and duplicate records are added to plane 0. ``early == 0`` (the leading
        setup and wave 0's setup) waits before every phase and before it
        triggers its dependents: main kernel 0 reads wave 0's sorted rows without
        a wait (see the module docstring). Plans with a wave-order probe take the
        ``*_alt`` (consecutive) wave begins and stride when the probe chose
        consecutive waves.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblocks, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self.rank_pdl):
            if early == Int32(0):
                cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        pitch = Int64(cute.round_up(seqlen_kv, 8))
        records, state, offsets, dup_records, row_list, dup_list, counters, codes, marks = self.rank_offsets(pitch, planes)
        words = cute.recast_ptr(mRows.iterator, dtype=Uint32)
        floats = mRows.iterator
        if cutlass.const_expr(self.rank_order_probed()):
            if self.rank_alt_order(words, counters) != Int32(0):
                prev_begin = prev_begin_alt
                cur_begin = cur_begin_alt
                next_begin = next_begin_alt
                stride = stride_alt
        topk_width = Int32(self.max_topk)
        slots = Int64(self.max_query_pairs_per_wave * self.max_topk)
        state_words = rank_state_words(self.max_query_pairs_per_wave)
        ordinal_words = rank_ordinal_words(self.max_query_pairs_per_wave)
        assert ordinal_words <= RANK_MAX_ORDINAL_WORDS
        state_pitch = pitch * Int64(state_words)
        state_prev = state + Int64((step + Int32(RANK_STATE_PARITIES - 1)) % Int32(RANK_STATE_PARITIES)) * state_pitch
        state_cur = state + Int64(step % Int32(RANK_STATE_PARITIES)) * state_pitch
        state_next = state + Int64((step + Int32(1)) % Int32(RANK_STATE_PARITIES)) * state_pitch
        list_prev = (step + Int32(RANK_LIST_PARITIES - 1)) % Int32(RANK_LIST_PARITIES)
        list_cur = step % Int32(RANK_LIST_PARITIES)
        list_next = (step + Int32(1)) % Int32(RANK_LIST_PARITIES)
        list_reset = (step + Int32(2)) % Int32(RANK_LIST_PARITIES)
        half_prev = Int64((step + Int32(1)) % Int32(2))
        half_cur = Int64(step % Int32(2))
        setup_threads = self.rank_threads()
        thread = bidx * Int32(setup_threads) + tidx
        nthreads = nblocks * Int32(setup_threads)
        work_threads = min(nblocks, Int32(self.rank_work_ctas())) * Int32(setup_threads)
        # Slot phases skip the leading sort CTAs of sorting plans.
        sort_ctas = Int32(self.rank_sort_ctas())
        slot_thread = thread - sort_ctas * Int32(setup_threads)
        slot_threads = min(nblocks - sort_ctas, Int32(self.rank_work_ctas())) * Int32(setup_threads)
        warp = thread // Int32(32)
        nwarps = work_threads // Int32(32)
        lane = tidx % Int32(32)
        cur_counters = counters + Int64(list_cur * Int32(RANK_COUNTER_STRIDE))
        rows_counter = cute.recast_ptr(words + cur_counters, dtype=Int32)
        records_counter = cute.recast_ptr(words + cur_counters + Int64(1), dtype=Int32)
        long_counter = cute.recast_ptr(words + cur_counters + Int64(3), dtype=Int32)
        cur_codes = codes + half_cur * slots
        cur_dup_records = dup_records + half_cur * slots
        cur_row_list = row_list + half_cur * Int64(4) * slots
        cur_offsets = offsets + half_cur * pitch
        sorted_rows = cute.recast_ptr(words, dtype=Int32)
        sorted_cur = Int64(0)
        if cutlass.const_expr(self.rank_sorts()):
            sorted_cur = self.rank_sorted_base(pitch, planes, step)
            # Next wave: the leading CTAs sort one query each for the next setup launch and main kernel.
            if bidx < sort_ctas:
                if bidx < next_size:
                    sort_smem = SmemAllocator()
                    sort_counts = sort_smem.allocate_tensor(cutlass.Int16, cute.make_layout(((RANK_SORT_BUCKETS + 1) * RANK_SETUP_THREADS,)), byte_alignment=16)
                    sort_sums = sort_smem.allocate_tensor(Int32, cute.make_layout((RANK_SETUP_THREADS + 4,)), byte_alignment=16)
                    sorted_next = self.rank_sorted_base(pitch, planes, step + Int32(1))
                    rotate_step = Int32(
                        RANK_SORT_ROTATE_TILES * RANK_SORT_BUCKETS * 64 // self.max_topk if self.rank_sort_rotates(mTopkLength is not None) else 0
                    )
                    self._rank_sort_query(
                        sort_counts,
                        sort_sums,
                        mTopkIdxs,
                        mTopkLength,
                        rank_wave_query(next_begin, bidx, stride),
                        seqlen_kv,
                        sorted_rows + sorted_next + Int64(bidx) * Int64(self.max_topk),
                        tidx,
                        ((step + Int32(1 - RANK_STEP_BIAS)) * rotate_step) % Int32(RANK_SORT_BUCKETS),
                    )
        next_dup_counter = cute.recast_ptr(words + counters + Int64(list_next * Int32(RANK_COUNTER_STRIDE) + Int32(2)), dtype=Int32)
        next_dup_list = dup_list + Int64(list_next) * slots
        # The three phases touch disjoint parities. One pass over the slot
        # indices issues every phase's independent loads before the dependent
        # state reads and updates, so their memory latencies overlap.
        span = max(prev_size, max(cur_size, next_size)) * topk_width
        slot_index = slot_thread
        if slot_thread < Int32(0) or slot_thread >= slot_threads:
            slot_index = span
        while slot_index < span:
            ordinal = slot_index // topk_width
            slot = slot_index - ordinal * topk_width
            # Extents are raw lengths: slot < min(max(length, 0), K) iff slot < length.
            prev_extent = topk_width
            prev_row = Int32(-1)
            if ordinal < prev_size:
                if cutlass.const_expr(mTopkLength is not None):
                    prev_extent = Int32(mTopkLength[rank_wave_query(prev_begin, ordinal, stride)])
                prev_row = mTopkIdxs[slot, (rank_wave_query(prev_begin, ordinal, stride), 0)]
            cur_extent = topk_width
            cur_row = Int32(-1)
            if ordinal < cur_size:
                if cutlass.const_expr(mTopkLength is not None):
                    cur_extent = Int32(mTopkLength[rank_wave_query(cur_begin, ordinal, stride)])
                if cutlass.const_expr(self.rank_sorts()):
                    cur_row = cute.arch.load(sorted_rows + sorted_cur + Int64(slot_index), Int32)
                else:
                    cur_row = mTopkIdxs[slot, (rank_wave_query(cur_begin, ordinal, stride), 0)]
            next_extent = topk_width
            next_row = Int32(-1)
            if ordinal < next_size:
                if cutlass.const_expr(mTopkLength is not None):
                    next_extent = Int32(mTopkLength[rank_wave_query(next_begin, ordinal, stride)])
                next_row = mTopkIdxs[slot, (rank_wave_query(next_begin, ordinal, stride), 0)]
            # Previous wave: clear its row state.
            if slot < prev_extent:
                if prev_row >= Int32(0) and prev_row < seqlen_kv:
                    for quad in cutlass.range_constexpr(state_words // 4):
                        _rank_store_zero_x4(words + state_prev + Int64(prev_row) * Int64(state_words) + Int64(4 * quad))
            # Current wave: read its row state and plane marks.
            cur_valid = Int32(0)
            state_0 = Uint32(0)
            state_1 = Uint32(0)
            state_2 = Uint32(0)
            state_3 = Uint32(0)
            state_4 = Uint32(0)
            state_5 = Uint32(0)
            state_6 = Uint32(0)
            duplicate = Uint32(0)
            mark_low = Int32(0)
            mark_stamp = Int32(0)
            if slot < cur_extent:
                if cur_row >= Int32(0) and cur_row < seqlen_kv:
                    cur_valid = Int32(1)
                    state_entry = words + state_cur + Int64(cur_row) * Int64(state_words)
                    if cutlass.const_expr(state_words == 4):
                        state_0, state_1, state_2, duplicate = _rank_load_u32x4(state_entry)
                    else:
                        state_0, state_1, state_2, state_3 = _rank_load_u32x4(state_entry)
                        state_4, state_5, state_6, duplicate = _rank_load_u32x4(state_entry + Int64(4))
                    mark_low, mark_stamp = _rank_load_mark(words + marks + Int64(cur_row) * Int64(RANK_HWM_WORDS))
            # Next wave: record its ordinal bit.
            next_valid = Int32(0)
            previous = Uint32(0)
            bit = Uint32(1) << Uint32(ordinal % Int32(32))
            if slot < next_extent:
                if next_row >= Int32(0) and next_row < seqlen_kv:
                    next_valid = Int32(1)
                    bit_entry = words + state_next + Int64(next_row) * Int64(state_words) + Int64(ordinal // Int32(32))
                    previous = cute.arch.atomic_or(bit_entry.llvm_ptr, bit, sem="relaxed", scope="gpu")
            if cur_valid != Int32(0):
                row = cur_row
                code_ptr = cute.recast_ptr(words + cur_codes + Int64(slot_index), dtype=Int32)
                if duplicate != Uint32(0):
                    record = cute.arch.atomic_add(records_counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                    cute.arch.store(cute.recast_ptr(words + cur_dup_records + Int64(slot_index), dtype=Int32), record)
                    cute.arch.store(code_ptr, planes * Int32(pitch) + record)
                else:
                    count = Int32(cute.arch.popc(state_0) + cute.arch.popc(state_1) + cute.arch.popc(state_2))
                    conflict = Int32(
                        cute.arch.popc(state_0 & _rank_lower_mask(ordinal, 0))
                        + cute.arch.popc(state_1 & _rank_lower_mask(ordinal, 1))
                        + cute.arch.popc(state_2 & _rank_lower_mask(ordinal, 2))
                    )
                    if cutlass.const_expr(ordinal_words > 3):
                        count = count + Int32(cute.arch.popc(state_3) + cute.arch.popc(state_4))
                        conflict = conflict + Int32(
                            cute.arch.popc(state_3 & _rank_lower_mask(ordinal, 3)) + cute.arch.popc(state_4 & _rank_lower_mask(ordinal, 4))
                        )
                    if cutlass.const_expr(ordinal_words > 5):
                        count = count + Int32(cute.arch.popc(state_5) + cute.arch.popc(state_6))
                        conflict = conflict + Int32(
                            cute.arch.popc(state_5 & _rank_lower_mask(ordinal, 5)) + cute.arch.popc(state_6 & _rank_lower_mask(ordinal, 6))
                        )
                    # Planes written before this wave: the stamped entry keeps the
                    # pre-wave mark, so all writers of this row agree on it.
                    written = mark_low & Int32(0xFFFF)
                    if mark_stamp == step:
                        written = (mark_low >> Int32(16)) & Int32(0xFFFF)
                    if conflict < planes:
                        if conflict >= max(written, Int32(1)):
                            cute.arch.store(code_ptr, Int32(RANK_FIRST_WRITE) + conflict * Int32(pitch) + row)
                        else:
                            cute.arch.store(code_ptr, conflict * Int32(pitch) + row)
                    else:
                        cute.arch.store(code_ptr, Int32(-2) - (conflict - planes))
                    if conflict == Int32(0):
                        raised = max(written, min(count, planes))
                        _rank_store_mark(words + marks + Int64(row) * Int64(RANK_HWM_WORDS), (written << Int32(16)) | raised, step)
                    if count > planes and conflict == planes:
                        first = cute.arch.atomic_add(records_counter.llvm_ptr, count - planes, sem="relaxed", scope="gpu")
                        list_index = Int64(0)
                        if count - planes > Int32(RANK_TAIL_LONG):
                            position = cute.arch.atomic_add(long_counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                            list_index = slots - Int64(1) - Int64(position)
                        else:
                            position = cute.arch.atomic_add(rows_counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                            list_index = Int64(position)
                        list_entry = cute.recast_ptr(words + cur_row_list + list_index * Int64(4), dtype=Int32)
                        cute.arch.store(list_entry, row)
                        cute.arch.store(list_entry + 1, first)
                        cute.arch.store(list_entry + 2, count - planes)
                        cute.arch.store(cute.recast_ptr(words + cur_offsets + Int64(row), dtype=Int32), first)
            if next_valid != Int32(0):
                if (previous & bit) != Uint32(0):
                    flag_entry = words + state_next + Int64(next_row) * Int64(state_words) + Int64(state_words - 1)
                    flagged = cute.arch.atomic_or(flag_entry.llvm_ptr, Uint32(1), sem="relaxed", scope="gpu")
                    if flagged == Uint32(0):
                        dup_position = cute.arch.atomic_add(next_dup_counter.llvm_ptr, Int32(1), sem="relaxed", scope="gpu")
                        cute.arch.store(cute.recast_ptr(words + next_dup_list + Int64(dup_position), dtype=Int32), next_row)
            slot_index += slot_threads
        if thread == Int32(0):
            _rank_store_zero_x4(words + counters + Int64(list_reset * Int32(RANK_COUNTER_STRIDE)))
        # The previous wave's overflow and duplicate row counts were final
        # before its main kernel started. CTAs that the tail assigns no work
        # leave without waiting; CTA 0 always waits, so this launch still
        # completes after the previous main kernel.
        prev_counters = counters + Int64(list_prev * Int32(RANK_COUNTER_STRIDE))
        overflow_rows = Int32(0)
        long_rows = Int32(0)
        duplicate_rows = Int32(0)
        if prev_size > Int32(0):
            overflow_rows = cute.arch.load(cute.recast_ptr(words + prev_counters, dtype=Int32), Int32)
            duplicate_rows = cute.arch.load(cute.recast_ptr(words + prev_counters + Int64(2), dtype=Int32), Int32)
            long_rows = cute.arch.load(cute.recast_ptr(words + prev_counters + Int64(3), dtype=Int32), Int32)
        columns = Int32(self.rank_record // 4)
        # Long chains take whole CTAs only when their items fit in one pass of
        # the grid; otherwise they join the per-thread units.
        long_items = long_rows * Int32(self.rank_tail_groups())
        thread_rows = overflow_rows
        if long_items > nblocks:
            long_items = Int32(0)
            thread_rows = overflow_rows + long_rows
        has_work = Int32(0)
        if bidx == Int32(0):
            has_work = Int32(1)
        if bidx < long_items:
            has_work = Int32(1)
        if bidx * Int32(setup_threads) < thread_rows * columns:
            has_work = Int32(1)
        if bidx < Int32(self.rank_work_ctas()):
            if bidx * Int32(setup_threads // 32) < duplicate_rows:
                has_work = Int32(1)
        if has_work != Int32(0):
            self._rank_finish_previous(
                floats,
                words,
                mTopkIdxs,
                mTopkLength,
                records,
                dup_records,
                row_list,
                dup_list,
                prev_counters,
                overflow_rows,
                long_rows,
                long_items,
                duplicate_rows,
                prev_begin,
                prev_size,
                stride,
                half_prev,
                list_prev,
                slots,
                bidx,
                nblocks,
                tidx,
                thread,
                nthreads,
                warp,
                nwarps,
                lane,
                self.rank_sorted_base(pitch, planes, step - Int32(1)),
            )

    @cute.jit
    def _rank_finish_previous(
        self,
        floats: cute.Pointer,
        words: cute.Pointer,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        records: Int64,
        dup_records: Int64,
        row_list: Int64,
        dup_list: Int64,
        prev_counters: Int64,
        overflow_rows: Int32,
        long_rows: Int32,
        long_items: Int32,
        duplicate_rows: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        stride: Int32,
        half_prev: Int64,
        list_prev: Int32,
        slots: Int64,
        bidx: Int32,
        nblocks: Int32,
        tidx: Int32,
        thread: Int32,
        nthreads: Int32,
        warp: Int32,
        nwarps: Int32,
        lane: Int32,
        sorted_prev: Int64,
    ):
        """Wait for the previous main kernel, then add its overflow and duplicate records to plane 0.

        The row-list entries were written by the previous wave's setup launch,
        which completed before the previous main kernel released this launch:
        the first entries are loaded before the grid dependency wait.
        """
        prev_row_list = words + row_list + half_prev * Int64(4) * slots
        groups = self.rank_tail_groups()
        columns = Int32(self.rank_record // 4)
        thread_rows = overflow_rows
        if long_items == Int32(0):
            thread_rows = overflow_rows + long_rows
        units = thread_rows * columns
        long_row = Uint32(0)
        long_first = Uint32(0)
        long_count = Uint32(0)
        entry_row = Uint32(0)
        entry_first = Uint32(0)
        entry_count = Uint32(0)
        if prev_size > Int32(0):
            if bidx < long_items:
                long_row, long_first, long_count, _ = _rank_load_u32x4(prev_row_list + (slots - Int64(1) - Int64(bidx // Int32(groups))) * Int64(4))
            if thread < units:
                entry_row, entry_first, entry_count, _ = _rank_load_u32x4(
                    prev_row_list + self._rank_list_index(thread // columns, overflow_rows, slots) * Int64(4)
                )
        if cutlass.const_expr(self.rank_pdl):
            cute.arch.griddepcontrol_wait()
        if prev_size > Int32(0):
            # ``long_items`` is zero when the long chains go to the per-thread units.
            self._rank_add_long_chains(
                floats, records // Int64(self.rank_record), prev_row_list, long_items, slots, bidx, nblocks, tidx, long_row, long_first, long_count
            )
            # One thread per (overflow row, 16-byte column): consecutive threads
            # read consecutive columns of the same record. Short rows come from
            # the front of the row list, then (without whole-CTA items) long
            # rows from its end. Each unit's (row, first, count) entry is loaded
            # one unit ahead.
            unit = thread
            while unit < units:
                row = Int32(entry_row)
                first = Int32(entry_first)
                count = Int32(entry_count)
                column = unit - (unit // columns) * columns
                unit += nthreads
                if unit < units:
                    entry_row, entry_first, entry_count, _ = _rank_load_u32x4(
                        prev_row_list + self._rank_list_index(unit // columns, overflow_rows, slots) * Int64(4)
                    )
                self._rank_add_records_column(floats, Int64(row), records // Int64(self.rank_record) + Int64(first), count, column)
            # Duplicate rows go to the slot-phase CTAs' warps only.
            item = warp
            if warp >= nwarps:
                item = duplicate_rows
            while item < duplicate_rows:
                row = cute.arch.load(cute.recast_ptr(words + dup_list + Int64(list_prev) * slots + Int64(item), dtype=Int32), Int32)
                self._rank_add_duplicate_row(
                    floats,
                    words,
                    mTopkIdxs,
                    mTopkLength,
                    records,
                    dup_records + half_prev * slots,
                    row,
                    prev_begin,
                    prev_size,
                    stride,
                    lane,
                    sorted_prev,
                )
                item += nwarps

    @cute.jit
    def launch_rank_init(self, mRows: cute.Tensor, seqlen_kv: Int32, planes: Int32, stream: cuda.CUstream):
        """Clear the conflict-rank workspace once per execution."""
        self.rank_init(mRows, seqlen_kv, planes).launch(grid=[RANK_INIT_CTAS, 1, 1], block=[RANK_INIT_THREADS, 1, 1], stream=stream)

    @cute.jit
    def launch_rank_setup(
        self,
        mRows: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        seqlen_kv: Int32,
        prev_begin: Int32,
        prev_size: Int32,
        cur_begin: Int32,
        cur_size: Int32,
        next_begin: Int32,
        next_size: Int32,
        step: Int32,
        stride: Int32,
        planes: Int32,
        early: Int32,
        prev_begin_alt: Int32,
        cur_begin_alt: Int32,
        next_begin_alt: Int32,
        stride_alt: Int32,
        stream: cuda.CUstream,
    ):
        """Launch the single per-wave setup (finish previous, codes for current, ordinals for next)."""
        if cutlass.const_expr(self.rank_thin()):
            # One thin CTA per SM pair: each fits beside a resident main CTA, and the
            # cluster partner SM runs its main CTA without a co-resident setup CTA.
            self.rank_setup(
                mRows,
                mTopkIdxs,
                mTopkLength,
                seqlen_kv,
                prev_begin,
                prev_size,
                cur_begin,
                cur_size,
                next_begin,
                next_size,
                step,
                stride,
                planes,
                early,
                prev_begin_alt,
                cur_begin_alt,
                next_begin_alt,
                stride_alt,
            ).launch(
                grid=[self.COMPACT_SLOTS, 1, 1],
                block=[RANK_THIN_THREADS, 1, 1],
                smem=self.rank_setup_smem_bytes(),
                stream=stream,
                use_pdl=self.rank_pdl,
                min_blocks_per_mp=RANK_THIN_MIN_BLOCKS,
                preferred_smem_carveout=RANK_THIN_CARVEOUT,
            )
        else:
            # Every other accepted top-k width (512 and wider) launches the wide setup.
            assert self.max_topk >= RANK_SETUP_WIDE_TOPK
            # Every CTA of this launch must start before the next main kernel can
            # launch. Below the widest top-k, plans whose dense planes cover at
            # least half a wave bound each row's overflow chain by half a wave,
            # so they skip the tail-only CTAs and size the slot-phase CTAs at
            # about two wave slots per thread.
            wide_ctas = Int32(RANK_SETUP_WIDE_CTAS)
            if cutlass.const_expr(self.max_topk < RANK_SETUP_TAIL_TOPK):
                if planes * Int32(2) >= Int32(self.max_query_pairs_per_wave):
                    wide_ctas = Int32(self.rank_setup_slot_ctas())
            if cutlass.const_expr(self.rank_sorts() and self.max_topk < RANK_SETUP_TAIL_TOPK):
                if planes * Int32(2) >= Int32(self.max_query_pairs_per_wave):
                    wide_ctas = Int32(self.rank_sort_ctas() + self.rank_setup_slot_ctas())
            self.rank_setup(
                mRows,
                mTopkIdxs,
                mTopkLength,
                seqlen_kv,
                prev_begin,
                prev_size,
                cur_begin,
                cur_size,
                next_begin,
                next_size,
                step,
                stride,
                planes,
                early,
                prev_begin_alt,
                cur_begin_alt,
                next_begin_alt,
                stride_alt,
            ).launch(
                grid=[wide_ctas, 1, 1],
                block=[RANK_SETUP_THREADS, 1, 1],
                smem=self.rank_setup_smem_bytes(),
                stream=stream,
                use_pdl=self.rank_pdl,
                min_blocks_per_mp=RANK_SETUP_WIDE_MIN_BLOCKS,
            )

    def rank_setup_slot_ctas(self) -> int:
        """Return slot-phase CTAs for about two wave slots per thread, within [RANK_SETUP_NARROW_MIN_CTAS, RANK_SETUP_WORK_CTAS]."""
        slots = self.max_query_pairs_per_wave * self.max_topk
        return max(RANK_SETUP_NARROW_MIN_CTAS, min(RANK_SETUP_WORK_CTAS, slots // (2 * RANK_SETUP_THREADS)))


@dsl_user_op
def rank_store_f32x2(destination: cute.Pointer, value_0: Float32, value_1: Float32, *, loc=None, ip=None) -> None:
    """Store two aligned FP32 values with one 64-bit store."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Float32(value_0).ir_value(loc=loc, ip=ip), Float32(value_1).ir_value(loc=loc, ip=ip)],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def rank_store_f32(destination: cute.Pointer, value: Float32, *, loc=None, ip=None) -> None:
    """Store one FP32 value."""
    destination_i64 = destination.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [destination_i64, Float32(value).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
