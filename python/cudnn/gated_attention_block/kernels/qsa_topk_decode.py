# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The decode top-k of the block-sparse indexer: one thread-block CLUSTER per query row, an exact radix select over the
row's fp32 block logits, the selected block ids written in CANONICAL order (ids strictly descending, ``-1`` padding last).

Contract (the decode scorer's output is this kernel's input; both modules pin it):

* ``logits [T, n_blocks_max]`` fp32, any row stride, unit column stride: row ``t`` holds one score per complete 4-token block in
  columns ``[0, counts[t])``; the columns at or past ``counts[t]`` are NEVER read (they are whatever the scorer left there);
* ``counts [T]`` int32: ``counts[t]`` is the VISIBLE count ``n_vis`` of row ``t`` (clamped in-kernel to ``[0, n_blocks_max]``) --
  NEVER the list length ``min(top_k, n_vis)``: the kernel cannot tell the two spellings apart, and the list-length spelling puts
  every row with ``n_vis > top_k`` in the identity region, so the kernel returns ``top_k - 1 .. 0`` (the OLDEST ``top_k`` blocks,
  not the top-k) bitwise reproducibly and with no error.  The list length is DERIVED from the output: ``min(top_k, counts[t])``
  non-``-1`` slots per row (the test module pins the trap);
* ``block_ids [T, top_k]`` int32 out: the ``n_sel = min(top_k, counts[t])`` selected block ids, STRICTLY DESCENDING in
  ``block_ids[t, :n_sel]``, then ``-1`` in ``block_ids[t, n_sel:]``.  The selected SET is the exact top-``n_sel`` of the fp32
  logits (every selected logit >= every unselected one); a tie at the k-th value keeps the SMALLER block ids; two launches are
  bitwise identical (no atomic decides a slot); a row with ``counts <= top_k`` selects every visible block (the identity
  region: ``counts - 1 .. 0``).  ``NaN`` logits have an unspecified rank (the scorer never writes one; the ordering key below
  sorts them with the infinities of their sign); ``-0.0`` ranks below ``+0.0``.
* ``top_k`` in ``[1, 2048]`` (the block declares a multiple of 4 in ``[4, 512]``; the kernel is generic), a compile-time constant
  read off ``block_ids.shape[1]``.

Why a new kernel: the radix top-k the standalone decode selector binds today runs ONE CTA per row and was measured 1.3-1.6x
behind the serving stack's own cluster top-k at 32K / 128K tokens; this kernel is its twin -- the row split over the CTAs of a
cluster, the per-CTA radix histograms merged through distributed shared memory (DSMEM) -- with the canonical order produced
in-kernel (the in-block indexer sorts its list with ``torch.sort`` today: four launches plus one per window and a 4 KB index
template per call).  No performance number is claimed here: the cluster size and the thread count are KNOBS (every value is the
same function, pinned bitwise by the tests), measured on an exclusive node before a default is quoted.

Algorithm (``C`` CTAs per row, ``THREADS`` threads each; every CTA owns the contiguous slice ``[rank * ceil(n / C), ...)`` of the
row, ``n = clamp(counts[t], 0, n_blocks_max)``):

1. ``n <= top_k`` -- the identity region: CTA 0 writes ``n - 1, ..., 0, -1, ...``; no shared memory, no barrier, the other CTAs
   of the cluster leave (the branch is uniform over the cluster: every CTA reads the same ``counts[t]``).
2. Otherwise every lane converts its slice elements to the order-preserving key ``u = bits ^ (bits >> 31 ? 0xFFFFFFFF :
   0x80000000)`` (a larger float is a larger unsigned key) and stages them in shared memory ONCE (``s_keys``); every later
   sweep re-reads the slice from there, never from global memory.
3. Radix select, 8 bits per pass, at most 4 passes (the DSA kernel's ``first_refine_shift = 24`` / ``num_refine_rounds = 4``
   geometry): pass ``p`` histograms the current candidates (the keys whose resolved top bits equal the prefix found so far)
   into 256 bins of their next byte with WARP-AGGREGATED shared atomics (``match.any.sync`` groups the lanes of one bin, the
   lowest lane adds the group's ``popc`` -- the top byte of a score row is spread over a handful of bins, so an unaggregated
   histogram would serialize 32-way); the cluster barrier publishes every CTA's histogram; lane ``b`` of every CTA then reads
   bin ``b`` of all ``C`` histograms through DSMEM (``ld.shared::cluster``), so EVERY CTA holds the same merged histogram and
   derives the same verdict (no leader, no second exchange): a descending block scan finds the bin that holds the k-th
   candidate, ``above`` (keys resolved strictly greater) and ``eq`` (keys in the bin) update, and the select stops when the bin
   is taken whole (``eq == k - above``) or the key is fully resolved (pass 4).  Each pass has its own histogram buffer, so no
   CTA overwrites a histogram a peer may still be reading (no write-after-read barrier is needed).
4. The exact verdict is a prefix ``P`` of ``8 x passes`` bits: GT keys (prefix above ``P``) are selected, EQ keys (prefix equal
   to ``P``) are selected in ascending block-id order up to ``need = k - above``, the rest are not.  Every CTA builds two
   32-bit-per-word SELECTION BITMAPS of its slice with one ``vote.ballot`` per warp per round (the 32 lanes of a warp hold 32
   CONSECUTIVE ids, so a ballot IS a bitmap word), scans the words' ``popc`` for per-word exclusive prefixes, publishes its
   (GT, EQ) totals into every peer's ``s_counts`` slot through DSMEM (``st.shared::cluster``), and after ONE more cluster
   barrier knows how many selected ids and how many EQ ids the lower slices hold.  A selected element at ascending rank
   ``pos`` among all selected ids of the row is written to ``block_ids[t, k - 1 - pos]`` -- the descending canonical order
   falls out of the ascending compaction, and a row with ``n > k`` fills exactly ``k`` slots.

Barrier table (REQUIRED pre-development table; no mbarrier is initialised by this kernel -- the DSMEM traffic is plain
``ld`` / ``st.shared::cluster`` ordered by release / acquire cluster barriers, and every CTA-local hand-off is ``bar.sync 0``):

| site | instruction | who arrives (SUM(issuing lanes) == expected) | what it orders |
|---|---|---|---|
| Z | ``bar.sync 0`` | every thread of the CTA, once: ``THREADS == THREADS`` | the zeroing of the four histogram buffers and the staging of ``s_keys`` before the first pass |
| CB_p (one per executed pass, p <= 4) | ``barrier.cluster.arrive.release.aligned`` + ``barrier.cluster.wait.acquire`` (``C > 1``); ``bar.sync 0`` at ``C == 1`` | every thread of every CTA of the cluster, once per pass: ``C x THREADS`` arrivals; the pass count is identical in every CTA because each CTA derives it from the SAME merged histogram | every CTA's pass-``p`` shared atomics before any peer's ``ld.shared::cluster`` of that buffer |
| B_p2 | ``bar.sync 0`` | ``THREADS`` | the warp sums of the 256-bin scan (``s_wsum``) before the cross-warp fix-up |
| B_p3 | ``bar.sync 0`` | ``THREADS`` | the found bin's record (``s_found``) before every thread reads it; also the reuse of ``s_wsum`` by the next pass |
| F1 | ``bar.sync 0`` | ``THREADS`` | the bitmap words (``s_bits_gt`` / ``s_bits_eq``) before the word scan |
| F2 | ``bar.sync 0`` | ``THREADS`` | the word scan's warp sums (``s_wsum``) before the cross-warp fix-up |
| CB_x | cluster arrive / wait as CB_p (``C > 1``); ``bar.sync 0`` at ``C == 1`` | ``C x THREADS``, once | every CTA's ``st.shared::cluster`` of its totals into the peers' ``s_counts`` before any peer reads them; also the per-word prefixes (``s_pfx_*``) before the write sweep |

Every cross-CTA access (the histogram reads of pass ``p``, the totals store before CB_x) sits BEFORE a cluster barrier every
CTA must still reach, so no CTA can exit while a peer reads or writes its shared memory (the P6 / P15 drain argument); after
CB_x no remote access remains and the CTAs exit independently.  The identity path performs no barrier of any kind.

SMEM table (REQUIRED pre-development table; dynamic shared memory through ``SmemAllocator``, ``SLICE = cols_cap / C``,
``WORDS = SLICE / 32``; no descriptor reads any buffer, so no swizzle matches a descriptor -- each row states the per-lane
stride that keeps the banks spread):

| name / dtype / elems | written by | read by | per-lane stride, bank argument |
|---|---|---|---|
| ``s_keys`` u32 x ``SLICE`` | per-lane ``st.shared`` in the round-strided map ``e = i x THREADS + tidx`` | per-lane ``ld.shared`` in the same map (the histogram passes, the flag sweep) | 4 B between lanes (consecutive words): conflict-free, no swizzle needed |
| ``s_hist`` i32 x ``4 x 256`` (one 256-bin buffer per pass) | warp-aggregated ``atom.shared.cta.add`` at the bin of each candidate (data-dependent address; the aggregation leaves one lane per distinct bin per instruction) | ``ld.shared::cluster`` by lane ``b`` of every CTA at bin ``255 - b`` of every peer's buffer | 4 B between lanes on the read side: conflict-free; the write side is a histogram (inherently bin-addressed) |
| ``s_wsum`` i32 x 32 | lane 31 of each warp at slot ``warp`` | every thread, slots ``< warp`` (broadcast) | one word per warp: trivial |
| ``s_found`` i32 x 4 | the one thread whose bin holds the k-th candidate | every thread (broadcast) | trivial |
| ``s_bits_gt`` / ``s_bits_eq`` u32 x ``WORDS`` | lane 0 of each warp, word ``i x (THREADS / 32) + warp`` per round | the word scan (thread ``t`` reads words ``t x WPT .. t x WPT + WPT - 1``) and the write sweep (broadcast per warp) | ``WPT x 4`` B between lanes: conflict-free at ``WPT = 1`` (``SLICE <= 32 x THREADS``), 4-way at ``WPT = 4`` -- a 4-load scan, immaterial |
| ``s_pfx_gt`` / ``s_pfx_eq`` i32 x ``WORDS`` | the word scan, the thread owning the word | the write sweep (broadcast per warp) | as the bitmaps |
| ``s_counts`` i32 x 8 | peer ``q``'s thread ``rank`` through ``st.shared::cluster`` into slot ``q`` (each slot written exactly once) | every thread, slots ``< C`` (broadcast) | trivial |

Degenerate inputs, each with its handling: ``counts[t] == 0`` -> every slot ``-1``, nothing read from the row; ``counts[t] < 0``
-> clamped to 0; ``counts[t] > n_blocks_max`` -> clamped (the row is never read past its width); ``counts[t] <= top_k`` ->
the identity list ``counts - 1 .. 0`` then ``-1``; fewer than ``k`` keys above the k-th value with ties AT it (a row of identical
logits, two distinct values, a boundary tie) -> the smaller ids of the tied value fill the list; a slice with no element
(``n`` not divisible by ``C``, the last CTAs) -> that CTA histograms nothing and selects nothing, its totals are zero; garbage
(``NaN``, huge, negative) past ``counts[t]`` is never read.  Negative logits order correctly (the key flips every bit of a negative
float).

Programmatic dependent launch (``use_pdl``): ``griddepcontrol.wait`` before the FIRST dependent global read (``counts[t]``: the
scorer writes both ``logits`` and ``counts``), ``griddepcontrol.launch_dependents`` once per CTA right after the ``counts`` read --
before the identity branch and before the slice is staged, so the identity path fires it too (the sparse tile that consumes the
list waits for THIS grid's completion; the trigger only lets it be scheduled early) -- traced under the ``use_pdl`` compile key
and launched with the attribute under the same flag; the two forms are bitwise.

Knobs (performance-only, the same function under every value -- the tests pin every value bitwise against the others):
``cluster_ctas`` in (1, 2, 4, 8) -- the portable cluster sizes; 16 needs the non-portable-cluster function attribute, which
the DSL launch does not set (the FROST occupancy probe in ``cudnn.frost.occupancy`` sets it on its own driver launch -- the
path a follow-up can reuse) and is a measured follow-up, not a default; ``threads_per_cta`` in (256, 512, 1024).  The defaults
(``DEFAULT_CLUSTER_CTAS = 8``, ``DEFAULT_THREADS_PER_CTA = 256``) are a starting point, not a measured optimum.
``cols_cap`` (the compiled row capacity, a power of two >= ``cluster_ctas x threads_per_cta`` and >= ``n_blocks_max``) sizes
``s_keys``; one artifact serves every ``n_blocks_max`` up to it.  Requirements: a CuTe DSL at or above the FROST floor with
the device's target (Rule 7, gated before anything else), a cc 9.0+ device for ``cluster_ctas > 1`` or ``use_pdl`` (thread-block
clusters, DSMEM and ``griddepcontrol`` exist from Hopper), and the per-CTA dynamic shared memory within the device's opt-in
limit (named in the decline otherwise).

Importing THIS module already needs the floor: it imports the CuTe DSL and ``cudnn.frost.tile_dsl`` at module scope, so on a DSL
below the floor the import fails inside the DSL before any gate here can name the version.  The entry a caller hits is the
DSL-free facade ``cudnn.gated_attention_block.qsa_topk`` (Rule 7): it gates the version first and imports this module lazily;
the gates in ``compile_qsa_topk_decode`` / ``qsa_topk_decode`` below are the second line, for a caller that imported this module
on a sufficient DSL.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn._cutlass_compat import SmemAllocator
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.frost.device import compute_capability, current_device, shared_memory_per_block_optin
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.tma import ld_global, st_global

__all__ = [
    "QSA_TOPK_DECODE_MAX_K",
    "MAX_ROWS",
    "CLUSTER_SIZES",
    "THREAD_COUNTS",
    "DEFAULT_CLUSTER_CTAS",
    "DEFAULT_THREADS_PER_CTA",
    "QsaTopkDecodeRecipe",
    "cols_cap_for",
    "smem_bytes_for",
    "compile_qsa_topk_decode",
    "run_qsa_topk_decode",
    "qsa_topk_decode",
]

QSA_TOPK_DECODE_MAX_K = 2048
"""The widest list this kernel writes (``block_ids.shape[1]``); the block's own declaration caps ``top_k`` at 512."""
CLUSTER_SIZES = (1, 2, 4, 8)
"""The portable thread-block cluster sizes; 16 needs the non-portable-cluster attribute (a measured follow-up)."""
THREAD_COUNTS = (256, 512, 1024)
DEFAULT_CLUSTER_CTAS = 8
DEFAULT_THREADS_PER_CTA = 256
RADIX_BITS = 8
BINS = 1 << RADIX_BITS
PASSES = 32 // RADIX_BITS
MAX_ROWS = 65535
"""The widest row block one launch serves: one cluster per row along ``grid.y``, whose extent the launch grid caps at 65535."""
_FULL_MASK = 0xFFFFFFFF
_FAKE_STREAM = None
compiled_cache = {}
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with the compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _is_pow2(x: int) -> bool:
    return x >= 1 and (x & (x - 1)) == 0


def cols_cap_for(n_blocks_max: int, cluster_ctas: int = DEFAULT_CLUSTER_CTAS, threads_per_cta: int = DEFAULT_THREADS_PER_CTA) -> int:
    """The smallest compiled row capacity that serves ``n_blocks_max`` columns: a power of two, at least one element per
    thread of the cluster (``cluster_ctas x threads_per_cta``, so every CTA's slice is a whole number of rounds)."""
    if n_blocks_max < 1:
        raise ValueError(f"n_blocks_max must be >= 1, got {n_blocks_max}")
    cap = 1
    while cap < n_blocks_max:
        cap *= 2
    return max(cap, int(cluster_ctas) * int(threads_per_cta))


def validate_topk_geometry(cols_cap: int, top_k: int, cluster_ctas: int, threads_per_cta: int) -> None:
    """Raise ``ValueError`` on any geometry the kernel cannot address (never an ``assert``: user-facing)."""
    if cluster_ctas not in CLUSTER_SIZES:
        raise ValueError(f"cluster_ctas must be one of {CLUSTER_SIZES} (the portable cluster sizes), got {cluster_ctas}")
    if threads_per_cta not in THREAD_COUNTS:
        raise ValueError(f"threads_per_cta must be one of {THREAD_COUNTS}, got {threads_per_cta}")
    if not 1 <= top_k <= QSA_TOPK_DECODE_MAX_K:
        raise ValueError(f"top_k must be in [1, {QSA_TOPK_DECODE_MAX_K}], got {top_k}")
    if not _is_pow2(cols_cap):
        raise ValueError(f"cols_cap must be a power of two, got {cols_cap}")
    if cols_cap % (cluster_ctas * threads_per_cta) != 0:
        raise ValueError(
            f"cols_cap ({cols_cap}) must be a multiple of cluster_ctas x threads_per_cta ({cluster_ctas * threads_per_cta}): every CTA's slice is whole rounds of the thread count"
        )
    if cols_cap // cluster_ctas > 0xFFFF:
        advice = (
            "use a larger cluster"
            if cluster_ctas < CLUSTER_SIZES[-1]
            else f"the row is wider than {CLUSTER_SIZES[-1]} x 65535 blocks, past this kernel's reach"
        )
        raise ValueError(f"a CTA slice of {cols_cap // cluster_ctas} elements overflows the 16-bit packed counts (65535 per slice); {advice}")


def smem_bytes_for(cols_cap: int, cluster_ctas: int, threads_per_cta: int) -> int:
    """Dynamic shared memory of one CTA (the SMEM table's rows, 16-B aligned each, plus the allocator's slack)."""
    slice_ = cols_cap // cluster_ctas
    words = slice_ // 32

    def al(x):
        return (x + 15) // 16 * 16

    total = al(slice_ * 4) + al(PASSES * BINS * 4) + al(32 * 4) + al(4 * 4) + 2 * al(words * 4) + 2 * al(words * 4) + al(8 * 4)
    return total + 128


@cute.kernel
def frost_qsa_topk_decode(
    mLogits: cute.Tensor,  # [T, n_blocks_max] fp32; row stride symbolic, unit column stride
    mCounts: cute.Tensor,  # [T] int32 contiguous
    mOut: cute.Tensor,  # [T, top_k] int32; row stride symbolic, unit column stride
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    cluster_ctas: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """Grid ``(cluster_ctas, T, 1)``, cluster ``(cluster_ctas, 1, 1)``: ``block_idx.x`` is the CTA's rank in its row's cluster
    (equal to the cluster rank because the cluster spans the whole x extent), ``block_idx.y`` the query row."""
    slice_cap = cutlass.const_expr(cols_cap // cluster_ctas)
    rounds = cutlass.const_expr(slice_cap // threads_per_cta)
    nwarps = cutlass.const_expr(threads_per_cta // 32)
    words = cutlass.const_expr(slice_cap // 32)
    wpt = cutlass.const_expr((words + threads_per_cta - 1) // threads_per_cta)
    multi = cutlass.const_expr(cluster_ctas > 1)

    smem = SmemAllocator()
    s_keys = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((slice_cap,)), byte_alignment=16)
    s_hist = smem.allocate_tensor(cutlass.Int32, cute.make_layout((PASSES * BINS,)), byte_alignment=16)
    s_wsum = smem.allocate_tensor(cutlass.Int32, cute.make_layout((32,)), byte_alignment=16)
    s_found = smem.allocate_tensor(cutlass.Int32, cute.make_layout((4,)), byte_alignment=16)
    s_bits_gt = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((words,)), byte_alignment=16)
    s_bits_eq = smem.allocate_tensor(cutlass.Uint32, cute.make_layout((words,)), byte_alignment=16)
    s_pfx_gt = smem.allocate_tensor(cutlass.Int32, cute.make_layout((words,)), byte_alignment=16)
    s_pfx_eq = smem.allocate_tensor(cutlass.Int32, cute.make_layout((words,)), byte_alignment=16)
    s_counts = smem.allocate_tensor(cutlass.Int32, cute.make_layout((8,)), byte_alignment=16)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = cutlass.Int32(cute.arch.lane_idx())
    warp = cutlass.Int32(cute.arch.make_warp_uniform(tidx // cutlass.Int32(32)))
    rank = cutlass.Int32(cute.arch.block_idx()[0])
    row = cutlass.Int32(cute.arch.block_idx()[1])
    my_bit = cutlass.Uint32(1) << lane.to(cutlass.Uint32)  # the bit of this thread within a warp mask
    lt_mask = my_bit - cutlass.Uint32(1)  # the lanes below this one (= %lanemask_lt, derived -- no DSL intrinsic needed)
    row64 = row.to(cutlass.Int64)
    four = cutlass.Int64(4)

    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    cols = cutlass.Int32(mLogits.shape[1])
    n = ld_global(mCounts.iterator.toint() + row64 * four, cutlass.Int32)
    n = n if n > cutlass.Int32(0) else cutlass.Int32(0)
    n = n if n < cols else cols
    out_row = mOut.iterator.toint() + row64 * cutlass.Int64(mOut.stride[0]) * four

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()

    if n <= cutlass.Int32(top_k):
        # the identity region: every visible block, ids descending, -1 past them; CTA 0 writes, the others leave
        if rank == cutlass.Int32(0):
            for j in range(tidx, top_k, threads_per_cta):
                val = n - cutlass.Int32(1) - j
                val = val if j < n else cutlass.Int32(-1)
                st_global(out_row + j.to(cutlass.Int64) * four, val, cutlass.Int32)
    else:
        # ---- the slice of this CTA ---------------------------------------------------------------------------------
        slice_rt = (n + cutlass.Int32(cluster_ctas - 1)) // cutlass.Int32(cluster_ctas)
        my_start = rank * slice_rt
        my_len = n - my_start
        my_len = my_len if my_len > cutlass.Int32(0) else cutlass.Int32(0)
        my_len = my_len if my_len < slice_rt else slice_rt
        row_base = mLogits.iterator.toint() + (row64 * cutlass.Int64(mLogits.stride[0]) + my_start.to(cutlass.Int64)) * four

        # zero the four histogram buffers (the stores of this CTA only; peers read them after the pass's cluster barrier)
        for i in range(tidx, PASSES * BINS, threads_per_cta):
            s_hist[i] = cutlass.Int32(0)
        # stage the slice as order-preserving keys, round-strided (consecutive lanes, consecutive columns)
        for i in range(rounds):
            e = i * cutlass.Int32(threads_per_cta) + tidx
            if e < my_len:
                bits = ld_global(row_base + e.to(cutlass.Int64) * four, cutlass.Uint32)
                sign = bits >> cutlass.Uint32(31)
                flip = (cutlass.Uint32(0) - sign) | cutlass.Uint32(0x80000000)
                s_keys[e] = bits ^ flip
        cute.arch.barrier()  # Z

        # ---- the radix select ---------------------------------------------------------------------------------------
        prefix = cutlass.Uint32(0)  # the resolved top bits of the k-th key
        pmask = cutlass.Uint32(0)  # which bits of `prefix` are resolved
        # `above`: keys resolved strictly greater than the k-th -- the select invariant is above + need == top_k; it is not read
        # after the loop (the compaction takes the per-slice totals through s_counts), kept for the invariant's readability
        above = cutlass.Int32(0)
        need = cutlass.Int32(top_k)  # how many of the current candidates are still wanted
        done = cutlass.Boolean(False)
        for p in cutlass.range_constexpr(PASSES):
            shift = cutlass.const_expr(32 - RADIX_BITS * (p + 1))
            if not done:
                # the candidates' next byte, warp-aggregated into this pass's buffer
                for i in range(rounds):
                    e = i * cutlass.Int32(threads_per_cta) + tidx
                    key = s_keys[e]
                    cand = (e < my_len) and ((key & pmask) == prefix)
                    bin_ = cutlass.Int32((key >> cutlass.Uint32(shift)) & cutlass.Uint32(0xFF))
                    tag = bin_ if cand else cutlass.Int32(BINS) + lane
                    peers = cute.arch.match_sync(_FULL_MASK, tag, "any")
                    leader = (peers & lt_mask) == cutlass.Uint32(0)
                    if cand and leader:
                        cute.arch.atomic_add(
                            s_hist.iterator + (cutlass.Int32(p * BINS) + bin_), cutlass.Int32(cute.arch.popc(peers)), sem="relaxed", scope="cta"
                        )
                # CB_p: every CTA's histogram published before any peer reads it
                if cutlass.const_expr(multi):
                    cute.arch.cluster_arrive(aligned=True)
                    cute.arch.cluster_wait()
                else:
                    cute.arch.barrier()
                # lane b of the first 256 threads merges bin 255 - b of every CTA (every CTA computes the same merged totals)
                b = cutlass.Int32(BINS - 1) - (tidx & cutlass.Int32(BINS - 1))
                off = cutlass.Int32(p * BINS) + b
                tot = cutlass.Int32(0)
                if tidx < cutlass.Int32(BINS):
                    if cutlass.const_expr(multi):
                        for q in cutlass.range_constexpr(cluster_ctas):
                            peer = cute.arch.map_dsmem_ptr(s_hist.iterator + off, cutlass.Int32(q))
                            tot = tot + peer.load()
                    else:
                        tot = s_hist[off]
                # descending inclusive scan over the bins: S(b) = sum of tot[b'] for b' >= b (thread order == descending bins)
                incl = tot
                for s in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_up(incl, 1 << s, mask_and_clamp=0)
                    if lane >= cutlass.Int32(1 << s):
                        incl = incl + other
                if (tidx < cutlass.Int32(BINS)) and (lane == cutlass.Int32(31)):
                    s_wsum[warp] = incl
                cute.arch.barrier()  # B_p2
                carry = cutlass.Int32(0)
                for w in cutlass.range_constexpr(BINS // 32):
                    if cutlass.Int32(w) < warp:
                        carry = carry + s_wsum[w]
                suffix = incl + carry  # keys of the candidates at or above bin b
                above_b = suffix - tot  # strictly above bin b
                if (tidx < cutlass.Int32(BINS)) and (above_b < need) and (need <= suffix):
                    s_found[0] = b
                    s_found[1] = above_b
                    s_found[2] = tot
                cute.arch.barrier()  # B_p3
                digit = s_found[0]
                above_in = s_found[1]
                eq = s_found[2]
                prefix = prefix | (digit.to(cutlass.Uint32) << cutlass.Uint32(shift))
                pmask = pmask | (cutlass.Uint32(0xFF) << cutlass.Uint32(shift))
                above = above + above_in
                need = need - above_in
                if cutlass.const_expr(p == PASSES - 1):
                    done = cutlass.Boolean(True)
                else:
                    done = eq == need

        # ---- the canonical compaction ------------------------------------------------------------------------------
        # sweep 1: the selection bitmaps (GT: resolved prefix above P; EQ: equal to P), one ballot word per warp per round
        for i in range(rounds):
            e = i * cutlass.Int32(threads_per_cta) + tidx
            key = s_keys[e]
            pre = key & pmask
            in_row = e < my_len
            w_gt = cutlass.Uint32(cute.arch.vote_ballot_sync(in_row and (pre > prefix)))
            w_eq = cutlass.Uint32(cute.arch.vote_ballot_sync(in_row and (pre == prefix)))
            if lane == cutlass.Int32(0):
                widx = i * cutlass.Int32(nwarps) + warp
                s_bits_gt[widx] = w_gt
                s_bits_eq[widx] = w_eq
        cute.arch.barrier()  # F1
        # the word scan: thread t owns words [t * wpt, t * wpt + wpt); packed (GT | EQ << 16) counts
        cnt = cutlass.Int32(0)
        for j in cutlass.range_constexpr(wpt):
            w = tidx * cutlass.Int32(wpt) + cutlass.Int32(j)
            if w < cutlass.Int32(words):
                cnt = cnt + cutlass.Int32(cute.arch.popc(s_bits_gt[w])) + (cutlass.Int32(cute.arch.popc(s_bits_eq[w])) << cutlass.Int32(16))
        incl2 = cnt
        for s in cutlass.range_constexpr(5):
            other2 = cute.arch.shuffle_sync_up(incl2, 1 << s, mask_and_clamp=0)
            if lane >= cutlass.Int32(1 << s):
                incl2 = incl2 + other2
        if lane == cutlass.Int32(31):
            s_wsum[warp] = incl2
        cute.arch.barrier()  # F2
        carry2 = cutlass.Int32(0)
        total = cutlass.Int32(0)
        for w in cutlass.range_constexpr(nwarps):
            v = s_wsum[w]
            total = total + v
            if cutlass.Int32(w) < warp:
                carry2 = carry2 + v
        run = incl2 - cnt + carry2  # the exclusive packed prefix of this thread's first word
        run_gt = run & cutlass.Int32(0xFFFF)
        run_eq = (run >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
        for j in cutlass.range_constexpr(wpt):
            w = tidx * cutlass.Int32(wpt) + cutlass.Int32(j)
            if w < cutlass.Int32(words):
                s_pfx_gt[w] = run_gt
                s_pfx_eq[w] = run_eq
                run_gt = run_gt + cutlass.Int32(cute.arch.popc(s_bits_gt[w]))
                run_eq = run_eq + cutlass.Int32(cute.arch.popc(s_bits_eq[w]))
        # the slice totals into every peer's slot `rank` (thread q serves peer q); CB_x publishes them and the prefixes
        if cutlass.const_expr(multi):
            if tidx < cutlass.Int32(cluster_ctas):
                dst = cute.arch.map_dsmem_ptr(s_counts.iterator + rank, tidx)
                dst.store(total)
            cute.arch.cluster_arrive(aligned=True)
            cute.arch.cluster_wait()
        else:
            if tidx == cutlass.Int32(0):
                s_counts[0] = total
            cute.arch.barrier()
        above_lt = cutlass.Int32(0)
        eq_lt = cutlass.Int32(0)
        for q in cutlass.range_constexpr(cluster_ctas):
            if cutlass.Int32(q) < rank:
                cq = s_counts[q]
                above_lt = above_lt + (cq & cutlass.Int32(0xFFFF))
                eq_lt = eq_lt + ((cq >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF))
        eq_s = (total >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
        take = need - eq_lt  # EQ ids of this slice to take, in ascending id order
        take = take if take > cutlass.Int32(0) else cutlass.Int32(0)
        take = take if take < eq_s else eq_s
        eq_sel_lt = eq_lt if eq_lt < need else need
        sel_lt = above_lt + eq_sel_lt  # selected ids in the lower slices
        # sweep 2: every selected id to its canonical slot (ascending rank pos -> slot top_k - 1 - pos)
        for i in range(rounds):
            e = i * cutlass.Int32(threads_per_cta) + tidx
            widx = i * cutlass.Int32(nwarps) + warp
            w_gt = s_bits_gt[widx]
            w_eq = s_bits_eq[widx]
            gt_before = s_pfx_gt[widx] + cutlass.Int32(cute.arch.popc(w_gt & lt_mask))
            eq_before = s_pfx_eq[widx] + cutlass.Int32(cute.arch.popc(w_eq & lt_mask))
            is_gt = (w_gt & my_bit) != cutlass.Uint32(0)
            is_eq = (w_eq & my_bit) != cutlass.Uint32(0)
            eq_sel_before = eq_before if eq_before < take else take
            pos = sel_lt + gt_before + eq_sel_before
            if is_gt or (is_eq and (eq_before < take)):
                slot = cutlass.Int32(top_k - 1) - pos
                st_global(out_row + slot.to(cutlass.Int64) * four, my_start + e, cutlass.Int32)


frost_qsa_topk_decode.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def qsa_topk_decode_launch(
    logits: cute.Tensor,
    counts: cute.Tensor,
    out: cute.Tensor,
    rows: cutlass.Int32,
    cols_cap: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
    cluster_ctas: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    smem_bytes: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    kernel = frost_qsa_topk_decode(logits, counts, out, cols_cap, top_k, cluster_ctas, threads_per_cta, use_pdl)
    if cutlass.const_expr(cluster_ctas > 1):
        kernel.launch(
            grid=(cluster_ctas, rows, 1), block=(threads_per_cta, 1, 1), cluster=(cluster_ctas, 1, 1), smem=smem_bytes, stream=stream, use_pdl=use_pdl
        )
    else:
        kernel.launch(grid=(1, rows, 1), block=(threads_per_cta, 1, 1), smem=smem_bytes, stream=stream, use_pdl=use_pdl)


class QsaTopkDecodeRecipe(NamedTuple):
    """Build-time facts of one artifact: everything is a compile-time constant of the kernel (the row count, the row
    strides and the column count enter as runtime tensor shapes -- one artifact serves every ``n_blocks_max <= cols_cap``)."""

    compiled: object
    cols_cap: int
    top_k: int
    cluster_ctas: int
    threads_per_cta: int
    use_pdl: bool
    smem_bytes: int
    device: int


def _fake_rows(dtype):
    """``[rows, cols]`` with a symbolic row stride and a unit column stride, 16-B aligned base."""
    return cute.runtime.make_fake_tensor(dtype=dtype, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(), 1), assumed_align=16)


def compile_qsa_topk_decode(
    *,
    cols_cap: int,
    top_k: int,
    cluster_ctas: int = DEFAULT_CLUSTER_CTAS,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
) -> QsaTopkDecodeRecipe:
    """Build the artifact from the geometry ALONE -- no device allocation, no launch.

    Order of the gates (Rule 7 first): the CuTe DSL floor, the geometry, the DSL's target for the current device, the device's
    cluster / PDL support (cc 9.0+), the dynamic shared memory against the device's opt-in limit.  ``NotImplementedError`` names
    the version, the target or the limit; ``ValueError`` names the malformed geometry.
    """
    global _FAKE_STREAM
    too_old = cutedsl_requirement_error("qsa_topk_decode")
    if too_old is not None:
        raise NotImplementedError(too_old)
    cols_cap, top_k, cluster_ctas, threads_per_cta = int(cols_cap), int(top_k), int(cluster_ctas), int(threads_per_cta)
    validate_topk_geometry(cols_cap, top_k, cluster_ctas, threads_per_cta)
    device = current_device()
    cc = compute_capability(device)
    arch_problem = cutedsl_arch_requirement_error(cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if cluster_ctas > 1 and cc < (9, 0):
        raise NotImplementedError(
            f"qsa_topk_decode with cluster_ctas = {cluster_ctas} needs thread-block clusters and DSMEM (cc 9.0+); found cc {cc[0]}.{cc[1]} -- use cluster_ctas = 1 there"
        )
    if use_pdl and cc < (9, 0):
        raise NotImplementedError(f"qsa_topk_decode with use_pdl=True needs griddepcontrol (cc 9.0+); found cc {cc[0]}.{cc[1]}")
    smem_bytes = smem_bytes_for(cols_cap, cluster_ctas, threads_per_cta)
    limit = shared_memory_per_block_optin(device)
    if smem_bytes > limit:
        raise NotImplementedError(
            f"qsa_topk_decode at cols_cap = {cols_cap} with cluster_ctas = {cluster_ctas} needs {smem_bytes} B of shared memory per CTA; the device allows {limit} B -- "
            + ("use a larger cluster" if cluster_ctas < CLUSTER_SIZES[-1] else "no larger portable cluster exists: the row is too wide for this device")
        )
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    key = (cols_cap, top_k, cluster_ctas, threads_per_cta, bool(use_pdl), device)
    if key not in compiled_cache:
        # the in-process memo; behind it the persistent compiled-plan cache keyed by this file's digest + the key's plain values
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_topk_decode")
        compiled_cache[key] = compile_cached(
            qsa_topk_decode_launch,
            _fake_rows(cutlass.Float32),
            cute.runtime.make_fake_compact_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(),), stride_order=(0,), assumed_align=4),
            _fake_rows(cutlass.Int32),
            cutlass.Int32(1),  # rows: runtime, the literal pins the type
            cols_cap,
            top_k,
            cluster_ctas,
            threads_per_cta,
            bool(use_pdl),
            smem_bytes,
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_qsa_topk_decode",
        )
    return QsaTopkDecodeRecipe(compiled_cache[key], cols_cap, top_k, cluster_ctas, threads_per_cta, bool(use_pdl), smem_bytes, device)


def _check_rows(name: str, x: torch.Tensor, *, dtype, rows: int, device) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor, got {None if not isinstance(x, torch.Tensor) else tuple(x.shape)}")
    if x.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {x.dtype}")
    if not x.is_cuda or x.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}, got {x.device}")
    if int(x.shape[0]) != rows:
        raise ValueError(f"{name} must hold {rows} rows (one per query row), got shape {tuple(x.shape)}")
    if x.stride(1) != 1:
        raise ValueError(f"{name} needs a unit column stride, got strides {tuple(x.stride())}")
    if x.data_ptr() % 16 != 0:
        raise ValueError(f"{name} must be 16-B aligned")


def run_qsa_topk_decode(r: QsaTopkDecodeRecipe, logits: torch.Tensor, counts: torch.Tensor, block_ids: torch.Tensor, *, stream) -> None:
    """The lowered launch: form checks and the launch -- no conversion, no allocation, no torch op (Rules 1 / 3 / 5).

    ``logits [T, n_blocks_max]`` fp32 with ``n_blocks_max <= r.cols_cap``, ``counts [T]`` int32 contiguous, ``block_ids
    [T, r.top_k]`` int32 (unit column stride; any row stride), all on the recipe's device, ``T <= MAX_ROWS`` (one cluster per row
    along ``grid.y``); ``stream`` is the raw handle the kernel launches on (``None`` = torch's current stream of that device).
    """
    if not isinstance(counts, torch.Tensor) or counts.ndim != 1 or counts.dtype != torch.int32 or not counts.is_cuda or not counts.is_contiguous():
        raise ValueError("counts must be a contiguous 1-D int32 CUDA tensor")
    rows = int(counts.shape[0])
    if rows > MAX_ROWS:
        raise ValueError(
            f"counts holds {rows} rows; the launch grid caps the row count at {MAX_ROWS} (one cluster per row along grid.y) -- split the row block"
        )
    device = counts.device
    _check_rows("logits", logits, dtype=torch.float32, rows=rows, device=device)
    _check_rows("block_ids", block_ids, dtype=torch.int32, rows=rows, device=device)
    n_blocks_max = int(logits.shape[1])
    if n_blocks_max < 1 or n_blocks_max > r.cols_cap:
        raise ValueError(f"logits holds {n_blocks_max} columns; this artifact was compiled for cols_cap = {r.cols_cap} (1 <= n_blocks_max <= cols_cap)")
    if int(block_ids.shape[1]) != r.top_k:
        raise ValueError(f"block_ids must be [T, top_k = {r.top_k}] for this artifact, got shape {tuple(block_ids.shape)}")
    dev_index = device.index if device.index is not None else torch.cuda.current_device()
    if dev_index != r.device:
        raise ValueError(f"the tensors live on device {dev_index} but the artifact was compiled for device {r.device} (compile per device)")
    if rows == 0:
        return
    handle = int(stream) if stream is not None else torch.cuda.current_stream(device).cuda_stream
    r.compiled(logits, counts, block_ids, cutlass.Int32(rows), cuda.CUstream(handle))


def qsa_topk_decode(
    logits: torch.Tensor,
    counts: torch.Tensor,
    block_ids: torch.Tensor,
    *,
    cluster_ctas: Optional[int] = None,
    threads_per_cta: Optional[int] = None,
    use_pdl: bool = False,
    stream=None,
) -> torch.Tensor:
    """Compile (cached) and run once -- the convenience form for the standalone selector, the tests and the benchmarks.

    ``block_ids [T, top_k]`` is the CALLER's buffer (nothing is allocated here); it is returned for convenience.  The two
    knobs default to a starting point (``DEFAULT_CLUSTER_CTAS``, ``DEFAULT_THREADS_PER_CTA``); every value is the same function.
    """
    too_old = cutedsl_requirement_error("qsa_topk_decode")
    if too_old is not None:
        raise NotImplementedError(too_old)
    if not isinstance(block_ids, torch.Tensor) or block_ids.ndim != 2:
        raise ValueError("block_ids must be a [T, top_k] int32 CUDA tensor")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("logits must be a [T, n_blocks_max] fp32 CUDA tensor")
    cluster_ctas = DEFAULT_CLUSTER_CTAS if cluster_ctas is None else int(cluster_ctas)
    threads_per_cta = DEFAULT_THREADS_PER_CTA if threads_per_cta is None else int(threads_per_cta)
    r = compile_qsa_topk_decode(
        cols_cap=cols_cap_for(int(logits.shape[1]), cluster_ctas, threads_per_cta),
        top_k=int(block_ids.shape[1]),
        cluster_ctas=cluster_ctas,
        threads_per_cta=threads_per_cta,
        use_pdl=use_pdl,
    )
    run_qsa_topk_decode(r, logits, counts, block_ids, stream=stream)
    return block_ids
