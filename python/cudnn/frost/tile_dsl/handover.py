# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""A hand-over between two CTAs of one grid through global memory.

One CTA stores data to global memory with ordinary stores; another CTA of the same grid loads it with ordinary loads.  Both
are weak operations: nothing orders them, no mbarrier is shared by the two CTAs, and a load may also be served from the
loading SM's first-level cache.  The PTX memory model orders such a pair through a word of global memory:

* every WRITING lane, behind its own stores, issues a release operation at GPU scope on the word (:func:`release_add`);
* the READING side polls the word with strong loads until the writers' sum has arrived and then issues ONE acquire load at
  GPU scope (:func:`poll_then_acquire`); its loads of the data follow that acquire.

    writer, every lane:    <stores of this lane>
                           release_add(word)                                   # atom.global.release.add
    reader:                count = poll_then_acquire(word, writing_lanes)       # ld.relaxed.gpu ... ; ld.acquire.gpu
                           src = src + consumed_zero(count, rz)                 # tile_dsl.consume
                           <loads from src>                                    #   or: + acquired_zero(count, writing_lanes)

THE CONTRACT.

* THE WORD IS A COUNTER.  It is zero before the first writer of the launch; every writing lane adds once; the reader waits
  for the number of writing lanes.  Whoever owns the word clears it for every launch (a kernel launched in front on the same
  stream, behind which the reading kernel issues ``griddepcontrol.wait``).  Neither form elects, masks or counts for the
  caller: SUM(adding lanes) == the value waited for is the caller's ledger, exactly as an mbarrier's init count is.
* THE READER'S LOADS TAKE A VALUE FROM THE ACQUIRE LOAD.  Source order is not execution order: a load whose address does not
  depend on the acquired value may be scheduled in front of it.  Add a zero that is computed from the acquired value to the
  address or the index the loads are formed from: ``tile_dsl.consume.consumed_zero(count, rz)`` (the count masked by the
  kernel's release-zero: 0 for any count), or :func:`acquired_zero` (the count minus the number of adding lanes, for a
  reader that must not read the release-zero: 0 only for the ledger's count).  When other lanes load, the acquiring lane
  stores such a value and releases them through a barrier of the CTA (an mbarrier arrive is a release at CTA scope, its
  wait an acquire).  A TMA load is formed from its coordinates: the zero goes into what they are computed from.
* A READER THROUGH ANOTHER PROXY NEEDS THE WRITER'S PROXY FENCE.  Ordinary stores go through the generic proxy; a TMA or bulk
  copy reads through the async proxy, and a release orders nothing across proxies.  A writer whose data is read that way
  calls ``release_add(word, async_reader=True)``: each lane issues ``fence.proxy.async.global`` behind its stores and in
  front of its add, so the fence lies on the path from the stores to the reader's acquire.
* A WRITER NEVER WAITS FOR ITS READER.  Whatever the reader polls for must be reached by its writer on a path without a poll,
  or the two CTAs wait for each other for ever: nothing here has a time limit.
* ONE RELEASE PER WARP, NOT PER LANE, in the generated code: the assembler folds the lanes' adds of one warp into one
  reduction of the lane count behind one GPU-scope memory barrier.  That barrier drains the SM: keep the form off per-chunk
  and per-iteration paths.

``test/python/sdpa/frost/test_tile_dsl_handover.py`` pins what the forms compile to.
"""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm

__all__ = ["acquired_zero", "poll_then_acquire", "release_add"]


@cute.jit
def release_add(word, amount: cutlass.Constexpr[int] = 1, async_reader: cutlass.Constexpr[bool] = False):
    """Every calling lane adds ``amount`` to the int32 ``word`` of global memory with RELEASE at GPU scope
    (``atom.global.release.add``); the previous value is not used.  The lane's earlier stores are ordered in front of every
    acquire that observes the sum.  ``async_reader=True``: the reader loads the data through the async proxy (a TMA or bulk
    copy), so the lane first issues ``fence.proxy.async.global`` -- behind its stores, in front of the add."""
    if cutlass.const_expr(async_reader):
        nvvm.fence_proxy("async.global")
    nvvm.atomicrmw("add", word, cutlass.Int32(amount), mem_order="release", syncscope="gpu")


@cute.jit
def poll_then_acquire(word, at_least, warp_uniform: cutlass.Constexpr[bool] = False):
    """Every calling lane reads the int32 ``word`` of global memory with relaxed loads at GPU scope until it reads at least
    ``at_least``, then with ONE acquire load at GPU scope, whose value is returned.  The loop has no time limit.

    ``warp_uniform=True`` is for a WHOLE warp that calls with one ``word`` and one ``at_least`` in converged control flow:
    the warp leaves the loop together, on a vote that every lane has read the count (``vote.sync.all``), so it is still
    converged behind the loop.  A warp whose later code elects a lane needs that: a loop that lanes may leave at different
    turns -- or a loop inside the branch of one elected lane -- makes the assembler treat everything behind it as possibly
    diverged (every later elect and predicated operation is then lowered per lane, at the price of registers), and lanes
    that really left at different turns would elect twice.  The lanes read the word one after another and may see different
    counts at one turn; the vote then takes one more turn (the counter only grows while it is polled)."""
    seen = cutlass.Int32(nvvm.load_ext(word, dtype=cutlass.Int32, order="relaxed", scope="gpu"))
    if cutlass.const_expr(warp_uniform):
        reached = cutlass.Boolean(nvvm.vote_sync(0xFFFFFFFF, seen >= at_least, "all"))
        while not reached:
            seen = cutlass.Int32(nvvm.load_ext(word, dtype=cutlass.Int32, order="relaxed", scope="gpu"))
            reached = cutlass.Boolean(nvvm.vote_sync(0xFFFFFFFF, seen >= at_least, "all"))
    else:
        while seen < at_least:
            seen = cutlass.Int32(nvvm.load_ext(word, dtype=cutlass.Int32, order="relaxed", scope="gpu"))
    return cutlass.Int32(nvvm.load_ext(word, dtype=cutlass.Int32, order="acquire", scope="gpu"))


@cute.jit
def acquired_zero(count, adding_lanes: cutlass.Constexpr[int]):
    """An ``Int32`` that is 0 at run time and that the generated code computes from ``count``, the value
    :func:`poll_then_acquire` returned for a counter that exactly ``adding_lanes`` lanes add 1 to: ``count - adding_lanes``.
    Add it to the address, the index or the stored word that must not be formed before the acquire load -- the anchor of
    the hand-over for a reader that must not read the kernel's release-zero (``tile_dsl.consume.consumed_zero`` is the
    masked form, 0 for any count).

    It is 0 only while the ledger holds: the counter is cleared for every launch, exactly ``adding_lanes`` lanes add to it
    and ``poll_then_acquire`` waited for that number.  A counter that reads more moves whatever the zero is added to: a
    broken ledger shows as a wrong index instead of passing silently."""
    return count - cutlass.Int32(adding_lanes)
