# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD / varlen (packed ``[T, H, D]`` + ``cu_seqlens``) device primitives.

Pass- and op-neutral: the metadata buffer, the batch ranking, the unit decode,
the persistent claim counter, and the per-sequence TMA-descriptor patchers.
Everything here is a ``@cute.jit`` macro that emits IR at the call site, so the
CALLER owns its elect / barrier discipline — each docstring names what it wants.

Promoted out of ``cudnn/sdpa/fwd/kernels/thd_sm100.py`` (issue #552's device-side
setup) when the backward needed the same pieces: a second copy of a metadata
LAYOUT is a silent-wrong-answer waiting to happen, since a reader offset and a
writer offset that disagree by one word decode garbage batches rather than
failing.  ``cudnn/linear_attention/frost/common/thd.py`` still carries an
Apache-2.0 sibling of :func:`emit_seq_descs`; collapsing it onto this one is a
cross-license move and therefore a maintainer's call, not a drive-by.

Metadata buffer, in int32 words (``THD_META_WORDS(B)`` of them)::

    [ seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1) | batch_remap(B) | live | ctr ]

``batch_remap`` is a permutation of ``[0, B)`` by DESCENDING Q length, so the
tile scheduler walks the longest sequences first (longest-processing-time): the
tail of a THD launch is then made of short sequences, which is what bounds the
ragged last wave.  ``live`` is the total unit count (device-computed — the host
cannot know ``SUM_b ceil(s_b/tile)*QH`` without a D2H) and ``ctr`` is the
persistent scheduler's claim counter.
"""

from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

# int64 words per 128-byte TMA descriptor.
TENSOR_MAP_QWORDS = 128 // 8

THD_META_WORDS = lambda b: 4 * b + 4  # noqa: E731
THD_REMAP_OFF = lambda b: 3 * b + 2  # noqa: E731
THD_LIVE_OFF = lambda b: 4 * b + 2  # noqa: E731   live unit total (device-computed)
THD_CTR_OFF = lambda b: 4 * b + 3  # noqa: E731    persistent-scheduler claim counter

# --- optional BACKWARD extension -------------------------------------------
# The d512 backward's S/dS workspace is BLOCKED over packed Q tokens: sequence b
# owns ``ceil(s_q[b]/gran)*gran`` rows starting at ``row_off[b]``, blocks packed
# end to end (see write_thd_row_offsets).  Three readers -- stage 2's workspace
# store, stage 3's A operand, and the host's reservation -- must agree on where
# that array lives, so the offsets are defined HERE with the rest of the layout
# rather than in any one of them.  The extension only APPENDS: every forward
# offset above is unchanged, so one buffer serves both passes.
#
#   [ ...the forward layout... | row_off(B+1) ]
THD_ROWOFF_OFF = lambda b: 4 * b + 4  # noqa: E731
THD_BWD_META_WORDS = lambda b: 5 * b + 5  # noqa: E731

# Threads for a THD setup launch. The metadata write itself is one elected
# thread; the batch-remap ranking that follows is parallel over batches, so the
# block is sized for that (B > THD_SETUP_THREADS just loops).
THD_SETUP_THREADS = 256


@cute.jit
def write_thd_meta(meta, ql, kl, lens_form: cutlass.Int32, n_batch: cutlass.Int32) -> None:
    """Single-thread body of the device-side THD metadata build (issue #552).

    Writes ``[seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)]`` from the
    caller's length tensors — ``(B,)`` per-batch lengths (serial cumsum; B is
    small) or the ``(B+1,)`` cu prefix-sum form, per side via ``lens_form``
    (bit 0: Q is cu, bit 1: KV is cu).  cu prefixes are NORMALIZED (element 0
    subtracted): the packed buffers are addressed from token 0, so a cu tensor
    sliced from a larger prefix means the same lengths — and the host can no
    longer validate ``cu[0] == 0`` (Rule 3), so an unnormalized base must not
    leak into the offsets the tiles and the dead-unit sentinel read.

    Callers run this under ``elect_sync``.
    """
    cuq0 = n_batch
    cuk0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    q_is_cu = (lens_form & cutlass.Int32(1)) != cutlass.Int32(0)
    kv_is_cu = (lens_form & cutlass.Int32(2)) != cutlass.Int32(0)
    if q_is_cu:
        base_q = cutlass.Int32(ql[0])
        for b in cutlass.range(0, n_batch + cutlass.Int32(1), 1, unroll=1):
            meta[cuq0 + b] = cutlass.Int32(ql[b]) - base_q
    else:
        acc = cutlass.Int32(0)
        meta[cuq0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            acc = acc + cutlass.Int32(ql[b])
            meta[cuq0 + b + cutlass.Int32(1)] = acc
    if kv_is_cu:
        base_k = cutlass.Int32(kl[0])
        meta[cuk0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            meta[cuk0 + b + cutlass.Int32(1)] = cutlass.Int32(kl[b + cutlass.Int32(1)]) - base_k
            meta[b] = cutlass.Int32(kl[b + cutlass.Int32(1)]) - cutlass.Int32(kl[b])
    else:
        acc_k = cutlass.Int32(0)
        meta[cuk0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            lkv = cutlass.Int32(kl[b])
            meta[b] = lkv
            acc_k = acc_k + lkv
            meta[cuk0 + b + cutlass.Int32(1)] = acc_k


@cute.jit
def write_thd_row_offsets(meta, n_batch: cutlass.Int32, gran: cutlass.Int32) -> None:
    """Fill ``row_off(B+1)``: where each sequence's block starts in a ragged,
    row-BLOCKED workspace, plus the total at ``row_off[B]``.

    ``row_off[b] = SUM_{i<b} ceil(s_q[i] / gran) * gran``.  Rounding each block
    UP is what keeps a sequence's tail tile inside its own block: at an
    unrounded offset the tail would overlap the next sequence's first rows and
    quietly corrupt them.

    **Choosing ``gran`` -- it is bracketed from BOTH sides, and the padding is
    pure waste, so it wants to be the smallest legal value:**

    * **At least the CONSUMER's k-tile.**  The reader walks whole k-tiles, so
      the rows between ``s_q[b]`` and the k-tile boundary must be inside the
      block and must hold the zeros the producer wrote there.  Any smaller
      ``gran`` and the reader's last k-tile reaches into the NEXT sequence's
      live rows -- nonzero data, silently summed into the wrong gradient.
    * **At least the PRODUCER's per-CTA store box, if you want to stay
      descriptor-free.**  With ``gran`` == that box height, every box is wholly
      inside its block or wholly outside it, so an out-of-range box is a
      boolean SKIP at the store site.  Below it a box STRADDLES the boundary,
      which no predicate can express -- it needs a per-sequence descriptor with
      a clipped ``GLOBAL_DIM``, i.e. a GMEM descriptor read plus a
      GENERIC->TENSORMAP acquire fence on every store.  That is real hot-path
      cost to save (box - k_tile) rows per sequence.

    For the SM100 d512 backward that is ``gran = TILE_M = 128`` (k-tile 64,
    per-CTA box 128).  The cluster spans ``TILE_M * CTA_MMA = 256`` rows, but
    each CTA stores its own 128 -- padding to 256 would double the waste for
    nothing.  At B=64 short sequences that is 20 % of the workspace rather
    than 33 %.

    Must run AFTER :func:`write_thd_meta` (it reads the ``cu_seqlens_q`` it
    wrote); callers run it on the same elected thread, where program order
    suffices.
    """
    cuq0 = n_batch
    off0 = cutlass.Int32(4) * n_batch + cutlass.Int32(4)
    run = cutlass.Int32(0)
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        meta[off0 + b] = run
        s_b = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
        run = run + ((s_b + gran - cutlass.Int32(1)) // gran) * gran
    meta[off0 + n_batch] = run


@cute.jit
def write_thd_batch_remap(meta, n_batch: cutlass.Int32, tid: cutlass.Int32, nthreads: cutlass.Int32) -> None:
    """Fill ``batch_remap`` with ``[0, B)`` sorted by descending Q length.

    Rank-by-counting rather than a sort network: each thread owns a batch and
    counts how many sequences outrank it, which is O(B^2) comparisons but fully
    parallel, branch-free and trivially deterministic.  Ties break on the
    original index, so the permutation is stable and reproducible run to run.

    WHOLE-BLOCK (not elected).  Must be called AFTER :func:`write_thd_meta`
    (it reads the ``cu_seqlens_q`` it wrote) with a barrier in between.
    """
    cuq0 = n_batch
    remap0 = cutlass.Int32(3) * n_batch + cutlass.Int32(2)
    i = tid
    while i < n_batch:
        len_i = cutlass.Int32(meta[cuq0 + i + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + i])
        rank = cutlass.Int32(0)
        for j in cutlass.range(0, n_batch, 1, unroll=1):
            len_j = cutlass.Int32(meta[cuq0 + j + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + j])
            # Descending by length; ties resolved by the lower original index.
            outranks = (len_j > len_i) | ((len_j == len_i) & (j < i))
            rank = rank + cutlass.Int32(arith.select(outranks.ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
        meta[remap0 + rank] = i
        i = i + nthreads


@cute.jit
def write_thd_live_and_ctr(
    meta,
    n_batch: cutlass.Int32,
    n_qh: cutlass.Int32,
    unit_rows: cutlass.Int32,
    n_ctas: cutlass.Int32,
    tidx: cutlass.Int32,
) -> None:
    """Publish the live-unit total and seed the persistent claim counter.

    A unit is ``unit_rows`` Q rows of one head of one sequence, so
    ``live = SUM_b ceil(s_q[b] / unit_rows) * n_qh`` — which the host cannot
    know without a D2H (issue #552), hence the kernel reading its own bound
    from here.  The counter starts at ``n_ctas``: cluster ``c`` takes unit ``c``
    from its blockIdx, then pulls from the counter.

    Leaving these two words unwritten hands out units off uninitialized
    workspace — an illegal-instruction fault, not a silent wrong answer.

    Guards on ``tidx == 0`` internally, deliberately WITHOUT ``elect_sync``:
    ``elect.sync`` picks an implementation-defined lane, so conjoining it with
    ``tidx == 0`` can select no thread at all.  The caller must have barriered
    after :func:`write_thd_meta` so the ``cu_seqlens_q`` read below is visible.
    """
    if tidx == cutlass.Int32(0):
        cuq0 = n_batch
        live = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            s_b = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
            live = live + ((s_b + unit_rows - cutlass.Int32(1)) // unit_rows) * n_qh
        meta[cutlass.Int32(4) * n_batch + cutlass.Int32(2)] = live
        meta[cutlass.Int32(4) * n_batch + cutlass.Int32(3)] = n_ctas


@cute.jit
def thd_decode_unit(
    meta,
    n_batch: cutlass.Int32,
    uid: cutlass.Int32,
    n_qh: cutlass.Int32,
    q_tile: cutlass.Int32,
    reverse_rows: bool,
) -> tuple:
    """Map a linear unit id to ``(q_tile_idx, batch, head)`` through ``batch_remap``.

    A unit is ``q_tile`` rows of one head of one sequence.  Sequences are walked
    LONGEST FIRST (the remap), and the head is the major axis within a sequence
    so consecutive units sweep the Q tiles of a single head — those share a K/V
    head, which is what keeps the claim order L2-friendly.  ``reverse_rows``
    walks a sequence's tiles from the diagonal back, putting the causal-heavy
    tiles first.

    A uid past the live total keeps ``batch == n_batch``; the caller is expected
    to bound uid against the live count instead of relying on that sentinel.
    """
    cuq0 = n_batch
    remap0 = cutlass.Int32(3) * n_batch + cutlass.Int32(2)
    f_batch = n_batch
    f_head = cutlass.Int32(0)
    f_qt = cutlass.Int32(0)
    done = cutlass.Int32(0)
    acc = cutlass.Int32(0)
    for i in cutlass.range(0, n_batch, 1, unroll=1):
        b = cutlass.Int32(meta[remap0 + i])
        s_i = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
        tb = (s_i + q_tile - cutlass.Int32(1)) // q_tile
        # A zero-length sequence contributes no unit; keep the divisor legal
        # anyway, since both quotients below are evaluated before the select.
        tb_nz = cute.math.max(tb, cutlass.Int32(1))
        units_b = tb * n_qh
        in_rng = (done == cutlass.Int32(0)) & (uid < acc + units_b)
        local = uid - acc
        qt = local % tb_nz
        if cutlass.const_expr(reverse_rows):
            qt = tb - cutlass.Int32(1) - qt
        f_batch = cutlass.Int32(arith.select(in_rng.ir_value(), b.ir_value(), f_batch.ir_value()))
        f_head = cutlass.Int32(arith.select(in_rng.ir_value(), (local // tb_nz).ir_value(), f_head.ir_value()))
        f_qt = cutlass.Int32(arith.select(in_rng.ir_value(), qt.ir_value(), f_qt.ir_value()))
        done = cutlass.Int32(arith.select(in_rng.ir_value(), cutlass.Int32(1).ir_value(), done.ir_value()))
        acc = acc + units_b
    return f_qt, f_batch, f_head


@cute.jit
def thd_claim_next(meta_t: cute.Tensor, ctr_off: cutlass.Int32, slot, tidx: cutlass.Int32) -> cutlass.Int32:
    """Take the next unit from the device-side claim counter.

    One atomic for the whole CTA, broadcast through a single SMEM word.  The
    leading barrier also separates the previous unit's use of the shared K/V
    staging from the next unit's, so the caller does not need its own.
    """
    ctr_ptr = Pointer(meta_t.iterator.raw_ptr(), dtype=cutlass.Int32) + ctr_off
    nvvm.barrier_cta_sync(0)
    if tidx == cutlass.Int32(0):
        slot[0] = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
    nvvm.barrier_cta_sync(0)
    return cutlass.Int32(slot[0])


# ---------------------------------------------------------------------------
# Per-sequence TMA descriptor arrays
#
# A packed output tensor cannot be reached by a batch COORDINATE (the row base
# cu[b] is irregular), and the last tile of a sequence overshoots into the NEXT
# sequence's rows with a live accumulator behind it.  Both are solved by giving
# each sequence its own descriptor: GLOBAL_ADDRESS at that sequence's first row,
# GLOBAL_DIM[seq] at its length, so the overshoot is TMA-clipped in hardware —
# dense packing with no predicated epilogue.
# ---------------------------------------------------------------------------


@cute.jit
def emit_seq_descs(
    base_desc,
    desc_words,
    cu,
    cu0: cutlass.Int32,
    base_ptr,
    n_batch: cutlass.Int32,
    row_stride: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
    slot_base=0,
) -> None:
    """Build a per-BATCH descriptor array over a packed ``[T, H, D]`` tensor.

    ``cu`` is an int32 array VIEW and ``cu0`` the offset of the relevant
    ``(B+1,)`` prefix inside it — so the same helper serves a standalone
    cu_seqlens tensor (``cu0 = 0``) and a prefix living inside the THD metadata
    buffer (``cu0 = THD cu_q / cu_k offset``).  ``row_stride`` is in ELEMENTS of
    the tensor's dtype (``.raw_ptr()`` is element-addressed).  ``seq_ord`` is
    innermost-first, so for ``[1, T, H, D]`` with D contiguous the sequence axis
    is **2**, not 1.  ``slot_base`` (a RUNTIME value -- the arrays are B slots long and B is not
    a compile-time constant) lets several arrays share one buffer.

    Runs on ONE elected thread; the caller elects and issues the
    ``GENERIC -> TENSORMAP`` release fence afterwards (one fence covers every
    array a setup kernel builds).
    """
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    base = base_ptr.iterator.raw_ptr()
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        cu_b = cutlass.Int32(cu[cu0 + b])
        s_b = cutlass.Int32(cu[cu0 + b + cutlass.Int32(1)]) - cu_b
        dptr = desc_base + (b + cutlass.Int32(slot_base)) * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        # Int64 fold: the product is in ELEMENTS, and a packed tensor can carry
        # more than 2^31 of them (128k tokens x 128 heads x 128 lanes already
        # does), so the Int32 form the inlined SDPA-forward version used was a
        # latent overflow. Matches the linear-attention sibling.
        row_base = base + cutlass.Int64(cu_b) * cutlass.Int64(row_stride)
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=row_base.toint(cutlass.Int64),
        )
        # Clamped to >= 1: a tensor map with a ZERO extent is not merely empty,
        # it is INVALID, and any access through it raises
        # cudaErrorIllegalInstruction -- so a zero-length sequence cannot be
        # left to the hardware clip.  The extent-1 descriptor is structurally
        # valid and never dereferenced: a consumer that can reach an empty
        # sequence's tiles must skip the access itself (stage 3's epilogue
        # does; the forward's scheduler hands out no unit for one).
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=cute.math.max(s_b, cutlass.Int32(1)),
            ord=seq_ord,
        )


@cute.jit
def emit_clamped_desc(
    base_desc,
    desc_words,
    slot: cutlass.Int32,
    extent: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Copy a base descriptor into ``slot`` with its sequence extent clamped.

    Issue #624: a THD caller binds K/V (and, in the backward, Q/dO) at buffer
    CAPACITY, not at the packed total, so a tile-tail load steps into
    caller-owned bytes that may never have been written.  Masked score columns
    are NaN-safe (the mask is a select) but the MMA is not — ``0 * NaN == NaN``
    wipes every valid row of the tile.  Clamping GLOBAL_DIM[seq] to the packed
    total makes those rows TMA-OOB, so they land as EXACT ZEROS without touching
    memory: no fill kernel, and nothing written into the caller's buffer.

    Runs on ONE elected thread; the caller elects and fences.
    """
    dptr = desc_words.iterator.raw_ptr() + slot * cutlass.Int32(TENSOR_MAP_QWORDS)
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
        (dptr + i).store((src_words + i).load())
    nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, dptr, new_value=extent, ord=seq_ord)


__all__ = [
    "TENSOR_MAP_QWORDS",
    "THD_BWD_META_WORDS",
    "THD_CTR_OFF",
    "THD_LIVE_OFF",
    "THD_META_WORDS",
    "THD_REMAP_OFF",
    "THD_ROWOFF_OFF",
    "THD_SETUP_THREADS",
    "emit_clamped_desc",
    "emit_seq_descs",
    "thd_claim_next",
    "thd_decode_unit",
    "write_thd_batch_remap",
    "write_thd_live_and_ctr",
    "write_thd_meta",
    "write_thd_row_offsets",
]
