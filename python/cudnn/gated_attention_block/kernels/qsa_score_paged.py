# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_score_paged`` -- the decode indexer's scorer over the PAGED compressed-key cache; its standalone selector is the host entry
``gated_attention_block/qsa_select_paged.py``.

A decode step of a block-sparse (``QsaSpec``) layer scores each of its ``S_q`` query rows per sequence against the compressed
keys of the complete 4-token blocks the row can see, and the top-k of those scores is the row's block list.  The serving
stacks keep the compressed keys PAGED on the main block table (one compressed row per 4 tokens, ``page_size / 4`` rows per
page), so this kernel reads them there -- once, in page order, never through a dense per-sequence copy::

    pos_t   = kv_lens[b] - S_q + s                       (row s of sequence b; t = b * S_q + s)
    n_vis_t = clamp(floor((pos_t + 1) / 4), 0, n_blocks_max)
    logits[t, j] = scale * sum_h relu(qi[t, h] . kbar[b, j])        for j < n_vis_t, kbar[b, j] = cache[table[b, j // R], j % R]
    counts[t]     = n_vis_t                                          (R = page_size / 4 compressed rows per page)
    block_lens[t] = min(top_k, n_vis_t)                              (optional output)

The ``logits [T, n_blocks_max]`` fp32 + ``counts [T]`` int32 pair is the input contract of the decode top-k that follows
(the radix kernel today, a FROST top-k later): the top-k reads row ``t``'s columns ``[0, counts[t])`` -- nothing else of the
row is written here -- and pads its output with ``-1`` past ``min(top_k, counts[t])``.  The host entry
``qsa_select_paged.qsa_select_decode_paged`` (a DSL-free module, so its import path resolves on every installed DSL) pairs the
scorer with the deterministic radix top-k (the smaller block id wins a tie at the cutoff, the selected ids come back in ASCENDING
order, so two calls are bitwise) and no torch prologue: the positions and the counts are derived on device by this kernel.

The dense-cache form of the same selection (``qsa_select_decode``, over ``kbar [B, n_blocks_max, D]``) scores through the
MMA-based indexer scorer; a decode step has 1-4 query rows per sequence, 4 indexer heads each, so an MMA tile is nearly
empty and the score is 512 FMAs per key row -- CUDA-core work.  This kernel is a bandwidth kernel: its roofline term is
the HBM bytes of the compressed rows it streams (plus the fp32 logits it writes).

Work decomposition (every constant below is a module constant or a compile key; nothing is inferred at launch):

* grid ``(ceil(n_blocks_max / blocks_per_cta), B)``, CTA = 4 warps = 128 threads, ``blocks_per_cta = 4 x 16 x units_per_warp``
  compressed rows (64 / 128 / 256; ``units_per_warp`` in {1, 2, 4} is a performance knob: the same function);
* the CTA reads ``kv_lens[b]`` and derives every row's ``n_vis_t`` and the CTA-uniform ``n_vis_max`` (the last row's); a CTA
  whose first row is at or past ``n_vis_max`` exits (after the counts write of chunk 0) -- nothing of ``qi`` / the table /
  the cache is read for a sequence with ``kv_lens == 0``;
* the prologue stages the sequence's ``S_q x H_i`` query rows ONCE in shared memory as fp32 (one ``ld.global.v4`` per lane
  per 16-byte bf16 chunk, unpacked, two ``st.shared.v4``), then ONE CTA barrier;
* each warp then streams ``units_per_warp`` units of 16 consecutive compressed rows: a warp-step covers ``R_step = min(R, 8)``
  rows with ``L = 32 / R_step`` lanes per row (4 rows x 8 lanes at page 16, 8 x 4 at page 32 and up, 2 x 16 at page 8), lane
  ``l`` of a row holding the 16-byte chunks ``k L + l`` (``16 / L`` of them, interleaved so one warp load instruction reads
  ``16 L`` contiguous bytes per row -- whole 32-byte sectors); all of a unit's loads (one ``ld.global`` of the table per
  row, one ``ld.global.v4`` per chunk) are issued before its first FMA;
* per key row and (s, h): each lane's partial dot in fp32 (``fma`` over its 128 / L elements in chunk order), the group total by
  ``log2(L)`` butterfly shuffles (``tile_dsl.pointwise.lane_group_sum``), ``relu``, the sum over heads in head order, times
  ``scale`` -- the op order of the dense selector's reference; lane ``s % L`` of the row group stores row ``s``'s logit
  (``st.global.f32``) iff ``j < n_vis_s``.

**SMEM buffer table** (one row; `frost-kernels.md` section 5):

======  =======  ====================  ======================================  ==================================  ==========================================================================
name    dtype    elems                 written by                              read by                             per-lane stride / swizzle
======  =======  ====================  ======================================  ==================================  ==========================================================================
``sQ``  fp32     ``S_q x H_i x 128``   the prologue: per lane two              per key-row step: per lane two      16 B (consecutive lanes = consecutive 16-B granules): the writes fill a 128-B
                 (= ``S_q H_i`` x      ``st.shared.v4`` per 16-B bf16 chunk,   ``ld.shared.v4`` per chunk, lane    wavefront per quarter-warp exactly; the reads of the L lanes of a row group are
                 512 B; 8 KiB at       at element ``row x 128 + 4 c`` (the     ``l`` reading chunks ``k L + l``    L consecutive granules and the R_step groups of the warp read the SAME addresses
                 S_q 4 x H_i 4, at     first 4 fp32 of chunk ``c``) and        of row ``(s, h)``                   (broadcast): one wavefront per instruction at L 4 / 8, two (the minimum for 256
                 most 32 KiB)          ``+ 64`` (the second 4): the                                                distinct bytes) at L 16.  NO swizzle: no descriptor reads it, and the stride is
                                       "half-split" row layout                                                     16 B, never a multiple of 128 B -- the half-split layout exists for exactly that
                                                                                                                   (a plain 8-fp32-per-chunk layout gives a 32-B lane stride: 2-way conflicts at L 8)
======  =======  ====================  ======================================  ==================================  ==========================================================================

**Barrier table** (one row; `frost-kernels.md` section 4) -- no mbarrier, no TMA, no tcgen05, no cluster:

==============  =========================  =====================================  =======================  =============================================================================
name            kind                       issuing lanes                          init / expected count    where
==============  =========================  =====================================  =======================  =============================================================================
``bar.sync 0``  the CTA barrier            all 128 threads (4 warps x 32), once   128 == the CTA size      after the prologue's last ``st.shared`` and before the first ``ld.shared`` of the
                (``cute.arch.barrier()``)  per CTA -- SUM(issuing lanes) = 128                             mainloop; every thread of a live CTA reaches it exactly once (a dead CTA skips the
                                                                                                           WHOLE body uniformly, so no thread waits for one that exited)
==============  =========================  =====================================  =======================  =============================================================================

PDL sites (under the ``use_pdl`` compile key, the launch attribute set with it): ``griddepcontrol.wait`` is the kernel's first
instruction -- ahead of the ``kv_lens`` read and so of every ``qi`` / table / cache read --, ``griddepcontrol.launch_dependents``
follows the barrier (the staging done; a dead CTA exits instead, which releases its share).  Without PDL (``use_pdl=False``, the
standalone default) neither instruction is traced and the kernel runs on any CuTe DSL device.

Degenerate inputs, each with its handling: ``kv_lens[b] == 0`` (a padding sequence) -> every ``pos_t < 0``, ``n_vis = 0``: no
logits column written, ``counts = block_lens = 0``, the sequence's CTAs exit after the counts write; ``kv_lens[b] < S_q`` (a
contract violation) -> the rows at a negative position get ``n_vis = 0`` likewise; ``n_vis`` inside a page -> the lanes past
``n_vis_max`` compute on a clamped page-0 row and store nothing; a ``-1`` table entry (or one at or past ``num_pages``) for a
VISIBLE block -> the key reads as ZERO (page 0 is loaded, the words zeroed by a select): logit exactly 0, still written --
the sparse core's dead-page convention, never an out-of-bounds read; a dead page past ``n_vis_max`` is never consulted; fewer
than ``top_k`` visible blocks (the identity region) -> ``block_lens = n_vis``, the paired top-k lists every visible id
ascending then ``-1``; ``n_vis`` past ``n_blocks_max`` -> clamped to the capacity (the host checks the table's reach
``max_pages x R >= n_blocks_max``).  Refused at the host before any launch: a table narrower than the capacity, a page size
that is not a power of two (at least 8), ``S_q x H_i`` past the SMEM envelope (``MAX_QUERY_ROWS``), a head dim other than 128,
misaligned or non-unit-stride operands, a mixed dtype pair, and -- by name, a typed ``NotImplementedError`` -- the e4m3 pair
(the fp8 indexer cache's read arm is a follow-up of this kernel: the bf16 pair is served).

Bytes per launch (the SOL denominator; no perf claim is made here until one is measured on an exclusive GPU):
``sum_b n_vis_max(b) x 256 B`` of compressed rows read once; ``sum_t n_vis_t x 4 B`` of logits written; the queries re-staged
per CTA (``B x n_chunks x S_q x H_i x 256 B``, L2-resident) and the table words (``sum_b ceil(n_vis_max(b) / R) x 4 B`` per
warp) -- ``qsa_select_paged.moved_bytes``.

Rule 7: this module imports the DSL at import time like its neighbours, so the documented host entry lives in the DSL-free
front door ``gated_attention_block/qsa_select_paged.py`` -- ``qsa_select_decode_paged`` gates the DSL floor, the ``sm_107a``
target on cc 10.7 and the radix top-k's cc 9.0 BEFORE importing this module -- and the kernel's host-side facts (the geometry,
the compile-key validator, the knob default, the byte model) are defined there and imported here, never the other way round
at module scope.  :func:`compile_qsa_score_paged`, the kernel-level entry a caller may reach directly, gates the floor and the
target again BEFORE any compile, so a too-old DSL reads as a version decline there too; the tests skip below the floor.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from cudnn.frost.compiled_cache import compile_cached, template_key
from cudnn.frost.device import compute_capability, current_device
from cudnn.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, lane_group_sum
from cudnn.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global

# the kernel's host-side facts live in the DSL-free front door (Rule 7: a declaration reads them without importing this module)
from ..qsa_select_paged import (
    BLOCK_SIZE,
    CHUNK_ELEMS,
    CHUNKS_PER_ROW,
    HALF,
    INDEX_HEAD_DIM,
    ROWS_PER_UNIT,
    THREADS,
    TOP_K_KERNEL_MAX,
    _ceil_div,
    _check_i32_1d,
    _kbar_rows_view,
    blocks_per_cta,
    lane_geometry,
    validate_params,
)

__all__ = [
    "QsaScorePagedRecipe",
    "dsl_decline_reason",
    "compile_qsa_score_paged",
    "run_qsa_score_paged",
]

_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with a compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------- the kernel


@cute.kernel
def frost_qsa_score_paged(
    mQ: cute.Tensor,  # [T, H_i, D] bf16; token / head strides symbolic (Int64), D contiguous
    mKbar: cute.Tensor,  # [num_pages, R, D] bf16; page / row strides symbolic (Int64), D contiguous
    mTable: cute.Tensor,  # [B, max_pages] int32; row stride symbolic (Int64), unit column stride
    mKvLens: cute.Tensor,  # [B] int32, contiguous
    mLogits: cute.Tensor,  # [T, n_blocks_max] fp32; row stride symbolic (Int64), unit column stride
    mCounts: cute.Tensor,  # [T] int32, contiguous
    mBlockLens: Optional[cute.Tensor],  # [T] int32, contiguous, or None (folded out)
    top_k: cutlass.Int32,
    scale: cutlass.Float32,
    h_i: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One CTA = chunk ``blockIdx.x`` of ``blocks_per_cta`` compressed rows of sequence ``blockIdx.y`` (module docstring)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
    r_step, lanes, chunks, steps = lane_geometry(rows_per_page)
    has_lens = cutlass.const_expr(mBlockLens is not None)
    # the ONE SMEM buffer (declared ahead of every branch): the sequence's (s, h) query rows as fp32, half-split rows
    sQ = cutlass.Array(cutlass.Float32, s_q * h_i * INDEX_HEAD_DIM, alignment=16, space=cutlass.AddressSpace.smem)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    bidx, bidy, _ = cute.arch.block_idx()
    chunk_id = cutlass.Int32(bidx)
    b = cutlass.Int32(bidy)
    n_blocks_max = cutlass.Int32(mLogits.shape[1])
    num_pages = cutlass.Int32(mKbar.shape[0])
    two = cutlass.Int64(2)  # bf16 bytes per element
    four = cutlass.Int64(4)  # int32 / fp32 bytes
    b64 = b.to(cutlass.Int64)
    zero_i = cutlass.Int32(0)
    kv_len = ld_global(mKvLens.iterator.toint() + b64 * four, cutlass.Int32)
    # the per-row visible counts (CTA-uniform; row s at position kv_len - s_q + s): clamp((pos + 1) / 4, 0, n_blocks_max) -- for a
    # negative pos + 1 the quotient is 0 or negative whichever way the division rounds, and the clamp takes it to 0: a dead
    # (kv_len 0) or a short sequence reads 0
    n_vis = []
    for s in cutlass.range_constexpr(s_q):
        pos1 = kv_len - cutlass.Int32(s_q) + cutlass.Int32(s + 1)
        nv = pos1 // cutlass.Int32(BLOCK_SIZE)
        nv = nv if nv > zero_i else zero_i
        nv = nv if nv < n_blocks_max else n_blocks_max
        n_vis.append(nv)
    n_vis_max = n_vis[s_q - 1]
    t0 = b * cutlass.Int32(s_q)
    # the counts (and block_lens) of the sequence's rows: chunk 0, one thread per row
    if (chunk_id == zero_i) & (tidx < cutlass.Int32(s_q)):
        pos1_t = kv_len - cutlass.Int32(s_q) + tidx + cutlass.Int32(1)
        nv_t = pos1_t // cutlass.Int32(BLOCK_SIZE)
        nv_t = nv_t if nv_t > zero_i else zero_i
        nv_t = nv_t if nv_t < n_blocks_max else n_blocks_max
        row64 = (t0 + tidx).to(cutlass.Int64)
        st_global(mCounts.iterator.toint() + row64 * four, nv_t, cutlass.Int32)
        if cutlass.const_expr(has_lens):
            bl = nv_t if nv_t < top_k else top_k
            st_global(mBlockLens.iterator.toint() + row64 * four, bl, cutlass.Int32)
    j_cta0 = chunk_id * cutlass.Int32(blocks_per_cta(units_per_warp))
    if j_cta0 < n_vis_max:
        # ---- prologue: stage the (s, h) query rows as fp32 (one 16-B bf16 chunk per lane per step -> two 16-B halves)
        q_base = mQ.iterator.toint()
        q_s0 = cutlass.Int64(mQ.stride[0])
        q_s1 = cutlass.Int64(mQ.stride[1])
        n_chunks_total = s_q * h_i * CHUNKS_PER_ROW
        for i in cutlass.range_constexpr(_ceil_div(n_chunks_total, THREADS)):
            idx = cutlass.Int32(i * THREADS) + tidx
            if idx < cutlass.Int32(n_chunks_total):
                row = idx // cutlass.Int32(CHUNKS_PER_ROW)  # the packed (s, h) row
                c = idx - row * cutlass.Int32(CHUNKS_PER_ROW)
                s = row // cutlass.Int32(h_i)
                h = row - s * cutlass.Int32(h_i)
                addr = q_base + ((t0 + s).to(cutlass.Int64) * q_s0 + h.to(cutlass.Int64) * q_s1 + c.to(cutlass.Int64) * cutlass.Int64(CHUNK_ELEMS)) * two
                w = ld_global_v4(addr, cutlass.Int32)
                e0, e1 = f16x2_to_f32(w[0], dtype=cutlass.BFloat16)
                e2, e3 = f16x2_to_f32(w[1], dtype=cutlass.BFloat16)
                e4, e5 = f16x2_to_f32(w[2], dtype=cutlass.BFloat16)
                e6, e7 = f16x2_to_f32(w[3], dtype=cutlass.BFloat16)
                off = row * cutlass.Int32(INDEX_HEAD_DIM) + c * cutlass.Int32(4)
                sQ.store(cutlass.Vector.from_elements((e0, e1, e2, e3), cutlass.Float32), off, vector_size=4, alignment=16)
                sQ.store(cutlass.Vector.from_elements((e4, e5, e6, e7), cutlass.Float32), off + cutlass.Int32(HALF), vector_size=4, alignment=16)
        cute.arch.barrier()  # the ONE barrier: every thread's st.shared above is visible to every thread's ld.shared below
        if cutlass.const_expr(use_pdl):
            launch_dependent_grids()
        # ---- the mainloop: this warp's units of 16 consecutive compressed rows
        kb_base = mKbar.iterator.toint()
        kb_s0 = cutlass.Int64(mKbar.stride[0])
        kb_s1 = cutlass.Int64(mKbar.stride[1])
        tb_row = mTable.iterator.toint() + b64 * cutlass.Int64(mTable.stride[0]) * four
        lg_base = mLogits.iterator.toint()
        lg_s0 = cutlass.Int64(mLogits.stride[0])
        row_in_step = lane // cutlass.Int32(lanes)
        l_in = lane - row_in_step * cutlass.Int32(lanes)
        j_warp0 = j_cta0 + warp * cutlass.Int32(units_per_warp * ROWS_PER_UNIT)
        for u in cutlass.range_constexpr(units_per_warp):
            j_unit0 = j_warp0 + cutlass.Int32(u * ROWS_PER_UNIT)
            # every load of the unit first: the page word of each step's row, then each lane's chunks of that row
            words = []
            oks = []
            js = []
            for st in cutlass.range_constexpr(steps):
                j = j_unit0 + cutlass.Int32(st * r_step) + row_in_step
                live = j < n_vis_max
                jc = j if j < n_blocks_max else n_blocks_max - cutlass.Int32(1)  # a lane past the capacity indexes a real table word
                pg = jc // cutlass.Int32(rows_per_page)
                rin = jc - pg * cutlass.Int32(rows_per_page)
                page = ld_global(tb_row + pg.to(cutlass.Int64) * four, cutlass.Int32)
                ok = live & (page >= zero_i) & (page < num_pages)
                page_c = page if ok else zero_i  # a dead or dropped page reads page 0 and is zeroed below
                kaddr = kb_base + (page_c.to(cutlass.Int64) * kb_s0 + rin.to(cutlass.Int64) * kb_s1) * two
                ws = []
                for k in cutlass.range_constexpr(chunks):
                    c = cutlass.Int32(k * lanes) + l_in
                    ws.append(ld_global_v4(kaddr + c.to(cutlass.Int64) * cutlass.Int64(CHUNK_ELEMS * 2), cutlass.Int32))
                words.append(ws)
                oks.append(ok)
                js.append(j)
            # then the arithmetic, step by step
            for st in cutlass.range_constexpr(steps):
                ok = oks[st]
                j = js[st]
                kf = []
                for k in cutlass.range_constexpr(chunks):
                    w = words[st][k]
                    f = []
                    for q in cutlass.range_constexpr(4):
                        wq = w[q] if ok else zero_i
                        lo, hi = f16x2_to_f32(wq, dtype=cutlass.BFloat16)
                        f.append(lo)
                        f.append(hi)
                    kf.append(f)
                for s in cutlass.range_constexpr(s_q):
                    total = cutlass.Float32(0.0)
                    for h in cutlass.range_constexpr(h_i):
                        acc = cutlass.Float32(0.0)
                        rbase = cutlass.Int32((s * h_i + h) * INDEX_HEAD_DIM)
                        for k in cutlass.range_constexpr(chunks):
                            c = cutlass.Int32(k * lanes) + l_in
                            off = rbase + c * cutlass.Int32(4)
                            qa = sQ.load(off, vector_size=4, alignment=16).to_elements()
                            qb = sQ.load(off + cutlass.Int32(HALF), vector_size=4, alignment=16).to_elements()
                            for i in cutlass.range_constexpr(4):
                                acc = cute.math.fma(qa[i], kf[k][i], acc)
                            for i in cutlass.range_constexpr(4):
                                acc = cute.math.fma(qb[i], kf[k][4 + i], acc)
                        acc = lane_group_sum(acc, lanes)  # the row's L lanes: every lane ends with the full dot
                        acc = acc if acc > cutlass.Float32(0.0) else cutlass.Float32(0.0)  # relu (a select)
                        total = total + acc
                    score = total * scale
                    # lane s % L of the row group writes row s's logit, iff the block is visible to THAT row
                    if (l_in == cutlass.Int32(s % lanes)) & (j < n_vis[s]):
                        st_global(lg_base + ((t0 + cutlass.Int32(s)).to(cutlass.Int64) * lg_s0 + j.to(cutlass.Int64)) * four, score, cutlass.Float32)


frost_qsa_score_paged.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def qsa_score_paged_launch(
    mQ: cute.Tensor,
    mKbar: cute.Tensor,
    mTable: cute.Tensor,
    mKvLens: cute.Tensor,
    mLogits: cute.Tensor,
    mCounts: cute.Tensor,
    mBlockLens: Optional[cute.Tensor],
    top_k: cutlass.Int32,
    scale: cutlass.Float32,
    n_chunks: cutlass.Int32,
    batch: cutlass.Int32,
    h_i: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_score_paged(mQ, mKbar, mTable, mKvLens, mLogits, mCounts, mBlockLens, top_k, scale, h_i, s_q, rows_per_page, units_per_warp, use_pdl).launch(
        grid=(n_chunks, batch, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl
    )


# --------------------------------------------------------------------------------------------- compile + run

compiled_cache = {}


class QsaScorePagedRecipe(NamedTuple):
    """Build-time facts of one artifact (Rule 4: one artifact per (H_i, S_q, page size, units knob, block_lens presence, PDL));
    the batch, the cache capacity, the table width, the page count and every stride ride in at runtime."""

    compiled: object
    h_i: int
    s_q: int
    page_size: int
    rows_per_page: int
    units_per_warp: int
    has_block_lens: bool
    use_pdl: bool
    device: int


def _fake_q(h_i: int):
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.BFloat16, shape=(cute.sym_int(), int(h_i), INDEX_HEAD_DIM), stride=(cute.sym_int(width=64), cute.sym_int(width=64), 1), assumed_align=16
    )


def _fake_kbar(rows_per_page: int):
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.BFloat16,
        shape=(cute.sym_int(), int(rows_per_page), INDEX_HEAD_DIM),
        stride=(cute.sym_int(width=64), cute.sym_int(width=64), 1),
        assumed_align=16,
    )


def _fake_i32_2d():
    return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=4)


def _fake_f32_2d():
    return cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=16)


def _fake_i32_1d():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4)


def dsl_decline_reason(device: Optional[int] = None) -> Optional[str]:
    """Rule 7: None when this DSL can build the kernel for this device, else the typed reason -- the version floor first
    (device-free), then the ``sm_107a`` target on cc 10.7.  The kernel itself is plain CUDA-core code (loads, shuffles, one
    barrier) and runs on any CuTe DSL device; PDL adds the cc 9.0 requirement (:func:`compile_qsa_score_paged`)."""
    too_old = cutedsl_requirement_error("qsa_score_paged")
    if too_old is not None:
        return too_old
    dev = current_device() if device is None else int(device)
    return cutedsl_arch_requirement_error(compute_capability(dev))


def compile_qsa_score_paged(
    *,
    h_i: int,
    s_q: int,
    page_size: int,
    units_per_warp: int = 1,
    has_block_lens: bool = True,
    use_pdl: bool = False,
    device: Optional[int] = None,
) -> QsaScorePagedRecipe:
    """Build from SHAPES ALONE (no allocation, no launch).  The DSL / arch gate runs BEFORE the compile (Rule 7): a decline
    is a ``NotImplementedError`` naming the version or the target, never an error from inside the DSL."""
    global _FAKE_STREAM
    validate_params(h_i=h_i, s_q=s_q, page_size=page_size, units_per_warp=units_per_warp)
    dev = current_device() if device is None else int(device)
    why = dsl_decline_reason(dev)
    if why is not None:
        raise NotImplementedError(why)
    cc = compute_capability(dev)
    if use_pdl and cc < (9, 0):
        raise NotImplementedError(
            f"use_pdl=True traces the programmatic-dependent-launch pair (griddepcontrol), an instruction of cc 9.0 and later; this device is cc {cc[0]}.{cc[1]}"
        )
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    rows_per_page = int(page_size) // BLOCK_SIZE
    key = (int(h_i), int(s_q), int(page_size), int(units_per_warp), bool(has_block_lens), bool(use_pdl), dev)
    if key not in compiled_cache:
        # The dict is the in-process memo; behind it sits the persistent compiled-plan cache, keyed by this file's digest
        # (``FROST_SOURCE_DIGEST`` above) + the key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_score_paged")
        compiled_cache[key] = compile_cached(
            qsa_score_paged_launch,
            _fake_q(h_i),
            _fake_kbar(rows_per_page),
            _fake_i32_2d(),  # the table [B, max_pages]
            _fake_i32_1d(),  # kv_lens
            _fake_f32_2d(),  # logits [T, n_blocks_max]
            _fake_i32_1d(),  # counts
            _fake_i32_1d() if has_block_lens else None,
            cutlass.Int32(0),  # top_k    ) runtime; the zeros pin the TYPE only
            cutlass.Float32(0.0),  # scale    )
            cutlass.Int32(0),  # n_chunks )
            cutlass.Int32(0),  # batch    )
            int(h_i),
            int(s_q),
            int(rows_per_page),
            int(units_per_warp),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_block_qsa_score_paged",
        )
    return QsaScorePagedRecipe(
        compiled=compiled_cache[key],
        h_i=int(h_i),
        s_q=int(s_q),
        page_size=int(page_size),
        rows_per_page=rows_per_page,
        units_per_warp=int(units_per_warp),
        has_block_lens=bool(has_block_lens),
        use_pdl=bool(use_pdl),
        device=dev,
    )


def run_qsa_score_paged(
    r: QsaScorePagedRecipe,
    qi: torch.Tensor,
    kbar: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    logits: torch.Tensor,
    counts: torch.Tensor,
    block_lens: Optional[torch.Tensor],
    *,
    top_k: int,
    scale: float,
    stream,
) -> None:
    """Validate the FORM of every operand (no device read, Rule 3) and launch (Rule 1: no conversion, no allocation).

    ``qi [T, H_i, D]`` bf16 with a unit last stride and 16-byte-aligned token / head strides (a column slice of a wider slab
    reads in place), ``T = B x s_q``; ``kbar`` the paged cache (:func:`_kbar_rows_view`); ``block_table [B, max_pages]`` int32
    with a unit column stride, ``max_pages x R >= n_blocks_max``; ``kv_lens [B]`` int32; ``logits [T, n_blocks_max]`` fp32 with a
    unit column stride and 16-byte-aligned rows; ``counts [T]`` int32; ``block_lens [T]`` int32 iff the artifact traces it;
    ``top_k`` in ``[1, 2048]``; ``scale > 0`` (the ordering is scale-free); ``stream`` a raw CUDA stream handle (an ``int`` or a
    ``cuda.CUstream``; the host entry resolves a ``torch.cuda.Stream`` to it)."""
    if not isinstance(qi, torch.Tensor) or qi.dim() != 3:
        raise ValueError(f"qi must be a [T, H_i, {INDEX_HEAD_DIM}] bf16 tensor, got {getattr(qi, 'shape', type(qi).__name__)}")
    if qi.dtype != torch.bfloat16:
        raise ValueError(f"qi must be bfloat16 for this artifact, got {qi.dtype}")
    if not qi.is_cuda:
        raise ValueError(f"qi must be a CUDA tensor (every operand is read through a device pointer), got device {qi.device}")
    device = qi.device
    t, h_i, d = (int(x) for x in qi.shape)
    if h_i != r.h_i or d != INDEX_HEAD_DIM:
        raise ValueError(f"qi must be [T, H_i={r.h_i}, {INDEX_HEAD_DIM}] (H_i is fixed per artifact), got {tuple(qi.shape)}")
    qs0, qs1, qs2 = (int(x) for x in qi.stride())
    if qs2 != 1 or (qs0 * 2) % 16 or (qs1 * 2) % 16 or qi.data_ptr() % 16:
        raise ValueError(
            f"qi needs a unit last stride and 16-byte-aligned token / head strides and base, got strides {tuple(qi.stride())}, data_ptr {qi.data_ptr():#x}"
        )
    if not isinstance(block_table, torch.Tensor) or block_table.dim() != 2 or block_table.dtype != torch.int32:
        raise ValueError(
            f"block_table must be an int32 [B, max_pages] tensor, got {getattr(block_table, 'shape', type(block_table).__name__)} {getattr(block_table, 'dtype', '')}"
        )
    batch, max_pages = (int(x) for x in block_table.shape)
    if batch < 1 or max_pages < 1:
        raise ValueError(f"block_table must hold at least one sequence and one page column, got {tuple(block_table.shape)}")
    if int(block_table.stride(1)) != 1 or block_table.device != device:
        raise ValueError(f"block_table needs a unit column stride on {device}, got strides {tuple(block_table.stride())} on {block_table.device}")
    if t != batch * r.s_q:
        raise ValueError(
            f"qi holds T = {t} rows but block_table / kv_lens hold B = {batch} sequences of s_q = {r.s_q} rows (T must be B x s_q = {batch * r.s_q})"
        )
    _check_i32_1d(kv_lens, "kv_lens", batch, device)
    kbar_view = _kbar_rows_view(kbar, r.rows_per_page, device)
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.dtype != torch.float32:
        raise ValueError(
            f"logits must be an fp32 [T, n_blocks_max] tensor, got {getattr(logits, 'shape', type(logits).__name__)} {getattr(logits, 'dtype', '')}"
        )
    lt, n_blocks_max = (int(x) for x in logits.shape)
    if lt != t or n_blocks_max < 1:
        raise ValueError(f"logits must be [T={t}, n_blocks_max >= 1], got {tuple(logits.shape)}")
    ls0, ls1 = (int(x) for x in logits.stride())
    if ls1 != 1 or (ls0 * 4) % 16 or logits.data_ptr() % 16 or logits.device != device:
        raise ValueError(
            f"logits needs a unit column stride, 16-byte-aligned rows and base, on {device}; got strides {tuple(logits.stride())}, data_ptr {logits.data_ptr():#x}, {logits.device}"
        )
    if max_pages * r.rows_per_page < n_blocks_max:
        raise ValueError(
            f"block_table maps max_pages x R = {max_pages} x {r.rows_per_page} = {max_pages * r.rows_per_page} compressed rows, fewer than the "
            f"n_blocks_max = {n_blocks_max} columns of logits: widen the table or shrink the logits"
        )
    _check_i32_1d(counts, "counts", t, device)
    if r.has_block_lens != (block_lens is not None):
        raise ValueError(f"this artifact was compiled {'with' if r.has_block_lens else 'without'} block_lens; the call must match (Rule 1: no silent fallback)")
    if block_lens is not None:
        _check_i32_1d(block_lens, "block_lens", t, device)
    if not 1 <= int(top_k) <= TOP_K_KERNEL_MAX:
        raise ValueError(f"top_k must be in [1, {TOP_K_KERNEL_MAX}], got {top_k}")
    if not float(scale) > 0.0:
        raise ValueError(f"scale must be positive (the ordering is scale-free), got {scale}")
    n_chunks = _ceil_div(n_blocks_max, blocks_per_cta(r.units_per_warp))
    r.compiled(
        qi,
        kbar_view,
        block_table,
        kv_lens,
        logits,
        counts,
        block_lens,
        cutlass.Int32(int(top_k)),
        cutlass.Float32(float(scale)),
        cutlass.Int32(n_chunks),
        cutlass.Int32(batch),
        cuda.CUstream(int(stream)),
    )
