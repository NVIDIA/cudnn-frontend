# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD / varlen setup launch for the SM100 SDPA backward chain.

One launch per execute, ahead of stages 1-3, building the facts only the DEVICE
can know: the packed totals and per-sequence lengths live in ``cu_seqlens``, and
reading them on the host would be a D2H sync per iteration (issue #552).

* the shared THD metadata buffer (:mod:`cudnn.frost.tile_dsl.thd`), extended
  with the ``row_off(B+1)`` block offsets of the blocked S/dS workspace;
* the persistent scheduler's live-unit total and claim counter;
* on the MXFP8 row only, the per-sequence scale-factor TILE prefixes
  ``sf_meta = [cu_sf_q(B+1) | cu_sf_k(B+1)]`` (``cu_sf[b] = SUM_{i<b} ceil(s_i / 128)``,
  the packed per-sequence-TILE-padded SF layout's tile base of sequence ``b``:
  ``config_sm100.STAGE3_THD_SF_*`` -- the layout the block-scale stage-3 arm and
  the MXFP8 main kernel both read) in a SEPARATE buffer: the shared metadata
  layout above is fixed, so anything new goes beside it, never inside it.

Also here, the MXFP8 row's two THD-only device passes: :func:`pad_sf_atoms_thd_host`,
the per-sequence SF pad staging (every E8M0 byte scaling a position past a
sequence's length, in that sequence's LAST tile, zeroed into a packed staging copy
-- the obligation with no bf16 counterpart, sdpa-invariants s2), and
:func:`thd_claim_unit`, the one-lane claim-counter atomic both THD bodies' scheduler
warps issue.

**Descriptors are deliberately NOT built here.**  A TMA descriptor's box dims,
swizzle and dim ORDER are the consuming kernel's private geometry -- stage 2
addresses ``(d, head, seq, batch)`` so its sequence axis is ``ord=2``, while
stage 3's C operand is ``(n, m, h, b)`` and its sequence axis is ``ord=1`` -- so
a shared builder would either take those from a caller with no business knowing
them, or hardcode one stage's answer and silently mis-patch the other.  Each
stage patches its own descriptors from its own ``_host`` with
``tile_dsl.thd.emit_seq_descs`` / ``emit_clamped_desc``; this launch publishes
the metadata they all read.
"""

from typing import Optional

from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

from typing import Optional

import cutlass
import cutlass.cute as cute

from cudnn.frost.tile_dsl.thd import (
    THD_BWD_META_WORDS,
    THD_ROWOFF_OFF,
    THD_SETUP_THREADS,
    write_thd_batch_remap,
    write_thd_live_and_ctr,
    write_thd_meta,
    write_thd_prefix_warp,
    write_thd_port_origins,
    write_thd_row_offsets,
)
from cudnn.sdpa.bwd.config_sm100 import STAGE3_THD_SF_CU_K_OFF, STAGE3_THD_SF_CU_Q_OFF, STAGE3_THD_SF_META_WORDS
from cudnn.sdpa.bwd.config_sm107 import MX_BLOCK, SF_ATOM_BYTES, SF_ATOM_COLS, SF_ATOM_ROWS

__all__ = [
    "THD_BWD_META_WORDS",
    "THD_ROWOFF_OFF",
    "THD_SETUP_THREADS",
    "THD_SF_META_WORDS",
    "THD_SF_CU_Q_OFF",
    "THD_SF_CU_K_OFF",
    "THD_SF_TILE",
    "THD_SF_SLAB_BYTES",
    "build_thd_bwd_setup_kernel",
    "thd_bwd_setup_host",
    "build_thd_meta_kernel",
    "thd_meta_host",
    "thd_claim_unit",
    "pad_sf_atoms_thd_host",
    "ORIGIN_PORTS",
    "build_thd_meta_origins_kernel",
    "thd_meta_origins_host",
]

# The MXFP8 row's per-sequence SF TILE prefixes: ONE layout, the block-scale stage-3 arm's (``config_sm100.STAGE3_THD_SF_*``),
# re-exported under the chain's names so the setup launch (writer), the pre-pass and the main kernel (readers) spell it once.
THD_SF_META_WORDS = STAGE3_THD_SF_META_WORDS  # 2 * (B + 1) int32 words
THD_SF_CU_Q_OFF = STAGE3_THD_SF_CU_Q_OFF  # cu_sf_q[0 .. B] at 0
THD_SF_CU_K_OFF = STAGE3_THD_SF_CU_K_OFF  # cu_sf_k[0 .. B] at B + 1
# The packed SF convention pads every sequence to whole 128-token tiles (one F8_128x4 atom row of tiles): the unit of the prefixes.
THD_SF_TILE = SF_ATOM_ROWS  # 128
# One (head, tile) SF slab at d = 256: two 512-B F8_128x4 atoms (rowwise: the two 4-group d-chunks; columnwise: the two D-planes,
# packed contiguously under THD) -- ``sf_smem_size`` of the d=256 kernels, derived from the atom geometry, never a literal.
_SF_D = 256
THD_SF_SLAB_BYTES = (_SF_D // SF_ATOM_ROWS) * SF_ATOM_BYTES  # 1024
_PAD_SF_THREADS = THD_SF_SLAB_BYTES // 4  # 256: one 4-byte word of the slab per thread (the 4 scale columns of one atom row)


@cute.jit
def _write_thd_sf_tile_prefix_warp(sfm, lens, n_batch: cutlass.Int32, prefix_offset, is_cu: cutlass.Boolean, lane: cutlass.Int32) -> None:
    """One FULL warp writes ``sfm[prefix_offset + b] = SUM_{i<b} ceil(s_i / THD_SF_TILE)`` for ``b in [0, B]``.

    The shared scan (:func:`write_thd_prefix_warp` with ``round_to = THD_SF_TILE``) yields the prefix of TILE-ROUNDED TOKENS;
    the second pass divides each entry by the tile.  Lane ``b % 32`` wrote entry ``b + 1`` and divides it again -- the same
    lane, program order -- so no barrier sits between the two passes; entry 0 is 0 either way.  Integer arithmetic matches
    the shared scan's (``ceil`` as ``(s + T - 1) // T * T`` then ``// T``), so the prefix is exact for any length."""
    off = cutlass.Int32(prefix_offset)
    tile = cutlass.Int32(THD_SF_TILE)
    write_thd_prefix_warp(sfm, lens, n_batch, off, is_cu, lane, store_lengths=False, round_to=tile)
    for start in cutlass.range(0, n_batch, 32, unroll=1):
        b = start + lane
        if b < n_batch:
            sfm[off + b + cutlass.Int32(1)] = cutlass.Int32(sfm[off + b + cutlass.Int32(1)]) // tile


@cute.jit
def _write_thd_sf_tile_prefixes_serial(meta, sfm, n_batch: cutlass.Int32) -> None:
    """Single-thread twin of the warp pass for ``n_batch <= 1`` (the shared layout's own single-thread path): both SF tile
    prefixes from the ``cu_q`` / ``cu_k`` prefixes :func:`write_thd_meta` wrote just before on the SAME thread (program order)."""
    cu_q0 = n_batch
    cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    sfq0 = cutlass.Int32(THD_SF_CU_Q_OFF)
    sfk0 = cutlass.Int32(THD_SF_CU_K_OFF(n_batch))
    tile = cutlass.Int32(THD_SF_TILE)
    run_q = cutlass.Int32(0)
    run_k = cutlass.Int32(0)
    sfm[sfq0] = run_q
    sfm[sfk0] = run_k
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        s_q = cutlass.Int32(meta[cu_q0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cu_q0 + b])
        s_kv = cutlass.Int32(meta[cu_k0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cu_k0 + b])
        run_q = run_q + (s_q + tile - cutlass.Int32(1)) // tile
        run_k = run_k + (s_kv + tile - cutlass.Int32(1)) // tile
        sfm[sfq0 + b + cutlass.Int32(1)] = run_q
        sfm[sfk0 + b + cutlass.Int32(1)] = run_k


@cute.kernel
def build_thd_bwd_setup_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    ws_gran: cutlass.Int32,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
    kv_blocked: cutlass.Constexpr[bool] = False,
    sf_meta_t: Optional[cute.Tensor] = None,
) -> None:
    """Metadata, blocked-workspace row offsets, live-unit total, claim counter.

    ``ws_gran`` is the S/dS block granularity (stage 2's per-CTA store box);
    ``cga_tile_m`` is stage 2's unit height.  They are the same number today but
    are passed separately so a tile change cannot silently redefine the
    workspace layout.

    ``kv_blocked`` (appended, default False = the SM100 d512 chain byte for
    byte) names the axis the consumer blocks: False = the S/dS workspace is
    blocked over packed Q tokens and a unit is ``cga_tile_m`` Q rows; True = the
    workspace is blocked over packed KV tokens and a unit is a ``cga_tile_m``-row
    kv block (the sm107 d256 backward), so both the ``row_off`` prefix and the
    live-unit total are computed from the KV lengths.  The batch ranking stays
    by descending Q length either way (a kv block's cost is its q-tile count).

    ``sf_meta_t`` (appended, default None = every chain but the MXFP8 row's) is
    the int32 ``[cu_sf_q(B+1) | cu_sf_k(B+1)]`` buffer of per-sequence scale-factor
    TILE prefixes (``THD_SF_META_WORDS(B)`` words; ``cu_sf[b] = SUM_{i<b}
    ceil(s_i / THD_SF_TILE)``).  Warps 3 and 4 of the block write it from the
    length tensors directly (warps 0-2 own the shared layout's three prefixes):
    the shared scan with ``round_to = THD_SF_TILE`` yields ROUNDED TOKENS, so
    each lane divides the entries it wrote by the tile -- same lane, program
    order, no cross-lane hazard.  Under ``n_batch <= 1`` the elected thread
    writes both prefixes serially after the metadata.  Tiles, not tokens:
    ``cu[b] // 128`` is short by one tile for every ragged sequence before ``b``
    (finite, plausible, wrong scales).

    ``n_qh`` is the head extent ONE stage-2 launch decodes with -- the head
    CHUNK, not the plan's head count: the chain loops ``heads // chunk`` stage-2
    launches over this one buffer and each hands its kernel ``n_qh = chunk``, so
    ``live`` has to be the per-launch total or the scheduler hands out dead
    units.  The claim counter seeded here serves the first launch only; stage
    2's clamp kernel re-seeds it before every launch
    (``_clamp_thd_input_descs``), so this launch stays once per execute.
    """
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    meta = cutlass.make_array_view(meta_t)
    # The prefix the blocked workspace and the unit count follow: cu_q at B, cu_k at 2B+1.
    blk_cu0 = (cutlass.Int32(2) * n_batch + cutlass.Int32(1)) if cutlass.const_expr(kv_blocked) else n_batch
    if n_batch <= cutlass.Int32(1):
        if nvvm.elect_sync() and tidx < cutlass.Int32(32):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
            write_thd_row_offsets(meta, n_batch, ws_gran, cu0=blk_cu0)
            if cutlass.const_expr(sf_meta_t is not None):
                _write_thd_sf_tile_prefixes_serial(meta, cutlass.make_array_view(sf_meta_t), n_batch)
    else:
        warp = cutlass.Int32(tidx) // cutlass.Int32(32)
        lane = cutlass.Int32(tidx) % cutlass.Int32(32)
        if warp == cutlass.Int32(0):
            write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
        if warp == cutlass.Int32(1):
            write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
        if warp == cutlass.Int32(2):
            # Read the blocked axis's lengths directly: all three prefix regions are
            # disjoint, so the existing publication barrier orders them together.
            if cutlass.const_expr(kv_blocked):
                write_thd_prefix_warp(
                    meta, cutlass.make_array_view(kv_lens_t), n_batch, 4 * n_batch + 4, (lens_form & 2) != 0, lane, store_lengths=False, round_to=ws_gran
                )
            else:
                write_thd_prefix_warp(
                    meta, cutlass.make_array_view(q_lens_t), n_batch, 4 * n_batch + 4, (lens_form & 1) != 0, lane, store_lengths=False, round_to=ws_gran
                )
        if cutlass.const_expr(sf_meta_t is not None):
            # The MXFP8 row's SF TILE prefixes, from the length tensors (its own buffer: disjoint from every shared-layout write).
            if warp == cutlass.Int32(3):
                _write_thd_sf_tile_prefix_warp(
                    cutlass.make_array_view(sf_meta_t), cutlass.make_array_view(q_lens_t), n_batch, THD_SF_CU_Q_OFF, (lens_form & 1) != 0, lane
                )
            if warp == cutlass.Int32(4):
                _write_thd_sf_tile_prefix_warp(
                    cutlass.make_array_view(sf_meta_t), cutlass.make_array_view(kv_lens_t), n_batch, THD_SF_CU_K_OFF(n_batch), (lens_form & 2) != 0, lane
                )
    # Outside the elect: every thread helps rank the batches.  The barrier makes
    # the cu_seqlens_q written above visible to the whole block first.  Stage 2's
    # decode walks this permutation, so skipping it leaves the region
    # uninitialized and units decode garbage batches.
    cute.arch.barrier()
    write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    cute.arch.barrier()
    write_thd_live_and_ctr(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx), cu0=blk_cu0)


build_thd_bwd_setup_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def thd_bwd_setup_host(
    meta_t,
    q_lens_t,
    kv_lens_t,
    lens_form,
    n_qh,
    n_batch,
    ws_gran,
    cga_tile_m,
    n_clusters,
    stream=None,
    kv_blocked: cutlass.Constexpr[bool] = False,
    sf_meta_t=None,
):
    """One-block launch of the metadata builder (``kv_blocked`` / ``sf_meta_t``: see the kernel; both appended, both default to
    the pre-existing chains' behaviour).  The tensor parameters are unannotated like the three before them: this entry is
    called both from a ``@cute.jit`` host (cute tensors) and directly from Python (torch tensors, converted at the boundary).

    Lives here rather than in the adapter because a `@cute.jit` defined inside a
    method closes over the kernel and its block width, and the DSL requires a
    code object with no free variables.
    """
    build_thd_bwd_setup_kernel(meta_t, q_lens_t, kv_lens_t, lens_form, n_qh, n_batch, ws_gran, cga_tile_m, n_clusters, kv_blocked, sf_meta_t).launch(
        grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream
    )


@cute.kernel
def build_thd_meta_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
) -> None:
    """Lengths -> ``[seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)]`` only.

    The metadata a kernel that takes ``cu_seqlens`` on device needs and
    nothing else: no blocked-workspace row offsets, no batch ranking, no claim
    counter. One warp builds the two prefixes for batches; B <= 1 retains
    the single-thread path. The SM80 backward reads ``cu_q`` / ``cu_k``
    straight out of this buffer.
    """
    tidx, _, _ = cute.arch.thread_idx()
    meta = cutlass.make_array_view(meta_t)
    if n_batch <= cutlass.Int32(1):
        if tidx == cutlass.Int32(0):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
    elif tidx < cutlass.Int32(32):
        write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, cutlass.Int32(tidx), store_lengths=False)
        write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, cutlass.Int32(tidx), store_lengths=True)


build_thd_meta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def thd_meta_host(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch, stream=None):
    """One-warp launch of :func:`build_thd_meta_kernel` (see ``thd_bwd_setup_host``
    for why the launch wrapper lives at module level)."""
    build_thd_meta_kernel(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


# --- the persistent claim counter --------------------------------------------------------------------------------------


@cute.jit
def thd_claim_unit(ctr_ptr) -> cutlass.Int32:
    """``uid = atomicAdd(ctr, 1)``: the next unit id off the device claim counter, for ONE lane.

    The THD main kernels' scheduler warp replaces the CLC try_cancel by this counter (``tile_dsl.thd.THD_CTR_OFF``, seeded at
    the cluster count by :func:`write_thd_live_and_ctr`): the cluster-first CTA's elected lane claims, compares with the live
    total and ships the unit to every CTA of the cluster by DSMEM stores.  A trace-time macro that emits ONE ``atom.add`` at the
    call site -- the CALLER elects (an un-gated call is 32 claims and 31 units nobody runs); ``ctr_ptr`` is the int32 pointer to
    the counter word (``Pointer(meta.iterator.raw_ptr(), dtype=Int32) + THD_CTR_OFF(B)``).  Shared so a body whose contract has no
    atomic of its own (the MXFP8 row produces no amax) spells the scheduler's one claim through the helper, not inline."""
    return cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))


# --- the MXFP8 row's per-sequence scale-factor pad staging -------------------------------------------------------------


@cute.kernel
def _pad_sf_atoms_thd(
    src: cute.Tensor,
    dst: cute.Tensor,
    meta_t: cute.Tensor,
    sf_meta_t: cute.Tensor,
    n_batch: cutlass.Int32,
    n_heads: cutlass.Int32,
    n_tiles: cutlass.Int32,
    kv_side: cutlass.Constexpr[bool],
    columnwise: cutlass.Constexpr[bool],
) -> None:
    """``dst`` = the PACKED F8_128x4 scale-factor slabs of ``src`` with every byte that scales a position PAST ITS SEQUENCE's
    length zeroed, and every tile past the live tile total zeroed -- the THD twin of the dense ``_pad_sf_atoms``
    (``sm107/prepared_host.py``), per sequence (sdpa-invariants s2).

    Layout (the forward's per-sequence-TILE-padded packed convention, one batch): ``[H, T_sf, THD_SF_SLAB_BYTES]``, sequence
    ``b`` owning tiles ``[cu_sf[b], cu_sf[b+1])`` of every head; a slab is the tile's ``D / 128`` atoms -- rowwise the 4-group
    d-chunks, columnwise the D-planes, both contiguous.  Byte ``(r % 32) * 16 + (r // 32) * 4 + c`` of an atom scales row ``r``
    (of 128), column ``c`` (of 4) (``tile_dsl.sf_layout``):
      rowwise  (scales along D; sf_v, sf_do): position ``s = tile_local * 128 + r`` -> pad iff ``s >= s_b``;
      columnwise (scales along S; sf_do_T, sf_q_T, sf_k_T): ``r`` is the d within the plane, the 32-token group is
        ``tile_local * 4 + c`` -> pad iff ``(tile_local * 4 + c) * 32 >= s_b``.
    A tile at or past ``cu_sf[B]`` (the capacity slack) belongs to no sequence: written as zeros (the clamped SF maps never read it).

    One block per (head, tile) slab, grid-strided; every thread resolves the slab's sequence once (a linear scan over the B
    prefixes, selects only -- B is small) and then its 4 bytes of the 1024-B slab (the 4 scale columns of one atom row, so the
    rowwise verdict is per word and the columnwise one per byte).  Reads ``meta_t`` for the lengths (``cu_q`` / ``cu_k``,
    ``kv_side``) and ``sf_meta_t`` for the tile prefixes; both were written by :func:`build_thd_bwd_setup_kernel` earlier on the
    stream (kernel-boundary ordering).

    Why a pre-pass and not an in-SMEM fix-up: the kernel's SF slabs are TMA-landed UTCCP sources read by the async proxy; a byte
    store from the 40-register service warps would need a new TMA-land -> store -> fence -> UTCCP handshake row in the barrier
    table.  Why it is needed at all: the producer's pad bytes are undefined and the main kernel READS them -- sf_v / sf_do pads
    reach dP (``0 x NaN`` -> dS NaN inside the last live tile), sf_do_T pads reach dV, and the block-scale GEMM's whole-atom SFB
    reads reach dK / dQ through sf_q_T / sf_k_T; sf_q / sf_k pads are harmless (S NaN -> the P select-zero).
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    blocks, _, _ = cute.arch.grid_dim()
    meta = cutlass.make_array_view(meta_t)
    sfm = cutlass.make_array_view(sf_meta_t)
    cu0 = (cutlass.Int32(2) * n_batch + cutlass.Int32(1)) if cutlass.const_expr(kv_side) else n_batch
    sf0 = cutlass.Int32(THD_SF_CU_K_OFF(n_batch)) if cutlass.const_expr(kv_side) else cutlass.Int32(THD_SF_CU_Q_OFF)
    src_ptr = src.iterator.raw_ptr()
    dst_ptr = dst.iterator.raw_ptr()
    total = cutlass.Int64(n_heads) * cutlass.Int64(n_tiles)
    slab = cutlass.Int64(bid)
    while slab < total:
        tile = cutlass.Int32(slab % cutlass.Int64(n_tiles))
        # The slab's sequence: tile in [cu_sf[b], cu_sf[b+1]).  A tile past the live total keeps the sentinel (s_b = 0: all pad).
        s_b = cutlass.Int32(0)
        tile_lo = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            lo = cutlass.Int32(sfm[sf0 + b])
            hi = cutlass.Int32(sfm[sf0 + b + cutlass.Int32(1)])
            hit = (tile >= lo) & (tile < hi)
            len_b = cutlass.Int32(meta[cu0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cu0 + b])
            s_b = cutlass.Int32(arith.select(hit.ir_value(), len_b.ir_value(), s_b.ir_value()))
            tile_lo = cutlass.Int32(arith.select(hit.ir_value(), lo.ir_value(), tile_lo.ir_value()))
        tile_local = tile - tile_lo
        slab_base = slab * cutlass.Int64(THD_SF_SLAB_BYTES)
        for c in cutlass.range_constexpr(SF_ATOM_COLS):
            off1 = cutlass.Int32(tid) * cutlass.Int32(SF_ATOM_COLS) + cutlass.Int32(c)  # byte inside the slab: word tid, column c
            off = off1 % cutlass.Int32(SF_ATOM_BYTES)  # byte inside its atom (atom = off1 // 512: the d-chunk or the D-plane)
            r = (off // cutlass.Int32(16)) + ((off % cutlass.Int32(16)) // cutlass.Int32(4)) * cutlass.Int32(32)
            if cutlass.const_expr(columnwise):
                keep = (tile_local * cutlass.Int32(SF_ATOM_COLS) + cutlass.Int32(c)) * cutlass.Int32(MX_BLOCK) < s_b
            else:
                keep = tile_local * cutlass.Int32(SF_ATOM_ROWS) + r < s_b
            i = slab_base + cutlass.Int64(off1)
            v = cutlass.Int8(0)  # raw byte pointers load / store Int8 (the type the DSL's raw_ptr carries); a pure byte copy
            if keep:
                v = (src_ptr + i).load()
            (dst_ptr + i).store(v)
        slab += cutlass.Int64(blocks)


_pad_sf_atoms_thd.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def pad_sf_atoms_thd_host(
    src,
    dst,
    meta_t,
    sf_meta_t,
    n_batch,
    n_heads,
    n_tiles,
    kv_side: cutlass.Constexpr[bool],
    columnwise: cutlass.Constexpr[bool],
    stream=None,
) -> None:
    """Launch :func:`_pad_sf_atoms_thd`: ``src`` the caller's packed SF bytes (any view; only the base address is read), ``dst``
    the staging copy of the SAME packed byte count (``n_heads * n_tiles * THD_SF_SLAB_BYTES``); ``n_tiles`` the bound buffer's
    PACKED tile count (the binder's, derived from its byte size -- never a capacity rounding), ``kv_side`` which length prefix
    the sequences' pads follow (K / V and their transposes: ``cu_k``; Q / dO: ``cu_q``).  Host ints or traced ``Int32`` alike;
    an empty side (0 tiles) launches one block that finds no slab.  Tensor parameters unannotated (cute tensors from a jit host,
    torch tensors from Python -- :func:`thd_bwd_setup_host`'s convention)."""
    n_slabs = cutlass.Int32(n_heads) * cutlass.Int32(n_tiles)
    grid_x = cute.math.max(cute.math.min(n_slabs, cutlass.Int32(4096)), cutlass.Int32(1))
    _pad_sf_atoms_thd(src, dst, meta_t, sf_meta_t, n_batch, n_heads, n_tiles, kv_side, columnwise).launch(
        grid=(grid_x, 1, 1), block=(_PAD_SF_THREADS, 1, 1), stream=stream
    )


# Ports whose caller buffers a backward may address at bound ragged offsets, in
# the order the origin spec and the offset pointers use.
ORIGIN_PORTS = ("q", "k", "v", "o", "do", "dq", "dk", "dv", "stats")


@cute.kernel
def build_thd_meta_origins_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    org_t: cute.Tensor,
    ro_q: Optional[cute.Tensor],
    ro_k: Optional[cute.Tensor],
    ro_v: Optional[cute.Tensor],
    ro_o: Optional[cute.Tensor],
    ro_do: Optional[cute.Tensor],
    ro_dq: Optional[cute.Tensor],
    ro_dk: Optional[cute.Tensor],
    ro_dv: Optional[cute.Tensor],
    ro_stats: Optional[cute.Tensor],
    origins: cutlass.Constexpr,
) -> None:
    """:func:`build_thd_meta_kernel` plus the caller-buffer token origins of the
    ports ``origins`` materializes (one ``(row, mult, ts)`` or ``None`` per
    :data:`ORIGIN_PORTS` entry).  Warp 0 builds the compact metadata exactly as
    :func:`build_thd_meta_kernel` does; every thread then writes origins, which
    do not depend on it."""
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    meta = cutlass.make_array_view(meta_t)
    if n_batch <= cutlass.Int32(1):
        if tidx == cutlass.Int32(0):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
    elif tidx < cutlass.Int32(32):
        write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, cutlass.Int32(tidx), store_lengths=False)
        write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, cutlass.Int32(tidx), store_lengths=True)
    ro = (ro_q, ro_k, ro_v, ro_o, ro_do, ro_dq, ro_dk, ro_dv, ro_stats)
    for i in cutlass.range_constexpr(len(ORIGIN_PORTS)):
        if cutlass.const_expr(origins[i] is not None):
            write_thd_port_origins(org_t, ro[i], origins[i][0], origins[i][1], origins[i][2], n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))


build_thd_meta_origins_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def thd_meta_origins_host(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch, org_t, ro, origins: cutlass.Constexpr, stream=None):
    """One-block launch of :func:`build_thd_meta_origins_kernel`; ``ro`` is the
    nine offset tensors (``None`` for a port without origins)."""
    build_thd_meta_origins_kernel(meta_t, q_lens_t, kv_lens_t, lens_form, n_batch, org_t, *ro, origins).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
