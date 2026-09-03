# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver

from cudnn.frost.tile_dsl.swizzle import swizzle_xor

from cudnn.frost.tile_dsl.thd import (
    TENSOR_MAP_QWORDS,
    THD_CTR_OFF,
    THD_LIVE_OFF,
    THD_META_WORDS,
    THD_REMAP_OFF,
    THD_SETUP_THREADS,
    emit_clamped_desc,
    emit_seq_descs,
    thd_claim_next,
    thd_decode_unit,
    write_thd_batch_remap,
    write_thd_live_and_ctr,
    write_thd_meta,
)

# The THD metadata layout, the batch ranking, the unit decode, the claim counter
# and the per-sequence descriptor patchers moved to ``frost/tile_dsl/thd.py``
# when the SM100 backward needed the same pieces -- a second copy of a metadata
# LAYOUT is a silent-wrong-answer waiting to happen, since a reader offset and a
# writer offset that disagree by one word decode garbage batches rather than
# failing.  Re-exported here so the kernels importing them from this module keep
# working.  What stays below is genuinely forward-specific: the SM120 sV tail
# sanitizer (its SMEM order and swizzle are that kernel's), and the per-flavor
# setup LAUNCHES.
__all__ = [
    "TENSOR_MAP_QWORDS",
    "THD_CTR_OFF",
    "THD_LIVE_OFF",
    "THD_META_WORDS",
    "THD_REMAP_OFF",
    "THD_SETUP_THREADS",
    "build_thd_meta_kernel",
    "build_thd_meta_o_descs_kernel",
    "build_thd_meta_o_kv_descs_kernel",
    "sanitize_v_tail",
    "thd_claim_next",
    "thd_decode_unit",
    "write_thd_batch_remap",
    "write_thd_live_and_ctr",
    "write_thd_meta",
]


@cute.jit
def sanitize_v_tail(
    sV,
    lane: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    kv_seq_idx: cutlass.Int32,
    in_dtype,
    head_tile_v,
    kv_tile,
    v_swizzle_chunk_elems,
) -> None:
    """Zero ``sV`` rows at/past this tile's valid KV extent before P @ V.

    Shared by the SM120 f16 and FP8 flavors (their sV tiles use the same
    ``[I][kv_tile][C]`` TMA order and per-row swizzle; the element size rides
    ``in_dtype``). The K/V TMA descriptors span the bound buffers' CAPACITY
    (under THD the packed totals are device values, so the views bind
    whole-buffer extents), so rows between the valid KV length and the tile
    end can carry UNINITIALIZED storage — including NaN bit patterns. The
    S-side mask overwrites those columns with -inf (a select, NaN-safe),
    but P @ V still multiplies P = 0 against the NaN V row and
    ``0 * NaN = NaN`` poisons the whole accumulator column-free. Reached
    only from THD specializations' first masked step (see the call
    sites): dense descriptors carry the declared S_kv, so their overhang
    loads zero-fill in hardware — a dense PADDED graph's pad rows are
    user memory and deliberately NOT sanitized here (whether the
    contract requires tolerating NaN bit patterns there is an open
    question for the sibling kernels too).

    Every compute warp redundantly zeroes the full overhang (idempotent
    zero stores race benignly), so a warp-level sync is enough for each
    warp's own ``ldmatrix`` lanes to observe the zeros.
    """
    seg_elems = 16 // in_dtype.bytes  # elements per 16-byte store
    segs_per_row = head_tile_v // seg_elems
    row_lo = cute.math.max(cutlass.Int32(0), seqlen_k - kv_seq_idx)
    for r_it in cutlass.range_constexpr(kv_tile // 32):
        row = cutlass.Int32(r_it * 32) + lane
        for seg in cutlass.range_constexpr(segs_per_row):
            col = seg * seg_elems
            phys_row = (col // v_swizzle_chunk_elems) * kv_tile + row
            sv_ptr = (
                sV.data_ptr()
                + phys_row * v_swizzle_chunk_elems
                + swizzle_xor(
                    phys_row,
                    col % v_swizzle_chunk_elems,
                    v_swizzle_chunk_elems,
                    in_dtype.bytes,
                )
            )
            zero16 = cutlass.Vector.from_elements(
                tuple(in_dtype(0.0) for _ in range(seg_elems)),
                in_dtype,
            )
            if row >= row_lo:
                sv_ptr.store(zero16, alignment=16)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.kernel
def build_thd_meta_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    n_qh: cutlass.Int32,
    q_tile: cutlass.Int32,
    n_ctas: cutlass.Int32,
) -> None:
    """Meta-only THD setup (SM120: no per-batch O TMA descriptors — O stores
    are raw pointer writes predicated per row). The metadata write is one
    elected thread; the batch remap and the live-unit total that follow are
    whole-block. The main kernel launched after it on the same stream sees the
    writes by kernel boundary ordering."""
    meta = cutlass.make_array_view(meta_t)
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    # elect_sync elects one thread PER WARP, and this block is THD_SETUP_THREADS
    # wide so the ranking below can run in parallel — narrow the single-thread
    # body to warp 0's leader. Every warp still evaluates elect_sync (it is warp
    # -uniform); only the added predicate is what excludes warps 1..N.
    if nvvm.elect_sync() and tidx < cutlass.Int32(32):
        write_thd_meta(
            meta,
            cutlass.make_array_view(q_lens_t),
            cutlass.make_array_view(kv_lens_t),
            lens_form,
            n_batch,
        )
    # Barrier first: the ranking reads the cu_seqlens_q written above.
    cute.arch.barrier()
    write_thd_batch_remap(meta, n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    # Live unit total + claim counter, as on SM100 — a SM120 unit is q_tile
    # rows of one head, so the same count applies with cga_tile_m := q_tile.
    cute.arch.barrier()
    write_thd_live_and_ctr(meta, n_batch, n_qh, q_tile, n_ctas, cutlass.Int32(tidx))


build_thd_meta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def build_thd_meta_o_kv_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    o_row_stride: cutlass.Int32,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
    k_seq_dim: cutlass.Constexpr[int] = 2,
    v_seq_dim: cutlass.Constexpr[int] = 2,
) -> None:
    """THD setup for the FP8/MXFP8 flavors: metadata, per-batch O descriptors
    and the packed-total-clamped K/V descriptors.

    Like ``build_thd_meta_o_descs_kernel``, this publishes the persistent
    scheduler's live-unit total and claim counter. Both kernels also clamp K/V
    (issue #624).

    The K/V loads tile in TILE_N rows, so the LAST sequence's tile steps past
    the packed KV total into the buffer's capacity tail — caller-owned bytes
    that may be NaN (test_mhas_v2 poisons them deliberately). Masked S columns
    are NaN-safe (the mask is a select), but BMM2's ``P·V`` is not
    (``0 · NaN == NaN`` wipes every valid row of the tile), and on cc10.3 the
    fused-LDTM row-max reduces S BEFORE the mask. So the setup thread also
    copies the K and V base descriptors into ``o_desc_words`` slots
    ``n_batch+1`` / ``n_batch+2`` with each descriptor's sequence extent
    patched to the packed total ``cu_k[B]`` — tile-tail loads past it become
    TMA-OOB and land as EXACT ZEROS in SMEM (zero V nulls the masked P·V
    terms; zero K keeps the pre-mask row-max finite). Slot ``n_batch`` stays
    the never-built dead-unit pad slot."""
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    # Warp 0's leader only. This flavor launches one warp, so elect_sync alone
    # would do; the predicate keeps all three setup kernels safe under any block
    # width, since elect_sync elects one thread PER WARP.
    if nvvm.elect_sync() and tidx < cutlass.Int32(32):
        meta = cutlass.make_array_view(meta_t)
        write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
        # Per-batch O descriptors, from the cu_q values written above (same
        # thread -- plain program order, no fence needed for the meta reads).
        emit_seq_descs(base_o_desc, o_desc_words, meta, n_batch, o_tensor, n_batch, o_row_stride, seq_ord=2)
        t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])  # cu_k[B]
        emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=k_seq_dim)
        emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=v_seq_dim)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )
    # Outside the elect: every thread helps rank the batches. The barrier makes
    # the cu_seqlens_q written above visible to the whole block first. The
    # decode walks this permutation on every THD flavor, so a setup that skips
    # it leaves the region uninitialized and units decode garbage batches.
    cute.arch.barrier()
    write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    # Live unit total + claim counter for the persistent scheduler, exactly as
    # build_thd_meta_o_descs_kernel writes them. The scheduler reads meta[4B+2]
    # as its bound, so leaving these two words unwritten hands out units off
    # uninitialized workspace (an illegal-instruction fault, not a silent
    # wrong answer). The counter starts at n_clusters: cluster c takes unit c
    # from its blockIdx, then pulls from the counter.
    cute.arch.barrier()
    write_thd_live_and_ctr(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx))


build_thd_meta_o_kv_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def build_thd_meta_o_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    o_row_stride: cutlass.Int32,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
) -> None:
    """Per-execute THD setup for the f16/bf16 flavors, one elected thread —
    ``build_thd_meta_o_kv_descs_kernel`` plus the persistent scheduler's
    live-unit total and claim counter (issue #552, D2H removal):
    build the [seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)] metadata
    buffer DEVICE-side from the caller's length tensors — ``(B,)`` per-batch
    lengths (serial cumsum; B is small) or the ``(B+1,)`` cu prefix-sum form
    (NORMALIZED by subtracting element 0 — the packed buffers are addressed
    from token 0, so a cu tensor sliced from a larger prefix means the same
    lengths, and the host can no longer validate ``cu[0] == 0`` (Rule 3), so
    an unnormalized base must not leak into the offsets the tiles and the
    dead-unit sentinel read; per-batch KV lengths are adjacent diffs either
    way), per side via
    ``lens_form`` (bit 0: Q is cu, bit 1: KV is cu) — then build the per-batch
    O TMA descriptors from the cu_q values just written (same thread, program
    order). Replaces the host tolist → cumsum → H2D round-trip with work
    inside the setup launch that already existed for the descriptors."""
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    # elect_sync elects one thread PER WARP, and this block is THD_SETUP_THREADS
    # wide so the ranking below can run in parallel — narrow the single-thread
    # body to warp 0's leader. Without this, one warp's descriptor base-copy can
    # land after another's tensormap_replace and revert the patched address.
    if nvvm.elect_sync() and tidx < cutlass.Int32(32):
        meta = cutlass.make_array_view(meta_t)
        write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
        # Per-batch O descriptors, from the cu_q values written above (same
        # thread -- plain program order, no fence needed for the meta reads).
        emit_seq_descs(base_o_desc, o_desc_words, meta, n_batch, o_tensor, n_batch, o_row_stride, seq_ord=2)
        # Packed-total-clamped K/V runtime descriptors (issue #624). K/V load
        # in TILE_N rows, so the LAST sequence's tile steps past the packed KV
        # total into the buffer's capacity tail — caller-owned bytes that may
        # never have been written. Masked S columns are NaN-safe (the mask is
        # a select), but BMM2's P·V is not: 0 · NaN == NaN wipes every valid
        # row of the tile. Patching the seq extent (GLOBAL_DIM ord=2) to
        # cu_k[B] makes those rows TMA-OOB, so they land as EXACT ZEROS
        # without touching memory — no fill kernel, and nothing written into
        # the caller's buffer. Mirrors build_thd_meta_o_kv_descs_kernel, which
        # the FP8/MXFP8 flavors have used for this since they were written.
        t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])  # cu_k[B]
        emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=2)
        emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )
    # Outside the elect: every thread helps rank the batches. The barrier makes
    # the cu_seqlens_q written above visible to the whole block first.
    cute.arch.barrier()
    write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    # Live unit total + claim counter for the persistent scheduler. The host
    # cannot know Sigma_b ceil(s_b/tile)*QH without a D2H (issue #552), so the
    # kernel reads its own bound from here. The counter starts at n_clusters:
    # cluster c takes unit c from its blockIdx, then pulls from the counter.
    cute.arch.barrier()
    write_thd_live_and_ctr(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx))


build_thd_meta_o_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
