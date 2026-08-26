# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
import cuda.bindings.driver as _cuda_driver

TENSOR_MAP_QWORDS = 128 // 8

# THD metadata layout, in int32 words:
#   [ seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1) | batch_remap(B) ]
# The trailing batch_remap is a permutation of [0, B) ordered by DESCENDING
# Q length, so the tile scheduler walks the longest sequences first (longest
# processing time): the tail of a THD launch is then made of short sequences,
# which is what bounds the ragged last wave.
THD_META_WORDS = lambda b: 4 * b + 4  # noqa: E731
THD_REMAP_OFF = lambda b: 3 * b + 2  # noqa: E731
THD_LIVE_OFF = lambda b: 4 * b + 2  # noqa: E731   live unit total (device-computed)
THD_CTR_OFF = lambda b: 4 * b + 3  # noqa: E731    persistent-scheduler claim counter

# Threads for the THD setup launch. The metadata write itself is one elected
# thread; the batch-remap ranking that follows is parallel over batches, so the
# block is sized for that (B > THD_SETUP_THREADS just loops).
THD_SETUP_THREADS = 256


@cute.jit
def write_thd_batch_remap(meta, n_batch: cutlass.Int32, tid: cutlass.Int32, nthreads: cutlass.Int32) -> None:
    """Fill batch_remap with [0, B) sorted by descending Q length.

    Rank-by-counting rather than a sort network: each thread owns a batch and
    counts how many sequences outrank it, which is O(B^2) comparisons but fully
    parallel, branch-free and trivially deterministic. Ties break on the
    original index, so the permutation is stable and reproducible run to run.

    Must be called AFTER write_thd_meta (it reads the cu_seqlens_q it wrote)
    with a barrier in between.
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
def write_thd_meta(meta, ql, kl, lens_form: cutlass.Int32, n_batch: cutlass.Int32) -> None:
    """Single-thread body of the device-side THD metadata build (issue #552),
    shared by the SM100 setup kernel (which follows it with the per-batch O
    TMA descriptors) and the SM120 meta-only kernel. Writes the
    [seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)] buffer from the
    caller's length tensors — ``(B,)`` per-batch lengths (serial cumsum; B
    is small) or the ``(B+1,)`` cu prefix-sum form, per side via
    ``lens_form`` (bit 0: Q is cu, bit 1: KV is cu). cu prefixes are
    NORMALIZED (element 0 subtracted): the packed buffers are addressed from
    token 0, so a cu tensor sliced from a larger prefix means the same
    lengths — and the host can no longer validate ``cu[0] == 0`` (Rule 3),
    so an unnormalized base must not leak into the offsets the tiles and
    the dead-unit sentinel read. Callers run this under ``elect_sync``."""
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
def thd_decode_unit(
    meta,
    n_batch: cutlass.Int32,
    uid: cutlass.Int32,
    n_qh: cutlass.Int32,
    q_tile: cutlass.Int32,
    reverse_rows: bool,
) -> tuple:
    """Map a linear unit id to ``(q_tile_idx, batch, head)`` through batch_remap.

    A unit is ``q_tile`` rows of one head of one sequence. Sequences are walked
    LONGEST FIRST (the remap), and the head is the major axis within a sequence
    so consecutive units sweep the Q tiles of a single head — those share a K/V
    head, which is what keeps the claim order L2-friendly. ``reverse_rows``
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

    One atomic for the whole CTA, broadcast through a single SMEM word. The
    leading barrier also separates the previous unit's use of the shared K/V
    staging from the next unit's, so the caller does not need its own.
    """
    ctr_ptr = Pointer(meta_t.iterator.raw_ptr(), dtype=cutlass.Int32) + ctr_off
    nvvm.barrier_cta_sync(0)
    if tidx == cutlass.Int32(0):
        slot[0] = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
    nvvm.barrier_cta_sync(0)
    return cutlass.Int32(slot[0])


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
    # Thread 0 alone, WITHOUT elect_sync: elect.sync picks an
    # implementation-defined lane, so conjoining it with tidx == 0 can
    # select no thread at all and leave these words unwritten.
    if tidx == cutlass.Int32(0):
        live = cutlass.Int32(0)
        cuq0 = n_batch
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            s_b = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
            live = live + ((s_b + q_tile - cutlass.Int32(1)) // q_tile) * n_qh
        meta[cutlass.Int32(4) * n_batch + cutlass.Int32(2)] = live
        meta[cutlass.Int32(4) * n_batch + cutlass.Int32(3)] = n_ctas


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
) -> None:
    """``build_thd_meta_o_descs_kernel`` + packed-total-clamped K/V descriptors
    (the FP8/MXFP8 THD flavors).

    The K/V loads tile in TILE_N rows, so the LAST sequence's tile steps past
    the packed KV total into the buffer's capacity tail — caller-owned bytes
    that may be NaN (test_mhas_v2 poisons them deliberately). Masked S columns
    are NaN-safe (the mask is a select), but BMM2's ``P·V`` is not
    (``0 · NaN == NaN`` wipes every valid row of the tile), and on cc10.3 the
    fused-LDTM row-max reduces S BEFORE the mask. So the setup thread also
    copies the K and V base descriptors into ``o_desc_words`` slots
    ``n_batch+1`` / ``n_batch+2`` with their seq extent (GLOBAL_DIM ord=2)
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
        cuq0 = n_batch
        o_ptr = o_tensor.iterator.raw_ptr()
        desc_base = o_desc_words.iterator.raw_ptr()
        src_words = Pointer(base_o_desc.get_ptr(), dtype=cutlass.Int64)
        row_elems = o_row_stride
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
            for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                (dptr + i).store((src_words + i).load())
            cu_q_b = cutlass.Int32(meta[cuq0 + b])
            s_i = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cu_q_b
            row_base = o_ptr + cu_q_b * row_elems
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_ADDRESS,
                dptr,
                new_value=row_base.toint(cutlass.Int64),
            )
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_DIM,
                dptr,
                new_value=s_i,
                ord=2,
            )
        t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])  # cu_k[B]
        k_dptr = desc_base + (n_batch + cutlass.Int32(1)) * cutlass.Int32(TENSOR_MAP_QWORDS)
        k_src = Pointer(base_k_desc.get_ptr(), dtype=cutlass.Int64)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (k_dptr + i).store((k_src + i).load())
        nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, k_dptr, new_value=t_kv, ord=2)
        v_dptr = desc_base + (n_batch + cutlass.Int32(2)) * cutlass.Int32(TENSOR_MAP_QWORDS)
        v_src = Pointer(base_v_desc.get_ptr(), dtype=cutlass.Int64)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (v_dptr + i).store((v_src + i).load())
        nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, v_dptr, new_value=t_kv, ord=2)
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


@cute.kernel
def build_thd_meta_o_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
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
    """Per-execute THD setup, one elected thread (issue #552, D2H removal):
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
        cuq0 = n_batch
        # Per-batch O descriptors, from the cu_q values written above (same
        # thread — plain program order, no fence needed for the meta reads).
        o_ptr = o_tensor.iterator.raw_ptr()
        desc_base = o_desc_words.iterator.raw_ptr()
        src_words = Pointer(base_o_desc.get_ptr(), dtype=cutlass.Int64)
        row_elems = o_row_stride
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
            for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                (dptr + i).store((src_words + i).load())
            cu_q_b = cutlass.Int32(meta[cuq0 + b])
            s_i = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cu_q_b
            row_base = o_ptr + cu_q_b * row_elems
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_ADDRESS,
                dptr,
                new_value=row_base.toint(cutlass.Int64),
            )
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_DIM,
                dptr,
                new_value=s_i,
                ord=2,
            )
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
    # Thread 0 alone, WITHOUT elect_sync: elect.sync picks an
    # implementation-defined lane, so conjoining it with tidx == 0 can
    # select no thread at all and leave these words unwritten.
    if tidx == cutlass.Int32(0):
        meta_w = cutlass.make_array_view(meta_t)
        cuq0 = n_batch
        live = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            s_b = cutlass.Int32(meta_w[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta_w[cuq0 + b])
            live = live + ((s_b + cga_tile_m - cutlass.Int32(1)) // cga_tile_m) * n_qh
        meta_w[cutlass.Int32(4) * n_batch + cutlass.Int32(2)] = live
        meta_w[cutlass.Int32(4) * n_batch + cutlass.Int32(3)] = n_clusters
