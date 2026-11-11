# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aligned bitmap compaction with sparse bit scans and vector prefetch."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from cuda.bindings import driver as cuda_driver

I32 = cutlass.Int32
I64 = cutlass.Int64
U32 = cutlass.Uint32
WARPS = 4


@cute.kernel
def frost_qsa_fast_compact(
    g_sel: cute.Tensor,
    g_lo: cute.Tensor,
    g_hi: cute.Tensor,
    g_first: cute.Tensor,
    g_last: cute.Tensor,
    g_ids: cute.Tensor,
    g_counts: cute.Tensor,
    T: I32,
    NB: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    row = bid * WARPS + warp
    if row < T:
        p_sel = g_sel.iterator.raw_ptr()
        p_lo = g_lo.iterator.raw_ptr()
        p_hi = g_hi.iterator.raw_ptr()
        p_first = g_first.iterator.raw_ptr()
        p_last = g_last.iterator.raw_ptr()
        p_ids = g_ids.iterator.raw_ptr()
        p_counts = g_counts.iterator.raw_ptr()
        low = (p_lo + row).load()
        high = (p_hi + row).load()
        count = I32(0)
        if low <= high:
            nvec = NB // 16
            words = cute.make_ptr(
                U32,
                I64(p_sel.toint()),
                cutlass.AddressSpace.gmem,
                assumed_align=16,
            )
            wrow = words + I64(row) * I64(NB // 4)
            irow = p_ids + I64(row) * I64(NB)
            # Software prefetch: keep the next 16 B vector in flight while this one is
            # processed (index clamped in range; unused lanes' data is ignored).
            v0 = lane
            if v0 >= nvec:
                v0 = I32(0)
            w_next = prims.load_ext(wrow + I64(v0) * 4, count=4)
            for cstart in cutlass.range(0, nvec, 32, unroll=1):
                v = cstart + lane
                w4 = w_next
                vn = v + 32
                if vn >= nvec:
                    vn = I32(0)
                w_next = prims.load_ext(wrow + I64(vn) * 4, count=4)
                valid = I32(0)
                if v < nvec:
                    mask16 = U32(0)
                    for k in cutlass.range_constexpr(4):
                        w = w4[k]
                        t = ((w & U32(0x7F7F7F7F)) + U32(0x7F7F7F7F)) | w
                        x = (t & U32(0x80808080)) >> U32(7)
                        m = (x & U32(1)) | ((x >> U32(7)) & U32(2)) | ((x >> U32(14)) & U32(4)) | ((x >> U32(21)) & U32(8))
                        mask16 = mask16 | (m << U32(4 * k))
                    # Visit only the set bits (about 1.6% of blocks at T=131072).
                    pending = mask16
                    for _bit in cutlass.range(cute.arch.popc(mask16), unroll=1):
                        lowbit = pending & (U32(0) - pending)
                        i = I32(cute.arch.popc(lowbit - U32(1)))
                        blk = v * 16 + i
                        if ((p_last + blk).load() >= low) & ((p_first + blk).load() <= high):
                            valid = valid | (I32(1) << i)
                        pending = pending ^ lowbit
                n = cute.arch.popc(valid)
                incl = I32(n)
                for d in cutlass.range_constexpr(5):
                    y = prims.shfl_sync(0xFFFFFFFF, incl, 1 << d, 0, prims.Shfl.UP)
                    if lane >= (1 << d):
                        incl = incl + y
                total = prims.shfl_sync(0xFFFFFFFF, incl, 31, 0x1F, prims.Shfl.IDX)
                run = count + (incl - I32(n))
                pending_valid = valid
                for _bit in cutlass.range(n, unroll=1):
                    lowbit = pending_valid & (I32(0) - pending_valid)
                    i = I32(cute.arch.popc(lowbit - I32(1)))
                    (irow + I64(run)).store(v * 16 + i)
                    run = run + 1
                    pending_valid = pending_valid ^ lowbit
                count = count + total
        if lane == 0:
            (p_counts + row).store(count)


frost_qsa_fast_compact.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def frost_qsa_fast_compact_host(
    sel: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    total = sel.shape[0]
    blocks = sel.shape[1]
    frost_qsa_fast_compact(sel, lo, hi, first, last, ids, counts, I32(total), I32(blocks)).launch(
        grid=(cute.ceil_div(total, WARPS), 1, 1),
        block=(WARPS * 32, 1, 1),
        stream=stream,
    )
