# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prep kernel of the SM100 / SM103 D128 MXFP8 backward chain: delta = rowsum(dO * O), the -LSE / -delta terms and
the LSE copy per row; it also zeroes the dQ unit counter and the requested amax outputs."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims
from cudnn.frost.tile_dsl.tma import st_global
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.common import (
    _add_ftz,
    _fma_ftz,
    _ldg_f32_if,
    _ldg_nc_v4,
    _mul_ftz,
    _select_f32,
    _select_u32,
)

HD = 128
LOG2E = 1.4426950408889634
_F32_NEG_INF = 0xFF800000  # -LSE term of the padded tail rows: P = 0 in dK/dV
_F32_POS_INF = 0x7F800000  # LSE copy of the padded tail rows: P = 0 in dQ


@cute.kernel
def prep_kernel(
    o: Int64,  # bf16, BHSD-compact (heads folded with the batch)
    dof: Int64,  # bf16 dO, same layout
    stats: Int64,  # fp32 LSE (natural log), [heads, S]
    ws: Int64,  # workspace base: delta | -LSE term | -delta term | LSE copy (rows each) | dQ unit counter
    amax_dq: Int64,
    amax_dk: Int64,
    amax_dv: Int64,
    g: cutlass.Constexpr,
):
    """8 warps x 4 rows per CTA, 16 lanes per row (16-byte loads), two rows per half-warp, every load issued before
    the first reduction.  Tail builds (S % 256 != 0): rows are the padded workspace rows (S_ALLOC per head); rows >= S
    get -LSE term = -inf (P = 0 in dK/dV), delta = -delta term = 0 and an LSE copy of +inf (P = 0 in dQ)."""
    rows, seq, s_alloc, ndelta_scale, tail = g.rows, g.seq, g.s_alloc, g.ndelta_scale, g.tail
    cute.arch.griddepcontrol_launch_dependents()  # dK/dV may start its prologue
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tidx = Int32(tidx)
    bidx = Int32(bidx)
    warp = tidx >> 5
    lane = tidx & 31
    half = lane >> 4
    sub = lane & 15
    row0 = bidx * 32 + warp * 4 + half  # rows row0 and row0 + 2
    o_base, do_base, st_base = o, dof, stats
    delta, nlse, ndelta, lse_ws = ws, ws + Int64(4 * rows), ws + Int64(8 * rows), ws + Int64(12 * rows)
    counter = ws + Int64(16 * rows)
    if (bidx | tidx) == 0:
        st_global(counter, Uint32(Uint32(0)), Uint32)
        if cutlass.const_expr(g.amax):
            st_global(amax_dq, Uint32(Uint32(0)), Uint32)  # max-reduced by dK/dV and dQ after their PDL waits
            st_global(amax_dk, Uint32(Uint32(0)), Uint32)
            st_global(amax_dv, Uint32(Uint32(0)), Uint32)

    lane_bytes = Int64(sub * 16)  # 8 bf16 per lane
    words = []
    sts = []
    oks = []
    for i in cutlass.range_constexpr(2):
        pr = row0 + 2 * i
        if cutlass.const_expr(tail):
            pr_u = Uint32(pr)  # unsigned: '//' on a signed value adds a floor correction
            h = pr_u // s_alloc
            s = pr_u - h * s_alloc
            ok = Uint32(s < seq)
            # A padded row (s >= S) reads row S - 1 of its head instead: its outputs are the constants selected at the
            # stores, so no load needs a predicate (predicated loads schedule worse: +5 % on this bandwidth-bound kernel).
            idx = Int64(h) * seq + Int64(_select_u32(ok, s, Uint32(seq - 1)))
            a = _ldg_nc_v4(o_base + idx * (HD * 2) + lane_bytes)
            b = _ldg_nc_v4(do_base + idx * (HD * 2) + lane_bytes)
        else:
            ok = Uint32(1)
            idx = Int64(pr)
            a = _ldg_nc_v4(o_base + idx * (HD * 2) + lane_bytes)
            b = _ldg_nc_v4(do_base + idx * (HD * 2) + lane_bytes)
        words.append((a, b))
        oks.append((ok, idx))
    for i in cutlass.range_constexpr(2):  # the four 16-byte loads first, then the stats
        ok, idx = oks[i]
        sts.append(_ldg_f32_if(Uint32(sub == 0), st_base + idx * 4))  # one lane per row reads the stats
        oks[i] = ok
    for i in cutlass.range_constexpr(2):
        a, b = words[i]
        acc = Float32(0.0)
        for u in cutlass.range_constexpr(4):
            acc = _fma_ftz((Uint32(a[u]) << Uint32(16)).bitcast(Float32), (Uint32(b[u]) << Uint32(16)).bitcast(Float32), acc)
            acc = _fma_ftz((Uint32(a[u]) & Uint32(0xFFFF0000)).bitcast(Float32), (Uint32(b[u]) & Uint32(0xFFFF0000)).bitcast(Float32), acc)
        for k in cutlass.range_constexpr(4):
            acc = _add_ftz(acc, prims.shfl_sync(thread_mask=0xFFFFFFFF, val=acc, offset=8 >> k, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY))
        if sub == 0:
            row = Int64(row0 + 2 * i) * 4
            nl = _fma_ftz(sts[i], Float32(-LOG2E), Float32(8.0))
            nd = _mul_ftz(acc, Float32(ndelta_scale))
            if cutlass.const_expr(tail):
                ok = oks[i]
                st_global(delta + row, Float32(_select_f32(ok, acc, Float32(0.0))), Float32)
                st_global(nlse + row, Uint32(_select_u32(ok, Float32(nl).bitcast(Uint32), Uint32(_F32_NEG_INF))), Uint32)
                st_global(ndelta + row, Float32(_select_f32(ok, nd, Float32(0.0))), Float32)
                st_global(lse_ws + row, Uint32(_select_u32(ok, Float32(sts[i]).bitcast(Uint32), Uint32(_F32_POS_INF))), Uint32)
            else:
                st_global(delta + row, Float32(acc), Float32)
                st_global(nlse + row, Float32(nl), Float32)
                st_global(ndelta + row, Float32(nd), Float32)


prep_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
