# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm backward, sm_100, CUTLASS primitives.

Given ``dy`` and the forward's saved ``x``, ``gamma``, ``mean``, ``rstd`` this
produces ``dx`` plus ``dgamma``/``dbeta`` (reduced across every group sharing a
channel, via fp32 atomics). Per group ``r`` with ``xhat=(x-mean)*rstd`` and
``dxhat=dy*gamma``::

    dgamma_c += sum_j dy*xhat ;  dbeta_c += sum_j dy      (atomics)
    a = mean_j(dxhat)   (0 for RMSNorm) ;  b = mean_j(dxhat*xhat)
    dx = rstd * (dxhat - a - xhat*b)

X and DY are staged into smem once (``STAGE_BULK`` = two ``cp.async.bulk`` on one
mbarrier; ``STAGE_CPASYNC`` = per-thread cp.async) when they fit; otherwise read
from global (``STAGE_NONE``). The DX store (and, unstaged, the X/DY loads) are
128-bit vectorized when ``M % V == 0``. Also serves RMSNorm (``has_mean=False``).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import ws_bytes, ws_empty, ws_zeros, cur_stream, dyn, fake_stream, smem_capacity
from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm._common_sm100 import (
    affine_gamma,
    STAGE_BULK,
    STAGE_NONE,
    block_reduce_sum2,
    red_scratch_len,
    stage_row,
    stage_two_bulk,
)

_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_CTA_SS = nvvm.SharedSpace.shared_cta


@cute.kernel
def _ln_bwd_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mGamma: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDX: cute.Tensor,
    mDGamma: cute.Tensor,
    mDBeta: cute.Tensor,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr,
    stage_mode: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    affine: cutlass.Constexpr = None,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    per_row: cutlass.Constexpr = affine is not None and affine[3]
    gamma_step: cutlass.Constexpr = 0 if per_row else 1
    gamma_base = cutlass.Int64(0)
    if cutlass.const_expr(per_row):
        gamma_base = cutlass.Int64(r)
    elif cutlass.const_expr(affine is not None):
        gamma_base = cutlass.Int64((r // affine[1]) % affine[0]) * M
    staged: cutlass.Constexpr = stage_mode != STAGE_NONE
    nv: cutlass.Constexpr = M // V

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(red_scratch_len(bt)), byte_alignment=8)
    sX = None
    sDY = None
    if cutlass.const_expr(staged):
        if cutlass.const_expr(stage_mode == STAGE_BULK):
            mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
            sX = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)
            sDY = smem.allocate_tensor(mDY.element_type, cute.make_layout(M), byte_alignment=16)
            stage_two_bulk(sX, sDY, mbar, mX[r, None], mDY[r, None], M * elem_bytes, tid)
        else:
            sX = smem.allocate_tensor(mX.element_type, cute.make_layout(M), byte_alignment=16)
            sDY = smem.allocate_tensor(mDY.element_type, cute.make_layout(M), byte_alignment=16)
            stage_row(sX, mX[r, None], M, tid, V, bt, elem_bytes)
            stage_row(sDY, mDY[r, None], M, tid, V, bt, elem_bytes)

    Mf = cutlass.Float32(M)
    mean = cutlass.Float32(0.0)
    if cutlass.const_expr(has_mean):
        mean = mMean[r]
    rstd = mRstd[r]

    # --- pass 1: reductions + dgamma/dbeta atomics ---
    s_dxhat = cutlass.Float32(0.0)
    s_dxhat_xhat = cutlass.Float32(0.0)
    if cutlass.const_expr(staged):
        j = tid
        while j < M:
            x = sX[j].to(cutlass.Float32)
            dy = sDY[j].to(cutlass.Float32)
            g = affine_gamma(mGamma[gamma_base + j * gamma_step], affine)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            if cutlass.const_expr(per_row):
                s_dxhat = s_dxhat + dy
                s_dxhat_xhat = s_dxhat_xhat + dy * xhat
            else:
                s_dxhat = s_dxhat + dxhat
                s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
                cute.arch.atomic_add(mDGamma.iterator + gamma_base + j * gamma_step, dy * xhat)
                if cutlass.const_expr(has_beta):
                    cute.arch.atomic_add(mDBeta.iterator + gamma_base + j * gamma_step, dy)
            j = j + bt
    elif cutlass.const_expr(vec):
        xv = cute.zipped_divide(mX[r, None], (V,))
        dyv = cute.zipped_divide(mDY[r, None], (V,))
        vi = tid
        while vi < nv:
            j0 = vi * V
            xf = cute.make_fragment_like(xv[None, vi])
            dyf = cute.make_fragment_like(dyv[None, vi])
            cute.autovec_copy(xv[None, vi], xf)
            cute.autovec_copy(dyv[None, vi], dyf)
            for e in cutlass.range_constexpr(V):
                x = xf[e].to(cutlass.Float32)
                dy = dyf[e].to(cutlass.Float32)
                g = affine_gamma(mGamma[gamma_base + (j0 + e) * gamma_step], affine)
                xhat = (x - mean) * rstd
                dxhat = dy * g
                if cutlass.const_expr(per_row):
                    s_dxhat = s_dxhat + dy
                    s_dxhat_xhat = s_dxhat_xhat + dy * xhat
                else:
                    s_dxhat = s_dxhat + dxhat
                    s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
                    cute.arch.atomic_add(mDGamma.iterator + gamma_base + (j0 + e) * gamma_step, dy * xhat)
                    if cutlass.const_expr(has_beta):
                        cute.arch.atomic_add(mDBeta.iterator + gamma_base + (j0 + e) * gamma_step, dy)
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = mX[r, j].to(cutlass.Float32)
            dy = mDY[r, j].to(cutlass.Float32)
            g = affine_gamma(mGamma[gamma_base + j * gamma_step], affine)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            if cutlass.const_expr(per_row):
                s_dxhat = s_dxhat + dy
                s_dxhat_xhat = s_dxhat_xhat + dy * xhat
            else:
                s_dxhat = s_dxhat + dxhat
                s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
                cute.arch.atomic_add(mDGamma.iterator + gamma_base + j * gamma_step, dy * xhat)
                if cutlass.const_expr(has_beta):
                    cute.arch.atomic_add(mDBeta.iterator + gamma_base + j * gamma_step, dy)
            j = j + bt

    s_dxhat, s_dxhat_xhat = block_reduce_sum2(s_dxhat, s_dxhat_xhat, tid, red, bt)
    if cutlass.const_expr(per_row):
        # One CTA owns this sample/channel, so write its spatial reductions
        # directly. Multiplying afterward also handles zero effective gamma.
        if tid == 0:
            mDGamma[gamma_base] = s_dxhat_xhat
            if cutlass.const_expr(has_beta):
                mDBeta[gamma_base] = s_dxhat
        row_gamma = affine_gamma(mGamma[gamma_base], affine)
        s_dxhat = s_dxhat * row_gamma
        s_dxhat_xhat = s_dxhat_xhat * row_gamma
    a = cutlass.Float32(0.0)
    if cutlass.const_expr(has_mean):
        a = s_dxhat / Mf
    b = s_dxhat_xhat / Mf

    # --- pass 2: dx ---
    if cutlass.const_expr(vec):
        dxv = cute.zipped_divide(mDX[r, None], (V,))
        xvg = cute.zipped_divide(mX[r, None], (V,)) if cutlass.const_expr(not staged) else None
        dyvg = cute.zipped_divide(mDY[r, None], (V,)) if cutlass.const_expr(not staged) else None
        vi = tid
        while vi < nv:
            j0 = vi * V
            dxf = cute.make_fragment_like(dxv[None, vi])
            if cutlass.const_expr(not staged):
                xf = cute.make_fragment_like(xvg[None, vi])
                dyf = cute.make_fragment_like(dyvg[None, vi])
                cute.autovec_copy(xvg[None, vi], xf)
                cute.autovec_copy(dyvg[None, vi], dyf)
            for e in cutlass.range_constexpr(V):
                x = (sX[j0 + e] if cutlass.const_expr(staged) else xf[e]).to(cutlass.Float32)
                dy = (sDY[j0 + e] if cutlass.const_expr(staged) else dyf[e]).to(cutlass.Float32)
                g = affine_gamma(mGamma[gamma_base + (j0 + e) * gamma_step], affine)
                xhat = (x - mean) * rstd
                dxhat = dy * g
                dxf[e] = (rstd * (dxhat - a - xhat * b)).to(mDX.element_type)
            cute.autovec_copy(dxf, dxv[None, vi])
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = (sX[j] if cutlass.const_expr(staged) else mX[r, j]).to(cutlass.Float32)
            dy = (sDY[j] if cutlass.const_expr(staged) else mDY[r, j]).to(cutlass.Float32)
            g = affine_gamma(mGamma[gamma_base + j * gamma_step], affine)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            mDX[r, j] = (rstd * (dxhat - a - xhat * b)).to(mDX.element_type)
            j = j + bt


_ln_bwd_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _ln_bwd_host(
    mDY,
    mX,
    mGamma,
    mMean,
    mRstd,
    mDX,
    mDGamma,
    mDBeta,
    R: cutlass.Int32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr,
    stage_mode: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    _ln_bwd_kernel(
        mDY,
        mX,
        mGamma,
        mMean,
        mRstd,
        mDX,
        mDGamma,
        mDBeta,
        M,
        V,
        bt,
        elem_bytes,
        stage_mode,
        vec,
        has_mean,
        has_beta,
    ).launch(stream=stream, grid=(R, 1, 1), block=(bt, 1, 1))


# ---------------------------------------------------------------------------
# Warp-specialized software-pipelined backward (mirrors the forward pipeline).
#
# The DMA warp bulk-loads BOTH dy and x into double-buffered smem; the compute
# warps do a two-pass row body (pass1: reductions c1[LN]/c2 + accumulate
# dgamma/dbeta REGISTER partials across grid-stride rows; pass2: re-read smem and
# write dx) and, after the loop, flush the register partials to a [ctas, C] global
# buffer. A finalize kernel sums those partials -> dgamma/dbeta -- NO atomics
# (atomic contention made the huge-N/small-C backward ~0.06x). Selected for wn>1.
# ---------------------------------------------------------------------------


@cute.kernel
def _ln_bwd_pipe_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mDXi: cute.Tensor,
    mGi: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    mDGp: cute.Tensor,
    mDBp: cute.Tensor,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    eb: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    cache_xd: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    smem = SmemAllocator()
    xbuf = smem.allocate_tensor(it_ty, cute.make_layout(STAGES * C), byte_alignment=16)
    dybuf = smem.allocate_tensor(it_ty, cute.make_layout(STAGES * C), byte_alignment=16)
    sG = smem.allocate_tensor(it_ty, cute.make_layout(C), byte_alignment=16)
    red = None
    if cutlass.const_expr(wn > 1):
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(wn * (2 if has_mean else 1)), byte_alignment=8)
    mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2 * STAGES + 1), byte_alignment=8)
    GBAR: cutlass.Constexpr = 2 * STAGES
    if tid == 0:
        for j in cutlass.range_constexpr(2 * STAGES + 1):
            nvvm.mbarrier_init(mbar.iterator + j, 1)
        nvvm.mbarrier_arrive_expect_tx(mbar.iterator + GBAR, C * eb)
        nvvm.cp_async_bulk_shared_cluster_global(sG.iterator, mGi.iterator, mbar.iterator + GBAR, C * eb)
    cute.arch.sync_threads()
    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + GBAR, 0):
        pass
    NB: cutlass.Constexpr = C * eb
    stride = ctas
    Mf = cutlass.Float32(C)
    rn = 1.0 / Mf

    if warp == wn:  # --- DMA warp: load dy AND x per row ---
        if lane == 0:
            i = cutlass.Int32(0)
            row = bid
            while row < R:
                s = i % STAGES
                if i >= STAGES:
                    ep = ((i // STAGES) - 1) & 1
                    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + (STAGES + s), ep):
                        pass
                nvvm.mbarrier_arrive_expect_tx(mbar.iterator + s, 2 * NB)
                nvvm.cp_async_bulk_shared_cluster_global(xbuf.iterator + s * C, mX.iterator + cutlass.Int64(row) * C, mbar.iterator + s, NB)
                nvvm.cp_async_bulk_shared_cluster_global(dybuf.iterator + s * C, mDY.iterator + cutlass.Int64(row) * C, mbar.iterator + s, NB)
                i = i + 1
                row = row + stride
    else:  # --- compute warps ---
        dgp = [cutlass.Float32(0.0)] * (ldgs * V)
        dbp = [cutlass.Float32(0.0)] * (ldgs * V) if cutlass.const_expr(has_beta) else None
        i = cutlass.Int32(0)
        row = bid
        while row < R:
            s = i % STAGES
            while not nvvm.mbarrier_try_wait_parity(mbar.iterator + s, (i // STAGES) & 1):
                pass
            boff = s * C
            mean = mMean[row] if cutlass.const_expr(has_mean) else cutlass.Float32(0.0)
            rstd = mRstd[row]
            if cutlass.const_expr(cache_xd):
                xhat_c = [cutlass.Float32(0.0)] * (ldgs * V)
                dxhat_c = [cutlass.Float32(0.0)] * (ldgs * V)
            c1 = cutlass.Float32(0.0)
            c2 = cutlass.Float32(0.0)
            for it in cutlass.range_constexpr(ldgs):
                col0 = (it * tpr + tid) * V
                xv = nvvm.load_ext(xbuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                dyv = nvvm.load_ext(dybuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                gv = nvvm.load_ext(sG.iterator + col0, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                for e in cutlass.range_constexpr(V):
                    x = xv[e].to(cutlass.Float32)
                    dy = dyv[e].to(cutlass.Float32)
                    g = gv[e].to(cutlass.Float32)
                    xhat = (x - mean) * rstd
                    dxhat = dy * g
                    if cutlass.const_expr(cache_xd):
                        xhat_c[it * V + e] = xhat
                        dxhat_c[it * V + e] = dxhat
                    if cutlass.const_expr(has_mean):
                        c1 = c1 + dxhat
                    c2 = c2 + dxhat * xhat
                    dgp[it * V + e] = dgp[it * V + e] + dy * xhat
                    if cutlass.const_expr(has_beta):
                        dbp[it * V + e] = dbp[it * V + e] + dy
            for k in cutlass.range_constexpr(5):
                off = 32 >> (k + 1)
                c2 = c2 + nvvm.shfl_sync(0xFFFFFFFF, c2, off, 0x1F, nvvm.Shfl.BFLY)
                if cutlass.const_expr(has_mean):
                    c1 = c1 + nvvm.shfl_sync(0xFFFFFFFF, c1, off, 0x1F, nvvm.Shfl.BFLY)
            if cutlass.const_expr(wn > 1):
                if lane == 0:
                    red[warp] = c2
                    if cutlass.const_expr(has_mean):
                        red[wn + warp] = c1
                nvvm.barrier_cta_sync(1, thread_count=tpr)
                c2 = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(wn):
                    c2 = c2 + red[j]
                if cutlass.const_expr(has_mean):
                    c1 = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(wn):
                        c1 = c1 + red[wn + j]
                # Fence the READS before the next row writes ``red`` again. The
                # barrier above orders the writes against the reads WITHIN a row and
                # does nothing for the row after it. The non-cached path happens to be
                # covered by the barrier it takes before freeing the staging buffer;
                # the cached path frees that buffer immediately and then touches only
                # registers, so nothing stood between this row's reads and the next
                # row's writes (racecheck: WAR on __shared__ 0xe404, 21024 errors).
                nvvm.barrier_cta_sync(1, thread_count=tpr)
            # With caching, pass2 reads registers (not smem), so the buffer is free
            # right after pass1 -> signal empty now so the DMA warp prefetches the next
            # row *during* pass2 (freeing after pass2 would kill the overlap large N
            # depends on). The barrier just above fenced this row's reduction reads.
            if cutlass.const_expr(cache_xd):
                # Fence BEFORE releasing the buffer. ``mbarrier_arrive`` does not order
                # the shared loads pass1 just issued, and only thread 0 arrives -- so
                # without this the DMA warp can start refilling the stage while this
                # warp's loads are still outstanding. wn==1 is not exempt: one warp is
                # issue-synchronous but its loads are not COMPLETE, which is what
                # racecheck caught (WAR on __shared__ 0x410, compute warp reading while
                # the DMA warp wrote, 2.3M errors at C=512 fp32).
                nvvm.barrier_cta_sync(1, thread_count=tpr)
                if tid == 0:
                    nvvm.mbarrier_arrive(mbar.iterator + (STAGES + s))
            a = c1 * rn if cutlass.const_expr(has_mean) else cutlass.Float32(0.0)
            b = c2 * rn
            base = cutlass.Int64(row) * C
            for it in cutlass.range_constexpr(ldgs):
                col0 = (it * tpr + tid) * V
                if cutlass.const_expr(not cache_xd):
                    xv = nvvm.load_ext(xbuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                    dyv = nvvm.load_ext(dybuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                    gv = nvvm.load_ext(sG.iterator + col0, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                ys = []
                for e in cutlass.range_constexpr(V):
                    if cutlass.const_expr(cache_xd):
                        dxhat = dxhat_c[it * V + e]
                        xhat = xhat_c[it * V + e]
                    else:
                        x = xv[e].to(cutlass.Float32)
                        dy = dyv[e].to(cutlass.Float32)
                        g = gv[e].to(cutlass.Float32)
                        xhat = (x - mean) * rstd
                        dxhat = dy * g
                    ys.append((rstd * (dxhat - a - xhat * b)).to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + col0))
            # Non-cached pass2 RE-READS smem, so every compute warp must finish before
            # the buffer is freed -- else the DMA reloads it mid-read and dx is garbage.
            # This barrier is NOT conditional on wn>1: a single warp is issue-synchronous
            # but its loads still have to complete, and thread 0 arriving says nothing
            # about the other 31. (Cached path freed the buffer after pass1.)
            if cutlass.const_expr(not cache_xd):
                nvvm.barrier_cta_sync(1, thread_count=tpr)
                if tid == 0:
                    nvvm.mbarrier_arrive(mbar.iterator + (STAGES + s))
            i = i + 1
            row = row + stride
        # flush register partials -> [ctas, C] (compute warps only)
        pbase = cutlass.Int64(bid) * C
        # Wide stores rather than one STG.32 per element: the [ctas, C] partials are
        # fp32 and C is large, so this flush is a real fraction of the kernel.
        #
        # SEG is the widest store V divides by, NOT a fixed 4. ``_bwd_pipe_cfg``
        # hands back V=2 for fp32 at small C, and a hard-coded ``V // 4`` made NSEG
        # zero there -- the loop body never ran, so the partials were never written
        # and the finalize summed whatever the allocator last left in them. dgamma
        # and dbeta came back as garbage for C=64 fp32 while dx stayed correct,
        # which is exactly the shape that hides: the gradient that IS checked by
        # most tests was fine.
        SEG: cutlass.Constexpr = 4 if V % 4 == 0 else (2 if V % 2 == 0 else 1)
        NSEG: cutlass.Constexpr = V // SEG
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tid) * V
            for h in cutlass.range_constexpr(NSEG):
                if cutlass.const_expr(SEG == 1):
                    mDGp[pbase + col0 + h] = dgp[it * V + h]
                    if cutlass.const_expr(has_beta):
                        mDBp[pbase + col0 + h] = dbp[it * V + h]
                else:
                    gseg = cutlass.Vector.from_elements(tuple(dgp[it * V + h * SEG + j] for j in range(SEG)), cutlass.Float32)
                    nvvm.store_ext(gseg, mDGp.iterator + (pbase + col0 + h * SEG))
                    if cutlass.const_expr(has_beta):
                        bseg = cutlass.Vector.from_elements(tuple(dbp[it * V + h * SEG + j] for j in range(SEG)), cutlass.Float32)
                        nvvm.store_ext(bseg, mDBp.iterator + (pbase + col0 + h * SEG))


_ln_bwd_pipe_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def _ln_bwd_finalize_kernel(
    mDGp: cute.Tensor,
    mDBp: cute.Tensor,
    mDG: cute.Tensor,
    mDB: cute.Tensor,
    nparts: cutlass.Int32,
    C: cutlass.Constexpr,
    FB: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    # 2D grid: x = column tile (coalesced, thread-per-column), y = partition chunk.
    # Splitting the reduction over y and atomic-adding into the tiny [C] output keeps
    # every SM busy -- a single-block finalize starves memory when C is small and the
    # partition count is large (C=128 was reducing 1776 partials on one SM).
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    c = bx * FB + tid
    if c < C:
        p0 = by * CHUNK
        p1 = p0 + CHUNK
        if p1 > nparts:
            p1 = nparts
        gsum = cutlass.Float32(0.0)
        bsum = cutlass.Float32(0.0)
        p = p0
        while p < p1:
            off = cutlass.Int64(p) * C + c
            gsum = gsum + mDGp[off]
            if cutlass.const_expr(has_beta):
                bsum = bsum + mDBp[off]
            p = p + 1
        cute.arch.atomic_add(mDG.iterator + c, gsum)
        if cutlass.const_expr(has_beta):
            cute.arch.atomic_add(mDB.iterator + c, bsum)


_ln_bwd_finalize_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _ln_bwd_pipe_host(
    mDY,
    mX,
    mDX,
    mGamma,
    mMean,
    mRstd,
    mDGp,
    mDBp,
    mDGamma,
    mDBeta,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    eb: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    cache_xd: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    FB: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
    nchunk: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    mDXi = cute.recast_tensor(mDX, it_ty)
    mGi = cute.recast_tensor(mGamma, it_ty)
    _ln_bwd_pipe_kernel(
        mDY,
        mX,
        mDXi,
        mGi,
        mMean,
        mRstd,
        mDGp,
        mDBp,
        R,
        ctas,
        C,
        V,
        tpr,
        wn,
        ldgs,
        block_threads,
        et,
        it_ty,
        eb,
        STAGES,
        cache_xd,
        has_mean,
        has_beta,
    ).launch(stream=stream, grid=(ctas, 1, 1), block=(block_threads, 1, 1), smem=smem_bytes)
    _ln_bwd_finalize_kernel(mDGp, mDBp, mDGamma, mDBeta, ctas, C, FB, CHUNK, has_beta).launch(stream=stream, grid=(fgrid, nchunk, 1), block=(FB, 1, 1))


# ---------------------------------------------------------------------------
# Tiny-C (sub-warp, wn==1) RMS backward with MULTI-ROW TILING. One cp.async.bulk
# loads RPT contiguous rows at once (a big transfer) instead of RPT tiny per-row
# copies whose fixed overhead dominates at C=128 (bwd 0.45-0.58x -> 0.82-0.96x).
# One compute warp processes the tile's rows in an inner loop; dgamma partials
# accumulate in one warp (no cross-warp reduce). RMS only (has_mean=False).
# ---------------------------------------------------------------------------


@cute.kernel
def _ln_bwd_tiled_kernel(
    mDY: cute.Tensor,
    mX: cute.Tensor,
    mDXi: cute.Tensor,
    mGi: cute.Tensor,
    mRstd: cute.Tensor,
    mDGp: cute.Tensor,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    RPT: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    eb: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    TC: cutlass.Constexpr = RPT * C
    smem = SmemAllocator()
    xbuf = smem.allocate_tensor(it_ty, cute.make_layout(STAGES * TC), byte_alignment=16)
    dybuf = smem.allocate_tensor(it_ty, cute.make_layout(STAGES * TC), byte_alignment=16)
    sG = smem.allocate_tensor(it_ty, cute.make_layout(C), byte_alignment=16)
    mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2 * STAGES + 1), byte_alignment=8)
    GBAR: cutlass.Constexpr = 2 * STAGES
    if tid == 0:
        for j in cutlass.range_constexpr(2 * STAGES + 1):
            nvvm.mbarrier_init(mbar.iterator + j, 1)
        nvvm.mbarrier_arrive_expect_tx(mbar.iterator + GBAR, C * eb)
        nvvm.cp_async_bulk_shared_cluster_global(sG.iterator, mGi.iterator, mbar.iterator + GBAR, C * eb)
    cute.arch.sync_threads()
    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + GBAR, 0):
        pass
    NB: cutlass.Constexpr = TC * eb
    ntiles = R // RPT
    stride = ctas
    Cf = cutlass.Float32(C)
    rn = 1.0 / Cf
    dgp = [cutlass.Float32(0.0)] * (ldgs * V)

    if warp == 1:  # DMA warp
        if lane == 0:
            i = cutlass.Int32(0)
            t = bid
            while t < ntiles:
                s = i % STAGES
                if i >= STAGES:
                    ep = ((i // STAGES) - 1) & 1
                    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + (STAGES + s), ep):
                        pass
                nvvm.mbarrier_arrive_expect_tx(mbar.iterator + s, 2 * NB)
                base = cutlass.Int64(t) * TC
                nvvm.cp_async_bulk_shared_cluster_global(xbuf.iterator + s * TC, mX.iterator + base, mbar.iterator + s, NB)
                nvvm.cp_async_bulk_shared_cluster_global(dybuf.iterator + s * TC, mDY.iterator + base, mbar.iterator + s, NB)
                i = i + 1
                t = t + stride
    else:  # single compute warp
        i = cutlass.Int32(0)
        t = bid
        while t < ntiles:
            s = i % STAGES
            while not nvvm.mbarrier_try_wait_parity(mbar.iterator + s, (i // STAGES) & 1):
                pass
            sbase = s * TC
            for rr in cutlass.range_constexpr(RPT):
                row = t * RPT + rr
                roff = sbase + rr * C
                rstd = mRstd[row]
                c2 = cutlass.Float32(0.0)
                xh = [cutlass.Float32(0.0)] * (ldgs * V)
                dh = [cutlass.Float32(0.0)] * (ldgs * V)
                for it in cutlass.range_constexpr(ldgs):
                    col0 = (it * tpr + tid) * V
                    xv = nvvm.load_ext(xbuf.iterator + (roff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                    dyv = nvvm.load_ext(dybuf.iterator + (roff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                    gv = nvvm.load_ext(sG.iterator + col0, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                    for e in cutlass.range_constexpr(V):
                        x = xv[e].to(cutlass.Float32)
                        dy = dyv[e].to(cutlass.Float32)
                        g = gv[e].to(cutlass.Float32)
                        xhat = x * rstd
                        dxhat = dy * g
                        xh[it * V + e] = xhat
                        dh[it * V + e] = dxhat
                        c2 = c2 + dxhat * xhat
                        dgp[it * V + e] = dgp[it * V + e] + dy * xhat
                for k in cutlass.range_constexpr(5):
                    off = 32 >> (k + 1)
                    c2 = c2 + nvvm.shfl_sync(0xFFFFFFFF, c2, off, 0x1F, nvvm.Shfl.BFLY)
                b = c2 * rn
                base = cutlass.Int64(row) * C
                for it in cutlass.range_constexpr(ldgs):
                    col0 = (it * tpr + tid) * V
                    ys = []
                    for e in cutlass.range_constexpr(V):
                        ys.append((rstd * (dh[it * V + e] - xh[it * V + e] * b)).to(et))
                    nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + col0))
            if tid == 0:
                nvvm.mbarrier_arrive(mbar.iterator + (STAGES + s))
            i = i + 1
            t = t + stride
        pbase = cutlass.Int64(bid) * C
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tid) * V
            for e in cutlass.range_constexpr(V):
                mDGp[pbase + (col0 + e)] = dgp[it * V + e]


_ln_bwd_tiled_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _ln_bwd_tiled_host(
    mDY,
    mX,
    mDX,
    mGamma,
    mRstd,
    mDGp,
    mDGamma,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    RPT: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    eb: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    FB: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
    nchunk: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    mDXi = cute.recast_tensor(mDX, it_ty)
    mGi = cute.recast_tensor(mGamma, it_ty)
    _ln_bwd_tiled_kernel(
        mDY,
        mX,
        mDXi,
        mGi,
        mRstd,
        mDGp,
        R,
        ctas,
        C,
        V,
        tpr,
        ldgs,
        RPT,
        et,
        it_ty,
        eb,
        STAGES,
    ).launch(stream=stream, grid=(ctas, 1, 1), block=(64, 1, 1), smem=smem_bytes)
    _ln_bwd_finalize_kernel(mDGp, mDGp, mDGamma, mDGamma, ctas, C, FB, CHUNK, False).launch(stream=stream, grid=(fgrid, nchunk, 1), block=(FB, 1, 1))


_PIPE_STAGES = 2
# Per-block smem limit, queried per architecture rather than fixed at Blackwell's.
_SM_COUNT = None
_KPIPE = {}


def _pipe_bwd_smem(C, wn, STAGES, eb):
    return 2 * STAGES * C * eb + C * eb + wn * 8 + (2 * STAGES + 1) * 8 + 64


def _bwd_cache_xd(ldgs, V, R, C, has_beta, wn, eb):
    """Whether pass2 caches xhat/dxhat in registers (skips the smem re-read). Costs
    (2+npb)*ldgs*V fp32 regs and lowers occupancy, so it's a win only when enough
    occupancy remains to hide latency at this work size. The crossover scales with the
    CACHED occupancy: high-occ (small C) tolerates caching at much larger R*C than
    low-occ (large C). Measured: C4096 (occ4) caches at 64M (llama3-8b 0.89->0.92),
    but C8192 (occ2) caches only up to ~48M (nemotronh 32M helps, llama3-70b 64M
    regresses 0.95->0.86). -> threshold ~ 24M * cached_occ. Register budget gates spills."""
    npb = 2 if has_beta else 1
    if (2 + npb) * ldgs * V > 120:
        return False
    occ = max(1, smem_capacity() // _pipe_bwd_smem(C, wn, 3, eb))
    return R * C <= 24 * 1024 * 1024 * occ


def _pipe_bwd_actual_stages(cache_xd):
    # 3 stages ONLY on the cached path: caching already spent occupancy on register
    # state, so the extra buffer's deeper overlap of the read-heavy (dy+x) load is a
    # net win (mixtral C4096 0.79->0.82, nemotronh C8192 0.80->0.83). The non-cached
    # path (large N, R*C>48M e.g. llama3-8b/llama3-70b) is occupancy-critical -- a 3rd
    # buffer drops it to 2 CTAs/SM and regresses hard (llama3-8b 0.88->0.67). Cached
    # shapes always have small ldgs, so C is bounded (<=~10K) and 3 buffers fit smem.
    return 3 if cache_xd else _PIPE_STAGES


def _bwd_pipe_cfg(C, eb):
    """Pipeline geometry for the backward: (tpr, wn, ldgs, V). Unlike the forward
    warp cfg this covers tiny C too -- shrink the vector width V until vec_cols is a
    multiple of 32 so the row maps to whole warps (wn>=1), which lets even C=128 use
    the pipeline (killing the sub-warp atomic-contention path). None if C<8B-aligned."""
    V = 16 // eb  # 128-bit: 8 for bf16/fp16
    while V > 1 and (C % (V * 32) != 0):
        V //= 2
    if C % V != 0 or (C // V) % 32 != 0:
        return None
    vec_cols = C // V
    best = None
    for wn in (1, 2, 4, 8):
        tpr = wn * 32
        if tpr > 256 or vec_cols % tpr != 0:
            continue
        ldgs = vec_cols // tpr
        if ldgs < 1 or ldgs > 16:
            continue
        if best is None or abs(ldgs - 4) < abs(best[2] - 4):
            best = (tpr, wn, ldgs)
    if best is None:
        return None
    tpr, wn, ldgs = best
    return (tpr, wn, ldgs, V)


def _pipe_bwd_eligible(C, wn, ldgs, V, R, has_beta, eb):
    # Use the ACTUAL stage count (cached -> 3) so the smem bound matches what the kernel
    # allocates -- a fixed STAGES=3 bound would wrongly reject non-cached large C
    # (llama31 C=16384 actually runs at 2 = 160KB, but 3 = 229KB > cap).
    STAGES = _pipe_bwd_actual_stages(_bwd_cache_xd(ldgs, V, R, C, has_beta, wn, eb))
    # V == 1 cannot be compiled by this kernel at all: it stages through
    # ``load_ext``, whose count must be one of {2, 4, 8} (PTX .v2/.v4/.v8), so a
    # one-wide vector raises at trace time rather than running slowly. C=32 and
    # C=96 fp32 land there. The plain route below has no such constraint and is
    # correct for them, so decline instead of failing the compile.
    return wn >= 1 and V >= 2 and _pipe_bwd_smem(C, wn, STAGES, eb) <= smem_capacity()


def _pipe_bwd_cap(R, wn, C, cache_xd, eb):
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    NSM = _SM_COUNT
    if cache_xd:
        # Occupancy-aware cap: the optimum balances three limits (measured by a
        # per-shape cap sweep -- this formula reproduces every mid-C optimum exactly).
        #   occ    -- don't launch more CTAs/SM than smem allows (extra waves don't help
        #             a memory-bound kernel and only grow the finalize).
        #   budget -- the finalize reads the [ctas,C] partials, so cap ctas*C to a fixed
        #             traffic budget (~2M elems) -- this is why smaller C wants MORE CTAs
        #             (mixtral C4096 -> 3x) and larger C fewer (nemotronh C8192 -> 1x).
        #   rows   -- keep >=~6 rows/CTA so the software pipeline amortizes its prologue
        #             (8 was too aggressive -- capped deepseek-2048 to 1x vs its 2x opt).
        smem = _pipe_bwd_smem(C, wn, 3, eb)
        occ = max(1, smem_capacity() // smem)
        budget = max(1, (2 * 1024 * 1024) // (NSM * C))
        rows = max(1, R // (NSM * 6))
        mult = max(1, min(occ, budget, rows))
    else:
        # Non-cached (large N): N-driven keeps the partials a ~constant fraction of the
        # data traffic. wn<=2 (tiny C) needs a higher floor to reach occupancy.
        floor = 2 if wn <= 2 else 1
        mult = max(floor, min(12, round(R / 4096)))
    return min(R, NSM * mult)


def _backward_pipe(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, params, wcfg, out_dx=None, out_dgamma=None, out_dbeta=None, ws=None):
    import torch

    tpr, wn, ldgs, V = wcfg
    R, C = spec.R, spec.M
    eb = DTYPE_BYTES[params.io_dtype]
    cache_xd = _bwd_cache_xd(ldgs, V, R, C, has_beta, wn, eb)
    STAGES = _pipe_bwd_actual_stages(cache_xd)
    block_threads = (wn + 1) * 32
    ctas = _pipe_bwd_cap(R, wn, C, cache_xd, eb)
    if mean is None:
        mean = rstd

    dx = torch.empty_like(x2d) if out_dx is None else out_dx
    # finalize atomic-accumulates over partition chunks -> outputs start at zero. Zero
    # dgamma+dbeta in ONE fill (a contiguous [2,gamma_len] buffer, sliced to views) so
    # LN pays a single zeroing-kernel launch instead of two -- the launch overhead is a
    # measurable fraction of the small-N LN backward (gpt3). RMS aliases dbeta->dgamma.
    if has_beta and out_dgamma is not None and out_dbeta is not None:
        # Caller owns both: two small zero_ launches, but it saves the two copies
        # the engine would otherwise make -- and the allocation.
        dgamma, dbeta = out_dgamma.zero_(), out_dbeta.zero_()
    elif has_beta:
        _gb = ws_zeros(ws, 2 * spec.gamma_len, torch.float32, x2d.device).reshape(2, spec.gamma_len)
        dgamma, dbeta = _gb[0], _gb[1]
    else:
        dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device) if out_dgamma is None else out_dgamma.zero_()
        dbeta = dgamma
    dgp = ws_empty(ws, ctas * C, torch.float32, x2d.device)
    dbp = ws_empty(ws, ctas * C, torch.float32, x2d.device) if has_beta else dgp

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    smem_bytes = _pipe_bwd_smem(C, wn, STAGES, eb)
    FB = 128 if C < 256 else 256
    fgrid = (C + FB - 1) // FB
    # split the finalize's partition reduction across the y-grid so it isn't stuck on
    # one SM when C is small; more chunks for few column-tiles (small C).
    nchunk = max(1, min(32, 256 // fgrid))
    if nchunk > ctas:
        nchunk = ctas
    CHUNK = (ctas + nchunk - 1) // nchunk

    args = (dyn(dy2d), dyn(x2d), dyn(dx), dyn(gamma), dyn(mean), dyn(rstd), dyn(dgp), dyn(dbp), dyn(dgamma), dyn(dbeta), cutlass.Int32(R), cutlass.Int32(ctas))
    ce = (C, V, tpr, wn, ldgs, block_threads, et, it_ty, eb, STAGES, cache_xd, spec.has_mean, has_beta, FB, fgrid, nchunk, CHUNK, smem_bytes)
    key = ("pipe", params.io_dtype) + ce
    fn = _KPIPE.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_pipe_host, *args, *ce, fake_stream())
        _KPIPE[key] = fn
    fn(*args, cur_stream())
    return dx, dgamma, (dbeta if has_beta else None)


_KTILED = {}


def _backward_tiled(spec, dy2d, x2d, gamma, rstd, *, params, wcfg, out_dx=None, out_dgamma=None, out_dbeta=None, ws=None):
    """Tiny-C (wn==1) RMS backward: multi-row tiled bulk-loads. Returns dgamma only
    (RMS has no dbeta)."""
    import torch

    tpr, wn, ldgs, V = wcfg
    R, C = spec.R, spec.M
    eb = DTYPE_BYTES[params.io_dtype]
    STAGES = 2
    RPT = 1
    for r in (8, 4, 2):
        if R % r == 0 and 2 * STAGES * r * C * eb <= smem_capacity():
            RPT = r
            break
    ntiles = R // RPT
    ctas = min(ntiles, _sm_count() * 12)

    dx = torch.empty_like(x2d) if out_dx is None else out_dx
    dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device) if out_dgamma is None else out_dgamma.zero_()
    dgp = ws_empty(ws, ctas * C, torch.float32, x2d.device)

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[DTYPE_BYTES[params.io_dtype]]
    FB = 128 if C < 256 else 256
    fgrid = (C + FB - 1) // FB
    nchunk = max(1, min(32, 256 // fgrid))
    if nchunk > ctas:
        nchunk = ctas
    CHUNK = (ctas + nchunk - 1) // nchunk
    smem_bytes = 2 * STAGES * RPT * C * eb + C * eb + (2 * STAGES + 1) * 8 + 64

    args = (dyn(dy2d), dyn(x2d), dyn(dx), dyn(gamma), dyn(rstd), dyn(dgp), dyn(dgamma), cutlass.Int32(R), cutlass.Int32(ctas))
    ce = (C, V, tpr, ldgs, RPT, et, it_ty, eb, STAGES, FB, fgrid, nchunk, CHUNK, smem_bytes)
    key = ("tiled", params.io_dtype) + ce
    fn = _KTILED.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_tiled_host, *args, *ce, fake_stream())
        _KTILED[key] = fn
    fn(*args, cur_stream())
    return dx, dgamma, None


def _sm_count():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


_KCACHE = {}


def scratch_bytes(spec, params, shape=None, has_beta=False, knobs=None):
    """Upper bound on the workspace this backward may carve.

    Mirrors ``backward``'s own dispatch rather than taking a blanket max over
    the three routes: the plain route carves no partials at all, and the tiled
    route's grid cap (sm*12) is far above the pipelined one's, so a blanket max
    reported 58 MB where the chosen route used 9.7. The conditions live a dozen
    lines below in this same file, so they cannot drift the way a cross-module
    guess would.
    """
    eb = DTYPE_BYTES[params.io_dtype]
    C, R = spec.M, spec.R
    wcfg = _bwd_pipe_cfg(C, eb)
    if wcfg is None:
        return ws_bytes((R, 4))  # plain route: only the RMS mean placeholder
    _tpr, wn, ldgs, V = wcfg
    if wn == 1 and V >= 2 and not spec.has_mean and (R % 2 == 0) and not has_beta:
        RPT = 1
        for r in (8, 4, 2):
            if R % r == 0 and 2 * 2 * r * C * eb <= smem_capacity():
                RPT = r
                break
        ctas = min(R // RPT, _sm_count() * 12)
        return ws_bytes((ctas * C, 4), (2 * spec.gamma_len, 4))
    if _pipe_bwd_eligible(C, wn, ldgs, V, R, has_beta, eb):
        ctas = _pipe_bwd_cap(R, wn, C, _bwd_cache_xd(ldgs, V, R, C, has_beta, wn, eb), eb)
        parts = [(ctas * C, 4)] * (2 if has_beta else 1)
        return ws_bytes(*parts, (2 * spec.gamma_len, 4))
    return ws_bytes((R, 4))  # plain route


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params, out_dx=None, out_dgamma=None, out_dbeta=None, ws=None):
    """Launch LayerNorm/RMSNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    wcfg = _bwd_pipe_cfg(spec.M, DTYPE_BYTES[params.io_dtype])
    if wcfg is not None and wcfg[1] == 1 and wcfg[3] >= 2 and not spec.has_mean and (spec.R % 2 == 0) and not has_beta:
        # tiny C (single warp per row) RMS: multi-row tiling beats the per-row pipeline.
        # V >= 2 because this route stages through ``load_ext`` too, whose count must
        # be one of {2, 4, 8}; C=32 and C=96 fp32 give V=1 and would fail the compile.
        # Only without beta -- this path returns dgamma alone, and selecting it for a
        # bias-carrying RMSNorm silently dropped dbeta to None (the caller then saw an
        # unwritten gradient, not an error). RMSNorm with bias is legal in cuDNN.
        return _backward_tiled(spec, dy2d, x2d, gamma, rstd, params=params, wcfg=wcfg, out_dx=out_dx, out_dgamma=out_dgamma, out_dbeta=out_dbeta, ws=ws)
    if wcfg is not None and _pipe_bwd_eligible(spec.M, wcfg[1], wcfg[2], wcfg[3], spec.R, has_beta, DTYPE_BYTES[params.io_dtype]):
        return _backward_pipe(
            spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, params=params, wcfg=wcfg, out_dx=out_dx, out_dgamma=out_dgamma, out_dbeta=out_dbeta, ws=ws
        )

    if mean is None:  # RMSNorm has no centering; kernel ignores it when has_mean=False
        mean = ws_empty(ws, spec.R, torch.float32, x2d.device)

    dx = torch.empty_like(x2d) if out_dx is None else out_dx
    dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device) if out_dgamma is None else out_dgamma.zero_()
    dbeta = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device) if out_dbeta is None else out_dbeta.zero_()

    args = (
        dyn(dy2d),
        dyn(x2d),
        dyn(gamma),
        dyn(mean),
        dyn(rstd),
        dyn(dx),
        dyn(dgamma),
        dyn(dbeta),
        cutlass.Int32(spec.R),
    )
    ce = (spec.M, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, spec.has_mean, has_beta)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_host, *args, *ce, fake_stream())
        _KCACHE[key] = fn
    fn(*args, cur_stream())
    return dx, dgamma, (dbeta if has_beta else None)
