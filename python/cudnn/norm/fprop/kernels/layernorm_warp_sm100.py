# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm/RMSNorm forward, sm_100 — warp-per-row, CUTLASS primitives.

Modeled on cuDNN's ln_fwd kernel. Built on ``cutlass.primitives`` (nvvm):

- ``tpr`` threads own one row. For large C the row is spread over ``wn = tpr//32``
  warps (``THREADS_PER_ROW = wn*32``) so each thread caches only ~``ldgs*V``
  elements (small register footprint — avoids the spill that halves bandwidth at
  large C). For tiny C, ``tpr<32`` (sub-warp) and a CTA packs several rows/warp.
- reduction: ``nvvm.shfl_sync`` butterfly within the ``intra``-lane group, then
  (``wn>1``) a cross-warp combine of the ``wn`` partials through smem.
- X loaded once via ``nvvm.load_ext`` (vectorized; recast->int + bitcast) into
  registers, reduction fused into the load; gamma/beta loaded once into smem.
- FMA-fused affine; vectorized store via ``nvvm.store_ext``. Grid-stride persistent.

Selected by ``make_warp_cfg`` for LN/RMS (norm dim C == gamma length).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn

_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}


@cute.kernel
def _warp_fwd_kernel(
    mXi: cute.Tensor,
    mYi: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    eps: cutlass.Float32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    intra: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    rpc: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    nsteps: cutlass.Constexpr,
    full_warp: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    rit = tid // tpr  # row within the CTA's tile
    tir = tid % tpr  # thread within the row group
    warp_in_row = tir // 32
    lane = tir % 32

    smem = SmemAllocator()
    sG = smem.allocate_tensor(mGamma.element_type, cute.make_layout(C), byte_alignment=16)
    sB = None
    if cutlass.const_expr(has_beta):
        sB = smem.allocate_tensor(mBeta.element_type, cute.make_layout(C), byte_alignment=16)
    red = None
    if cutlass.const_expr(wn > 1):
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(rpc * wn * (2 if has_mean else 1)), byte_alignment=8)

    i = tid
    while i < C:
        sG[i] = mGamma[i]
        if cutlass.const_expr(has_beta):
            sB[i] = mBeta[i]
        i = i + block_threads
    cute.arch.sync_threads()

    Cf = cutlass.Float32(C)
    rn = 1.0 / Cf
    stride = ctas * rpc
    row = bid * rpc + rit
    while row < R:
        base = cutlass.Int64(row) * C

        # --- pass 1: load X to registers (vectorized), fused reduce ---
        # mean/s1/neg are const_expr-guarded so the RMS path carries zero extra
        # live registers (a dead ``neg``/``s1`` tips ldgs=5 into a spill: -23% BW).
        xs = [cutlass.Float32(0.0)] * (ldgs * V)
        s2 = cutlass.Float32(0.0)
        if cutlass.const_expr(has_mean):
            s1 = cutlass.Float32(0.0)
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tir) * V
            xvec = nvvm.load_ext(mXi.iterator + (base + col0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xvec[e].to(cutlass.Float32)
                xs[it * V + e] = x
                if cutlass.const_expr(has_mean):
                    s1 = s1 + x
                s2 = s2 + x * x
        if cutlass.const_expr(full_warp):
            # wn>1 / full-warp: literal offsets & clamp fold to compile-time constants.
            # (``intra`` is a Constexpr *param* that arrives as a runtime Int32, so
            # ``intra >> k`` would NOT fold — the live intra/clamp/offset registers tip
            # ldgs=5 over the 8-CTA/SM occupancy cliff: -19% BW. Literals avoid that.)
            for k in cutlass.range_constexpr(5):
                off = 32 >> (k + 1)
                if cutlass.const_expr(has_mean):
                    s1 = s1 + nvvm.shfl_sync(0xFFFFFFFF, s1, off, 0x1F, nvvm.Shfl.BFLY)
                s2 = s2 + nvvm.shfl_sync(0xFFFFFFFF, s2, off, 0x1F, nvvm.Shfl.BFLY)
        else:
            clamp: cutlass.Constexpr = 0x1F | ((32 - intra) << 8)  # sub-warp (tiny C)
            for k in cutlass.range_constexpr(nsteps):
                o = intra >> (k + 1)
                if cutlass.const_expr(has_mean):
                    s1 = s1 + nvvm.shfl_sync(0xFFFFFFFF, s1, o, clamp, nvvm.Shfl.BFLY)
                s2 = s2 + nvvm.shfl_sync(0xFFFFFFFF, s2, o, clamp, nvvm.Shfl.BFLY)

        if cutlass.const_expr(wn > 1):  # combine the wn warp-partials via smem
            if lane == 0:
                red[rit * wn + warp_in_row] = s2
                if cutlass.const_expr(has_mean):
                    red[rpc * wn + rit * wn + warp_in_row] = s1
            cute.arch.sync_threads()
            s2 = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(wn):
                s2 = s2 + red[rit * wn + j]
            if cutlass.const_expr(has_mean):
                s1 = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(wn):
                    s1 = s1 + red[rpc * wn + rit * wn + j]

        if cutlass.const_expr(has_mean):
            mean = s1 * rn
            rstd = cute.math.rsqrt(s2 * rn - mean * mean + eps)
            neg = -rstd * mean
            if tir == 0:
                mMean[row] = mean
                mRstd[row] = rstd
        else:
            rstd = cute.math.rsqrt(s2 * rn + eps)
            if tir == 0:
                mRstd[row] = rstd

        # --- pass 2: normalize from registers + affine, vectorized store ---
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tir) * V
            ys = []
            for e in cutlass.range_constexpr(V):
                if cutlass.const_expr(has_mean):
                    y = rstd * xs[it * V + e] + neg
                else:
                    y = rstd * xs[it * V + e]
                y = sG[col0 + e].to(cutlass.Float32) * y
                if cutlass.const_expr(has_beta):
                    y = y + sB[col0 + e].to(cutlass.Float32)
                ys.append(y.to(et))
            yvec = cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty)
            nvvm.store_ext(yvec, mYi.iterator + (base + col0))
        if cutlass.const_expr(wn > 1):
            cute.arch.sync_threads()  # red reused next iteration
        row = row + stride


@cute.jit
def _warp_fwd_host(
    mX,
    mY,
    mGamma,
    mBeta,
    mMean,
    mRstd,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    eps: cutlass.Float32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    intra: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    rpc: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    nsteps: cutlass.Constexpr,
    full_warp: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _warp_fwd_kernel(
        mXi,
        mYi,
        mGamma,
        mBeta,
        mMean,
        mRstd,
        R,
        ctas,
        eps,
        C,
        V,
        tpr,
        wn,
        intra,
        ldgs,
        rpc,
        block_threads,
        et,
        it_ty,
        nsteps,
        full_warp,
        has_mean,
        has_beta,
    ).launch(grid=(ctas, 1, 1), block=(block_threads, 1, 1), min_blocks_per_mp=mbpm)


# ---------------------------------------------------------------------------
# Warp-specialized software-pipelined kernel (cuDNN ln_tma_fwd style).
#
# One CTA owns a row; the *last* warp is a dedicated DMA warp that streams whole
# rows into double-buffered smem via ``cp.async.bulk`` (1D bulk copy, no TMA
# descriptor), coordinated with the ``wn`` compute warps through full/empty
# mbarriers. The compute warps read smem VECTORIZED, reduce (shfl + a *named*
# barrier that excludes the DMA warp), free the buffer, and normalize from
# registers. This overlaps the next row's load with the current row's compute,
# closing the large-C bandwidth gap the single-CTA kernel leaves (~0.83 -> ~0.98x
# at C=16384; small-N LN/RMS 0.71/0.82 -> 1.08/1.14x). gamma/beta are bulk-loaded
# to smem once and read vectorized -- element-wise gamma reads were the pipeline's
# compute bottleneck. Selected for wn>1 (large C) by ``forward``.
_CTA_SS = nvvm.SharedSpace.shared_cta


@cute.kernel
def _warp_fwd_pipe_kernel(
    mX: cute.Tensor,
    mXi: cute.Tensor,
    mYi: cute.Tensor,
    mGi: cute.Tensor,
    mBi: cute.Tensor,
    mMean: cute.Tensor,
    mRstd: cute.Tensor,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    eps: cutlass.Float32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32

    smem = SmemAllocator()
    xbuf = smem.allocate_tensor(cutlass.Int16, cute.make_layout(STAGES * C), byte_alignment=16)
    sG = smem.allocate_tensor(cutlass.Int16, cute.make_layout(C), byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.Int16, cute.make_layout(C), byte_alignment=16) if cutlass.const_expr(has_beta) else None
    red = None
    if cutlass.const_expr(wn > 1):
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(wn * (2 if has_mean else 1)), byte_alignment=8)
    npb: cutlass.Constexpr = 2 if has_beta else 1
    mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2 * STAGES + npb), byte_alignment=8)
    GBAR: cutlass.Constexpr = 2 * STAGES

    if tid == 0:
        for j in cutlass.range_constexpr(2 * STAGES + npb):
            nvvm.mbarrier_init(mbar.iterator + j, 1)
        nvvm.mbarrier_arrive_expect_tx(mbar.iterator + GBAR, C * 2)
        nvvm.cp_async_bulk_shared_cluster_global(sG.iterator, mGi.iterator, mbar.iterator + GBAR, C * 2)
        if cutlass.const_expr(has_beta):
            nvvm.mbarrier_arrive_expect_tx(mbar.iterator + (GBAR + 1), C * 2)
            nvvm.cp_async_bulk_shared_cluster_global(sB.iterator, mBi.iterator, mbar.iterator + (GBAR + 1), C * 2)
    cute.arch.sync_threads()
    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + GBAR, 0):
        pass
    if cutlass.const_expr(has_beta):
        while not nvvm.mbarrier_try_wait_parity(mbar.iterator + (GBAR + 1), 0):
            pass

    NB: cutlass.Constexpr = C * 2
    stride = ctas
    Cf = cutlass.Float32(C)
    rn = 1.0 / Cf

    if warp == wn:  # --- DMA warp: producer ---
        if lane == 0:
            i = cutlass.Int32(0)
            row = bid
            while row < R:
                s = i % STAGES
                if i >= STAGES:  # buffer reused -> wait for compute to free it
                    ep = ((i // STAGES) - 1) & 1
                    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + (STAGES + s), ep):
                        pass
                nvvm.mbarrier_arrive_expect_tx(mbar.iterator + s, NB)
                nvvm.cp_async_bulk_shared_cluster_global(xbuf.iterator + s * C, mX.iterator + cutlass.Int64(row) * C, mbar.iterator + s, NB)
                i = i + 1
                row = row + stride
    else:  # --- compute warps: consumer ---
        i = cutlass.Int32(0)
        row = bid
        while row < R:
            s = i % STAGES
            while not nvvm.mbarrier_try_wait_parity(mbar.iterator + s, (i // STAGES) & 1):
                pass
            boff = s * C

            xs = [cutlass.Float32(0.0)] * (ldgs * V)
            s2 = cutlass.Float32(0.0)
            if cutlass.const_expr(has_mean):
                s1 = cutlass.Float32(0.0)
            for it in cutlass.range_constexpr(ldgs):
                col0 = (it * tpr + tid) * V
                xv = nvvm.load_ext(xbuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                for e in cutlass.range_constexpr(V):
                    x = xv[e].to(cutlass.Float32)
                    xs[it * V + e] = x
                    if cutlass.const_expr(has_mean):
                        s1 = s1 + x
                    s2 = s2 + x * x
            for k in cutlass.range_constexpr(5):
                off = 32 >> (k + 1)
                if cutlass.const_expr(has_mean):
                    s1 = s1 + nvvm.shfl_sync(0xFFFFFFFF, s1, off, 0x1F, nvvm.Shfl.BFLY)
                s2 = s2 + nvvm.shfl_sync(0xFFFFFFFF, s2, off, 0x1F, nvvm.Shfl.BFLY)
            if cutlass.const_expr(wn > 1):
                if lane == 0:
                    red[warp] = s2
                    if cutlass.const_expr(has_mean):
                        red[wn + warp] = s1
                nvvm.barrier_cta_sync(1, thread_count=tpr)  # compute warps only (excl. DMA)
                s2 = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(wn):
                    s2 = s2 + red[j]
                if cutlass.const_expr(has_mean):
                    s1 = cutlass.Float32(0.0)
                    for j in cutlass.range_constexpr(wn):
                        s1 = s1 + red[wn + j]

            if cutlass.const_expr(has_mean):
                mean = s1 * rn
                rstd = cute.math.rsqrt(s2 * rn - mean * mean + eps)
                neg = -rstd * mean
                if tid == 0:
                    mMean[row] = mean
                    mRstd[row] = rstd
            else:
                rstd = cute.math.rsqrt(s2 * rn + eps)
                if tid == 0:
                    mRstd[row] = rstd
            if tid == 0:  # buffer fully read -> free it for the DMA warp
                nvvm.mbarrier_arrive(mbar.iterator + (STAGES + s))

            base = cutlass.Int64(row) * C
            for it in cutlass.range_constexpr(ldgs):
                col0 = (it * tpr + tid) * V
                gv = nvvm.load_ext(sG.iterator + col0, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                bv = None
                if cutlass.const_expr(has_beta):
                    bv = nvvm.load_ext(sB.iterator + col0, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                ys = []
                for e in cutlass.range_constexpr(V):
                    if cutlass.const_expr(has_mean):
                        y = rstd * xs[it * V + e] + neg
                    else:
                        y = rstd * xs[it * V + e]
                    y = gv[e].to(cutlass.Float32) * y
                    if cutlass.const_expr(has_beta):
                        y = y + bv[e].to(cutlass.Float32)
                    ys.append(y.to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (base + col0))
            i = i + 1
            row = row + stride


@cute.jit
def _warp_fwd_pipe_host(
    mX,
    mY,
    mGamma,
    mBeta,
    mMean,
    mRstd,
    R: cutlass.Int32,
    ctas: cutlass.Int32,
    eps: cutlass.Float32,
    C: cutlass.Constexpr,
    V: cutlass.Constexpr,
    tpr: cutlass.Constexpr,
    wn: cutlass.Constexpr,
    ldgs: cutlass.Constexpr,
    block_threads: cutlass.Constexpr,
    et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    STAGES: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    mGi = cute.recast_tensor(mGamma, it_ty)
    mBi = cute.recast_tensor(mBeta, it_ty)
    _warp_fwd_pipe_kernel(
        mX,
        mXi,
        mYi,
        mGi,
        mBi,
        mMean,
        mRstd,
        R,
        ctas,
        eps,
        C,
        V,
        tpr,
        wn,
        ldgs,
        block_threads,
        et,
        it_ty,
        STAGES,
        has_mean,
        has_beta,
    ).launch(grid=(ctas, 1, 1), block=(block_threads, 1, 1), smem=smem_bytes)


# min CTAs/SM (launch_bounds minnctapersm). 0 = compiler default (measured: forcing
# it spills and hurts, so left off).
_MBPM = 0
# Persistent grid: cap the grid so each CTA grid-strides over many rows, overlapping
# row i's reduction latency with row i+1's loads (a one-tile-per-CTA grid exposes
# that latency and loses ~30% bandwidth). The cap multiple of SM_COUNT is picked to
# hold a *constant working set per SM*: ``mult * (block_threads * ldgs) ~ 4096``, i.e.
# ``mult ~ 4096 / (block_threads*ldgs)`` (== round(32768/C) at rpc=1). This matches
# the measured occupancy sweet spot for every large-C shape and, crucially, lands on
# the safe side of a sharp occupancy cliff: e.g. C=5120 peaks at 6-7x (0.91x) but
# 8x collapses to 0.77x. For wn==1 (small C, cheap 1-2-ldg rows) latency is hidden by
# raw occupancy, so fill the machine (mult=8; clamped to full_ctas for huge-N shapes).
_CTAS_CAP = 0  # test override; 0 = use the formula
_SM_COUNT = None


def _persist_cap(full_ctas, block_threads, wn, ldgs):
    global _SM_COUNT
    if _CTAS_CAP:
        return min(full_ctas, _CTAS_CAP)
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    if wn > 1:
        mult = max(2, min(16, round(4096 / (block_threads * ldgs))))
    else:
        mult = 8
    return min(full_ctas, _SM_COUNT * mult)


_KCACHE = {}

# Warp-specialized pipeline knobs. STAGES=2 (double-buffer) is the measured sweet
# spot (deeper hurts occupancy, matching cuDNN's ">2 negligible"). The pipeline is
# used for wn>1 (large C) when its smem (STAGES row buffers + gamma[+beta]) fits;
# tiny/sub-warp C (wn==1) keeps the single-CTA kernel. Persist cap ~ round(16384/C)
# clamped [2,8] (the pipeline's bigger block+smem wants ~half the single-CTA cap).
_PIPE_STAGES = 2
_PIPE_SMEM_MAX = 228 * 1024
_USE_PIPE = True  # test override


def _pipe_smem(C, has_beta, wn, STAGES):
    return STAGES * C * 2 + (2 if has_beta else 1) * C * 2 + wn * 8 + (2 * STAGES + (2 if has_beta else 1)) * 8 + 64


def _pipe_stages(C):
    # STAGES=2 (double-buffer) is the sweet spot; the smallest C (tiny row buffers,
    # smem ~free) needs a deeper pipeline to keep enough loads in flight.
    return 3 if C <= 2048 else _PIPE_STAGES


def _pipe_cap(full_ctas, C):
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    mult = max(2, min(6, round(16384 / C)))  # >6x overshoots at tiny C (qwen-2048)
    return min(full_ctas, _SM_COUNT * mult)


def _pipe_eligible(C, wn, has_beta):
    return _USE_PIPE and wn > 1 and _pipe_smem(C, has_beta, wn, _pipe_stages(C)) <= _PIPE_SMEM_MAX


def forward(spec, x2d, gamma, beta, *, eps, wcfg, params):
    """Launch the warp-per-row LN/RMS forward. ``wcfg`` = (tpr, wn, intra, ldgs, rpc, block_threads, V)."""
    import torch

    tpr, wn, intra, ldgs, rpc, block_threads, V = wcfg
    has_beta = beta is not None
    R, C = spec.R, spec.M

    if _pipe_eligible(C, wn, has_beta):
        return _forward_pipe(spec, x2d, gamma, beta, eps=eps, wcfg=wcfg, params=params, has_beta=has_beta)

    ctas = _persist_cap((R + rpc - 1) // rpc, block_threads, wn, ldgs)  # persistent grid
    y = torch.empty_like(x2d)
    rstd = torch.empty(R, dtype=torch.float32, device=x2d.device)
    # Unused kernel operands (beta for RMS, mean for RMS) alias an existing tensor:
    # a distinct dummy alloc adds a per-call memset (counted in device-time) and an
    # extra pointer register that tips ldgs=5 below the 8-CTA/SM occupancy cliff.
    mean = torch.empty(R, dtype=torch.float32, device=x2d.device) if spec.has_mean else rstd
    if beta is None:
        beta = gamma

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[DTYPE_BYTES[params.io_dtype]]
    nsteps = int(intra).bit_length() - 1
    full_warp = int(intra) == 32

    args = (dyn(x2d), dyn(y), dyn(gamma), dyn(beta), dyn(mean), dyn(rstd), cutlass.Int32(R), cutlass.Int32(ctas), cutlass.Float32(eps))
    ce = (C, V, tpr, wn, intra, ldgs, rpc, block_threads, et, it_ty, nsteps, full_warp, spec.has_mean, has_beta, _MBPM)
    key = (params.io_dtype, C, tpr, wn, intra, ldgs, rpc, block_threads, spec.has_mean, has_beta, _MBPM)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_warp_fwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd


def _forward_pipe(spec, x2d, gamma, beta, *, eps, wcfg, params, has_beta):
    """Warp-specialized software-pipelined launch (large C). block=(wn+1)*32 so the
    last warp is a dedicated DMA warp; one row per CTA (rpc=1), grid-strided."""
    import torch

    tpr, wn, intra, ldgs, rpc, _bt, V = wcfg
    R, C = spec.R, spec.M
    STAGES = _pipe_stages(C)
    block_threads = (wn + 1) * 32
    ctas = _pipe_cap(R, C)

    y = torch.empty_like(x2d)
    rstd = torch.empty(R, dtype=torch.float32, device=x2d.device)
    mean = torch.empty(R, dtype=torch.float32, device=x2d.device) if spec.has_mean else rstd
    if beta is None:
        beta = gamma  # unused (not bulk-loaded) when has_beta is False

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[DTYPE_BYTES[params.io_dtype]]
    smem_bytes = _pipe_smem(C, has_beta, wn, STAGES)

    args = (dyn(x2d), dyn(y), dyn(gamma), dyn(beta), dyn(mean), dyn(rstd), cutlass.Int32(R), cutlass.Int32(ctas), cutlass.Float32(eps))
    ce = (C, V, tpr, wn, ldgs, block_threads, et, it_ty, STAGES, spec.has_mean, has_beta, smem_bytes)
    key = ("pipe", params.io_dtype, C, tpr, wn, ldgs, block_threads, STAGES, spec.has_mean, has_beta)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_warp_fwd_pipe_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
