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

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm._common_sm100 import (
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
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
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
            g = mGamma[j].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            s_dxhat = s_dxhat + dxhat
            s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
            cute.arch.atomic_add(mDGamma.iterator + j, dy * xhat)
            if cutlass.const_expr(has_beta):
                cute.arch.atomic_add(mDBeta.iterator + j, dy)
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
                g = mGamma[j0 + e].to(cutlass.Float32)
                xhat = (x - mean) * rstd
                dxhat = dy * g
                s_dxhat = s_dxhat + dxhat
                s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
                cute.arch.atomic_add(mDGamma.iterator + (j0 + e), dy * xhat)
                if cutlass.const_expr(has_beta):
                    cute.arch.atomic_add(mDBeta.iterator + (j0 + e), dy)
            vi = vi + bt
    else:
        j = tid
        while j < M:
            x = mX[r, j].to(cutlass.Float32)
            dy = mDY[r, j].to(cutlass.Float32)
            g = mGamma[j].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            s_dxhat = s_dxhat + dxhat
            s_dxhat_xhat = s_dxhat_xhat + dxhat * xhat
            cute.arch.atomic_add(mDGamma.iterator + j, dy * xhat)
            if cutlass.const_expr(has_beta):
                cute.arch.atomic_add(mDBeta.iterator + j, dy)
            j = j + bt

    s_dxhat, s_dxhat_xhat = block_reduce_sum2(s_dxhat, s_dxhat_xhat, tid, red, bt)
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
                g = mGamma[j0 + e].to(cutlass.Float32)
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
            g = mGamma[j].to(cutlass.Float32)
            xhat = (x - mean) * rstd
            dxhat = dy * g
            mDX[r, j] = (rstd * (dxhat - a - xhat * b)).to(mDX.element_type)
            j = j + bt


@cute.jit
def _ln_bwd_host(
    mDY, mX, mGamma, mMean, mRstd, mDX, mDGamma, mDBeta,
    R: cutlass.Int32,
    M: cutlass.Constexpr, V: cutlass.Constexpr, bt: cutlass.Constexpr,
    elem_bytes: cutlass.Constexpr, stage_mode: cutlass.Constexpr, vec: cutlass.Constexpr,
    has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    _ln_bwd_kernel(
        mDY, mX, mGamma, mMean, mRstd, mDX, mDGamma, mDBeta,
        M, V, bt, elem_bytes, stage_mode, vec, has_mean, has_beta,
    ).launch(grid=(R, 1, 1), block=(bt, 1, 1))


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
    mDY: cute.Tensor, mX: cute.Tensor, mDXi: cute.Tensor, mGi: cute.Tensor,
    mMean: cute.Tensor, mRstd: cute.Tensor, mDGp: cute.Tensor, mDBp: cute.Tensor,
    R: cutlass.Int32, ctas: cutlass.Int32,
    C: cutlass.Constexpr, V: cutlass.Constexpr, tpr: cutlass.Constexpr, wn: cutlass.Constexpr,
    ldgs: cutlass.Constexpr, block_threads: cutlass.Constexpr, et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr, STAGES: cutlass.Constexpr,
    has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    smem = SmemAllocator()
    xbuf = smem.allocate_tensor(cutlass.Int16, cute.make_layout(STAGES * C), byte_alignment=16)
    dybuf = smem.allocate_tensor(cutlass.Int16, cute.make_layout(STAGES * C), byte_alignment=16)
    sG = smem.allocate_tensor(cutlass.Int16, cute.make_layout(C), byte_alignment=16)
    red = None
    if cutlass.const_expr(wn > 1):
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(wn * (2 if has_mean else 1)), byte_alignment=8)
    mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2 * STAGES + 1), byte_alignment=8)
    GBAR: cutlass.Constexpr = 2 * STAGES
    if tid == 0:
        for j in cutlass.range_constexpr(2 * STAGES + 1):
            nvvm.mbarrier_init(mbar.iterator + j, 1)
        nvvm.mbarrier_arrive_expect_tx(mbar.iterator + GBAR, C * 2)
        nvvm.cp_async_bulk_shared_cluster_global(sG.iterator, mGi.iterator, mbar.iterator + GBAR, C * 2)
    cute.arch.sync_threads()
    while not nvvm.mbarrier_try_wait_parity(mbar.iterator + GBAR, 0):
        pass
    NB: cutlass.Constexpr = C * 2
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
            a = c1 * rn if cutlass.const_expr(has_mean) else cutlass.Float32(0.0)
            b = c2 * rn
            base = cutlass.Int64(row) * C
            for it in cutlass.range_constexpr(ldgs):
                col0 = (it * tpr + tid) * V
                xv = nvvm.load_ext(xbuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                dyv = nvvm.load_ext(dybuf.iterator + (boff + col0), dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                gv = nvvm.load_ext(sG.iterator + col0, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                ys = []
                for e in cutlass.range_constexpr(V):
                    x = xv[e].to(cutlass.Float32)
                    dy = dyv[e].to(cutlass.Float32)
                    g = gv[e].to(cutlass.Float32)
                    xhat = (x - mean) * rstd
                    dxhat = dy * g
                    ys.append((rstd * (dxhat - a - xhat * b)).to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + col0))
            # pass2 RE-READS smem, so all compute warps must finish before the buffer
            # is freed. wn==1 is a single warp (SIMT-synchronous) and needs no barrier;
            # wn>1 does (else the DMA warp reloads the buffer mid-read -> garbage dx).
            if cutlass.const_expr(wn > 1):
                nvvm.barrier_cta_sync(1, thread_count=tpr)
            if tid == 0:
                nvvm.mbarrier_arrive(mbar.iterator + (STAGES + s))
            i = i + 1
            row = row + stride
        # flush register partials -> [ctas, C] (compute warps only)
        pbase = cutlass.Int64(bid) * C
        for it in cutlass.range_constexpr(ldgs):
            col0 = (it * tpr + tid) * V
            for e in cutlass.range_constexpr(V):
                mDGp[pbase + (col0 + e)] = dgp[it * V + e]
                if cutlass.const_expr(has_beta):
                    mDBp[pbase + (col0 + e)] = dbp[it * V + e]


@cute.kernel
def _ln_bwd_finalize_kernel(
    mDGp: cute.Tensor, mDBp: cute.Tensor, mDG: cute.Tensor, mDB: cute.Tensor,
    nparts: cutlass.Int32, C: cutlass.Constexpr, FB: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    c = bid * FB + tid
    if c < C:
        gsum = cutlass.Float32(0.0)
        bsum = cutlass.Float32(0.0)
        p = cutlass.Int32(0)
        while p < nparts:
            off = cutlass.Int64(p) * C + c
            gsum = gsum + mDGp[off]
            if cutlass.const_expr(has_beta):
                bsum = bsum + mDBp[off]
            p = p + 1
        mDG[c] = gsum
        if cutlass.const_expr(has_beta):
            mDB[c] = bsum


@cute.jit
def _ln_bwd_pipe_host(
    mDY, mX, mDX, mGamma, mMean, mRstd, mDGp, mDBp, mDGamma, mDBeta,
    R: cutlass.Int32, ctas: cutlass.Int32,
    C: cutlass.Constexpr, V: cutlass.Constexpr, tpr: cutlass.Constexpr, wn: cutlass.Constexpr,
    ldgs: cutlass.Constexpr, block_threads: cutlass.Constexpr, et: cutlass.Constexpr,
    it_ty: cutlass.Constexpr, STAGES: cutlass.Constexpr, has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr, FB: cutlass.Constexpr, fgrid: cutlass.Constexpr, smem_bytes: cutlass.Constexpr,
) -> None:
    mDXi = cute.recast_tensor(mDX, it_ty)
    mGi = cute.recast_tensor(mGamma, it_ty)
    _ln_bwd_pipe_kernel(
        mDY, mX, mDXi, mGi, mMean, mRstd, mDGp, mDBp, R, ctas,
        C, V, tpr, wn, ldgs, block_threads, et, it_ty, STAGES, has_mean, has_beta,
    ).launch(grid=(ctas, 1, 1), block=(block_threads, 1, 1), smem=smem_bytes)
    _ln_bwd_finalize_kernel(mDGp, mDBp, mDGamma, mDBeta, ctas, C, FB, has_beta).launch(
        grid=(fgrid, 1, 1), block=(FB, 1, 1))


_PIPE_STAGES = 2
_PIPE_SMEM_MAX = 228 * 1024
_SM_COUNT = None
_KPIPE = {}


def _pipe_bwd_smem(C, wn, STAGES):
    return 2 * STAGES * C * 2 + C * 2 + wn * 8 + (2 * STAGES + 1) * 8 + 64


def _pipe_bwd_stages(C):
    return 3 if C <= 2048 else _PIPE_STAGES


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


def _pipe_bwd_eligible(C, wn):
    return wn >= 1 and _pipe_bwd_smem(C, wn, _pipe_bwd_stages(C)) <= _PIPE_SMEM_MAX


def _pipe_bwd_cap(R, C):
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch
        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    mult = max(2, min(6, round(16384 / C)))
    return min(R, _SM_COUNT * mult)


def _backward_pipe(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, params, wcfg):
    import torch

    tpr, wn, ldgs, V = wcfg
    R, C = spec.R, spec.M
    STAGES = _pipe_bwd_stages(C)
    block_threads = (wn + 1) * 32
    ctas = _pipe_bwd_cap(R, C)
    if mean is None:
        mean = rstd

    dx = torch.empty_like(x2d)
    dgamma = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dgp = torch.empty(ctas * C, dtype=torch.float32, device=x2d.device)
    dbp = torch.empty(ctas * C, dtype=torch.float32, device=x2d.device) if has_beta else dgp

    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[DTYPE_BYTES[params.io_dtype]]
    smem_bytes = _pipe_bwd_smem(C, wn, STAGES)
    FB = 256
    fgrid = (C + FB - 1) // FB

    args = (dyn(dy2d), dyn(x2d), dyn(dx), dyn(gamma), dyn(mean), dyn(rstd),
            dyn(dgp), dyn(dbp), dyn(dgamma), dyn(dbeta),
            cutlass.Int32(R), cutlass.Int32(ctas))
    ce = (C, V, tpr, wn, ldgs, block_threads, et, it_ty, STAGES, spec.has_mean, has_beta, FB, fgrid, smem_bytes)
    key = ("pipe", params.io_dtype, C, tpr, wn, ldgs, block_threads, STAGES, spec.has_mean, has_beta)
    fn = _KPIPE.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_pipe_host, *args, *ce)
        _KPIPE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)


_KCACHE = {}


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params):
    """Launch LayerNorm/RMSNorm backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    wcfg = _bwd_pipe_cfg(spec.M, DTYPE_BYTES[params.io_dtype])
    if wcfg is not None and _pipe_bwd_eligible(spec.M, wcfg[1]):
        return _backward_pipe(spec, dy2d, x2d, gamma, mean, rstd, has_beta=has_beta, params=params, wcfg=wcfg)

    if mean is None:  # RMSNorm has no centering; kernel ignores it when has_mean=False
        mean = torch.empty(spec.R, dtype=torch.float32, device=x2d.device)

    dx = torch.empty_like(x2d)
    dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device)

    args = (
        dyn(dy2d), dyn(x2d), dyn(gamma), dyn(mean), dyn(rstd),
        dyn(dx), dyn(dgamma), dyn(dbeta), cutlass.Int32(spec.R),
    )
    ce = (spec.M, cfg.V, cfg.block_threads, cfg.elem_bytes, cfg.stage_mode, cfg.vec, spec.has_mean, has_beta)
    key = (params.io_dtype,) + ce
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
