# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm / RMSNorm backward with the row split across a CGA, sm_100.

The backward counterpart of :mod:`cudnn.norm.fprop.kernels.layernorm_cga_sm100`, and
it needs the same cross-CTA reduction. With ``xhat = (x - mean) * rstd`` and
``dxhat = dy * gamma``::

    c1 = sum_j(dxhat) / C          c2 = sum_j(dxhat * xhat) / C     (per row)
    dx = rstd * (dxhat - c1 - xhat * c2)
    dgamma_j = sum_rows(dy * xhat)  dbeta_j = sum_rows(dy)          (per column)

RMSNorm drops ``c1`` and ``dbeta`` and uses ``xhat = x * rstd``.

So ``c1``/``c2`` are exactly the ``(sum, sum_sq)``-shaped pair the forward already
reduces through distributed shared memory, and the split carries over unchanged: each
CTA reduces its ``C/CGA`` slice, publishes one pair, and all CTAs read every peer's
pair via ``mapa`` and finalise redundantly.

The per-COLUMN gradients need no cross-CTA traffic at all, which is what makes this
split fit the backward so cleanly: CTA rank ``r`` owns columns ``[r*CS, (r+1)*CS)``
for every row it ever sees, so its ``dgamma``/``dbeta`` register partials are already
complete for those columns. A cluster collectively writes one whole ``[C]`` row of
partials, so the buffer is ``[nclusters, C]`` and the existing finalize reduces it
with ``nparts = nclusters``. No atomics.

Splitting also *relieves* register pressure here, unlike in the forward: a thread
holds ``2 * VPT * V`` per-column accumulators, and ``VPT`` shrinks with the slice.
That is why the cluster is sized a step wider than the forward's rule asks for.

``c1``/``c2`` are not known until the whole row has been reduced, so pass 2 needs
``x`` and ``dy`` a second time. Keeping them in registers costs ``2*VPT*V`` on top of
the per-column partials, and re-reading them from global turns the minimal 3 units of
traffic (read x, read dy, write dx) into 5 -- which measured exactly as badly as that
implies. Instead the slice is STAGED IN SHARED MEMORY, which is affordable precisely
because the split made it small: ``KS`` vectors per thread are staged under a fixed
byte budget and only the remainder, if any, is re-read.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.fprop.kernels.layernorm_cga_sm100 import _MAX_CGA, _pick_cga
from cudnn.norm.utils import dyn, sm_count

from .layernorm_sm100 import _ln_bwd_finalize_kernel

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_BT = 256
_STAGE_CAP = 64 * 1024  # smem for the x+dy slice stage; ~3 CTAs/SM at this size
_MAX_VPT = 4  # per-column partials cost 2*VPT*V registers

# Crossover measured against the existing pipelined backward (bf16, fraction of the
# achievable copy; the backward moves 3 units against a 2-unit ceiling):
#
#   R=64    D=8192 .24/.19   16384 .20/.14   32768 .12/.06   131072 .08/.03
#   R=128          .19/.19         .14/.15         .13/.09
#   R=256          .16/.21         .15/.18         .11/.10
#   R=512          .16/.26         .12/.23         .14/.13   131072 .21/.11
#   R=4096         .31/.52         .23/.60         .21/.17   131072 .31/.17
#
# At D>=32768 the split wins at every row count. Below that it only wins when there
# are too few rows to fill the machine without it -- the existing warp-specialised
# pipeline is simply a better kernel for short-ish rows, and routing D=16384 here
# would cost 2.6x at R=4096. Hence two thresholds rather than the forward's one.
_CGA_MIN_C = 32768
_CGA_MIN_C_FEW_ROWS = 16384
_FEW_ROWS = 128


@cute.jit
def _block_sum2(v1, v2, tid, red, bt: cutlass.Constexpr):
    """Reduce two fp32 partials across the CTA; nvvm primitives only."""
    nwarps: cutlass.Constexpr = bt // 32
    warp = tid // 32
    lane = tid % 32
    for d in cutlass.range_constexpr(5):
        off = 1 << d
        v1 = v1 + nvvm.shfl_sync(_FULL, v1, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
        v2 = v2 + nvvm.shfl_sync(_FULL, v2, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
    if cutlass.const_expr(nwarps == 1):
        return v1, v2
    if lane == 0:
        red[warp] = v1
        red[nwarps + warp] = v2
    nvvm.barrier_cta_sync_aligned(0)
    a = cutlass.Float32(0.0)
    b = cutlass.Float32(0.0)
    for w in cutlass.range_constexpr(nwarps):
        a = a + red[w]
        b = b + red[nwarps + w]
    nvvm.barrier_cta_sync_aligned(0)
    return a, b


def cga_cfg(C, R, eb, block_threads=_BT):
    """``(CGA, CS, V, VPT, KS)`` for a backward row split, or None."""
    V = 16 // eb
    lanes = block_threads * V
    if C % lanes:
        return None
    cga = _pick_cga(C, R, V, block_threads)
    # One step wider than the forward wants: the per-column dgamma/dbeta partials are
    # 2*VPT*V registers per thread, so a narrower slice buys register headroom that
    # the forward (which holds no per-column state) does not need.
    while cga < _MAX_CGA and (C // cga) > _MAX_VPT * lanes and C % (cga * 2 * lanes) == 0:
        cga *= 2
    while cga > 1 and (C % (cga * lanes) or (C // cga) < lanes):
        cga //= 2
    if cga < 2:
        return None
    cs = C // cga
    vpt = cs // lanes
    # Stage as much of the slice as the byte budget allows; the rest is re-read.
    ks = min(vpt, _STAGE_CAP // (2 * lanes * eb))
    return cga, cs, V, vpt, ks


def eligible(C, R, eb, block_threads=_BT):
    return cga_cfg(C, R, eb, block_threads) is not None


def should_use(C, R, eb, block_threads=_BT):
    """True when splitting the row across a cluster beats the pipelined backward."""
    if not eligible(C, R, eb, block_threads):
        return False
    if C >= _CGA_MIN_C:
        return True
    return C >= _CGA_MIN_C_FEW_ROWS and R < _FEW_ROWS


@cute.kernel
def _ln_bwd_cga_kernel(
    mDYi, mXi, mDXi, mG, mMean, mRstd, mDGp, mDBp,
    R: cutlass.Int32, nclusters: cutlass.Int32,
    C: cutlass.Constexpr, CS: cutlass.Constexpr, CGA: cutlass.Constexpr,
    BT: cutlass.Constexpr, V: cutlass.Constexpr, VPT: cutlass.Constexpr,
    KS: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    Cf: cutlass.Constexpr, has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    nwarps: cutlass.Constexpr = BT // 32

    smem = SmemAllocator()
    part = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2), byte_alignment=16)
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * nwarps), byte_alignment=16)
    sx = smem.allocate_tensor(it_ty, cute.make_layout(KS * BT * V), byte_alignment=16)
    sd = smem.allocate_tensor(it_ty, cute.make_layout(KS * BT * V), byte_alignment=16)

    rank = nvvm.cluster_ctarank()
    cl = bx // CGA
    c0 = rank * CS  # this CTA's column slice, fixed for every row it sees

    gc = []
    for k in cutlass.range_constexpr(VPT):
        gv = nvvm.load_ext(mG.iterator + (c0 + (tid + k * BT) * V), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            gc.append(gv[e].to(cutlass.Float32))

    dg = [cutlass.Float32(0.0)] * (VPT * V)
    db = [cutlass.Float32(0.0)] * (VPT * V)

    row = cl
    while row < R:
        base = cutlass.Int64(row) * C + c0
        mn = mMean[row] if cutlass.const_expr(has_mean) else cutlass.Float32(0.0)
        rs = mRstd[row]

        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)

        # ---- pass 1: row slice -> (c1, c2) partials + per-column accumulation ----
        for k in cutlass.range_constexpr(VPT):
            off = (tid + k * BT) * V
            rx = nvvm.load_ext(mXi.iterator + (base + off), dtype=it_ty, count=V)
            rd = nvvm.load_ext(mDYi.iterator + (base + off), dtype=it_ty, count=V)
            xv = rx.bitcast(et)
            dv = rd.bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                d = dv[e].to(cutlass.Float32)
                xh = (x - mn) * rs
                dxh = d * gc[k * V + e]
                s1 = s1 + dxh
                s2 = s2 + dxh * xh
                dg[k * V + e] = dg[k * V + e] + d * xh
                db[k * V + e] = db[k * V + e] + d
            if cutlass.const_expr(k < KS):
                nvvm.store_ext(rx, sx.iterator + (tid + k * BT) * V, shared_space=_CTA_SS)
                nvvm.store_ext(rd, sd.iterator + (tid + k * BT) * V, shared_space=_CTA_SS)

        s1, s2 = _block_sum2(s1, s2, tid, red, BT)
        if tid == 0:
            part[0] = s1
            part[1] = s2
        nvvm.barrier_cta_sync_aligned(0)
        nvvm.fence_sc_cluster()
        nvvm.barrier_cluster_arrive_aligned()
        nvvm.barrier_cluster_wait_aligned()
        t1 = cutlass.Float32(0.0)
        t2 = cutlass.Float32(0.0)
        for r in cutlass.range_constexpr(CGA):
            peer = nvvm.mapa(part.iterator, cutlass.Int32(r))
            t1 = t1 + peer[0]  # indexing, not load_ext: addrspace-7 pointer
            t2 = t2 + peer[1]
        # Peers must finish READING before anyone rewrites `part` on the next row.
        nvvm.barrier_cluster_arrive_aligned()
        nvvm.barrier_cluster_wait_aligned()

        c1 = t1 / Cf
        c2 = t2 / Cf

        # ---- pass 2 ----
        for k in cutlass.range_constexpr(VPT):
            off = (tid + k * BT) * V
            if cutlass.const_expr(k < KS):
                xv = nvvm.load_ext(sx.iterator + (tid + k * BT) * V, dtype=it_ty, count=V,
                                   shared_space=_CTA_SS).bitcast(et)
                dv = nvvm.load_ext(sd.iterator + (tid + k * BT) * V, dtype=it_ty, count=V,
                                   shared_space=_CTA_SS).bitcast(et)
            else:
                xv = nvvm.load_ext(mXi.iterator + (base + off), dtype=it_ty, count=V).bitcast(et)
                dv = nvvm.load_ext(mDYi.iterator + (base + off), dtype=it_ty, count=V).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xv[e].to(cutlass.Float32) - mn) * rs
                dxh = dv[e].to(cutlass.Float32) * gc[k * V + e]
                g_ = dxh - xh * c2
                if cutlass.const_expr(has_mean):
                    g_ = g_ - c1
                ys.append((rs * g_).to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                           mDXi.iterator + (base + off))
        row = row + nclusters

    # ---- flush per-column partials: the cluster writes one whole [C] row ----
    pbase = cutlass.Int64(cl) * C + c0
    for k in cutlass.range_constexpr(VPT):
        off = (tid + k * BT) * V
        for e in cutlass.range_constexpr(V):
            mDGp[pbase + off + e] = dg[k * V + e]
            if cutlass.const_expr(has_beta):
                mDBp[pbase + off + e] = db[k * V + e]


@cute.jit
def _ln_bwd_cga_host(
    mDY, mX, mDX, mG, mMean, mRstd, mDGp, mDBp, mDGamma, mDBeta, R, nclusters,
    C: cutlass.Constexpr, CS: cutlass.Constexpr, CGA: cutlass.Constexpr,
    BT: cutlass.Constexpr, V: cutlass.Constexpr, VPT: cutlass.Constexpr,
    KS: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    Cf: cutlass.Constexpr, has_mean: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    nctas: cutlass.Constexpr, FB: cutlass.Constexpr, fgrid: cutlass.Constexpr,
    nchunk: cutlass.Constexpr, CHUNK: cutlass.Constexpr, smem_bytes: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    mGi = cute.recast_tensor(mG, it_ty)
    _ln_bwd_cga_kernel(
        mDYi, mXi, mDXi, mGi, mMean, mRstd, mDGp, mDBp, R, nclusters,
        C, CS, CGA, BT, V, VPT, KS, it_ty, et, Cf, has_mean, has_beta,
    ).launch(grid=(nctas, 1, 1), block=(BT, 1, 1), cluster=(CGA, 1, 1), smem=smem_bytes)
    _ln_bwd_finalize_kernel(mDGp, mDBp, mDGamma, mDBeta, nclusters, C, FB, CHUNK, has_beta).launch(
        grid=(fgrid, nchunk, 1), block=(FB, 1, 1)
    )


_KCACHE = {}


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, params, knobs=None):
    """Launch the CGA row-split LN/RMS backward. ``dy2d``/``x2d`` are ``[R, C]``."""
    import torch

    R, C = int(spec.R), int(spec.M)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    cfg = cga_cfg(C, R, eb, _BT)
    if cfg is None:
        raise ValueError(f"CGA LayerNorm backward: C={C} not splittable")
    CGA, CS, V, VPT, KS = cfg
    if knobs is not None:
        KS = min(VPT, int(knobs))
    smem_bytes = 2 * KS * _BT * V * eb + 2 * 4 + 2 * (_BT // 32) * 4 + 128

    # One cluster per row up to a persistent cap, then grid-stride.
    nclusters = min(R, max(1, 4 * sm_count() // CGA))
    dx = torch.empty_like(x2d)
    dgamma = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.zeros(spec.gamma_len, dtype=torch.float32, device=x2d.device) if has_beta else dgamma
    dgp = torch.empty(nclusters * C, dtype=torch.float32, device=x2d.device)
    dbp = torch.empty(nclusters * C, dtype=torch.float32, device=x2d.device) if has_beta else dgp

    FB = 128 if C < 256 else 256
    fgrid = (C + FB - 1) // FB
    nchunk = max(1, min(32, 256 // fgrid))
    if nchunk > nclusters:
        nchunk = nclusters
    CHUNK = (nclusters + nchunk - 1) // nchunk

    args = (dyn(dy2d.reshape(-1)), dyn(x2d.reshape(-1)), dyn(dx.reshape(-1)), dyn(gamma),
            dyn(mean), dyn(rstd), dyn(dgp), dyn(dbp), dyn(dgamma), dyn(dbeta),
            cutlass.Int32(R), cutlass.Int32(nclusters))
    ce = (C, CS, CGA, _BT, V, VPT, KS, it_ty, et, float(C), bool(spec.has_mean), has_beta,
          nclusters * CGA, FB, fgrid, nchunk, CHUNK, smem_bytes)
    key = (params.io_dtype, C, R, CGA, KS, bool(spec.has_mean), has_beta, nclusters, nchunk, CHUNK)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_ln_bwd_cga_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
