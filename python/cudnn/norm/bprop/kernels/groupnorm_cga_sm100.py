# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroupNorm / InstanceNorm backward for long rows and few rows, sm_100.

:mod:`groupnorm_fast_sm100` is the good atomic-free backward, and it declines in two
situations that together leave the worst shapes on the ORIGINAL one-atomic-per-element
kernel at 0.00-0.01 of achievable:

* **The row does not fit shared memory.** It stages the whole row so the dx pass does
  not re-read global, which needs ``2*M*eb`` bytes -- 1.6 MB at ``M=401408``.
* **Too few rows.** Even when it fits, ``ctas`` is capped at ``R``, so GroupNorm with
  ``N=2, C=256, G=2`` would run 4 CTAs on 148 SMs.

This kernel keeps the thing that makes the fast kernel work -- the map that fixes a
CTA's channel set, so per-channel gradients stay register-local and no atomics are
needed -- and relaxes both limits:

* staging is **bounded**, ``KS`` vectors per thread under a byte budget, with the
  remainder re-read in pass 2;
* a row is split across a **cluster** of ``CGA`` CTAs along its span, with the
  row-wide ``sum(dxhat)`` / ``sum(dxhat*xhat)`` reduced through distributed shared
  memory.

The partials layout is what lets the EXISTING finalize be reused untouched. It sums
contributors at stride ``GPS`` over a ``[parts, CPG]`` buffer, which works because a
CTA's group index is ``cta % GPS``. Writing partition ``p = rank * ncta_rows + cl``
with ``ncta_rows`` a multiple of ``GPS`` preserves exactly that: ``p % GPS == cl % GPS``,
the group of every row that cluster slot touches. Every rank of every cluster lands in
the right residue class, so ``nparts = ncta_rows * CGA`` and nothing else changes.

``CGA == 1`` is a supported configuration, not a degenerate one: it covers the
row-too-big-but-enough-rows case, where the bounded staging alone is the fix.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn, sm_count, smem_capacity

from .groupnorm_fast_sm100 import _gn_bwd_finalize_kernel, _vec_for, red_scratch_len

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_MAX_CGA = 8
_STAGE_CAP = 64 * 1024  # bounded smem stage; ~3 CTAs/SM at this size
_FILL_TARGET = 4


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


def cga_cfg(spec, eb):
    """``(V, bt, CGA, ncta_rows, PER, KS, NVS)`` or None when this does not apply."""
    CPG = int(spec.channels_per_group)
    SPAN = int(spec.gamma_inner_span)
    GPS = int(spec.groups_per_sample)
    R = int(spec.R)
    if CPG < 1 or CPG > 1024 or SPAN < 1:
        return None
    V = _vec_for(SPAN, eb)
    if V < 2 or SPAN % V:  # scalar spans keep the existing path
        return None
    NVS = SPAN // V

    bt = min(256, max(32, ((max(1, int(spec.M) // V) + 31) // 32) * 32))
    if bt < CPG:
        bt = CPG
    if bt % CPG:
        bt = ((bt + CPG - 1) // CPG) * CPG
    bt = min(bt, 1024)
    if bt % 32:
        return None

    sm = sm_count()
    ncta_rows = max(GPS, (min(_FILL_TARGET * sm, R) // GPS) * GPS)
    if ncta_rows > R:
        ncta_rows = max(GPS, (R // GPS) * GPS) or GPS
    cga = 1
    while cga < _MAX_CGA and ncta_rows * cga < _FILL_TARGET * sm:
        cga *= 2
    # Each rank needs real work, else the extra CTAs are only barrier participants.
    while cga > 1 and (NVS + cga - 1) // cga < (bt // CPG):
        cga //= 2
    per = (NVS + cga - 1) // cga
    ks = min(per, max(1, _STAGE_CAP // (2 * bt * V * eb)))
    return V, bt, cga, ncta_rows, per, ks, NVS


def eligible(spec, eb):
    return cga_cfg(spec, eb) is not None


def should_use(spec, eb):
    """True only where groupnorm_fast_sm100 does badly or not at all.

    Where the fast kernel applies AND has rows to work with it stays ahead -- it
    stages the entire row, so its dx pass never returns to global. Measured on
    1024-row shapes: 0.44 against this kernel's 0.41, and 0.36 against 0.28. So this
    takes over only when the row will not fit that stage, or when there are too few
    rows to fill the machine (at R=32 it is 0.33 against 0.26).
    """
    from .groupnorm_fast_sm100 import eligible as _fast_eligible

    if not eligible(spec, eb):
        return False
    return (not _fast_eligible(spec, eb)) or int(spec.R) < _FILL_TARGET * sm_count()


@cute.kernel
def _gn_bwd_cga_kernel(
    mDYi,
    mXi,
    mDXi,
    mG,
    mMean,
    mRstd,
    mDGp,
    mDBp,
    R: cutlass.Int32,
    ncta_rows: cutlass.Int32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    CPG: cutlass.Constexpr,
    SPAN: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    CGA: cutlass.Constexpr,
    PER: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    NVS: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    TPS: cutlass.Constexpr = bt // CPG  # threads per channel segment
    Mf = cutlass.Float32(M)

    smem = SmemAllocator()
    part = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2), byte_alignment=16)
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(red_scratch_len(bt)), byte_alignment=8)
    seg_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * bt), byte_alignment=16)
    sX = smem.allocate_tensor(it_ty, cute.make_layout(KS * bt * V), byte_alignment=16)
    sD = smem.allocate_tensor(it_ty, cute.make_layout(KS * bt * V), byte_alignment=16)

    if cutlass.const_expr(CGA > 1):
        rank = nvvm.cluster_ctarank()
        cl = bid // CGA
    else:
        rank = cutlass.Int32(0)
        cl = bid

    seg = tid // TPS  # which channel of the group this thread serves
    lane = tid % TPS
    base_c = (cl % GPS) * CPG  # FIXED for this slot -- that is what kills the atomics
    g_seg = mG[base_c + seg].to(cutlass.Float32)

    v0 = rank * PER  # this CTA's vector range within one channel's span
    v1 = v0 + PER
    if v1 > NVS:
        v1 = NVS

    dg = cutlass.Float32(0.0)
    db = cutlass.Float32(0.0)

    row = cl
    while row < R:
        mean = cutlass.Float32(0.0)
        if cutlass.const_expr(has_mean):
            mean = mMean[row]
        rstd = mRstd[row]
        base = cutlass.Int64(row) * M

        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)

        # ---- pass 1: row-wide reduction + this thread's channel sums ----
        kv = v0 + lane
        ks = cutlass.Int32(0)
        while kv < v1:
            off = seg * SPAN + kv * V
            rx = nvvm.load_ext(mXi.iterator + (base + off), dtype=it_ty, count=V)
            rd = nvvm.load_ext(mDYi.iterator + (base + off), dtype=it_ty, count=V)
            xv = rx.bitcast(et)
            dv = rd.bitcast(et)
            for e in cutlass.range_constexpr(V):
                xh = (xv[e].to(cutlass.Float32) - mean) * rstd
                dy = dv[e].to(cutlass.Float32)
                dxh = dy * g_seg
                s1 = s1 + dxh
                s2 = s2 + dxh * xh
                dg = dg + dy * xh
                db = db + dy
            if ks < KS:
                nvvm.store_ext(rx, sX.iterator + (tid + ks * bt) * V, shared_space=_CTA_SS)
                nvvm.store_ext(rd, sD.iterator + (tid + ks * bt) * V, shared_space=_CTA_SS)
            ks = ks + 1
            kv = kv + TPS

        s1, s2 = _block_sum2(s1, s2, tid, red, bt)
        if cutlass.const_expr(CGA > 1):
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
            # Peers must finish reading before the next row overwrites `part`.
            nvvm.barrier_cluster_arrive_aligned()
            nvvm.barrier_cluster_wait_aligned()
            s1 = t1
            s2 = t2
        a = s1 / Mf
        b = s2 / Mf

        # ---- pass 2: dx, from the bounded stage where it reaches ----
        kv = v0 + lane
        ks = cutlass.Int32(0)
        while kv < v1:
            off = seg * SPAN + kv * V
            # Both branches carry the whole body: a value assigned inside a staged
            # `if` does not escape it, so a shared tail reading `xv` would not compile.
            if ks < KS:
                xv = nvvm.load_ext(sX.iterator + (tid + ks * bt) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                dv = nvvm.load_ext(sD.iterator + (tid + ks * bt) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
                ys = []
                for e in cutlass.range_constexpr(V):
                    xh = (xv[e].to(cutlass.Float32) - mean) * rstd
                    dxh = dv[e].to(cutlass.Float32) * g_seg
                    ys.append((rstd * (dxh - a - xh * b)).to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + off))
            else:
                xg = nvvm.load_ext(mXi.iterator + (base + off), dtype=it_ty, count=V).bitcast(et)
                dg_ = nvvm.load_ext(mDYi.iterator + (base + off), dtype=it_ty, count=V).bitcast(et)
                zs = []
                for e in cutlass.range_constexpr(V):
                    xh = (xg[e].to(cutlass.Float32) - mean) * rstd
                    dxh = dg_[e].to(cutlass.Float32) * g_seg
                    zs.append((rstd * (dxh - a - xh * b)).to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(zs), et).bitcast(it_ty), mDXi.iterator + (base + off))
            ks = ks + 1
            kv = kv + TPS
        nvvm.barrier_cta_sync_aligned(0)
        row = row + ncta_rows

    # ---- flush: reduce each channel segment's TPS threads, write [parts, CPG] ----
    # p = rank*ncta_rows + cl keeps p % GPS == cl % GPS, which is what the shared
    # finalize relies on when it sums contributors at stride GPS.
    seg_red[tid] = dg
    seg_red[bt + tid] = db
    nvvm.barrier_cta_sync_aligned(0)
    if tid < CPG:
        agg = cutlass.Float32(0.0)
        aggb = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(TPS):
            agg = agg + seg_red[tid * TPS + k]
            aggb = aggb + seg_red[bt + tid * TPS + k]
        p = rank * ncta_rows + cl
        mDGp[p * CPG + tid] = agg
        if cutlass.const_expr(has_beta):
            mDBp[p * CPG + tid] = aggb


_gn_bwd_cga_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _gn_bwd_cga_host(
    mDY,
    mX,
    mDX,
    mG,
    mMean,
    mRstd,
    mDGp,
    mDBp,
    mDGamma,
    mDBeta,
    R,
    ncta_rows,
    nparts,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    CPG: cutlass.Constexpr,
    SPAN: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    CGA: cutlass.Constexpr,
    PER: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    NVS: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    nctas: cutlass.Constexpr,
    smem_bytes: cutlass.Constexpr,
    fbt: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    _gn_bwd_cga_kernel(
        mDYi,
        mXi,
        mDXi,
        mG,
        mMean,
        mRstd,
        mDGp,
        mDBp,
        R,
        ncta_rows,
        M,
        V,
        CPG,
        SPAN,
        GPS,
        bt,
        CGA,
        PER,
        KS,
        NVS,
        it_ty,
        et,
        has_mean,
        has_beta,
    ).launch(grid=(nctas, 1, 1), block=(bt, 1, 1), smem=smem_bytes, cluster=(CGA, 1, 1))
    _gn_bwd_finalize_kernel(mDGp, mDBp, mDGamma, mDBeta, nparts, CPG, GPS, fbt, has_beta).launch(grid=(fgrid, 1, 1), block=(fbt, 1, 1))


_KCACHE = {}


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, params, knobs=None):
    """Launch the long-row / few-row GN/IN backward. Returns ``(dx, dgamma, dbeta)``."""
    import torch

    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    cfg = cga_cfg(spec, eb)
    if cfg is None:
        raise ValueError("CGA GroupNorm backward: shape not supported")
    V, bt, CGA, ncta_rows, PER, KS, NVS = cfg
    if knobs is not None:
        KS = min(PER, int(knobs))
    M = int(spec.M)
    CPG = int(spec.channels_per_group)
    SPAN = int(spec.gamma_inner_span)
    GPS = int(spec.groups_per_sample)
    R = int(spec.R)

    nparts = ncta_rows * CGA
    dx = torch.empty_like(x2d)
    dgamma = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dgp = torch.zeros(nparts * CPG, dtype=torch.float32, device=x2d.device)
    dbp = torch.zeros(nparts * CPG, dtype=torch.float32, device=x2d.device)
    if mean is None:
        mean = rstd

    smem_bytes = 2 * KS * bt * V * eb + 2 * 4 + red_scratch_len(bt) * 4 + 2 * bt * 4 + 256
    assert smem_bytes <= smem_capacity(), "stage exceeds this architecture's shared memory"
    fbt = 128
    fgrid = (GPS * CPG + fbt - 1) // fbt

    args = (
        dyn(dy2d.reshape(-1)),
        dyn(x2d.reshape(-1)),
        dyn(dx.reshape(-1)),
        dyn(gamma),
        dyn(mean),
        dyn(rstd),
        dyn(dgp),
        dyn(dbp),
        dyn(dgamma),
        dyn(dbeta),
        cutlass.Int32(R),
        cutlass.Int32(ncta_rows),
        cutlass.Int32(nparts),
    )
    ce = (M, V, CPG, SPAN, GPS, bt, CGA, PER, KS, NVS, it_ty, et, bool(spec.has_mean), has_beta, nparts, smem_bytes, fbt, fgrid)
    key = (params.io_dtype, M, CPG, SPAN, GPS, R, bt, CGA, PER, KS, bool(spec.has_mean), has_beta)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_gn_bwd_cga_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
