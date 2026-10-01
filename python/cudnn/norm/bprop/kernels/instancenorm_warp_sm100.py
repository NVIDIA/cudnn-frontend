# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp-per-row InstanceNorm backward, sm_100, CUTLASS primitives.

InstanceNorm is GroupNorm with ``channels_per_group == 1``: a rowwise row is one
``(sample, channel)`` pair of ``M = H*W`` elements. The block-per-row backward
(``groupnorm_fast_sm100``) therefore spends a whole CTA and a block reduce -- two
barriers through shared memory -- on a row that can be as short as 49 elements.
At 7x7 that is almost all overhead: measured 0.11-0.17 of achievable.

Here a WARP owns a row, so:

  * the row reduction is a butterfly shuffle -- **no barriers and no shared memory**;
  * a block runs ``WPB`` rows concurrently, so one row's reduction latency is hidden
    by its neighbours' loads instead of stalling the whole CTA.

The atomic-free channel map carries over. Warp ``gw = bid*WPB + w`` walks rows
``gw, gw + W, gw + 2W, ...`` where ``W`` is the total warp count; if ``W`` is a
multiple of ``gps`` then every row a warp sees has the same ``r % gps``, so the
warp's channel is FIXED and its ``dgamma``/``dbeta`` live in two registers. The
partials are ``[W]`` -- one float per warp -- and a tiny finalize sums the ``W/gps``
warps that share a channel. No atomics anywhere.

``KREG`` of each lane's vectors are kept in registers across the two passes, so a
short row is read exactly once.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn

_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_WPB = 8
_MAX_KREG = 8
_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


@cute.jit
def _warp_sum2(a, b):
    for d in cutlass.range_constexpr(5):
        off = 1 << d
        a = a + nvvm.shfl_sync(_FULL, a, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
        b = b + nvvm.shfl_sync(_FULL, b, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
    return a, b


@cute.kernel
def _in_bwd_warp_kernel(
    mDY,
    mDYi,
    mX,
    mXi,
    mDX,
    mDXi,
    mG,
    mMean,
    mRstd,
    mDGp,
    mDBp,
    R: cutlass.Int32,
    W: cutlass.Int32,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    WPB: cutlass.Constexpr,
    KREG: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    NV: cutlass.Constexpr = M // V
    Mf = cutlass.Float32(M)
    w = tid // 32
    lane = tid % 32
    gw = bid * WPB + w
    ch = gw % GPS  # FIXED for this warp -- that is what removes the atomics
    g_c = mG[ch].to(cutlass.Float32)

    dg = cutlass.Float32(0.0)
    db = cutlass.Float32(0.0)

    row = gw
    while row < R:
        mean = cutlass.Float32(0.0)
        if cutlass.const_expr(has_mean):
            mean = mMean[row]
        rstd = mRstd[row]
        base = cutlass.Int64(row) * M

        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)
        xr = [cutlass.Float32(0.0)] * (KREG * V)
        dr = [cutlass.Float32(0.0)] * (KREG * V)

        # ---- pass 1: register-cache the first KREG lane-steps, stream the rest ----
        for k in cutlass.range_constexpr(KREG):
            kv = k * 32 + lane
            if kv < NV:
                if cutlass.const_expr(V == 1):
                    xs = [mX[base + kv].to(cutlass.Float32)]
                    ds = [mDY[base + kv].to(cutlass.Float32)]
                else:
                    xs = [e.to(cutlass.Float32) for e in nvvm.load_ext(mXi.iterator + (base + kv * V), dtype=it_ty, count=V).bitcast(et)]
                    ds = [e.to(cutlass.Float32) for e in nvvm.load_ext(mDYi.iterator + (base + kv * V), dtype=it_ty, count=V).bitcast(et)]
                for e in cutlass.range_constexpr(V):
                    xh = (xs[e] - mean) * rstd
                    dxh = ds[e] * g_c
                    s1 = s1 + dxh
                    s2 = s2 + dxh * xh
                    dg = dg + ds[e] * xh
                    db = db + ds[e]
                    xr[k * V + e] = xs[e]
                    dr[k * V + e] = ds[e]
        kv = KREG * 32 + lane
        while kv < NV:
            if cutlass.const_expr(V == 1):
                xs = [mX[base + kv].to(cutlass.Float32)]
                ds = [mDY[base + kv].to(cutlass.Float32)]
            else:
                xs = [e.to(cutlass.Float32) for e in nvvm.load_ext(mXi.iterator + (base + kv * V), dtype=it_ty, count=V).bitcast(et)]
                ds = [e.to(cutlass.Float32) for e in nvvm.load_ext(mDYi.iterator + (base + kv * V), dtype=it_ty, count=V).bitcast(et)]
            for e in cutlass.range_constexpr(V):
                xh = (xs[e] - mean) * rstd
                dxh = ds[e] * g_c
                s1 = s1 + dxh
                s2 = s2 + dxh * xh
                dg = dg + ds[e] * xh
                db = db + ds[e]
            kv = kv + 32

        s1, s2 = _warp_sum2(s1, s2)  # butterfly only -- no barrier, no shared memory
        a = s1 / Mf
        b = s2 / Mf

        # ---- pass 2 ----
        for k in cutlass.range_constexpr(KREG):
            kv = k * 32 + lane
            if kv < NV:
                ys = []
                for e in cutlass.range_constexpr(V):
                    xh = (xr[k * V + e] - mean) * rstd
                    ys.append((rstd * (dr[k * V + e] * g_c - a - xh * b)).to(et))
                if cutlass.const_expr(V == 1):
                    mDX[base + kv] = ys[0]
                else:
                    nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + kv * V))
        kv = KREG * 32 + lane
        while kv < NV:
            if cutlass.const_expr(V == 1):
                xs = [mX[base + kv].to(cutlass.Float32)]
                ds = [mDY[base + kv].to(cutlass.Float32)]
            else:
                xs = [e.to(cutlass.Float32) for e in nvvm.load_ext(mXi.iterator + (base + kv * V), dtype=it_ty, count=V).bitcast(et)]
                ds = [e.to(cutlass.Float32) for e in nvvm.load_ext(mDYi.iterator + (base + kv * V), dtype=it_ty, count=V).bitcast(et)]
            ys = []
            for e in cutlass.range_constexpr(V):
                xh = (xs[e] - mean) * rstd
                ys.append((rstd * (ds[e] * g_c - a - xh * b)).to(et))
            if cutlass.const_expr(V == 1):
                mDX[base + kv] = ys[0]
            else:
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mDXi.iterator + (base + kv * V))
            kv = kv + 32
        row = row + W

    dg, db = _warp_sum2(dg, db)
    if lane == 0:
        mDGp[gw] = dg
        if cutlass.const_expr(has_beta):
            mDBp[gw] = db


@cute.kernel
def _in_bwd_finalize_kernel(
    mDGp,
    mDBp,
    mDGamma,
    mDBeta,
    W: cutlass.Int32,
    GPS: cutlass.Constexpr,
    bt: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    c = bid * bt + tid
    if c < GPS:
        sg = cutlass.Float32(0.0)
        sb = cutlass.Float32(0.0)
        j = cutlass.Int32(c)
        while j < W:
            sg = sg + mDGp[j]
            if cutlass.const_expr(has_beta):
                sb = sb + mDBp[j]
            j = j + GPS
        mDGamma[c] = sg
        if cutlass.const_expr(has_beta):
            mDBeta[c] = sb


@cute.jit
def _in_bwd_warp_host(
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
    W,
    M: cutlass.Constexpr,
    V: cutlass.Constexpr,
    GPS: cutlass.Constexpr,
    WPB: cutlass.Constexpr,
    KREG: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    ctas: cutlass.Constexpr,
    fgrid: cutlass.Constexpr,
    fbt: cutlass.Constexpr,
) -> None:
    mDYi = cute.recast_tensor(mDY, it_ty)
    mXi = cute.recast_tensor(mX, it_ty)
    mDXi = cute.recast_tensor(mDX, it_ty)
    _in_bwd_warp_kernel(
        mDY,
        mDYi,
        mX,
        mXi,
        mDX,
        mDXi,
        mG,
        mMean,
        mRstd,
        mDGp,
        mDBp,
        R,
        W,
        M,
        V,
        GPS,
        WPB,
        KREG,
        it_ty,
        et,
        has_mean,
        has_beta,
    ).launch(grid=(ctas, 1, 1), block=(WPB * 32, 1, 1))
    _in_bwd_finalize_kernel(mDGp, mDBp, mDGamma, mDBeta, W, GPS, fbt, has_beta).launch(grid=(fgrid, 1, 1), block=(fbt, 1, 1))


_KCACHE = {}


def _vec_for(M, eb):
    v = 16 // eb
    while v > 1 and M % v != 0:
        v //= 2
    return v


def eligible(spec, elem_bytes):
    """InstanceNorm only (one channel per row), and short enough rows that the
    block-per-row kernel's barriers dominate."""
    if int(spec.channels_per_group) != 1:
        return False
    M = int(spec.M)
    V = _vec_for(M, elem_bytes)
    return (M // V + 31) // 32 <= _MAX_KREG


def backward(spec, dy2d, x2d, gamma, mean, rstd, *, has_beta, cfg, params):
    """Launch the warp-per-row InstanceNorm backward."""
    import torch

    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    M = int(spec.M)
    V = _vec_for(M, eb)
    GPS = int(spec.groups_per_sample)
    R = int(spec.R)
    NV = M // V
    KREG = min(_MAX_KREG, (NV + 31) // 32)
    WPB = _WPB

    # Total warps must be a multiple of gps so each warp's channel is fixed.
    want = max(1, min(4 * _nsm() * WPB, R))
    W = max(GPS, (want // GPS) * GPS)
    ctas = (W + WPB - 1) // WPB
    W = ctas * WPB
    if W % GPS:
        W = ((W + GPS - 1) // GPS) * GPS
        ctas = (W + WPB - 1) // WPB
        W = ctas * WPB

    dx = torch.empty_like(x2d)
    dgamma = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dbeta = torch.empty(spec.gamma_len, dtype=torch.float32, device=x2d.device)
    dgp = torch.zeros(W, dtype=torch.float32, device=x2d.device)
    dbp = torch.zeros(W, dtype=torch.float32, device=x2d.device)
    if mean is None:
        mean = rstd
    dyf, xf, dxf = dy2d.reshape(-1), x2d.reshape(-1), dx.reshape(-1)

    fbt = 128
    fgrid = (GPS + fbt - 1) // fbt
    args = (dyn(dyf), dyn(xf), dyn(dxf), dyn(gamma), dyn(mean), dyn(rstd), dyn(dgp), dyn(dbp), dyn(dgamma), dyn(dbeta), cutlass.Int32(R), cutlass.Int32(W))
    ce = (M, V, GPS, WPB, KREG, it_ty, et, spec.has_mean, has_beta, ctas, fgrid, fbt)
    key = (params.io_dtype, M, V, GPS, WPB, KREG, spec.has_mean, has_beta, ctas)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_in_bwd_warp_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return dx, dgamma, (dbeta if has_beta else None)
