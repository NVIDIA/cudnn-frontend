# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GroupNorm / InstanceNorm forward with the row split across a CGA, sm_100.

The rowwise forward launches ``grid=(R, 1, 1)`` -- one CTA per ``(sample, group)``.
That is fine when ``R = N*G`` is large, and collapses when it is not: GroupNorm with
``N=2, C=256, G=2`` has FOUR rows, so four CTAs run on 148 SMs and the measured
utilisation is 0.01 of the achievable copy. The rows are not small -- each is 401408
elements -- so the work is there; it simply has nowhere to go.

This is the same shape of problem the LayerNorm forward had, and the same fix: give
one row to a CLUSTER of ``CGA`` CTAs, each reducing its ``M/CGA`` slice, publishing a
single ``(sum, sum_sq)`` pair, and reading every peer's pair back through distributed
shared memory. The grid becomes ``R*CGA``, which is what makes the machine reachable
at all for small ``R``.

A cluster is at most 8 CTAs, so the split itself buys at most 8x the grid. The
measured gains run higher than that -- 17x at ``N=2, C=256, G=2`` -- because the
generic rowwise kernel this replaces is also carrying a staged inner loop sized for
a different regime, so the win is the split AND a plainer vectorised pass. The split
is what makes it reachable; do not read 17x as a parallelism factor.

Engaged only while ``R`` cannot fill the machine. Above that the rowwise kernel has
the parallelism already and the split would only add a barrier -- at 1024 rows it
runs 0.45 and is left alone.

The affine index is the one difference from the LayerNorm version: the channel for
element ``j`` of row ``r`` is ``(r % gps) * cpg + j // span``, so gamma/beta are
gathered per element rather than held per column. ``j`` is the GLOBAL element index
within the row, not the slice-local one.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn, sm_count

_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_BT = 256
_MAX_CGA = 8
_KC = 4
# Engage only when the one-CTA-per-row grid cannot fill the machine. Above this the
# rowwise kernel already has the parallelism and the split would only add a barrier.
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


def cga_cfg(M, R, eb, block_threads=_BT):
    """``(CGA, NVEC, V, PER)`` for a rowwise split, or None when it does not apply.

    The split is over 128-bit VECTORS with a bounds-checked grid stride, not over an
    exact division of the row. Requiring ``M % (BT*V) == 0`` instead would reject most
    real shapes outright -- GroupNorm rows are ``cpg * H*W``, and 12544, 784 and 196
    are none of them multiples of 2048.
    """
    V = 16 // eb
    if M % V:  # rows must still be 128-bit vectorisable
        return None
    sm = sm_count()
    if R >= _FILL_TARGET * sm:  # the rowwise grid already fills the machine
        return None
    nvec = M // V
    cga = 2
    while cga < _MAX_CGA and R * cga < _FILL_TARGET * sm:
        cga *= 2
    # Each CTA should still get at least one full vector pass, else the extra CTAs
    # are just barrier participants.
    while cga > 1 and (nvec + cga - 1) // cga < block_threads:
        cga //= 2
    if cga < 2:
        return None
    per = (nvec + cga - 1) // cga
    return cga, nvec, V, per


def eligible(M, R, eb, block_threads=_BT):
    return cga_cfg(M, R, eb, block_threads) is not None


@cute.kernel
def _gn_fwd_cga_kernel(
    mXi,
    mYi,
    mG,
    mB,
    mMean,
    mRstd,
    gps: cutlass.Int32,
    cpg: cutlass.Int32,
    span: cutlass.Int32,
    M: cutlass.Constexpr,
    NVEC: cutlass.Constexpr,
    CGA: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    V: cutlass.Constexpr,
    PER: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    nwarps: cutlass.Constexpr = BT // 32

    smem = SmemAllocator()
    # Same offset in every CTA, so mapa of this pointer lands on the peer's copy.
    part = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2), byte_alignment=16)
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * nwarps), byte_alignment=16)

    rank = nvvm.cluster_ctarank()
    row = bx // CGA
    rbase = cutlass.Int64(row) * M
    base_c = (row % gps) * cpg
    v0 = rank * PER  # this CTA's first VECTOR within the row
    v1 = v0 + PER
    if v1 > NVEC:
        v1 = NVEC

    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    xc = [cutlass.Float32(0.0)] * (KC * V)

    for k in cutlass.range_constexpr(KC):
        iv = v0 + tid + k * BT
        if iv < v1:
            xv = nvvm.load_ext(mXi.iterator + (rbase + iv * V), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                s1 = s1 + x
                s2 = s2 + x * x
                xc[k * V + e] = x
    iv = v0 + tid + KC * BT
    while iv < v1:
        xv = nvvm.load_ext(mXi.iterator + (rbase + iv * V), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x
        iv = iv + BT

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
    # No CTA may exit while a peer is still reading its shared memory.
    nvvm.barrier_cluster_arrive_aligned()
    nvvm.barrier_cluster_wait_aligned()

    if cutlass.const_expr(has_mean):
        mean = t1 / Mf
        rstd = cute.math.rsqrt(t2 / Mf - mean * mean + eps)
    else:
        mean = cutlass.Float32(0.0)
        rstd = cute.math.rsqrt(t2 / Mf + eps)
    if rank == 0 and tid == 0:
        if cutlass.const_expr(has_mean):
            mMean[row] = mean
        mRstd[row] = rstd

    for k in cutlass.range_constexpr(KC):
        iv = v0 + tid + k * BT
        if iv < v1:
            ys = []
            for e in cutlass.range_constexpr(V):
                c = base_c + ((iv * V + e) // span)
                y = mG[c].to(cutlass.Float32) * ((xc[k * V + e] - mean) * rstd)
                if cutlass.const_expr(has_beta):
                    y = y + mB[c].to(cutlass.Float32)
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (rbase + iv * V))
    iv = v0 + tid + KC * BT
    while iv < v1:
        xv = nvvm.load_ext(mXi.iterator + (rbase + iv * V), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            c = base_c + ((iv * V + e) // span)
            y = mG[c].to(cutlass.Float32) * ((xv[e].to(cutlass.Float32) - mean) * rstd)
            if cutlass.const_expr(has_beta):
                y = y + mB[c].to(cutlass.Float32)
            ys.append(y.to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (rbase + iv * V))
        iv = iv + BT


_gn_fwd_cga_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _gn_fwd_cga_host(
    mX,
    mY,
    mG,
    mB,
    mMean,
    mRstd,
    gps,
    cpg,
    span,
    M: cutlass.Constexpr,
    NVEC: cutlass.Constexpr,
    CGA: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    V: cutlass.Constexpr,
    PER: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Mf: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    nctas: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _gn_fwd_cga_kernel(
        mXi,
        mYi,
        mG,
        mB,
        mMean,
        mRstd,
        gps,
        cpg,
        span,
        M,
        NVEC,
        CGA,
        BT,
        V,
        PER,
        KC,
        it_ty,
        et,
        Mf,
        eps,
        has_mean,
        has_beta,
    ).launch(grid=(nctas, 1, 1), block=(BT, 1, 1), cluster=(CGA, 1, 1))


_KCACHE = {}


def forward(spec, x2d, gamma, beta, *, eps, params, knobs=None):
    """Launch the CGA row-split GN/IN forward. ``x2d`` is ``[R, M]``."""
    import torch

    R, M = int(spec.R), int(spec.M)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    cfg = cga_cfg(M, R, eb, _BT)
    if cfg is None:
        raise ValueError(f"CGA GroupNorm forward: M={M} R={R} not splittable")
    CGA, NVEC, V, PER = cfg
    KC = _KC if knobs is None else int(knobs)
    has_beta = beta is not None
    if beta is None:
        beta = gamma

    y = torch.empty_like(x2d)
    rstd = torch.empty(R, dtype=torch.float32, device=x2d.device)
    mean = torch.empty(R, dtype=torch.float32, device=x2d.device) if spec.has_mean else rstd

    args = (
        dyn(x2d.reshape(-1)),
        dyn(y.reshape(-1)),
        dyn(gamma),
        dyn(beta),
        dyn(mean),
        dyn(rstd),
        cutlass.Int32(spec.groups_per_sample),
        cutlass.Int32(spec.channels_per_group),
        cutlass.Int32(spec.gamma_inner_span),
    )
    ce = (M, NVEC, CGA, _BT, V, PER, KC, it_ty, et, float(M), float(eps), bool(spec.has_mean), has_beta, R * CGA)
    key = (params.io_dtype, M, R, CGA, KC, bool(spec.has_mean), has_beta)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_gn_fwd_cga_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
