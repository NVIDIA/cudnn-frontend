# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LayerNorm / RMSNorm forward with the row split across a CGA, sm_100.

One CTA per row is the basic design and it runs out of road twice as the row grows.
The row is staged in one CTA's shared memory, so past ~232 KB it cannot be staged at
all; and parallelism is capped at one CTA per row, so a long row means few CTAs and
a thread holding ever more of it. Measured on the single-CTA path at a fixed 512 rows
(bf16, fraction of the achievable copy): 0.64 at D=8192, 0.59 at 16384, 0.28 at
32768, and past the smem wall the streaming fallback sits at ~0.17.

So split ONE row across ``CGA`` CTAs of a cluster. Each CTA reduces its own
``CS = C/CGA`` slice, publishes a single ``(sum, sum_sq)`` pair into shared memory,
and every CTA then reads all ``CGA`` pairs through DISTRIBUTED shared memory -- an
``mapa.shared::cluster`` translation of its own ``part`` pointer into each peer --
and finalises redundantly. No global round-trip, no grid barrier, and the cluster
barrier is hardware rather than an atomic spin.

Two consequences beyond speed: the slice is ``CGA`` times smaller, so a row that
could never be staged now fits comfortably (D=65536 bf16 at CGA=8 is a 16 KB slice);
and the grid becomes ``R*CGA`` CTAs instead of ``R``.

**DSMEM access is by INDEXING the mapped pointer** (``peer[0]``). ``nvvm.load_ext``
fails IR verification against an addrspace-7 pointer whatever ``shared_space`` says,
which is the one mechanical surprise here.

The trailing cluster barrier is not optional: a CTA must not exit while a peer is
still reading its shared memory, and dropping it is a use-after-free that will not
reproduce reliably.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS
from cudnn.norm.utils import dyn

_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F
_BT = 256
_MAX_CGA = 8  # portable cluster size; 16 is sm_100-only and needs an opt-in launch
_TARGET_SLICE = 8192  # elements per CTA: where the single-CTA path still runs well
_KC = 4  # vectors per thread cached in registers across the two passes


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


def _pick_cga(C, R, V, bt):
    """Cluster size for an ``R x C`` problem.

    The optimum depends on BOTH dimensions, which a slice-size heuristic cannot
    express. Measured util (bf16, fraction of achievable copy; * = chosen):

        R=64    D=16384  CGA2 0.47  CGA4 0.56*  CGA8 0.45
        R=64    D=131072 CGA2 0.29  CGA4 0.45   CGA8 0.62*
        R=512   D=32768  CGA2 0.62* CGA4 0.59   CGA8 0.41
        R=4096  D=65536  CGA2 0.74  CGA4 0.79*  CGA8 0.53

    Few rows need a WIDER split just to fill the machine; many rows prefer a narrow
    one, because every extra CTA in the cluster is another participant in the
    barrier. Three terms, in order:

    1. widen until the grid is a few waves deep (``R*CGA`` CTAs over the SMs);
    2. but never past ``VPT < 2`` -- slicing below two vectors per thread adds
       cluster-barrier cost without reducing the pass-2 re-read it is meant to buy;
    3. a long row takes one more split than occupancy alone suggests, since pass 2
       re-reads whatever the register cache could not hold.

    Reproduces the best measured cluster size in 11 of 12 cells (worst 0.93x).
    """
    from cudnn.norm.utils import sm_count

    sm = sm_count()
    cga = 2
    while cga < _MAX_CGA and R * cga < 4 * sm:
        cga *= 2
    while cga > 2 and (C // cga) < 2 * bt * V:
        cga //= 2
    if C >= 32 * bt * V:
        cga = max(cga, 4)
    return cga


def cga_cfg(C, R, eb, block_threads=_BT):
    """``(CGA, CS, V, VPT)`` for a row split across a cluster, or None."""
    V = 16 // eb
    lanes = block_threads * V  # elements one CTA covers per vector pass
    if C % lanes:
        return None
    cga = _pick_cga(C, R, V, block_threads)
    while cga > 1 and (C % (cga * lanes) or (C // cga) < lanes):
        cga //= 2
    if cga < 2:
        return None
    cs = C // cga
    return cga, cs, V, cs // lanes


# Below ~16K elements the single-CTA warp kernel is still ahead (0.70 vs 0.56 at
# D=4096): splitting a short row only buys a cluster barrier. At 16K they tie, and
# above it the warp map falls off a cliff -- 0.29 at 32768, 0.17 at 65536 -- while
# the split holds 0.6-0.8. So that is where this takes over.
_CGA_MIN_C = 16384


def eligible(C, R, eb, block_threads=_BT):
    return cga_cfg(C, R, eb, block_threads) is not None


def should_use(C, R, eb, block_threads=_BT):
    """True when the row is long enough that splitting it across a cluster wins."""
    return C >= _CGA_MIN_C and eligible(C, R, eb, block_threads)


@cute.kernel
def _ln_fwd_cga_kernel(
    mXi,
    mYi,
    mG,
    mB,
    mMean,
    mRstd,
    C: cutlass.Constexpr,
    CS: cutlass.Constexpr,
    CGA: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    V: cutlass.Constexpr,
    VPT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Cf: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    nwarps: cutlass.Constexpr = BT // 32

    smem = SmemAllocator()
    # `part` must sit at the same offset in every CTA -- mapa translates THIS CTA's
    # pointer into a peer, so the layouts have to agree. Identical allocation order
    # gives that for free.
    part = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2), byte_alignment=16)
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * nwarps), byte_alignment=16)

    rank = nvvm.cluster_ctarank()
    row = bx // CGA
    base = cutlass.Int64(row) * C + rank * CS

    s1 = cutlass.Float32(0.0)
    s2 = cutlass.Float32(0.0)
    xc = [cutlass.Float32(0.0)] * (KC * V)

    # ---- pass 1: this CTA's slice, first KC vectors kept in registers ----
    for k in cutlass.range_constexpr(KC):
        if cutlass.const_expr(k < VPT):
            xv = nvvm.load_ext(mXi.iterator + (base + (tid + k * BT) * V), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                s1 = s1 + x
                s2 = s2 + x * x
                xc[k * V + e] = x
    for k in cutlass.range_constexpr(VPT - KC if VPT > KC else 0):
        xv = nvvm.load_ext(mXi.iterator + (base + (tid + (KC + k) * BT) * V), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s1 = s1 + x
            s2 = s2 + x * x

    s1, s2 = _block_sum2(s1, s2, tid, red, BT)
    if tid == 0:
        part[0] = s1
        part[1] = s2
    nvvm.barrier_cta_sync_aligned(0)

    # ---- cross-CTA reduce through distributed shared memory ----
    nvvm.fence_sc_cluster()
    nvvm.barrier_cluster_arrive_aligned()
    nvvm.barrier_cluster_wait_aligned()
    t1 = cutlass.Float32(0.0)
    t2 = cutlass.Float32(0.0)
    for r in cutlass.range_constexpr(CGA):
        peer = nvvm.mapa(part.iterator, cutlass.Int32(r))
        t1 = t1 + peer[0]  # indexing, not load_ext: see module docstring
        t2 = t2 + peer[1]
    # Every CTA must finish READING peer smem before any CTA may exit.
    nvvm.barrier_cluster_arrive_aligned()
    nvvm.barrier_cluster_wait_aligned()

    if cutlass.const_expr(has_mean):
        mean = t1 / Cf
        rstd = cute.math.rsqrt(t2 / Cf - mean * mean + eps)
    else:
        mean = cutlass.Float32(0.0)
        rstd = cute.math.rsqrt(t2 / Cf + eps)
    if rank == 0 and tid == 0:
        if cutlass.const_expr(has_mean):
            mMean[row] = mean
        mRstd[row] = rstd

    # ---- pass 2 ----
    for k in cutlass.range_constexpr(KC):
        if cutlass.const_expr(k < VPT):
            off = (tid + k * BT) * V
            gv = nvvm.load_ext(mG.iterator + (rank * CS + off), dtype=it_ty, count=V).bitcast(et)
            bv = nvvm.load_ext(mB.iterator + (rank * CS + off), dtype=it_ty, count=V).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(V):
                y = (xc[k * V + e] - mean) * rstd * gv[e].to(cutlass.Float32)
                if cutlass.const_expr(has_beta):
                    y = y + bv[e].to(cutlass.Float32)
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (base + off))
    for k in cutlass.range_constexpr(VPT - KC if VPT > KC else 0):
        off = (tid + (KC + k) * BT) * V
        xv = nvvm.load_ext(mXi.iterator + (base + off), dtype=it_ty, count=V).bitcast(et)
        gv = nvvm.load_ext(mG.iterator + (rank * CS + off), dtype=it_ty, count=V).bitcast(et)
        bv = nvvm.load_ext(mB.iterator + (rank * CS + off), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            y = (xv[e].to(cutlass.Float32) - mean) * rstd * gv[e].to(cutlass.Float32)
            if cutlass.const_expr(has_beta):
                y = y + bv[e].to(cutlass.Float32)
            ys.append(y.to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (base + off))


_ln_fwd_cga_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _ln_fwd_cga_host(
    mX,
    mY,
    mG,
    mB,
    mMean,
    mRstd,
    C: cutlass.Constexpr,
    CS: cutlass.Constexpr,
    CGA: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    V: cutlass.Constexpr,
    VPT: cutlass.Constexpr,
    KC: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr,
    Cf: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    has_mean: cutlass.Constexpr,
    has_beta: cutlass.Constexpr,
    nctas: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    mGi = cute.recast_tensor(mG, it_ty)
    mBi = cute.recast_tensor(mB, it_ty)
    _ln_fwd_cga_kernel(
        mXi,
        mYi,
        mGi,
        mBi,
        mMean,
        mRstd,
        C,
        CS,
        CGA,
        BT,
        V,
        VPT,
        KC,
        it_ty,
        et,
        Cf,
        eps,
        has_mean,
        has_beta,
    ).launch(grid=(nctas, 1, 1), block=(BT, 1, 1), cluster=(CGA, 1, 1))


_KCACHE = {}


def forward(spec, x2d, gamma, beta, *, eps, params, knobs=None):
    """Launch the CGA row-split LN/RMS forward. ``x2d`` is ``[R, C]``."""
    import torch

    R, C = int(spec.R), int(spec.M)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    cfg = cga_cfg(C, R, eb, _BT)
    if cfg is None:
        raise ValueError(f"CGA LayerNorm forward: C={C} not splittable")
    CGA, CS, V, VPT = cfg
    KC = _KC if knobs is None else int(knobs)
    has_beta = beta is not None
    if beta is None:
        beta = gamma

    y = torch.empty_like(x2d)
    rstd = torch.empty(R, dtype=torch.float32, device=x2d.device)
    mean = torch.empty(R, dtype=torch.float32, device=x2d.device) if spec.has_mean else rstd

    args = (dyn(x2d.reshape(-1)), dyn(y.reshape(-1)), dyn(gamma), dyn(beta), dyn(mean), dyn(rstd))
    ce = (C, CS, CGA, _BT, V, VPT, KC, it_ty, et, float(C), float(eps), bool(spec.has_mean), has_beta, R * CGA)
    key = (params.io_dtype, C, R, CGA, KC, bool(spec.has_mean), has_beta)
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_ln_fwd_cga_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y, mean, rstd
