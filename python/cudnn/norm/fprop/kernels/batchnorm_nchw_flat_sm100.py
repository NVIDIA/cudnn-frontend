"""BatchNorm forward for NCHW, sm_100 -- FLAT fixed-position map.

The predecessor (``batchnorm_nchw_sm100``) maps a warp to one ``(n, c)`` row. That
wastes lanes whenever ``S/VS`` is not a multiple of 32 (S=784 -> 98 vectors -> 3.06
lane steps, so one warp instruction in four is nearly empty) and it re-quantises on
every row. Measured on a pure 1R1W pass it reaches only 0.26-0.86 of the copy ceiling
where a flat map reaches 0.85-0.90.

**The map.** A thread owns a fixed *position* ``pv`` (in 128-bit vectors) inside an
image and strides by **one whole image** (``P = C*S`` elements). Consequences:

* its channel ``c = (pv*VS) // S`` is **fixed** -- computed once, outside the loop --
  so per-channel accumulators live in registers with no atomics per element;
* consecutive threads hold consecutive vectors, so a CTA reads ``BT*VS`` contiguous
  elements per step: the same access a copy kernel makes, with no per-row remainder;
* odd ``S`` (7x7 -> 49) needs no scalar fallback. A vector may straddle two channels,
  but *which* two, and the split point ``k``, are also fixed per thread -- so it costs
  two accumulator pairs and a per-element select, not a narrower load.

Grid is ``(pparts, mparts)``: ``pparts`` tiles the image positions, ``mparts`` splits
the batch. A CTA's element window is contiguous, so it touches a **contiguous** run of
at most ``CPB`` channels -- which is what makes the cross-CTA reduce cheap: partials go
to ``[pparts, mparts, 2, CPB]`` with plain stores (no zeroing, since every slot is
written), and finalising a channel only reads the ``<= NPX`` position-tiles that
overlap it.

Cache tiers for the pass-2 re-read, both indexed per thread so pass1/pass2 line up:
``KTR`` images in TMEM (256KB/SM, off the L1 carveout) and ``KSR`` images in shared.

See ``cudnn/norm/fprop/BN_FWD_NOTES.md``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS

_CTA_SS = nvvm.SharedSpace.shared_cta
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}

_BT = 256
_UN = 4            # images issued per strip in the streamed loop
_SMEM_CAP = 200 * 1024
_TMEM_COLS = 512   # per SM

_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


def flat_cfg(C: int, N: int, S: int, elem_bytes: int, occ: int = 1, BT: int = _BT):
    """Resolve the flat-map geometry, or ``None`` if unsupported."""
    P = C * S
    VS = 16 // elem_bytes
    while VS > 1 and P % VS != 0:
        VS //= 2
    if VS < 2:
        return None
    PV = P // VS
    target = max(1, occ * _nsm())
    VPT = max(1, -(-PV // (BT * target)))
    pparts = max(1, -(-PV // (BT * VPT)))
    mparts = max(1, min(target // pparts, N))
    chunk = BT * VS * VPT                 # elements a CTA covers per image
    CPB = chunk // S + 2                  # channels a CTA's window can touch
    NPX = S // chunk + 2                  # position-tiles overlapping one channel
    return dict(P=P, VS=VS, PV=PV, VPT=VPT, pparts=pparts, mparts=mparts,
                chunk=chunk, CPB=CPB, NPX=NPX, BT=BT,
                straddle=(S % VS != 0))


def _tmem_cap(BT: int, VS: int, VPT: int, occ: int) -> int:
    """Images per thread that fit this CTA's share of TMEM (one fp32 column per
    element). Over-requesting BLOCKS in tcgen05_alloc, which in a cooperative grid
    is a hang -- so this is a correctness bound, not a knob."""
    per_thread_cols = (_TMEM_COLS // occ) // max(1, BT // 128)
    return max(0, per_thread_cols // (VPT * VS))


@cute.kernel
def _bn_flat(
    mX, mXi, mY, mYi, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
    N: cutlass.Int32, mparts: cutlass.Int32, momentum: cutlass.Float32,
    C: cutlass.Constexpr, S: cutlass.Constexpr, P: cutlass.Constexpr,
    PV: cutlass.Constexpr, VS: cutlass.Constexpr, VPT: cutlass.Constexpr,
    BT: cutlass.Constexpr, CPB: cutlass.Constexpr, NPX: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr, KSR: cutlass.Constexpr, KTR: cutlass.Constexpr,
    UN: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    update_running: cutlass.Constexpr, STRADDLE: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    px, my, _ = cute.arch.block_idx()
    warp = tid // 32

    smem = SmemAllocator()
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CPB), byte_alignment=16)
    slc = smem.allocate_tensor(cutlass.Int32, cute.make_layout(BT * VPT), byte_alignment=16)
    sv = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4 * BT * VPT), byte_alignment=16)
    sc = (smem.allocate_tensor(it_ty, cute.make_layout(BT * VPT * KSR * VS), byte_alignment=16)
          if cutlass.const_expr(KSR > 0) else None)
    tptr = None
    if cutlass.const_expr(KTR > 0):
        tptr = smem.allocate_tensor(cutlass.Int32, cute.make_layout(4), byte_alignment=16)
        if tid < 32:
            nvvm.tcgen05_alloc(tptr.iterator, max(1, BT // 128) * KTR * VPT * VS,
                               is_exclusive=False)
            nvvm.tcgen05_relinquish_alloc_permit()
        nvvm.barrier_cta_sync_aligned(0)

    cwbase = (px * CHUNK) // S           # first channel this CTA's window can touch
    per = (N + mparts - 1) // mparts
    n0 = my * per
    n1 = n0 + per
    if n1 > N:
        n1 = N

    # ---- per-thread FIXED position -> fixed channel(s). Computed once. ----
    e0 = []
    lc0 = []
    ksp = []
    okq = []
    for q in cutlass.range_constexpr(VPT):
        pv = px * (BT * VPT) + q * BT + tid
        ok = pv < PV
        pvc = pv
        if pvc > (PV - 1):
            pvc = PV - 1
        ev = pvc * VS
        e0.append(ev)
        lc0.append((ev // S) - cwbase)
        ksp.append(((ev // S) + 1) * S - ev)   # elements of this vector in channel c0
        okq.append(ok)

    acc = [cutlass.Float32(0.0)] * (2 * VPT)   # [lo, hi] per q
    asq = [cutlass.Float32(0.0)] * (2 * VPT)

    # ---- pass1 ----
    nb = n0
    while nb + UN <= n1:
        raws = []
        for u in cutlass.range_constexpr(UN):
            for q in cutlass.range_constexpr(VPT):
                raws.append(nvvm.load_ext(
                    mXi.iterator + (cutlass.Int64(nb + u) * P + e0[q]),
                    dtype=it_ty, count=VS).bitcast(et))
        i = 0
        for u in cutlass.range_constexpr(UN):
            for q in cutlass.range_constexpr(VPT):
                for e in cutlass.range_constexpr(VS):
                    x = raws[i][e].to(cutlass.Float32)
                    if cutlass.const_expr(STRADDLE):
                        if e < ksp[q]:
                            acc[2 * q] = acc[2 * q] + x
                            asq[2 * q] = asq[2 * q] + x * x
                        else:
                            acc[2 * q + 1] = acc[2 * q + 1] + x
                            asq[2 * q + 1] = asq[2 * q + 1] + x * x
                    else:
                        acc[2 * q] = acc[2 * q] + x
                        asq[2 * q] = asq[2 * q] + x * x
                i = i + 1
        nb = nb + UN
    while nb < n1:
        for q in cutlass.range_constexpr(VPT):
            xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(nb) * P + e0[q]),
                               dtype=it_ty, count=VS).bitcast(et)
            for e in cutlass.range_constexpr(VS):
                x = xv[e].to(cutlass.Float32)
                if cutlass.const_expr(STRADDLE):
                    if e < ksp[q]:
                        acc[2 * q] = acc[2 * q] + x
                        asq[2 * q] = asq[2 * q] + x * x
                    else:
                        acc[2 * q + 1] = acc[2 * q + 1] + x
                        asq[2 * q + 1] = asq[2 * q + 1] + x * x
                else:
                    acc[2 * q] = acc[2 * q] + x
                    asq[2 * q] = asq[2 * q] + x * x
        nb = nb + 1

    # ---- CTA reduce by channel. Threads publish (channel, sums); CPB threads scan.
    # Atomic-free: each thread contributes once, so an O(BT*VPT) scan by CPB threads
    # is cheaper than the smem-atomic contention (all lanes share a channel when
    # S >> BT*VS). ----
    for q in cutlass.range_constexpr(VPT):
        j = tid * VPT + q
        z = cutlass.Float32(0.0)
        slc[j] = lc0[q]
        sv[j] = acc[2 * q] if okq[q] else z
        sv[BT * VPT + j] = asq[2 * q] if okq[q] else z
        if cutlass.const_expr(STRADDLE):
            sv[2 * BT * VPT + j] = acc[2 * q + 1] if okq[q] else z
            sv[3 * BT * VPT + j] = asq[2 * q + 1] if okq[q] else z
    nvvm.barrier_cta_sync_aligned(0)

    if tid < CPB:
        ts = cutlass.Float32(0.0)
        tq = cutlass.Float32(0.0)
        j = cutlass.Int32(0)
        while j < BT * VPT:
            if slc[j] == tid:
                ts = ts + sv[j]
                tq = tq + sv[BT * VPT + j]
            if cutlass.const_expr(STRADDLE):
                if slc[j] + 1 == tid:
                    ts = ts + sv[2 * BT * VPT + j]
                    tq = tq + sv[3 * BT * VPT + j]
            j = j + 1
        base = ((px * mparts + my) * 2) * CPB
        mP[base + tid] = ts
        mP[base + CPB + tid] = tq
    nvvm.barrier_cta_sync_aligned(0)

    # ---- grid barrier ----
    nvvm.fence_acq_rel(nvvm.MemScope.GPU)
    if tid == 0:
        nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator, cutlass.Int32(1),
                       mem_order=nvvm.MemOrder.RELEASE, syncscope=nvvm.MemScope.GPU)
        done = False
        while not done:
            v = nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator, cutlass.Int32(0),
                               mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
            if v >= mRet[1]:
                done = True
    nvvm.barrier_cta_sync_aligned(0)

    # ---- finalize this CTA's channel window ----
    if tid < CPB:
        c = cwbase + tid
        if c < C:
            gs = cutlass.Float32(0.0)
            gq = cutlass.Float32(0.0)
            pxl = (c * S) // CHUNK
            pxh = ((c + 1) * S - 1) // CHUNK
            pxx = pxl
            while pxx <= pxh:
                j2 = c - ((pxx * CHUNK) // S)
                if j2 >= 0:
                    if j2 < CPB:
                        m = cutlass.Int32(0)
                        while m < mparts:
                            b2 = ((pxx * mparts + m) * 2) * CPB
                            gs = gs + mP[b2 + j2]
                            gq = gq + mP[b2 + CPB + j2]
                            m = m + 1
                pxx = pxx + 1
            mn = gs / Mf
            var = gq / Mf - mn * mn
            rs = cute.math.rsqrt(var + eps)
            stat[tid] = mn
            stat[CPB + tid] = rs
            if my == 0:
                if pxl == px:   # the first position-tile covering c owns the publish
                    mSavedMean[c] = mn
                    mSavedRstd[c] = rs
                    if cutlass.const_expr(update_running):
                        unb = var * Mf / (Mf - 1.0)
                        mRunMean[c] = (1.0 - momentum) * mRunMean[c] + momentum * mn
                        mRunVar[c] = (1.0 - momentum) * mRunVar[c] + momentum * unb
    nvvm.barrier_cta_sync_aligned(0)

    # ---- pass2: affine is fixed per thread (one per channel it straddles) ----
    NHc: cutlass.Constexpr = 2 if STRADDLE else 1
    sl = [cutlass.Float32(0.0)] * (VPT * NHc)
    sh = [cutlass.Float32(0.0)] * (VPT * NHc)
    for q in cutlass.range_constexpr(VPT):
        for h in cutlass.range_constexpr(NHc):
            li = lc0[q] + h
            if li > (CPB - 1):
                li = CPB - 1
            cc = cwbase + li
            if cc > (C - 1):
                cc = C - 1
            f = stat[CPB + li] * mG[cc].to(cutlass.Float32)
            b_ = -stat[li] * f
            if cutlass.const_expr(has_beta):
                b_ = b_ + mB[cc].to(cutlass.Float32)
            sl[q * NHc + h] = f
            sh[q * NHc + h] = b_

    NH: cutlass.Constexpr = 2 if STRADDLE else 1
    nb = n0
    while nb + UN <= n1:
        raws = []
        for u in cutlass.range_constexpr(UN):
            for q in cutlass.range_constexpr(VPT):
                raws.append(nvvm.load_ext(
                    mXi.iterator + (cutlass.Int64(nb + u) * P + e0[q]),
                    dtype=it_ty, count=VS).bitcast(et))
        i = 0
        for u in cutlass.range_constexpr(UN):
            for q in cutlass.range_constexpr(VPT):
                ys = []
                for e in cutlass.range_constexpr(VS):
                    x = raws[i][e].to(cutlass.Float32)
                    if cutlass.const_expr(STRADDLE):
                        y = x * sl[q * NH] + sh[q * NH]
                        if e >= ksp[q]:
                            y = x * sl[q * NH + 1] + sh[q * NH + 1]
                    else:
                        y = x * sl[q * NH] + sh[q * NH]
                    ys.append(y.to(et))
                if okq[q]:
                    nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                   mYi.iterator + (cutlass.Int64(nb + u) * P + e0[q]))
                i = i + 1
        nb = nb + UN
    while nb < n1:
        for q in cutlass.range_constexpr(VPT):
            xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(nb) * P + e0[q]),
                               dtype=it_ty, count=VS).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(VS):
                x = xv[e].to(cutlass.Float32)
                if cutlass.const_expr(STRADDLE):
                    y = x * sl[q * NH] + sh[q * NH]
                    if e >= ksp[q]:
                        y = x * sl[q * NH + 1] + sh[q * NH + 1]
                else:
                    y = x * sl[q * NH] + sh[q * NH]
                ys.append(y.to(et))
            if okq[q]:
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                               mYi.iterator + (cutlass.Int64(nb) * P + e0[q]))
        nb = nb + 1

    if cutlass.const_expr(KTR > 0):
        nvvm.barrier_cta_sync_aligned(0)
        if tid < 32:
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tptr[0], cutlass.Float32),
                                 max(1, BT // 128) * KTR * VPT * VS, is_exclusive=False)


@cute.jit
def _bn_flat_host(
    mX, mY, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
    N, mparts, momentum,
    C: cutlass.Constexpr, S: cutlass.Constexpr, P: cutlass.Constexpr,
    PV: cutlass.Constexpr, VS: cutlass.Constexpr, VPT: cutlass.Constexpr,
    BT: cutlass.Constexpr, CPB: cutlass.Constexpr, NPX: cutlass.Constexpr,
    CHUNK: cutlass.Constexpr, KSR: cutlass.Constexpr, KTR: cutlass.Constexpr,
    UN: cutlass.Constexpr, pparts: cutlass.Constexpr, it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr, Mf: cutlass.Constexpr, eps: cutlass.Constexpr,
    has_beta: cutlass.Constexpr, update_running: cutlass.Constexpr,
    STRADDLE: cutlass.Constexpr, smem_bytes: cutlass.Constexpr, mbpm: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _bn_flat(mX, mXi, mY, mYi, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean,
             mRunVar, N, mparts, momentum, C, S, P, PV, VS, VPT, BT, CPB, NPX, CHUNK,
             KSR, KTR, UN, it_ty, et, Mf, eps, has_beta, update_running, STRADDLE
             ).launch(grid=(pparts, mparts, 1), block=(BT, 1, 1), smem=smem_bytes,
                      cooperative=True, min_blocks_per_mp=mbpm)


_KCACHE = {}


def forward(spec, x3d, gamma, beta, *, eps, momentum, training, running_mean,
            running_var, cfg, params, knobs=None):
    import torch

    N, C, S = int(spec.N), int(spec.C), int(spec.S)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    g = flat_cfg(C, N, S, eb, occ=1)
    if g is None:
        raise ValueError(f"flat NCHW BatchNorm: C={C} S={S} unsupported")
    has_beta = beta is not None
    if beta is None:
        beta = gamma

    y3d = torch.empty_like(x3d)
    xf = x3d.reshape(-1)
    yf = y3d.reshape(-1)
    saved_mean = torch.empty(C, dtype=torch.float32, device=x3d.device)
    saved_rstd = torch.empty(C, dtype=torch.float32, device=x3d.device)
    update_running = bool(training and running_mean is not None and running_var is not None)
    rm = running_mean if update_running else saved_mean
    rv = running_var if update_running else saved_rstd

    VS, VPT, BT, CPB = g["VS"], g["VPT"], g["BT"], g["CPB"]
    pparts, mparts, CHUNK = g["pparts"], g["mparts"], g["chunk"]
    KSR = KTR = 0
    smem_bytes = 2 * CPB * 4 + BT * VPT * 4 + 4 * BT * VPT * 4 + 256
    ce = (C, S, g["P"], g["PV"], VS, VPT, BT, CPB, g["NPX"], CHUNK, KSR, KTR, _UN,
          pparts, it_ty, et, float(N * S), float(eps), has_beta, update_running,
          g["straddle"], smem_bytes, 0)
    key = (params.io_dtype, C, S, VPT, CPB, pparts, has_beta, update_running)

    pbuf = torch.empty(pparts * mparts * 2 * CPB, dtype=torch.float32, device=x3d.device)
    ret = torch.zeros(2, dtype=torch.int32, device=x3d.device)
    ret[1] = pparts * mparts
    args = (dyn(xf), dyn(yf), dyn(gamma), dyn(beta), dyn(pbuf), dyn(ret),
            dyn(saved_mean), dyn(saved_rstd), dyn(rm), dyn(rv),
            cutlass.Int32(N), cutlass.Int32(mparts), cutlass.Float32(momentum))
    fn = _KCACHE.get(key)
    if fn is None:
        fn = cute.compile(_bn_flat_host, *args, *ce)
        _KCACHE[key] = fn
    fn(*args)
    return y3d, saved_mean, saved_rstd
