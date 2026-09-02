"""BatchNorm forward for NHWC (channels-last), sm_100, CUTLASS primitives.

A single **cooperative** kernel with this cross-CTA structure:

    pass1 (reduce this CTA's NHW slice, caching the first ``KC`` pixels/thread in
    registers and the next ``KS`` in shared) -> write this CTA's partial to its OWN
    slot in a ``[cblks, 2, mparts, C_tile]`` buffer (plain vector stores, NO atomics)
    -> gpu-scope fence + atomic-spin GRID BARRIER -> REDUNDANT finalize (every
    part-CTA re-reduces all ``mparts`` slots for its channel tile, parallelised across
    the ``PPL`` pixel groups; the partials are L2-resident) -> broadcast mean/rstd
    through shared -> pass2 normalize, reading the cached pixels back out of
    registers/shared and re-reading only the overflow.

NHWC flattens to ``[M = N*H*W, C]`` and the reduction runs down ``M`` per channel, so
``C`` is the fast (coalesced, 128-bit vectorised) axis. The thread map is cuDNN's:
``TPP = C_tile/V`` threads span the contiguous channel tile and ``PPL = BT/TPP``
pixels are in flight per CTA, with ``mparts`` CTAs splitting ``M`` (split-K).

Inference (``training=False``) needs no reduction at all, so it takes a plain
non-cooperative elementwise kernel with the affine staged once per thread.

See ``cudnn/norm/fprop/BN_FWD_NOTES.md`` for the measurements behind the knobs.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.primitives as nvvm
from cutlass.memory import SmemAllocator

from cudnn.norm.utils import dyn
from cudnn.norm.dtypes import DTYPE_BYTES, DTYPE_TO_CUTLASS

_CTA_SS = nvvm.SharedSpace.shared_cta

# elem_bytes -> the integer type the raw 128-bit vector loads/stores are typed as.
_INT_TY = {2: cutlass.Int16, 4: cutlass.Int32}

_SM_COUNT = None


def _nsm():
    global _SM_COUNT
    if _SM_COUNT is None:
        import torch

        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------

# Cap the channel tile so TPP stays small and PPL (pixel parallelism) stays large:
# a whole-C-per-CTA tile gave PPL=1 for C=2048 and cost ~0.2x (see BN_FWD_NOTES).
_CPC_MAX = 256
_BT = 512  # cuDNN's THREADS_PER_CTA; halves the per-thread slice vs 256
_SMEM_PER_SM = 228 * 1024
_UR = 4  # pixels issued per strip in the uncached remainder loops


def nhwc_cfg(C: int, elem_bytes: int, block_threads: int = _BT):
    """Resolve ``(V, CPC, TPP, PPL, cblks, BT)`` for an NHWC BatchNorm launch.

    Returns ``None`` when ``C`` cannot be tiled for 128-bit vector access, in which
    case the caller falls back to the generic (layout-agnostic) kernel.
    """
    V = 16 // elem_bytes
    if C % V != 0:
        return None
    # Largest divisor of C that is <= _CPC_MAX and a multiple of V.
    cpc = 0
    for cand in range(min(C, _CPC_MAX), V - 1, -V):
        if C % cand == 0:
            cpc = cand
            break
    if cpc == 0:
        return None
    tpp = cpc // V
    if tpp > block_threads or block_threads % tpp != 0:
        return None
    return V, cpc, tpp, block_threads // tpp, C // cpc, block_threads


def _mparts(M: int, PPL: int, cblks: int, occ: int) -> int:
    """CTAs per channel tile. Cooperative launch requires the whole grid to
    co-reside, so ``cblks * mparts <= occ * NSM``."""
    return max(1, min(occ * _nsm() // max(1, cblks), (M + PPL - 1) // PPL))


# ---------------------------------------------------------------------------
# Training: fused cooperative kernel
# ---------------------------------------------------------------------------


@cute.kernel
def _bn_nhwc_kernel(
    mXi, mYi, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
    M: cutlass.Int32, mparts: cutlass.Int32, momentum: cutlass.Float32,
    C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr, BT: cutlass.Constexpr, KC: cutlass.Constexpr,
    KS: cutlass.Constexpr, KT: cutlass.Constexpr, UR: cutlass.Constexpr,
    CPC: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    update_running: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    NSEG: cutlass.Constexpr = V // 4  # 128-bit fp32 segments per V-vector
    CPP: cutlass.Constexpr = V  # TMEM columns per pixel (one fp32 per element)

    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BT * V), byte_alignment=16)
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CPC), byte_alignment=16)
    sc = (
        smem.allocate_tensor(it_ty, cute.make_layout(BT * KS * V), byte_alignment=16)
        if cutlass.const_expr(KS > 0)
        else None
    )
    # ---- TMEM tier. 256KB/SM that does NOT come out of the smem/L1 carveout, at
    # ~40-67 TB/s, and (with is_exclusive=False) at no occupancy cost -- so unlike the
    # smem tier it can be made large without starving L1. Each thread owns one TMEM
    # lane; warps beyond the first four share those lanes via disjoint column windows.
    tptr = None
    tcol0 = 0
    if cutlass.const_expr(KT > 0):
        tptr = smem.allocate_tensor(cutlass.Int32, cute.make_layout(4), byte_alignment=16)
        if tid < 32:
            nvvm.tcgen05_alloc(tptr.iterator, (BT // 128) * KT * CPP, is_exclusive=False)
            nvvm.tcgen05_relinquish_alloc_permit()
        nvvm.barrier_cta_sync_aligned(0)

    lane = tid % TPP  # which V-wide chunk of the channel tile this thread owns
    tp = tid // TPP  # which of the PPL parallel pixels
    c0 = cx * (TPP * V) + lane * V
    per = (M + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > M:
        r1 = M

    # ---- pass1: reduce; cache first KC pixels in regs, next KS in smem ----
    s = [cutlass.Float32(0.0)] * V
    sq = [cutlass.Float32(0.0)] * V
    cache = [cutlass.Float32(0.0)] * (KC * V)
    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(rk) * C + c0), dtype=it_ty, count=V).bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                s[e] = s[e] + x
                sq[e] = sq[e] + x * x
                cache[kk * V + e] = x
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            raw = nvvm.load_ext(mXi.iterator + (cutlass.Int64(rk) * C + c0), dtype=it_ty, count=V)
            xv = raw.bitcast(et)
            for e in cutlass.range_constexpr(V):
                x = xv[e].to(cutlass.Float32)
                s[e] = s[e] + x
                sq[e] = sq[e] + x * x
            nvvm.store_ext(raw, sc.iterator + (tid * KS + ks) * V, shared_space=_CTA_SS)
    if cutlass.const_expr(KT > 0):
        wg: cutlass.Constexpr = BT // 128  # warp-groups sharing the 128 TMEM lanes
        warp = tid // 32
        wcol = (warp // 4) * (KT * CPP)
        for kt in cutlass.range_constexpr(KT):
            rk = r0 + tp + (KC + KS + kt) * PPL
            # tcgen05 ld/st are warp-ALIGNED: every lane must participate. When
            # TPP < 32 a warp spans several pixel groups, so ``rk < r1`` is divergent
            # and guarding the tcgen05 op with it HANGS. Clamp the row instead, run
            # the TMEM op unconditionally, and predicate only the accumulate.
            rr = rk
            if rr >= r1:
                rr = r1 - 1
            if rr < 0:
                rr = 0
            tvin = nvvm.load_ext(mXi.iterator + (cutlass.Int64(rr) * C + c0),
                                 dtype=it_ty, count=V).bitcast(et)
            fs = []
            for e in cutlass.range_constexpr(V):
                fs.append(tvin[e].to(cutlass.Float32))
            if rk < r1:
                for e in cutlass.range_constexpr(V):
                    s[e] = s[e] + fs[e]
                    sq[e] = sq[e] + fs[e] * fs[e]
            tp_kt = nvvm.make_tmem_ptr_from_warp_row_col(
                tptr[0], warp % 4, wcol + kt * CPP, cutlass.Float32)
            nvvm.tcgen05_st(nvvm.Tcgen05LdStShape.SHAPE_32X32B, tp_kt,
                            cutlass.Vector.from_elements(tuple(fs), cutlass.Float32))
        nvvm.tcgen05_wait(nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
    # The remainder (pixels past the cache tiers) is strip-mined: UR pixels are
    # ISSUED before any is consumed. A plain one-load-per-iteration loop here is
    # latency-bound and dominates whenever the slice exceeds the cache depth.
    # The strip bound is deliberately tp-independent so it stays warp-uniform.
    rbase = r0 + (KC + KS + KT) * PPL
    jb = cutlass.Int32(0)
    while rbase + (PPL - 1) + (jb + (UR - 1)) * PPL < r1:
        raws = []
        for u in cutlass.range_constexpr(UR):
            raws.append(nvvm.load_ext(
                mXi.iterator + (cutlass.Int64(rbase + tp + (jb + u) * PPL) * C + c0),
                dtype=it_ty, count=V).bitcast(et))
        for u in cutlass.range_constexpr(UR):
            for e in cutlass.range_constexpr(V):
                x = raws[u][e].to(cutlass.Float32)
                s[e] = s[e] + x
                sq[e] = sq[e] + x * x
        jb = jb + UR
    row = rbase + tp + jb * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s[e] = s[e] + x
            sq[e] = sq[e] + x * x
        row = row + PPL

    # ---- reduce across the PPL pixel groups (reused by pass1 stats AND finalize) ----
    def _reduce(vals):
        if cutlass.const_expr(PPL > 1):
            for e in cutlass.range_constexpr(V):
                red[(tp * TPP + lane) * V + e] = vals[e]
            nvvm.barrier_cta_sync_aligned(0)
            if tp == 0:
                for e in cutlass.range_constexpr(V):
                    acc = vals[e]
                    for j in cutlass.range_constexpr(PPL - 1):
                        acc = acc + red[((j + 1) * TPP + lane) * V + e]
                    vals[e] = acc
            nvvm.barrier_cta_sync_aligned(0)
        return vals

    s = _reduce(s)
    sq = _reduce(sq)

    # ---- cross-CTA reduce: this CTA's partials -> its OWN slot, plain stores ----
    psum = cx * (mparts * TPP * V * 2)
    psq = psum + mparts * TPP * V
    if tp == 0:
        off = (my * TPP + lane) * V
        for h in cutlass.range_constexpr(NSEG):
            sseg = cutlass.Vector.from_elements(tuple(s[h * 4 + j] for j in range(4)), cutlass.Float32)
            qseg = cutlass.Vector.from_elements(tuple(sq[h * 4 + j] for j in range(4)), cutlass.Float32)
            nvvm.store_ext(sseg, mP.iterator + (psum + off + h * 4))
            nvvm.store_ext(qseg, mP.iterator + (psq + off + h * 4))

    # ---- grid barrier: gpu-scope fence so the partials land, then a retired-CTA spin ----
    nvvm.fence_acq_rel(nvvm.MemScope.GPU)
    nvvm.barrier_cta_sync_aligned(0)
    if tid == 0:
        nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + cx, cutlass.Int32(1),
                       mem_order=nvvm.MemOrder.RELEASE, syncscope=nvvm.MemScope.GPU)
        done = False
        while not done:
            v = nvvm.atomicrmw(nvvm.AtomicOp.ADD, mRet.iterator + cx, cutlass.Int32(0),
                               mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
            if v >= mparts:
                done = True
    nvvm.barrier_cta_sync_aligned(0)

    # ---- REDUNDANT finalize: every part-CTA re-reduces all mparts slots for its tile.
    # Fully parallel across part-CTAs, vectorised, L2-resident, and needs no 2nd grid
    # barrier -- measured faster than a non-redundant single-CTA finalize (STEP 6). ----
    facc = [cutlass.Float32(0.0)] * V
    faccsq = [cutlass.Float32(0.0)] * V
    part = tp
    while part < mparts:
        off = (part * TPP + lane) * V
        for h in cutlass.range_constexpr(NSEG):
            sv = nvvm.load_ext(mP.iterator + (psum + off + h * 4), dtype=cutlass.Float32, count=4)
            qv = nvvm.load_ext(mP.iterator + (psq + off + h * 4), dtype=cutlass.Float32, count=4)
            for j in cutlass.range_constexpr(4):
                facc[h * 4 + j] = facc[h * 4 + j] + sv[j]
                faccsq[h * 4 + j] = faccsq[h * 4 + j] + qv[j]
        part = part + PPL
    facc = _reduce(facc)
    faccsq = _reduce(faccsq)

    # tp==0 now holds the full per-channel reduction -> mean/rstd; broadcast via smem.
    if tp == 0:
        for e in cutlass.range_constexpr(V):
            mn = facc[e] / Mf
            var = faccsq[e] / Mf - mn * mn
            rs = cute.math.rsqrt(var + eps)
            stat[lane * V + e] = mn
            stat[CPC + lane * V + e] = rs
            # One CTA per channel tile publishes the saved / running statistics.
            if my == 0:
                mSavedMean[c0 + e] = mn
                mSavedRstd[c0 + e] = rs
                if cutlass.const_expr(update_running):
                    # PyTorch semantics: running_var tracks the UNBIASED estimator.
                    unb = var * Mf / (Mf - 1.0)
                    mRunMean[c0 + e] = (1.0 - momentum) * mRunMean[c0 + e] + momentum * mn
                    mRunVar[c0 + e] = (1.0 - momentum) * mRunVar[c0 + e] + momentum * unb
    nvvm.barrier_cta_sync_aligned(0)

    mean = [cutlass.Float32(0.0)] * V
    rstd = [cutlass.Float32(0.0)] * V
    for e in cutlass.range_constexpr(V):
        mean[e] = stat[lane * V + e]
        rstd[e] = stat[CPC + lane * V + e]
    g = []
    bb = []
    for e in cutlass.range_constexpr(V):
        g.append(mG[c0 + e].to(cutlass.Float32))
        if cutlass.const_expr(has_beta):
            bb.append(mB[c0 + e].to(cutlass.Float32))

    # ---- pass2: normalize from the register cache, then smem, then re-read.
    # The store is written out three times rather than factored into a helper: a
    # closure would capture ``C``/``c0``, which the DSL cannot stage inside an ``if``.
    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            ys = []
            for e in cutlass.range_constexpr(V):
                y = (cache[kk * V + e] - mean[e]) * rstd[e] * g[e]
                if cutlass.const_expr(has_beta):
                    y = y + bb[e]
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                           mYi.iterator + (cutlass.Int64(rk) * C + c0))
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            xv = nvvm.load_ext(sc.iterator + (tid * KS + ks) * V, dtype=it_ty, count=V,
                               shared_space=_CTA_SS).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(V):
                y = (xv[e].to(cutlass.Float32) - mean[e]) * rstd[e] * g[e]
                if cutlass.const_expr(has_beta):
                    y = y + bb[e]
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                           mYi.iterator + (cutlass.Int64(rk) * C + c0))
    if cutlass.const_expr(KT > 0):
        warp = tid // 32
        wcol = (warp // 4) * (KT * CPP)
        for kt in cutlass.range_constexpr(KT):
            rk = r0 + tp + (KC + KS + kt) * PPL
            tp_kt = nvvm.make_tmem_ptr_from_warp_row_col(
                tptr[0], warp % 4, wcol + kt * CPP, cutlass.Float32)
            # Unconditional (warp-aligned); only the global store is predicated.
            tvout = nvvm.tcgen05_ld(nvvm.Tcgen05LdStShape.SHAPE_32X32B, tp_kt, num=CPP)
            if rk < r1:
                ys = []
                for e in cutlass.range_constexpr(V):
                    y = (tvout[e] - mean[e]) * rstd[e] * g[e]
                    if cutlass.const_expr(has_beta):
                        y = y + bb[e]
                    ys.append(y.to(et))
                nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                               mYi.iterator + (cutlass.Int64(rk) * C + c0))
        nvvm.tcgen05_wait(nvvm.Tcgen05Wait.LOAD)

    rbase = r0 + (KC + KS + KT) * PPL
    jb = cutlass.Int32(0)
    while rbase + (PPL - 1) + (jb + (UR - 1)) * PPL < r1:
        raws = []
        for u in cutlass.range_constexpr(UR):
            raws.append(nvvm.load_ext(
                mXi.iterator + (cutlass.Int64(rbase + tp + (jb + u) * PPL) * C + c0),
                dtype=it_ty, count=V).bitcast(et))
        for u in cutlass.range_constexpr(UR):
            ys = []
            for e in cutlass.range_constexpr(V):
                y = (raws[u][e].to(cutlass.Float32) - mean[e]) * rstd[e] * g[e]
                if cutlass.const_expr(has_beta):
                    y = y + bb[e]
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                           mYi.iterator + (cutlass.Int64(rbase + tp + (jb + u) * PPL) * C + c0))
        jb = jb + UR
    row = rbase + tp + jb * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            y = (xv[e].to(cutlass.Float32) - mean[e]) * rstd[e] * g[e]
            if cutlass.const_expr(has_beta):
                y = y + bb[e]
            ys.append(y.to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                       mYi.iterator + (cutlass.Int64(row) * C + c0))
        row = row + PPL

    if cutlass.const_expr(KT > 0):
        nvvm.barrier_cta_sync_aligned(0)
        if tid < 32:
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tptr[0], cutlass.Float32),
                                 (BT // 128) * KT * CPP, is_exclusive=False)


@cute.jit
def _bn_nhwc_host(
    mX, mY, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
    M, mparts, momentum,
    C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr, BT: cutlass.Constexpr, KC: cutlass.Constexpr,
    KS: cutlass.Constexpr, KT: cutlass.Constexpr, UR: cutlass.Constexpr,
    CPC: cutlass.Constexpr, cblks: cutlass.Constexpr,
    it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    update_running: cutlass.Constexpr, smem_bytes: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _bn_nhwc_kernel(
        mXi, mYi, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
        M, mparts, momentum, C, V, TPP, PPL, BT, KC, KS, KT, UR, CPC, it_ty, et, Mf, eps,
        has_beta, update_running,
    ).launch(grid=(cblks, mparts, 1), block=(BT, 1, 1), smem=smem_bytes,
             cooperative=True, min_blocks_per_mp=mbpm)


# ---------------------------------------------------------------------------
# Inference: plain elementwise kernel (no reduction, no cooperative launch)
# ---------------------------------------------------------------------------


@cute.kernel
def _bn_nhwc_infer_kernel(
    mXi, mYi, mG, mB, mRunMean, mRunVar, M: cutlass.Int32,
    C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr, BT: cutlass.Constexpr, it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    _, nblk, _ = cute.arch.grid_dim()  # grid is (cblks, nblk, 1) -- the M-split is .y
    lane = tid % TPP
    tp = tid // TPP
    c0 = cx * (TPP * V) + lane * V
    # Stage the affine once per thread, then stream pixels.
    scale = []
    shift = []
    for e in cutlass.range_constexpr(V):
        rs = cute.math.rsqrt(mRunVar[c0 + e].to(cutlass.Float32) + eps)
        gg = mG[c0 + e].to(cutlass.Float32) * rs
        scale.append(gg)
        bo = -mRunMean[c0 + e].to(cutlass.Float32) * gg
        if cutlass.const_expr(has_beta):
            bo = bo + mB[c0 + e].to(cutlass.Float32)
        shift.append(bo)
    row = my * PPL + tp
    stride = nblk * PPL
    while row < M:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            ys.append((xv[e].to(cutlass.Float32) * scale[e] + shift[e]).to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                       mYi.iterator + (cutlass.Int64(row) * C + c0))
        row = row + stride


@cute.jit
def _bn_nhwc_infer_host(
    mX, mY, mG, mB, mRunMean, mRunVar, M,
    C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr,
    PPL: cutlass.Constexpr, BT: cutlass.Constexpr, cblks: cutlass.Constexpr,
    nblk: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _bn_nhwc_infer_kernel(mXi, mYi, mG, mB, mRunMean, mRunVar, M, C, V, TPP, PPL,
                          BT, it_ty, et, eps, has_beta).launch(
        grid=(cblks, nblk, 1), block=(BT, 1, 1))


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

_KCACHE = {}
_OCC = {}

#
# (min_blocks_per_mp, KC, KS) -- occupancy, register-cache depth, smem-cache depth.
#
#   occ=1 + an 8-pixel register cache + a big L1  ... small and mid M
#   occ=2 + no register cache + a 6-pixel smem cache ... large M, and large C
#
# At occ=1 the ~238KB smem/L1 carveout is left almost entirely to L1, which caches
# the pass-2 re-reads better than any explicit smem tile. At occ=2 the per-thread
# slice halves, so a *moderate* smem cache covers a real fraction of it without
# starving L1 -- a large one (>=100KB/CTA) regresses on both counts.
# (min_blocks_per_mp, KC regs, KS smem pixels, KT tmem pixels) -- occupancy,
# register-cache depth and smem-cache depth, plus the
# TMEM tier. Measured per-shape optimum on the RN50 set (see BN_FWD_NOTES.md); the
# spread between candidates is ~5-10%, so this is a heuristic over measured bands
# rather than a fit.
_BIG_M = 131072


def _knobs_for(M: int, C: int):
    if M > _BIG_M:
        return (2, 0, 6, 0)   # large M: occ=2 + a moderate smem cache
    if C >= 1024:
        return (2, 0, 0, 8)   # wide C: occ=2 + the TMEM cache
    if M <= 32768:
        return (0, 8, 0, 0)   # small M: occ=1, registers only, big L1
    return (2, 0, 0, 8)       # mid: occ=2 + the TMEM cache


def _tmem_cap(BT: int, V: int, occ: int) -> int:
    """Pixels per thread that fit in this CTA's share of TMEM.

    TMEM is 512 columns x 128 lanes per SM. A CTA of ``BT`` threads has ``BT//128``
    warp-groups sharing those 128 lanes, so it needs ``(BT//128) * KT * V`` columns
    (one fp32 column per element). Over-requesting BLOCKS in ``tcgen05_alloc``, which
    in a cooperative grid is a hang -- so this cap is a correctness constraint, not a
    tuning knob.
    """
    return max(0, (512 // occ) // (max(1, BT // 128) * V))


def _smem_bytes(BT, V, CPC, KS, elem_bytes):
    return BT * V * 4 + 2 * CPC * 4 + BT * KS * V * elem_bytes + 128


def forward(spec, x2d, gamma, beta, *, eps, momentum, training, running_mean,
            running_var, cfg, params, knobs=None):
    """Launch the NHWC BatchNorm forward on ``x2d`` viewed as ``[M, C]``.

    Returns ``(y2d, saved_mean, saved_rstd)``. ``running_mean``/``running_var`` are
    updated in place when ``training`` and both are supplied.
    """
    import torch

    M, C = int(x2d.shape[0]), int(x2d.shape[1])
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    geo = nhwc_cfg(C, eb)
    if geo is None:
        raise ValueError(f"NHWC BatchNorm: C={C} is not 128-bit tileable")
    V, CPC, TPP, PPL, cblks, BT = geo
    has_beta = beta is not None
    if beta is None:
        beta = gamma

    y2d = torch.empty_like(x2d)
    saved_mean = torch.empty(C, dtype=torch.float32, device=x2d.device)
    saved_rstd = torch.empty(C, dtype=torch.float32, device=x2d.device)

    if not training:
        rm = running_mean if running_mean is not None else torch.zeros(C, dtype=torch.float32, device=x2d.device)
        rv = running_var if running_var is not None else torch.ones(C, dtype=torch.float32, device=x2d.device)
        nblk = max(1, min(_nsm() * 4 // max(1, cblks), (M + PPL - 1) // PPL))
        ce = (C, V, TPP, PPL, BT, cblks, nblk, it_ty, et, float(eps), has_beta)
        key = ("infer", params.io_dtype) + ce[:-4] + (has_beta,)
        args = (dyn(x2d), dyn(y2d), dyn(gamma), dyn(beta), dyn(rm), dyn(rv), cutlass.Int32(M))
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_bn_nhwc_infer_host, *args, *ce)
            _KCACHE[key] = fn
        fn(*args)
        # Inference still reports the statistics it normalised with.
        saved_mean.copy_(rm.float())
        saved_rstd.copy_((rv.float() + eps).rsqrt())
        return y2d, saved_mean, saved_rstd

    update_running = bool(running_mean is not None and running_var is not None)
    rm = running_mean if update_running else saved_mean
    rv = running_var if update_running else saved_rstd

    mbpm, KC, KS, KT = knobs if knobs is not None else _knobs_for(M, C)
    smem_bytes = _smem_bytes(BT, V, CPC, KS, eb)

    ce_head = (C, V, TPP, PPL, BT, KC, KS)
    ce_tail = (CPC, cblks, it_ty, et, float(M), float(eps), has_beta, update_running,
               smem_bytes, mbpm)
    key_head = (params.io_dtype, C, KC, KS)
    key_tail = (CPC, has_beta, update_running, mbpm)

    okey = key_head + key_tail + (M,)
    occ0 = _OCC.get(okey, 2 if mbpm != 1 else 1)
    can2 = smem_bytes * 2 <= 224 * 1024
    for occ in ([2, 1] if (occ0 == 2 and can2) else [1]):
        KT_occ = min(KT, _tmem_cap(BT, V, occ))
        smem_bytes = _smem_bytes(BT, V, CPC, KS, eb)
        if KT_occ > 0:
            # Same hazard as NCHW: without shared memory pinning the occupancy, a
            # 3rd co-resident CTA can exhaust the SM's 512 TMEM columns and its
            # tcgen05_alloc leaves a garbage base address.
            smem_bytes = max(smem_bytes, _SMEM_PER_SM // (occ + 1) + 1)
        ce = ce_head + (KT_occ, _UR) + ce_tail[:-2] + (smem_bytes, mbpm)
        key = key_head + (KT_occ, _UR) + key_tail
        fn = _KCACHE.get(key)
        mparts = _mparts(M, PPL, cblks, occ)
        # Partials [cblks, 2, mparts, C_tile]; every slot is written, so no zeroing.
        pbuf = torch.empty(cblks * mparts * TPP * V * 2, dtype=torch.float32, device=x2d.device)
        ret = torch.zeros(cblks, dtype=torch.int32, device=x2d.device)
        args = (dyn(x2d), dyn(y2d), dyn(gamma), dyn(beta), dyn(pbuf), dyn(ret),
                dyn(saved_mean), dyn(saved_rstd), dyn(rm), dyn(rv),
                cutlass.Int32(M), cutlass.Int32(mparts), cutlass.Float32(momentum))
        if fn is None:
            fn = cute.compile(_bn_nhwc_host, *args, *ce)
            _KCACHE[key] = fn
        try:
            fn(*args)
            _OCC[okey] = occ
            return y2d, saved_mean, saved_rstd
        except Exception as e:  # COOPERATIVE_LAUNCH_TOO_LARGE -> retry at occ=1
            if "TOO_LARGE" in str(e) and occ == 2:
                continue
            raise
    return y2d, saved_mean, saved_rstd
