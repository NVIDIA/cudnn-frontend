"""BatchNorm forward for NCHW, sm_100, CUTLASS primitives.

NCHW is the *easy* layout for BatchNorm even though cuDNN treats it as a fallback:
for channel ``c`` the elements of image ``n`` are the ``S = H*W`` **contiguous**
values at ``(n*C + c) * S``. So the reduction set is a union of contiguous rows and
a warp-per-row map is perfectly coalesced.

Work map -- a CTA owns ``CT = WPB * CPW`` channels and a range of images:

* warp ``w`` owns channels ``c0 + cw*WPB + w`` for ``cw < CPW`` and *only* those, so
  its per-channel accumulators stay in **registers** for the whole pass and the
  cross-CTA reduce needs **no atomics anywhere** (unlike the NHWC kernel, whose
  channel tile is shared by every warp and needs a partials buffer per pixel group).
* at any instant the CTA's warps read ``WPB`` adjacent channels of one image, i.e.
  one contiguous ``WPB*S`` span -- so the CTA is coalesced as a whole too.
* ``mparts`` CTAs split the image axis (split-K); partials go to a
  ``[cparts, 2, mparts, CT]`` buffer with plain stores.

Then: gpu-scope fence + atomic-spin grid barrier -> finalize (redundant, but each
CTA only re-reduces ``mparts * 2 * CT`` floats, versus the NHWC kernel's 256-channel
tile) -> pass2 normalize. Because each row belongs to exactly ONE channel, pass2's
affine is a **scalar per row**, staged into registers once, so the normalize is a
pure streaming copy.

**Memory-level parallelism.** Every element loop is strip-mined: ``UF`` lane steps
are *issued* before any is consumed, and ``CPW`` channels are unrolled on top, so a
warp keeps ~``UF*CPW`` loads in flight. A naive one-load-per-iteration loop is purely
latency-bound here -- it measured 455 GB/s on 2048x7x7 against a 4986 GB/s ceiling.

``KSR`` whole image-rows per warp are cached in shared memory on the pass-1 read so
pass2 does not go back to global for them.

Vector width: ``VS`` is the widest 128-bit-or-narrower vector that divides ``S``.
``load_ext`` only accepts counts of 2/4/8, and a row at an odd multiple of ``S`` is
only ``elem_bytes``-aligned, so odd ``S`` (e.g. 7x7 -> 49) takes a scalar path --
still coalesced, since 32 lanes cover 32 consecutive elements.

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
_FULL = 0xFFFFFFFF
_BFLY_CLAMP = 0x1F  # shfl width == 32

_BT = 256
_SMEM_CAP = 200 * 1024  # headroom under the 228KB dynamic-smem opt-in cap
_SMEM_PER_SM = 228 * 1024
_INFLIGHT = 8  # target loads in flight per warp (UF * CPW)

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


def _vec_width(S: int, elem_bytes: int) -> int:
    """Widest <=128-bit vector dividing S; 1 (scalar path) when S is odd."""
    v = 16 // elem_bytes
    while v > 1 and S % v != 0:
        v //= 2
    return v


def nchw_cfg(C: int, N: int, S: int, elem_bytes: int, block_threads: int = _BT):
    """Resolve ``(VS, UF, WPB, CPW, CT, cparts, BT)`` or ``None`` if unsupported."""
    wpb = block_threads // 32
    if C % wpb != 0 or S < 1:
        return None
    vs = _vec_width(S, elem_bytes)
    nv = S // vs                                  # lane steps per row
    # UF is capped by register pressure AND by the row actually containing a full
    # strip -- a strip loop that never fires leaves every lane in the scalar tail.
    uf = max(1, min(max(2, 32 // vs), nv // 32))
    # The channel unroll supplies the rest of the in-flight loads, then keeps growing
    # while the channel-tile count is too large for a cooperative grid.
    want = max(1, -(-_INFLIGHT // uf))
    cpw = 1
    while cpw < want and C // (wpb * cpw) > 1 and (C // (wpb * cpw)) % 2 == 0:
        cpw *= 2
    while C // (wpb * cpw) > 2 * _nsm() and (C // (wpb * cpw)) % 2 == 0:
        cpw *= 2
    ct = wpb * cpw
    if C % ct != 0:
        return None
    return vs, uf, wpb, cpw, ct, C // ct, block_threads


def _split(N: int, cparts: int, occ: int) -> int:
    return max(1, min(occ * _nsm() // max(1, cparts), N))


@cute.jit
def _warp_sum2(a, b):
    """Butterfly all-reduce of two fp32 partials across a full warp."""
    for d in cutlass.range_constexpr(5):
        off = 1 << d
        a = a + nvvm.shfl_sync(_FULL, a, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
        b = b + nvvm.shfl_sync(_FULL, b, off, _BFLY_CLAMP, nvvm.Shfl.BFLY)
    return a, b


# ---------------------------------------------------------------------------
# Training kernel
# ---------------------------------------------------------------------------


@cute.kernel
def _bn_nchw_kernel(
    mX, mXi, mY, mYi, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
    N: cutlass.Int32, mparts: cutlass.Int32, momentum: cutlass.Float32,
    C: cutlass.Constexpr, S: cutlass.Constexpr, VS: cutlass.Constexpr,
    UF: cutlass.Constexpr, WPB: cutlass.Constexpr, CPW: cutlass.Constexpr,
    CT: cutlass.Constexpr, BT: cutlass.Constexpr, KSR: cutlass.Constexpr,
    KTR: cutlass.Constexpr, LPT: cutlass.Constexpr, TCOLS: cutlass.Constexpr,
    it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr, Mf: cutlass.Constexpr, eps: cutlass.Constexpr,
    has_beta: cutlass.Constexpr, update_running: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    w = tid // 32
    lane = tid % 32
    NV: cutlass.Constexpr = S // VS  # lane steps per row (== S on the scalar path)
    STEP: cutlass.Constexpr = UF * 32
    CSZ: cutlass.Constexpr = WPB * CPW * KSR * S

    smem = SmemAllocator()
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CT), byte_alignment=16)
    part = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CT), byte_alignment=16)
    sc = None
    if cutlass.const_expr(KSR > 0):
        cty: cutlass.Constexpr = et if VS == 1 else it_ty
        sc = smem.allocate_tensor(cty, cute.make_layout(CSZ), byte_alignment=16)
    # TMEM tier: 256KB/SM that does NOT come out of the smem/L1 carveout, so it raises
    # the cached fraction without the L1 starvation that capped the smem tier. Columns
    # per thread = CPW * LPT * VS (one fp32 column per element).
    tptr = None
    if cutlass.const_expr(KTR > 0):
        tptr = smem.allocate_tensor(cutlass.Int32, cute.make_layout(4), byte_alignment=16)
        if tid < 32:
            nvvm.tcgen05_alloc(tptr.iterator, TCOLS, is_exclusive=False)
            nvvm.tcgen05_relinquish_alloc_permit()
        nvvm.barrier_cta_sync_aligned(0)

    c0 = cx * CT
    per = (N + mparts - 1) // mparts
    n0 = my * per
    n1 = n0 + per
    if n1 > N:
        n1 = N
    nt = n0 + KTR
    if nt > n1:
        nt = n1
    nc = nt + KSR
    if nc > n1:
        nc = n1

    acc = [cutlass.Float32(0.0)] * CPW
    accsq = [cutlass.Float32(0.0)] * CPW

    # ---- pass1t: the TMEM-cached image rows.
    # The row loop is a CONSTEXPR LPT sweep with a clamped index rather than the
    # dynamic strip+tail: tcgen05 ld/st are warp-ALIGNED, and the tail loop's bound
    # (kv = kb + lane) is divergent, which would hang. Clamp, store unconditionally,
    # predicate only the accumulate. ----
    if cutlass.const_expr(KTR > 0):
        warpg = tid // 32
        wcol = (warpg // 4) * (KTR * CPW * LPT * VS)
        n = n0
        r = cutlass.Int32(0)
        while n < nt:
            bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
            for cw in cutlass.range_constexpr(CPW):
                for j in cutlass.range_constexpr(LPT):
                    kv = j * 32 + lane
                    kvc = kv
                    if kvc > (NV - 1):
                        kvc = NV - 1
                    if cutlass.const_expr(VS == 1):
                        fs = [mX[bs[cw] + kvc].to(cutlass.Float32)]
                    else:
                        tin = nvvm.load_ext(mXi.iterator + (bs[cw] + kvc * VS),
                                            dtype=it_ty, count=VS).bitcast(et)
                        fs = [tin[e].to(cutlass.Float32) for e in range(VS)]
                    if kv < NV:
                        for e in cutlass.range_constexpr(VS):
                            acc[cw] = acc[cw] + fs[e]
                            accsq[cw] = accsq[cw] + fs[e] * fs[e]
                    col = wcol + (((r * CPW + cw) * LPT) + j) * VS
                    tpj = nvvm.make_tmem_ptr_from_warp_row_col(
                        tptr[0], warpg % 4, col, cutlass.Float32)
                    nvvm.tcgen05_st(nvvm.Tcgen05LdStShape.SHAPE_32X32B, tpj,
                                    cutlass.Vector.from_elements(tuple(fs), cutlass.Float32))
            n = n + 1
            r = r + 1
        nvvm.tcgen05_wait(nvvm.Tcgen05Wait.STORE)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

    # ---- pass1a: the shared-memory-cached image rows ----
    # The CPW channel unroll lives INSIDE the element loop so all UF*CPW loads land
    # in one basic block. With the channel loop on the outside each channel sat in
    # its own dynamic loop and contributed no memory-level parallelism at all.
    if cutlass.const_expr(KSR > 0):
        n = nt   # smem tier starts AFTER the TMEM-cached images
        r = cutlass.Int32(0)
        while n < nc:
            bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
            cbs = [((w * CPW + cw) * KSR + r) * S for cw in range(CPW)]
            kb = cutlass.Int32(0)
            while kb + STEP <= NV:
                raws = []
                for j in cutlass.range_constexpr(UF):
                    for cw in cutlass.range_constexpr(CPW):
                        o = kb + j * 32 + lane
                        if cutlass.const_expr(VS == 1):
                            raws.append(mX[bs[cw] + o])
                        else:
                            raws.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS),
                                                      dtype=it_ty, count=VS))
                i = 0
                for j in cutlass.range_constexpr(UF):
                    for cw in cutlass.range_constexpr(CPW):
                        o = kb + j * 32 + lane
                        if cutlass.const_expr(VS == 1):
                            sc[cbs[cw] + o] = raws[i]
                            x = raws[i].to(cutlass.Float32)
                            acc[cw] = acc[cw] + x
                            accsq[cw] = accsq[cw] + x * x
                        else:
                            nvvm.store_ext(raws[i], sc.iterator + (cbs[cw] + o * VS),
                                           shared_space=_CTA_SS)
                            xv = raws[i].bitcast(et)
                            for e in cutlass.range_constexpr(VS):
                                x = xv[e].to(cutlass.Float32)
                                acc[cw] = acc[cw] + x
                                accsq[cw] = accsq[cw] + x * x
                        i = i + 1
                kb = kb + STEP
            kv = kb + lane
            while kv < NV:
                for cw in cutlass.range_constexpr(CPW):
                    if cutlass.const_expr(VS == 1):
                        raw = mX[bs[cw] + kv]
                        sc[cbs[cw] + kv] = raw
                        x = raw.to(cutlass.Float32)
                        acc[cw] = acc[cw] + x
                        accsq[cw] = accsq[cw] + x * x
                    else:
                        raw = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS)
                        nvvm.store_ext(raw, sc.iterator + (cbs[cw] + kv * VS), shared_space=_CTA_SS)
                        xv = raw.bitcast(et)
                        for e in cutlass.range_constexpr(VS):
                            x = xv[e].to(cutlass.Float32)
                            acc[cw] = acc[cw] + x
                            accsq[cw] = accsq[cw] + x * x
                kv = kv + 32
            n = n + 1
            r = r + 1

    # ---- pass1b: the streamed (uncached) image rows ----
    n = nc
    while n < n1:
        bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
        kb = cutlass.Int32(0)
        while kb + STEP <= NV:
            raws = []
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        raws.append(mX[bs[cw] + o].to(cutlass.Float32))
                    else:
                        raws.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS),
                                                  dtype=it_ty, count=VS).bitcast(et))
            i = 0
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    if cutlass.const_expr(VS == 1):
                        acc[cw] = acc[cw] + raws[i]
                        accsq[cw] = accsq[cw] + raws[i] * raws[i]
                    else:
                        for e in cutlass.range_constexpr(VS):
                            x = raws[i][e].to(cutlass.Float32)
                            acc[cw] = acc[cw] + x
                            accsq[cw] = accsq[cw] + x * x
                    i = i + 1
            kb = kb + STEP
        kv = kb + lane
        while kv < NV:
            for cw in cutlass.range_constexpr(CPW):
                if cutlass.const_expr(VS == 1):
                    x = mX[bs[cw] + kv].to(cutlass.Float32)
                    acc[cw] = acc[cw] + x
                    accsq[cw] = accsq[cw] + x * x
                else:
                    xv = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    for e in cutlass.range_constexpr(VS):
                        x = xv[e].to(cutlass.Float32)
                        acc[cw] = acc[cw] + x
                        accsq[cw] = accsq[cw] + x * x
            kv = kv + 32
        n = n + 1

    for cw in cutlass.range_constexpr(CPW):
        a, b = _warp_sum2(acc[cw], accsq[cw])
        if lane == 0:
            part[cw * WPB + w] = a
            part[CT + cw * WPB + w] = b
    nvvm.barrier_cta_sync_aligned(0)

    # ---- this CTA's partials -> its own slot in [cparts, 2, mparts, CT] ----
    if tid < 2 * CT:
        h = tid // CT
        ct = tid % CT
        mP[cx * (2 * mparts * CT) + h * (mparts * CT) + my * CT + ct] = part[tid]

    # ---- grid barrier ----
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

    # ---- finalize: only mparts*2*CT floats per CTA, so redundancy is free ----
    if tid < CT:
        ssum = cutlass.Float32(0.0)
        ssq = cutlass.Float32(0.0)
        p = cutlass.Int32(0)
        while p < mparts:
            ssum = ssum + mP[cx * (2 * mparts * CT) + p * CT + tid]
            ssq = ssq + mP[cx * (2 * mparts * CT) + mparts * CT + p * CT + tid]
            p = p + 1
        mn = ssum / Mf
        var = ssq / Mf - mn * mn
        rs = cute.math.rsqrt(var + eps)
        stat[tid] = mn
        stat[CT + tid] = rs
        if my == 0:  # one CTA per channel tile publishes the statistics
            mSavedMean[c0 + tid] = mn
            mSavedRstd[c0 + tid] = rs
            if cutlass.const_expr(update_running):
                # PyTorch semantics: running_var tracks the UNBIASED estimator.
                unb = var * Mf / (Mf - 1.0)
                mRunMean[c0 + tid] = (1.0 - momentum) * mRunMean[c0 + tid] + momentum * mn
                mRunVar[c0 + tid] = (1.0 - momentum) * mRunVar[c0 + tid] + momentum * unb
    nvvm.barrier_cta_sync_aligned(0)

    # ---- pass2: one channel per row -> scalar affine staged in registers ----
    scale = [cutlass.Float32(0.0)] * CPW
    shift = [cutlass.Float32(0.0)] * CPW
    for cw in cutlass.range_constexpr(CPW):
        ci = cw * WPB + w
        sfac = stat[CT + ci] * mG[c0 + ci].to(cutlass.Float32)
        scale[cw] = sfac
        sh = -stat[ci] * sfac
        if cutlass.const_expr(has_beta):
            sh = sh + mB[c0 + ci].to(cutlass.Float32)
        shift[cw] = sh

    # Cached rows and re-read rows are separate loops: selecting the source inside a
    # single loop would need a value to escape a staged ``if``.
    if cutlass.const_expr(KTR > 0):
        warpg = tid // 32
        wcol = (warpg // 4) * (KTR * CPW * LPT * VS)
        n = n0
        r = cutlass.Int32(0)
        while n < nt:
            bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
            for cw in cutlass.range_constexpr(CPW):
                for j in cutlass.range_constexpr(LPT):
                    kv = j * 32 + lane
                    col = wcol + (((r * CPW + cw) * LPT) + j) * VS
                    tpj = nvvm.make_tmem_ptr_from_warp_row_col(
                        tptr[0], warpg % 4, col, cutlass.Float32)
                    tout = nvvm.tcgen05_ld(nvvm.Tcgen05LdStShape.SHAPE_32X32B, tpj, num=VS)
                    if kv < NV:
                        if cutlass.const_expr(VS == 1):
                            mY[bs[cw] + kv] = (tout[0] * scale[cw] + shift[cw]).to(et)
                        else:
                            ys = []
                            for e in cutlass.range_constexpr(VS):
                                ys.append((tout[e] * scale[cw] + shift[cw]).to(et))
                            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                           mYi.iterator + (bs[cw] + kv * VS))
            n = n + 1
            r = r + 1
        nvvm.tcgen05_wait(nvvm.Tcgen05Wait.LOAD)

    if cutlass.const_expr(KSR > 0):
        n = nt
        r = cutlass.Int32(0)
        while n < nc:
            bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
            cbs = [((w * CPW + cw) * KSR + r) * S for cw in range(CPW)]
            kb = cutlass.Int32(0)
            while kb + STEP <= NV:
                for j in cutlass.range_constexpr(UF):
                    for cw in cutlass.range_constexpr(CPW):
                        o = kb + j * 32 + lane
                        if cutlass.const_expr(VS == 1):
                            y = sc[cbs[cw] + o].to(cutlass.Float32) * scale[cw] + shift[cw]
                            mY[bs[cw] + o] = y.to(et)
                        else:
                            xv = nvvm.load_ext(sc.iterator + (cbs[cw] + o * VS), dtype=it_ty,
                                               count=VS, shared_space=_CTA_SS).bitcast(et)
                            ys = []
                            for e in cutlass.range_constexpr(VS):
                                ys.append((xv[e].to(cutlass.Float32) * scale[cw] + shift[cw]).to(et))
                            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                           mYi.iterator + (bs[cw] + o * VS))
                kb = kb + STEP
            kv = kb + lane
            while kv < NV:
                for cw in cutlass.range_constexpr(CPW):
                    if cutlass.const_expr(VS == 1):
                        y = sc[cbs[cw] + kv].to(cutlass.Float32) * scale[cw] + shift[cw]
                        mY[bs[cw] + kv] = y.to(et)
                    else:
                        xv = nvvm.load_ext(sc.iterator + (cbs[cw] + kv * VS), dtype=it_ty,
                                           count=VS, shared_space=_CTA_SS).bitcast(et)
                        ys = []
                        for e in cutlass.range_constexpr(VS):
                            ys.append((xv[e].to(cutlass.Float32) * scale[cw] + shift[cw]).to(et))
                        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                       mYi.iterator + (bs[cw] + kv * VS))
                kv = kv + 32
            n = n + 1
            r = r + 1

    n = nc
    while n < n1:
        bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
        kb = cutlass.Int32(0)
        while kb + STEP <= NV:
            raws = []
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        raws.append(mX[bs[cw] + o].to(cutlass.Float32))
                    else:
                        raws.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS),
                                                  dtype=it_ty, count=VS).bitcast(et))
            i = 0
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        mY[bs[cw] + o] = (raws[i] * scale[cw] + shift[cw]).to(et)
                    else:
                        ys = []
                        for e in cutlass.range_constexpr(VS):
                            ys.append((raws[i][e].to(cutlass.Float32) * scale[cw] + shift[cw]).to(et))
                        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                       mYi.iterator + (bs[cw] + o * VS))
                    i = i + 1
            kb = kb + STEP
        kv = kb + lane
        while kv < NV:
            for cw in cutlass.range_constexpr(CPW):
                if cutlass.const_expr(VS == 1):
                    y = mX[bs[cw] + kv].to(cutlass.Float32) * scale[cw] + shift[cw]
                    mY[bs[cw] + kv] = y.to(et)
                else:
                    xv = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    ys = []
                    for e in cutlass.range_constexpr(VS):
                        ys.append((xv[e].to(cutlass.Float32) * scale[cw] + shift[cw]).to(et))
                    nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                   mYi.iterator + (bs[cw] + kv * VS))
            kv = kv + 32
        n = n + 1

    if cutlass.const_expr(KTR > 0):
        nvvm.barrier_cta_sync_aligned(0)
        if tid < 32:
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tptr[0], cutlass.Float32),
                                 TCOLS, is_exclusive=False)


@cute.jit
def _bn_nchw_host(
    mX, mY, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
    N, mparts, momentum,
    C: cutlass.Constexpr, S: cutlass.Constexpr, VS: cutlass.Constexpr,
    UF: cutlass.Constexpr, WPB: cutlass.Constexpr, CPW: cutlass.Constexpr,
    CT: cutlass.Constexpr, BT: cutlass.Constexpr, KSR: cutlass.Constexpr,
    KTR: cutlass.Constexpr, LPT: cutlass.Constexpr, TCOLS: cutlass.Constexpr,
    cparts: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
    update_running: cutlass.Constexpr, smem_bytes: cutlass.Constexpr,
    mbpm: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _bn_nchw_kernel(
        mX, mXi, mY, mYi, mG, mB, mP, mRet, mSavedMean, mSavedRstd, mRunMean, mRunVar,
        N, mparts, momentum, C, S, VS, UF, WPB, CPW, CT, BT, KSR, KTR, LPT, TCOLS, it_ty, et, Mf, eps,
        has_beta, update_running,
    ).launch(grid=(cparts, mparts, 1), block=(BT, 1, 1), smem=smem_bytes,
             cooperative=True, min_blocks_per_mp=mbpm)


# ---------------------------------------------------------------------------
# Inference kernel (no reduction, no cooperative launch)
# ---------------------------------------------------------------------------


@cute.kernel
def _bn_nchw_infer_kernel(
    mX, mXi, mY, mYi, mG, mB, mRunMean, mRunVar, N: cutlass.Int32,
    C: cutlass.Constexpr, S: cutlass.Constexpr, VS: cutlass.Constexpr,
    UF: cutlass.Constexpr, WPB: cutlass.Constexpr, CPW: cutlass.Constexpr,
    CT: cutlass.Constexpr, BT: cutlass.Constexpr, it_ty: cutlass.Constexpr,
    et: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    _, nblk, _ = cute.arch.grid_dim()
    w = tid // 32
    lane = tid % 32
    NV: cutlass.Constexpr = S // VS
    STEP: cutlass.Constexpr = UF * 32
    c0 = cx * CT
    scale = [cutlass.Float32(0.0)] * CPW
    shift = [cutlass.Float32(0.0)] * CPW
    for cw in cutlass.range_constexpr(CPW):
        ci = cw * WPB + w
        rs = cute.math.rsqrt(mRunVar[c0 + ci].to(cutlass.Float32) + eps)
        sfac = rs * mG[c0 + ci].to(cutlass.Float32)
        scale[cw] = sfac
        sh = -mRunMean[c0 + ci].to(cutlass.Float32) * sfac
        if cutlass.const_expr(has_beta):
            sh = sh + mB[c0 + ci].to(cutlass.Float32)
        shift[cw] = sh
    n = my
    while n < N:
        bs = [(cutlass.Int64(n) * C + (c0 + cw * WPB + w)) * S for cw in range(CPW)]
        kb = cutlass.Int32(0)
        while kb + STEP <= NV:
            raws = []
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        raws.append(mX[bs[cw] + o].to(cutlass.Float32))
                    else:
                        raws.append(nvvm.load_ext(mXi.iterator + (bs[cw] + o * VS),
                                                  dtype=it_ty, count=VS).bitcast(et))
            i = 0
            for j in cutlass.range_constexpr(UF):
                for cw in cutlass.range_constexpr(CPW):
                    o = kb + j * 32 + lane
                    if cutlass.const_expr(VS == 1):
                        mY[bs[cw] + o] = (raws[i] * scale[cw] + shift[cw]).to(et)
                    else:
                        ys = []
                        for e in cutlass.range_constexpr(VS):
                            ys.append((raws[i][e].to(cutlass.Float32) * scale[cw] + shift[cw]).to(et))
                        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                       mYi.iterator + (bs[cw] + o * VS))
                    i = i + 1
            kb = kb + STEP
        kv = kb + lane
        while kv < NV:
            for cw in cutlass.range_constexpr(CPW):
                if cutlass.const_expr(VS == 1):
                    y = mX[bs[cw] + kv].to(cutlass.Float32) * scale[cw] + shift[cw]
                    mY[bs[cw] + kv] = y.to(et)
                else:
                    xv = nvvm.load_ext(mXi.iterator + (bs[cw] + kv * VS), dtype=it_ty, count=VS).bitcast(et)
                    ys = []
                    for e in cutlass.range_constexpr(VS):
                        ys.append((xv[e].to(cutlass.Float32) * scale[cw] + shift[cw]).to(et))
                    nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty),
                                   mYi.iterator + (bs[cw] + kv * VS))
            kv = kv + 32
        n = n + nblk


@cute.jit
def _bn_nchw_infer_host(
    mX, mY, mG, mB, mRunMean, mRunVar, N,
    C: cutlass.Constexpr, S: cutlass.Constexpr, VS: cutlass.Constexpr,
    UF: cutlass.Constexpr, WPB: cutlass.Constexpr, CPW: cutlass.Constexpr,
    CT: cutlass.Constexpr, BT: cutlass.Constexpr, cparts: cutlass.Constexpr,
    nblk: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
    eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
) -> None:
    mXi = cute.recast_tensor(mX, it_ty)
    mYi = cute.recast_tensor(mY, it_ty)
    _bn_nchw_infer_kernel(mX, mXi, mY, mYi, mG, mB, mRunMean, mRunVar, N, C, S, VS,
                          UF, WPB, CPW, CT, BT, it_ty, et, eps, has_beta).launch(
        grid=(cparts, nblk, 1), block=(BT, 1, 1))


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

_KCACHE = {}
_OCC = {}


def _pow2_up(n: int) -> int:
    p = 32
    while p < n:
        p *= 2
    return p


def _tmem_alloc_cols(BT: int, KTR: int, CPW: int, LPT: int, VS: int) -> int:
    """Columns tcgen05_alloc must be asked for: a power of 2 in [32, 512]."""
    return _pow2_up(max(1, BT // 128) * KTR * CPW * LPT * VS)


# MEASURED: the TMEM tier is a net LOSS for the NCHW row map, so it is OFF by
# default (the code path is kept and still validates). Isolated at N=128 by capping
# KTR: 2048x7x7 1794 -> 988 GB/s (-45%), 512x14x14 2365 -> 2053 (-13%), 64x56x56
# neutral. Why it loses here but helps NHWC: this kernel's TMEM traffic is far more
# fine-grained (one fp32 column per ELEMENT, and VS=1 on odd S, so 7x7 stores a
# column per 2-byte input), and the constexpr LPT sweep the warp-aligned tcgen05 ops
# require re-loads clamped tail lanes. On top of that, enforcing the per-CTA TMEM
# budget needs shared memory padded to pin occupancy, which costs L1 -- the same
# L1-starvation that capped the plain smem cache.
_USE_TMEM = False


def _tmem_rows(BT: int, CPW: int, LPT: int, VS: int, occ: int) -> int:
    """Image rows per thread that fit this CTA's share of TMEM (one fp32 column per
    element), AFTER the power-of-2 rounding tcgen05_alloc requires. Over-requesting
    BLOCKS in tcgen05_alloc, which in a cooperative grid is a hang -- so this is a
    correctness bound, not a knob."""
    if not _USE_TMEM:
        return 0
    budget = 512 // occ
    best = 0
    k = 1
    while True:
        need = _tmem_alloc_cols(BT, k, CPW, LPT, VS)
        if need > budget or need > 512:
            break
        best = k
        k += 1
    return best


def _ksr(nloc: int, WPB: int, CPW: int, S: int, eb: int, occ: int) -> int:
    """Whole image-rows per warp to cache in shared, bounded by the smem budget."""
    per_row = WPB * CPW * S * eb
    if per_row == 0:
        return 0
    return max(0, min(nloc, (_SMEM_CAP // occ) // per_row))


def forward(spec, x3d, gamma, beta, *, eps, momentum, training, running_mean,
            running_var, cfg, params, knobs=None):
    """Launch the NCHW BatchNorm forward on ``x3d`` viewed as ``[N, C, S]``.

    Returns ``(y3d, saved_mean, saved_rstd)``.
    """
    import torch

    N, C, S = int(spec.N), int(spec.C), int(spec.S)
    eb = DTYPE_BYTES[params.io_dtype]
    et = DTYPE_TO_CUTLASS[params.io_dtype]
    it_ty = _INT_TY[eb]
    geo = nchw_cfg(C, N, S, eb)
    if geo is None:
        raise ValueError(f"NCHW BatchNorm: C={C} S={S} unsupported by this kernel")
    VS, UF, WPB, CPW, CT, cparts, BT = geo
    has_beta = beta is not None
    if beta is None:
        beta = gamma

    y3d = torch.empty_like(x3d)
    xf = x3d.reshape(-1)
    yf = y3d.reshape(-1)
    saved_mean = torch.empty(C, dtype=torch.float32, device=x3d.device)
    saved_rstd = torch.empty(C, dtype=torch.float32, device=x3d.device)

    if not training:
        rm = running_mean if running_mean is not None else torch.zeros(C, dtype=torch.float32, device=x3d.device)
        rv = running_var if running_var is not None else torch.ones(C, dtype=torch.float32, device=x3d.device)
        nblk = max(1, min(_nsm() * 4 // max(1, cparts), N))
        ce = (C, S, VS, UF, WPB, CPW, CT, BT, cparts, nblk, it_ty, et, float(eps), has_beta)
        key = ("infer", params.io_dtype, C, S, VS, UF, CPW, nblk, has_beta)
        args = (dyn(xf), dyn(yf), dyn(gamma), dyn(beta), dyn(rm), dyn(rv), cutlass.Int32(N))
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_bn_nchw_infer_host, *args, *ce)
            _KCACHE[key] = fn
        fn(*args)
        saved_mean.copy_(rm.float())
        saved_rstd.copy_((rv.float() + eps).rsqrt())
        return y3d, saved_mean, saved_rstd

    update_running = bool(running_mean is not None and running_var is not None)
    rm = running_mean if update_running else saved_mean
    rv = running_var if update_running else saved_rstd
    count = float(N * S)

    NV = S // VS
    LPT = (NV + 31) // 32          # lane steps per row (constexpr sweep for the TMEM tier)
    okey = (params.io_dtype, C, S, N, has_beta, update_running)

    # Try occ=2 first. A coverage-driven choice (prefer whichever occupancy caches
    # more) was tried and MEASURED WORSE -- it pushed 64x56x56 to occ=1 for one extra
    # cached image and lost 3035 -> 2589 GB/s, because the CTA count matters more
    # than the marginal cached row.
    occ0 = _OCC.get(okey, 2)
    for occ in ([2, 1] if occ0 == 2 else [1]):
        mparts = _split(N, cparts, occ)
        nloc = (N + mparts - 1) // mparts
        # TMEM first (it is free of L1 pressure), then shared for whatever is left.
        KTR = min(nloc, _tmem_rows(BT, CPW, LPT, VS, occ)) if knobs is None else 0
        KSR = (_ksr(nloc - KTR, WPB, CPW, S, eb, occ) if knobs is None else int(knobs))
        smem_bytes = 2 * (2 * CT) * 4 + WPB * CPW * KSR * S * eb + 128
        mbpm = 2 if occ == 2 else 0
        TCOLS = _tmem_alloc_cols(BT, KTR, CPW, LPT, VS) if KTR > 0 else 0
        if TCOLS > 0:
            # A cooperative launch guarantees co-residency but NOT an even CTA
            # distribution, so a 3rd CTA can land on an SM whose TMEM is already
            # spoken for; its tcgen05_alloc then fails and leaves a garbage base
            # address -> illegal access. Pad shared memory so it is the occupancy
            # limiter and the per-CTA TMEM budget is actually enforceable.
            smem_bytes = max(smem_bytes, _SMEM_PER_SM // (occ + 1) + 1)
        ce = (C, S, VS, UF, WPB, CPW, CT, BT, KSR, KTR, LPT, TCOLS, cparts, it_ty, et,
              count, float(eps), has_beta, update_running, smem_bytes, mbpm)
        key = (params.io_dtype, C, S, VS, UF, CPW, KSR, KTR, LPT, TCOLS, has_beta,
               update_running, mbpm)
        pbuf = torch.empty(cparts * 2 * mparts * CT, dtype=torch.float32, device=x3d.device)
        ret = torch.zeros(cparts, dtype=torch.int32, device=x3d.device)
        args = (dyn(xf), dyn(yf), dyn(gamma), dyn(beta), dyn(pbuf), dyn(ret),
                dyn(saved_mean), dyn(saved_rstd), dyn(rm), dyn(rv),
                cutlass.Int32(N), cutlass.Int32(mparts), cutlass.Float32(momentum))
        fn = _KCACHE.get(key)
        if fn is None:
            fn = cute.compile(_bn_nchw_host, *args, *ce)
            _KCACHE[key] = fn
        try:
            fn(*args)
            _OCC[okey] = occ
            return y3d, saved_mean, saved_rstd
        except Exception as e:
            if "TOO_LARGE" in str(e) and occ == 2:
                continue
            raise
    return y3d, saved_mean, saved_rstd
