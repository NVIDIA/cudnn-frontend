# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared indices-only BF16 Top-K; device compute provenance: varlen_provenance.json.

Imported only after varlen_api checks the installed DSL and actual target.
The reviewed device function below is preserved byte-for-byte from safe8652.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.utils import SmemAllocator
from cudnn.frost.compiled_cache import compile_cached, template_key

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# fmt: off
THREADS = 1024
NWARP = THREADS // 32
UNROLL = 8
TILE = THREADS * UNROLL
NBIN1 = 2048          # 11-bit first radix digit
NBIN2 = 32            # 5-bit second digit


@cute.kernel
def _topk_kernel(
    mS: cute.Tensor,
    mLen: cute.Tensor,
    mOut: cute.Tensor,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    NN: cutlass.Constexpr,
    CR: cutlass.Constexpr,
    PACK: cutlass.Constexpr,
    NSLOT: cutlass.Constexpr,
    CACHE: cutlass.Constexpr,
    POW2: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    lane = tid % 32
    warp = tid // 32

    smem = SmemAllocator()
    hist = smem.allocate_array(cutlass.Int32, NBIN1)
    wsum = smem.allocate_array(cutlass.Int32, 64)
    ctrl = smem.allocate_array(cutlass.Int32, 16)
    cand = smem.allocate_array(cutlass.Int32, CAP * THREADS)
    if cutlass.const_expr(CACHE):
        kc = smem.allocate_array(cutlass.Int32, NSLOT)
    else:
        kc = smem.allocate_array(cutlass.Int32, 1)

    wreg = cute.make_rmem_tensor((UNROLL,), cutlass.Int32)
    one = cutlass.Int32(1)

    # ---- effective row length (64-bit metadata arithmetic) --------------------
    req = r // NN
    off = r % NN
    raw = mLen[req].to(cutlass.Int64) - cutlass.Int64(NN - 1) + off.to(cutlass.Int64)
    L = cutlass.Int32(0)
    if raw > cutlass.Int64(0):
        q = raw // cutlass.Int64(CR)
        if q > cutlass.Int64(N):
            q = cutlass.Int64(N)
        L = q.to(cutlass.Int32)

    # ---- trivial answer: index p when p < L, else -1 -------------------------
    for p0 in cutlass.range((K + THREADS - 1) // THREADS):
        p = p0 * THREADS + tid
        if p < K:
            v = cutlass.Int32(-1)
            if p < L:
                v = p
            mOut[r, p] = v

    for z in cutlass.range_constexpr(NBIN1 // THREADS):
        hist[z * THREADS + tid] = cutlass.Int32(0)
    if tid < 16:
        ctrl[tid] = cutlass.Int32(0)

    # rows with L <= K are already answered; make every following loop zero-trip.
    Lr = cutlass.Int32(0)
    if L > K:
        Lr = L
    if cutlass.const_expr(PACK):
        ns = (Lr + 1) // 2
        ns2 = Lr >> 1
    else:
        ns = Lr
        ns2 = Lr
    smax = ns - 1
    ni = (ns + (TILE - 1)) // TILE
    nfull = ns2 // TILE
    tstart = nfull * TILE
    ntail = cutlass.Int32(0)
    if ns > tstart:
        ntail = cutlass.Int32(1)

    cute.arch.barrier()

    # ================= pass 1: raw first-digit histogram ======================
    for j in cutlass.range(nfull):
        bs = j * TILE + tid
        for m in cutlass.range_constexpr(UNROLL):
            wreg[m] = mS[r, bs + m * THREADS].to(cutlass.Int32)
        for m in cutlass.range_constexpr(UNROLL):
            w = wreg[m]
            if cutlass.const_expr(CACHE):
                kc[bs + m * THREADS] = w
            cute.arch.atomic_add(hist + ((w >> 5) & 0x7FF), one)
            if cutlass.const_expr(PACK):
                cute.arch.atomic_add(hist + ((w >> 21) & 0x7FF), one)
    for j in cutlass.range(ntail):
        bs = tstart + tid
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if cutlass.const_expr(POW2):
                cm = sm & (NSLOT - 1)
            else:
                dm = smax - sm
                cm = sm + (dm & (dm >> 31))
            wreg[m] = mS[r, cm].to(cutlass.Int32)
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if sm < ns:
                w = wreg[m]
                if cutlass.const_expr(CACHE):
                    kc[sm] = w
                cute.arch.atomic_add(hist + ((w >> 5) & 0x7FF), one)
                if cutlass.const_expr(PACK):
                    if sm < ns2:
                        cute.arch.atomic_add(hist + ((w >> 21) & 0x7FF), one)
    cute.arch.barrier()

    # ---- block-wide suffix scan over NBIN1 bins, in float order --------------
    # sorted position p maps back to raw bin  p-1024 (p>=1024) or 2047-p (p<1024)
    p0 = 2 * tid
    c0 = cutlass.Int32(0)
    c1 = cutlass.Int32(0)
    if p0 < 1024:
        c0 = hist[2047 - p0]
        c1 = hist[2046 - p0]
    else:
        c0 = hist[p0 - 1024]
        c1 = hist[p0 - 1023]
    g = c0 + c1
    x = g
    for sh in cutlass.range_constexpr(5):
        y = cute.arch.shuffle_sync_down(x, 1 << sh)
        if lane + (1 << sh) < 32:
            x = x + y
    if lane == 0:
        wsum[warp] = x
    cute.arch.barrier()
    if warp == 0:
        t0 = wsum[lane]
        u = t0
        for sh in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_down(u, 1 << sh)
            if lane + (1 << sh) < 32:
                u = u + y
        wsum[lane] = u - t0
    cute.arch.barrier()
    after1 = wsum[warp] + x - g
    s1 = after1 + c1
    s0 = s1 + c0
    if s1 >= K:
        if after1 < K:
            ctrl[0] = 2 * tid + 1
            ctrl[1] = after1
    if s0 >= K:
        if s1 < K:
            ctrl[0] = 2 * tid
            ctrl[1] = s1
    if tid < NBIN2:
        hist[tid] = cutlass.Int32(0)
    cute.arch.barrier()

    b = ctrl[0]
    rem = K - ctrl[1]

    # bucket test "ordered bucket >= b" as a window on the raw 16-bit word
    tb = cutlass.Int32(0)
    LO = cutlass.Int32(0)
    HI = cutlass.Int32(0)
    if b >= 1024:
        tb = b - cutlass.Int32(1024)
        LO = tb << 5
        HI = cutlass.Int32(0x7FFF)
    else:
        tb = cutlass.Int32(2047) - b
        LO = cutlass.Int32(0)
        HI = (tb << 5) | cutlass.Int32(31)

    # ====== pass 2: second-digit histogram + candidate capture ===============
    nc = cutlass.Int32(0)
    for j in cutlass.range(nfull):
        bs = j * TILE + tid
        for m in cutlass.range_constexpr(UNROLL):
            if cutlass.const_expr(CACHE):
                wreg[m] = kc[bs + m * THREADS]
            else:
                wreg[m] = mS[r, bs + m * THREADS].to(cutlass.Int32)
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            w = wreg[m]
            if cutlass.const_expr(PACK):
                i0 = 2 * sm
            else:
                i0 = sm
            a0 = w & 0xFFFF
            if a0 >= LO:
                if a0 <= HI:
                    t = a0 >> 5
                    lf0 = a0 & 0x1F
                    if t >= 1024:
                        lf0 = lf0 ^ 0x1F
                    if t != tb:
                        lf0 = lf0 + 32
                    else:
                        cute.arch.atomic_add(hist + lf0, one)
                    if nc < CAP:
                        cand[nc * THREADS + tid] = i0 | (lf0 << 25)
                    nc = nc + 1
            if cutlass.const_expr(PACK):
                a1 = (w >> 16) & 0xFFFF
                if a1 >= LO:
                    if a1 <= HI:
                        t = a1 >> 5
                        lf1 = a1 & 0x1F
                        if t >= 1024:
                            lf1 = lf1 ^ 0x1F
                        if t != tb:
                            lf1 = lf1 + 32
                        else:
                            cute.arch.atomic_add(hist + lf1, one)
                        if nc < CAP:
                            cand[nc * THREADS + tid] = (i0 + 1) | (lf1 << 25)
                        nc = nc + 1
    for j in cutlass.range(ntail):
        bs = tstart + tid
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if cutlass.const_expr(POW2):
                cm = sm & (NSLOT - 1)
            else:
                dm = smax - sm
                cm = sm + (dm & (dm >> 31))
            if cutlass.const_expr(CACHE):
                dm = smax - cm
                cm = cm + (dm & (dm >> 31))
                wreg[m] = kc[cm]
            else:
                wreg[m] = mS[r, cm].to(cutlass.Int32)
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if sm < ns:
                w = wreg[m]
                if cutlass.const_expr(PACK):
                    i0 = 2 * sm
                else:
                    i0 = sm
                a0 = w & 0xFFFF
                if a0 >= LO:
                    if a0 <= HI:
                        t = a0 >> 5
                        lf0 = a0 & 0x1F
                        if t >= 1024:
                            lf0 = lf0 ^ 0x1F
                        if t != tb:
                            lf0 = lf0 + 32
                        else:
                            cute.arch.atomic_add(hist + lf0, one)
                        if nc < CAP:
                            cand[nc * THREADS + tid] = i0 | (lf0 << 25)
                        nc = nc + 1
                if cutlass.const_expr(PACK):
                    if sm < ns2:
                        a1 = (w >> 16) & 0xFFFF
                        if a1 >= LO:
                            if a1 <= HI:
                                t = a1 >> 5
                                lf1 = a1 & 0x1F
                                if t >= 1024:
                                    lf1 = lf1 ^ 0x1F
                                if t != tb:
                                    lf1 = lf1 + 32
                                else:
                                    cute.arch.atomic_add(hist + lf1, one)
                                if nc < CAP:
                                    cand[nc * THREADS + tid] = (i0 + 1) | (lf1 << 25)
                                nc = nc + 1
    if nc > CAP:
        cute.arch.atomic_add(ctrl + 5, one)
    cute.arch.barrier()

    if warp == 0:
        cc = hist[lane]
        u = cc
        for sh in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_down(u, 1 << sh)
            if lane + (1 << sh) < 32:
                u = u + y
        if u >= rem:
            if u - cc < rem:
                ctrl[2] = lane
                ctrl[3] = u - cc
    cute.arch.barrier()

    l = ctrl[2]
    T = (b << 5) | l
    abtot = ctrl[1] + ctrl[3]
    need2 = K - abtot
    ovf = ctrl[5]

    # candidate-list trip count, or 0 when we must fall back to a full re-scan
    ncl = nc
    if ovf != 0:
        ncl = cutlass.Int32(0)
    nfull2 = ni
    if ovf == 0:
        nfull2 = cutlass.Int32(0)

    # ================= count (key > T) and (key == T) =========================
    ca = cutlass.Int32(0)
    ce = cutlass.Int32(0)
    for j in cutlass.range(ncl):
        lf = cand[j * THREADS + tid] >> 25
        if lf > l:
            ca = ca + 1
        elif lf == l:
            ce = ce + 1
    for j in cutlass.range(nfull2):
        bs = j * TILE + tid
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if cutlass.const_expr(POW2):
                cm = sm & (NSLOT - 1)
            else:
                dm = smax - sm
                cm = sm + (dm & (dm >> 31))
            if cutlass.const_expr(CACHE):
                dm = smax - cm
                cm = cm + (dm & (dm >> 31))
                wreg[m] = kc[cm]
            else:
                wreg[m] = mS[r, cm].to(cutlass.Int32)
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if sm < ns:
                w = wreg[m]
                kw = w ^ ((((w >> 15) & 0x00010001) * 0xFFFF) | 0x80008000)
                k0 = kw & 0xFFFF
                if k0 > T:
                    ca = ca + 1
                elif k0 == T:
                    ce = ce + 1
                if cutlass.const_expr(PACK):
                    if sm < ns2:
                        k1 = (kw >> 16) & 0xFFFF
                        if k1 > T:
                            ca = ca + 1
                        elif k1 == T:
                            ce = ce + 1

    # ---- block exclusive scans of ca and ce ---------------------------------
    x = ca
    for sh in cutlass.range_constexpr(5):
        y = cute.arch.shuffle_sync_up(x, 1 << sh)
        if lane >= (1 << sh):
            x = x + y
    if lane == 31:
        wsum[warp] = x
    cute.arch.barrier()
    if warp == 0:
        t0 = wsum[lane]
        u = t0
        for sh in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_up(u, 1 << sh)
            if lane >= (1 << sh):
                u = u + y
        wsum[lane] = u - t0
    cute.arch.barrier()
    baseA = wsum[warp] + x - ca
    cute.arch.barrier()

    x = ce
    for sh in cutlass.range_constexpr(5):
        y = cute.arch.shuffle_sync_up(x, 1 << sh)
        if lane >= (1 << sh):
            x = x + y
    if lane == 31:
        wsum[warp] = x
    cute.arch.barrier()
    if warp == 0:
        t0 = wsum[lane]
        u = t0
        for sh in cutlass.range_constexpr(5):
            y = cute.arch.shuffle_sync_up(u, 1 << sh)
            if lane >= (1 << sh):
                u = u + y
        wsum[lane] = u - t0
    cute.arch.barrier()
    baseE = wsum[warp] + x - ce
    cute.arch.barrier()

    # ================= emit ===================================================
    pa = baseA
    pe = baseE
    for j in cutlass.range(ncl):
        v = cand[j * THREADS + tid]
        lf = v >> 25
        if lf > l:
            mOut[r, pa] = v & 0x1FFFFFF
            pa = pa + 1
        elif lf == l:
            if pe < need2:
                mOut[r, abtot + pe] = v & 0x1FFFFFF
            pe = pe + 1
    for j in cutlass.range(nfull2):
        bs = j * TILE + tid
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if cutlass.const_expr(POW2):
                cm = sm & (NSLOT - 1)
            else:
                dm = smax - sm
                cm = sm + (dm & (dm >> 31))
            if cutlass.const_expr(CACHE):
                dm = smax - cm
                cm = cm + (dm & (dm >> 31))
                wreg[m] = kc[cm]
            else:
                wreg[m] = mS[r, cm].to(cutlass.Int32)
        for m in cutlass.range_constexpr(UNROLL):
            sm = bs + m * THREADS
            if sm < ns:
                w = wreg[m]
                kw = w ^ ((((w >> 15) & 0x00010001) * 0xFFFF) | 0x80008000)
                k0 = kw & 0xFFFF
                if cutlass.const_expr(PACK):
                    i0 = 2 * sm
                else:
                    i0 = sm
                if k0 > T:
                    mOut[r, pa] = i0
                    pa = pa + 1
                elif k0 == T:
                    if pe < need2:
                        mOut[r, abtot + pe] = i0
                    pe = pe + 1
                if cutlass.const_expr(PACK):
                    if sm < ns2:
                        k1 = (kw >> 16) & 0xFFFF
                        if k1 > T:
                            mOut[r, pa] = i0 + 1
                            pa = pa + 1
                        elif k1 == T:
                            if pe < need2:
                                mOut[r, abtot + pe] = i0 + 1
                            pe = pe + 1

# fmt: on

_topk_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _prepared_host(
    scores: cute.Pointer,
    lengths: cute.Pointer,
    indices: cute.Pointer,
    rows: cutlass.Int32,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    NN: cutlass.Constexpr,
    CR: cutlass.Constexpr,
    PACK: cutlass.Constexpr,
    NSLOT: cutlass.Constexpr,
    CACHE: cutlass.Constexpr,
    POW2: cutlass.Constexpr,
    CAP: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    # Runtime row/batch extents do not specialize the device algorithm.
    mS = cute.make_tensor(scores, cute.make_layout((rows, NSLOT), stride=(NSLOT, 1)))
    mLen = cute.make_tensor(lengths, cute.make_layout((rows // NN,), stride=(1,)))
    mOut = cute.make_tensor(indices, cute.make_layout((rows, K), stride=(K, 1)))
    _topk_kernel(mS, mLen, mOut, N, K, NN, CR, PACK, NSLOT, CACHE, POW2, CAP).launch(grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def compile_topk(cols, top_k, next_n, compress_ratio, device_index, arch):
    key = template_key(globals(), locals(), "compile_topk")
    pack = cols % 2 == 0
    slots = cols // 2 if pack else cols
    capacity = 8 if top_k == 512 else 16
    cache = slots * 4 + capacity * THREADS * 4 <= 190000
    pow2 = slots & (slots - 1) == 0
    score_ptr = cute.runtime.make_ptr(cutlass.Int32 if pack else cutlass.Int16, 16, cute.AddressSpace.gmem, assumed_align=16)
    integer_ptr = cute.runtime.make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
    return compile_cached(
        _prepared_host,
        score_ptr,
        integer_ptr,
        integer_ptr,
        cutlass.Int32(0),
        cols,
        top_k,
        next_n,
        compress_ratio,
        pack,
        slots,
        cache,
        pow2,
        capacity,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
        cache_key=key,
        symbol="cudnn_indexer_top_k_varlen_bf16",
    )
