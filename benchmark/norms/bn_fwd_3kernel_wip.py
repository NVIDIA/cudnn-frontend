"""BatchNorm NHWC forward de-risk (cutlass primitives, sm_100). NHWC x flattens to
[M, C] (M=N*H*W pixels, C channels contiguous). Reduce over M per channel.

v1 (correctness + baseline): 3 kernels.
  A stats:  split-K over M x channel-tile. Thread map = cuDNN's: TPP=C_tile/V threads
            cover the contiguous C-tile (128-bit coalesced), PPL=BT/TPP pixels processed
            in parallel. Per-thread sum/sumsq over its M-slice, smem-reduce over the PPL
            pixel groups, atomicAdd the CTA partial to global sum/sumsq[C] (fp32).
  M meanrstd: mean=sum/M, rstd=rsqrt(sumsq/M - mean^2 + eps).
  B norm:   y = (x-mean)*rstd*gamma + beta, per-channel affine, coalesced.
Traffic 2R+1W -> ~2/3 of copy ceiling target. Single-pass (cache X) is a later step."""
import sys, types
D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python" / "cudnn")
stub = types.ModuleType("cudnn"); stub.__path__ = [D]; stub.pygraph = type("pygraph", (), {}); sys.modules["cudnn"] = stub
import cutlass, cutlass.cute as cute, cutlass.primitives as nvvm, torch, numpy as np
from torch.profiler import profile, ProfilerActivity, record_function
from cutlass.memory import SmemAllocator
from cutlass.cute.runtime import from_dlpack
_CTA = nvvm.SharedSpace.shared_cta
def _dyn(t): return from_dlpack(t, assumed_align=16).mark_layout_dynamic()
_C = {}


@cute.kernel
def _bn_stats(mXi, mPart, M: cutlass.Int32, mparts: cutlass.Int32,
              C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr,
              PPL: cutlass.Constexpr, BT: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BT * V * 2), byte_alignment=16)
    tc = (tid % TPP) * V             # base channel for this thread (within tile)
    tp = tid // TPP                  # which pixel-in-parallel
    c0 = cx * (TPP * V) + tc         # global base channel
    # M-slice for this (cx,my)
    per = (M + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > M:
        r1 = M
    s = [cutlass.Float32(0.0)] * V
    sq = [cutlass.Float32(0.0)] * V
    UNR: cutlass.Constexpr = 4
    row = r0 + tp
    # main loop: load UNR pixels into registers first (independent loads -> ILP keeps
    # memory saturated), THEN accumulate. Fixes the dependent per-pixel chain.
    while row + (UNR - 1) * PPL < r1:
        xvs = []
        for u in cutlass.range_constexpr(UNR):
            off_u = cutlass.Int64(row + u * PPL) * C + c0
            xvs.append(nvvm.load_ext(mXi.iterator + off_u, dtype=it_ty, count=V).bitcast(et))
        for u in cutlass.range_constexpr(UNR):
            for e in cutlass.range_constexpr(V):
                x = xvs[u][e].to(cutlass.Float32)
                s[e] = s[e] + x
                sq[e] = sq[e] + x * x
        row = row + UNR * PPL
    while row < r1:  # remainder
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s[e] = s[e] + x
            sq[e] = sq[e] + x * x
        row = row + PPL
    # reduce over the PPL pixel-groups (threads sharing tc, differing tp) via smem
    if cutlass.const_expr(PPL > 1):
        for e in cutlass.range_constexpr(V):
            red[(tp * TPP + (tid % TPP)) * V + e] = s[e]           # store sum, laid out [tp][channel]
        cute.arch.sync_threads()
        if tp == 0:
            for e in cutlass.range_constexpr(V):
                acc = s[e]
                for k in cutlass.range_constexpr(PPL - 1):
                    acc = acc + red[((k + 1) * TPP + (tid % TPP)) * V + e]
                s[e] = acc
        cute.arch.sync_threads()
        for e in cutlass.range_constexpr(V):
            red[(tp * TPP + (tid % TPP)) * V + e] = sq[e]
        cute.arch.sync_threads()
        if tp == 0:
            for e in cutlass.range_constexpr(V):
                acc = sq[e]
                for k in cutlass.range_constexpr(PPL - 1):
                    acc = acc + red[((k + 1) * TPP + (tid % TPP)) * V + e]
                sq[e] = acc
    # write the CTA partial (no atomics): mPart[my, 0:C]=sum, mPart[my, C:2C]=sumsq
    if tp == 0:
        pbase = cutlass.Int64(my) * (2 * C) + c0
        for e in cutlass.range_constexpr(V):
            mPart[pbase + e] = s[e]
            mPart[pbase + C + e] = sq[e]


@cute.kernel
def _bn_finalize(mPart, mSum, mSq, mparts: cutlass.Int32, C: cutlass.Constexpr,
                 CHUNK: cutlass.Constexpr):
    # 2D grid: bx = C-tile (thread-per-channel), by = partition chunk. Splitting the
    # mparts reduction across the y-grid + atomic-add keeps every SM busy (the 1-block
    # version ran the whole reduction on 1 SM -> 19us for 600KB).
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    c = bx * 256 + tid
    if c < C:
        p0 = by * CHUNK; p1 = p0 + CHUNK
        if p1 > mparts:
            p1 = mparts
        ssum = cutlass.Float32(0.0); ssq = cutlass.Float32(0.0)
        p = p0
        while p < p1:
            base = cutlass.Int64(p) * (2 * C) + c
            ssum = ssum + mPart[base]
            ssq = ssq + mPart[base + C]
            p = p + 1
        cute.arch.atomic_add(mSum.iterator + c, ssum)
        cute.arch.atomic_add(mSq.iterator + c, ssq)


@cute.kernel
def _bn_meanrstd(mSum, mSq, mMean, mRstd, C: cutlass.Constexpr, Mf: cutlass.Constexpr, eps: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    c = bx * 256 + tid
    if c < C:
        mean = mSum[c] / Mf
        var = mSq[c] / Mf - mean * mean
        mMean[c] = mean
        mRstd[c] = cute.math.rsqrt(var + eps)


@cute.kernel
def _bn_norm(mXi, mYi, mMean, mRstd, mG, mB, M: cutlass.Int32, mnorm: cutlass.Int32,
             C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr, PPL: cutlass.Constexpr,
             BT: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr, has_beta: cutlass.Constexpr):
    # channel-tile map (same as stats): each thread owns V fixed channels -> stage its
    # affine params (mean,rstd,gamma,beta) ONCE into registers, then stream pixels down
    # the M-slice (coalesced along C). Kills the per-element global affine reads.
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    tc = (tid % TPP) * V
    tp = tid // TPP
    c0 = cx * (TPP * V) + tc
    m = []; r = []; g = []; bb = []
    for e in cutlass.range_constexpr(V):
        m.append(mMean[c0 + e])
        r.append(mRstd[c0 + e])
        g.append(mG[c0 + e].to(cutlass.Float32))
        if cutlass.const_expr(has_beta):
            bb.append(mB[c0 + e].to(cutlass.Float32))
    per = (M + mnorm - 1) // mnorm
    r0 = my * per
    r1 = r0 + per
    if r1 > M:
        r1 = M
    row = r0 + tp
    while row < r1:
        off = cutlass.Int64(row) * C + c0
        xv = nvvm.load_ext(mXi.iterator + off, dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            y = (xv[e].to(cutlass.Float32) - m[e]) * r[e] * g[e]
            if cutlass.const_expr(has_beta):
                y = y + bb[e]
            ys.append(y.to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + off)
        row = row + PPL


@cute.jit
def _bn_host(mX, mY, mG, mB, mPart, mSum, mSq, mMean, mRstd, M, mparts, nblk,
             C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr, PPL: cutlass.Constexpr,
             BT: cutlass.Constexpr, cblks: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
             Mf: cutlass.Constexpr, eps: cutlass.Constexpr, cgrid: cutlass.Constexpr, has_beta: cutlass.Constexpr,
             smem_bytes: cutlass.Constexpr, nchunk: cutlass.Constexpr, CHUNK: cutlass.Constexpr):
    mXi = cute.recast_tensor(mX, it_ty); mYi = cute.recast_tensor(mY, it_ty)
    _bn_stats(mXi, mPart, M, mparts, C, V, TPP, PPL, BT, it_ty, et).launch(
        grid=(cblks, mparts, 1), block=(BT, 1, 1), smem=smem_bytes)
    _bn_finalize(mPart, mSum, mSq, mparts, C, CHUNK).launch(grid=(cgrid, nchunk, 1), block=(256, 1, 1))
    _bn_meanrstd(mSum, mSq, mMean, mRstd, C, Mf, eps).launch(grid=(cgrid, 1, 1), block=(256, 1, 1))
    # gamma/beta passed as native bf16 (indexed + .to(f32)); X/Y use recast Int16 + bitcast
    _bn_norm(mXi, mYi, mMean, mRstd, mG, mB, M, nblk, C, V, TPP, PPL, BT, it_ty, et, has_beta).launch(
        grid=(cblks, nblk, 1), block=(BT, 1, 1))


def run_bn(x, g, b, eps=1e-5):
    import cutlass as _c
    M, C = x.shape; V = 8; BT = 256; NSM = 148
    # C_PER_CTA knob: channel-tile per CTA, decoupled from C. Cap at 256 so TPP<=32 and
    # PPL>=8 (the old "whole C per CTA" gave PPL=1 for C=2048 -> no pixel parallelism).
    # Smaller CPC also means more channel-tiles -> fewer partials/channel (cheaper finalize).
    CPC = min(C, 256)
    TPP = CPC // V
    PPL = BT // TPP
    cblks = C // CPC
    # split-K over M: more CTAs -> better stats occupancy (finalize is now parallelized).
    # M-aware on PIXELS/CTA (=M/mparts, not groups): keep >=~64 pixels/CTA so small-M
    # (e.g. 7x7) isn't over-partitioned, floored at NSM to fill the machine, capped 4x.
    ngroups = (M + PPL - 1) // PPL
    mparts = max(1, min(4 * NSM // max(1, cblks), max(NSM, M // 64)))
    mparts = min(mparts, ngroups)
    nblk = max(1, NSM * 4 // max(1, cblks))  # normalize grid is (cblks, nblk) -> divide by cblks
    y = torch.empty_like(x)
    part = torch.empty(mparts * 2 * C, dtype=torch.float32, device="cuda")
    ssum = torch.zeros(C, dtype=torch.float32, device="cuda")
    ssq = torch.zeros(C, dtype=torch.float32, device="cuda")
    mean = torch.empty(C, dtype=torch.float32, device="cuda")
    rstd = torch.empty(C, dtype=torch.float32, device="cuda")
    cgrid = (C + 255) // 256
    nchunk = max(1, min(32, 256 // cgrid)); nchunk = min(nchunk, mparts)
    CHUNK = (mparts + nchunk - 1) // nchunk
    smem_bytes = BT * V * 2 * 4 + 128
    ra = (_dyn(x), _dyn(y), _dyn(g), _dyn(b), _dyn(part), _dyn(ssum), _dyn(ssq), _dyn(mean), _dyn(rstd),
          _c.Int32(M), _c.Int32(mparts), _c.Int32(nblk))
    ce = (C, V, TPP, PPL, BT, cblks, _c.Int16, _c.BFloat16, float(M), float(eps), cgrid, True, smem_bytes, nchunk, CHUNK)
    key = (C, M)
    fn = _C.get(key); fn = fn or cute.compile(_bn_host, *ra, *ce); _C[key] = fn; fn(*ra)
    return y, mean, rstd


def main():
    l2 = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    def dev(fn, it=40, wu=12):
        for _ in range(wu): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(it):
            l2.zero_()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                with record_function("op"): fn()
                torch.cuda.synchronize()
            ka = p.key_averages(); ts.append(sum(i.device_time for i in ka if i.device_time > 0) / 1000)
        return float(np.median(ts))
    N = 128
    for C, H, W, cud, ceil in [(64,56,56,2877,5480),(256,56,56,3448,6155),(128,28,28,2222,5097),
                                (256,28,28,2455,5517),(512,14,14,2351,5113),(2048,7,7,2254,5113)]:
        M = N * H * W
        x = torch.randn(M, C, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(C, device="cuda", dtype=torch.bfloat16); b = torch.randn(C, device="cuda", dtype=torch.bfloat16)
        # reference
        xf = x.float()
        mean_r = xf.mean(0); var_r = xf.var(0, unbiased=False); rstd_r = 1.0 / torch.sqrt(var_r + 1e-5)
        yr = (xf - mean_r) * rstd_r * g.float() + b.float()
        y, mean, rstd = run_bn(x, g, b)
        yerr = (y.float() - yr).abs().max().item()
        merr = (mean - mean_r).abs().max().item()
        rerr = (rstd - rstd_r).abs().max().item()
        if C == 64:
            print(f"  dbg mean f={mean[:3].tolist()} r={mean_r[:3].tolist()}")
            print(f"  dbg rstd f={rstd[:3].tolist()} r={rstd_r[:3].tolist()}")
        gb = 2 * M * C * 2 / (dev(lambda: run_bn(x, g, b)[0]) * 1e-3) / 1e9
        print(f"C={C:5} {H}x{W}: {gb:.0f} GB/s ({gb/cud:.2f}x cuDNN, {gb/ceil:.2f} ceil) yerr={yerr:.2f} merr={merr:.4f} rerr={rerr:.4f}", flush=True)


if __name__ == "__main__":
    main()
