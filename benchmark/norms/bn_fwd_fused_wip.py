"""BatchNorm NHWC forward -- SINGLE-PASS cooperative de-risk (cutlass primitives).
One cooperative kernel fuses: pass1 (reduce slice + cache first KCACHE pixels/thread in
regs) -> write [mparts,2,C] partial -> GRID BARRIER (atomic-spin, gpu scope) -> in-kernel
finalize (each thread sums mparts partials for its V channels; L2-resident) -> pass2
normalize (cached pixels from regs, overflow re-read). Cuts the 2nd X read for the cached
fraction + kills the 2 extra launches. RMS-free BN, bf16."""
import sys, types
D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python" / "cudnn")
stub = types.ModuleType("cudnn"); stub.__path__ = [D]; stub.pygraph = type("pygraph", (), {}); sys.modules["cudnn"] = stub
import cutlass, cutlass.cute as cute, cutlass.primitives as nvvm, torch, numpy as np
from torch.profiler import profile, ProfilerActivity, record_function
from cutlass.memory import SmemAllocator
from cutlass.cute.runtime import from_dlpack
def _dyn(t): return from_dlpack(t, assumed_align=16).mark_layout_dynamic()
_C = {}


@cute.kernel
def _bn_sp(mXi, mYi, mG, mB, mPart, mRet, M: cutlass.Int32, mparts: cutlass.Int32,
          C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr, PPL: cutlass.Constexpr,
          BT: cutlass.Constexpr, KC: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
          Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BT * V), byte_alignment=16)
    tc = (tid % TPP) * V
    tp = tid // TPP
    c0 = cx * (TPP * V) + tc
    per = (M + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > M:
        r1 = M
    # ---- pass1: reduce + cache first KC pixels (unrolled -> constexpr cache index) ----
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
    row = r0 + tp + KC * PPL  # streaming remainder
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s[e] = s[e] + x
            sq[e] = sq[e] + x * x
        row = row + PPL
    # ---- reduce over PPL pixel-groups (sum then sumsq, reuse red) ----
    def _reduce(vals):
        if cutlass.const_expr(PPL > 1):
            for e in cutlass.range_constexpr(V):
                red[(tp * TPP + (tid % TPP)) * V + e] = vals[e]
            cute.arch.sync_threads()
            if tp == 0:
                for e in cutlass.range_constexpr(V):
                    acc = vals[e]
                    for j in cutlass.range_constexpr(PPL - 1):
                        acc = acc + red[((j + 1) * TPP + (tid % TPP)) * V + e]
                    vals[e] = acc
            cute.arch.sync_threads()
        return vals
    s = _reduce(s)
    sq = _reduce(sq)
    if tp == 0:
        pb = cutlass.Int64(my) * (2 * C) + c0
        for e in cutlass.range_constexpr(V):
            mPart[pb + e] = s[e]
            mPart[pb + C + e] = sq[e]
    # ---- grid barrier (atomic-spin, gpu scope) ----
    cute.arch.sync_threads()
    if tid == 0:
        cute.arch.atomic_add(mRet.iterator + cx, cutlass.Int32(1), sem="release", scope="gpu")
        done = False
        while not done:
            v = cute.arch.atomic_add(mRet.iterator + cx, cutlass.Int32(0), sem="acquire", scope="gpu")
            if v >= mparts:
                done = True
    cute.arch.sync_threads()
    # ---- finalize: this thread's V channels, sum mparts partials (L2-resident) ----
    ssum = [cutlass.Float32(0.0)] * V
    ssq = [cutlass.Float32(0.0)] * V
    p = cutlass.Int32(0)
    while p < mparts:
        base = cutlass.Int64(p) * (2 * C) + c0
        for e in cutlass.range_constexpr(V):
            ssum[e] = ssum[e] + mPart[base + e]
            ssq[e] = ssq[e] + mPart[base + C + e]
        p = p + 1
    mean = [cutlass.Float32(0.0)] * V
    rstd = [cutlass.Float32(0.0)] * V
    for e in cutlass.range_constexpr(V):
        mn = ssum[e] / Mf
        mean[e] = mn
        rstd[e] = cute.math.rsqrt(ssq[e] / Mf - mn * mn + eps)
    g = []
    bb = []
    for e in cutlass.range_constexpr(V):
        g.append(mG[c0 + e].to(cutlass.Float32))
        if cutlass.const_expr(has_beta):
            bb.append(mB[c0 + e].to(cutlass.Float32))
    # ---- pass2: normalize -- cached pixels from regs (unrolled), overflow re-read ----
    for kk in cutlass.range_constexpr(KC):
        rk = r0 + tp + kk * PPL
        if rk < r1:
            ys = []
            for e in cutlass.range_constexpr(V):
                y = (cache[kk * V + e] - mean[e]) * rstd[e] * g[e]
                if cutlass.const_expr(has_beta):
                    y = y + bb[e]
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (cutlass.Int64(rk) * C + c0))
    row = r0 + tp + KC * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        ys = []
        for e in cutlass.range_constexpr(V):
            y = (xv[e].to(cutlass.Float32) - mean[e]) * rstd[e] * g[e]
            if cutlass.const_expr(has_beta):
                y = y + bb[e]
            ys.append(y.to(et))
        nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (cutlass.Int64(row) * C + c0))
        row = row + PPL


@cute.jit
def _bn_sp_host(mX, mY, mG, mB, mPart, mRet, M, mparts,
                C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr, PPL: cutlass.Constexpr,
                BT: cutlass.Constexpr, KC: cutlass.Constexpr, cblks: cutlass.Constexpr, it_ty: cutlass.Constexpr,
                et: cutlass.Constexpr, Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
                smem_bytes: cutlass.Constexpr):
    mXi = cute.recast_tensor(mX, it_ty); mYi = cute.recast_tensor(mY, it_ty)
    _bn_sp(mXi, mYi, mG, mB, mPart, mRet, M, mparts, C, V, TPP, PPL, BT, KC, it_ty, et, Mf, eps, has_beta).launch(
        grid=(cblks, mparts, 1), block=(BT, 1, 1), smem=smem_bytes, cooperative=True)


def run_bn_sp(x, g, b, KC=8, eps=1e-5):
    import cutlass as _c
    M, C = x.shape; V = 8; BT = 256; NSM = 148
    # C_PER_CTA knob: more channel-tiles -> fewer NHW-partials per tile -> the in-kernel
    # (redundant O(mparts^2)) finalize gets much cheaper for large C.
    CPC = min(C, 256)
    TPP = CPC // V
    PPL = BT // TPP
    cblks = C // CPC
    # cooperative: all CTAs co-resident. cap total grid at 2*NSM -> mparts/tile shrinks
    # as cblks grows (the key to a cheap in-kernel finalize).
    mparts = max(1, min(2 * NSM // max(1, cblks), (M + PPL - 1) // PPL))
    y = torch.empty_like(x)
    part = torch.empty(mparts * 2 * C, dtype=torch.float32, device="cuda")
    ret = torch.zeros(cblks, dtype=torch.int32, device="cuda")
    smem_bytes = BT * V * 4 + 128
    ra = (_dyn(x), _dyn(y), _dyn(g), _dyn(b), _dyn(part), _dyn(ret), _c.Int32(M), _c.Int32(mparts))
    ce = (C, V, TPP, PPL, BT, KC, cblks, _c.Int16, _c.BFloat16, float(M), float(eps), True, smem_bytes)
    key = (C, M, KC)
    fn = _C.get(key); fn = fn or cute.compile(_bn_sp_host, *ra, *ce); _C[key] = fn
    ret.zero_(); fn(*ra)
    return y


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
        xf = x.float(); mean_r = xf.mean(0); rstd_r = 1.0 / torch.sqrt(xf.var(0, unbiased=False) + 1e-5)
        yr = (xf - mean_r) * rstd_r * g.float() + b.float()
        try:
            y = run_bn_sp(x, g, b, KC=8)
            yerr = (y.float() - yr).abs().max().item()
            gb = 2 * M * C * 2 / (dev(lambda: run_bn_sp(x, g, b, KC=8)) * 1e-3) / 1e9
            print(f"C={C:5} {H}x{W}: {gb:.0f} GB/s ({gb/cud:.2f}x cuDNN, {gb/ceil:.2f} ceil) yerr={yerr:.3f}", flush=True)
        except Exception as e:
            print(f"C={C:5} {H}x{W}: ERR {str(e)[:60]}", flush=True)


if __name__ == "__main__":
    main()
