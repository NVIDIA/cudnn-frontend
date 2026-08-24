"""BatchNorm NHWC forward -- SINGLE-PASS cooperative kernel
(cutlass primitives). One cooperative kernel fuses: pass1 (reduce slice + cache first KC
pixels/thread in regs, next KS in smem) -> write THIS CTA's partial to its OWN slot in a
[cblks,2,mparts,C-tile] partials buffer (plain stores, NO atomics -> zero channel
contention, unlike the earlier atomic-finalize) -> GPU-fence + atomic-spin GRID BARRIER ->
REDUNDANT finalize (every part-CTA re-reduces all mparts slots for its channel tile,
parallelized across the PPL pixel-groups; partials L2-resident) -> broadcast mean/rstd via
smem -> pass2 normalize (cached pixels from regs/smem, overflow re-read). This is exactly
cuDNN's cross-CTA reduction structure (partials + inter-block sync + redundant finalize),
now folded into a single launch. RMS-free BN, bf16."""
import sys, types
D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python" / "cudnn")
stub = types.ModuleType("cudnn"); stub.__path__ = [D]; stub.pygraph = type("pygraph", (), {}); sys.modules["cudnn"] = stub
import cutlass, cutlass.cute as cute, cutlass.primitives as nvvm, torch, numpy as np
from torch.profiler import profile, ProfilerActivity, record_function
from cutlass.memory import SmemAllocator
from cutlass.cute.runtime import from_dlpack
_CTA_SS = nvvm.SharedSpace.shared_cta
def _dyn(t): return from_dlpack(t, assumed_align=16).mark_layout_dynamic()
_C = {}
_OCC = {}


@cute.kernel
def _bn_sp(mXi, mYi, mG, mB, mP, mRet, M: cutlass.Int32, mparts: cutlass.Int32,
          C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr, PPL: cutlass.Constexpr,
          BT: cutlass.Constexpr, KC: cutlass.Constexpr, KS: cutlass.Constexpr, CPC: cutlass.Constexpr,
          it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
          Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    cx, my, _ = cute.arch.block_idx()
    smem = SmemAllocator()
    red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BT * V), byte_alignment=16)
    stat = smem.allocate_tensor(cutlass.Float32, cute.make_layout(2 * CPC), byte_alignment=16)
    sc = smem.allocate_tensor(cutlass.Int16, cute.make_layout(BT * KS * V), byte_alignment=16) if cutlass.const_expr(KS > 0) else None
    lane = tid % TPP
    tp = tid // TPP
    c0 = cx * (TPP * V) + lane * V
    per = (M + mparts - 1) // mparts
    r0 = my * per
    r1 = r0 + per
    if r1 > M:
        r1 = M
    # ---- pass1: reduce + cache first KC pixels (regs), next KS (smem), stream remainder ----
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
    row = r0 + tp + (KC + KS) * PPL
    while row < r1:
        xv = nvvm.load_ext(mXi.iterator + (cutlass.Int64(row) * C + c0), dtype=it_ty, count=V).bitcast(et)
        for e in cutlass.range_constexpr(V):
            x = xv[e].to(cutlass.Float32)
            s[e] = s[e] + x
            sq[e] = sq[e] + x * x
        row = row + PPL
    # ---- reduce over PPL pixel-groups (reused for pass1 stats AND finalize) ----
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
    # ---- cuDNN-style cross-CTA reduce: write THIS CTA's partials to its OWN slot
    # mP[cx][{sum,sq}][my][lane] with plain 128-bit stores (NO atomics -> no contention) ----
    psum = cx * (mparts * TPP * V * 2)
    psq = psum + mparts * TPP * V
    if tp == 0:
        off = (my * TPP + lane) * V
        for h in cutlass.range_constexpr(2):
            sseg = cutlass.Vector.from_elements(tuple(s[h * 4 + j] for j in range(4)), cutlass.Float32)
            qseg = cutlass.Vector.from_elements(tuple(sq[h * 4 + j] for j in range(4)), cutlass.Float32)
            nvvm.store_ext(sseg, mP.iterator + (psum + off + h * 4))
            nvvm.store_ext(qseg, mP.iterator + (psq + off + h * 4))
    # ---- grid barrier (gpu-fence so partial stores land, then atomic-spin retired counter) ----
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
    # ---- REDUNDANT finalize (FASTER than a 1-CTA non-redundant one for cblks=1: fully parallel
    # across all part-CTAs, vectorized L2 loads, no 2nd grid barrier): every part-CTA re-reduces
    # ALL mparts slots for its channel tile, parallelized across the PPL pixel-groups. ----
    facc = [cutlass.Float32(0.0)] * V
    faccsq = [cutlass.Float32(0.0)] * V
    part = tp
    while part < mparts:
        off = (part * TPP + lane) * V
        for h in cutlass.range_constexpr(2):
            sv = nvvm.load_ext(mP.iterator + (psum + off + h * 4), dtype=cutlass.Float32, count=4)
            qv = nvvm.load_ext(mP.iterator + (psq + off + h * 4), dtype=cutlass.Float32, count=4)
            for j in cutlass.range_constexpr(4):
                facc[h * 4 + j] = facc[h * 4 + j] + sv[j]
                faccsq[h * 4 + j] = faccsq[h * 4 + j] + qv[j]
        part = part + PPL
    facc = _reduce(facc)
    faccsq = _reduce(faccsq)
    # tp==0 has the full per-channel reduction -> mean/rstd; broadcast to all PPL via smem
    if tp == 0:
        for e in cutlass.range_constexpr(V):
            mn = facc[e] / Mf
            stat[lane * V + e] = mn
            stat[CPC + lane * V + e] = cute.math.rsqrt(faccsq[e] / Mf - mn * mn + eps)
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
    # ---- pass2: normalize -- cached pixels from regs (unrolled), then smem, then re-read ----
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
    for ks in cutlass.range_constexpr(KS):
        rk = r0 + tp + (KC + ks) * PPL
        if rk < r1:
            xv = nvvm.load_ext(sc.iterator + (tid * KS + ks) * V, dtype=it_ty, count=V, shared_space=_CTA_SS).bitcast(et)
            ys = []
            for e in cutlass.range_constexpr(V):
                y = (xv[e].to(cutlass.Float32) - mean[e]) * rstd[e] * g[e]
                if cutlass.const_expr(has_beta):
                    y = y + bb[e]
                ys.append(y.to(et))
            nvvm.store_ext(cutlass.Vector.from_elements(tuple(ys), et).bitcast(it_ty), mYi.iterator + (cutlass.Int64(rk) * C + c0))
    row = r0 + tp + (KC + KS) * PPL
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
def _bn_sp_host(mX, mY, mG, mB, mP, mRet, M, mparts,
                C: cutlass.Constexpr, V: cutlass.Constexpr, TPP: cutlass.Constexpr, PPL: cutlass.Constexpr,
                BT: cutlass.Constexpr, KC: cutlass.Constexpr, KS: cutlass.Constexpr, CPC: cutlass.Constexpr,
                cblks: cutlass.Constexpr, it_ty: cutlass.Constexpr, et: cutlass.Constexpr,
                Mf: cutlass.Constexpr, eps: cutlass.Constexpr, has_beta: cutlass.Constexpr,
                smem_bytes: cutlass.Constexpr, mbpm: cutlass.Constexpr):
    mXi = cute.recast_tensor(mX, it_ty); mYi = cute.recast_tensor(mY, it_ty)
    _bn_sp(mXi, mYi, mG, mB, mP, mRet, M, mparts, C, V, TPP, PPL, BT, KC, KS, CPC, it_ty, et, Mf, eps, has_beta).launch(
        grid=(cblks, mparts, 1), block=(BT, 1, 1), smem=smem_bytes, cooperative=True, min_blocks_per_mp=mbpm)


def run_bn_sp(x, g, b, KC=8, KS=0, eps=1e-5, mbpm=2):
    import cutlass as _c
    M, C = x.shape; V = 8; BT = 512; NSM = 148  # BT=512 halves the per-thread slice (cuDNN's PPL)
    # C_PER_CTA knob: more channel-tiles -> fewer NHW-partials per tile -> cheaper finalize.
    CPC = min(C, 256)
    TPP = CPC // V
    PPL = BT // TPP
    cblks = C // CPC
    red_b = BT * V * 4; stat_b = 2 * CPC * 4
    smem_bytes = red_b + stat_b + BT * KS * V * 2 + 128  # red + stat + sc cache
    y = torch.empty_like(x)
    ce = (C, V, TPP, PPL, BT, KC, KS, CPC, cblks, _c.Int16, _c.BFloat16, float(M), float(eps), True, smem_bytes, mbpm)
    key = (C, M, KC, KS, mbpm)
    # cooperative: ALL CTAs must co-reside -> grid <= occupancy*NSM. Occupancy is config-
    # dependent (reg/smem), so TRY occ=2 then fall back to occ=1 on TOO_LARGE (mparts is a
    # runtime arg -> the same compiled kernel just re-launches with a smaller grid).
    import os
    fn = _C.get(key)
    # occupancy is config-dependent; cache the working occ per key so timed iters don't
    # re-throw a failed occ=2 launch every call (which pollutes the measurement).
    occ0 = _OCC.get(key, 2)
    can2 = smem_bytes * 2 <= 224 * 1024  # occ=2 only if 2 CTAs' smem co-resides
    for occ in ([2, 1] if (occ0 == 2 and can2) else [1]):
        mparts = max(1, min(occ * NSM // max(1, cblks), (M + PPL - 1) // PPL))
        # partials [cblks, 2, mparts, TPP*V] fp32; every slot written -> no zeroing needed.
        pbuf = torch.empty(cblks * mparts * TPP * V * 2, dtype=torch.float32, device="cuda")
        ret = torch.zeros(cblks, dtype=torch.int32, device="cuda")
        ra = (_dyn(x), _dyn(y), _dyn(g), _dyn(b), _dyn(pbuf), _dyn(ret), _c.Int32(M), _c.Int32(mparts))
        if fn is None:
            fn = cute.compile(_bn_sp_host, *ra, *ce); _C[key] = fn
        try:
            ret.zero_(); fn(*ra)
            was_new = key not in _OCC
            _OCC[key] = occ
            if os.environ.get("DBG") and was_new: print(f"    [dbg C={C} occ={occ} mparts={mparts} cblks={cblks} PPL={PPL} KS={KS} smem={smem_bytes//1024}KB]", flush=True)
            return y
        except Exception as e:
            if "TOO_LARGE" in str(e) and occ == 2:
                continue
            raise
    return y


# measured cuDNN NHWC BN GB/s references (separate process, profiler device-time)
_CUDNN = {
    128: {(64,56):2877,(256,56):3448,(128,28):2222,(256,28):2455,(512,14):2351,(2048,7):2254},
    8:   {(64,56):551,(256,56):1526,(128,28):273,(256,28):513,(512,14):271,(2048,7):267},
    16:  {(64,56):926,(256,56):2365,(128,28):543,(256,28):928,(512,14):544,(2048,7):538},
    32:  {(64,56):1524,(256,56):2046,(128,28):930,(256,28):1545,(512,14):923,(2048,7):907},
}


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
    import os
    KC = int(os.environ.get("KC", 8))
    MBPM = int(os.environ.get("MBPM", 2))
    KS = int(os.environ.get("KS", 0))
    AUTO = int(os.environ.get("AUTO", 0))
    # (mbpm, KC, KS) candidates -- the knob space cuDNN autotunes (occupancy, reg-cache,
    # smem-cache depth). occ=1 big-L1 for small/mid-M; occ=2 + moderate smem cache for large-M.
    CANDS = [(0, 8, 0), (2, 0, 4), (2, 0, 6), (2, 0, 8), (2, 2, 0)]
    Ns = [int(a) for a in sys.argv[1:]] or [128]
    print(f"### {'AUTOTUNE '+str(CANDS) if AUTO else f'KC={KC} KS={KS} MBPM={MBPM}'} BT=512", flush=True)
    for N in Ns:
        cud = _CUDNN.get(N, {})
        print(f"===== N={N} =====", flush=True)
        for C, H, W in [(64,56,56),(256,56,56),(128,28,28),(256,28,28),(512,14,14),(2048,7,7)]:
            M = N * H * W
            x = torch.randn(M, C, device="cuda", dtype=torch.bfloat16)
            g = torch.randn(C, device="cuda", dtype=torch.bfloat16); b = torch.randn(C, device="cuda", dtype=torch.bfloat16)
            xf = x.float(); mean_r = xf.mean(0); rstd_r = 1.0 / torch.sqrt(xf.var(0, unbiased=False) + 1e-5)
            yr = (xf - mean_r) * rstd_r * g.float() + b.float()
            yc = torch.empty_like(x)
            ceil = 2 * M * C * 2 / (dev(lambda: yc.copy_(x)) * 1e-3) / 1e9
            cands = CANDS if AUTO else [(MBPM, KC, KS)]
            best_gb = 0.0; best_cfg = None; best_yerr = 9.9
            for (mb, kc, ks) in cands:
                try:
                    y = run_bn_sp(x, g, b, KC=kc, KS=ks, mbpm=mb)
                    ye = (y.float() - yr).abs().max().item()
                    gb = 2 * M * C * 2 / (dev(lambda: run_bn_sp(x, g, b, KC=kc, KS=ks, mbpm=mb)) * 1e-3) / 1e9
                    if gb > best_gb: best_gb = gb; best_cfg = (mb, kc, ks); best_yerr = ye
                except Exception as e:
                    if not AUTO: print(f"  C={C:5} {H}x{W}: ERR {str(e)[:50]}", flush=True)
            cv = cud.get((C, H), 0)
            xr = f"{best_gb/cv:.2f}x cuDNN, " if cv else ""
            print(f"  C={C:5} {H}x{W}: frost={best_gb:.0f} ({xr}{best_gb/ceil:.2f} ceil) cfg(mbpm,KC,KS)={best_cfg} yerr={best_yerr:.3f}", flush=True)


if __name__ == "__main__":
    main()
