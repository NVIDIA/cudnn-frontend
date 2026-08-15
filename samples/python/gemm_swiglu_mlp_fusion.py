# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense bf16 SwiGLU-MLP as a cuDNN autograd op (fwd + bwd), built from the cuDNN
graph API — a prototype for the GEMM owner: it shows what the graph fuses today and,
with measured numbers, exactly where a dense fused GEMM+SwiGLU training op is gated.

What the cuDNN graph fuses today (no new kernel):
  * forward  h = silu(x @ Wg^T) * (x @ Wu^T)  -> gate GEMM + up GEMM + SiLU + mul compile
    into ONE cuDNN kernel (a fused GEMM+pointwise engine; on SM100 with the default engine
    set this is the FORT-native runtime-fusion kernel — the demo prints the actual kernel
    name so the count/attribution is evidence, not assumed). The down projection
    `out = h @ Wd^T` is a separate GEMM (a 3-GEMM single graph does NOT compile).
  * backward dSwiGLU (dup, dgate from dh, gate, up) -> two single-output cuDNN pointwise
    kernels here; and as a matmul EPILOGUE, `matmul(dout,Wd)->dh` fused with the dSwiGLU
    is one kernel and ~2.3x the unfused dh-GEMM + elementwise (see fla_shim_proto probes).

Each cuDNN graph is autotuned (build ALL plans, time with `execute_plan_at_index`, keep
the fastest); on these Qwen3.5 shapes the top heuristic plan is already ~optimal (~1.01x),
so autotune is here for correctness/robustness, not as the lever.

Measured on B200 at the Qwen3.5-27B MLP shape (H5120 I17408), vs torch+cuBLAS:
  * forward fused is a real win: 1.05-1.20x eager AND under CUDA-graph replay, across token
    counts M in {2048..16384} (fusing gate+up+act into one kernel beats 2 cuBLAS GEMMs + act);
  * forward+backward eager is ~0.86x (a regression) — NOT because the kernels are slow. The
    cuDNN backend execute is already at cuBLAS parity (~8.4us vs ~7.6us for a 256^3 matmul);
    the gap is removable per-call FE wrapper cost (redundant set_stream ~5us, the generic
    execute vs a pinned execute_plan_at_index ~5us, output alloc + varpack churn), which the
    backward pays across 6-8 plain GEMMs, plus a bwd recompute-vs-save-activations tradeoff.
So the lever is (a) a memoized matmul hot path to erase the per-call wrapper cost (cf. the
grouped-GEMM/cuTeDSL fast paths), and (b) a cuBLAS-class fused GEMM+epilogue for the backward
(the MoE `gemm/cutedsl/grouped/{swiglu,dswiglu}` path is this, grouped+quantized). The dSwiGLU
itself already fuses well (backward dACT-as-matmul-epilogue is ~2.3x), so it is not the gap.

Numerically matches torch to bf16 noise (fwd + all four gradients). Verified on B200.
"""

import torch
import torch.nn.functional as F
import cudnn

BF16 = cudnn.data_type.BFLOAT16
FP32 = cudnn.data_type.FLOAT
_HANDLES = {}  # device index -> cudnn handle (one per device; plans/workspaces are device-bound)
_MM_CACHE = {}
_SWIGLU_CACHE = {}
_DSWIGLU_CACHE = {}
_AUTOTUNE_ITERS = 20
# label -> (num_plans, best_index, heuristic_first_ms, best_ms); for reporting only.
_AUTOTUNE_LOG = {}


def _handle(device):
    """One cuDNN handle per CUDA device, bound to that device's current stream. Graphs and
    workspaces are device-specific, so every cache key below also includes device.index."""
    h = _HANDLES.get(device.index)
    if h is None:
        with torch.cuda.device(device):
            h = cudnn.create_handle()
        _HANDLES[device.index] = h
    cudnn.set_stream(handle=h, stream=torch.cuda.current_stream(device).cuda_stream)
    return h


def _autotune(g, handle, var_pack, label):
    """Build ALL candidate plans, time each, and return (best_index, workspace).

    The top heuristic plan (index 0) is not always the fastest, so we time every plan
    that builds and executes, and keep the fastest. `var_pack` is a real variant pack
    (device buffers) used for the timing run.
    """
    g.check_support()
    g.build_plans(cudnn.build_plan_policy.ALL)
    n = g.get_execution_plan_count()
    dev = next(iter(var_pack.values())).device
    ws = torch.empty(max(g.get_workspace_size_plan_at_index(i) for i in range(n)), device=dev, dtype=torch.uint8)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = [float("inf")] * n
    for i in range(n):
        try:
            g.execute_plan_at_index(var_pack, ws, index=i, handle=handle)  # warm up / validity
            torch.cuda.synchronize()
            start.record()
            for _ in range(_AUTOTUNE_ITERS):
                g.execute_plan_at_index(var_pack, ws, index=i, handle=handle)
            stop.record()
            stop.synchronize()
            times[i] = start.elapsed_time(stop) / _AUTOTUNE_ITERS
        except Exception:
            pass  # a plan may build but fail to execute on this shape; skip it
    best = min(range(n), key=times.__getitem__)
    heur_first = next((t for t in times if t != float("inf")), float("inf"))
    _AUTOTUNE_LOG[label] = (n, best, heur_first, times[best])
    return best, ws


def _mm(a2, b2):
    """[M,K] @ [K,N] -> [M,N] on cuDNN (cached, autotuned graph keyed by shape/stride/dtype/device)."""
    h = _handle(a2.device)
    a, b = a2.unsqueeze(0), b2.unsqueeze(0)
    key = (tuple(a.shape), tuple(a.stride()), tuple(b.shape), tuple(b.stride()), a.dtype, a2.device.index)
    e = _MM_CACHE.get(key)
    if e is None:
        g = cudnn.pygraph(handle=h, compute_data_type=FP32)
        A = g.tensor(dim=list(a.shape), stride=list(a.stride()), data_type=BF16)
        B = g.tensor(dim=list(b.shape), stride=list(b.stride()), data_type=BF16)
        C = g.matmul(name="mm", A=A, B=B, compute_data_type=FP32)
        C.set_output(True).set_data_type(BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        out = torch.empty((1, a.shape[1], b.shape[2]), device=a2.device, dtype=a2.dtype)
        best, ws = _autotune(g, h, {A: a, B: b, C: out}, ("mm", a.shape[1], a.shape[2], b.shape[2]))
        e = (g, A, B, C, best, ws)
        _MM_CACHE[key] = e
    g, A, B, C, best, ws = e
    out = torch.empty((1, a.shape[1], b.shape[2]), device=a2.device, dtype=a2.dtype)
    g.execute_plan_at_index({A: a, B: b, C: out}, ws, index=best, handle=h)
    return out.squeeze(0)


def swiglu_act(x, Wg, Wu):
    """Fused h = silu(x@Wg^T) * (x@Wu^T) in ONE cuDNN kernel. x:[M,H] Wg,Wu:[I,H].

    Wg/Wu are the [I,H] nn.Linear weights; they enter the GEMM transposed. We bind them
    as strided ``.t()`` views ([H,I], column-major), so cuDNN reads them transposed with
    no copy — materializing ``Wg.t().contiguous()`` would add a [H,I] transpose kernel
    that costs more than the fused GEMM itself.
    """
    h = _handle(x.device)
    M, H = x.shape
    I = Wg.shape[0]
    xv = x.unsqueeze(0)
    wg, wu = Wg.t().unsqueeze(0), Wu.t().unsqueeze(0)  # [1,H,I] column-major views, no copy
    # key on every bound tensor's actual strides + device: cached descriptors encode them
    key = (M, H, I, x.dtype, xv.stride(), wg.stride(), wu.stride(), x.device.index)
    e = _SWIGLU_CACHE.get(key)
    if e is None:
        g = cudnn.pygraph(handle=h, compute_data_type=FP32)
        X = g.tensor(dim=[1, M, H], stride=list(xv.stride()), data_type=BF16)
        WG = g.tensor(dim=[1, H, I], stride=list(wg.stride()), data_type=BF16)
        WU = g.tensor(dim=[1, H, I], stride=list(wu.stride()), data_type=BF16)
        gate = g.matmul(name="gate", A=X, B=WG, compute_data_type=FP32)
        sg = g.mul(a=gate, b=g.sigmoid(input=gate))  # SiLU = gate * sigmoid(gate)
        up = g.matmul(name="up", A=X, B=WU, compute_data_type=FP32)
        hh = g.mul(a=sg, b=up)
        hh.set_output(True).set_data_type(BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        out = torch.empty(M, I, device=x.device, dtype=x.dtype)
        best, ws = _autotune(g, h, {X: xv, WG: wg, WU: wu, hh: out.unsqueeze(0)}, ("swiglu", M, H, I))
        e = (g, X, WG, WU, hh, best, ws)
        _SWIGLU_CACHE[key] = e
    g, X, WG, WU, hh, best, ws = e
    out = torch.empty(M, I, device=x.device, dtype=x.dtype)
    g.execute_plan_at_index({X: xv, WG: wg, WU: wu, hh: out.unsqueeze(0)}, ws, index=best, handle=h)
    return out


def _dswiglu(dh, gate, up):
    """Fused dSwiGLU: dup = dh*silu(gate), dgate = dh*up*silu'(gate), given dh,gate,up
    ([M,I]). A naive impl spreads this over ~6 fp32 torch kernels; here it is two
    single-output cuDNN pointwise kernels (bf16 io / fp32 compute). One graph writing
    both outputs hits "unsupported multi-output fusion", so dup and dgate are separate
    graphs; each fuses fully and they share the cheap sigmoid(gate) recompute."""
    h = _handle(dh.device)
    M, I = dh.shape
    key = (M, I, dh.dtype, dh.device.index)
    e = _DSWIGLU_CACHE.get(key)
    if e is None:

        def build(fn):
            g = cudnn.pygraph(handle=h, compute_data_type=FP32)
            DH = g.tensor(dim=[1, M, I], stride=[M * I, I, 1], data_type=BF16)
            GATE = g.tensor(dim=[1, M, I], stride=[M * I, I, 1], data_type=BF16)
            UP = g.tensor(dim=[1, M, I], stride=[M * I, I, 1], data_type=BF16)
            out = fn(g, DH, GATE, UP)
            out.set_output(True).set_data_type(BF16)
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            return g, DH, GATE, UP, out

        def dup_fn(g, DH, GATE, UP):
            return g.mul(a=DH, b=g.mul(a=GATE, b=g.sigmoid(input=GATE)))  # dh*silu(gate)

        def dgate_fn(g, DH, GATE, UP):
            s = g.sigmoid(input=GATE)
            silu = g.mul(a=GATE, b=s)
            silup = g.add(a=s, b=g.sub(a=silu, b=g.mul(a=silu, b=s)))  # silu' = s + silu*(1-s)
            return g.mul(a=g.mul(a=DH, b=UP), b=silup)

        scratch = torch.empty(M, I, device=dh.device, dtype=dh.dtype)
        built = {}
        for name, fn in (("dup", dup_fn), ("dgate", dgate_fn)):
            g, DH, GATE, UP, out = build(fn)
            vp = {DH: dh.unsqueeze(0), GATE: gate.unsqueeze(0), UP: up.unsqueeze(0), out: scratch.unsqueeze(0)}
            best, ws = _autotune(g, h, vp, ("dswiglu-" + name, M, I))
            built[name] = (g, DH, GATE, UP, out, best, ws)
        e = built
        _DSWIGLU_CACHE[key] = e
    outs = {}
    for name in ("dup", "dgate"):
        g, DH, GATE, UP, out, best, ws = e[name]
        buf = torch.empty(M, I, device=dh.device, dtype=dh.dtype)
        g.execute_plan_at_index({DH: dh.unsqueeze(0), GATE: gate.unsqueeze(0), UP: up.unsqueeze(0), out: buf.unsqueeze(0)}, ws, index=best, handle=h)
        outs[name] = buf
    return outs["dgate"], outs["dup"]


class SwigluMLP(torch.autograd.Function):
    """out = (silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T, all GEMMs + the SwiGLU on cuDNN."""

    @staticmethod
    def forward(ctx, x, Wg, Wu, Wd):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        h = swiglu_act(x2, Wg, Wu)  # fused: 1 kernel
        out = _mm(h, Wd.t())  # transposed weight view; cuDNN reads it column-major, no copy
        ctx.save_for_backward(x2, Wg, Wu, Wd, h)
        ctx.shp = shp
        return out.reshape(*shp[:-1], Wd.shape[0])

    @staticmethod
    def backward(ctx, dout):
        x2, Wg, Wu, Wd, h = ctx.saved_tensors
        dout2 = dout.reshape(-1, Wd.shape[0])
        dh = _mm(dout2, Wd)
        dWd = _mm(dout2.t(), h)
        gate = _mm(x2, Wg.t())  # recompute (no fwd materialization)
        up = _mm(x2, Wu.t())
        dgate, dup = _dswiglu(dh, gate, up)  # fused: 1 cuDNN kernel
        dx = _mm(dgate, Wg) + _mm(dup, Wu)
        dWg = _mm(dgate.t(), x2)
        dWu = _mm(dup.t(), x2)
        return dx.reshape(*ctx.shp), dWg, dWu, dWd


def _demo():
    dev = next((torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).major >= 10), None)
    if dev is None:
        print("no SM100+ (Blackwell) GPU found; the fused runtime-fusion engine needs one — skipping.")
        return
    M, H, I = 2048, 5120, 17408  # Qwen3.5-27B MLP shape
    torch.manual_seed(0)
    x = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    Wg = (torch.randn(I, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wu = (torch.randn(I, H, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    Wd = (torch.randn(H, I, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    do = torch.randn(1, M, H, device=dev, dtype=torch.bfloat16)
    xr, Wgr, Wur, Wdr = (t.detach().clone().requires_grad_(True) for t in (x, Wg, Wu, Wd))

    SwigluMLP.apply(x, Wg, Wu, Wd).backward(do)
    ((F.silu(xr @ Wgr.t()) * (xr @ Wur.t())) @ Wdr.t()).backward(do)

    def rel(a, b):
        return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-9)

    print(f"device {torch.cuda.get_device_properties(dev).name}; SwiGLU-MLP M{M} H{H} I{I}")
    print(f"fwd  rel={rel(SwigluMLP.apply(x, Wg, Wu, Wd), (F.silu(xr @ Wgr.t()) * (xr @ Wur.t())) @ Wdr.t()):.2e}")
    for n, a, b in [("dx", x.grad, xr.grad), ("dWg", Wg.grad, Wgr.grad), ("dWu", Wu.grad, Wur.grad), ("dWd", Wd.grad, Wdr.grad)]:
        print(f"bwd {n:3} rel={rel(a, b):.2e}")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        swiglu_act(x.detach().reshape(M, H), Wg.detach(), Wu.detach())
        torch.cuda.synchronize()
    # count actual GPU launches (ev.count), not distinct keys, and name them — so "1 kernel"
    # is evidence, and the kernel name shows which engine served the fusion (not assumed).
    kernels = [(ev.key, ev.count) for ev in prof.key_averages() if ev.self_device_time_total > 0]
    launches = sum(c for _, c in kernels)
    print(f"fused swiglu_act (gate+up+silu+mul) -> {launches} GPU launch(es): {', '.join(k[:60] for k, _ in kernels)}")
    n, best, heur, bt = _AUTOTUNE_LOG[("swiglu", M, H, I)]
    print(f"autotune: {n} plans, heuristic-first {heur:.3f} ms -> tuned {bt:.3f} ms (idx {best}, {heur / bt:.2f}x)")


if __name__ == "__main__":
    _demo()
