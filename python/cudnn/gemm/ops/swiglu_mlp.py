# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense bf16 SwiGLU-MLP as a cuDNN autograd op.

``out = (silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T`` — the Qwen/LLaMA-style gated MLP,
with all GEMMs and the SwiGLU running on cuDNN.

The forward fuses the gate GEMM, the up GEMM, ``SiLU`` and the multiply into ONE
cuDNN kernel (the FORT-native runtime-fusion engine on SM100), which beats two
cuBLAS GEMMs plus a torch activation by ~1.05-1.20x on the Qwen3.5-27B MLP shape;
the down projection is a separate GEMM (a three-GEMM single graph does not
compile). Backward recomputes gate/up and fuses the dSwiGLU into two cuDNN
pointwise kernels. Numerically matches torch to bf16 noise on the output and all
four gradients.

Weights are the ``[I, H]`` / ``[H, I]`` ``nn.Linear`` tensors; they enter the
GEMMs transposed and are bound as strided ``.t()`` views (cuDNN reads them
column-major), so no transpose copy is materialized.

Requires an SM100 (Blackwell) device for the fused runtime-fusion engine; on
other architectures the graph build declines and the op raises.
"""

import torch

import cudnn

_BF16 = cudnn.data_type.BFLOAT16
_FP32 = cudnn.data_type.FLOAT

# One cuDNN handle per device; plans/workspaces are device-bound so every cache
# key below also carries device.index. Value: [handle, last_stream].
_HANDLES = {}
_MM_CACHE = {}
_SWIGLU_CACHE = {}
_DSWIGLU_CACHE = {}
_AUTOTUNE_ITERS = 20


def _handle(device):
    """One cuDNN handle per CUDA device. ``set_stream`` is called only when the
    current stream changes (it costs ~5 us/call). Assumes a handle is not used
    from two streams concurrently (true for a single-stream training loop)."""
    entry = _HANDLES.get(device.index)
    if entry is None:
        with torch.cuda.device(device):
            entry = [cudnn.create_handle(), None]  # [handle, last_stream]
        _HANDLES[device.index] = entry
    h, last_stream = entry
    stream = torch.cuda.current_stream(device).cuda_stream
    if stream != last_stream:
        cudnn.set_stream(handle=h, stream=stream)
        entry[1] = stream
    return h


def _autotune(g, handle, var_pack):
    """Build every candidate plan, time each on the graph's device, and return
    ``(best_index, workspace)``. The top heuristic plan is not always fastest, so
    every plan that builds and executes is timed and the fastest kept. If no plan
    executes, raise instead of caching a known-failing index."""
    g.check_support()
    g.build_plans(cudnn.build_plan_policy.ALL)
    n = g.get_execution_plan_count()
    dev = next(iter(var_pack.values())).device
    times = [float("inf")] * n
    errors = {}
    with torch.cuda.device(dev):  # events/workspace/sync must be on the graph's device
        ws = torch.empty(max(g.get_workspace_size_plan_at_index(i) for i in range(n)), device=dev, dtype=torch.uint8)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for i in range(n):
            try:
                g.execute_plan_at_index(var_pack, ws, index=i, handle=handle)  # warm up / validity
                torch.cuda.synchronize(dev)
                start.record()
                for _ in range(_AUTOTUNE_ITERS):
                    g.execute_plan_at_index(var_pack, ws, index=i, handle=handle)
                stop.record()
                stop.synchronize()
                times[i] = start.elapsed_time(stop) / _AUTOTUNE_ITERS
            except Exception as exc:  # a plan may build but fail to execute on this shape; skip it
                errors[i] = repr(exc)
    best = min(range(n), key=times.__getitem__)
    if times[best] == float("inf"):
        raise RuntimeError(f"cudnn.gemm.swiglu_mlp: all {n} autotune plans failed to execute; errors: {errors}")
    return best, ws


def _mm(a2, b2):
    """``[M,K] @ [K,N] -> [M,N]`` on cuDNN (cached, autotuned graph keyed by shape/stride/dtype/device)."""
    h = _handle(a2.device)
    a, b = a2.unsqueeze(0), b2.unsqueeze(0)
    key = (tuple(a.shape), tuple(a.stride()), tuple(b.shape), tuple(b.stride()), a.dtype, a2.device.index)
    e = _MM_CACHE.get(key)
    if e is None:
        g = cudnn.pygraph(handle=h, compute_data_type=_FP32)
        A = g.tensor(dim=list(a.shape), stride=list(a.stride()), data_type=_BF16)
        B = g.tensor(dim=list(b.shape), stride=list(b.stride()), data_type=_BF16)
        C = g.matmul(name="mm", A=A, B=B, compute_data_type=_FP32)
        C.set_output(True).set_data_type(_BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        out = torch.empty((1, a.shape[1], b.shape[2]), device=a2.device, dtype=a2.dtype)
        best, ws = _autotune(g, h, {A: a, B: b, C: out})
        e = (g, A, B, C, best, ws)
        _MM_CACHE[key] = e
    g, A, B, C, best, ws = e
    out = torch.empty((1, a.shape[1], b.shape[2]), device=a2.device, dtype=a2.dtype)
    g.execute_plan_at_index({A: a, B: b, C: out}, ws, index=best, handle=h)
    return out.squeeze(0)


def _swiglu_act(x, Wg, Wu):
    """Fused ``h = silu(x@Wg^T) * (x@Wu^T)`` in ONE cuDNN kernel. ``x:[M,H]``,
    ``Wg,Wu:[I,H]`` bound as strided ``.t()`` views (no transpose copy)."""
    h = _handle(x.device)
    M, H = x.shape
    interm = Wg.shape[0]
    xv = x.unsqueeze(0)
    wg, wu = Wg.t().unsqueeze(0), Wu.t().unsqueeze(0)  # [1,H,interm] column-major views, no copy
    key = (M, H, interm, x.dtype, xv.stride(), wg.stride(), wu.stride(), x.device.index)
    e = _SWIGLU_CACHE.get(key)
    if e is None:
        g = cudnn.pygraph(handle=h, compute_data_type=_FP32)
        X = g.tensor(dim=[1, M, H], stride=list(xv.stride()), data_type=_BF16)
        WG = g.tensor(dim=[1, H, interm], stride=list(wg.stride()), data_type=_BF16)
        WU = g.tensor(dim=[1, H, interm], stride=list(wu.stride()), data_type=_BF16)
        gate = g.matmul(name="gate", A=X, B=WG, compute_data_type=_FP32)
        sg = g.mul(a=gate, b=g.sigmoid(input=gate))  # SiLU = gate * sigmoid(gate)
        up = g.matmul(name="up", A=X, B=WU, compute_data_type=_FP32)
        hh = g.mul(a=sg, b=up)
        hh.set_output(True).set_data_type(_BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        out = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        best, ws = _autotune(g, h, {X: xv, WG: wg, WU: wu, hh: out.unsqueeze(0)})
        e = (g, X, WG, WU, hh, best, ws)
        _SWIGLU_CACHE[key] = e
    g, X, WG, WU, hh, best, ws = e
    out = torch.empty(M, interm, device=x.device, dtype=x.dtype)
    g.execute_plan_at_index({X: xv, WG: wg, WU: wu, hh: out.unsqueeze(0)}, ws, index=best, handle=h)
    return out


def _dswiglu(dh, gate, up):
    """Fused dSwiGLU: ``dup = dh*silu(gate)``, ``dgate = dh*up*silu'(gate)`` given
    ``dh,gate,up:[M,I]`` — two single-output cuDNN pointwise kernels (a single
    graph writing both outputs hits unsupported multi-output fusion)."""
    h = _handle(dh.device)
    M, interm = dh.shape
    key = (M, interm, dh.dtype, dh.device.index)
    e = _DSWIGLU_CACHE.get(key)
    if e is None:

        def build(fn):
            g = cudnn.pygraph(handle=h, compute_data_type=_FP32)
            DH = g.tensor(dim=[1, M, interm], stride=[M * interm, interm, 1], data_type=_BF16)
            GATE = g.tensor(dim=[1, M, interm], stride=[M * interm, interm, 1], data_type=_BF16)
            UP = g.tensor(dim=[1, M, interm], stride=[M * interm, interm, 1], data_type=_BF16)
            out = fn(g, DH, GATE, UP)
            out.set_output(True).set_data_type(_BF16)
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

        scratch = torch.empty(M, interm, device=dh.device, dtype=dh.dtype)
        built = {}
        for name, fn in (("dup", dup_fn), ("dgate", dgate_fn)):
            g, DH, GATE, UP, out = build(fn)
            vp = {DH: dh.unsqueeze(0), GATE: gate.unsqueeze(0), UP: up.unsqueeze(0), out: scratch.unsqueeze(0)}
            best, ws = _autotune(g, h, vp)
            built[name] = (g, DH, GATE, UP, out, best, ws)
        e = built
        _DSWIGLU_CACHE[key] = e
    outs = {}
    for name in ("dup", "dgate"):
        g, DH, GATE, UP, out, best, ws = e[name]
        buf = torch.empty(M, interm, device=dh.device, dtype=dh.dtype)
        g.execute_plan_at_index({DH: dh.unsqueeze(0), GATE: gate.unsqueeze(0), UP: up.unsqueeze(0), out: buf.unsqueeze(0)}, ws, index=best, handle=h)
        outs[name] = buf
    return outs["dgate"], outs["dup"]


class _SwigluMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Wg, Wu, Wd):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        h = _swiglu_act(x2, Wg, Wu)  # fused: 1 kernel
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
        dgate, dup = _dswiglu(dh, gate, up)  # fused dSwiGLU: two cuDNN pointwise kernels
        dx = _mm(dgate, Wg) + _mm(dup, Wu)
        dWg = _mm(dgate.t(), x2)
        dWu = _mm(dup.t(), x2)
        return dx.reshape(*ctx.shp), dWg, dWu, dWd


def swiglu_mlp(x, Wg, Wu, Wd):
    """Dense bf16 SwiGLU-MLP ``(silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T`` on cuDNN.

    Args:
        x: input activations ``[..., H]`` (bf16).
        Wg, Wu: gate / up ``nn.Linear`` weights ``[I, H]`` (bf16).
        Wd: down ``nn.Linear`` weight ``[H, I]`` (bf16).

    Returns:
        ``[..., H]`` (bf16). Differentiable w.r.t. all four inputs.

    Requires an SM100 (Blackwell) device.
    """
    return _SwigluMLP.apply(x, Wg, Wu, Wd)
