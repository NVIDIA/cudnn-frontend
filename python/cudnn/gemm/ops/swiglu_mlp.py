# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense bf16 SwiGLU-MLP as a cuDNN autograd op.

``out = (silu(x @ Wg^T) * (x @ Wu^T)) @ Wd^T`` — the Qwen/LLaMA-style gated MLP,
with all GEMMs and the SwiGLU running on cuDNN.

The forward fuses the gate GEMM, the up GEMM, ``SiLU`` and the multiply into ONE
cuDNN kernel (the FORT-native runtime-fusion engine on SM100), which beats two
cuBLAS GEMMs plus a torch activation by ~1.05-1.20x on the Qwen3.5-27B MLP shape;
the down projection is a separate GEMM (a three-GEMM single graph does not
compile). That fused kernel also emits the two pre-activations ``gate = x@Wg^T``
and ``up = x@Wu^T`` (the GEMM accumulators it already computes) as extra outputs,
so the backward reads them instead of recomputing two GEMMs -- still one kernel,
the accumulators stored in the epilogue. This keeps the full backward at parity
with a torch autograd MLP, which likewise saves its activations (~0.99-1.02x
backward on the Qwen3.5-27B shape); an earlier revision that recomputed gate/up
paid two extra GEMMs and ran ~1.3x slower on the backward, ~1.17-1.19x on the full
fwd+bwd step. Saving ``{h, gate, up}`` costs ~3x[M,I] of activation memory, less
than the ~4x[M,I] torch autograd already keeps. The backward runs the dSwiGLU as
ONE two-output cuDNN pointwise kernel (``dup`` and ``dgate`` from a single graph;
cuDNN's tensor-ir engine declines multi-output but another engine serves it),
which reads the inputs once and is ~2x the two single-output kernels it replaces.
With that and the saved pre-activations, the full fwd+bwd step runs ~0.96-0.98x a
torch autograd MLP on the Qwen3.5-27B shape (B200). (An opt-in
``CUDNN_GEMM_SWIGLU_FROST_BWD=1`` path fuses the ``dh = dout @ Wd`` dgrad GEMM with
the dSwiGLU into one FROST kernel, taking the natural down weight directly, no
transpose. It avoids materialising ``dh`` to HBM but its cuTeDSL GEMM currently
ties the separate nvjet GEMM + one-kernel pointwise (~1.15ms each, B200 M8192
stage), so with the backward GEMM-bound it does not move the full step; off by
default pending FROST GEMM tuning.) Numerically matches torch to bf16 noise on the
output and all four gradients.

Weights are the ``[I, H]`` / ``[H, I]`` ``nn.Linear`` tensors; they enter the
GEMMs transposed and are bound as strided ``.t()`` views (cuDNN reads them
column-major), so no transpose copy is materialized.

Requires an SM100 (Blackwell) device for the fused runtime-fusion engine; on
other architectures the graph build declines and the op raises.
"""

import os

import torch

import cudnn

_BF16 = cudnn.data_type.BFLOAT16
_FP32 = cudnn.data_type.FLOAT

# One cuDNN handle -- and its plans + workspaces -- per (device, stream). The
# handle's stream is bound once at creation, and the caches below key on the same
# (device.index, stream), so two streams that use the op concurrently never share a
# handle or a scratch workspace (which would race and silently corrupt grads under
# DDP comm streams / explicit torch.cuda.stream() regions / multi-threaded backward).
_HANDLES = {}
_MM_CACHE = {}
_SWIGLU_CACHE = {}
_DSWIGLU_CACHE = {}
_AUTOTUNE_ITERS = 20


def _handle(device):
    """The cuDNN handle for ``(device, current stream)``, its stream bound once at
    creation. Returns ``(handle, stream)``; callers fold ``stream`` into their plan/
    workspace cache key so per-stream state stays isolated. A stream serialises its
    own kernels on the GPU, so a per-stream cached workspace is safe to reuse; the
    only unsafe sharing -- one workspace/handle across two concurrent streams -- can
    no longer happen."""
    stream = torch.cuda.current_stream(device).cuda_stream
    key = (device.index, stream)
    h = _HANDLES.get(key)
    if h is None:
        with torch.cuda.device(device):
            h = cudnn.create_handle()
            cudnn.set_stream(handle=h, stream=stream)
        _HANDLES[key] = h
    return h, stream


def _autotune(g, handle, var_pack):
    """Build every candidate plan, time each on the graph's device, and return
    ``(best_index, workspace)``. The top heuristic plan is not always fastest, so
    every plan that builds and executes is timed and the fastest kept. If no plan
    executes, raise instead of caching a known-failing index."""
    g.check_support()
    g.build_plans(cudnn.build_plan_policy.ALL)
    n = g.get_execution_plan_count()
    if n == 0:
        raise RuntimeError("cudnn.gemm.swiglu_mlp: no execution plan was generated for this graph")
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
    h, stream = _handle(a2.device)
    a, b = a2.unsqueeze(0), b2.unsqueeze(0)
    key = (tuple(a.shape), tuple(a.stride()), tuple(b.shape), tuple(b.stride()), a.dtype, a2.device.index, stream)
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
    """Fused ``h = silu(x@Wg^T) * (x@Wu^T)`` in ONE cuDNN kernel, also emitting the
    two pre-activations ``gate = x@Wg^T`` and ``up = x@Wu^T`` (the GEMM
    accumulators the fusion already computes) so the backward reads them instead of
    recomputing two GEMMs. All three come from the single fused plan -- the extra
    epilogue stores add ~14% to the forward but drop the two ~25% recompute GEMMs
    from the backward (measured 1-kernel on SM100; the accumulators are stored in
    the epilogue, no copy kernel appended). ``x:[M,H]``, ``Wg,Wu:[I,H]`` bound as
    strided ``.t()`` views (no transpose copy). Returns ``(h, gate, up)`` ``:[M,I]``."""
    h, stream = _handle(x.device)
    M, H = x.shape
    interm = Wg.shape[0]
    xv = x.unsqueeze(0)
    wg, wu = Wg.t().unsqueeze(0), Wu.t().unsqueeze(0)  # [1,H,interm] column-major views, no copy
    key = (M, H, interm, x.dtype, xv.stride(), wg.stride(), wu.stride(), x.device.index, stream)
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
        gate.set_output(True).set_data_type(_BF16)  # saved for backward -> no recompute GEMM
        up.set_output(True).set_data_type(_BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        hb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        gb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        ub = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        best, ws = _autotune(g, h, {X: xv, WG: wg, WU: wu, hh: hb.unsqueeze(0), gate: gb.unsqueeze(0), up: ub.unsqueeze(0)})
        e = (g, X, WG, WU, hh, gate, up, best, ws)
        _SWIGLU_CACHE[key] = e
    g, X, WG, WU, hh, gate, up, best, ws = e
    hb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
    gb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
    ub = torch.empty(M, interm, device=x.device, dtype=x.dtype)
    g.execute_plan_at_index({X: xv, WG: wg, WU: wu, hh: hb.unsqueeze(0), gate: gb.unsqueeze(0), up: ub.unsqueeze(0)}, ws, index=best, handle=h)
    return hb, gb, ub


def _dswiglu(dh, gate, up):
    """Fused dSwiGLU in ONE cuDNN kernel: ``dup = dh*silu(gate)`` and
    ``dgate = dh*up*silu'(gate)`` (``silu'(g) = s + silu*(1-s)``, ``s=sigmoid(g)``)
    as the two outputs of a single multi-output pointwise graph. ``dh,gate,up:[M,I]``
    are dense and same-shape (elementwise, no broadcast). cuDNN's tensor-ir engine
    declines multi-output (it logs ``unsupported multi-output fusion``), but another
    engine serves the graph as one kernel that reads the inputs once -- ~2x the two
    single-output kernels it replaces, which re-read the inputs and recompute the
    sigmoid. Returns ``(dgate, dup)``."""
    h, stream = _handle(dh.device)
    M, interm = dh.shape
    key = (M, interm, dh.dtype, dh.device.index, stream)
    e = _DSWIGLU_CACHE.get(key)
    if e is None:
        g = cudnn.pygraph(handle=h, compute_data_type=_FP32)
        DH = g.tensor(dim=[1, M, interm], stride=[M * interm, interm, 1], data_type=_BF16)
        GATE = g.tensor(dim=[1, M, interm], stride=[M * interm, interm, 1], data_type=_BF16)
        UP = g.tensor(dim=[1, M, interm], stride=[M * interm, interm, 1], data_type=_BF16)
        s = g.sigmoid(input=GATE)
        silu = g.mul(a=GATE, b=s)
        dup = g.mul(a=DH, b=silu)  # dh*silu(gate)
        silup = g.add(a=s, b=g.sub(a=silu, b=g.mul(a=silu, b=s)))  # silu' = s + silu*(1-s)
        dgate = g.mul(a=g.mul(a=DH, b=UP), b=silup)  # dh*up*silu'
        dup.set_output(True).set_data_type(_BF16)
        dgate.set_output(True).set_data_type(_BF16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        dgb = torch.empty(M, interm, device=dh.device, dtype=dh.dtype)
        dub = torch.empty(M, interm, device=dh.device, dtype=dh.dtype)
        vp = {DH: dh.unsqueeze(0), GATE: gate.unsqueeze(0), UP: up.unsqueeze(0), dup: dub.unsqueeze(0), dgate: dgb.unsqueeze(0)}
        best, ws = _autotune(g, h, vp)
        e = (g, DH, GATE, UP, dup, dgate, best, ws)
        _DSWIGLU_CACHE[key] = e
    g, DH, GATE, UP, dup, dgate, best, ws = e
    dgb = torch.empty(M, interm, device=dh.device, dtype=dh.dtype)
    dub = torch.empty(M, interm, device=dh.device, dtype=dh.dtype)
    g.execute_plan_at_index(
        {DH: dh.unsqueeze(0), GATE: gate.unsqueeze(0), UP: up.unsqueeze(0), dup: dub.unsqueeze(0), dgate: dgb.unsqueeze(0)}, ws, index=best, handle=h
    )
    return dgb, dub


_FROST_DSWIGLU_CACHE = {}
# ON by default (set CUDNN_GEMM_SWIGLU_FROST_BWD=0 to force the pointwise path). On
# this dense bf16 shape the FROST fused dgrad+dSwiGLU (one cuTeDSL kernel, dh never
# materialised to HBM) ties the separate nvjet GEMM + one-kernel pointwise (~1.15ms
# each, B200 M8192 stage) -- ~1% behind only because its cuTeDSL GEMM trails nvjet.
# It is the default so it stays exercised and Yanqin can close that GEMM gap, and
# because the fusion advantage grows as the workload gets pointwise-heavier (fp8
# halves the GEMM and adds quant/scale pointwise; MoE grouped GEMMs are smaller and
# more memory-bound). Falls back to the pointwise path if FROST cannot serve a
# shape/arch/thread (e.g. no CUDA context on an autograd worker thread).
_FROST_BWD = os.environ.get("CUDNN_GEMM_SWIGLU_FROST_BWD", "1") != "0"


def _frost_dswiglu(dout2, Wd, gate, up):
    """Fused backward dgrad + dSwiGLU in ONE FROST kernel: ``dh = dout2 @ Wd`` with
    ``dup = dh*silu(gate)`` and ``dgate = dh*silu'(gate)*up`` as the GEMM epilogue,
    so the separate dh GEMM + two dSwiGLU pointwise kernels of :func:`_dswiglu`
    collapse into a single cuTeDSL bare-launch kernel. ``dout2:[M,H]``, ``Wd:[H,I]``
    (natural down weight), ``gate, up:[M,I]``. Returns ``(dgate, dup):[M,I]``. Raises
    on any unsupported case so the caller falls back to the pointwise path.

    FROST takes the natural down weight directly (an N-major / I-contiguous B -- no
    transpose copy; the downstream ``dg.t()`` wgrad operand is likewise a free strided
    view). The fused stage beats the separate dh GEMM + two pointwise kernels
    (~1.15-2.64x, Qwen3.5-27B, SM100), but the full backward is GEMM-bound so the fused
    stage does not move it: the forward already saves gate/up, so the two recompute
    GEMMs that once dominated the backward are gone, and what remains (dWd, dh, dx, dWg,
    dWu) is pure GEMM that FROST's dgrad+epilogue fusion cannot shrink. Kept as an
    opt-in experiment, not a win. Not a transpose problem (dense bf16 needs none;
    ``dg.t()`` is a free view) and not a host-overhead one (host is ~1% of these
    compute-bound GEMMs, and #612 already cut it)."""
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
    from cudnn.gemm.frost.tile_config import CATALOG

    M, H = dout2.shape
    interm = gate.shape[1]
    key = (M, H, interm, dout2.dtype, dout2.device.index)
    e = _FROST_DSWIGLU_CACHE.get(key)
    if e is None:
        tn = 256 if interm >= 256 else 128
        cfg = next(c for c in CATALOG if (c.cta_tile_m, c.cta_tile_n, c.cta_tile_k_bytes, c.cgrp_size_m, c.cgrp_size_n) == (128, tn, 128, 1, 1))
        g = cudnn.pygraph(io_data_type=_BF16, intermediate_data_type=_FP32, compute_data_type=_FP32)
        DY = g.tensor(name="dy", dim=[1, M, H], stride=[M * H, H, 1])
        # Natural down weight [H,I] as an N-major (I-contiguous) B -- FROST takes
        # arbitrary t/n operand layouts, so no transpose copy is needed.
        WD = g.tensor(name="Wd", dim=[1, H, interm], stride=[H * interm, interm, 1])
        G = g.tensor(name="gate_pre", dim=[1, M, interm], stride=[M * interm, interm, 1])
        U = g.tensor(name="up_pre", dim=[1, M, interm], stride=[M * interm, interm, 1])
        dh = g.matmul(A=DY, B=WD, name="dgrad")
        g.mul(a=dh, b=g.swish(input=G), name="dup").set_output(True).set_data_type(_BF16)
        g.mul(a=g.swish_backward(loss=dh, input=G), b=U, name="dgate").set_output(True).set_data_type(_BF16)
        compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=1, scheduler="clc")
        bd = compiled.binding
        out_by = {o.get_name().split("::")[0]: o for o in bd.outputs}
        aux_by = {a.get_name(): a for a in bd.aux}
        e = (compiled, bd, out_by, aux_by)
        _FROST_DSWIGLU_CACHE[key] = e
    compiled, bd, out_by, aux_by = e
    dgate = torch.empty(1, M, interm, device=dout2.device, dtype=dout2.dtype)
    dup = torch.empty(1, M, interm, device=dout2.device, dtype=dout2.dtype)
    compiled(
        {
            bd.a_operands[0]: dout2.unsqueeze(0),
            bd.b_operands[0]: Wd.unsqueeze(0),  # natural [1,H,I], N-major — no transpose
            out_by["dup"]: dup,
            out_by["dgate"]: dgate,
            aux_by["gate_pre"]: gate.unsqueeze(0),
            aux_by["up_pre"]: up.unsqueeze(0),
        }
    )
    return dgate.squeeze(0), dup.squeeze(0)


class _SwigluMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Wg, Wu, Wd):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        h, gate, up = _swiglu_act(x2, Wg, Wu)  # fused: 1 kernel, also emits gate/up for the backward
        out = _mm(h, Wd.t())  # transposed weight view; cuDNN reads it column-major, no copy
        ctx.save_for_backward(x2, Wg, Wu, Wd, h, gate, up)
        ctx.shp = shp
        return out.reshape(*shp[:-1], Wd.shape[0])

    @staticmethod
    def backward(ctx, dout):
        x2, Wg, Wu, Wd, h, gate, up = ctx.saved_tensors  # gate/up saved by the fused forward, not recomputed
        dout2 = dout.reshape(-1, Wd.shape[0])
        dWd = _mm(dout2.t(), h)
        # dh = dout@Wd + dSwiGLU as ONE FROST kernel; fall back to a separate dh
        # GEMM + two pointwise kernels if FROST cannot serve this shape/arch.
        if _FROST_BWD:
            try:
                dgate, dup = _frost_dswiglu(dout2, Wd, gate, up)
            except Exception:
                dgate, dup = _dswiglu(_mm(dout2, Wd), gate, up)
        else:
            dgate, dup = _dswiglu(_mm(dout2, Wd), gate, up)
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

    Requires an SM100 (Blackwell) device. Raises on a non-bf16 / non-CUDA input
    or a shape mismatch (the kernels are bf16-only; other dtypes are not silently
    reinterpreted).
    """
    for name, t in (("x", x), ("Wg", Wg), ("Wu", Wu), ("Wd", Wd)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"cudnn.gemm.swiglu_mlp: {name} must be bfloat16, got {t.dtype}")
        if t.device.type != "cuda":
            raise ValueError(f"cudnn.gemm.swiglu_mlp: {name} must be a CUDA tensor, got device {t.device}")
    H = x.shape[-1]
    if x.dim() < 2 or Wg.dim() != 2 or Wu.dim() != 2 or Wd.dim() != 2:
        raise ValueError(
            f"cudnn.gemm.swiglu_mlp: expected x[...,H] and 2-D Wg/Wu[I,H], Wd[H,I]; got x{tuple(x.shape)} Wg{tuple(Wg.shape)} Wu{tuple(Wu.shape)} Wd{tuple(Wd.shape)}"
        )
    if Wg.shape != Wu.shape or Wg.shape[1] != H or Wd.shape[0] != H or Wd.shape[1] != Wg.shape[0]:
        raise ValueError(
            f"cudnn.gemm.swiglu_mlp: shape mismatch — x[...,{H}], Wg{tuple(Wg.shape)}, Wu{tuple(Wu.shape)}, Wd{tuple(Wd.shape)}; "
            f"expected Wg/Wu = [I, {H}] and Wd = [{H}, I]"
        )
    return _SwigluMLP.apply(x, Wg, Wu, Wd)
