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
the accumulators stored in the epilogue. This matches torch autograd's
saved-activation policy; an earlier revision that
recomputed gate/up paid two extra GEMMs and ran ~1.3x slower on the backward,
~1.17-1.19x on the full fwd+bwd step. Saving ``{h, gate, up}`` costs ~3x[M,I] of
activation memory, less than the ~4x[M,I] torch autograd already keeps. The
pointwise fallback computes ``dup`` and ``dgate`` in one two-output graph, reading
the inputs once and running ~2x faster than two single-output kernels. By default,
``dh = dout @ Wd`` dgrad GEMM and the dSwiGLU are fused into one FROST (cuTeDSL)
kernel that never materialises ``dh`` to HBM (set
``CUDNN_GEMM_SWIGLU_FROST_BWD=0`` for the separate nvjet GEMM + one-kernel
pointwise, which it falls back to anyway if FROST cannot serve a
shape/arch/thread). The dense large-M path uses the B200-tuned 2-CTA
M128/N256/K128, cluster2x1, CLC strategy; its bare GEMM is at nvjet parity and
the fused epilogue turns the avoided ``dh`` round-trip into a net win. A balanced
B200 run at M=8192, H=5120, I=17408 measured 1.077x backward and 1.109x full
fwd+bwd versus eager torch (the exact ratio is workload/runtime dependent).
Numerically matches torch to bf16 noise on the output and all four gradients.

At the public call boundary the op snapshots GradMode and each input's
``requires_grad`` into a small mask. Inference and Wd-only training use an h-only
forward graph; partial-gradient training retains only the tensors its requested
input gradients consume and skips unrelated backward GEMMs. Full all-gradient
training takes the same kernels as above. The mask follows ``requires_grad`` at
forward time; it cannot infer a narrower target list passed later to
``torch.autograd.grad``.

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

_GRAD_X = 1 << 0
_GRAD_WG = 1 << 1
_GRAD_WU = 1 << 2
_GRAD_WD = 1 << 3
_GRAD_FC1 = _GRAD_X | _GRAD_WG | _GRAD_WU


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


def _swiglu_act(x, Wg, Wu, *, save_preacts=True):
    """Fused ``h = silu(x@Wg^T) * (x@Wu^T)`` in ONE cuDNN kernel, also emitting the
    two pre-activations ``gate = x@Wg^T`` and ``up = x@Wu^T`` when
    ``save_preacts=True``. All requested outputs come from the single fused plan;
    their cost is plan/runtime dependent, but they drop the two recompute GEMMs
    from a full backward. In inference or a Wd-only backward,
    ``save_preacts=False`` selects an h-only graph and avoids those side stores.
    ``x:[M,H]``, ``Wg,Wu:[I,H]`` are bound as strided ``.t()`` views (no transpose
    copy). Returns ``(h, gate, up)`` ``:[M,I]``; gate/up are ``None`` for h-only."""
    h, stream = _handle(x.device)
    M, H = x.shape
    interm = Wg.shape[0]
    xv = x.unsqueeze(0)
    wg, wu = Wg.t().unsqueeze(0), Wu.t().unsqueeze(0)  # [1,H,interm] column-major views, no copy
    key = (M, H, interm, x.dtype, xv.stride(), wg.stride(), wu.stride(), bool(save_preacts), x.device.index, stream)
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
        # Keep h numerically independent of whether the pre-activations are exposed:
        # the full-training graph consumes BF16-rounded gate/up, so the h-only graph
        # must retain those intermediate dtypes even though it does not store them.
        gate.set_data_type(_BF16)
        up.set_data_type(_BF16)
        if save_preacts:
            gate.set_output(True)  # saved for backward -> no recompute GEMM
            up.set_output(True)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        hb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        vp = {X: xv, WG: wg, WU: wu, hh: hb.unsqueeze(0)}
        if save_preacts:
            gb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
            ub = torch.empty(M, interm, device=x.device, dtype=x.dtype)
            vp.update({gate: gb.unsqueeze(0), up: ub.unsqueeze(0)})
        best, ws = _autotune(g, h, vp)
        e = (g, X, WG, WU, hh, gate, up, best, ws)
        _SWIGLU_CACHE[key] = e
    g, X, WG, WU, hh, gate, up, best, ws = e
    hb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
    vp = {X: xv, WG: wg, WU: wu, hh: hb.unsqueeze(0)}
    if save_preacts:
        gb = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        ub = torch.empty(M, interm, device=x.device, dtype=x.dtype)
        vp.update({gate: gb.unsqueeze(0), up: ub.unsqueeze(0)})
    else:
        gb = ub = None
    g.execute_plan_at_index(vp, ws, index=best, handle=h)
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
# this dense bf16 shape the B200-tuned 2-CTA strategy brings the FROST GEMM core to
# nvjet parity while the fused epilogue avoids materialising ``dh``. It is the
# default so the fusion saving is captured, and because that saving grows as the
# workload gets pointwise-heavier (fp8 halves the GEMM and adds quant/scale
# pointwise; MoE grouped GEMMs are smaller and more memory-bound). Falls back to
# the pointwise path if FROST explicitly declines a shape, architecture, layout,
# or optional dependency.
_FROST_BWD = os.environ.get("CUDNN_GEMM_SWIGLU_FROST_BWD", "1") != "0"
_FROST_DECLINE_ERRORS = (NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError)


def _frost_dswiglu(dout2, Wd, gate, up):
    """Fused backward dgrad + dSwiGLU in ONE FROST kernel: ``dh = dout2 @ Wd`` with
    ``dup = dh*silu(gate)`` and ``dgate = dh*silu'(gate)*up`` as the GEMM epilogue,
    so the separate dh GEMM + two dSwiGLU pointwise kernels of :func:`_dswiglu`
    collapse into a single cuTeDSL bare-launch kernel. ``dout2:[M,H]``, ``Wd:[H,I]``
    (natural down weight), ``gate, up:[M,I]``. Returns ``(dgate, dup):[M,I]``. Raises
    on any unsupported case so the caller falls back to the pointwise path.

    FROST takes the natural down weight directly (an N-major / I-contiguous B -- no
    transpose copy; the downstream ``dg.t()`` wgrad operand is likewise a free strided
    view) and keeps ``dh`` on-chip (no HBM round-trip). This is the DEFAULT backward
    stage. The large-M path deliberately pins the B200-tuned
    M128/N256/Kbytes128, cluster2x1, 2CTAMMA, CLC strategy. The previous geometry-
    only sweep fixed 1CTAMMA/cluster1x1 and therefore did not cover this execution-
    strategy win. Small M retains the 1CTAMMA/cluster1x1 strategy. Not a transpose
    problem (dense bf16 needs none; ``dg.t()`` is a free view) and not a host-
    overhead one (host is ~1% of these compute-bound GEMMs, and #612 already cut
    it)."""
    operands = (("dout", dout2), ("Wd", Wd), ("gate", gate), ("up", up))
    if any(t.dim() != 2 for _, t in operands):
        raise NotImplementedError("cudnn.gemm.swiglu_mlp: FROST backward requires rank-2 dout/Wd/gate/up operands")
    M, H = dout2.shape
    interm = gate.shape[1]
    if Wd.shape != (H, interm) or gate.shape != (M, interm) or up.shape != (M, interm):
        raise NotImplementedError(
            "cudnn.gemm.swiglu_mlp: FROST backward operand shapes must be "
            f"dout[{M},{H}], Wd[{H},I], gate/up[{M},I]; got Wd{tuple(Wd.shape)}, gate{tuple(gate.shape)}, up{tuple(up.shape)}"
        )
    device = dout2.device
    if device.type != "cuda" or any(t.device != device for _, t in operands):
        raise NotImplementedError("cudnn.gemm.swiglu_mlp: FROST backward requires all operands on one CUDA device")
    if any(t.dtype != torch.bfloat16 for _, t in operands):
        raise NotImplementedError("cudnn.gemm.swiglu_mlp: FROST backward requires bfloat16 dout/Wd/gate/up operands")
    bad_strides = [f"{name}{tuple(tensor.stride())}" for name, tensor in operands if tuple(tensor.stride()) != (tensor.shape[1], 1)]
    if bad_strides:
        # The graph below deliberately declares dense strides. Do not bind a view
        # with different strides to that descriptor: square transposes preserve
        # the shape and otherwise make this mismatch particularly easy to miss.
        raise NotImplementedError("cudnn.gemm.swiglu_mlp: FROST backward requires exact row-major strides (cols, 1); " f"got {', '.join(bad_strides)}")
    if H % 8 or interm % 8:
        raise NotImplementedError("cudnn.gemm.swiglu_mlp: FROST backward's bf16 TMA operands require " f"H and I to be multiples of 8; got H={H}, I={interm}")
    # CuTeDSL currently derives its compilation target from visible CUDA
    # ordinal 0, not from the active context. A same-process heterogeneous-GPU
    # call must decline instead of compiling for one architecture and launching
    # that artifact on another; the caller's cuDNN backend path remains valid.
    if torch.cuda.get_device_capability(device) != torch.cuda.get_device_capability(0):
        raise NotImplementedError(
            "cudnn.gemm.swiglu_mlp: FROST backward requires the operand device " "architecture to match visible CUDA device 0's CuTeDSL JIT target"
        )
    misaligned = [name for name, tensor in operands if tensor.data_ptr() % 32]
    if misaligned:
        # 32 B is the maximum vector width used by this dense bf16 epilogue and
        # is stronger than the 16 B TMA base-pointer floor. Standard torch
        # allocations satisfy it; offset views decline rather than reaching a
        # runtime descriptor/alignment ValueError after JIT.
        raise NotImplementedError("cudnn.gemm.swiglu_mlp: FROST backward requires 32-byte-aligned operands; " f"misaligned: {', '.join(misaligned)}")

    # FROST's direct JIT path chooses its compile device from the current CUDA
    # context, unlike the cuDNN-handle paths above. Keep cache construction,
    # allocations, and launch under the tensor's device and pass the caller's
    # PyTorch stream explicitly (None would launch on CUDA stream 0).
    with torch.cuda.device(device):
        from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
        from cudnn.gemm.frost.tile_config import CATALOG

        stream = torch.cuda.current_stream(device).cuda_stream
        key = (M, H, interm, dout2.dtype, device.index)
        e = _FROST_DSWIGLU_CACHE.get(key)
        if e is None:
            tn = 256 if interm >= 256 else 128
            cta_group = 2 if M > 128 else 1
            cluster_m = 2 if cta_group == 2 else 1
            cfg = next(c for c in CATALOG if (c.cta_tile_m, c.cta_tile_n, c.cta_tile_k_bytes, c.cgrp_size_m, c.cgrp_size_n) == (128, tn, 128, cluster_m, 1))
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
            compiled = jit_from_cudnn_graph(g, config=cfg, cta_group=cta_group)
            bd = compiled.binding
            out_by = {o.get_name().split("::")[0]: o for o in bd.outputs}
            aux_by = {a.get_name(): a for a in bd.aux}
            e = (compiled, bd, out_by, aux_by)
            _FROST_DSWIGLU_CACHE[key] = e
        compiled, bd, out_by, aux_by = e
        dgate = torch.empty(1, M, interm, device=device, dtype=dout2.dtype)
        dup = torch.empty(1, M, interm, device=device, dtype=dout2.dtype)
        compiled(
            {
                bd.a_operands[0]: dout2.unsqueeze(0),
                bd.b_operands[0]: Wd.unsqueeze(0),  # natural [1,H,I], N-major — no transpose
                out_by["dup"]: dup,
                out_by["dgate"]: dgate,
                aux_by["gate_pre"]: gate.unsqueeze(0),
                aux_by["up_pre"]: up.unsqueeze(0),
            },
            stream=stream,
        )
    return dgate.squeeze(0), dup.squeeze(0)


class _SwigluMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Wg, Wu, Wd, grad_mask):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        need_fc1_grad = bool(grad_mask & _GRAD_FC1)
        h, gate, up = _swiglu_act(
            x2,
            Wg,
            Wu,
            save_preacts=need_fc1_grad,
        )  # fused: h-only unless an FC1-related backward needs gate/up
        out = _mm(h, Wd.t())  # transposed weight view; cuDNN reads it column-major, no copy

        saved_names = []
        saved_tensors = []

        def save(name, tensor):
            saved_names.append(name)
            saved_tensors.append(tensor)

        if grad_mask & (_GRAD_WG | _GRAD_WU):
            save("x2", x2)
        if grad_mask & _GRAD_X:
            save("Wg", Wg)
            save("Wu", Wu)
        if need_fc1_grad:
            save("Wd", Wd)
            save("gate", gate)
            # P1 keeps the existing dual-output dSwiGLU for every FC1-related
            # backward. A later single-output specialization can omit up for a
            # Wu-only gradient.
            save("up", up)
        if grad_mask & _GRAD_WD:
            save("h", h)

        ctx.save_for_backward(*saved_tensors)
        ctx.saved_names = tuple(saved_names)
        ctx.grad_mask = grad_mask
        ctx.shp = shp
        ctx.out_features = Wd.shape[0]
        return out.reshape(*shp[:-1], Wd.shape[0])

    @staticmethod
    def backward(ctx, dout):
        saved = dict(zip(ctx.saved_names, ctx.saved_tensors))
        grad_mask = ctx.grad_mask
        # Autograd may supply an expanded/zero-stride gradient (for example from
        # out.sum()). Backend GEMMs and the direct FROST descriptor require a
        # dense matrix; contiguous() is a no-op for the normal dense case.
        dout2 = dout.reshape(-1, ctx.out_features).contiguous()

        dWd = _mm(dout2.t(), saved["h"]) if grad_mask & _GRAD_WD else None

        if grad_mask & _GRAD_FC1:
            Wd, gate, up = (saved[name] for name in ("Wd", "gate", "up"))
            # Saved-tensor hooks may restore the internally dense preactivations
            # with a different valid stride. Both the FROST descriptor and the
            # pointwise fallback declare dense gate/up tensors, so normalize at
            # this boundary (a no-op for the ordinary path).
            gate, up = gate.contiguous(), up.contiguous()
            # P1 retains the existing dual-output dSwiGLU whenever either output
            # is needed. It still skips the whole stage for a Wd-only backward.
            if _FROST_BWD:
                try:
                    dgate, dup = _frost_dswiglu(dout2, Wd, gate, up)
                except _FROST_DECLINE_ERRORS:
                    dgate, dup = _dswiglu(_mm(dout2, Wd), gate, up)
            else:
                dgate, dup = _dswiglu(_mm(dout2, Wd), gate, up)
        else:
            dgate = dup = None

        if grad_mask & _GRAD_X:
            dx = _mm(dgate, saved["Wg"]) + _mm(dup, saved["Wu"])
            dx = dx.reshape(*ctx.shp)
        else:
            dx = None
        dWg = _mm(dgate.t(), saved["x2"]) if grad_mask & _GRAD_WG else None
        dWu = _mm(dup.t(), saved["x2"]) if grad_mask & _GRAD_WU else None
        return dx, dWg, dWu, dWd, None


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
    if any(t.device != x.device for t in (Wg, Wu, Wd)):
        raise ValueError(
            "cudnn.gemm.swiglu_mlp: x, Wg, Wu, and Wd must be on the same CUDA device; " f"got x={x.device}, Wg={Wg.device}, Wu={Wu.device}, Wd={Wd.device}"
        )
    if x.dim() < 2 or Wg.dim() != 2 or Wu.dim() != 2 or Wd.dim() != 2:
        raise ValueError(
            f"cudnn.gemm.swiglu_mlp: expected x[...,H] and 2-D Wg/Wu[I,H], Wd[H,I]; got x{tuple(x.shape)} Wg{tuple(Wg.shape)} Wu{tuple(Wu.shape)} Wd{tuple(Wd.shape)}"
        )
    H = x.shape[-1]
    if Wg.shape != Wu.shape or Wg.shape[1] != H or Wd.shape[0] != H or Wd.shape[1] != Wg.shape[0]:
        raise ValueError(
            f"cudnn.gemm.swiglu_mlp: shape mismatch — x[...,{H}], Wg{tuple(Wg.shape)}, Wu{tuple(Wu.shape)}, Wd{tuple(Wd.shape)}; "
            f"expected Wg/Wu = [I, {H}] and Wd = [{H}, I]"
        )
    # A custom Function's forward always runs with GradMode disabled, while
    # ctx.needs_input_grad still mirrors the inputs' requires_grad flags even under
    # an outer no_grad()/inference_mode(). Capture the outer state here so inference
    # selects the h-only graph even when model parameters remain trainable.
    grad_mask = 0
    if torch.is_grad_enabled():
        for bit, tensor in ((_GRAD_X, x), (_GRAD_WG, Wg), (_GRAD_WU, Wu), (_GRAD_WD, Wd)):
            if tensor.requires_grad:
                grad_mask |= bit
    return _SwigluMLP.apply(x, Wg, Wu, Wd, grad_mask)
