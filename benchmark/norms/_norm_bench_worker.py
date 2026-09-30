# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-backend norm benchmark worker (cuDNN frontend vs cuDNN frost).

Measures fwd + bwd GPU time (CUDA events, l2-flushed, median) for a single
(norm_type, N, C, dtype) shape and prints a RESULT line. Run one process per
backend so `import cudnn` means the pip frontend (backend=cudnn) or the repo's
frost package (backend=frost) without collision.

    python _norm_bench_worker.py --backend cudnn --norm_type rms_norm --N 16384 --C 4096 --dtype bfloat16
"""

import argparse
import sys
import types

import numpy as np

_REPO_CUDNN = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python" / "cudnn")


def _median_ms(fn, iters, warmup, l2buf):
    """Median GPU *kernel* time (ms) via the profiler device-time — excludes host
    dispatch overhead, matching benchmark_single_norm's methodology. Per-iter L2
    flush."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        l2buf.zero_()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with record_function("op"):
                fn()
            torch.cuda.synchronize()
        ka = prof.key_averages()
        ev = [i for i in ka if i.key == "op"]
        dt_us = ev[0].device_time if ev else sum(i.device_time for i in ka if i.device_time > 0)
        times.append(dt_us / 1000.0)
    return float(np.median(times))


def run_frost(a):
    # Frost lives in the repo as cudnn.norm; stub the cudnn package BEFORE import
    # so it resolves to the repo (not the pip frontend), with a dummy pygraph so
    # the cudnn.frost lifecycle patch installs.
    stub = types.ModuleType("cudnn")
    stub.__path__ = [_REPO_CUDNN]
    stub.pygraph = type("pygraph", (), {})
    sys.modules["cudnn"] = stub

    import torch
    import torch.nn.functional as F
    from cudnn.norm import NormVariant, norm_bprop, norm_fprop

    dt = getattr(torch, a.dtype)
    dev = "cuda"
    variant = NormVariant.RMS_NORM if a.norm_type == "rms_norm" else NormVariant.LAYER_NORM
    torch.manual_seed(0)
    x = torch.randn(a.N, a.C, device=dev, dtype=dt)
    g = torch.randn(a.C, device=dev, dtype=dt)
    b = torch.randn(a.C, device=dev, dtype=dt) if a.has_bias else None
    l2 = torch.empty(256 * 1024 * 1024, device=dev, dtype=torch.int8)

    # correctness vs torch
    y, mean, rstd = norm_fprop(variant, x, g, b, normalized_shape=[a.C], eps=a.epsilon)
    if a.norm_type == "rms_norm":
        yref = F.rms_norm(x.float(), (a.C,), g.float(), eps=a.epsilon)
    else:
        yref = F.layer_norm(x.float(), (a.C,), g.float(), b.float() if b is not None else None, a.epsilon)
    fwd_maxabs = (y.float() - yref).abs().max().item()

    dy = torch.randn_like(x)
    dx, dgamma, dbeta = norm_bprop(variant, dy, x, g, mean, rstd, normalized_shape=[a.C], has_beta=a.has_bias)
    xr = x.float().detach().requires_grad_(True)
    gr = g.float().detach().requires_grad_(True)
    if a.norm_type == "rms_norm":
        yr = F.rms_norm(xr, (a.C,), gr, eps=a.epsilon)
    else:
        yr = F.layer_norm(xr, (a.C,), gr, None, a.epsilon)
    dxr = torch.autograd.grad(yr, [xr], grad_outputs=dy.float())[0]
    bwd_maxabs = (dx.float() - dxr).abs().max().item() / max(1.0, dxr.abs().max().item())

    fwd_ms = _median_ms(lambda: norm_fprop(variant, x, g, b, normalized_shape=[a.C], eps=a.epsilon), a.iters, a.warmup, l2)
    bwd_ms = _median_ms(lambda: norm_bprop(variant, dy, x, g, mean, rstd, normalized_shape=[a.C], has_beta=a.has_bias), a.iters, a.warmup, l2)
    return fwd_ms, bwd_ms, fwd_maxabs, bwd_maxabs


def run_cudnn(a):
    import torch
    import cudnn

    dt = getattr(torch, a.dtype)
    dev = torch.device("cuda")
    N, C = a.N, a.C
    l2 = torch.empty(256 * 1024 * 1024, device=dev, dtype=torch.int8)

    x = torch.randn(N, C, 1, 1, dtype=dt, device=dev)
    scale = torch.randn(1, C, 1, 1, dtype=dt, device=dev)
    bias = torch.randn(1, C, 1, 1, dtype=dt, device=dev) if a.has_bias else None
    eps = torch.full((1, 1, 1, 1), a.epsilon, dtype=torch.float32, device="cpu")

    # ---- forward graph ----
    gf = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    X = gf.tensor_like(x.detach())
    S = gf.tensor_like(scale.detach())
    Bt = gf.tensor_like(bias.detach()) if a.has_bias else None
    E = gf.tensor_like(eps)
    if a.norm_type == "rms_norm":
        Y, INV = gf.rmsnorm(name="RMS", norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=S, bias=Bt, epsilon=E)
        MEAN = None
    else:
        Y, MEAN, INV = gf.layernorm(name="LN", norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=S, bias=Bt, epsilon=E)
    Y.set_output(True).set_data_type(dt)
    INV.set_output(True).set_data_type(torch.float32)
    if MEAN is not None:
        MEAN.set_output(True).set_data_type(torch.float32)
    gf.validate()
    gf.build_operation_graph()
    gf.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    gf.check_support()
    gf.build_plans()

    y = torch.empty(N, C, 1, 1, dtype=dt, device=dev)
    invv = torch.empty(N, 1, 1, 1, dtype=torch.float32, device=dev)
    meanv = torch.empty(N, 1, 1, 1, dtype=torch.float32, device=dev) if a.norm_type == "layer_norm" else None
    vpf = {X: x.detach(), S: scale.detach(), E: eps, Y: y, INV: invv}
    if a.has_bias:
        vpf[Bt] = bias.detach()
    if MEAN is not None:
        vpf[MEAN] = meanv

    # ---- backward graph ----
    dy = torch.randn(N, C, 1, 1, dtype=dt, device=dev)
    gb = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    DY = gb.tensor_like(dy.detach())
    Xb = gb.tensor_like(x.detach())
    Sb = gb.tensor_like(scale.detach())
    INVb = gb.tensor_like(invv)
    if a.norm_type == "rms_norm":
        DX, DS, DB = gb.rmsnorm_backward(name="DRMS", grad=DY, input=Xb, scale=Sb, inv_variance=INVb, has_dbias=a.has_bias)
        MEANb = None
    else:
        MEANb = gb.tensor_like(meanv)
        DX, DS, DB = gb.layernorm_backward(name="DLN", grad=DY, input=Xb, scale=Sb, mean=MEANb, inv_variance=INVb)
    # cuDNN norm-backward engines want the parameter grads in the IO dtype (the
    # backend declines fp32 DScale with CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE).
    DX.set_output(True).set_data_type(dt)
    DS.set_output(True).set_data_type(dt)
    if DB is not None:
        DB.set_output(True).set_data_type(dt)
    gb.validate()
    gb.build_operation_graph()
    gb.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    gb.check_support()
    gb.build_plans()

    dxb = torch.empty_like(x)
    dsb = torch.empty(1, C, 1, 1, dtype=dt, device=dev)
    dbb = torch.empty(1, C, 1, 1, dtype=dt, device=dev) if a.has_bias else None
    vpb = {DY: dy.detach(), Xb: x.detach(), Sb: scale.detach(), INVb: invv, DX: dxb, DS: dsb}
    if DB is not None and dbb is not None:
        vpb[DB] = dbb
    if MEANb is not None:
        vpb[MEANb] = meanv

    ws_bytes = max(gf.get_workspace_size(), gb.get_workspace_size())
    ws = torch.empty(ws_bytes, device=dev, dtype=torch.uint8)

    # run fwd once to populate invv/meanv for a meaningful bwd
    gf.execute(vpf, ws)
    torch.cuda.synchronize()

    fwd_ms = _median_ms(lambda: gf.execute(vpf, ws), a.iters, a.warmup, l2)
    bwd_ms = _median_ms(lambda: gb.execute(vpb, ws), a.iters, a.warmup, l2)
    return fwd_ms, bwd_ms, -1.0, -1.0  # correctness checked in the frost worker


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["cudnn", "frost"])
    p.add_argument("--norm_type", required=True, choices=["rms_norm", "layer_norm"])
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--C", type=int, required=True)
    p.add_argument("--has_bias", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--epsilon", type=float, default=1e-5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    a = p.parse_args()
    a.has_bias = bool(a.has_bias)
    fwd_ms, bwd_ms, fwd_maxabs, bwd_maxabs = (run_cudnn if a.backend == "cudnn" else run_frost)(a)
    print(f"RESULT backend={a.backend} fwd_ms={fwd_ms:.5f} bwd_ms={bwd_ms:.5f} " f"fwd_maxabs={fwd_maxabs:.3e} bwd_maxrel={bwd_maxabs:.3e}")


if __name__ == "__main__":
    main()
