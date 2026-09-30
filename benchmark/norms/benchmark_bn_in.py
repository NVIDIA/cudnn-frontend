# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BatchNorm / InstanceNorm forward: frost vs cuDNN vs copy-kernel ceiling, on RN50
shapes, for BOTH NCHW and NHWC layouts. GroupNorm has no cuDNN reference so it is
benchmarked vs the ceiling only (optional).

Metric: GB/s counting the MINIMAL 1R+1W traffic (2*N*C*H*W*elem_bytes), so a kernel
that hits the pure-copy ceiling reports ~ the copy GB/s; a 2-pass (re-read x) kernel
reports ~2/3 of it. Times are profiler device-time (kernel-only, dispatch-excluded).

Run inside the frost venv WITHOUT PYTHONPATH pollution (the cudnn frontend is the pip
package; frost is imported via a stub only in the frost path, in a subprocess-free way
here since we don't need both cudnn packages simultaneously per-process).
"""

import argparse, sys, types, statistics
import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function

# RN50 FP8 BN layers (N,C,H,W); N shrinkable via --n
RN50 = [(64, 56, 56), (256, 56, 56), (128, 28, 28), (256, 28, 28), (512, 14, 14), (512, 7, 7), (2048, 7, 7)]

_l2 = None


def _dev_time(fn, iters=40, warmup=12):
    global _l2
    if _l2 is None:
        _l2 = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        _l2.zero_()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            with record_function("op"):
                fn()
            torch.cuda.synchronize()
        ka = p.key_averages()
        ts.append(sum(i.device_time for i in ka if i.device_time > 0) / 1000.0)
    return float(statistics.median(ts))


def _mk_x(N, C, H, W, dt, layout):
    x = torch.randn(N, C, H, W, dtype=dt, device="cuda")
    if layout == "nhwc":
        x = x.to(memory_format=torch.channels_last)
    return x


def copy_ceiling(N, C, H, W, dt, layout):
    x = _mk_x(N, C, H, W, dt, layout)
    y = torch.empty_like(x)
    t = _dev_time(lambda: y.copy_(x))
    gb = 2 * N * C * H * W * x.element_size() / (t * 1e-3) / 1e9
    return gb


def run_cudnn(variant, N, C, H, W, dt, layout):
    import cudnn

    x = _mk_x(N, C, H, W, dt, layout)
    scale = torch.ones(1, C, 1, 1, dtype=dt, device="cuda")
    bias = torch.zeros(1, C, 1, 1, dtype=dt, device="cuda")
    eps = torch.full((1, 1, 1, 1), 1e-5, dtype=torch.float32, device="cpu")
    g = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    X = g.tensor_like(x.detach())
    S = g.tensor_like(scale.detach())
    Bt = g.tensor_like(bias.detach())
    E = g.tensor_like(eps)
    if variant == "bn":
        rm = torch.zeros(1, C, 1, 1, dtype=torch.float32, device="cuda")
        rv = torch.ones(1, C, 1, 1, dtype=torch.float32, device="cuda")
        RM = g.tensor_like(rm)
        RV = g.tensor_like(rv)
        mom = torch.full((1, 1, 1, 1), 0.1, dtype=torch.float32, device="cpu")
        M = g.tensor_like(mom)
        Y, MEAN, INV, NRM, NRV = g.batchnorm(name="BN", input=X, scale=S, bias=Bt, in_running_mean=RM, in_running_var=RV, epsilon=E, momentum=M)
        outs = [NRM, NRV]
    else:
        Y, MEAN, INV = g.instancenorm(name="IN", norm_forward_phase=cudnn.norm_forward_phase.TRAINING, input=X, scale=S, bias=Bt, epsilon=E)
        outs = []
    Y.set_output(True).set_data_type(dt)
    MEAN.set_output(True).set_data_type(torch.float32)
    INV.set_output(True).set_data_type(torch.float32)
    for o in outs:
        o.set_output(True).set_data_type(torch.float32)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    y = torch.empty_like(x)
    statshape = (1, C, 1, 1) if variant == "bn" else (N, C, 1, 1)
    meanv = torch.empty(statshape, dtype=torch.float32, device="cuda")
    invv = torch.empty(statshape, dtype=torch.float32, device="cuda")
    vp = {X: x.detach(), S: scale.detach(), Bt: bias.detach(), E: eps, Y: y, MEAN: meanv, INV: invv}
    if variant == "bn":
        vp[RM] = rm
        vp[RV] = rv
        vp[M] = mom
        vp[NRM] = torch.empty(1, C, 1, 1, dtype=torch.float32, device="cuda")
        vp[NRV] = torch.empty(1, C, 1, 1, dtype=torch.float32, device="cuda")
    ws = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute(vp, ws)
    torch.cuda.synchronize()
    t = _dev_time(lambda: g.execute(vp, ws))
    gb = 2 * N * C * H * W * x.element_size() / (t * 1e-3) / 1e9
    return gb


def run_frost(variant, N, C, H, W, dt, layout):
    D = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "python" / "cudnn")
    if "cudnn" not in sys.modules or not hasattr(sys.modules["cudnn"], "norm"):
        stub = types.ModuleType("cudnn")
        stub.__path__ = [D]
        stub.pygraph = type("pygraph", (), {})
        sys.modules["cudnn"] = stub
    from cudnn.norm import NormVariant, norm_fprop

    nv = NormVariant.BATCH_NORM if variant == "bn" else NormVariant.INSTANCE_NORM
    x = _mk_x(N, C, H, W, dt, layout)
    g = torch.ones(C, dtype=dt, device="cuda")
    b = torch.zeros(C, dtype=dt, device="cuda")
    kw = {}
    if variant == "bn":
        kw = dict(
            running_mean=torch.zeros(C, dtype=torch.float32, device="cuda"),
            running_var=torch.ones(C, dtype=torch.float32, device="cuda"),
            momentum=0.1,
            training=True,
        )
    fn = lambda: norm_fprop(nv, x, g, b, eps=1e-5, **kw)
    fn()  # warm/validate
    t = _dev_time(fn)
    gb = 2 * N * C * H * W * x.element_size() / (t * 1e-3) / 1e9
    return gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cudnn", "frost", "copy"], required=True)
    ap.add_argument("--variant", choices=["bn", "in"], required=True)
    ap.add_argument("--layout", choices=["nchw", "nhwc"], required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    for C, H, W in RN50:
        tag = f"{a.variant}/{a.layout} N={a.n} C={C:4} {H}x{W}"
        try:
            if a.backend == "copy":
                gb = copy_ceiling(a.n, C, H, W, dt, a.layout)
            elif a.backend == "cudnn":
                gb = run_cudnn(a.variant, a.n, C, H, W, dt, a.layout)
            else:
                gb = run_frost(a.variant, a.n, C, H, W, dt, a.layout)
            print(f"{tag}: {gb:.0f} GB/s", flush=True)
        except Exception as e:
            print(f"{tag}: ERR {str(e)[:70]}", flush=True)


if __name__ == "__main__":
    main()
