# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Back-to-back causal-conv comparison, including the stored-output backward contract."""

import argparse
import hashlib
import ctypes
import importlib.util
import json
from pathlib import Path
import sys
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from probe import reference
from tune import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="2,7,65")
    parser.add_argument("--projection-width", type=int, default=3)
    parser.add_argument("--mixer-width", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--backend-library", required=True)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    B, D, L = map(int, args.shape.split(","))
    Kp, Km = args.projection_width, args.mixer_width
    dtype = getattr(torch, args.dtype)
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    gen = torch.Generator().manual_seed(20261022)

    def rand(shape):
        return (torch.randn(shape, generator=gen) * 0.25).to(device="cuda", dtype=dtype)

    x, wp, wm, skip = rand((B, 3 * D, L)), rand((3 * D, Kp)), rand((D, Km)), rand((D,))
    y = torch.empty((B, D, L), device="cuda", dtype=dtype)
    yg = torch.empty_like(y)
    dy = rand(y.shape)
    dx, dwp, dwm, ds = torch.empty_like(x), torch.empty_like(wp, dtype=compute), torch.empty_like(wm, dtype=compute), torch.empty_like(skip, dtype=compute)
    path = Path(__file__).resolve().parents[2] / "python/cudnn/causal_conv1d_se/b2b_kernel.py"
    spec = importlib.util.spec_from_file_location("causal_se_b2b", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    forward = cute.compile(
        module.launch_b2b_fwd, *(from_dlpack(t, assumed_align=16) for t in (x, wp, wm, skip, y, yg)), stream, Kp, Km, options="--enable-tvm-ffi"
    )
    backward = cute.compile(
        module.launch_b2b_bwd,
        *(from_dlpack(t, assumed_align=16) for t in (x, wp, wm, skip, y, dy, dx, dwp, dwm, ds)),
        stream,
        Kp,
        Km,
        options="--enable-tvm-ffi",
    )

    def fwd():
        forward(x, wp, wm, skip, y, yg, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    def bwd():
        backward(x, wp, wm, skip, y, dy, dx, dwp, dwm, ds, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    fwd()
    xx, wpp, wmm, ss = (t.double().requires_grad_() for t in (x, wp, wm, skip))
    projection = reference(xx, wpp, torch.zeros(3 * D, device="cuda", dtype=torch.float64), False, False).reshape(B, D, 3, L)
    p1, p2, pv = projection.unbind(2)
    product = p2 * pv
    ref_y = reference(product, wmm, torch.zeros(D, device="cuda", dtype=torch.float64), False, False) + product * ss[None, :, None]
    ref_yg = ref_y * p1
    tol = {"bfloat16": (0.003, 0.03), "float16": (0.0003, 0.003), "float32": (2e-6, 2e-5), "float64": (1e-12, 1e-12)}[args.dtype]
    for out, ref in ((y, ref_y), (yg, ref_yg)):
        torch.testing.assert_close(out, ref.to(dtype), atol=tol[0], rtol=tol[1])
    # Backend differentiates the gated output using the saved, rounded ungated output for d(projected x1).
    grad_outputs = torch.stack((dy.double() * y.double(), torch.zeros_like(p2), torch.zeros_like(pv)), dim=2)
    g_product = torch.autograd.grad(ref_y, product, dy.double() * p1, retain_graph=True)[0]
    grad_outputs[:, :, 1] = g_product * pv
    grad_outputs[:, :, 2] = g_product * p2
    ref_dx, ref_dwp = torch.autograd.grad(projection, (xx, wpp), grad_outputs, retain_graph=True)
    ref_dwm, ref_ds = torch.autograd.grad(ref_y, (wmm, ss), dy.double() * p1)
    refs = (ref_dx, ref_dwp, ref_dwm, ref_ds)
    bwd()
    for out, ref in zip((dx, dwp, dwm, ds), refs):
        torch.testing.assert_close(out, ref.to(out.dtype), atol=tol[0], rtol=tol[1])
    lib = ctypes.CDLL(str(Path(args.backend_library).resolve(strict=True)))
    be_y, be_yg = torch.empty_like(y), torch.empty_like(yg)
    be_out = [torch.empty_like(t) for t in (dx, dwp, dwm, ds)]
    typecode = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 9}[dtype]
    bf = lib.cudnnB2BCausalConv1dForward
    bf.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 6
    bf.restype = ctypes.c_int
    bb = lib.cudnnB2BCausalConv1dBackward
    bb.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 7
    bb.restype = ctypes.c_int

    def be_fwd():
        status = bf(torch.cuda.current_stream().cuda_stream, *(t.data_ptr() for t in (x, wp, wm, skip, be_y, be_yg)), B, D, L, Kp, Km, typecode)
        if status:
            raise RuntimeError(f"backend forward status {status}")

    def be_bwd():
        for t in be_out[1:]:
            t.zero_()
        status = bb(
            torch.cuda.current_stream().cuda_stream,
            *(t.data_ptr() for t in (x, wp, wm, skip, y, dy, *be_out)),
            B,
            D,
            L,
            Kp,
            Km,
            typecode,
            1 if compute == torch.float64 else 0,
        )
        if status:
            raise RuntimeError(f"backend backward status {status}")

    be_fwd()
    be_bwd()
    errors = {}
    for name, out, ref in zip(("y", "yg", "dx", "dwp", "dwm", "ds"), (y, yg, dx, dwp, dwm, ds), (be_y, be_yg, *be_out)):
        torch.testing.assert_close(out, ref, atol=tol[0], rtol=tol[1])
        errors[name] = (out.double() - ref.double()).abs().max().item()
    measurements = measure({"candidate_fwd": fwd, "backend_fwd": be_fwd, "candidate_bwd": bwd, "backend_bwd": be_bwd}) if args.timing else {}
    print(
        json.dumps(
            {
                "args": vars(args),
                "gpu": torch.cuda.get_device_name(),
                "cc": torch.cuda.get_device_capability(),
                "backend_max_abs": errors,
                "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, Path(__file__))},
                "timing": measurements,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
