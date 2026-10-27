# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare all three backward outputs with identical preallocated buffers."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from probe import load_kernel, reference
from backend import Backend
from tune import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="2,768,4096")
    parser.add_argument("--layout", choices=["ncw", "nwc"], default="nwc")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--backend-library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend", action="store_true")
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--rows", default="8,16,32")
    args = parser.parse_args()
    B, D, L = map(int, args.shape.split(","))
    nwh = args.layout == "nwc"
    row_choices = list(map(int, args.rows.split(",")))
    if any(r <= 0 or (nwh and args.width <= 16 and r % 8 != 0) for r in row_choices):
        raise ValueError("Backward tiles must be positive; short-filter NWH tiles must be multiples of eight")
    dtype = getattr(torch, args.dtype)
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    generator = torch.Generator().manual_seed(20261022)

    def rand(shape):
        return torch.randn(shape, generator=generator).to(device="cuda", dtype=dtype)

    x = rand((B, L, D) if nwh else (B, D, L))
    w = rand((args.width, D) if nwh else (D, args.width))
    b = rand((D,))
    dy = rand(x.shape)
    # Channels are independent: bound FP64 autograd memory for the throughput-sized cases.
    refs = (torch.empty_like(x, dtype=torch.float64), torch.empty_like(w, dtype=torch.float64), torch.empty_like(b, dtype=torch.float64))
    channel_chunk = max(1, min(D, 2_000_000 // (B * L)))
    for first in range(0, D, channel_chunk):
        last = min(D, first + channel_chunk)
        xs = x[:, :, first:last] if nwh else x[:, first:last, :]
        ws = w[:, first:last] if nwh else w[first:last, :]
        dys = dy[:, :, first:last] if nwh else dy[:, first:last, :]
        xx, ww, bb = (t.double().requires_grad_() for t in (xs, ws, b[first:last]))
        grads = torch.autograd.grad(reference(xx, ww, bb, not args.identity, nwh), (xx, ww, bb), dys.double())
        if nwh:
            refs[0][:, :, first:last].copy_(grads[0])
            refs[1][:, first:last].copy_(grads[1])
        else:
            refs[0][:, first:last, :].copy_(grads[0])
            refs[1][first:last, :].copy_(grads[1])
        refs[2][first:last].copy_(grads[2])
        del xx, ww, bb, grads
    tol = {"bfloat16": (0.03, 0.02), "float16": (0.003, 0.002), "float32": (2e-4, 2e-4), "float64": (1e-10, 1e-10)}[args.dtype]

    def check(outputs):
        errors = {}
        for name, out, ref in zip(("dx", "dw", "db"), outputs, refs):
            # Parameter gradients stay FP32: allow summation error proportional to sqrt(B*L).
            atol = tol[0] if name == "dx" else (1e-10 if compute == torch.float64 else 2e-5 * (B * L) ** 0.5)
            rtol = tol[1] if name == "dx" else (1e-10 if compute == torch.float64 else 2e-4)
            torch.testing.assert_close(out, ref.to(out.dtype), atol=atol, rtol=rtol)
            errors[name] = (out.double() - ref).abs().max().item()
        return errors

    backend = Backend(args.backend_library)
    bo = (torch.empty_like(x), torch.empty_like(w, dtype=compute), torch.empty_like(b, dtype=compute))

    def be_run():
        backend.backward(x, w, b, dy, *bo, channel_last=nwh, silu=not args.identity)

    be_run()
    errors = {"backend": check(bo)}
    functions = {"backend": be_run}
    retained = []
    module = load_kernel(True)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    for rows in map(int, args.rows.split(",")):
        dz = torch.empty_like(x, dtype=compute)
        outputs = (torch.empty_like(x), torch.empty_like(w, dtype=compute), torch.empty_like(b, dtype=compute))
        fn = cute.compile(
            module.launch_bwd,
            *(from_dlpack(t, assumed_align=16) for t in (x, w, b, dy, dz, *outputs)),
            stream,
            args.width,
            not args.identity,
            nwh,
            rows,
            options="--enable-tvm-ffi",
        )

        def run(fn=fn, dz=dz, outputs=outputs):
            fn(x, w, b, dy, dz, *outputs, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        run()
        name = f"rows{rows}"
        errors[name] = check(outputs)
        functions[name] = run
        retained.append((fn, dz, outputs))
        print(f"{name}: correct", flush=True)
    if args.frontend:
        if not nwh or args.width != 4 or dtype != torch.bfloat16 or args.identity:
            raise ValueError("Existing frontend comparison requires NWC BF16 K4 SiLU")
        from cudnn.causal_conv1d_bulk_sm100.backward import CausalConv1dBulkBwdPrototype

        ow = w.T.contiguous()
        outputs = (torch.empty_like(x), torch.empty_like(ow, dtype=compute), torch.empty_like(b, dtype=compute))
        old = CausalConv1dBulkBwdPrototype(x, ow, dy, sample_bias=b)
        old.check_support()
        old.compile()
        workspace = torch.empty(old.dweight_workspace_numel, device="cuda", dtype=torch.float32) if old.dweight_workspace_numel else None

        def fe_run():
            old.execute(x, ow, dy, outputs[0], outputs[1], bias=b, db_accum=outputs[2], dweight_workspace=workspace)

        fe_run()
        errors["frontend"] = check((outputs[0], outputs[1].T, outputs[2]))
        functions["frontend"] = fe_run
    results = measure(functions, cold=args.cold)
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "max_abs_vs_fp64": errors,
        "results": results,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(module.__file__), Path(__file__))},
        "cache_policy": "512 MiB eviction before every sample" if args.cold else "warm repeated inputs; no eviction",
        "repeats": 1 if args.cold else 100,
        "rounds": 21,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: round(v["median_us"], 3) for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
