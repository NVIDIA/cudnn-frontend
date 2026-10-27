# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare a direct CuTe DSL schedule with an independent convolution reference."""

import argparse
import importlib.util
import json
import statistics
import sys
import warnings
from pathlib import Path

import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


def load_kernel(backward=False):
    """Load the forward or backward candidate module from this worktree."""
    path = Path(__file__).resolve().parents[2] / "python/cudnn/causal_conv1d_se" / ("backward_kernel.py" if backward else "kernel.py")
    spec = importlib.util.spec_from_file_location("causal_se_backward" if backward else "causal_se_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference(x, w, bias, silu, nwh):
    """Compute causal depthwise convolution and optional SiLU with FP64 PyTorch operations."""
    xx = x.transpose(1, 2) if nwh else x
    ww = w.T if nwh else w
    acc = bias.double()[None, :, None].expand_as(xx).clone()
    for tap in range(ww.shape[1]):
        shift = ww.shape[1] - 1 - tap
        if shift < xx.shape[2]:
            acc[:, :, shift:] += xx[:, :, : xx.shape[2] - shift].double() * ww[:, tap].double()[None, :, None]
    if silu:
        acc = torch.nn.functional.silu(acc)
    return acc.transpose(1, 2) if nwh else acc


def graph_time(fn):
    """Return median device microseconds per call from repeated CUDA-graph replays."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
        with torch.cuda.graph(graph):
            for _ in range(20):
                fn()
    vals = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        vals.append(a.elapsed_time(b) * 1000 / 20)
    return statistics.median(vals)


def main():
    """Check one candidate against the FP64 reference and optional backend and frontend comparisons."""
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="2,7,65")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layout", choices=["ncw", "nwc"], default="ncw")
    p.add_argument("--identity", action="store_true")
    p.add_argument("--tile", type=int, default=8)
    p.add_argument("--vector", type=int, default=1)
    p.add_argument("--packed", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--backend-library")
    p.add_argument("--frontend", action="store_true")
    p.add_argument("--backward", action="store_true")
    args = p.parse_args()
    if args.backward and args.frontend:
        raise ValueError("Use tune_bwd.py --frontend for the three-way backward comparison")
    if args.vector not in (1, 2, 4, 8):
        raise ValueError("vector must be 1, 2, 4, or 8")
    if args.packed and (args.layout != "nwc" or args.vector == 1 or torch.cuda.get_device_capability()[0] < 10):
        raise ValueError("Packed FP32 arithmetic requires vectorized NWC on SM100 or newer")
    if not 2 <= args.width <= (128 if args.layout == "nwc" else 256):
        raise ValueError("Filter width is outside the backend's supported range")
    B, D, L = map(int, args.shape.split(","))
    nwh = args.layout == "nwc"
    generator = torch.Generator().manual_seed(20261022)
    dtype = getattr(torch, args.dtype)
    if args.vector > 1:
        if nwh and (D % args.vector or dtype == torch.float64):
            raise ValueError("NWC vector schedule requires divisible channels and BF16/FP16/FP32")
        if not nwh and (L % 8 or args.tile != 8 or dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)):
            raise ValueError("NCW vector schedule requires 8-aligned length, tile 8, and a supported floating-point dtype")
    x = torch.randn((B, L, D) if nwh else (B, D, L), generator=generator).to(device="cuda", dtype=dtype)
    w = torch.randn((args.width, D) if nwh else (D, args.width), generator=generator).to(device="cuda", dtype=dtype)
    b = torch.randn((D,), generator=generator).to(device="cuda", dtype=dtype)
    y = torch.empty_like(x)
    module = load_kernel(args.backward)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if args.backward:
        dy = torch.randn(x.shape, generator=generator).to(device="cuda", dtype=dtype)
        compute = torch.float64 if dtype == torch.float64 else torch.float32
        dz = torch.empty_like(x, dtype=compute)
        dw, db = torch.empty_like(w, dtype=compute), torch.empty_like(b, dtype=compute)
        fn = cute.compile(
            module.launch_bwd,
            *(from_dlpack(t, assumed_align=16) for t in (x, w, b, dy, dz, y, dw, db)),
            stream,
            args.width,
            not args.identity,
            nwh,
            args.tile,
            options="--enable-tvm-ffi",
        )

        def run_bwd():
            """Launch candidate backward into the preallocated gradient buffers."""
            fn(x, w, b, dy, dz, y, dw, db, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

        run_bwd()
        xx, ww, bb = (t.double().requires_grad_() for t in (x, w, b))
        ref_y = reference(xx, ww, bb, not args.identity, nwh)
        refs = torch.autograd.grad(ref_y, (xx, ww, bb), dy.double())
        tol = {"bfloat16": (0.03, 0.02), "float16": (0.003, 0.002), "float32": (2e-4, 2e-4), "float64": (1e-10, 1e-10)}[args.dtype]
        errors = {}
        for name, out, ref in zip(("dx", "dw", "db"), (y, dw, db), refs):
            torch.testing.assert_close(
                out,
                ref.to(out.dtype),
                atol=tol[0] if name == "dx" else (1e-10 if compute == torch.float64 else 2e-3),
                rtol=tol[1] if name == "dx" else (1e-10 if compute == torch.float64 else 2e-4),
            )
            errors[name] = (out.double() - ref).abs().max().item()
        comparison = {}
        if args.backend_library:
            from backend import Backend

            be = Backend(args.backend_library)
            be_out = [torch.empty_like(t) for t in (y, dw, db)]

            def be_run():
                """Run backend backward with the selected layout and activation."""
                be.backward(x, w, b, dy, *be_out, channel_last=nwh, silu=not args.identity)

            be_run()
            for name, out, ref in zip(("dx", "dw", "db"), (y, dw, db), be_out):
                torch.testing.assert_close(out, ref, atol=tol[0], rtol=tol[1])
                comparison[name + "_backend_max_abs"] = (out.double() - ref.double()).abs().max().item()
            comparison["backend_us"] = graph_time(be_run) if args.timing else None
        print(
            json.dumps(
                {
                    "args": vars(args),
                    "gpu": torch.cuda.get_device_name(),
                    "cc": torch.cuda.get_device_capability(),
                    "dsl": cutlass.__version__,
                    "max_abs_vs_fp64": errors,
                    "us": graph_time(run_bwd) if args.timing else None,
                    **comparison,
                }
            ),
            flush=True,
        )
        return
    fn = cute.compile(
        module.launch_fwd,
        *(from_dlpack(t, assumed_align=16) for t in (x, w, b, y)),
        stream,
        args.width,
        not args.identity,
        nwh,
        args.tile,
        args.vector,
        args.packed,
        options="--enable-tvm-ffi",
    )

    def run():
        """Launch candidate forward into the preallocated output buffer."""
        fn(x, w, b, y, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    run()
    expected = reference(x, w, b, not args.identity, nwh)
    torch.cuda.synchronize()
    error = (y.double() - expected).abs()
    tol = {"bfloat16": (0.02, 0.02), "float16": (0.002, 0.002), "float32": (1e-5, 1e-5), "float64": (1e-12, 1e-12)}[args.dtype]
    torch.testing.assert_close(y, expected.to(dtype), atol=tol[0], rtol=tol[1])
    comparison = {}
    if args.backend_library:
        from backend import Backend

        backend = Backend(args.backend_library)
        backend_output = torch.empty_like(x)

        def backend_run():
            """Run backend forward into its comparison output."""
            backend.forward(x, w, b, backend_output, channel_last=nwh, silu=not args.identity)

        backend_run()
        torch.testing.assert_close(y, backend_output, atol=tol[0], rtol=tol[1])
        comparison["backend_max_abs"] = (y.double() - backend_output.double()).abs().max().item()
        comparison["backend_us"] = graph_time(backend_run) if args.timing else None
    if args.frontend:
        if not nwh or args.width != 4 or args.dtype != "bfloat16" or args.identity:
            raise ValueError("Existing frontend native arm requires NWC BF16 K4 SiLU")
        from cudnn.causal_conv1d_bulk_sm100 import CausalConv1dBulkFwdSm100

        old_w = w.T.contiguous()
        old_y = torch.empty_like(x)
        old = CausalConv1dBulkFwdSm100(sample_x=x, sample_weight=old_w, sample_bias=b, sample_output=old_y)
        old.check_support()
        old.compile()

        def frontend_run():
            """Run the existing frontend forward implementation into its comparison output."""
            old.execute(x, old_w, old_y, bias_tensor=b)

        frontend_run()
        torch.testing.assert_close(y, old_y, atol=tol[0], rtol=tol[1])
        comparison["frontend_max_abs"] = (y.double() - old_y.double()).abs().max().item()
        comparison["frontend_us"] = graph_time(frontend_run) if args.timing else None
    print(
        json.dumps(
            {
                "args": vars(args),
                "gpu": torch.cuda.get_device_name(),
                "cc": torch.cuda.get_device_capability(),
                "dsl": cutlass.__version__,
                "max_abs_vs_fp64": error.max().item(),
                "us": graph_time(run) if args.timing else None,
                **comparison,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
