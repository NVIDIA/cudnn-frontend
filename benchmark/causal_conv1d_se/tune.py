# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interleaved, preallocated CUDA-graph comparison; compilation is outside timing."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import warnings
import torch
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from probe import load_kernel, reference
from backend import Backend


def measure(functions, repeats=100, rounds=21, cold=False):
    eviction = torch.empty(512 * 1024 * 1024, device="cuda", dtype=torch.uint8) if cold else None
    if cold:
        repeats = 1
    graphs = {}
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="The CUDA Graph is empty.*")
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    fn()
        graphs[name] = graph
    start = time.monotonic()
    while time.monotonic() - start < 0.3:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    rng = random.Random(20261022)
    samples = {name: [] for name in graphs}
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            if eviction is not None:
                eviction.zero_()
            begin.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) * 1000 / repeats)
    return {name: {"median_us": statistics.median(vals), "samples_us": vals} for name, vals in samples.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="2,768,4096")
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--layout", choices=["ncw", "nwc"], default="nwc")
    p.add_argument("--identity", action="store_true")
    p.add_argument("--backend-library", required=True)
    p.add_argument("--frontend", action="store_true")
    p.add_argument("--cold", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--schedules", default="1:4,1:8,1:16,2:4,2:8,4:4,4:8,8:2,8:4,8:8")
    args = p.parse_args()
    B, D, L = map(int, args.shape.split(","))
    nwh = args.layout == "nwc"
    dtype = getattr(torch, args.dtype)
    gen = torch.Generator().manual_seed(20261022)
    x = torch.randn((B, L, D) if nwh else (B, D, L), generator=gen).to(device="cuda", dtype=dtype)
    w = torch.randn((args.width, D) if nwh else (D, args.width), generator=gen).to(device="cuda", dtype=dtype)
    b = torch.randn(D, generator=gen).to(device="cuda", dtype=dtype)
    expected = reference(x, w, b, not args.identity, nwh).to(dtype)
    tol = {"bfloat16": (0.02, 0.02), "float16": (0.002, 0.002), "float32": (1e-5, 1e-5), "float64": (1e-12, 1e-12)}[args.dtype]
    backend = Backend(args.backend_library)
    by = torch.empty_like(x)

    def be_run():
        backend.forward(x, w, b, by, channel_last=nwh, silu=not args.identity)

    be_run()
    torch.testing.assert_close(by, expected, atol=tol[0], rtol=tol[1])
    functions = {"backend": be_run}
    retained = []
    if args.frontend:
        if not nwh or args.width != 4 or dtype != torch.bfloat16 or args.identity:
            raise ValueError("Existing frontend comparison requires NWC BF16 K4 SiLU")
        from cudnn.causal_conv1d_bulk_sm100 import CausalConv1dBulkFwdSm100

        old_w, old_y = w.T.contiguous(), torch.empty_like(x)
        old = CausalConv1dBulkFwdSm100(sample_x=x, sample_weight=old_w, sample_bias=b, sample_output=old_y)
        old.check_support()
        old.compile()

        def fe_run():
            old.execute(x, old_w, old_y, bias_tensor=b)

        fe_run()
        torch.testing.assert_close(old_y, expected, atol=tol[0], rtol=tol[1])
        functions["frontend"] = fe_run
    module = load_kernel()
    fixed = args.schedules == "auto"
    if fixed:
        if not nwh:
            vector, rows = (8 if L % 8 == 0 and dtype != torch.float64 else 1), 8
        elif args.width >= 64:
            vector, rows = 1, 4
        else:
            vector = 4 if D % 4 == 0 and (D > 768 or B * D * L > 10_000_000) else (2 if D % 2 == 0 else 1)
            if dtype == torch.float64:
                vector = 1
            rows = 4 if args.width >= 32 else (16 if dtype == torch.float32 and B * D * L >= 8_000_000 else (32 if B * D * L >= 32_000_000 else 8))
        configs = [f"{vector}:{rows}"]
    else:
        configs = args.schedules.split(",")
    for config in configs:
        vector, rows = map(int, config.split(":"))
        if nwh and D % vector:
            continue
        available_packed = nwh and vector > 1 and torch.cuda.get_device_capability()[0] >= 10
        for packed in ((available_packed,) if fixed else ((False, True) if available_packed else (False,))):
            name = f"v{vector}_r{rows}_p{int(packed)}"
            y = torch.empty_like(x)
            fn = cute.compile(
                module.launch_fwd,
                *(from_dlpack(t, assumed_align=16) for t in (x, w, b, y)),
                cuda.CUstream(torch.cuda.current_stream().cuda_stream),
                args.width,
                not args.identity,
                nwh,
                rows,
                vector,
                packed,
                options="--enable-tvm-ffi",
            )

            def run(fn=fn, y=y):
                fn(x, w, b, y, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

            run()
            torch.testing.assert_close(y, expected, atol=tol[0], rtol=tol[1])
            retained.append((fn, y))
            functions[name] = run
            print(f"{name}: correct", flush=True)
    results = measure(functions, cold=args.cold)
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "gpu": torch.cuda.get_device_name(),
        "cc": torch.cuda.get_device_capability(),
        "dsl": cutlass.__version__,
        "cache_policy": "512 MiB eviction before every sample" if args.cold else "warm repeated inputs; no eviction",
        "repeats": 1 if args.cold else 100,
        "rounds": 21,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(module.__file__), Path(__file__)]},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: round(v["median_us"], 3) for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
