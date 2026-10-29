# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeakyReLU on GEMM operands, with an optional two-stage baseline.

Run on a data-center Blackwell GPU:
  CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 python samples/frost/gemm/07_matmul_leaky_relu_mainloop.py --benchmark

Both paths use FROST GEMM and preserve BF16 activation outputs. Timings use
CUDA-graph replay, exclude preparation/JIT, and include the baseline's separate
PyTorch activation kernels. They are shape-specific, not a speedup guarantee.
"""

from __future__ import annotations

import argparse
import math

import cudnn
import torch


def _build_graph(m, n, k, side, slope, fuse):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="a", dim=[1, m, k], stride=[m * k, k, 1])
    b = g.tensor(name="b", dim=[1, k, n], stride=[k * n, 1, k])
    ai = g.leaky_relu(input=a, negative_slope=slope).set_data_type(cudnn.data_type.BFLOAT16) if fuse and "a" in side else a
    bi = g.leaky_relu(input=b, negative_slope=slope).set_data_type(cudnn.data_type.BFLOAT16) if fuse and "b" in side else b
    y = g.matmul(A=ai, B=bi).set_data_type(cudnn.data_type.FLOAT).set_output(True)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    candidates = [i for i, name in enumerate(names) if name.split("[")[0] == "frost_gemm"]
    if not candidates:
        raise RuntimeError("No FROST GEMM plan; set CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 and check GPU/DSL support")
    g.select_plan(candidates[0])
    g.check_support()
    g.build_plans()
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    return g, a, b, y, workspace


def _replay_us(call, iterations=100):
    # Warm and capture on a side stream, retaining the capture pool until timing ends.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured, stream=stream):
        call()
    for _ in range(5):
        captured.replay()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        captured.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=264)
    parser.add_argument("--side", choices=("a", "b", "ab"), default="ab")
    parser.add_argument("--slope", type=float, default=0.1)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0 or not math.isfinite(args.slope):
        parser.error("dimensions must be positive and slope must be finite")
    if not torch.cuda.is_available():
        parser.error("this sample needs a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor < 120:
        parser.error("this sample needs a data-center Blackwell-family GPU (100 <= SM < 120)")

    torch.manual_seed(0)
    av = torch.randn((1, args.m, args.k), device="cuda", dtype=torch.bfloat16)
    bv = torch.randn((1, args.n, args.k), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    fused_out = torch.empty((1, args.m, args.n), device="cuda", dtype=torch.float32)
    g, a, b, y, workspace = _build_graph(args.m, args.n, args.k, args.side, args.slope, True)

    def fused():
        g.execute({a: av, b: bv, y: fused_out}, workspace)

    ar = torch.nn.functional.leaky_relu(av.float(), args.slope).to(av.dtype) if "a" in args.side else av
    br = torch.nn.functional.leaky_relu(bv.float(), args.slope).to(bv.dtype) if "b" in args.side else bv
    ref = ar.float() @ br.float()
    fused()
    torch.testing.assert_close(fused_out, ref, atol=1e-2, rtol=1e-2)
    print(f"PASS: side={args.side}, slope={args.slope}, M/N/K={args.m}/{args.n}/{args.k}")

    if args.benchmark:
        base, ba, bb, by, base_workspace = _build_graph(args.m, args.n, args.k, args.side, args.slope, False)
        base_out = torch.empty_like(fused_out)

        def separate():
            aa = torch.nn.functional.leaky_relu(av, args.slope) if "a" in args.side else av
            ab = torch.nn.functional.leaky_relu(bv, args.slope) if "b" in args.side else bv
            base.execute({ba: aa, bb: ab, by: base_out}, base_workspace)

        separate()
        torch.testing.assert_close(base_out, ref, atol=1e-2, rtol=1e-2)
        fused_us, separate_us = _replay_us(fused), _replay_us(separate)
        torch.testing.assert_close(fused_out, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(base_out, ref, atol=1e-2, rtol=1e-2)
        print(f"CUDA-graph replay: fused={fused_us:.2f} us, separate={separate_us:.2f} us, ratio={separate_us / fused_us:.3f}x")


if __name__ == "__main__":
    main()
