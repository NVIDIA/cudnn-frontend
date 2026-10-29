# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example 07: parameterized LeakyReLU on GEMM operands (pure frontend API).

Run on a data-center Blackwell GPU:
  CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 python samples/frost/gemm/07_matmul_leaky_relu_mainloop.py --benchmark

Both paths use FROST GEMM and preserve BF16 activation outputs. Timings use
CUDA-graph replay, exclude preparation/JIT, and include the baseline's separate
PyTorch activation kernels. They are shape-specific, not a speedup guarantee.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import cudnn
import torch


def _build_plans(g: cudnn.pygraph) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    candidates = [i for i, name in enumerate(names) if name.split("[")[0] == "frost_gemm"]
    if not candidates:
        raise RuntimeError("No FROST GEMM plan; set CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 and check GPU/DSL support")
    g.select_plan(candidates[0])  # Pin FROST for the matched-provider comparison.
    g.check_support()
    g.build_plans()


def _build_graph(M: int, N: int, K: int, side: str, negative_slope: float, fuse: bool):
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    B = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    Ai, Bi = A, B
    if fuse and side in ("a", "ab"):
        Ai = g.leaky_relu(input=A, negative_slope=negative_slope, name="leakyA").set_data_type(cudnn.data_type.BFLOAT16)
    if fuse and side in ("b", "ab"):
        Bi = g.leaky_relu(input=B, negative_slope=negative_slope, name="leakyB").set_data_type(cudnn.data_type.BFLOAT16)
    C = g.matmul(A=Ai, B=Bi, name="mm").set_data_type(cudnn.data_type.FLOAT).set_output(True)
    return g, A, B, C


def _replay_us(call: Callable[[], None], iterations: int = 100) -> float:
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


def main(M: int = 512, N: int = 512, K: int = 264, side: str = "ab", negative_slope: float = 0.1, benchmark: bool = False) -> None:
    if min(M, N, K) <= 0 or not math.isfinite(negative_slope):
        raise ValueError("dimensions must be positive and negative_slope must be finite")
    if side not in ("a", "b", "ab"):
        raise ValueError("side must be 'a', 'b', or 'ab'")
    if not torch.cuda.is_available():
        print("[07] SKIP  needs a CUDA GPU")
        return
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor < 120:
        print("[07] SKIP  needs a data-center Blackwell-family GPU (100 <= SM < 120)")
        return

    # Graph description and plan construction precede all execution/timing.
    g, A, B, C = _build_graph(M, N, K, side, negative_slope, fuse=True)
    _build_plans(g)
    torch.manual_seed(0)
    a = torch.randn((1, M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((1, N, K), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    c = torch.empty((1, M, N), device="cuda", dtype=torch.float32)
    workspace = torch.empty(max(g.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

    def fused() -> None:
        g.execute({A: a, B: b, C: c}, workspace)

    a_ref = torch.nn.functional.leaky_relu(a.float(), negative_slope).to(a.dtype) if side in ("a", "ab") else a
    b_ref = torch.nn.functional.leaky_relu(b.float(), negative_slope).to(b.dtype) if side in ("b", "ab") else b
    ref = a_ref.float() @ b_ref.float()
    fused()
    torch.cuda.synchronize()
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    print(f"[07] PASS  leaky_relu operands  side={side} slope={negative_slope} M={M} N={N} K={K}")

    if benchmark:
        base, A_base, B_base, C_base = _build_graph(M, N, K, side, negative_slope, fuse=False)
        _build_plans(base)
        c_base = torch.empty_like(c)
        base_workspace = torch.empty(max(base.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)

        def separate() -> None:
            a_activated = torch.nn.functional.leaky_relu(a, negative_slope) if side in ("a", "ab") else a
            b_activated = torch.nn.functional.leaky_relu(b, negative_slope) if side in ("b", "ab") else b
            base.execute({A_base: a_activated, B_base: b_activated, C_base: c_base}, base_workspace)

        separate()
        torch.cuda.synchronize()
        torch.testing.assert_close(c_base, ref, atol=1e-2, rtol=1e-2)
        fused_us, separate_us = _replay_us(fused), _replay_us(separate)
        torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(c_base, ref, atol=1e-2, rtol=1e-2)
        print(f"[07] CUDA-graph replay: fused={fused_us:.2f} us, separate={separate_us:.2f} us, ratio={separate_us / fused_us:.3f}x")


if __name__ == "__main__":
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
    main(args.m, args.n, args.k, args.side, args.slope, args.benchmark)
