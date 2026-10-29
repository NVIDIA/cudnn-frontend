# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A-side per-K input scaling, fused into the SM100 GEMM mainloop.

Graph: A[1,M,K] * scale[1,1,K] -> BF16 intermediate -> GEMM(B[1,K,N]).
The FP32 multiply rounds to BF16 before MMA, just like a materialized graph.
The scale is loaded using logical K coordinates after undoing the SMEM swizzle;
loads are predicated on K, and the transformed padded K lanes are zeroed.

Initial scope: single GEMM, batch=1, M>1, BF16 A/B/scale, K-major A,
contiguous scale, K divisible by 8, fixed M/N/K, SM100-family mainloop template.
Not arbitrary broadcast, B-side scaling, MoE, FP8 or swap-AB.

Run on Blackwell: python samples/frost/gemm/07_matmul_per_k_input_scale.py
The timing baseline uses a separate torch.mul (with a preallocated BF16 output)
and a plain FROST GEMM. Both paths include their launches, exclude compilation,
and use caller-owned output/workspace. It is an experiment, not a speedup claim.
"""

import argparse
import cudnn
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name


def _plan(M, N, K, *, scaled):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    b = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    s = g.tensor(name="scale", dim=[1, 1, K], stride=[K, K, 1]) if scaled else None
    operand = g.mul(a=a, b=s).set_data_type(cudnn.data_type.BFLOAT16) if scaled else a
    y = g.matmul(A=operand, B=b).set_output(True)
    cfg = by_name("CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma")
    return jit_from_cudnn_graph(g, cfg), (a, b, s, y)


def _time_ms(fn, repeats):
    for _ in range(10):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main(M=512, N=512, K=264, repeats=100):
    if not torch.cuda.is_available() or not 100 <= sum(v * w for v, w in zip(torch.cuda.get_device_capability(), (10, 1))) < 120:
        raise SystemExit("This sample requires a Blackwell-family GPU (100 <= SM < 120).")
    fused, (a, b, s, y) = _plan(M, N, K, scaled=True)
    plain, (pa, pb, _, py) = _plan(M, N, K, scaled=False)
    torch.manual_seed(0)
    av = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    bv = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)
    sv = ((torch.arange(K, device="cuda") % 17 - 8).float() / 7).reshape(1, 1, K).to(torch.bfloat16)
    out, baseline, materialized = (
        torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16),
        torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16),
        torch.empty_like(av),
    )
    fused_pack, plain_pack = {a: av, b: bv, s: sv, y: out}, {pa: materialized, pb: bv, py: baseline}

    def fused_run():
        fused(fused_pack)

    def separate_run():
        torch.mul(av, sv, out=materialized)
        plain(plain_pack)

    fused_run()
    separate_run()
    ref = ((av.float() * sv.float()).to(torch.bfloat16).float() @ bv.float().transpose(-1, -2)).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=0.25, rtol=0.02)
    torch.testing.assert_close(baseline, ref, atol=0.25, rtol=0.02)
    assert torch.isfinite(out).all()
    print(f"PASS M={M} N={N} K={K}; fused={_time_ms(fused_run, repeats):.4f} ms, separate={_time_ms(separate_run, repeats):.4f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=264)
    p.add_argument("--repeats", type=int, default=100)
    args = p.parse_args()
    main(args.m, args.n, args.k, args.repeats)
