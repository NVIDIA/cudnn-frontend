# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-K input scaling on A, B, or both, fused into the SM100 GEMM mainloop.

Graph: (A[1,M,K] * scale_a[1,1,K]) @ (B[1,K,N] * scale_b[1,K,1]).
Each FP32 multiply rounds to the operand dtype before MMA, just like a
materialized graph. Scale loads use logical K after undoing the SMEM swizzle;
loads are bounded by K, and the transformed padded K lanes are zeroed.

The template supports K-major BF16/FP16 scaled operands, contiguous per-K
BF16/FP16/FP32 scales, broadcast across the batch, dynamic M/N/K, and swap-AB.
K must satisfy the existing TMA alignment (divisible by 8 for 16-bit inputs).
Scaled A needs M>1; scaled B needs N>1 to distinguish the scale from its input.
MoE, block scaling and parallel multi-GEMM are not mainloop template routes.

Run on Blackwell: python samples/frost/gemm/07_matmul_per_k_input_scale.py --side ab
The baseline uses separate torch.mul calls (preallocated outputs) and a plain
FROST GEMM. Both paths include launches and exclude compilation. These timings
are an experiment, not a speedup claim. Try --side a or --side b separately.
"""

import argparse
import cudnn
import torch

from cudnn.gemm.frost.compiler import jit_from_cudnn_graph
from cudnn.gemm.frost.tile_config import by_name


def _plan(M, N, K, side):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    a = g.tensor(name="A", dim=[1, M, K], stride=[M * K, K, 1])
    b = g.tensor(name="B", dim=[1, K, N], stride=[K * N, 1, K])
    sa = g.tensor(name="scale_a", dim=[1, 1, K], stride=[K, K, 1]) if "a" in side else None
    sb = g.tensor(name="scale_b", dim=[1, K, 1], stride=[K, 1, K]) if "b" in side else None
    aa = g.mul(a=a, b=sa).set_data_type(cudnn.data_type.BFLOAT16) if sa is not None else a
    bb = g.mul(a=b, b=sb).set_data_type(cudnn.data_type.BFLOAT16) if sb is not None else b
    y = g.matmul(A=aa, B=bb).set_output(True)
    cfg = by_name("CONFIG_sm100_128x128x64_128x128x32_cluster1x1_1ctamma")
    return jit_from_cudnn_graph(g, cfg), (a, b, sa, sb, y)


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


def main(M=512, N=512, K=264, repeats=100, side="ab"):
    if not torch.cuda.is_available():
        raise SystemExit("This sample requires a Blackwell-family GPU (100 <= SM < 120).")
    major, minor = torch.cuda.get_device_capability()
    if not 100 <= major * 10 + minor < 120:
        raise SystemExit("This sample requires a Blackwell-family GPU (100 <= SM < 120).")
    fused, (a, b, sa, sb, y) = _plan(M, N, K, side)
    plain, (pa, pb, _, _, py) = _plan(M, N, K, "")
    torch.manual_seed(0)
    av = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    bv = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)
    sav = ((torch.arange(K, device="cuda") % 17 - 8).float() / 7).reshape(1, 1, K).to(torch.bfloat16)
    sbv = torch.linspace(0.7, 1.3, K, device="cuda").reshape(1, K, 1).to(torch.bfloat16)
    out, baseline = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16), torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)
    materialized_a, materialized_b = torch.empty_like(av), torch.empty_like(bv)
    fused_pack = {a: av, b: bv, y: out}
    if sa is not None:
        fused_pack[sa] = sav
    if sb is not None:
        fused_pack[sb] = sbv
    plain_pack = {pa: materialized_a if sa is not None else av, pb: materialized_b if sb is not None else bv, py: baseline}

    def fused_run():
        fused(fused_pack)

    def separate_run():
        if sa is not None:
            torch.mul(av, sav, out=materialized_a)
        if sb is not None:
            torch.mul(bv, sbv.transpose(1, 2), out=materialized_b)
        plain(plain_pack)

    fused_run()
    separate_run()
    aa = (av.float() * sav.float()).to(torch.bfloat16) if sa is not None else av
    bb = (bv.float() * sbv.transpose(1, 2).float()).to(torch.bfloat16) if sb is not None else bv
    ref = (aa.float() @ bb.float().transpose(-1, -2)).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=0.25, rtol=0.02)
    torch.testing.assert_close(baseline, ref, atol=0.25, rtol=0.02)
    assert torch.isfinite(out).all()
    print(f"PASS side={side} M={M} N={N} K={K}; fused={_time_ms(fused_run, repeats):.4f} ms, separate={_time_ms(separate_run, repeats):.4f} ms")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=264)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--side", choices=("a", "b", "ab"), default="ab")
    args = p.parse_args()
    main(args.m, args.n, args.k, args.repeats, args.side)
