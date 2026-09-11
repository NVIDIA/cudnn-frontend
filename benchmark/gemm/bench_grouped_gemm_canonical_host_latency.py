# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-latency comparison for the grouped GEMM SwiGLU wrapper: legacy pre-permuted
call (including the TransformerEngine-style per-call view/permute gymnastics the
legacy contract forces on the caller) vs the canonical natural-layout call.

DSv3 fc1 shape from MLPerf MoE: (sum_m, n, k) = (24576, 7168, 2048), MXFP8 inputs,
first dim overallocated 1.5x-4x for EP routing slack.

Usage: python bench_grouped_gemm_canonical_host_latency.py
"""

import time

import torch

from cudnn import grouped_gemm_swiglu_wrapper_sm100
from cudnn.api_base import ceil_div

VALID_M, N, K = 24576, 7168, 2048
EXPERTS = 8
SF_VEC = 32
WARMUP, ITERS = 20, 200


def make_buffers(tensor_m):
    dev = "cuda"
    rest_k = ceil_div(ceil_div(K, SF_VEC), 4)
    # Natural (canonical) buffers, as a framework owns them.
    a = torch.randint(0, 200, (tensor_m, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    b = torch.randint(0, 200, (EXPERTS, N, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    sfa = torch.randint(118, 132, (1, ceil_div(tensor_m, 128), rest_k, 32, 4, 4), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    sfb = torch.randint(118, 132, (EXPERTS, ceil_div(N, 128), rest_k, 32, 4, 4), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    group = VALID_M // EXPERTS
    offsets = torch.arange(group, VALID_M + 1, group, dtype=torch.int32, device=dev)
    alpha = torch.ones(EXPERTS, dtype=torch.float32, device=dev)
    prob = torch.rand(tensor_m, dtype=torch.float32, device=dev)
    norm_const = torch.tensor([0.01], dtype=torch.float32, device=dev)
    return dict(a=a, b=b, sfa=sfa, sfb=sfb, offsets=offsets, alpha=alpha, prob=prob, norm_const=norm_const)


def call_legacy(buf):
    # TE-style per-call layout gymnastics required by the legacy contract
    # (see transformer_engine grouped_mlp.py: 6-D SF view+permute, B permute).
    m = buf["a"].shape[0]
    a3d = buf["a"].view(m, K, 1)
    b_nkl = buf["b"].permute(1, 2, 0)
    sfa6d = buf["sfa"].view(torch.float8_e8m0fnu).view(1, ceil_div(m, 128), ceil_div(ceil_div(K, SF_VEC), 4), 32, 4, 4).permute(3, 4, 1, 5, 2, 0)
    sfb6d = buf["sfb"].view(torch.float8_e8m0fnu).view(EXPERTS, ceil_div(N, 128), ceil_div(ceil_div(K, SF_VEC), 4), 32, 4, 4).permute(3, 4, 1, 5, 2, 0)
    prob3d = buf["prob"].view(m, 1, 1)
    return grouped_gemm_swiglu_wrapper_sm100(
        a_tensor=a3d,
        b_tensor=b_nkl,
        sfa_tensor=sfa6d,
        sfb_tensor=sfb6d,
        padded_offsets=buf["offsets"],
        alpha_tensor=buf["alpha"],
        norm_const_tensor=buf["norm_const"],
        prob_tensor=prob3d,
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC,
    )


def call_canonical(buf):
    return grouped_gemm_swiglu_wrapper_sm100(
        a_tensor=buf["a"],
        b_tensor=buf["b"],
        sfa_tensor=buf["sfa"],
        sfb_tensor=buf["sfb"],
        padded_offsets=buf["offsets"],
        alpha_tensor=None,
        norm_const_tensor=buf["norm_const"],
        prob_tensor=buf["prob"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC,
    )


def bench(fn, buf):
    for _ in range(WARMUP):
        fn(buf)
    torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    times.sort()
    n = len(times)
    return times[n // 2] * 1e6, times[int(n * 0.9)] * 1e6


def main():
    torch.manual_seed(0)
    print(f"grouped_gemm_swiglu host latency, (sum_m, n, k)=({VALID_M}, {N}, {K}), {EXPERTS} experts, MXFP8")
    print(f"{'overalloc':>9} | {'tensor_m':>8} | {'legacy p50/p90 (us)':>22} | {'canonical p50/p90 (us)':>22}")
    for factor in (1.5, 2.0, 4.0):
        tensor_m = ceil_div(int(VALID_M * factor), 256) * 256
        buf = make_buffers(tensor_m)
        legacy_p50, legacy_p90 = bench(call_legacy, buf)
        canon_p50, canon_p90 = bench(call_canonical, buf)
        print(f"{factor:>8}x | {tensor_m:>8} | {legacy_p50:>10.1f} / {legacy_p90:>7.1f} | {canon_p50:>11.1f} / {canon_p90:>7.1f}")


if __name__ == "__main__":
    main()
