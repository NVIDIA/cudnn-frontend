# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host latency of the block-scaled grouped GEMM wrappers TransformerEngine's fused MoE
step calls per iteration: glu fwd, quant, dglu, wgrad.

DSv3 fc1 shape from MLPerf MoE: (sum_m, n, k) = (24576, 7168, 2048), MXFP8, 8 experts,
token dim overallocated 2x for EP routing slack. Each call is timed after a
torch.cuda.synchronize() so only the wrapper's own host work is measured.

Usage: python bench_grouped_gemm_wrapper_host_latency.py
"""

import time

import torch

from cudnn import (
    grouped_gemm_dglu_wrapper_sm100,
    grouped_gemm_glu_wrapper_sm100,
    grouped_gemm_quant_wrapper_sm100,
    grouped_gemm_wgrad_wrapper_sm100,
)
from cudnn.api_base import ceil_div

VALID_M, N, K = 24576, 7168, 2048
EXPERTS = 8
SF_VEC = 32
OVERALLOC = 2.0
WARMUP, ITERS = 20, 200


def sf6d(rows, cols, batch):
    dev = "cuda"
    raw = torch.randint(118, 132, (batch, ceil_div(rows, 128), ceil_div(ceil_div(cols, SF_VEC), 4), 32, 4, 4), dtype=torch.uint8, device=dev)
    return raw.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)


def fp8(*shape):
    return torch.randint(0, 200, shape, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)


def make_buffers(tensor_m):
    dev = "cuda"
    group = VALID_M // EXPERTS
    offsets = torch.arange(group, VALID_M + 1, group, dtype=torch.int32, device=dev)
    return dict(
        a=fp8(1, tensor_m, K).permute(1, 2, 0),
        sfa=sf6d(tensor_m, K, 1),
        b=fp8(EXPERTS, N, K).permute(1, 2, 0),
        sfb=sf6d(N, K, EXPERTS),
        # dglu: grad (m, k) x weight (n/2, k, l) against the fwd activations c (m, n); wgrad: (hidden, tokens) x (tokens, inter)
        grad=fp8(1, tensor_m, K).permute(1, 2, 0),
        sfgrad=sf6d(tensor_m, K, 1),
        c=torch.randn(1, tensor_m, N, dtype=torch.bfloat16, device=dev).permute(1, 2, 0),
        b_half=fp8(EXPERTS, N // 2, K).permute(1, 2, 0),
        sfb_half=sf6d(N // 2, K, EXPERTS),
        wg_a=fp8(K, tensor_m),
        wg_b=fp8(tensor_m, N).T.contiguous().T,
        wg_sfa=torch.randint(118, 132, (K, ceil_div(tensor_m, SF_VEC)), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu),
        wg_sfb=torch.randint(118, 132, (N, ceil_div(tensor_m, SF_VEC)), dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu),
        offsets=offsets,
        alpha=torch.ones(EXPERTS, dtype=torch.float32, device=dev),
        beta=torch.ones(EXPERTS, dtype=torch.float32, device=dev),
        prob=torch.rand(tensor_m, 1, 1, dtype=torch.float32, device=dev),
        dprob=torch.zeros(tensor_m, 1, 1, dtype=torch.float32, device=dev),
        norm_const=torch.tensor([0.01], dtype=torch.float32, device=dev),
    )


def call_glu(buf):
    return grouped_gemm_glu_wrapper_sm100(
        a_tensor=buf["a"],
        sfa_tensor=buf["sfa"],
        padded_offsets=buf["offsets"],
        alpha_tensor=buf["alpha"],
        b_tensor=buf["b"],
        sfb_tensor=buf["sfb"],
        norm_const_tensor=buf["norm_const"],
        prob_tensor=buf["prob"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC,
    )


def call_quant(buf):
    return grouped_gemm_quant_wrapper_sm100(
        a_tensor=buf["a"],
        sfa_tensor=buf["sfa"],
        padded_offsets=buf["offsets"],
        alpha_tensor=buf["alpha"],
        b_tensor=buf["b"],
        sfb_tensor=buf["sfb"],
        norm_const_tensor=buf["norm_const"],
        prob_tensor=buf["prob"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC,
    )


def call_dglu(buf):
    return grouped_gemm_dglu_wrapper_sm100(
        a_tensor=buf["grad"],
        c_tensor=buf["c"],
        sfa_tensor=buf["sfgrad"],
        padded_offsets=buf["offsets"],
        alpha_tensor=buf["alpha"],
        beta_tensor=buf["beta"],
        prob_tensor=buf["prob"],
        dprob_tensor=buf["dprob"],
        b_tensor=buf["b_half"],
        sfb_tensor=buf["sfb_half"],
        norm_const_tensor=buf["norm_const"],
        d_dtype=torch.float8_e4m3fn,
        sf_vec_size=SF_VEC,
    )


def call_wgrad(buf):
    return grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=buf["wg_a"],
        b_tensor=buf["wg_b"],
        sfa_tensor=buf["wg_sfa"],
        sfb_tensor=buf["wg_sfb"],
        offsets_tensor=buf["offsets"],
        wgrad_dtype=torch.bfloat16,
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
    tensor_m = ceil_div(int(VALID_M * OVERALLOC), 256) * 256
    buf = make_buffers(tensor_m)
    print(f"grouped GEMM wrapper host latency, (sum_m, n, k)=({VALID_M}, {N}, {K}), {EXPERTS} experts, MXFP8, tensor_m={tensor_m}")
    print(f"{'wrapper':>8} | {'p50 (us)':>8} | {'p90 (us)':>8}")
    for name, fn in (("glu", call_glu), ("quant", call_quant), ("dglu", call_dglu), ("wgrad", call_wgrad)):
        p50, p90 = bench(fn, buf)
        print(f"{name:>8} | {p50:>8.1f} | {p90:>8.1f}")


if __name__ == "__main__":
    main()
