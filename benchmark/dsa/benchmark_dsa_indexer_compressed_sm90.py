#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the SM90 packed-THD compact IndexerForward + Top-K path.

Compilation, allocations, candidate-offset construction, and CUDA Graph capture
are untimed. Report median warm-L2 graph replay time for one batch segment.
"""

import argparse
import json
import statistics

import torch

from cudnn import DSA


def make_inputs(tokens: int, precision: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_bf16 = (torch.randn(tokens, 64, 128, generator=generator, device="cuda") * 0.25).bfloat16()
    k_bf16 = (torch.randn(tokens, 1, 128, generator=generator, device="cuda") * 0.25).bfloat16()
    w_bf16 = (torch.rand(tokens, 64, generator=generator, device="cuda") + 0.1).bfloat16()
    w = w_bf16.float() if precision == "bf16" else w_bf16
    q_scale = k_scale = None
    if precision == "fp8":
        q = q_bf16.to(torch.float8_e4m3fn)
        k = k_bf16.to(torch.float8_e4m3fn)
        q_scale = torch.ones((tokens, 64), dtype=torch.float32, device="cuda")
        k_scale = torch.ones((tokens, 1), dtype=torch.float32, device="cuda")
    else:
        q, k = q_bf16, k_bf16
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    return q, k, w, q_scale, k_scale, cu


def benchmark(tokens: int, args):
    q, k, w, q_scale, k_scale, cu = make_inputs(tokens, args.precision, args.seed)
    offsets, candidate_count = DSA.compress_topk_cand_buffer_size_thd(cu, cu, 4)
    buffers = {
        "cand_batch_offsets": offsets,
        "cand_buffer": torch.empty(candidate_count, dtype=torch.float32, device="cuda"),
        "out_indices": torch.empty((tokens, args.top_k), dtype=torch.int32, device="cuda"),
        "out_logits": torch.empty((tokens, args.top_k), dtype=torch.float32, device="cuda"),
    }

    def call():
        return DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=args.top_k,
            ratio=4,
            qhead_per_kv_head=64,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=tokens,
            max_seqlen_k=tokens,
            precision=args.precision,
            q_scale=q_scale,
            k_scale=k_scale,
            sm_scale=128**-0.5,
            topk_indices_global=False,
            deterministic=True,
            return_softmax=False,
            **buffers,
        )

    for _ in range(args.warmup):
        output = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = call()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(args.samples):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        median_ms = statistics.median(times)
        assert output["indices"].data_ptr() == buffers["out_indices"].data_ptr()
        return {
            "tokens": tokens,
            "precision": args.precision,
            "top_k": args.top_k,
            "candidate_count": candidate_count,
            "candidate_mib": candidate_count * 4 / 2**20,
            "median_ms": median_ms,
            "candidate_gitems_s": candidate_count / median_ms / 1e6,
            "samples_ms": [round(value, 6) for value in times],
        }
    finally:
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(tokens < 1 for tokens in args.tokens) or not 0 < args.top_k <= 2048:
        parser.error("tokens must be positive and top-k must be in [1, 2048]")

    print(
        "# "
        + json.dumps(
            {
                "arch": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                **vars(args),
            }
        ),
        flush=True,
    )
    for tokens in args.tokens:
        print(json.dumps(benchmark(tokens, args)), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
