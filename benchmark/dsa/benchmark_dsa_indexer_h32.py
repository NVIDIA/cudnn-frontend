#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the compact-H32 SM100 indexer backward interface.

The default benchmark compares two end-to-end paths:

* ``compact_h32``: compact H32 tensors passed directly to the default backend;
* ``eager_pad_h64``: caller-side ``torch.nn.functional.pad`` to H64 before the
  established kernel.

Both paths reuse output storage. The H64 baseline deliberately omits output
compaction, favoring it when a consumer requires compact H32 gradients.
Measurements rotate case order and flush L2 before every sample. Profile mode
also exposes prebuilt H64 inputs as a kernel-only sanity control.

Use ``profile --profile-case ...`` to emit exactly one warmed invocation
between ``cudaProfilerStart``/``cudaProfilerStop`` for profilers that honor
the CUDA profiler range.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from cudnn import DSA


def _make_case(batch: int, seqlen_q: int, seqlen_k: int, topk: int, seed: int):
    """Allocate deterministic compact-H32 inputs and sparse-score operands."""
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q32 = torch.randn((batch, seqlen_q, 32, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    w32 = torch.randn((batch, seqlen_q, 32), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn((batch, seqlen_k, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    attn = torch.randn((batch, seqlen_q, topk), device="cuda", dtype=torch.float32, generator=generator)
    predict = torch.softmax(torch.randn(attn.shape, device="cuda", dtype=torch.float32, generator=generator), dim=-1)
    indices = torch.randint(0, seqlen_k, attn.shape, device="cuda", dtype=torch.int32, generator=generator)

    return q32, w32, k, attn, predict, indices


def _outputs(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor):
    """Allocate caller-owned dQ/dW buffers and an FP32 dK accumulator."""
    return (
        torch.empty_like(q),
        torch.empty_like(w),
        torch.empty_like(k, dtype=torch.float32),
    )


def _call(q, w, k, attn, predict, indices, grad_loss, outputs):
    """Launch the default indexer-backward backend into supplied outputs."""
    d_index_q, d_weights, d_index_k = outputs
    return DSA.indexer_backward_wrapper(
        q,
        w,
        k,
        attn,
        predict,
        indices,
        grad_loss,
        sm_scale=1.0,
        loss_coeff=float(q.shape[0] * q.shape[1]),
        block_I=128,
        topk_indices_global=False,
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k,
        backend="default",
    )


def _time_once(label, fn, attn, attn_base, l2_flush):
    """Restore destructive input state, flush L2, and time one call in microseconds."""
    l2_flush.zero_()
    attn.copy_(attn_base)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.cuda.nvtx.range(label):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def main():
    """Run the two-path benchmark or emit one profiler-delimited launch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="benchmark", choices=("benchmark", "profile"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=512)
    parser.add_argument("--seqlen-k", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile-case",
        choices=("compact_h32", "pre_padded_h64", "eager_pad_h64"),
        default="compact_h32",
    )
    args = parser.parse_args()

    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("benchmark requires an SM100-family GPU (compute capability 10.x)")
    if args.topk <= 0 or args.topk % 128 != 0:
        raise ValueError("topk must be a positive multiple of 128")

    q32, w32, k, attn_base, predict, indices = _make_case(args.batch, args.seqlen_q, args.seqlen_k, args.topk, args.seed)
    grad_loss = torch.ones((), device="cuda", dtype=torch.float32)
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    attn32 = attn_base.clone()
    attn64_eager = attn_base.clone()
    out32 = _outputs(q32, w32, k)
    q64_shape = (args.batch, args.seqlen_q, 64, 128)
    w64_shape = (args.batch, args.seqlen_q, 64)
    out64_eager = (
        torch.empty(q64_shape, device=q32.device, dtype=q32.dtype),
        torch.empty(w64_shape, device=w32.device, dtype=w32.dtype),
        torch.empty_like(k, dtype=torch.float32),
    )

    compact_h32 = lambda: _call(q32, w32, k, attn32, predict, indices, grad_loss, out32)

    def eager_pad_h64():
        """Materialize H64 inputs on every call before launching the baseline."""
        q64_runtime = torch.nn.functional.pad(q32, (0, 0, 0, 32))
        w64_runtime = torch.nn.functional.pad(w32, (0, 32))
        return _call(q64_runtime, w64_runtime, k, attn64_eager, predict, indices, grad_loss, out64_eager)

    cases = (
        ("compact_h32_us", "dsa_indexer_bwd_compact_h32", compact_h32, attn32),
        ("eager_pad_h64_us", "dsa_indexer_bwd_eager_pad_h64", eager_pad_h64, attn64_eager),
    )
    if args.mode == "profile":
        if args.profile_case == "pre_padded_h64":
            q64 = torch.zeros(q64_shape, device=q32.device, dtype=q32.dtype)
            w64 = torch.zeros(w64_shape, device=w32.device, dtype=w32.dtype)
            q64[:, :, :32].copy_(q32)
            w64[:, :, :32].copy_(w32)
            attn64_pre = attn_base.clone()
            out64_pre = _outputs(q64, w64, k)
            profile_case = (
                "pre_padded_h64_us",
                "dsa_indexer_bwd_pre_padded_h64",
                lambda: _call(q64, w64, k, attn64_pre, predict, indices, grad_loss, out64_pre),
                attn64_pre,
            )
        else:
            profile_case = next(case for case in cases if case[0] == f"{args.profile_case}_us")
        name, label, fn, attn = profile_case
        # Compile and warm only the selected case. Besides keeping profiler
        # output focused, this lets Compute Sanitizer attribute a diagnostic
        # to H32 or to the established H64 control without a poisoned context
        # from another case.
        fn()
        torch.cuda.synchronize()
        attn.copy_(attn_base)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range(label):
            fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print(
            json.dumps(
                {
                    "profile_case": name.removesuffix("_us"),
                    "shape": {
                        "batch": args.batch,
                        "seqlen_q": args.seqlen_q,
                        "seqlen_k": args.seqlen_k,
                        "heads": 32,
                        "head_dim": 128,
                        "topk": args.topk,
                    },
                },
                indent=2,
            )
        )
        return

    # Compile every shape and verify the compact interface against the
    # established H64 formulation before collecting timings.
    compact_h32()
    eager_pad_h64()
    torch.cuda.synchronize()
    torch.testing.assert_close(out32[0], out64_eager[0][:, :, :32], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(out32[1], out64_eager[1][:, :, :32], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(out32[2], out64_eager[2], rtol=3e-2, atol=3e-2)

    for _ in range(args.warmup):
        for _, _, fn, attn in cases:
            attn.copy_(attn_base)
            fn()
    torch.cuda.synchronize()

    timings = {name: [] for name, _, _, _ in cases}
    for repeat in range(args.repeats):
        shift = repeat % len(cases)
        for name, label, fn, attn in cases[shift:] + cases[:shift]:
            timings[name].append(_time_once(label, fn, attn, attn_base, l2_flush))

    medians = {name: statistics.median(values) for name, values in timings.items()}
    compact = medians["compact_h32_us"]
    baseline = medians["eager_pad_h64_us"]
    # Three GEMMs each contribute 2*B*Sq*H*D*TopK FLOPs. Report logical H32
    # work for every path so adapter comparisons remain apples-to-apples; the
    # H64 controls execute twice that many tensor-core operations internally.
    logical_gemm_flops = 6 * args.batch * args.seqlen_q * 32 * 128 * args.topk
    result = {
        "shape": {
            "batch": args.batch,
            "seqlen_q": args.seqlen_q,
            "seqlen_k": args.seqlen_k,
            "heads": 32,
            "head_dim": 128,
            "topk": args.topk,
        },
        "samples": timings,
        "median_us": medians,
        "logical_gemm_tflops": {name.removesuffix("_us"): logical_gemm_flops / value / 1.0e6 for name, value in medians.items()},
        "speedup_percent": (baseline / compact - 1.0) * 100.0,
        "latency_reduction_percent": (1.0 - compact / baseline) * 100.0,
        "notes": [
            "eager_pad_h64 allocates H64 inputs through torch.nn.functional.pad on every call",
            "H64 output compaction is excluded from the baseline",
            "TFLOP/s uses logical H32 work: 6 * B * Sq * H * D * TopK",
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
