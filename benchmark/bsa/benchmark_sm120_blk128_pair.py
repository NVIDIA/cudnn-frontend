# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interleave a saved SM120 blk128 kernel source and the installed checkout.

Usage and baseline preparation: test/python/fe_api/bsa/test_sm120_blk128_pair_benchmark.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import torch

from cudnn import BSA
from cudnn.block_sparse_attention import _interface
from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128 import bsa_fwd_sm120_fa4
from benchmark_sm120_blk128 import _make_block_indices, _rounded_topk, _validate_samples


def _parse_args(argv=None):
    """Validate paired-run options and prevent JSON from replacing source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=142720)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--densities", type=float, nargs="+", default=[0.15, 0.20])
    parser.add_argument("--patterns", choices=["strided", "local"], nargs="+", default=["strided", "local"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--fail-below-target", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.sequence % 128 or args.heads <= 0:
        parser.error("sequence must be a positive multiple of 128, and heads must be positive")
    if args.warmup < 0 or args.repeats < 3 or not math.isfinite(args.min_speedup) or args.min_speedup <= 0:
        parser.error("warmup must be nonnegative, repeats >= 3, and min-speedup positive")
    if any(not 0 < density <= 1 for density in args.densities):
        parser.error("densities must be in (0, 1]")
    if not args.baseline_source.is_file():
        parser.error("baseline-source must be an existing saved kernel source file")
    sources = {args.baseline_source.resolve(), Path(bsa_fwd_sm120_fa4.__file__).resolve()}
    if args.json and args.json.resolve() in sources:
        parser.error("JSON output must not overwrite a kernel source")
    return args


def _summarize(baseline_ms, candidate_ms):
    """Validate paired timings and report median-based speedup and latency."""
    if not baseline_ms or len(baseline_ms) != len(candidate_ms):
        raise ValueError("timings must contain the same positive number of paired samples")
    if any(not math.isfinite(value) or value <= 0 for value in (*baseline_ms, *candidate_ms)):
        raise ValueError("timings must be finite and positive")
    baseline = statistics.median(baseline_ms)
    candidate = statistics.median(candidate_ms)
    return {
        "baseline_median_ms": baseline,
        "candidate_median_ms": candidate,
        "speedup": baseline / candidate,
        "latency_reduction_pct": 100 * (1 - candidate / baseline),
        "baseline_samples_ms": baseline_ms,
        "candidate_samples_ms": candidate_ms,
    }


@torch.no_grad()
def main(argv=None):
    """Interleave saved and current kernels, restoring dispatch and cache state."""
    args = _parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 GPU")
    spec = importlib.util.spec_from_file_location("_sm120_blk128_baseline", args.baseline_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline_cls = module.BlockSparseAttnForwardSm120Blk128Fa4
    candidate_cls = bsa_fwd_sm120_fa4.BlockSparseAttnForwardSm120Blk128Fa4
    original_cache = dict(_interface.bsa_attn_fwd.compile_cache)
    torch.manual_seed(args.seed)
    shape = (1, args.heads, args.sequence, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    records = []

    def call(q2k, topk):
        """Invoke the public blk128 wrapper on this run's shared Q/K/V tensors."""
        return BSA.block_sparse_attention_forward(q, k, v, q2k, block_sparse_num=topk, sparse_block_size=128)

    try:
        for density in args.densities:
            topk = _rounded_topk(args.sequence // 128, density)
            for pattern in args.patterns:
                q2k = _make_block_indices(args.heads, args.sequence // 128, topk, pattern, q.device)
                variants = {}
                for name, kernel_cls in (("baseline", baseline_cls), ("candidate", candidate_cls)):
                    bsa_fwd_sm120_fa4.BlockSparseAttnForwardSm120Blk128Fa4 = kernel_cls
                    _interface.bsa_attn_fwd.compile_cache.clear()
                    result = call(q2k, topk)
                    torch.cuda.synchronize()
                    if len(_interface.bsa_attn_fwd.compile_cache) != 1:
                        raise RuntimeError("Expected exactly one compiled SM120 forward kernel")
                    variants[name] = next(iter(_interface.bsa_attn_fwd.compile_cache.items()))
                    _validate_samples(q, k, v, result["o_tensor"], result["lse_tensor"], q2k, 128)

                def run(name):
                    """Select a precompiled variant without compiling inside timing."""
                    key, compiled = variants[name]
                    _interface.bsa_attn_fwd.compile_cache.clear()
                    _interface.bsa_attn_fwd.compile_cache[key] = compiled
                    return call(q2k, topk)

                for _ in range(args.warmup):
                    run("baseline")
                    run("candidate")
                torch.cuda.synchronize()
                events = {
                    name: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)] for name in variants
                }
                for iteration in range(args.repeats):
                    order = ("baseline", "candidate") if iteration % 2 == 0 else ("candidate", "baseline")
                    for name in order:
                        start, end = events[name][iteration]
                        start.record()
                        run(name)
                        end.record()
                torch.cuda.synchronize()
                timings = {name: [start.elapsed_time(end) for start, end in pairs] for name, pairs in events.items()}
                record = {"density": topk / (args.sequence // 128), "topk": topk, "pattern": pattern, **_summarize(timings["baseline"], timings["candidate"])}
                for name in variants:
                    result = run(name)
                    out_error, lse_error = _validate_samples(q, k, v, result["o_tensor"], result["lse_tensor"], q2k, 128)
                    record[f"{name}_sample_max_o_error"] = out_error
                    record[f"{name}_sample_max_lse_error"] = lse_error
                record["meets_target"] = record["speedup"] >= args.min_speedup
                records.append(record)
                print(
                    f"density={record['density']:.6f} pattern={pattern} baseline_ms={record['baseline_median_ms']:.6f} candidate_ms={record['candidate_median_ms']:.6f} speedup={record['speedup']:.6f}x target={'PASS' if record['meets_target'] else 'FAIL'}",
                    flush=True,
                )
    finally:
        bsa_fwd_sm120_fa4.BlockSparseAttnForwardSm120Blk128Fa4 = candidate_cls
        _interface.bsa_attn_fwd.compile_cache.clear()
        _interface.bsa_attn_fwd.compile_cache.update(original_cache)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "gpu_model": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "baseline_source_sha256": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
            "candidate_source_sha256": hashlib.sha256(Path(bsa_fwd_sm120_fa4.__file__).read_bytes()).hexdigest(),
            "shape": shape,
            "dtype": "bf16",
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "min_speedup": args.min_speedup,
            "records": records,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if args.fail_below_target and any(not record["meets_target"] for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
