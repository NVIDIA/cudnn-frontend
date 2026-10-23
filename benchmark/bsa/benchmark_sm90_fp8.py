# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time the Hopper Sage FP8 block-sparse attention kernels (blk64 and blk128).

Run with this checkout imported, for example:
    PYTHONPATH=python python benchmark/bsa/benchmark_sm90_fp8.py --output /tmp/bsa-sm90-fp8.json

Q/K/V are quantized once outside the timed region. Each block size gets its own random
sparse graph with the same number of active KV tokens per query block, so the effective
TFLOPS (4 * B * H * Sq * active_tokens * D, sparse QK and PV only) are comparable. Times
are CUDA-event measurements of warmed CUDA graph replay of the attention kernel alone:
they exclude quantization, JIT compilation, host allocation and Python launch overhead.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import cutlass
import cudnn
import torch

from cudnn.block_sparse_attention._fp8_quant import _quantize_sage_bhsd
from cudnn.block_sparse_attention._interface import _bsa_attn_fwd_sm90_fp8_blk64, _bsa_attn_fwd_sm90_fp8_blk128

HEAD_DIM = 128
# (batch, heads, seqlen_q, seqlen_k, active KV tokens per query block)
QUICK_SHAPES = [(1, 4, 256, 8192, 2048), (1, 8, 2048, 8192, 2048), (1, 8, 8192, 8192, 2048)]
SHAPES = QUICK_SHAPES + [(2, 8, 8192, 8192, 384), (1, 4, 128, 32768, 8192), (1, 8, 32768, 32768, 2048)]
KERNELS = {64: _bsa_attn_fwd_sm90_fp8_blk64, 128: _bsa_attn_fwd_sm90_fp8_blk128}


def measure(fn, repeats, iterations):
    """Median/min/max microseconds per call of one captured CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn()
        for _ in range(5):
            graph.replay()
        samples = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / iterations)
    torch.cuda.current_stream().wait_stream(stream)
    # Keep graph outputs alive until the measurement has completed.
    del output
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="Run only the three smallest shapes")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=529)
    args = parser.parse_args()
    if args.repeats < 1 or args.iterations < 1:
        parser.error("repeats and iterations must be positive")
    if torch.cuda.get_device_capability()[0] != 9:
        parser.error("this benchmark requires Hopper SM90")
    root = Path(__file__).resolve().parents[2]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "cuda": torch.version.cuda,
        "cutlass_dsl": cutlass.__version__,
        "cudnn_import": cudnn.__file__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "timing": "FP8 attention kernel only, warmed CUDA graph replay",
        "repeats": args.repeats,
        "iterations": args.iterations,
        "seed": args.seed,
        "results": [],
    }
    for batch, heads, sq, sk, active_tokens in QUICK_SHAPES if args.quick else SHAPES:
        torch.manual_seed(args.seed)
        q = torch.randn(batch, heads, sq, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5
        k = torch.randn(batch, heads, sk, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.5
        v = torch.randn_like(k) * 0.5
        quantized = _quantize_sage_bhsd(q, k, v, v_layout="bhds")
        row = {"shape_B_H_Sq_Sk_D": [batch, heads, sq, sk, HEAD_DIM], "active_tokens": active_tokens}
        for block, kernel in KERNELS.items():
            count = active_tokens // block
            ids = torch.rand(batch, heads, sq // block, sk // block, device="cuda").argsort(-1)[..., :count].int().contiguous()
            timing = measure(lambda: kernel(*quantized, ids, count), args.repeats, args.iterations)
            flops = 4 * batch * heads * sq * count * block * HEAD_DIM
            row[f"blk{block}"] = {**timing, "tflops": flops / (timing["median_us"] * 1e6)}
        row["blk128_speedup_vs_blk64"] = row["blk64"]["median_us"] / row["blk128"]["median_us"]
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
