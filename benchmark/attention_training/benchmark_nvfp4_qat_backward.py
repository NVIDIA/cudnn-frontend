# SPDX-License-Identifier: Apache-2.0

"""Benchmark the explicit Triton NVFP4 attention QAT backward API."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from cudnn import Nvfp4AttentionQatBackward


def _measure_ms(operation, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--kv-length", type=int, default=None)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if torch.cuda.get_device_capability() not in {(10, 0), (10, 3), (12, 0), (12, 1)}:
        raise RuntimeError("This benchmark requires an SM100, SM103, SM120, or SM121 GPU")

    seqlen_kv = args.query_length if args.kv_length is None else args.kv_length
    if args.causal and args.query_length != seqlen_kv:
        parser.error("--causal requires equal query and KV lengths")

    q_shape = (args.batch_size, args.heads, args.query_length, 128)
    kv_shape = (args.batch_size, args.heads, seqlen_kv, 128)
    q = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(q_shape, device="cuda", dtype=torch.bfloat16)
    high_precision_o = torch.zeros_like(q)
    lse = torch.full(
        q_shape[:-1],
        math.log(seqlen_kv) + 4.0,
        device="cuda",
        dtype=torch.float32,
    )
    if args.causal:
        lse = torch.arange(1, args.query_length + 1, device="cuda", dtype=torch.float32).log() + 4.0
        lse = lse.view(1, 1, -1).expand(args.batch_size, args.heads, -1).contiguous()

    scale = 1.0 / math.sqrt(128)
    op = Nvfp4AttentionQatBackward(q, k, v, high_precision_o, do, lse, is_causal=args.causal, softmax_scale=scale)
    op.check_support()
    op.compile()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    workspace = torch.empty(op.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)

    elapsed_ms = _measure_ms(
        lambda: op.execute(q, k, v, high_precision_o, do, lse, dq, dk, dv, workspace),
        args.warmup,
        args.iterations,
    )
    base_flops = args.batch_size * args.heads * args.query_length * seqlen_kv * 128
    qat_backward_tflops = 14 * base_flops / (elapsed_ms * 1.0e9)
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"q={q_shape}, k/v={kv_shape}, causal={args.causal}")
    print(f"median: {elapsed_ms:.3f} ms; QAT backward: {qat_backward_tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
