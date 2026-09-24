# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 blk64 forward TFLOP/s: QK^T + PV FLOPs / CUDA Graph forward time (including GPU metadata setup)."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from triton.testing import do_bench_cudagraph

import cudnn
from cudnn import BSA
from cudnn.block_sparse_attention.csrc.fwd.sm90_blk64 import bsa_fwd_sm90


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run the first three shapes")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--rep", type=int, default=20, help="Measurement duration in milliseconds")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    if args.rep < 1:
        parser.error("rep must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This benchmark requires an SM90 GPU")
    root = Path(__file__).resolve().parents[2]
    if Path(cudnn.__file__).resolve() != root / "python/cudnn/__init__.py":
        raise RuntimeError(f"cudnn resolves to {cudnn.__file__}; install this checkout with uv pip install -e .")

    shapes = [(1, 2, 256, 512, 6), (1, 8, 2048, 4096, 16), (1, 8, 8192, 8192, 32)]
    if not args.quick:
        shapes += [(2, 16, 16384, 16384, 64), (1, 8, 32768, 32768, 64), (1, 8, 8192, 8192, 128)]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cudnn_source=cudnn.__file__,
        kernel_source=bsa_fwd_sm90.__file__,
        dtype=args.dtype,
        timing=__doc__,
        flops_formula="4 * B * H * Sq * D * topk64 * 64",
        results=[],
    )
    print(json.dumps(report), flush=True)
    for b, h, sq, sk, topk in shapes:
        torch.manual_seed(20260917)
        q = torch.randn(b, h, sq, 128, device="cuda", dtype=dtype)
        k = torch.randn(b, h, sk, 128, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        indices = torch.rand(b, h, sq // 64, sk // 64, device="cuda").argsort(-1)[..., :topk].int().contiguous()

        def forward():
            return BSA.block_sparse_attention_forward(q, k, v, indices, topk, sparse_block_size=64)

        forward()
        torch.cuda.synchronize()
        ms = do_bench_cudagraph(forward, rep=args.rep, return_mode="median")
        flops = 4 * b * h * sq * 128 * topk * 64
        row = dict(shape=[b, h, sq, sk, 128], topk64=topk, forward_flops=flops, median_ms=ms, tflops=flops / (ms * 1e9))
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
