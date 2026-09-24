# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark native SM90 blk64 forward/backward/training; counts denote KV64 blocks per Q64 row."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
import cudnn
from cudnn import BSA
from benchmark_sm90_blk128 import measure


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/bsa-sm90-blk64-results.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rotate", type=int, default=1, help="Number of independent input buffers per graph cycle")
    args = parser.parse_args()
    if args.rotate < 1 or args.repeats < 1:
        parser.error("rotate and repeats must be positive")
    root = Path(__file__).resolve().parents[2]
    if Path(cudnn.__file__).resolve() != root / "python/cudnn/__init__.py":
        raise RuntimeError(f"cudnn resolves to {cudnn.__file__}; install this checkout with uv pip install -e .")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("This benchmark requires an SM90 GPU")
    shapes = [(1, 2, 256, 512, 6), (1, 8, 2048, 4096, 16), (1, 8, 8192, 8192, 32)]
    if not args.quick:
        shapes += [(2, 16, 16384, 16384, 64), (1, 8, 32768, 32768, 64), (1, 8, 8192, 8192, 128)]
    report = dict(
        gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        torch=torch.__version__,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cuda=torch.version.cuda,
        layout="bhsd",
        sparse_block_size=64,
        rotations=args.rotate,
        cudnn_source=cudnn.__file__,
        mode="CUDA Graph complete invocation; CSR/initialization/pre/post included; rotate=1 uses warm cache",
        sparse_pattern="native_blk64_random",
        active_blocks_unit="KV64 blocks per Q64 row",
        seed=20260917,
        repeats=args.repeats,
        samples=7,
        results=[],
    )
    for b, h, sq, sk, count in shapes:
        torch.manual_seed(20260917)
        q = torch.randn(b, h, sq, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, h, sk, 128, device="cuda", dtype=torch.bfloat16)
        v, do = torch.randn_like(k), torch.randn_like(q)
        ids64 = torch.rand(b, h, sq // 64, sk // 64, device="cuda").argsort(-1)[..., :count].int().contiguous()
        row = dict(shape=[b, h, sq, sk, 128], active_blocks=count, density=count * 64 / sk, dtype="bf16")
        pool = [(q, k, v, do)] + [(q.clone(), k.clone(), v.clone(), do.clone()) for _ in range(args.rotate - 1)]

        def fwd(index=0):
            q, k, v, do = pool[index]
            return BSA.block_sparse_attention_forward(q, k, v, ids64, count, sparse_block_size=64)

        f = [fwd(index) for index in range(args.rotate)]

        def bwd(index=0):
            q, k, v, do = pool[index]
            return BSA.block_sparse_attention_backward(do, q, k, v, f[index]["o_tensor"], f[index]["lse_tensor"], ids64, count, sparse_block_size=64)

        def train(index=0):
            q, k, v, do = pool[index]
            fo = fwd(index)
            return BSA.block_sparse_attention_backward(do, q, k, v, fo["o_tensor"], fo["lse_tensor"], ids64, count, sparse_block_size=64)

        fm, _ = measure(fwd, args.repeats, 7, args.rotate)
        bm, _ = measure(bwd, args.repeats, 7, args.rotate)
        tm, _ = measure(train, args.repeats, 7, args.rotate)
        row["64"] = dict(forward=fm, backward=bm, training=tm)
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        with open(args.output, "w") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
