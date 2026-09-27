# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure complete eager blk64 BSA invocations with reproducible synthetic inputs."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess


def measure(fn, warmup, repeats, rounds, flush):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(rounds):
        samples = []
        for _ in range(repeats):
            flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        measurements.append({"median_ms": statistics.median(samples), "samples_ms": samples})
    return measurements, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-root", type=Path, required=True, help="Fail if cudnn is imported from another checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sq", type=int, default=8192)
    parser.add_argument("--sk", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks-per-row", type=int, default=16)
    parser.add_argument("--layout", choices=("bhsd", "bshd"), default="bshd")
    parser.add_argument("--metadata", choices=("fixed", "variable"), default="variable")
    parser.add_argument("--bucket-size", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if min(args.sq, args.sk, args.heads, args.blocks_per_row, args.warmup, args.repeats, args.rounds) <= 0:
        parser.error("shapes and iteration counts must be positive")
    if args.bucket_size is not None and args.bucket_size <= 0:
        parser.error("bucket-size must be positive")

    import torch
    import cudnn
    from cudnn import BSA

    root = args.expected_source_root.resolve()
    expected_package = root / "python" / "cudnn"
    if Path(cudnn.__file__).resolve().parent != expected_package:
        raise RuntimeError(f"expected {expected_package}, imported {cudnn.__file__}")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("this benchmark targets SM100/SM103")
    nq, nk = (args.sq + 63) // 64, (args.sk + 63) // 64
    width = min(args.blocks_per_row, nk)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    host_tensors = [torch.randn((1, args.heads, length, 128), generator=generator, dtype=torch.bfloat16) for length in (args.sq, args.sk, args.sk, args.sq)]
    # Every row has unique KV IDs; cyclic neighborhoods include cross-window reuse.
    rows = torch.arange(nq).view(1, 1, nq, 1)
    heads = torch.arange(args.heads).view(1, args.heads, 1, 1)
    indices = ((rows + heads * 7 + torch.arange(width)) % nk).int().contiguous()
    counts = (width - (torch.arange(args.heads * nq) % min(width, 4))).int().view(1, args.heads, nq)
    sizes = torch.full((nk,), 64, dtype=torch.int32)
    sizes[-1] = args.sk - (nk - 1) * 64
    digest = hashlib.sha256()
    for tensor in (*host_tensors, indices, counts, sizes):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    if args.layout == "bshd":
        host_tensors = [t.transpose(1, 2).contiguous() for t in host_tensors]
    q, k, v, do = [t.cuda() for t in host_tensors]
    indices, sizes = indices.cuda(), sizes.cuda()
    counts = counts.cuda() if args.metadata == "variable" else None
    kwargs = dict(block_sparse_num=width, block_sizes=sizes, q2k_block_nums=counts, sparse_block_size=64, layout=args.layout)

    def forward():
        return BSA.block_sparse_attention_forward(q, k, v, indices, **kwargs)

    output, lse = forward()

    def backward():
        return BSA.block_sparse_attention_backward(do, q, k, v, output, lse, indices, **kwargs, bucket_size_blocks=args.bucket_size)

    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fwd_times, final_forward = measure(forward, args.warmup, args.repeats, args.rounds, flush)
    bwd_times, final_backward = measure(backward, args.warmup, args.repeats, args.rounds, flush)
    if not all(torch.isfinite(t).all().item() for t in (*final_forward, *final_backward)):
        raise RuntimeError("nonfinite output; discard timing")
    source_dir = expected_package / "block_sparse_attention"
    report = {
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "cudnn_source": cudnn.__file__,
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p.relative_to(source_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.rglob("*.py"))},
        "input_sha256": digest.hexdigest(),
        "gpu_state": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv"], text=True),
        "timing": "CUDA events around complete eager API calls; CSR, pairing, preprocessing and output conversion included; 256 MiB flush before each sample",
        "validation": "Finite outputs only; run the independent numerical tests before interpreting timings",
        "forward": fwd_times,
        "backward": bwd_times,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"forward_ms": [x["median_ms"] for x in fwd_times], "backward_ms": [x["median_ms"] for x in bwd_times]}, indent=2))


if __name__ == "__main__":
    main()
