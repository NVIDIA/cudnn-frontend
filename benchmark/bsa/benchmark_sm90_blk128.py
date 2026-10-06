# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native blk128 to the identical sparse graph expressed in blk64."""

import argparse
import hashlib
from pathlib import Path
import json
import statistics
import subprocess
import time
import importlib.metadata

import torch
import cudnn
from cudnn import BSA


def measure(fn, repeats, samples, rotations=1):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for index in range(rotations):
                fn(index)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for index in range(rotations):
                result = fn(index)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (repeats * rotations))
    start_wall = time.perf_counter()
    for index in range(20):
        fn(index % rotations)
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - start_wall) * 1e6 / 20
    return {
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "p95_us": sorted(times)[min(len(times) - 1, int(len(times) * 0.95))],
        "samples_us": times,
        "eager_wall_us": wall_us,
    }, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/bsa-sm90-results.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rotate", type=int, default=1, help="Number of independent input buffers per graph cycle")
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    if args.rotate < 1 or args.repeats < 1:
        parser.error("rotate and repeats must be positive")
    assert torch.cuda.get_device_capability()[0] == 9
    shapes = [(1, 2, 256, 512, 3), (1, 8, 2048, 4096, 8), (1, 8, 8192, 8192, 16)]
    if not args.quick:
        shapes += [(2, 16, 16384, 16384, 32), (1, 8, 32768, 32768, 32), (1, 8, 8192, 8192, 64)]
    report = dict(
        gpu=torch.cuda.get_device_name(),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        torch=torch.__version__,
        cutlass_dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cuda=torch.version.cuda,
        layout="bhsd",
        rotations=args.rotate,
        reverse_order=args.reverse_order,
        cudnn_source=cudnn.__file__,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        mode="CUDA Graph complete invocation; CSR/initialization/pre/post included; rotate=1 uses warm cache",
        results=[],
    )
    root = Path(__file__).resolve().parents[2]
    sources = [
        "api.py",
        "_interface.py",
        "_sm90_blk128.py",
        "csrc/bwd/bucketed_k2q_csr.py",
        "csrc/utils/sm90_barriers.py",
        "csrc/fwd/sm90_blk128/bsa_fwd_sm90.py",
        "csrc/bwd/sm90_blk128/bsa_bwd_sm90.py",
    ]
    report["source_sha256"] = {name: hashlib.sha256((root / "python/cudnn/block_sparse_attention" / name).read_bytes()).hexdigest() for name in sources}
    report["gpu_state"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv"], text=True)
    for b, h, sq, sk, count in shapes:
        torch.manual_seed(20260917)
        q = torch.randn(b, h, sq, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, h, sk, 128, device="cuda", dtype=torch.bfloat16)
        v, do = torch.randn_like(k), torch.randn_like(q)
        ids128 = torch.rand(b, h, sq // 128, sk // 128, device="cuda").argsort(-1)[..., :count].int().contiguous()
        ids64 = torch.stack((ids128 * 2, ids128 * 2 + 1), dim=-1).flatten(-2).repeat_interleave(2, dim=2).contiguous()
        row = dict(shape=[b, h, sq, sk, 128], active_blocks=count, dtype="bf16")
        outputs = {}
        pool = [(q, k, v, do)] + [(q.clone(), k.clone(), v.clone(), do.clone()) for _ in range(args.rotate - 1)]
        order = [(128, ids128)] if args.native_only else [(64, ids64), (128, ids128)]
        if args.reverse_order:
            order.reverse()
        for block, indices in order:

            def fwd(index=0):
                q, k, v, do = pool[index]
                return BSA.block_sparse_attention_forward(q, k, v, indices, indices.shape[-1], sparse_block_size=block)

            f = [fwd(index) for index in range(args.rotate)]

            def bwd(index=0):
                q, k, v, do = pool[index]
                return BSA.block_sparse_attention_backward(
                    do,
                    q,
                    k,
                    v,
                    f[index]["o_tensor"],
                    f[index]["lse_tensor"],
                    indices,
                    indices.shape[-1],
                    sparse_block_size=block,
                )

            def train(index=0):
                q, k, v, do = pool[index]
                fo = fwd(index)
                return BSA.block_sparse_attention_backward(do, q, k, v, fo["o_tensor"], fo["lse_tensor"], indices, indices.shape[-1], sparse_block_size=block)

            fm, fo = measure(fwd, args.repeats, 7, args.rotate)
            bm, bo = measure(bwd, args.repeats, 7, args.rotate)
            tm, _ = measure(train, args.repeats, 7, args.rotate)
            row[str(block)] = dict(forward=fm, backward=bm, training=tm)
            outputs[block] = fo, bo
        if not args.native_only:
            for name in ("o_tensor", "lse_tensor"):
                torch.testing.assert_close(outputs[64][0][name], outputs[128][0][name], atol=0.015, rtol=0.015)
            for a, z in zip(outputs[64][1], outputs[128][1]):
                torch.testing.assert_close(a, z, atol=0.03, rtol=0.03)
            row["speedup"] = {phase: row["64"][phase]["median_us"] / row["128"][phase]["median_us"] for phase in ("forward", "backward", "training")}
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        with open(args.output, "w") as file:
            json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
