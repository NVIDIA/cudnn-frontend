# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fully paired blk64 comparison. Run serially on an otherwise idle GPU.

The public grouped APIs include all permutations. The baseline retains its
native compiled calls after a one-time warmup, so neither side times CSR
construction. Cache interception is confined to this single-thread benchmark;
the implementation does not intercept or modify any native compile cache.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch
import cudnn
import tvm_ffi
from cuda.bindings import driver as cuda


def record_calls(api, cache):
    api()
    torch.cuda.synchronize()
    saved, calls = dict(cache), []

    def wrap(fn):
        def run(*args):
            calls.append((fn, args))
            return fn(*args)

        return run

    try:
        cache.update({key: wrap(fn) for key, fn in saved.items()})
        result = api()
    finally:
        cache.update(saved)
    if len(calls) != 1:
        raise RuntimeError(f"expected one native blk64 entry, recorded {len(calls)}")
    return result, calls[0]


class Native64:
    def __init__(self, q, k, v, routes, dout):
        from cudnn.block_sparse_attention import _interface

        options = dict(sparse_block_size=64, layout="bshd")
        output, (self.fwd, fargs) = record_calls(
            lambda: cudnn.block_sparse_attention_forward(q, k, v, routes, **options),
            _interface.bsa_attn_fwd_blk64_cutedsl.compile_cache,
        )
        self.out, self.lse = output
        self.fargs = fargs[:-1]
        gradients = [torch.empty_like(t) for t in (q, k, v)]

        def bwd():
            with tvm_ffi.use_torch_stream():
                return cudnn.block_sparse_attention_backward(
                    dout,
                    q,
                    k,
                    v,
                    self.out,
                    self.lse,
                    routes,
                    dq_tensor=gradients[0],
                    dk_tensor=gradients[1],
                    dv_tensor=gradients[2],
                    **options,
                )

        self.grads, (self.bwd, bargs) = record_calls(bwd, _interface._bsa_attn_bwd_bucketed_k2q_csr.compile_cache)
        if len(fargs) != 15 or len(bargs) != 16:
            raise RuntimeError("native call ABI changed; update this benchmark adapter")
        self.bargs = bargs[:-1]
        b, n, h, _ = q.shape
        self.reset = bargs[13].reshape(-1)[2 * b * h * n :]
        self.num_buckets = bargs[10].shape[2]

    def forward(self):
        self.fwd(*self.fargs, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return self.out, self.lse

    def backward(self):
        self.reset.zero_()
        self.bwd(*self.bargs, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return self.grads


class Cake64:
    def __init__(self, q, k, v, routes, module):
        _, length, heads, _ = q.shape
        topk = routes.shape[-1]
        self.out = torch.empty_like(q)
        self.stats = torch.empty((length, heads), device=q.device, dtype=torch.float32)
        self.lse = self.stats.T.unsqueeze(0)
        self.counts = torch.full(routes.shape[1:3], topk, device=q.device, dtype=torch.int32)
        self.sizes = torch.full((k.shape[1] // 64,), 64, device=q.device, dtype=torch.int32)
        total = (length // 64) * heads
        grid = min(total, torch.cuda.get_device_properties(q.device).multi_processor_count)
        self.module = module
        self.params = (
            q[0],
            k[0],
            v[0],
            self.out[0],
            self.stats,
            routes[0],
            self.counts,
            self.sizes,
            topk,
            length,
            length // 64,
            total,
            math.ceil(total / grid),
            heads,
            (128**-0.5) / math.log(2),
            1,
            grid,
            1,
            1,
        )

    def forward(self):
        with tvm_ffi.use_torch_stream():
            self.module.run(*self.params)
        return self.out, self.lse


def make_routes(heads, blocks, topk):
    scores = torch.rand((1, heads, blocks // 2, blocks // 2), device="cuda")
    scores.diagonal(dim1=-2, dim2=-1).fill_(2)
    ids = scores.topk(topk // 2, dim=-1).indices
    first = torch.cat((ids, blocks - 1 - ids), dim=-1).sort(-1).values.int()
    return torch.cat((first, first.flip(-2)), dim=-2)


def difference(actual, expected, *, lse=False, grad=False):
    total_error = total_ref = maximum = 0.0
    x, y = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, x.numel(), 8 * 1024**2):
        a, b = x[start : start + 8 * 1024**2].float(), y[start : start + 8 * 1024**2].float()
        torch.testing.assert_close(a, b, atol=1e-4 if lse else 0.03, rtol=1e-4 if lse else 0.03)
        delta = a - b
        maximum = max(maximum, delta.abs().max().item())
        total_error += delta.square().sum().item()
        total_ref += b.square().sum().item()
    relative = (total_error / max(total_ref, 1e-30)) ** 0.5
    assert relative < (0.001 if lse else 0.012 if grad else 0.008), relative
    return dict(max_abs=maximum, relative_l2=relative)


def measure(callables, rounds, seed):
    graphs, outputs = {}, {}
    samples = {name: [] for name in callables}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(stream):
            for name, fn in callables.items():
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = fn()
                graphs[name] = graph
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            rng = random.Random(seed)
            for _ in range(rounds):
                order = list(graphs)
                rng.shuffle(order)
                for name in order:
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))
        torch.cuda.current_stream().wait_stream(stream)
        return {name: dict(median_ms=statistics.median(values), samples_ms=values) for name, values in samples.items()}
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()


def run_case(args, topk, seed, cake_module):
    torch.manual_seed(seed)
    shape = (1, args.length, args.heads, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    routes = make_routes(args.heads, args.length // 64, topk)
    do = torch.randn_like(q)
    grouped, prepare_ms = {}, {}
    for name, bucket in (("grouped_default", None), ("grouped_one_bucket", args.length // 128)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        grouped[name] = cudnn.block_sparse_attention_grouped_prepare(routes, num_kv_blocks=args.length // 64, bucket_size_blocks=bucket)
        torch.cuda.synchronize()
        prepare_ms[name] = (time.perf_counter() - start) * 1000
    print(json.dumps(dict(event="prepared", topk=topk, seed=seed, prepare_ms=prepare_ms)), flush=True)
    control = Native64(q, k, v, routes, do)
    control.forward()
    control.backward()
    accuracy = {}
    for name, grouping in grouped.items():
        output = cudnn.block_sparse_attention_grouped_forward(q, k, v, grouping)
        grads = cudnn.block_sparse_attention_grouped_backward(do, q, k, v, control.out, control.lse, grouping)
        accuracy[name] = dict(
            output=difference(output[0], control.out),
            lse=difference(output[1], control.lse, lse=True),
            gradients=[difference(a, b, grad=True) for a, b in zip(grads, control.grads)],
        )
        del output, grads
    fwd_calls = {
        "native_blk64": control.forward,
        "grouped_blk128": lambda: cudnn.block_sparse_attention_grouped_forward(q, k, v, grouped["grouped_default"]),
    }
    if cake_module is not None:
        cake = Cake64(q, k, v, routes, cake_module)
        cake.forward()
        accuracy["cake_blk64"] = dict(output=difference(cake.out, control.out), lse=difference(cake.lse, control.lse, lse=True))
        fwd_calls["cake_blk64"] = cake.forward
    bwd_calls = {"native_blk64": control.backward}
    for name, grouping in grouped.items():
        bwd_calls[name] = lambda grouping=grouping: cudnn.block_sparse_attention_grouped_backward(do, q, k, v, control.out, control.lse, grouping)
    bwd = measure(bwd_calls, args.rounds, seed)
    fwd = measure(fwd_calls, args.rounds, seed)
    for phase, factor in ((fwd, 4), (bwd, 10)):
        for entry in phase.values():
            entry["tflops"] = factor * args.heads * args.length * topk * 64 * 128 / entry["median_ms"] / 1e9
    return dict(
        topk=topk,
        seed=seed,
        sparsity=1 - topk * 64 / args.length,
        prepare_ms=prepare_ms,
        accuracy=accuracy,
        fwd=fwd,
        bwd=bwd,
        buckets=dict(native_blk64=control.num_buckets, **{name: g.bucketed_k2q_offsets.shape[2] for name, g in grouped.items()}),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--topk", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4593, 17, 2026])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cake-checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.length % 128 or any(k % 2 or not 2 <= k <= args.length // 64 for k in args.topk):
        parser.error("length must be a multiple of 128; topK must be even and fit the KV64 block count")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        "python/cudnn/block_sparse_attention/api.py",
        "python/cudnn/block_sparse_attention/_grouped.py",
        "python/cudnn/block_sparse_attention/_grouped_reorder.py",
        "python/cudnn/block_sparse_attention/_interface.py",
        "python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py",
        "python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk128/bsa_fwd_sm100.py",
        "python/cudnn/block_sparse_attention/csrc/bwd/sm100_blk64/bsa_bwd_sm100.py",
        "python/cudnn/block_sparse_attention/csrc/bwd/sm100_blk128/bsa_bwd_sm100.py",
        "benchmark/bsa_grouped/bench.py",
    ]
    props = torch.cuda.get_device_properties(0)
    metadata = dict(
        args={k: v for k, v in vars(args).items() if k not in {"output", "cake_checkout"}},
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths},
        gpu=props.name,
        sm_count=props.multi_processor_count,
        capability=torch.cuda.get_device_capability(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn_frontend=cudnn.__version__,
        cudnn_backend=cudnn.backend_version(),
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        timing="CUDA Graph replay; 2 eager + 1 graph warmup; randomized order; medians across rounds then seeds",
        included="All grouped permutations; backward accumulator resets, preprocessing, main and conversion",
        excluded="Routing, CPU prepare/CSR, compilation, allocation, Python dispatch",
        pattern="100% paired on both axes, mirrored nonadjacent pairs; every KV used",
        flops="Active work: forward 4*B*H*N*topK*64*D; backward 10*B*H*N*topK*64*D (not dense-equivalent)",
    )
    cake_module = None
    if args.cake_checkout:
        sys.path.insert(0, str(args.cake_checkout.resolve()))
        from flashinfer import cake_vsa

        cake_module = cake_vsa._load_module("blk64_persistent", "sm_100a")
        source = args.cake_checkout / "csrc/cake_vsa/cake_vsa_blk64_persistent_sm_100a.cu"
        metadata["cake"] = dict(
            reference="https://github.com/flashinfer-ai/flashinfer/pull/4593",
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.cake_checkout, text=True).strip(),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            flags=["-cubin", "--std=c++17", "--use_fast_math", "-arch=sm_100a"],
        )
    result = dict(metadata=metadata, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for topk in args.topk:
        for seed in args.seeds:
            case = run_case(args, topk, seed, cake_module)
            result["cases"].append(case)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(dict(event="result", topk=topk, seed=seed, fwd=case["fwd"], bwd=case["bwd"])), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
