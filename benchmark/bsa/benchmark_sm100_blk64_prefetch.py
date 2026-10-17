# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interleaved, same-buffer SM100 blk64 forward A/B and independent A/A.

Install the current checkout, then run on an idle GPU, for example:
    python benchmark/bsa/benchmark_sm100_blk64_prefetch.py --baseline-ref <base-commit> --output results.json

The baseline is loaded from git into two independent modules. Routing, allocation,
compilation and reference checks are outside CUDA Graph timing. Every variant
uses identical BSHD inputs and ascending mean-pool top-k block indices.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cutlass
import cudnn
import torch
from cuda.bindings import driver as cuda
from cudnn.block_sparse_attention import _interface
from cudnn.block_sparse_attention.csrc.fwd.sm100_blk64 import bsa_fwd_sm100

ROOT = Path(__file__).resolve().parents[2]
KERNEL = Path("python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_sm100.py")


def load_baseline(source, directory, name):
    path = directory / f"{name}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.BlockSparseAttnForwardSm100Blk64


def gpu_snapshot():
    properties = torch.cuda.get_device_properties(0)
    pci_bus = f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0"
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            pci_bus,
            "--query-gpu=name,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


class Runner:
    def __init__(self, cls, inputs, indices):
        cache = _interface.bsa_attn_fwd_blk64_cutedsl.compile_cache
        original_cls = _interface.BlockSparseAttnForwardSm100Blk64
        cache.clear()

        def api():
            return cudnn.BSA.block_sparse_attention_forward(*inputs, indices, sparse_block_size=64, layout="bshd", kv_splits=1)

        try:
            _interface.BlockSparseAttnForwardSm100Blk64 = cls
            first = api()
            torch.cuda.synchronize()
            assert len(cache) == 1, "Expected one unsplit attention kernel."
            key, fn = next(iter(cache.items()))
            calls = []

            def record(*args):
                calls.append(args)
                return fn(*args)

            cache[key] = record
            result = api()
            torch.cuda.synchronize()
            assert len(calls) == 1
            self.fn, self.params = fn, calls[0][:-1]
            self.output, self.lse = result["o_tensor"], result["lse_tensor"]
            self.compile_key = str(key)
            assert torch.equal(first["o_tensor"], self.output)
            assert torch.equal(first["lse_tensor"], self.lse)
        finally:
            _interface.BlockSparseAttnForwardSm100Blk64 = original_cls
            cache.clear()

    def run(self):
        self.fn(*self.params, cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def reference_check(inputs, indices, runners):
    q, k, v = inputs
    length, heads = q.shape[1:3]
    records = {name: dict(output_max_abs=0.0, lse_max_abs=0.0, squared_error=0.0, squared_reference=0.0) for name in runners}
    for head in sorted({0, heads // 2, heads - 1}):
        for block in sorted({0, length // 128, length // 64 - 1}):
            rows = torch.tensor([block * 64, block * 64 + 31], device="cuda")
            keys = (indices[0, head, block].long()[:, None] * 64 + torch.arange(64, device="cuda")).flatten()
            logits = q[0, rows, head].float() @ k[0, keys, head].float().T / math.sqrt(128)
            expected = logits.softmax(-1) @ v[0, keys, head].float()
            expected_lse = logits.logsumexp(-1)
            for name, runner in runners.items():
                actual, lse = runner.output[0, rows, head].float(), runner.lse[0, head, rows]
                torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
                torch.testing.assert_close(lse, expected_lse, atol=0.0001, rtol=0.00001)
                rec = records[name]
                rec["output_max_abs"] = max(rec["output_max_abs"], (actual - expected).abs().max().item())
                rec["lse_max_abs"] = max(rec["lse_max_abs"], (lse - expected_lse).abs().max().item())
                rec["squared_error"] += (actual - expected).square().sum().item()
                rec["squared_reference"] += expected.square().sum().item()
    for rec in records.values():
        rec["output_relative_l2"] = math.sqrt(rec.pop("squared_error") / max(rec.pop("squared_reference"), 1e-30))
    return records


def measure(runners, rounds, repeats, seed):
    graphs, samples, orders = {}, {name: [] for name in runners}, []
    for name, runner in runners.items():
        for _ in range(5):
            runner.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                runner.run()
        graphs[name] = graph
        runner.output.fill_(float("nan"))
        runner.lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(runner.output).all() and torch.isfinite(runner.lse).all()
    until = time.perf_counter() + 1.0
    while time.perf_counter() < until:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    rng = random.Random(seed)
    for _ in range(rounds):
        order = list(runners)
        rng.shuffle(order)
        orders.append(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    return {name: dict(median_ms=statistics.median(values), samples_ms=values) for name, values in samples.items()}, orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--topks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4593, 17, 2026])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.length % 64 == 0 and all(0 < k <= args.length // 64 for k in args.topks)
    assert Path(bsa_fwd_sm100.__file__).resolve() == (ROOT / KERNEL).resolve(), "Install this checkout before benchmarking."
    baseline_sha = subprocess.check_output(["git", "rev-parse", args.baseline_ref], cwd=ROOT, text=True).strip()
    baseline_source = subprocess.check_output(["git", "show", f"{baseline_sha}:{KERNEL}"], cwd=ROOT)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = dict(
        baseline_commit=baseline_sha,
        baseline_source_sha256=hashlib.sha256(baseline_source).hexdigest(),
        candidate_source_sha256=hashlib.sha256((ROOT / KERNEL).read_bytes()).hexdigest(),
        imported_kernel=bsa_fwd_sm100.__file__,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cutlass=cutlass.__version__,
        gpu=str(torch.cuda.get_device_properties(0)),
        gpu_start=gpu_snapshot(),
        arguments=vars(args) | dict(output=str(args.output)),
        scope="CUDA Graph attention kernel, O and LSE; excludes routing, allocations, compilation and references",
        flops="4 * B * H * N * topK * 64 * D",
    )
    report = dict(metadata=metadata, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with tempfile.TemporaryDirectory(prefix="bsa-prefetch-") as temp:
        classes = {
            "baseline": load_baseline(baseline_source, Path(temp), "baseline_a"),
            "candidate": bsa_fwd_sm100.BlockSparseAttnForwardSm100Blk64,
            "baseline_repeat": load_baseline(baseline_source, Path(temp), "baseline_b"),
        }
        for topk in args.topks:
            for seed in args.seeds:
                print(f"TopK={topk}, seed={seed}", flush=True)
                torch.manual_seed(seed)
                inputs = [torch.randn(1, args.length, args.heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
                pooled_q, pooled_k = [t.view(1, args.length // 64, 64, args.heads, 128).float().mean(2).transpose(1, 2) for t in inputs[:2]]
                indices = (pooled_q @ pooled_k.transpose(-1, -2)).topk(topk, -1).indices.sort(-1).values.to(torch.int32)
                del pooled_q, pooled_k
                construction_order = list(classes)
                if args.seeds.index(seed) % 2:
                    construction_order.reverse()
                runners = {name: Runner(classes[name], inputs, indices) for name in construction_order}
                rec = dict(
                    topk=topk,
                    seed=seed,
                    gpu_before=gpu_snapshot(),
                    construction_order=construction_order,
                    compile_keys={name: r.compile_key for name, r in runners.items()},
                )
                rec["timing"], rec["orders"] = measure(runners, args.rounds, args.repeats, seed)
                rec["fp32_sampled"] = reference_check(inputs, indices, runners)
                baseline, candidate = runners["baseline"], runners["candidate"]
                torch.testing.assert_close(candidate.output, baseline.output, atol=0.001, rtol=0.001)
                torch.testing.assert_close(candidate.lse, baseline.lse, atol=0.0001, rtol=0.00001)
                rec["output_bitwise_equal"] = torch.equal(candidate.output, baseline.output)
                rec["lse_bitwise_equal"] = torch.equal(candidate.lse, baseline.lse)
                rec["gain_percent"] = 100 * (rec["timing"]["baseline"]["median_ms"] / rec["timing"]["candidate"]["median_ms"] - 1)
                rec["aa_percent"] = 100 * (rec["timing"]["baseline_repeat"]["median_ms"] / rec["timing"]["baseline"]["median_ms"] - 1)
                rec["gpu_after"] = gpu_snapshot()
                report["cases"].append(rec)
                save()
                print(
                    json.dumps({key: rec[key] for key in ("topk", "seed", "gain_percent", "aa_percent", "output_bitwise_equal", "lse_bitwise_equal")}),
                    flush=True,
                )
                del inputs, indices, runners, baseline, candidate
                gc.collect()
                torch.cuda.empty_cache()
    metadata["completed"] = True
    save()


if __name__ == "__main__":
    main()
