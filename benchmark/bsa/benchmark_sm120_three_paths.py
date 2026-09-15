# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native KV128, native KV64, and the pinned PR #1010 adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch

from benchmark_sm120_blk128 import _make_block_indices, _rounded_topk, _validate_samples
from cudnn import BSA
from cudnn.block_sparse_attention.csrc.fwd.sm120_blk128 import bsa_fwd_sm120_fa4
from cudnn.block_sparse_attention.csrc.fwd.sm120_blk64 import bsa_fwd_sm120

CHECKOUT = Path(__file__).resolve().parents[2]
PR1010_REVISION = "1d5ed9d596d51087bcc2f81cc44a1f7ea260189f"
LEGACY_FILES = ("api.py", "_interface.py", "csrc/fwd/sm120_blk128/metadata.py")
NAMES = ("native128", "native64", "pr1010")
ORDERS = tuple(itertools.permutations(NAMES))


def verify_sources(legacy_source_dir):
    """Verify pinned files before import and unchanged native64 dependencies."""
    hashes = {}
    for relative in LEGACY_FILES:
        try:
            source = subprocess.check_output(
                ["git", "-C", str(CHECKOUT), "show", f"{PR1010_REVISION}:python/cudnn/block_sparse_attention/{relative}"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Fetch the PR #1010 revision as described in SM120_THREE_PATHS.md before benchmarking") from exc
        saved = (legacy_source_dir / relative).read_bytes()
        if source != saved:
            raise ValueError(f"Archived PR source mismatch: {relative}")
        hashes[f"pr1010/{relative}"] = hashlib.sha256(saved).hexdigest()
    unchanged = subprocess.check_output(
        [
            "git",
            "-C",
            str(CHECKOUT),
            "diff",
            "--name-only",
            PR1010_REVISION,
            "--",
            "python/cudnn/block_sparse_attention/csrc/fwd/sm120_blk64",
            "python/cudnn/block_sparse_attention/csrc/utils",
        ],
        text=True,
    )
    if unchanged.strip():
        raise ValueError("Native64 kernel dependencies differ from PR #1010; this comparison is no longer isolated")
    for name, module in (("native128", bsa_fwd_sm120_fa4), ("native64", bsa_fwd_sm120)):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(CHECKOUT):
            raise RuntimeError("The imported kernel is not from this checkout")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["benchmark"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


@contextmanager
def load_legacy(legacy_source_dir):
    """Import the verified legacy API without replacing the installed BSA API."""
    package_name = "_sm120_pr1010_benchmark"
    if package_name in sys.modules:
        raise RuntimeError("A legacy benchmark package is already active")
    package = ModuleType(package_name)
    package.__path__ = [str(legacy_source_dir.resolve())]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        api = importlib.import_module(f"{package_name}.api")
        metadata = importlib.import_module(f"{package_name}.csrc.fwd.sm120_blk128.metadata")
        yield SimpleNamespace(api=api, lower=metadata.lower_sm120_blk128_metadata)
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]


def make_calls(q, k, v, indices128, topk, legacy):
    """Use exactly the same selected tokens; exclude preprocessing for native64."""
    lowered = legacy.lower(indices128, topk, None, None, seqlen_q=q.shape[2])
    indices64 = lowered.q2k_block_index
    torch.testing.assert_close(indices64[:, :, ::2, 0::2], indices128 * 2, rtol=0, atol=0)
    torch.testing.assert_close(indices64[:, :, ::2, 1::2], indices128 * 2 + 1, rtol=0, atol=0)
    torch.testing.assert_close(indices64[:, :, ::2], indices64[:, :, 1::2], rtol=0, atol=0)

    def native128():
        """Run the current native128 wrapper with the original logical mask."""
        return BSA.block_sparse_attention_forward(q, k, v, indices128, block_sparse_num=topk, sparse_block_size=128)

    def native64():
        """Run native64 with its equivalent mask already prepared outside timing."""
        return BSA.block_sparse_attention_forward(q, k, v, indices64, block_sparse_num=2 * topk, sparse_block_size=64)

    def pr1010():
        """Run the pinned adapter, including its per-call metadata conversion."""
        return legacy.api.block_sparse_attention_forward(q, k, v, indices128, block_sparse_num=topk, sparse_block_size=128)

    return dict(native128=native128, native64=native64, pr1010=pr1010)


def validate_results(q, k, v, indices128, results, *, full_reference=False):
    """Check FP32 samples, full cross-path outputs, and optional small references."""
    record = {}
    for name, result in results.items():
        max_o, max_lse = _validate_samples(q, k, v, result["o_tensor"], result["lse_tensor"], indices128, 128)
        record[name] = {"sample_max_o_error": max_o, "sample_max_lse_error": max_lse}
    for field in ("o_tensor", "lse_tensor"):
        # PR #1010 and native64 have the same kernel and the same physical mask.
        torch.testing.assert_close(results["pr1010"][field], results["native64"][field], rtol=0, atol=0)
    # KV64 and KV128 use different softmax reduction groupings, so equality
    # is numerical rather than bitwise for native128.
    # Use the existing BSA output bound for the small full-reference tests.
    # The target long-sequence outputs additionally permit a tighter check;
    # applying that absolute bound to top-k=1 rejects ordinary BF16 rounding.
    cross_o_atol = 3e-2 if q.shape[2] <= 1024 else 3e-4
    torch.testing.assert_close(results["native128"]["o_tensor"], results["native64"]["o_tensor"], rtol=3e-2, atol=cross_o_atol)
    torch.testing.assert_close(results["native128"]["lse_tensor"], results["native64"]["lse_tensor"], rtol=1e-6, atol=3e-6)
    record["native128_vs_native64"] = {
        "full_o_atol": cross_o_atol,
        "full_o_rtol": 3e-2,
        "full_max_o_diff": (results["native128"]["o_tensor"].float() - results["native64"]["o_tensor"].float()).abs().max().item(),
        "full_max_lse_diff": (results["native128"]["lse_tensor"] - results["native64"]["lse_tensor"]).abs().max().item(),
    }
    if full_reference:
        assert q.shape[2] <= 1024
        for head in range(q.shape[1]):
            for block in range(q.shape[2] // 128):
                tokens = (indices128[0, head, block].long()[:, None] * 128 + torch.arange(128, device=q.device)).flatten()
                rows = slice(block * 128, (block + 1) * 128)
                scores = q[0, head, rows].float() @ k[0, head, tokens].float().T * (128**-0.5)
                ref_o = scores.softmax(-1) @ v[0, head, tokens].float()
                ref_lse = scores.logsumexp(-1)
                for result in results.values():
                    torch.testing.assert_close(result["o_tensor"][0, head, rows].float(), ref_o, rtol=3e-2, atol=3e-2)
                    torch.testing.assert_close(result["lse_tensor"][0, head, rows], ref_lse, rtol=2e-3, atol=2e-3)
    return record


def summarize(samples):
    """Validate balanced samples and compute median-based three-path ratios."""
    if set(samples) != set(NAMES) or len({len(values) for values in samples.values()}) != 1:
        raise ValueError("All three paths must have equal sample counts")
    if any(not values or any(not math.isfinite(value) or value <= 0 for value in values) for values in samples.values()):
        raise ValueError("Timings must be finite and positive")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "median_ms": medians,
        "samples_ms": samples,
        "native128_speedup_vs_native64": medians["native64"] / medians["native128"],
        "native128_speedup_vs_pr1010": medians["pr1010"] / medians["native128"],
        "native128_latency_reduction_vs_native64_pct": 100 * (1 - medians["native128"] / medians["native64"]),
        "native128_latency_reduction_vs_pr1010_pct": 100 * (1 - medians["native128"] / medians["pr1010"]),
    }


def _parse_args(argv=None):
    """Validate the legacy archive, balanced timing options, and new JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-source-dir", type=Path, required=True, help="Archived PR #1010 block_sparse_attention package directory")
    parser.add_argument("--sequence", type=int, default=142720)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=102)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.sequence % 128 or args.heads <= 0:
        parser.error("sequence must be a positive multiple of 128 and heads must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.repeats % 6:
        parser.error("warmup must be nonnegative and repeats must be a positive multiple of six")
    if any(not (args.legacy_source_dir / relative).is_file() for relative in LEGACY_FILES):
        parser.error("legacy-source-dir must contain the archived API, interface, and metadata files")
    if args.json.suffix.lower() != ".json" or args.json.exists():
        parser.error("json must be a new .json output file, not an existing file")
    return args


@torch.no_grad()
def main(argv=None):
    """Verify the SM120 checkout and archive, then run an isolated comparison."""
    args = _parse_args(argv)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("This benchmark requires an SM120 GPU")
    hashes = verify_sources(args.legacy_source_dir)
    with load_legacy(args.legacy_source_dir) as legacy:
        return _run(args, hashes, legacy)


def _run(args, hashes, legacy):
    """Validate and time all paths, then exclusively create the JSON report."""
    torch.manual_seed(20260915)
    shape = (1, args.heads, args.sequence, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    records = []
    print(f"GPU={torch.cuda.get_device_name()} shape={shape} repeats={args.repeats}", flush=True)
    for density in (0.15, 0.20):
        topk = _rounded_topk(args.sequence // 128, density)
        for pattern in ("strided", "local"):
            indices = _make_block_indices(args.heads, args.sequence // 128, topk, pattern, q.device)
            calls = make_calls(q, k, v, indices, topk, legacy)
            results = {}
            for name, call in calls.items():
                print(f"Prepare density={density} pattern={pattern} path={name}", flush=True)
                results[name] = call()
                torch.cuda.synchronize()
            accuracy = validate_results(q, k, v, indices, results, full_reference=args.sequence <= 1024)
            del results
            print("Accuracy PASS: same-mask native64/PR1010 bitwise, native128 full-tensor tolerance", flush=True)
            for iteration in range(args.warmup):
                for name in ORDERS[iteration % 6]:
                    calls[name]()
            torch.cuda.synchronize()
            events = {name: [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)] for name in NAMES}
            for iteration in range(args.repeats):
                for name in ORDERS[iteration % 6]:
                    start, end = events[name][iteration]
                    start.record()
                    calls[name]()
                    end.record()
            torch.cuda.synchronize()
            samples = {name: [start.elapsed_time(end) for start, end in pairs] for name, pairs in events.items()}
            post_accuracy = validate_results(q, k, v, indices, {name: call() for name, call in calls.items()})
            record = {
                "density": topk / (args.sequence // 128),
                "pattern": pattern,
                "topk128": topk,
                "topk64": 2 * topk,
                "accuracy_before": accuracy,
                "accuracy_after": post_accuracy,
                **summarize(samples),
            }
            records.append(record)
            print(json.dumps({key: value for key, value in record.items() if key not in ("samples_ms", "accuracy_before", "accuracy_after")}), flush=True)
    report = {
        "gpu_model": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "shape": shape,
        "dtype": "bf16",
        "seed": 20260915,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "source_sha256": hashes,
        "candidate_revision": subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip(),
        "pr1010_revision": PR1010_REVISION,
        "method": "Same-process six-permutation CUDA-event public-wrapper timing; native64 preprocessing excluded; PR1010 conversion included; compile and validation excluded.",
        "records": records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
