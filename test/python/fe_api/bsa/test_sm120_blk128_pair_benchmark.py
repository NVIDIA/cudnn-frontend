# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Run paired native SM120 blk128 kernel comparisons and harness regressions.

Requires this checkout's CuTe BSA package, CUDA-enabled PyTorch, an SM120 GPU,
CuTe DSL >= 4.7.0, and pytest dependencies (see test_BSA_attention_forward.py).
From the repository root, run the small harness tests:

    (cd test/python && CUDA_VISIBLE_DEVICES=0 python -m pytest -q fe_api/bsa/test_sm120_blk128_pair_benchmark.py)

For an A/B measurement, save a trusted baseline kernel source locally. To
reproduce the two-fold-unroll baseline, extract commit 9869b9b6 without
switching branches:

    git fetch https://github.com/tiffany940107/cudnn-frontend.git bsa-sm120-native-blk128-fa4-style
    mkdir -p agent/agent_space agent/agent_benchmark
    VSA_BASELINE_ROOT="$(mktemp -d -p agent/agent_space blk128_baseline.XXXXXX)"
    git archive 9869b9b6 python/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120_fa4.py | tar -x -C "$VSA_BASELINE_ROOT"
    VSA_BASELINE_SOURCE="$VSA_BASELINE_ROOT/python/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120_fa4.py"
    CUDA_VISIBLE_DEVICES=0 python benchmark/bsa/benchmark_sm120_blk128_pair.py \
      --baseline-source "$VSA_BASELINE_SOURCE" --sequence 142720 --heads 8 \
      --densities 0.15 0.20 --patterns strided local --warmup 10 --repeats 101 \
      --seed 20260914 --min-speedup 1.05 --fail-below-target \
      --json agent/agent_benchmark/paired_repeat.json

Only use trusted baseline files: the harness imports and executes them.
Choose fresh report paths, use an idle GPU without a profiler, and repeat in
independent processes. The harness alternates A/B and B/A, excludes compilation,
checks sampled FP32-reference rows, and restores the kernel class/compile cache.
It compares a saved kernel file with the checkout, not complete environments;
its sampled accuracy checks do not establish full-tensor bitwise equivalence.

--fail-below-target returns nonzero if any case misses the speedup gate; that
alone is not a correctness failure. A 1.05x speedup is a 4.76% latency reduction;
use --min-speedup 1.052631579 for a full 5% latency reduction. See PR #1070 for
measured results and limitations; do not combine ratios from different baselines.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_benchmark(monkeypatch):
    """Load the checkout's paired harness only with a supported CuTe DSL."""
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("paired blk128 benchmark requires a supported CuTe DSL version")
    repo = Path(__file__).resolve().parents[4]
    benchmark_dir = repo / "benchmark" / "bsa"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    spec = importlib.util.spec_from_file_location("_pair_benchmark_test", benchmark_dir / "benchmark_sm120_blk128_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pair_benchmark_summary(monkeypatch):
    """Check median arithmetic and reject empty, nonpositive, or NaN timings."""
    benchmark = _load_benchmark(monkeypatch)
    report = benchmark._summarize([10.0, 12.0, 11.0], [5.0, 6.0, 5.5])
    assert report["speedup"] == 2.0
    assert report["latency_reduction_pct"] == 50.0
    with pytest.raises(ValueError):
        benchmark._summarize([], [])
    with pytest.raises(ValueError):
        benchmark._summarize([1.0], [0.0])
    with pytest.raises(ValueError):
        benchmark._summarize([float("nan")], [1.0])


@pytest.mark.gpu_exclusive
def test_pair_benchmark_sm120_smoke(monkeypatch, tmp_path):
    """Check target failure, report privacy, and kernel/cache restoration on SM120."""
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("paired blk128 benchmark requires SM120")
    benchmark = _load_benchmark(monkeypatch)
    original_class = benchmark.bsa_fwd_sm120_fa4.BlockSparseAttnForwardSm120Blk128Fa4
    original_cache = dict(benchmark._interface.bsa_attn_fwd.compile_cache)
    report_path = tmp_path / "paired.json"
    # An impossible target exercises the failure exit without relying on noise.
    status = benchmark.main(
        [
            "--baseline-source",
            benchmark.bsa_fwd_sm120_fa4.__file__,
            "--sequence",
            "512",
            "--heads",
            "2",
            "--densities",
            "0.5",
            "--patterns",
            "strided",
            "local",
            "--warmup",
            "1",
            "--repeats",
            "3",
            "--min-speedup",
            "1000",
            "--fail-below-target",
            "--json",
            str(report_path),
        ]
    )
    assert status == 1
    report = json.loads(report_path.read_text())
    assert len(report["records"]) == 2
    assert all(not record["meets_target"] for record in report["records"])
    assert all(len(record["baseline_samples_ms"]) == 3 for record in report["records"])
    assert report["baseline_source_sha256"] == report["candidate_source_sha256"]
    assert not {"hostname", "ip", "uuid", "baseline_source", "candidate_source"}.intersection(report)
    assert benchmark.bsa_fwd_sm120_fa4.BlockSparseAttnForwardSm120Blk128Fa4 is original_class
    assert benchmark._interface.bsa_attn_fwd.compile_cache == original_cache
