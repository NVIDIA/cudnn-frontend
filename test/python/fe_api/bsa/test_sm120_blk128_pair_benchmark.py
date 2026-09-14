# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_benchmark(monkeypatch):
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
