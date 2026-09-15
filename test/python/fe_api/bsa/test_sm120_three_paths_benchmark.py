# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch

pytestmark = pytest.mark.L0


def _load_benchmark(monkeypatch):
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("three-path benchmark requires a supported CuTe DSL version")
    benchmark_dir = Path(__file__).resolve().parents[4] / "benchmark/bsa"
    monkeypatch.syspath_prepend(str(benchmark_dir))
    spec = importlib.util.spec_from_file_location("_three_path_benchmark_test", benchmark_dir / "benchmark_sm120_three_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def legacy_source_dir():
    source = os.environ.get("SM120_PR1010_SOURCE_DIR")
    if source is None:
        pytest.skip("set SM120_PR1010_SOURCE_DIR to the archived PR #1010 BSA package for legacy comparisons")
    return Path(source)


def test_three_path_summary_and_balanced_order(monkeypatch):
    bench = _load_benchmark(monkeypatch)
    assert len(bench.ORDERS) == 6
    for position in range(3):
        assert all(sum(order[position] == name for order in bench.ORDERS) == 2 for name in bench.NAMES)
    result = bench.summarize(dict(native128=[10, 11, 9], native64=[20, 19, 21], pr1010=[25, 24, 26]))
    assert result["native128_speedup_vs_native64"] == 2
    assert result["native128_latency_reduction_vs_pr1010_pct"] == 60
    for samples in (
        {},
        dict(native128=[], native64=[], pr1010=[]),
        dict(native128=[0], native64=[1], pr1010=[1]),
        dict(native128=[float("nan")], native64=[1], pr1010=[1]),
        dict(native128=[1, 2], native64=[1], pr1010=[1]),
    ):
        with pytest.raises(ValueError):
            bench.summarize(samples)


@pytest.mark.parametrize("option,value", [("--sequence", "129"), ("--heads", "0"), ("--warmup", "-1"), ("--repeats", "7")])
def test_three_path_invalid_arguments(monkeypatch, tmp_path, option, value):
    bench = _load_benchmark(monkeypatch)
    with pytest.raises(SystemExit):
        bench._parse_args(["--legacy-source-dir", str(tmp_path), "--json", str(tmp_path / "output.json"), option, value])


def test_three_path_archive_and_output_guards(monkeypatch, tmp_path):
    bench = _load_benchmark(monkeypatch)
    output = tmp_path / "output.json"
    args = ["--legacy-source-dir", str(tmp_path), "--json", str(output)]
    with pytest.raises(SystemExit):
        bench._parse_args(args)
    for relative in bench.LEGACY_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Not a verified archive.\n")
    assert bench._parse_args(args).repeats == 102
    output.write_text("preserve me")
    with pytest.raises(SystemExit):
        bench._parse_args(args)
    assert output.read_text() == "preserve me"


def test_three_path_tampered_archive(monkeypatch, legacy_source_dir, tmp_path):
    bench = _load_benchmark(monkeypatch)
    for relative in bench.LEGACY_FILES:
        copied = tmp_path / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((legacy_source_dir / relative).read_bytes())
    bench.verify_sources(tmp_path)
    api_path = tmp_path / "api.py"
    api_path.write_bytes(api_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Archived PR source mismatch"):
        bench.verify_sources(tmp_path)


@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("pattern", ["strided", "local"])
@pytest.mark.parametrize("topk", [1, 3])
@torch.no_grad()
def test_three_path_full_reference(monkeypatch, legacy_source_dir, pattern, topk):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("three-path benchmark requires SM120")
    bench = _load_benchmark(monkeypatch)
    bench.verify_sources(legacy_source_dir)
    torch.manual_seed(20260915)
    q, k, v = [torch.randn((1, 2, 1024, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    indices = bench._make_block_indices(2, 8, topk, pattern, q.device)
    with bench.load_legacy(legacy_source_dir) as legacy:
        calls = bench.make_calls(q, k, v, indices, topk, legacy)
        results = {name: call() for name, call in calls.items()}
        torch.cuda.synchronize()
        bench.validate_results(q, k, v, indices, results, full_reference=True)
    assert not any(name.startswith("_sm120_pr1010_benchmark") for name in sys.modules)


@pytest.mark.gpu_exclusive
def test_three_path_cli_smoke(monkeypatch, legacy_source_dir, tmp_path):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("three-path benchmark requires SM120")
    bench = _load_benchmark(monkeypatch)
    output = tmp_path / "three_paths.json"
    bench.main(["--legacy-source-dir", str(legacy_source_dir), "--sequence", "512", "--heads", "2", "--warmup", "1", "--repeats", "6", "--json", str(output)])
    report = json.loads(output.read_text())
    assert len(report["records"]) == 4
    assert all(len(samples) == 6 for record in report["records"] for samples in record["samples_ms"].values())
    assert report["pr1010_revision"] == bench.PR1010_REVISION
    assert not {"hostname", "ip", "uuid", "legacy_source_dir"}.intersection(report)
    assert str(legacy_source_dir) not in output.read_text()
    assert not any(name.startswith("_sm120_pr1010_benchmark") for name in sys.modules)
