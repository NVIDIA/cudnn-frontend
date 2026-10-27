# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure accounting and profiler coverage for the SE benchmark tools."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0
BENCH = Path(__file__).resolve().parents[4] / "benchmark/causal_conv1d_se"


def _load(name):
    """Load a benchmark entry point without executing its command-line main."""
    spec = importlib.util.spec_from_file_location(f"se_tool_{name}", BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool_tree(tmp_path, monkeypatch):
    """Provide a tiny source snapshot and GPU identity for host-only orchestration tests."""
    bench = tmp_path / "benchmark/causal_conv1d_se"
    bench.mkdir(parents=True)
    kernels = tmp_path / "python/cudnn/causal_conv1d_se"
    kernels.mkdir(parents=True)
    for name in ("kernel.py", "backward_kernel.py", "b2b_kernel.py"):
        (kernels / name).write_text("# frozen test source\n")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    return bench


def test_controlled_timeout_is_recorded_and_sweep_continues(tool_tree, tmp_path, monkeypatch):
    """Keep timeout output and execute subsequent jobs instead of losing the sweep summary."""
    tool = _load("controlled_matrix")
    tool.__file__ = str(tool_tree / "controlled_matrix.py")
    case = dict(family="ncw", dtype="float32", shape=[1, 8, 17], width=3, silu=True, directions=["forward", "backward"])
    (tool_tree / "supplemental_workloads.json").write_text(json.dumps({"cases": [case]}))
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [tool.__file__, "--supplement", "--output", str(output), "--backend-library", "unused.so"])
    calls = []

    def run(command, **kwargs):
        """Time out the first direction and complete the second one."""
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 900, output=b"partial status 3010\n", stderr=b"partial error\n")
        return subprocess.CompletedProcess(command, 0, stdout="completed\n", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(SystemExit) as error:
        tool.main()
    assert error.value.code == 1
    records = json.loads((output / "summary.json").read_text())["results"]
    assert [r["status"] for r in records] == ["failed", "pass"]
    assert [r["returncode"] for r in records] == [124, 0]
    assert len(calls) == 2
    assert (output / "000-forward.log").read_text() == "partial status 3010\nTimed out after 900 seconds\npartial error\n"


def test_b2b_backward_rejection_is_backend_unsupported(tool_tree, tmp_path, monkeypatch):
    """Classify the backward status-3010 spelling without failing the remaining manifest jobs."""
    tool = _load("manifest_matrix")
    tool.__file__ = str(tool_tree / "manifest_matrix.py")
    cases = [
        dict(family="b2b", dtype="float64", shape=[1, 8, 17], width=3, mixer_width=4, silu=False, directions=["forward", "backward"]),
        dict(family="nwc", dtype="bfloat16", shape=[2, 768, 4096], width=4, silu=True, directions=["forward"]),
    ]
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": cases}))
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [tool.__file__, "--manifest", str(manifest), "--output", str(output), "--backend-library", "unused.so"])

    def run(command, **kwargs):
        """Return a backward admission rejection only for the fused probe."""
        fused = Path(command[1]).name == "probe_b2b.py"
        return subprocess.CompletedProcess(command, 1 if fused else 0, stdout="", stderr="backend backward status 3010" if fused else "")

    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(SystemExit) as error:
        tool.main()
    assert error.value.code == 0
    records = json.loads((output / "summary.json").read_text())["results"]
    assert [r["status"] for r in records] == ["backend_unsupported", "pass"]


def test_zero_regressions_replaces_stale_report(tmp_path, monkeypatch):
    """Produce an empty, readable report when every candidate is at least as fast as its backend."""
    tool = _load("summarize_regressions")
    baseline = tmp_path / "baseline.csv"
    baseline.write_text("candidate_us,backend_us\n1,2\n2,2\n")
    output = tmp_path / "output"
    output.mkdir()
    (output / "regressions.csv").write_text("stale,row\n1,2\n")
    monkeypatch.setattr(sys, "argv", [tool.__file__, "--baseline", str(baseline), "--replay", f"sm90={tmp_path}", "--output", str(output)])
    tool.main()
    with (output / "regressions.csv").open() as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == ["candidate_us", "backend_us"]
        assert list(reader) == []
    summary = json.loads((output / "summary.json").read_text())
    assert summary["original_regressions"] == summary["replayed"] == 0
    assert summary["status"] == summary["categories"] == {}
    assert json.loads((output / "compilation.json").read_text()) == []


@pytest.mark.parametrize("script,config", [("tune", {}), ("tune_bwd", {"rows": "32"}), ("probe_b2b", {"projection_width": 3})])
def test_replay_profiles_the_module_that_runs(script, config, tool_tree, tmp_path, monkeypatch):
    """Verify all replay entry points use the patched timer and emit profiler ranges."""
    tool = _load("investigate")
    events = []
    tune = ModuleType("tune")
    tune.measure = lambda functions, **kwargs: events.append("timed")
    cutlass = ModuleType("cutlass")
    cutlass.__version__ = "test"
    cutlass.cute = SimpleNamespace(compile=lambda fn: fn)
    backend = ModuleType("backend")
    backend.Backend = type("Backend", (), {"forward": lambda self: None, "backward": lambda self: None})
    module = tune if script == "tune" else ModuleType(script)
    module.main = lambda: tune.measure({"candidate": lambda: events.append("launch")})
    for name, value in {"cutlass": cutlass, "tune": tune, "backend": backend, script: module}.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "cudart", lambda: SimpleNamespace(cudaProfilerStart=lambda: events.append("start"), cudaProfilerStop=lambda: events.append("stop"))
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: events.append(name))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    evidence = tmp_path / "case.json"
    evidence.write_text(json.dumps({"args": config}))
    output = tmp_path / "replay"
    tool.worker(SimpleNamespace(evidence=evidence, output=output, profile=True, sweep=False))
    assert events == ["start", "candidate", *(["launch"] * 5), "pop", "stop", "timed"]
    record = json.loads((output / "compilation.json").read_text())
    assert record["command"][0] == f"{script}.py"
