# SPDX-FileCopyrightText: Copyright (c) 2026 Alfred Wong
# SPDX-License-Identifier: Apache-2.0

"""CPU subprocess tests for the standalone benchmark summary command."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.L0
SCRIPT = Path(__file__).with_name("summarize_gdn2_steep_decay.py")
LABELS = ("base-A1", "patch-B1", "patch-B2", "base-A2")


@pytest.fixture
def runs():
    """Create complete ABBA records with independently known timing ratios."""
    records = {}
    for label in LABELS:
        median = 2.0 if label.startswith("base") else 1.0
        records[label] = {
            "label": label,
            "script_sha256": "script",
            "helper_sha256": label.split("-")[0],
            **dict.fromkeys(("torch", "cuda", "dsl", "cudnn_backend", "gpu", "capability", "seed"), "same"),
            "results": [
                {
                    "tokens": tokens,
                    "steep_channels": channels,
                    "correct_within_one_percent": True,
                    "errors_fp64": {name: {"relative_frobenius": 0.005, "max_abs": 0.001} for name in ("output", "q", "k", "v", "g", "b", "w")},
                    "eager": {"median_ms": median},
                    "gpu_busy": {"median_ms": median},
                    "scheduler_records": [{"plan": "CompiledGdn2Bwd", "schedulers": {"scheduler_bwd": [0, 2 if channels else 0]}}],
                }
                for tokens in (16, 256)
                for channels in (0, 1, 64)
            ],
        }
    return records


def summarize(tmp_path, runs, optimized=False):
    """Execute the real CLI against four input files."""
    for label, record in runs.items():
        (tmp_path / f"{label}.json").write_text(json.dumps(record))
    return subprocess.run([sys.executable, *(["-O"] if optimized else []), str(SCRIPT), str(tmp_path)], capture_output=True, text=True)


def test_complete_summary(tmp_path, runs):
    """Preserve valid measurements and write JSON that strict readers accept."""
    result = summarize(tmp_path, runs)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text(), parse_constant=lambda value: pytest.fail(f"Nonstandard JSON: {value}"))
    assert len(summary["results"]) == 6
    for row in summary["results"]:
        for mode in ("eager", "gpu_busy"):
            assert row[mode]["base_ms"] == 2.0
            assert row[mode]["patch_ms"] == 1.0
            assert row[mode]["speedup"] == 2.0
            assert row[mode]["reduction_percent"] == 50.0
        assert row["max_fp64_relative_error"] == 0.005


@pytest.mark.parametrize("mode", ["eager", "gpu_busy"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_timing_does_not_replace_summary(tmp_path, runs, mode, value):
    """Reject invalid medians before overwriting a previous summary."""
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("previous summary\n")
    runs["patch-B1"]["results"][0][mode]["median_ms"] = value
    result = summarize(tmp_path, runs)
    assert result.returncode != 0
    assert "finite and positive" in result.stderr
    assert sentinel.read_text() == "previous summary\n"


@pytest.mark.parametrize("base,patch", [(1e308, 1e-308), (1e-308, 1e308)])
def test_derived_overflow_is_rejected(tmp_path, runs, base, patch):
    """Reject ratios that overflow despite finite positive inputs."""
    for label, run in runs.items():
        run["results"][0]["gpu_busy"]["median_ms"] = base if label.startswith("base") else patch
    result = summarize(tmp_path, runs)
    assert result.returncode != 0
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.001, 0.02])
def test_invalid_error_cannot_hide_behind_pass_flag(tmp_path, runs, value):
    """Check each relative error independently of the saved pass flag."""
    runs["patch-B2"]["results"][0]["errors_fp64"]["q"]["relative_frobenius"] = value
    result = summarize(tmp_path, runs)
    assert result.returncode != 0
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fault", ["environment", "helper", "case", "pass_flag", "scheduler", "missing_gradient"])
def test_validation_survives_optimized_python(tmp_path, runs, optimized, fault):
    """Reject incomplete or mismatched records in both Python execution modes."""
    run = runs["patch-B2"]
    if fault == "environment":
        run["torch"] = "different"
    elif fault == "helper":
        run["helper_sha256"] = "different"
    elif fault == "case":
        run["results"].pop()
    elif fault == "pass_flag":
        run["results"][0]["correct_within_one_percent"] = False
    elif fault == "scheduler":
        run["results"][0]["scheduler_records"] = []
    else:
        del run["results"][0]["errors_fp64"]["w"]
    result = summarize(tmp_path, runs, optimized)
    assert result.returncode != 0
    assert not (tmp_path / "summary.json").exists()
