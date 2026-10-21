# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker failure handling without a GPU."""

import json
import subprocess
from pathlib import Path

import pytest

from benchmark.qsa import benchmark_qsa as benchmark

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("exit_code", [0, -9, "timeout"])
def test_partial_worker_result_preserves_remaining_cases(tmp_path, monkeypatch, exit_code):
    output = tmp_path / "results.json"
    source = tmp_path / "miles.py"
    source.touch()
    monkeypatch.setattr(benchmark, "CASES", benchmark.CASES[:2])
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark_qsa.py", "--miles-source", str(source), "--output", str(output)])
    calls = []

    def run(command, stdout, timeout, **kwargs):
        calls.append(command)
        result = Path(command[command.index("--output") + 1])
        stdout.write("worker log\n")
        if len(calls) == 1:
            result.write_text('{"status":')
            if exit_code == "timeout":
                raise subprocess.TimeoutExpired(command, timeout)
            return subprocess.CompletedProcess(command, exit_code)
        result.write_text(json.dumps(dict(status="passed", input_sha256="inputs", saved_state_sha256="saved")))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert benchmark.main() == 1
    report = json.loads(output.read_text())
    assert len(calls) == 4 and len(report["cases"]) == 2
    failed = report["cases"][0]["implementations"]["native"]
    assert failed["status"] == "failed" and failed["exit_code"] == exit_code
    assert "JSON" in failed["error"]
    assert Path(failed["log_path"]).read_text() == "worker log\n"
    assert report["cases"][0]["implementations"]["miles"]["status"] == "passed"
    assert all(item["status"] == "passed" for item in report["cases"][1]["implementations"].values())
