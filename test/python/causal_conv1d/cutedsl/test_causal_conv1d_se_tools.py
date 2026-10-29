# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure accounting and backend attribution for the SE benchmark tools."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    """Classify the backward status-3010 spelling without failing the remaining sweep jobs."""
    tool = _load("controlled_matrix")
    tool.__file__ = str(tool_tree / "controlled_matrix.py")
    cases = [
        dict(family="b2b", dtype="float64", shape=[1, 8, 17], width=3, mixer_width=4, silu=False, directions=["forward", "backward"]),
        dict(family="nwc", dtype="bfloat16", shape=[2, 768, 4096], width=4, silu=True, directions=["forward"]),
    ]
    manifest = tool_tree / "supplemental_workloads.json"
    manifest.write_text(json.dumps({"cases": cases}))
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [tool.__file__, "--supplement", "--output", str(output), "--backend-library", "unused.so"])

    def run(command, **kwargs):
        """Return a backward admission rejection only for fused backward."""
        config = json.loads(Path(command[command.index("--case") + 1]).read_text())["args"]
        fused = "projection_width" in config and command[command.index("--direction") + 1] == "backward"
        return subprocess.CompletedProcess(command, 1 if fused else 0, stdout="", stderr="backend backward status 3010" if fused else "")

    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(SystemExit) as error:
        tool.main()
    assert error.value.code == 0
    records = json.loads((output / "summary.json").read_text())["results"]
    assert [r["status"] for r in records] == ["pass", "backend_unsupported", "pass"]


@pytest.mark.parametrize("selection", ["missing", "different", "ambiguous", "replaced", "matching_symlink"])
def test_loaded_backend_library_identity(selection, tmp_path, monkeypatch):
    """Reject false attribution, accepting an alias only when the loaded file identity matches."""
    import hashlib
    import os

    tool = _load("backend")
    loaded = tmp_path / "libcudnn_ext.so.9"
    loaded.write_bytes(b"loaded backend")
    other = tmp_path / "other/libcudnn_ext.so.9"
    other.parent.mkdir()
    other.write_bytes(b"other backend")
    alias = tmp_path / "backend.so"
    alias.symlink_to(loaded)
    stat = loaded.stat()
    device = f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}"
    inode = stat.st_ino + (selection == "replaced")
    mappings = f"0000-1000 r-xp 00000000 {device} {inode} {loaded}\n"
    if selection == "ambiguous":
        mappings += f"1000-2000 r-xp 00000000 {device} {other.stat().st_ino} {other}\n"
    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs: mappings if str(path) == "/proc/self/maps" else read_text(path, *args, **kwargs))
    requested = tmp_path / "missing.so" if selection == "missing" else (other if selection == "different" else alias)
    if selection == "matching_symlink":
        identity = tool.verify_loaded_library(requested, 92900)
        assert identity == dict(
            requested_path=str(loaded),
            loaded_path=str(loaded),
            sha256=hashlib.sha256(loaded.read_bytes()).hexdigest(),
            size_bytes=14,
            frontend_backend_version=92900,
        )
    else:
        with pytest.raises(FileNotFoundError if selection == "missing" else RuntimeError):
            tool.verify_loaded_library(requested, 92900)
