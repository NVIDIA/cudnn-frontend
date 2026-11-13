# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM90 dense score warp-role synchronization and shared-memory reuse."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from deepseek_sparse_attention.cutedsl import test_DSA_dense_score_recompute as dense_tests


def _require_sm90():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")


_CASES = [
    pytest.param("attention", 0, id="attention-unified"),
    pytest.param("attention", 1, id="attention-split"),
    pytest.param("indexer", 0, id="indexer-unified"),
    pytest.param("indexer", 1, id="indexer-split"),
]


@pytest.mark.L1
@pytest.mark.parametrize("score_type,split_producer", _CASES)
def test_dense_sm90_sync_reference(monkeypatch, score_type, split_producer):
    _require_sm90()
    from cudnn.deepseek_sparse_attention.score_recompute import _interface_sm90

    monkeypatch.setenv("DENSE3WG_USE_WARP_SPLIT_PROD", str(split_producer))
    monkeypatch.setattr(_interface_sm90._dense_score_recompute, "compile_cache", {})
    if score_type == "attention":
        # H128 uses two head tiles; the test includes wholly masked rows and a KV tail.
        dense_tests.test_DSA_dense_attn_sm90_masked_output(heads=128, ratio=1, is_thd=False)
    else:
        # Exercise singleton, short and partial KV shapes through the same cache.
        dense_tests.test_DSA_dense_indexer_sm90_singleton_and_mask_cache(heads=32, ratio=1, singleton_first=True, monkeypatch=monkeypatch)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("score_type,split_producer", _CASES)
@pytest.mark.parametrize("tool", ["synccheck", "racecheck"])
def test_dense_sm90_sync_sanitizer(score_type, split_producer, tool):
    _require_sm90()
    toolkit_sanitizer = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "compute-sanitizer"
    sanitizer = str(toolkit_sanitizer) if toolkit_sanitizer.is_file() and os.access(toolkit_sanitizer, os.X_OK) else shutil.which("compute-sanitizer")
    if sanitizer is None:
        sanitizer = next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/compute-sanitizer"))), None)
    if sanitizer is None:
        return pytest.skip("compute-sanitizer is not installed")
    case = f"{score_type}-{'split' if split_producer else 'unified'}"
    test_timeout = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
    timeout = min(300.0, 0.6 * test_timeout) if test_timeout > 0 else 300.0
    child = f"import faulthandler,sys; faulthandler.dump_traceback_later({timeout * 0.9!r}); import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    cmd = [
        sanitizer,
        "--tool",
        tool,
        "--report-api-errors",
        "no",
        "--error-exitcode",
        "9",
        "--print-limit",
        "10",
        sys.executable,
        "-u",
        "-c",
        child,
        f"{Path(__file__).resolve()}::test_dense_sm90_sync_reference[{case}]",
        "-n0",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        rest, _ = proc.communicate()
        prior = exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else (exc.output or "")
        raise AssertionError(f"{tool} timed out after {timeout}s\n{prior}{rest}") from None
    assert proc.returncode == 0, out[-8000:]
    assert "1 passed" in out, out[-8000:]
    if tool == "synccheck":
        assert "ERROR SUMMARY: 0 errors" in out, out[-8000:]
    else:
        assert "0 errors, 0 warnings" in out, out[-8000:]
