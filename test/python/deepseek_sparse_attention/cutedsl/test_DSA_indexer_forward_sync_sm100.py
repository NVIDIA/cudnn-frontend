# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check warp-role shared-memory publication in the SM100-family block-scaled indexer forward."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("tool", ["racecheck", "synccheck"])
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("test_DSA_indexer_forward_wrapper_block_scaled_matches_dequant_reference[qh4_b3_key_tail-mxfp8]", id="mxfp8-h4-key-tail"),
        pytest.param("test_DSA_indexer_forward_wrapper_block_scaled_matches_dequant_reference[qh4_b3_key_tail-mxfp4]", id="mxfp4-h4-empty-tail"),
        pytest.param("test_DSA_indexer_forward_wrapper_block_scaled_matches_dequant_reference[qh64_ratio4_persistent_mixed_blocks-mxfp4]", id="mxfp4-h64-persistent"),
    ],
)
def test_indexer_forward_sm100_blockscaled_sanitizer(case, tool):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    toolkit_sanitizer = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "compute-sanitizer"
    sanitizer = str(toolkit_sanitizer) if toolkit_sanitizer.is_file() and os.access(toolkit_sanitizer, os.X_OK) else shutil.which("compute-sanitizer")
    if sanitizer is None:
        sanitizer = next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/compute-sanitizer"))), None)
    if sanitizer is None:
        pytest.skip("compute-sanitizer is not installed")
    test_timeout = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
    timeout = min(300.0, 0.6 * test_timeout) if test_timeout > 0 else 300.0
    child = f"import faulthandler,sys; faulthandler.dump_traceback_later({timeout * 0.9!r}); import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    target = Path(__file__).with_name("test_DSA_indexer_forward.py").resolve()
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
        f"{target}::{case}",
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
    summary = "ERROR SUMMARY: 0 errors" if tool == "synccheck" else "0 errors, 0 warnings"
    assert summary in out, out[-8000:]
