# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Tile-aligned dense D192 backward cases under Compute Sanitizer memory/sync checks (L1)."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L1, requires_dsl]

_FILES = ["test_sdpa_bwd_d192_accum_sm100.py", "test_sdpa_bwd_d192_fp8_sm100.py"]
_SELECT = "(test_d192_accumulation and dense-bf16) or (test_dense and bf16)"


def _sanitizer():
    found = shutil.which("compute-sanitizer")
    if found is None:
        found = next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/compute-sanitizer"))), None)
    return found


def run_under_sanitizer(timeout, files=_FILES, select=_SELECT, tool="memcheck"):
    """Run a sanitizer over a pytest selection in a child process group; the child dumps its own stack before the parent deadline."""
    child = f"import faulthandler, sys\nfaulthandler.dump_traceback_later({timeout * 0.9!r})\nimport pytest\nsys.exit(pytest.main(sys.argv[1:]))\n"
    cmd = [_sanitizer(), "--tool", tool, "--report-api-errors", "no", "--error-exitcode", "9", sys.executable, "-u", "-c", child]
    cmd += [*files, "-k", select, "-p", "no:cacheprovider", "-q"]
    env = {**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    proc = subprocess.Popen(
        cmd, cwd=Path(__file__).resolve().parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        rest_out, rest_err = proc.communicate()

        def text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")

        raise AssertionError(
            f"sanitizer child timed out after {timeout}s\n--- child stdout ---\n{text(exc.stdout) + rest_out}\n--- child stderr / stack ---\n{text(exc.stderr) + rest_err}"
        ) from None
    return proc.returncode, out, err


@requires_pre_rubin_blackwell
@pytest.mark.parametrize("tool", ["memcheck", "synccheck"])
def test_tile_aligned_exact_allocations_under_compute_sanitizer(tool):
    if _sanitizer() is None:
        pytest.skip("compute-sanitizer is not installed")
    timeout = min(900.0, 0.6 * float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500")))
    code, out, err = run_under_sanitizer(timeout, tool=tool)
    assert code == 0, out[-4000:] + err[-2000:]
    assert "ERROR SUMMARY: 0 errors" in out, out[-4000:]
