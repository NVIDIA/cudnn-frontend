# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Tile-aligned dense D192 backward cases under Compute Sanitizer memcheck (L1)."""

import pytest

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L1, requires_dsl]


@requires_pre_rubin_blackwell
def test_tile_aligned_exact_allocations_under_compute_sanitizer():
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    sanitizer = shutil.which("compute-sanitizer") or next((str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/compute-sanitizer"))), None)
    if sanitizer is None:
        pytest.skip("compute-sanitizer is not installed")
    files = ["test_sdpa_bwd_d192_accum_sm100.py", "test_sdpa_bwd_d192_fp8_sm100.py"]
    select = "(test_d192_accumulation and dense-bf16) or (test_dense and bf16)"
    cmd = [
        sanitizer,
        "--tool",
        "memcheck",
        "--report-api-errors",
        "no",
        "--error-exitcode",
        "9",
        sys.executable,
        "-m",
        "pytest",
        *files,
        "-k",
        select,
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    env = {**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    run = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-2000:]
    assert "ERROR SUMMARY: 0 errors" in run.stdout, run.stdout[-4000:]
