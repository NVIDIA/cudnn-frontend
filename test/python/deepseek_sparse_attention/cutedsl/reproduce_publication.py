# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the six publication regressions under Compute Sanitizer racecheck."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    package = importlib.util.find_spec("cudnn")
    if package is None or Path(package.origin).resolve() != root / "python/cudnn/__init__.py":
        sys.exit("Install this checkout first: uv pip install -e .")
    os.environ.update(
        CUDNN_TEST_NO_ISOLATION="1",
        CUTE_DSL_LINEINFO="1",
        CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1",
        CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO="0",
    )
    report = root / "publication-results.xml"
    report.unlink(missing_ok=True)
    result = subprocess.run(
        [
            "compute-sanitizer",
            "--tool",
            "racecheck",
            "--error-exitcode",
            "86",
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-m",
            "L0 or L1",
            "-v",
            "-ra",
            f"--junitxml={report}",
            "deepseek_sparse_attention/cutedsl/test_DSA_indexer_forward.py::test_DSA_indexer_forward_per_head_publication",
        ],
        cwd=root / "test/python",
    )
    if result.returncode not in (0, 86):
        sys.exit(result.returncode)
    cases = ET.parse(report).findall(".//testcase")
    if len(cases) != 6 or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        sys.exit("Expected six passing cases with no skips; inspect publication-results.xml.")
    sys.exit(result.returncode)
