# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crash identity survives an outer timeout without changing pytest outcomes."""

import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.skipif(os.name != "posix", reason="isolated process-group controls require POSIX")]
_FOLLOWING = "[crash-report-control] following test started"


def _prepare(tmp_path, kind, stall):
    # Load the actual reporting functions without starting the GPU fixtures.
    # Absence of the new hook deliberately remains runnable for the RED control.
    source = Path(__file__).with_name("conftest.py").read_text()
    names = {"_log_to_real_stderr", "pytest_handlecrashitem"}
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert any(node.name == "_log_to_real_stderr" for node in functions)
    plugin = """import os, pytest
_stderr_fd = None
def pytest_configure(config):
    global _stderr_fd
    _stderr_fd = os.dup(2)
"""
    plugin += "\n\n".join(ast.unparse(node) for node in functions)
    (tmp_path / "conftest.py").write_text(plugin)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_child.py").write_text(f"""import faulthandler, os, pytest, time
import conftest
@pytest.mark.parametrize("kind", [{kind!r}])
def test_crash(kind):
    if kind == "hard-exit":
        os._exit(17)
    if kind == "watchdog":
        faulthandler.dump_traceback_later(0.2, exit=True, file=conftest._stderr_fd)
        time.sleep(60)
def test_following():
    if {stall!r}:
        conftest._log_to_real_stderr({_FOLLOWING!r})
        time.sleep(60)
""")


def _run(tmp_path, *, distributed, stall):
    if distributed:
        pytest.importorskip("xdist")
    # A pytest launched from an xdist worker is a fresh controller, not that worker.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_") and k not in ("CUDNN_TEST_RUN_ID", "CUDNN_TEST_TRACE_DIR")}
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    args = [sys.executable, "-m", "pytest", "-c", str(tmp_path / "pytest.ini"), "-q", "-p", "no:cacheprovider"]
    if distributed:
        args += ["-p", "xdist.plugin", "-n", "1", "--max-worker-restart=2"]
    args += [str(tmp_path / "test_child.py")]
    path = tmp_path / "child.log"
    following_started = False
    with path.open("w") as log:
        child = subprocess.Popen(args, cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while child.poll() is None and time.monotonic() < deadline:
                if stall and _FOLLOWING in path.read_text():
                    following_started = True
                    break
                time.sleep(0.05)
            completed = child.poll() is not None
        finally:
            if child.poll() is None:
                # Stop only this deliberately stalled child and its own workers.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
    output = path.read_text()
    assert following_started if stall else completed, output
    return output, child.returncode


@pytest.mark.parametrize("kind", ["hard-exit", "watchdog"])
def test_crash_item_is_visible_before_the_session_finishes(tmp_path, kind):
    _prepare(tmp_path, kind, stall=True)
    output, _ = _run(tmp_path, distributed=True, stall=True)
    assert "node down: Not properly terminated" in output, output
    assert f"[crash-guard] xdist crash item: test_child.py::test_crash[{kind}]" in output, output
    assert "short test summary info" not in output, "the identification must not depend on a terminal summary"


def test_crash_reporting_preserves_the_failure_and_following_pass(tmp_path):
    _prepare(tmp_path, "hard-exit", stall=False)
    output, code = _run(tmp_path, distributed=True, stall=False)
    assert code == 1 and "1 failed, 1 passed" in output, output
    assert output.count("[crash-guard] xdist crash item:") == 1, output


def test_crash_hook_is_optional_without_xdist(tmp_path):
    _prepare(tmp_path, "pass", stall=False)
    output, code = _run(tmp_path, distributed=False, stall=False)
    assert code == 0 and "2 passed" in output, output
    assert "[crash-guard] xdist crash item:" not in output, output
