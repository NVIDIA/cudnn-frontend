# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level cache contracts, with a fake compiler and real filesystem locks."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.skipif(os.name != "posix", reason="POSIX artifact locks")]
_SOURCE = Path(__file__).resolve().parents[4] / "python/cudnn/frost/compiled_cache.py"

_CHILD = r"""
import importlib.util, json, os, sys, time, types
from pathlib import Path
source, directory, rank, key, mode = sys.argv[1:]
root = Path(directory)
spec = importlib.util.spec_from_file_location('cache_under_test', source)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
cc.set_cache_dir(root / 'cache')
cc.environment_manifest = lambda: {'schema': cc._SCHEMA, 'cudnn_frontend': 'test'}
cc._memo_device = lambda: (0, 0)
cc._wrapper_spec_of = lambda *a: {'arg_names': [], 'arg_defaults': [], 'kwonly_names': [], 'kwonly_defaults': {}}
cc._prune_once = lambda *a: None
os.environ.pop(cc._ENV_DISABLE, None)
os.environ[cc._ENV_MEMO] = '0'

def wait(path):
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(.01)

first = True
def load(entry, expected_key, symbol):
    global first
    if first:
        first = False
        (root / ('ready-' + rank)).touch()
        wait(root / 'start')
    record = cc._read_json(entry / cc._ENTRY)
    if record and record['key'] == expected_key and (entry / cc._OBJECT).read_text() == expected_key:
        return 'reloaded'
    return None
cc._try_load = load

class Compiled:
    def export_to_c(self, path, function_name):
        Path(path).write_text(key)

def compile(*args, **kwargs):
    (root / ('compiled-' + rank)).touch()
    if mode == 'fail':
        raise RuntimeError('injected compiler failure')
    if mode == 'hold':
        wait(root / 'finish')
    # Give every simultaneous cold caller a chance to observe the miss.
    time.sleep(.1)
    return Compiled()

cute = types.ModuleType('cutlass.cute')
cute.compile = compile
cutlass = types.ModuleType('cutlass')
cutlass.cute = cute
sys.modules.update({'cutlass': cutlass, 'cutlass.cute': cute})
if mode == 'unavailable':
    (root / 'cache').mkdir(exist_ok=True)
    (root / 'cache' / '.locks').write_text('not a directory')
if mode == 'warm':
    def no_lock(*args):
        raise AssertionError('a cache hit acquired a compiler lock')
    cc._entry_lock = no_lock
result = cc.compile_cached(lambda: None, cache_key=key, options='--enable-tvm-ffi')
(root / ('result-' + rank)).write_text(json.dumps({'result': result, 'stats': cc.stats()}))
"""


def _start(root, rank, key="shared", mode="normal"):
    """Start an independent cache caller with a retained per-process log."""
    log = (root / f"log-{rank}").open("w")
    process = subprocess.Popen([sys.executable, "-c", _CHILD, str(_SOURCE), str(root), str(rank), key, mode], stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return process


def _wait_files(root, prefix, count, processes):
    """Wait for the child-process barrier while detecting early exits."""
    deadline = time.monotonic() + 20
    while len(list(root.glob(prefix + "*"))) < count:
        assert all(p.poll() is None for p in processes), [p.returncode for p in processes]
        assert time.monotonic() < deadline, f"timed out waiting for {prefix}"
        time.sleep(0.01)


def _finish(root, processes, expected=0):
    """Join callers and surface the failing child's diagnostic log."""
    for rank, process in processes:
        assert process.wait(timeout=30) == expected, (root / f"log-{rank}").read_text()


def _cleanup(processes):
    """Reap all children, including those left running after an assertion."""
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_frost_cold_processes_compile_one_artifact(tmp_path):
    """Cold callers share one compilation and a fresh warm caller takes no lock."""
    processes = [_start(tmp_path, rank) for rank in range(4)]
    try:
        _wait_files(tmp_path, "ready-", 4, processes)
        (tmp_path / "start").touch()
        _finish(tmp_path, list(enumerate(processes)))
        assert len(list(tmp_path.glob("compiled-*"))) == 1
        results = [json.loads((tmp_path / f"result-{rank}").read_text()) for rank in range(4)]
        assert all(r["result"] == "reloaded" for r in results)
        assert sum(r["stats"]["misses"] for r in results) == 1
        assert sum(r["stats"]["hits"] for r in results) == 3
        warm = _start(tmp_path, "warm", mode="warm")
        processes.append(warm)
        _finish(tmp_path, [("warm", warm)])
        assert not (tmp_path / "compiled-warm").exists()
    finally:
        _cleanup(processes)


def test_frost_different_keys_can_compile_concurrently(tmp_path):
    """Unrelated artifacts must enter their compilers without waiting for each other."""
    processes = [_start(tmp_path, rank, key=str(rank), mode="hold") for rank in range(2)]
    try:
        _wait_files(tmp_path, "ready-", 2, processes)
        (tmp_path / "start").touch()
        _wait_files(tmp_path, "compiled-", 2, processes)
        (tmp_path / "finish").touch()
        _finish(tmp_path, list(enumerate(processes)))
    finally:
        _cleanup(processes)


@pytest.mark.parametrize("failure", ["fail", "kill"])
def test_frost_compiler_failure_releases_the_artifact_lock(tmp_path, failure):
    """Compiler exceptions and process termination must not strand the next caller."""
    owner = _start(tmp_path, "owner", mode="hold" if failure == "kill" else "fail")
    processes = [owner]
    try:
        _wait_files(tmp_path, "ready-", 1, processes)
        (tmp_path / "start").touch()
        if failure == "kill":
            _wait_files(tmp_path, "compiled-", 1, processes)
            owner.kill()
            owner.wait(timeout=5)
        else:
            _finish(tmp_path, [("owner", owner)], expected=1)
            assert "injected compiler failure" in (tmp_path / "log-owner").read_text()
        retry = _start(tmp_path, "retry")
        processes.append(retry)
        _finish(tmp_path, [("retry", retry)])
        assert (tmp_path / "compiled-retry").exists()
        assert json.loads((tmp_path / "result-retry").read_text())["result"] == "reloaded"
    finally:
        _cleanup(processes)


def test_frost_unavailable_lock_keeps_compilation_functional(tmp_path):
    """An unusable lock directory preserves best-effort compilation and export."""
    process = _start(tmp_path, "fallback", mode="unavailable")
    try:
        _wait_files(tmp_path, "ready-", 1, [process])
        (tmp_path / "start").touch()
        _finish(tmp_path, [("fallback", process)])
        assert json.loads((tmp_path / "result-fallback").read_text())["result"] == "reloaded"
        assert "compiling without a process lock" in (tmp_path / "log-fallback").read_text()
    finally:
        _cleanup([process])
