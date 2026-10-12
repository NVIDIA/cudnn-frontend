# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only pins of what test/python/conftest.py does for SEVERAL pytest processes on one tree or one GPU (test/AGENTS.md):
the per-run FROST routing directory (no two runs share one -- a session a test starts with a worker's environment included; the
session-start sweep judges a pid only on the host that minted it, so another host's live run on a shared tree is never swept) and the
shared-GPU switch of the memory gate."""

import os
import re
import shutil
import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.L0

_HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def top_conftest(request):
    """The registered module object of test/python/conftest.py -- not ``import conftest``: pytest keeps only the LAST conftest it
    imported under that name in sys.modules, which may be a sub-directory's."""
    path = os.path.join(_HERE, "conftest.py")
    for plugin in request.config.pluginmanager.get_plugins():
        if os.path.abspath(getattr(plugin, "__file__", "") or "") == path:
            return plugin
    pytest.fail("test/python/conftest.py is not among the registered plugins")


def _child_pytest(*args, env=None):
    """A fresh pytest run on this tree (collect-only of this module: cheap, and it runs the conftest's session hooks)."""
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST_") and k != "CUDNN_TEST_RUN_ID"}
    child_env.update(env or {})
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-p", "no:cacheprovider", os.path.basename(__file__), *args]
    return subprocess.Popen(cmd, cwd=_HERE, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _run_id_of(output):
    m = re.search(r"^Test run id: (\S+)$", output, re.M)
    assert m, f"the conftest banner names no run id:\n{output[-2000:]}"
    return m.group(1)


# ---- the per-run routing directory


def _host_tag():
    """How the routing directory's name spells the owning host: this machine's hostname, every character outside ``[A-Za-z0-9.-]``
    replaced by ``-`` (so the name's ``_`` separator never occurs in it)."""
    return re.sub(r"[^A-Za-z0-9.-]", "-", socket.gethostname()) or "unknown-host"


def test_routing_dir_is_keyed_by_this_host_and_this_runs_id(top_conftest):
    base, path, run_id = top_conftest._FROST_ROUTING_BASE, top_conftest._FROST_ROUTING_DIR, top_conftest._FROST_RUN_ID
    assert run_id and path == f"{base}_{_host_tag()}_{run_id}", (base, path, run_id)
    assert top_conftest._ROUTING_HOST == _host_tag() and "_" not in top_conftest._ROUTING_HOST
    assert top_conftest._ROUTING_LEFTOVER_TTL_S == 24 * 60 * 60, "another host's leftover is garbage-collected after a day"
    # the controller exported it before spawning this worker (or this single process minted it): the two agree
    assert os.environ.get("CUDNN_TEST_RUN_ID") == run_id


def test_a_controller_mints_a_fresh_id_even_under_a_parent_pytest(top_conftest, monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("CUDNN_TEST_RUN_ID", "parent_run")  # what a test that spawns pytest hands down
    run_id = top_conftest._frost_routing_run_id()
    assert run_id != "parent_run" and run_id.split("_")[0] == str(os.getpid())
    assert os.environ["CUDNN_TEST_RUN_ID"] == run_id, "exported for the workers this controller spawns next"
    assert top_conftest._frost_routing_run_id() != run_id, "two runs never share an id"


def test_a_worker_inherits_the_controllers_id(top_conftest, monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw7")
    monkeypatch.setenv("CUDNN_TEST_RUN_ID", "4242_cafe0123")
    assert top_conftest._frost_routing_run_id() == "4242_cafe0123"
    monkeypatch.delenv("CUDNN_TEST_RUN_ID")
    monkeypatch.setenv("PYTEST_XDIST_TESTRUNUID", "xdist0uid")
    assert top_conftest._frost_routing_run_id() == "xdist0uid", "xdist's own id when no controller of ours exported one"


# ---- a pytest session a TEST starts inherits its worker's identity -- and must not pass for that worker

_WORKER_IDENTITY = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID", "CUDNN_TEST_RUN_ID")


class _NoWorkerinput:
    """A pytest config without xdist's ``workerinput``: this process is NOT the worker its environment names."""


class _Workerinput:
    workerinput = {"workerid": "gw3"}


def test_a_fresh_session_drops_the_worker_identity_it_inherited(top_conftest, monkeypatch):
    monkeypatch.setattr(top_conftest, "_TRACE_PATH", top_conftest._TRACE_PATH)  # the drop re-derives it; restored after the test
    for name, value in zip(_WORKER_IDENTITY, ("gw3", "4", "parentuid", "4242_parent00")):
        monkeypatch.setenv(name, value)
    assert set(top_conftest._drop_inherited_worker_identity(_NoWorkerinput())) == set(_WORKER_IDENTITY)
    assert not top_conftest._is_xdist_worker() and not any(name in os.environ for name in _WORKER_IDENTITY)
    assert top_conftest._drop_inherited_worker_identity(_NoWorkerinput()) == (), "idempotent: nothing left to drop"
    run_id = top_conftest._frost_routing_run_id()
    assert run_id != "4242_parent00" and run_id.split("_")[0] == str(os.getpid()), "a run of its own"


def test_a_real_xdist_worker_keeps_its_identity(top_conftest, monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.setenv("CUDNN_TEST_RUN_ID", "4242_parent00")
    assert top_conftest._drop_inherited_worker_identity(_Workerinput()) == ()
    assert os.environ["PYTEST_XDIST_WORKER"] == "gw3" and os.environ["CUDNN_TEST_RUN_ID"] == "4242_parent00"
    assert top_conftest._frost_routing_run_id() == "4242_parent00", "the controller's id, as before"


def test_a_child_session_started_with_a_workers_environment_is_its_own_run(top_conftest):
    # What a test that runs pytest in a subprocess WITHOUT scrubbing its environment hands down: this worker's identity and this
    # run's id.  The child is a fresh session all the same -- a run id of its own (its pid), never this run's.
    inherited = dict(
        zip(
            _WORKER_IDENTITY,
            (os.environ.get("PYTEST_XDIST_WORKER", "gw0"), "1", os.environ.get("PYTEST_XDIST_TESTRUNUID", "parentuid"), top_conftest._FROST_RUN_ID),
        )
    )
    child = _child_pytest(env=inherited)
    out = child.communicate(timeout=900)[0]
    assert child.returncode == 0, out[-2000:]
    run_id = _run_id_of(out)
    assert run_id != top_conftest._FROST_RUN_ID and run_id.split("_")[0] == str(child.pid), (run_id, top_conftest._FROST_RUN_ID, child.pid)


def test_two_concurrent_runs_on_one_tree_get_distinct_ids(top_conftest):
    a, b = _child_pytest(), _child_pytest()
    out_a, out_b = a.communicate(timeout=900)[0], b.communicate(timeout=900)[0]
    assert a.returncode == 0 and b.returncode == 0, out_a[-2000:] + out_b[-2000:]
    ids = {_run_id_of(out_a), _run_id_of(out_b), top_conftest._FROST_RUN_ID}
    assert len(ids) == 3, ids
    assert not any(i.split("_")[0] == str(os.getpid()) for i in ids - {top_conftest._FROST_RUN_ID}), "a child run is its own run"


def _exited_pid():
    """The pid of a process that has exited and been reaped: a controller that is gone."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _sweep(top_conftest, monkeypatch):
    """The session-start sweep, as the controller runs it (this process may itself be an xdist worker, which the hook skips)."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    top_conftest.pytest_sessionstart(None)


def _age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_session_start_sweeps_a_dead_runs_directory_and_keeps_a_live_ones(top_conftest):
    base, host = top_conftest._FROST_ROUTING_BASE, _host_tag()
    dead_dir, live_dir = f"{base}_{host}_{_exited_pid()}_deadrun", f"{base}_{host}_{os.getpid()}_liverun"
    # another host's run whose pid is dead HERE, and a run of the previous per-run layout (no host in its name -- it may be another
    # host's live run): neither is this host's to judge, so both stay
    foreign_dir, legacy_dir = f"{base}_otherhost-01_{_exited_pid()}_foreignrun", f"{base}_{_exited_pid()}_legacyrun"
    for d in (dead_dir, live_dir, foreign_dir, legacy_dir):
        os.makedirs(d, exist_ok=True)
    try:
        child = _child_pytest()
        out = child.communicate(timeout=900)[0]
        assert child.returncode == 0, out[-2000:]
        assert not os.path.isdir(dead_dir), "a crashed run's directory (this host, controller pid gone) is swept at the next session start"
        assert os.path.isdir(live_dir), "a live sibling run's directory is never touched"
        assert os.path.isdir(foreign_dir), "another host's directory is never judged by a pid that means nothing here"
        assert os.path.isdir(legacy_dir), "a directory without a host in its name may be another host's live run: kept until the TTL"
    finally:
        for d in (dead_dir, live_dir, foreign_dir, legacy_dir):
            shutil.rmtree(d, ignore_errors=True)


def test_the_sweep_removes_a_dead_run_of_this_host_and_keeps_a_live_one(top_conftest, monkeypatch):
    base, host = top_conftest._FROST_ROUTING_BASE, _host_tag()
    dead_dir, live_dir = f"{base}_{host}_{_exited_pid()}_deadunit", f"{base}_{host}_{os.getpid()}_liveunit"
    os.makedirs(dead_dir, exist_ok=True)
    os.makedirs(live_dir, exist_ok=True)
    _age(live_dir, 3 * 24 * 60 * 60)  # however old it looks: a live pid of this host is a live run
    try:
        _sweep(top_conftest, monkeypatch)
        assert not os.path.isdir(dead_dir) and os.path.isdir(live_dir)
    finally:
        shutil.rmtree(live_dir, ignore_errors=True)
        shutil.rmtree(dead_dir, ignore_errors=True)


def test_the_sweep_keeps_another_hosts_directory_whose_pid_is_dead_here(top_conftest, monkeypatch):
    """A tree on shared storage holds other machines' live runs; ``os.kill(pid, 0)`` answers for this machine's pids only."""
    base = top_conftest._FROST_ROUTING_BASE
    foreign_dir = f"{base}_otherhost-01_{_exited_pid()}_foreignunit"
    os.makedirs(foreign_dir, exist_ok=True)
    try:
        _sweep(top_conftest, monkeypatch)
        assert os.path.isdir(foreign_dir), "another host's run was swept because its pid is dead HERE"
    finally:
        shutil.rmtree(foreign_dir, ignore_errors=True)


def test_the_sweep_keeps_a_previous_layouts_directory_its_pid_may_be_another_hosts_live_run(top_conftest, monkeypatch):
    base = top_conftest._FROST_ROUTING_BASE
    legacy_dir = f"{base}_{_exited_pid()}_legacyunit"  # `.frost_routing_<pid>_<uid>`: no host to compare with
    os.makedirs(legacy_dir, exist_ok=True)
    try:
        _sweep(top_conftest, monkeypatch)
        assert os.path.isdir(legacy_dir), "a pid without a host is not this host's to judge"
    finally:
        shutil.rmtree(legacy_dir, ignore_errors=True)


def test_the_sweep_removes_another_hosts_leftover_only_after_the_ttl(top_conftest, monkeypatch):
    base, ttl = top_conftest._FROST_ROUTING_BASE, top_conftest._ROUTING_LEFTOVER_TTL_S
    fresh_dir, old_dir = f"{base}_otherhost-02_{_exited_pid()}_fresh", f"{base}_otherhost-02_{_exited_pid()}_old"
    old_legacy_dir = f"{base}_{_exited_pid()}_oldlegacy"  # the previous layout's name, past the TTL: a leftover too
    for d in (fresh_dir, old_dir, old_legacy_dir):
        os.makedirs(d, exist_ok=True)
    _age(fresh_dir, ttl - 60 * 60)
    _age(old_dir, ttl + 60 * 60)
    _age(old_legacy_dir, ttl + 60 * 60)
    try:
        _sweep(top_conftest, monkeypatch)
        assert os.path.isdir(fresh_dir), "within the TTL another host's directory stays, whatever its pid means here"
        assert not os.path.isdir(old_dir) and not os.path.isdir(
            old_legacy_dir
        ), "past the TTL a leftover of another host, or without a pid, is garbage-collected"
    finally:
        for d in (fresh_dir, old_dir, old_legacy_dir):
            shutil.rmtree(d, ignore_errors=True)


def test_an_impossible_pid_of_this_host_is_a_dead_run_not_a_crash_of_the_sweep(top_conftest, monkeypatch):
    """``os.kill(pid, 0)`` raises OverflowError past what a pid_t holds; a same-host directory carrying such a number (a crafted or
    corrupted name) must read as a dead run and be swept -- not break every session start on this host until someone removes it."""
    assert top_conftest._pid_alive(2**40) is False and top_conftest._pid_alive(os.getpid()) is True
    base, host = top_conftest._FROST_ROUTING_BASE, _host_tag()
    huge_dir = f"{base}_{host}_{2**40}_hugepid"
    os.makedirs(huge_dir, exist_ok=True)
    try:
        _sweep(top_conftest, monkeypatch)
        assert not os.path.isdir(huge_dir), "a pid no process can have is not a live run"
    finally:
        shutil.rmtree(huge_dir, ignore_errors=True)


def test_the_sweep_never_removes_this_runs_own_directory(top_conftest, monkeypatch):
    own = top_conftest._FROST_ROUTING_DIR
    existed = os.path.isdir(own)
    os.makedirs(own, exist_ok=True)
    _age(own, 3 * 24 * 60 * 60)
    try:
        _sweep(top_conftest, monkeypatch)
        assert os.path.isdir(own), "the live run's own directory, however old it looks"
    finally:
        if not existed:
            shutil.rmtree(own, ignore_errors=True)


# ---- the memory gate's shared-GPU switch


def test_under_xdist_flips_with_the_shared_gpu_switch(top_conftest, monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "1")
    monkeypatch.delenv("CUDNN_TEST_SHARED_GPU", raising=False)
    assert not top_conftest._under_xdist(), "a lone -n1 process: the gate is off"
    monkeypatch.setenv("CUDNN_TEST_SHARED_GPU", "1")
    assert top_conftest._under_xdist(), "CUDNN_TEST_SHARED_GPU=1 arms it for independent processes sharing the GPU"
    monkeypatch.setenv("CUDNN_TEST_SHARED_GPU", "0")
    assert not top_conftest._under_xdist()
    monkeypatch.setenv("PYTEST_XDIST_WORKER_COUNT", "4")
    assert top_conftest._under_xdist(), "several xdist workers of one run: armed as before"


def test_the_armed_gate_announces_itself_once_per_test_process():
    child = _child_pytest(env={"CUDNN_TEST_SHARED_GPU": "1"})
    out = child.communicate(timeout=900)[0]
    assert child.returncode == 0, out[-2000:]
    assert out.count("[mem-gate] armed by CUDNN_TEST_SHARED_GPU=1") == 1, out[-2000:]


# ---- the measured channel (frost_routing.measured -> the controller's terminal summary)


@pytest.mark.parametrize("workers", ["0", "2"], ids=["one_process", "xdist"])
def test_a_measurement_reaches_the_controllers_summary(workers):
    """frost_routing.measured(key, text) is printed under "measured" in the run's terminal summary -- from the single process and,
    aggregated through the per-run directory, from an xdist worker -- while the test that took it PASSES (its captured stdout is
    not in the log).  The child module lives under test/python so this tree's conftest governs it; it is removed afterwards."""
    tag = f"{os.getpid()}_{workers}"
    folder = os.path.join(_HERE, f".measured_pin_{tag}")
    os.makedirs(folder, exist_ok=True)
    module = os.path.join(folder, "test_measured_pin.py")
    try:
        with open(module, "w") as f:
            f.write(
                "import frost_routing\n"
                "import pytest\n"
                "pytestmark = pytest.mark.L0\n"
                "def test_records_a_measurement():\n"
                f"    frost_routing.measured('pin {tag}', 'alpha=crash   beta=planned(3)\\n  gamma=declined')\n"
                "    print('CAPTURED STDOUT OF A PASSING TEST')\n"
            )
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST_") and k != "CUDNN_TEST_RUN_ID"}
        cmd = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "-p", "no:randomly", "-n", workers, module]
        child = subprocess.Popen(cmd, cwd=_HERE, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = child.communicate(timeout=900)[0]
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert child.returncode == 0 and "1 passed" in out, out[-3000:]
    assert re.search(r"^=+ measured =+$", out, re.M), f"no 'measured' section:\n{out[-3000:]}"
    assert f"  pin {tag}: alpha=crash beta=planned(3) gamma=declined" in out, out[-3000:]  # one line, whitespace collapsed
    assert "CAPTURED STDOUT OF A PASSING TEST" not in out, "the channel exists because this line is NOT in the log"
