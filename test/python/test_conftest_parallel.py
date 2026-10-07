# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only pins of what test/python/conftest.py does for SEVERAL pytest processes on one tree or one GPU (test/AGENTS.md):
the per-run FROST routing directory (no two runs share one -- a session a test starts with a worker's environment included) and the
shared-GPU switch of the memory gate."""

import os
import re
import shutil
import subprocess
import sys

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


def test_routing_dir_is_keyed_by_this_runs_id(top_conftest):
    base, path, run_id = top_conftest._FROST_ROUTING_BASE, top_conftest._FROST_ROUTING_DIR, top_conftest._FROST_RUN_ID
    assert run_id and path == f"{base}_{run_id}", (base, path, run_id)
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


def test_session_start_sweeps_a_dead_runs_directory_and_keeps_a_live_ones(top_conftest):
    base = top_conftest._FROST_ROUTING_BASE
    dead_dir, live_dir = f"{base}_{_exited_pid()}_deadrun", f"{base}_{os.getpid()}_liverun"
    os.makedirs(dead_dir, exist_ok=True)
    os.makedirs(live_dir, exist_ok=True)
    try:
        child = _child_pytest()
        out = child.communicate(timeout=900)[0]
        assert child.returncode == 0, out[-2000:]
        assert not os.path.isdir(dead_dir), "a crashed run's directory (controller pid gone) is swept at the next session start"
        assert os.path.isdir(live_dir), "a live sibling run's directory is never touched"
    finally:
        shutil.rmtree(live_dir, ignore_errors=True)
        shutil.rmtree(dead_dir, ignore_errors=True)


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
