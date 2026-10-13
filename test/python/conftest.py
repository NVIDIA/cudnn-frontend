# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# Caps peak GPU memory across long pytest-xdist runs (e.g. test_mhas_v2 ~2.5k
# configs in one worker). Must precede any torch import (including the
# transitive one via transformer_engine below) -- PyTorch reads this env var
# once when its CUDA allocator initializes.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6",
)

# The JAX interop tests (**/test_*_jax.py) initialize XLA in the same pytest
# process as the torch suites; XLA's default 75%-of-GPU preallocation starves later
# torch tests of memory (CUDA_ERROR_OUT_OF_MEMORY at kernel-compile time). Must be set
# before jax initializes its backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Transformer Engine is used only for PyTorch quantization references. JAX
# interop tests do not need its optional JAX extension, even when JAX is installed.
os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")

import faulthandler
import gc
import re
import socket
import subprocess
import sys
import time
import uuid
import weakref
import pytest

# Import TransformerEngine BEFORE cudnn to avoid library loading conflicts
# TE requires specific CUDA library versions that conflict if cudnn is loaded first
try:
    import transformer_engine
except (ImportError, OSError):
    pass

import cudnn

# Caller-owned DSA plan contracts: host-sync and compile-allocation detectors.
from rule8_detector import _execute_never_blocks_the_host, compile_allocates_nothing  # noqa: E402,F401

# cudart via cuda-python instead of torch: torch is optional, and startup must
# not create a CUDA context so per-worker CUDA_VISIBLE_DEVICES routing works.
import cuda.bindings.driver as cuda_driver
import cuda.bindings.runtime as cudart


def _cudart_call(fn, *args):
    err, *rest = fn(*args)
    if int(err) != 0:
        raise RuntimeError(f"{fn.__name__} failed: {cudart.cudaGetErrorString(err)[1].decode()}")
    return rest[0] if len(rest) == 1 else tuple(rest)


# fmt: off

# =================== Crash isolation =====================
# Three failure modes are unrecoverable in-process and make every later test lie:
#   - a segfault or abort kills the interpreter;
#   - a faulting kernel (illegal memory access, device-side assert, ...) poisons
#     the CUDA context, so every later CUDA call -- fixture setup included --
#     returns the same sticky error;
#   - a hung test, CPU or GPU, blocks the session forever.
# So the tests run inside a pytest-xdist worker (one is injected below when the
# caller did not ask for -n), and the worker is killed as soon as its context is
# dead or a test overruns its deadline. The controller reports the offending
# test, replaces the worker, and the rest of the session runs in a fresh process
# with fresh fixtures. Escape hatches: -s / --pdb / -n<N> / CUDNN_TEST_NO_ISOLATION=1.
#
# Caveat for GPU hangs: killing the process does not stop a kernel that is still
# running; the driver keeps that context alive until the kernel finishes, and
# the next worker can block behind it. Nothing short of a GPU reset fixes that.

_TEST_TIMEOUT_S = float(os.environ.get("CUDNN_TEST_TIMEOUT", "1500"))
_xdist_controller = False
_stderr_fd = None  # dup of the real stderr, taken while pytest's capture is suspended

# Per-worker journal of started/finished tests, off unless CUDNN_TEST_TRACE_DIR
# names a writable directory. See _trace() for what it is for.
_TRACE_DIR = os.environ.get("CUDNN_TEST_TRACE_DIR")


def _trace_path():
    return os.path.join(_TRACE_DIR, f"worker-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}.trace") if _TRACE_DIR else None


_TRACE_PATH = _trace_path()


def _is_xdist_worker():
    return os.environ.get("PYTEST_XDIST_WORKER") is not None


# What an xdist worker carries in its environment -- and hands down to every process it starts.
_WORKER_IDENTITY_VARS = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID", "CUDNN_TEST_RUN_ID")


def _drop_inherited_worker_identity(config):
    # A pytest session that a TEST starts (a subprocess run from inside an xdist
    # worker) inherits the worker's PYTEST_XDIST_* and the run's CUDNN_TEST_RUN_ID,
    # and would pass for that worker in every check of this file: skip the
    # isolation worker, reuse the parent run's routing directory and overwrite
    # its worker file at session finish, suppress its own terminal summary.
    # xdist marks a real worker itself -- config.workerinput is set in the worker
    # bootstrap, before any hook runs -- so a process that carries the variables
    # WITHOUT it is a fresh session: drop them, and it is a run of its own (a
    # fresh run id, its own isolation worker). Idempotent; returns the names
    # dropped. Called from the first two hooks that see the config.
    global _TRACE_PATH
    if getattr(config, "workerinput", None) is not None:
        return ()
    dropped = tuple(name for name in _WORKER_IDENTITY_VARS if name in os.environ)
    for name in dropped:
        del os.environ[name]
    if dropped:
        _TRACE_PATH = _trace_path()  # was named after the parent's worker
    return dropped


def _log_to_real_stderr(msg):
    # print(..., file=sys.__stderr__) does NOT reach the CI log from an xdist
    # worker: pytest's global capture replaces fd 2 for the whole worker
    # process, so sys.__stderr__ writes land in a capture buffer that is thrown
    # away when the worker dies -- which is exactly when these messages matter.
    # _stderr_fd is the pre-capture dup, the same channel pytest's own
    # faulthandler plugin writes tracebacks to for this very reason. Verified
    # both ways against a worker that prints on both channels and then _exits:
    # only the duped fd survives.
    fd = _stderr_fd if _stderr_fd is not None else 2
    try:
        os.write(fd, (msg + "\n").encode("utf-8", errors="replace"))
    except OSError:
        pass  # a diagnostic must never be what breaks the run


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    # Runs before xdist's own tryfirst hook, which expands numprocesses into tx
    # specs. Workers re-enter this hook with numprocesses reset to None; skip
    # there, or a worker spawns workers of its own. A session that merely
    # INHERITED a worker's environment is not a worker: drop the identity first.
    _drop_inherited_worker_identity(config)
    opt = config.option
    if _is_xdist_worker() or os.environ.get("CUDNN_TEST_NO_ISOLATION"):
        return
    if not hasattr(opt, "numprocesses"):
        return  # pytest-xdist not installed
    if opt.maxworkerrestart is None:
        opt.maxworkerrestart = 100000  # xdist's default is 4x the worker count; every crashing test costs one
    if opt.numprocesses is not None or opt.tx or opt.collectonly:
        return
    if opt.capture == "no" or opt.usepdb:
        return  # interactive run: keep the tests in this process
    opt.numprocesses = 1


def _dead_cuda_context():
    # A sticky context error is returned by every later CUDA call, so one
    # synchronize is both the cheapest and the most complete probe; it also
    # surfaces async faults the test itself swallowed.
    # Probe only if this thread already holds a context; cudaDeviceSynchronize
    # would otherwise create one in a worker that never touched the GPU.
    err, ctx = cuda_driver.cuCtxGetCurrent()
    if err == cuda_driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return None
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        return cuda_driver.cuGetErrorString(err)[1].decode()
    if int(ctx) == 0:
        return None
    (err,) = cudart.cudaDeviceSynchronize()
    if int(err) == 0:
        return None
    return cudart.cudaGetErrorString(err)[1].decode()


def _trace(event, nodeid):
    # Opt-in per-worker journal of what each worker started and finished.
    #
    # When a worker dies, all the log says is "node down"; the journal survives
    # it and distinguishes the two deaths that otherwise look identical:
    #   - last line is a START, so the worker died *inside* that test (signal);
    #   - last line is DEAD-CONTEXT, so the crash guard below killed it after
    #     the test, and the line carries the CUDA error that condemned it.
    # It also pins down which test is at fault independently of xdist's own
    # attribution, which is worth having precisely because that attribution is
    # easy to reason about wrongly.
    # The controller replays every worker's reports, so logstart fires there
    # too, while logfinish returns early -- its journal would be nothing but
    # START lines and would read as a worker that died on every test.
    if _TRACE_PATH is None or _xdist_controller:
        return
    try:
        with open(_TRACE_PATH, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {event} {nodeid}\n")
    except OSError:
        pass  # a diagnostic must never be what breaks the run


def pytest_runtest_logstart(nodeid, location):
    _trace("START ", nodeid)
    # faulthandler's watchdog is a C thread that needs no GIL, so it fires even
    # while the main thread sits in a CUDA driver call; a Python thread or
    # SIGALRM handler would never get to run there. exit=True dumps every
    # thread's stack to the real stderr and then _exit()s the worker.
    if _TEST_TIMEOUT_S > 0 and not _xdist_controller:
        faulthandler.dump_traceback_later(_TEST_TIMEOUT_S, exit=True, file=_stderr_fd)


@pytest.hookimpl(trylast=True)
def pytest_runtest_logfinish(nodeid, location):
    if _xdist_controller:
        return  # only replays worker reports; ran no CUDA work of its own
    # The probe blocks on any kernel the test left running, so it runs with the
    # watchdog still armed; disarm only once it has returned.
    error = _dead_cuda_context()
    faulthandler.cancel_dump_traceback_later()
    if error is None:
        _trace("ok    ", nodeid)
        return
    msg = f"[crash-guard] {nodeid}: CUDA context is unusable ({error})"
    _trace(f"DEAD-CONTEXT ({error}) ", nodeid)
    if not _is_xdist_worker():
        pytest.exit(msg, returncode=1)  # no supervisor to restart us: stop cleanly instead of cascading
    _log_to_real_stderr(f"{msg}; killing the worker")
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.5)  # let this test's report drain to the xdist controller first
    os._exit(os.EX_SOFTWARE)

# =================== CUDA Graph lifetimes =====================
# A captured CUDAGraph that a test leaves in a reference cycle is destroyed
# whenever the cycle collector next runs -- possibly inside a later test's
# capture, which that destruction invalidates ("operation failed due to a
# previous error during capture", test/AGENTS.md "CUDA Graph test lifetimes").
# After a test that leaves a captured, never-reset graph alive, collect here,
# between tests where destruction is harmless, and fail the test that leaked it.
_new_cuda_graphs = []


def _track_cuda_graphs():
    try:
        import torch
    except ImportError:
        return
    cls = torch.cuda.CUDAGraph
    init, capture_end, reset = cls.__init__, cls.capture_end, cls.reset

    def tracked_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _new_cuda_graphs.append(weakref.ref(self))

    def tracked_capture_end(self, *args, **kwargs):
        capture_end(self, *args, **kwargs)
        self._fe_captured = True

    def tracked_reset(self, *args, **kwargs):
        reset(self, *args, **kwargs)
        self._fe_captured = False

    cls.__init__, cls.capture_end, cls.reset = tracked_init, tracked_capture_end, tracked_reset


def _collect_leaked_cuda_graphs():
    """Collect this test's captured, never-reset graphs now; return how many sat in a reference cycle."""
    refs = list(_new_cuda_graphs)
    _new_cuda_graphs.clear()
    captured = [ref for ref in refs if getattr(ref(), "_fe_captured", False)]
    if not captured:
        return 0
    gc.collect()
    return sum(ref() is None for ref in captured)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call" and report.failed:
        item._fe_call_failed = True
    return report


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_teardown(item, nextitem):
    try:
        result = yield
    except BaseException:
        _collect_leaked_cuda_graphs()  # still collect here; the teardown's own error is the report
        raise
    leaked = _collect_leaked_cuda_graphs()
    # A failed call's traceback often holds the graph in a cycle; its failure is the report.
    if leaked and not getattr(item, "_fe_call_failed", False):
        raise AssertionError(
            f"{leaked} captured CUDA graph(s) outlived this test in a reference cycle without reset(); a later "
            "capture would have been invalidated by their destruction. Reset test-owned graphs in a finally block "
            '(test/AGENTS.md "CUDA Graph test lifetimes").'
        )
    return result


# =================== Test tiers: smoke / nightly_only =====================
# Nested selections of the L0 matrix for LOCAL runs. CI keeps `-m L0`; no list
# changes what CI runs.
#   SMOKE    -m smoke                       one cell per code path (engine row x dtype x mask arm x layout), minutes on one GPU
#   FULL     -m "L0 and not nightly_only"   the functional matrix minus codegen pins that have a numerics twin and B/H/seed twins
#   NIGHTLY  -m L0 (or every level)         everything
# The two markers are applied HERE from the committed node-id lists in tiers/
# (smoke_<arch>.txt per compute capability, nightly_only.txt shared), so no
# test file carries a tier and moving a cell is a one-line list edit.
# tiers/README.md states the rules a cell must satisfy to be listed;
# test_tiers.py asserts every listed id still collects, so a renamed test
# cannot silently drop out of a tier.

_TIERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiers")


def _tier_ids(name):
    # The node ids in tiers/<name>: one per line, blank lines and `#` comments ignored; empty when the file is absent.
    try:
        with open(os.path.join(_TIERS_DIR, name)) as fh:
            return {line.strip() for line in fh if line.strip() and not line.lstrip().startswith("#")}
    except OSError:
        return set()


def _tier_arch_tag():
    # `cc<major><minor>` of device 0 -- the GPU this process runs on (CUDA_VISIBLE_DEVICES applied), e.g. `cc107`.
    # CUDNN_TEST_TIER_ARCH=cc<NNN> overrides it, to list another arch's smoke tier from any host.
    override = os.environ.get("CUDNN_TEST_TIER_ARCH")
    if override:
        return override
    try:
        prop = _cudart_call(cudart.cudaGetDeviceProperties, 0)
    except Exception:
        return None
    return f"cc{prop.major}{prop.minor}"


def _apply_tier_markers(config, items):
    arch = _tier_arch_tag()
    smoke_list = f"smoke_{arch}.txt" if arch else None
    smoke = _tier_ids(smoke_list) if smoke_list else set()
    nightly_only = _tier_ids("nightly_only.txt")
    if "smoke" in (getattr(config.option, "markexpr", "") or "") and not (smoke_list and os.path.isfile(os.path.join(_TIERS_DIR, smoke_list))):
        # A capability without a list has an EMPTY smoke tier: `-m smoke` would exit 5 with no tests and no word why.  One line,
        # once per run (the xdist workers collect, the controller does not; a single process is its own gw0).
        if not _is_xdist_worker() or os.environ.get("PYTEST_XDIST_WORKER") == "gw0":
            _log_to_real_stderr(
                f"[tiers] no SMOKE list for this GPU ({smoke_list or 'compute capability unknown'} is not in {_TIERS_DIR}): "
                "-m smoke selects NOTHING here; CUDNN_TEST_TIER_ARCH=cc<NNN> applies another arch's list (tiers/README.md)"
            )
    for item in items:
        if item.nodeid in smoke:
            item.add_marker(pytest.mark.smoke)
        if item.nodeid in nightly_only:
            item.add_marker(pytest.mark.nightly_only)


# =================== JAX/XLA target gate =====================
# XLA cannot compile for every GPU these tests run on, and when it cannot it
# does not raise -- it prints
#
#   LLVM Fatal Error ... PTX version 9.0 does not support target 'sm_107a'.
#   Minimum required PTX version is 9.4.
#
# and calls exit(1) from C. On Rubin (sm_107a) that happens on the very first
# XLA compilation, so every test_*_jax.py test takes the interpreter down with
# it: jaxlib 0.11.1 -- the newest release -- emits PTX 9.0 from its embedded
# LLVM, and no local CUDA or ptxas can change that.
#
# Under pytest-xdist the death is completely silent. The fatal text goes to
# fd 2, which pytest's capture has replaced, so the buffer dies with the worker
# and all the controller reports is "node down: Not properly terminated"
# against an arbitrary JAX test. That cost a long hunt through OOM and
# CUDA-context theories; skipping loudly is the point of this gate.
#
# There is no way to ask XLA whether it can target this device short of
# compiling something, and a failed attempt is unrecoverable in-process, so the
# probe runs once in a subprocess. It is a capability check rather than an
# arch/version blocklist so that the tests start running again by themselves on
# the first jaxlib that can emit PTX 9.4.

_JAX_PROBE = "import jax, jax.numpy as jnp; jax.jit(lambda x: x + 1)(jnp.ones(1)).block_until_ready()"
_jax_can_compile = None


def _jax_compiles_for_this_device():
    global _jax_can_compile
    if _jax_can_compile is not None:
        return _jax_can_compile
    try:
        import jax  # noqa: F401
    except ImportError:
        _jax_can_compile = True  # nothing to gate; the tests importorskip themselves
        return _jax_can_compile
    try:
        done = subprocess.run([sys.executable, "-c", _JAX_PROBE], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        _jax_can_compile = True  # cannot tell -- let the tests run and report for themselves
        return _jax_can_compile
    _jax_can_compile = done.returncode == 0
    if not _jax_can_compile:
        reason = next(
            (ln.strip() for ln in reversed((done.stderr or "").splitlines()) if "PTX" in ln or "Fatal" in ln),
            f"probe exited {done.returncode}",
        )
        _log_to_real_stderr(f"[jax-gate] XLA cannot compile for this GPU, skipping every *_jax.py test: {reason}")
    return _jax_can_compile


def _skip_jax_tests_when_xla_cannot_compile(items):
    if not any(item.fspath.basename.endswith("_jax.py") for item in items):
        return  # do not pay for the probe on runs with no JAX tests
    if _jax_compiles_for_this_device():
        return
    skip = pytest.mark.skip(reason="XLA cannot compile for this GPU (jaxlib emits PTX 9.0; sm_107a needs >= 9.4)")
    for item in items:
        if item.fspath.basename.endswith("_jax.py"):
            item.add_marker(skip)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # The tier markers must be ON the items before the built-in mark plugin's impl of this hook deselects by `-m`
    # (`_pytest/mark/__init__.py::deselect_by_mark`).  A plain conftest impl runs first today only because plain impls are
    # called last-registered-first and that impl carries no tryfirst; tryfirst makes the order explicit on every pytest
    # (pinned by test_tiers.py).  The JAX skip markers are order-independent.
    _apply_tier_markers(config, items)
    _skip_jax_tests_when_xla_cannot_compile(items)


# =================== GPU memory gate (pytest-xdist, shared GPUs) =====================
# Several xdist workers share one GPU. A memory-hungry test in one worker (a
# large sdpa bwd config can legitimately hold >12 GiB of a 16 GiB device) makes
# unrelated tests in the other workers fail with OutOfMemoryError on tiny
# allocations. Two mitigations, both no-ops in single-process runs:
#   1. Before each test, wait (bounded) until a floor of device memory is free,
#      so tests do not start while a sibling worker holds the GPU.
#   2. If a test still hits torch.OutOfMemoryError, wait for the pressure to
#      clear and re-run it once (pytest_runtest_call hookwrapper below).
# Teardown returns this worker's cached blocks to the driver after every test
# (effective because expandable_segments is set above), so a worker's
# high-water mark is not held against the siblings for the rest of the session.
#
# The siblings need not be xdist workers of THIS run. K independent pytest
# processes on one GPU -- each with the injected -n1 worker, so every one of
# them sees PYTEST_XDIST_WORKER_COUNT == 1 and the gate OFF -- hold their
# allocator high-water marks against each other in exactly the same way.
# CUDNN_TEST_SHARED_GPU=1 (set by whoever launches several processes per GPU)
# arms the gate for them: the floor wait, the empty_cache() and the one OOM
# retry, nothing else changes.

_MEM_GATE_FRACTION = float(os.environ.get("CUDNN_TEST_MEM_GATE_FRACTION", "0.2"))
_MEM_GATE_TIMEOUT_S = float(os.environ.get("CUDNN_TEST_MEM_GATE_TIMEOUT", "30"))


def _shared_gpu():
    return os.environ.get("CUDNN_TEST_SHARED_GPU") == "1"


def _under_xdist():
    # Several xdist workers of this run share the GPU -- or, with CUDNN_TEST_SHARED_GPU=1, other pytest processes do.
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")) > 1 or _shared_gpu()


def _torch_empty_cache():
    # No-op unless torch (and thus its caching allocator) is loaded.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.cuda.empty_cache()


def _wait_for_free_gpu_memory(context):
    # Best effort: proceed after the timeout even if the floor was not reached,
    # so a worker can never deadlock the run; the allocation itself then either
    # succeeds or fails with the usual OOM.
    free, total = _cudart_call(cudart.cudaMemGetInfo)
    floor = _MEM_GATE_FRACTION * total
    if free >= floor:
        return
    _torch_empty_cache()
    deadline = time.monotonic() + _MEM_GATE_TIMEOUT_S
    waited = False
    while time.monotonic() < deadline:
        free, _ = _cudart_call(cudart.cudaMemGetInfo)
        if free >= floor:
            break
        waited = True
        time.sleep(2)
    if waited:
        _log_to_real_stderr(
            f"[mem-gate] {context}: waited for GPU memory "
            f"(free {free / 2**30:.2f} GiB, floor {floor / 2**30:.2f} GiB)"
        )


def _is_cuda_oom(exc):
    # torch.OutOfMemoryError only exists on newer torch; the cuda alias is old.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)


@pytest.fixture(autouse=True)
def _gpu_memory_gate(request):
    if _under_xdist():
        _wait_for_free_gpu_memory(request.node.name)
    yield
    if _under_xdist():
        _torch_empty_cache()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    if not _under_xdist() or outcome.excinfo is None or not _is_cuda_oom(outcome.excinfo[1]):
        return
    # OOM under xdist is usually transient sibling-worker pressure, not a
    # property of this test: release our cache, wait for the device, retry once.
    _log_to_real_stderr(f"[mem-gate] {item.nodeid}: OOM under xdist, retrying once")
    _torch_empty_cache()
    _wait_for_free_gpu_memory(item.nodeid)
    try:
        item.runtest()
    except Exception:
        return  # keep the original OOM report
    outcome.force_result(None)


# =================== Fixtures =====================
@pytest.fixture(scope="session", autouse=True)
def cudnn_handle():
    try:
        _ = cudnn.backend_version()
    except Exception:
        # cuDNN not available; do not create a handle so tests not requiring it can run
        yield None
        return
    
    # Create CUDA stream and graph objects
    stream = _cudart_call(cudart.cudaStreamCreateWithFlags, cudart.cudaStreamNonBlocking)
    cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(handle=cudnn_handle, stream=int(stream))
    yield cudnn_handle
    cudnn.destroy_handle(cudnn_handle)
    _cudart_call(cudart.cudaStreamDestroy, stream)


# =================== PyTest Hooks =====================

def pytest_configure(config):
    global _xdist_controller, _stderr_fd, _FROST_RUN_ID, _FROST_ROUTING_DIR
    _drop_inherited_worker_identity(config)  # idempotent: pytest_cmdline_main is firstresult, a plugin answering it first skips the call there
    _xdist_controller = not _is_xdist_worker() and bool(getattr(config.option, "tx", None))
    _stderr_fd = os.dup(sys.__stderr__.fileno())
    _FROST_RUN_ID = _frost_routing_run_id()
    _FROST_ROUTING_DIR = _frost_routing_dir(_FROST_RUN_ID)
    if _shared_gpu() and not _xdist_controller:
        _log_to_real_stderr("[mem-gate] armed by CUDNN_TEST_SHARED_GPU=1: other pytest processes share this GPU")

    assert _cudart_call(cudart.cudaGetDeviceCount) > 0
    _track_cuda_graphs()

    print("===== cudnn-frontend conftest.py ====")
    print(f"cuDNN Frontend Version: {cudnn.__version__}")
    print(f"cuDNN Frontend Path: {cudnn.__file__}")
    print(f"Test run id: {_FROST_RUN_ID}")
    try:
        print(f"cuDNN Backend Version: {cudnn.backend_version()}")
    except Exception as e:
        print(f"cuDNN Backend not available: {e}")
    prop = _cudart_call(cudart.cudaGetDeviceProperties, 0)
    print(f"GPU Name: {prop.name.decode().rstrip(chr(0))}")
    print(f"SM Arch Version: {(prop.major, prop.minor)}")
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
    else:
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch Path: {torch.__file__}")
        print(f"PyTorch CUDA Version: {torch.version.cuda}")
        print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")

# fmt: off
def pytest_addoption(parser):
    # Generic options that may be used by all scripts.
    parser.addoption("--dryrun", action="store", nargs="?", const=1, type=int, default=0, help="show repro commands when 1, 2, or 3 (use with '-s')")
    parser.addoption("--diffs", action="store", type=int, default=10, help="set number of numerical mismatches to display")
    parser.addoption("--repro", action="store", type=str, default=None, help="specify config string to run repro function")
    parser.addoption("--seed", action="store", type=int, default=None, help="[fuzzer] random seed for reproducibility")
    parser.addoption("--num-tests", action="store", type=int, default=100, help="[fuzzer] number of random tests to run")
    parser.addoption("--perf", action="store_true", help="enable performance profiling")
    parser.addoption("--timing_method", action="store", type=str, default="cupti", choices=["events", "cupti"], help="timing method: 'cupti' (torch.profiler device_time, default) or 'events' (CUDA events)")

    # MHA command line options to overwrite specific test dimensions in test_mhas.py and test_mhas_v2.py.
    parser.addoption("--b", default=None, type=int, help="[sdpa tests] batch dimension")
    parser.addoption("--s_q", default=None, type=int, help="[sdpa tests] query sequence length")
    parser.addoption("--s_kv", default=None, type=int, help="[sdpa tests] key/value sequence length")
    parser.addoption("--d_qk", default=None, type=int, help="[sdpa tests] query/key embedding dimension per head")
    parser.addoption("--d_v", default=None, type=int, help="[sdpa tests] value embedding dimension per head")
    parser.addoption("--h_q", default=None, type=int, help="[sdpa tests] query number of heads")
    parser.addoption("--h_k", default=None, type=int, help="[sdpa tests] key number of heads")
    parser.addoption("--h_v", default=None, type=int, help="[sdpa tests] value number of heads")
    parser.addoption("--deterministic", default=None, type=int, choices=[0, 1], help="[sdpa tests] force deterministic algorithm")
    parser.addoption("--block_size", default=None, type=int, help="[sdpa tests] block size for paged attention")
    parser.addoption("--left_bound", default=None, type=int, help="[sdpa tests] size of the window to the left of the diagonal")
    parser.addoption("--right_bound", default=None, type=int, help="[sdpa tests] size of the window to the right of the diagonal")

    parser.addoption("--implementation", action="store", default=None, type=str, choices=["AUTO", "COMPOSITE", "UNIFIED"], help="[test_mhas_v2.py], overwrites implementation")

    parser.addoption("--skip-ref", action="store_true", help="[NSA, DSA, gemm_swiglu, gemm_amax, grouped_gemm_swiglu, sdpa_fwd, sdpa_bwd] Skip reference computation for performance testing")

    # NSA (Native Sparse Attention) command line options for test_NSA_selection_attention.py, test_NSA_swa.py
    parser.addoption("--nsa-b", action="store", default=None, type=int, help="[NSA] Batch size")
    parser.addoption("--nsa-s_q", action="store", default=None, type=int, help="[NSA] Query sequence length")
    parser.addoption("--nsa-s_kv", action="store", default=None, type=int, help="[NSA] Key/value sequence length")
    parser.addoption("--nsa-d_qk", action="store", default=None, type=int, help="[NSA] Query/key embedding dimension per head")
    parser.addoption("--nsa-d_v", action="store", default=None, type=int, help="[NSA] Value embedding dimension per head")
    parser.addoption("--nsa-h_q", action="store", default=None, type=int, help="[NSA] Number of query heads")
    parser.addoption("--nsa-h_k", action="store", default=None, type=int, help="[NSA] Number of key heads")
    parser.addoption("--nsa-h_v", action="store", default=None, type=int, help="[NSA] Number of value heads")

    # DSA (DeepSeek Sparse Attention) command line options for test_DSA_*.py
    parser.addoption("--dsa-b", action="store", default=None, type=int, help="[DSA] Batch size")
    parser.addoption("--dsa-s_q", action="store", default=None, type=int, help="[DSA] Query sequence length")
    parser.addoption("--dsa-s_kv", action="store", default=None, type=int, help="[DSA] Key/value sequence length")
    parser.addoption("--dsa-h_q", action="store", default=None, type=int, help="[DSA] Number of query heads")
    parser.addoption("--dsa-h_kv", action="store", default=None, type=int, help="[DSA] Number of KV heads")
    parser.addoption("--dsa-d_qk", action="store", default=None, type=int, help="[DSA] Query/key embedding dimension per head")
    parser.addoption("--dsa-d_v", action="store", default=None, type=int, help="[DSA] Value embedding dimension per head")
    parser.addoption("--dsa-topk", action="store", default=None, type=int, help="[DSA] Top-K count")
    parser.addoption("--dsa-ratio", action="store", default=None, type=int, help="[DSA] Indexer compression ratio")

    # GEMM SwiGLU command line options for test_gemm_swiglu.py
    parser.addoption("--gemm-swiglu-mnkl", action="store", default=None, type=str, help="[test_gemm_swiglu.py] M,N,K,L dimensions as comma-separated values (e.g., '256,256,512,1')")
    parser.addoption("--gemm-swiglu-mma-tiler", action="store", default=None, type=str, help="[test_gemm_swiglu.py] MMA tiler (M,N) dimensions as comma-separated values (e.g., '128,128')")
    parser.addoption("--gemm-swiglu-cluster-shape", action="store", default=None, type=str, help="[test_gemm_swiglu.py] Cluster shape (M,N) dimensions as comma-separated values (e.g., '1,1')")
    parser.addoption("--gemm-swiglu-alpha", action="store", default=None, type=float, help="[test_gemm_swiglu.py] Alpha scaling factor")

    # GEMM Amax command line options for test_gemm_amax.py
    parser.addoption("--gemm-amax-mnkl", action="store", default=None, type=str, help="[test_gemm_amax.py] M,N,K,L dimensions as comma-separated values (e.g., '512,256,256,1')")
    parser.addoption("--gemm-amax-mma-tiler", action="store", default=None, type=str, help="[test_gemm_amax.py] MMA tiler (M,N) dimensions as comma-separated values (e.g., '128,128')")
    parser.addoption("--gemm-amax-cluster-shape", action="store", default=None, type=str, help="[test_gemm_amax.py] Cluster shape (M,N) dimensions as comma-separated values (e.g., '1,1')")

    # Grouped GEMM SwiGLU command line options for test_grouped_gemm_swiglu.py
    parser.addoption("--grouped-gemm-nkl", action="store", default=None, type=str, help="[test_grouped_gemm_swiglu.py] N,K,L dimensions as comma-separated values (e.g., '512,512,4')")
    parser.addoption("--grouped-gemm-group-m", action="store", default=None, type=str, help="[test_grouped_gemm_swiglu.py] M values per group as comma-separated values (e.g., '256,512,256,256')")
# fmt: on


# =================== FROST routing summary =====================
# We are transitioning ops from the native cuDNN backend to FROST engines; the
# end state is every graph on FROST. This summary shows, per test run, how many
# graphs each path served ("frost:<engine>" vs "native:<harness site>"), so the
# remaining native population is visible per op family. Counts come from the
# test-side frost_routing tally (recorded by the sdpa harness after build_plans,
# once the plan walk has resolved — the cudnn package itself is not
# instrumented). Under pytest-xdist each worker persists its per-process counts
# to a file at session finish and the controller aggregates them in the terminal
# summary.
#
# The files live in a PER-RUN directory, `.frost_routing_<host>_<run id>` beside
# this file. The run id is minted once per run by the controller (or the single
# process) in pytest_configure and exported as CUDNN_TEST_RUN_ID, which the
# xdist workers -- spawned after that hook -- inherit; a worker without it falls
# back to xdist's own PYTEST_XDIST_TESTRUNUID. Concurrent pytest processes on one
# tree therefore never touch each other's files. A pytest session a TEST starts
# inherits its worker's PYTEST_XDIST_* and this id; _drop_inherited_worker_identity
# strips them (xdist marks a real worker with config.workerinput), so it is a
# run of its own, not a second writer of this run's worker file. With ONE
# shared directory a second process's session start deleted the first one's
# worker files (lost counts), and a worker that lost the race between makedirs
# and open died with FileNotFoundError at session finish -- a red run with
# every test green.
#
# The name carries the OWNING HOST (_ROUTING_HOST: this machine's hostname, every
# character outside [A-Za-z0-9.-] replaced by "-", so the name's "_" separator
# never occurs in it) because the tree may be shared storage that holds LIVE runs
# of other machines, and a pid is a fact only on the host that minted it:
# os.kill(pid, 0) answers for this machine's pids, so a remote controller's pid
# that happens to be absent here would read as a crashed run. The session-start
# sweep (_sweep_stale_routing_dirs) therefore removes a dead-pid directory only
# when its name carries THIS host (a pid os.kill cannot even represent -- a
# crafted or corrupted name -- counts as dead), never removes a directory of
# this host whose pid is alive, whatever its age (a recycled pid keeps such a
# leftover until that process ends), and leaves every other host's directory
# alone (a directory of the previous per-run layout, `.frost_routing_<pid>_<uid>`,
# has no host in its name and counts as another host's: it may be another
# machine's live run) until it is older than _ROUTING_LEFTOVER_TTL_S (a day:
# longer than any pytest run on one tree). The owner is the bare hostname: two
# machines that report the same name and share a tree judge each other's pids as
# before. xdist workers are local processes, so they derive the same name from
# the same host.

_FROST_ROUTING_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".frost_routing")
_ROUTING_HOST = re.sub(r"[^A-Za-z0-9.-]", "-", socket.gethostname()) or "unknown-host"
_ROUTING_LEFTOVER_TTL_S = 24 * 60 * 60
_FROST_RUN_ID = None  # set in pytest_configure
_FROST_ROUTING_DIR = _FROST_ROUTING_BASE  # re-keyed per run in pytest_configure


def _frost_routing_dir(run_id):
    return f"{_FROST_ROUTING_BASE}_{_ROUTING_HOST}_{run_id}"


def _routing_dir_owner(name):
    """(host, pid) a per-run routing directory's NAME declares; pid is None when the name carries none (xdist's own id as the run
    id).  A name of the previous layout, `.frost_routing_<pid>_<uid>`, reads its pid as the host here -- never equal to this host's
    tag, so it is treated as another host's directory."""
    rest = name[len(os.path.basename(_FROST_ROUTING_BASE)) + 1 :]
    host, _, run_id = rest.partition("_")
    pid = run_id.split("_")[0]
    return host, (int(pid) if pid.isdigit() else None)


def _frost_routing_run_id():
    if _is_xdist_worker():
        # A real worker (a merely inherited identity was dropped in pytest_cmdline_main): the id the controller that spawned
        # it exported; xdist's own id is the fallback (a controller without this conftest); a fresh id last, so the worker
        # still has somewhere to write.
        return os.environ.get("CUDNN_TEST_RUN_ID") or os.environ.get("PYTEST_XDIST_TESTRUNUID") or f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    # Controller or single process: a NEW run, whatever a parent pytest (a test that spawns pytest) exported.
    run_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    os.environ["CUDNN_TEST_RUN_ID"] = run_id  # the xdist workers are spawned after pytest_configure and inherit the environment
    return run_id


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False  # gone -- or a number no pid_t can hold (a crafted or corrupted directory name): not a live run either way
    except OSError:
        pass  # exists but is not ours (EPERM), or no signal support here: treat as alive
    return True


def _frost_routing_counts():
    try:
        import frost_routing

        return frost_routing.snapshot()
    except Exception:
        return None


def _frost_routing_measured():
    try:
        import frost_routing

        return dict(frost_routing.MEASURED)
    except Exception:
        return None


def _sweep_stale_routing_dirs(now=None):
    """Drop what a crashed earlier run left beside this file: a per-run directory of THIS host whose controller pid is gone, the
    shared directory of the previous layout, and a directory of another host (or without a pid in its name) older than
    _ROUTING_LEFTOVER_TTL_S.  A LIVE run's directory is never touched: a live sibling of this host has a live pid (kept whatever its
    age), and another host's run has a pid this host cannot judge, so its directory is left alone until the TTL.  Returns the names
    removed."""
    import shutil

    now = time.time() if now is None else now
    parent, prefix = os.path.dirname(_FROST_ROUTING_BASE), os.path.basename(_FROST_ROUTING_BASE)
    try:
        names = os.listdir(parent)
    except OSError:
        return []
    swept = []
    for name in names:
        path = os.path.join(parent, name)
        if not name.startswith(prefix) or path == _FROST_ROUTING_DIR or not os.path.isdir(path):
            continue
        host, pid = _routing_dir_owner(name)
        if name == prefix:
            stale = True  # the shared directory of the previous layout
        elif host == _ROUTING_HOST and pid is not None:
            stale = not _pid_alive(pid)  # this host's run: its pid is a fact here
        else:
            try:
                stale = now - os.stat(path).st_mtime > _ROUTING_LEFTOVER_TTL_S  # another host's, or no pid to judge: by age only
            except OSError:
                continue
        if stale:
            shutil.rmtree(path, ignore_errors=True)
            swept.append(name)
    return swept


def pytest_sessionstart(session):
    # Controller (or single-process run): drop what a crashed earlier run left behind -- see _sweep_stale_routing_dirs. A LIVE
    # run's directory, this host's or another's, is never touched, so concurrent runs on one tree do not race here.
    if _is_xdist_worker():
        return
    _sweep_stale_routing_dirs()


def pytest_sessionfinish(session, exitstatus):
    counts = _frost_routing_counts()
    measured = _frost_routing_measured()
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return
    import json

    # `<worker>.json` = the routing counts, `<worker>.measured.json` = the measurements (frost_routing.measured); the
    # controller tells them apart by the suffix.
    for payload, suffix in ((counts, ".json"), (measured, ".measured.json")):
        if payload:
            os.makedirs(_FROST_ROUTING_DIR, exist_ok=True)
            with open(os.path.join(_FROST_ROUTING_DIR, f"{worker}{suffix}"), "w") as f:
                json.dump(payload, f)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        return  # workers report via files; only the controller prints
    counts = dict(_frost_routing_counts() or {})
    measured = dict(_frost_routing_measured() or {})
    if os.path.isdir(_FROST_ROUTING_DIR):
        import json
        import shutil

        for fname in sorted(os.listdir(_FROST_ROUTING_DIR)):
            try:
                with open(os.path.join(_FROST_ROUTING_DIR, fname)) as f:
                    payload = json.load(f)
                if fname.endswith(".measured.json"):
                    measured.update(payload)
                else:
                    for key, n in payload.items():
                        counts[key] = counts.get(key, 0) + n
            except Exception:
                pass
        shutil.rmtree(_FROST_ROUTING_DIR, ignore_errors=True)
    if counts:
        total = sum(counts.values())
        frost_total = sum(n for key, n in counts.items() if key.startswith("frost:"))
        terminalreporter.section("FROST routing")
        terminalreporter.write_line(f"graphs on FROST engines: {frost_total}/{total} ({100.0 * frost_total / total:.1f}%) -- transition goal is all-FROST")
        for key in sorted(counts):
            terminalreporter.write_line(f"  {key}: {counts[key]}")
    if measured:
        terminalreporter.section("measured")  # frost_routing.measured: records a passing test wants in this log
        for key in sorted(measured):
            terminalreporter.write_line(f"  {key}: {measured[key]}")
