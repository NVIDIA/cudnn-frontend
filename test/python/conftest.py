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

# The JAX interop tests (fe_api/**/test_*_jax.py) initialize XLA in the same pytest
# process as the torch suites; XLA's default 75%-of-GPU preallocation starves later
# torch tests of memory (CUDA_ERROR_OUT_OF_MEMORY at kernel-compile time). Must be set
# before jax initializes its backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import time
import traceback
import pytest

# Import TransformerEngine BEFORE cudnn to avoid library loading conflicts
# TE requires specific CUDA library versions that conflict if cudnn is loaded first
try:
    import transformer_engine
except (ImportError, OSError):
    pass

import cudnn
import torch

# fmt: off

# =================== CUDA Synchronize Guard =====================
torch.cuda.synchronize_unsafe = torch.cuda.synchronize

def cuda_synchronize_safe(*args, **kwargs):
    try:
        torch.cuda.synchronize_unsafe(*args, **kwargs)
    except Exception as e:
        entries = traceback.extract_stack(sys._getframe(1))
        test_entries = [f for f in entries if "/test/python/" in f.filename]
        print("Traceback (most recent call last):", flush=True)
        print(*traceback.format_list(test_entries), end="", flush=True)
        print(e, flush=True)
        os._exit(os.EX_SOFTWARE)

torch.cuda.synchronize = cuda_synchronize_safe

# =================== GPU memory gate (pytest-xdist) =====================
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

_MEM_GATE_FRACTION = float(os.environ.get("CUDNN_TEST_MEM_GATE_FRACTION", "0.2"))
_MEM_GATE_TIMEOUT_S = float(os.environ.get("CUDNN_TEST_MEM_GATE_TIMEOUT", "30"))


def _under_xdist():
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")) > 1


def _wait_for_free_gpu_memory(context):
    # Best effort: proceed after the timeout even if the floor was not reached,
    # so a worker can never deadlock the run; the allocation itself then either
    # succeeds or fails with the usual OOM.
    free, total = torch.cuda.mem_get_info()
    floor = _MEM_GATE_FRACTION * total
    if free >= floor:
        return
    torch.cuda.empty_cache()
    deadline = time.monotonic() + _MEM_GATE_TIMEOUT_S
    waited = False
    while time.monotonic() < deadline:
        free, _ = torch.cuda.mem_get_info()
        if free >= floor:
            break
        waited = True
        time.sleep(2)
    if waited:
        # sys.__stderr__ bypasses pytest capture so the message reaches the CI log.
        print(
            f"[mem-gate] {context}: waited for GPU memory "
            f"(free {free / 2**30:.2f} GiB, floor {floor / 2**30:.2f} GiB)",
            file=sys.__stderr__,
            flush=True,
        )


def _is_cuda_oom(exc):
    # torch.OutOfMemoryError only exists on newer torch; the cuda alias is old.
    return isinstance(exc, torch.cuda.OutOfMemoryError)


@pytest.fixture(autouse=True)
def _gpu_memory_gate(request):
    if _under_xdist():
        _wait_for_free_gpu_memory(request.node.name)
    yield
    if _under_xdist():
        torch.cuda.empty_cache()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    if not _under_xdist() or outcome.excinfo is None or not _is_cuda_oom(outcome.excinfo[1]):
        return
    # OOM under xdist is usually transient sibling-worker pressure, not a
    # property of this test: release our cache, wait for the device, retry once.
    print(f"[mem-gate] {item.nodeid}: OOM under xdist, retrying once", file=sys.__stderr__, flush=True)
    torch.cuda.empty_cache()
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
    stream = torch.cuda.Stream()
    cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(handle=cudnn_handle, stream=stream.cuda_stream)
    yield cudnn_handle
    cudnn.destroy_handle(cudnn_handle)


# =================== PyTest Hooks =====================
def pytest_load_initial_conftests(args, early_config, parser):
    if not any(arg.startswith("--tb=") for arg in args):
        args.insert(0, "--tb=short")
    if "--no-header" not in args:
        args.insert(0, "--no-header")


def pytest_configure(config):
    assert torch.cuda.is_available()

    print("===== cudnn-frontend conftest.py ====")
    print(f"cuDNN Frontend Version: {cudnn.__version__}")
    print(f"cuDNN Frontend Path: {cudnn.__file__}")
    try:
        print(f"cuDNN Backend Version: {cudnn.backend_version()}")
    except Exception as e:
        print(f"cuDNN Backend not available: {e}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Path: {torch.__file__}")
    print(f"PyTorch GPU Name: {torch.cuda.get_device_name()}")
    print(f"PyTorch SM Arch Version: {torch.cuda.get_device_capability()}")
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
    parser.addoption("--b", default=None, type=int, help="[test_mhas.py] batch dimension")
    parser.addoption("--s_q", default=None, type=int, help="[test_mhas.py] query sequence length")
    parser.addoption("--s_kv", default=None, type=int, help="[test_mhas.py] key/value sequence length")
    parser.addoption("--d_qk", default=None, type=int, help="[test_mhas.py] query/key embedding dimension per head")
    parser.addoption("--d_v", default=None, type=int, help="[test_mhas.py] value embedding dimension per head")
    parser.addoption("--h_q", default=None, type=int, help="[test_mhas.py] query number of heads")
    parser.addoption("--h_k", default=None, type=int, help="[test_mhas.py] key number of heads")
    parser.addoption("--h_v", default=None, type=int, help="[test_mhas.py] value number of heads")
    parser.addoption("--deterministic", default=None, type=int, choices=[0, 1], help="[test_mhas.py] force deterministic algorithm")
    parser.addoption("--block_size", default=None, type=int, help="[test_mhas.py] block size for paged attention")
    parser.addoption("--left_bound", default=None, type=int, help="[test_mhas.py] size of the window to the left of the diagonal")
    parser.addoption("--right_bound", default=None, type=int, help="[test_mhas.py] size of the window to the right of the diagonal")

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

_FROST_ROUTING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".frost_routing")


def _frost_routing_counts():
    try:
        import frost_routing

        return frost_routing.snapshot()
    except Exception:
        return None


def pytest_sessionstart(session):
    # Controller (or single-process run): drop stale worker files from a previous run.
    if os.environ.get("PYTEST_XDIST_WORKER") is None and os.path.isdir(_FROST_ROUTING_DIR):
        import shutil

        shutil.rmtree(_FROST_ROUTING_DIR, ignore_errors=True)


def pytest_sessionfinish(session, exitstatus):
    counts = _frost_routing_counts()
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if counts and worker is not None:
        import json

        os.makedirs(_FROST_ROUTING_DIR, exist_ok=True)
        with open(os.path.join(_FROST_ROUTING_DIR, f"{worker}.json"), "w") as f:
            json.dump(counts, f)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        return  # workers report via files; only the controller prints
    counts = dict(_frost_routing_counts() or {})
    if os.path.isdir(_FROST_ROUTING_DIR):
        import json
        import shutil

        for fname in sorted(os.listdir(_FROST_ROUTING_DIR)):
            try:
                with open(os.path.join(_FROST_ROUTING_DIR, fname)) as f:
                    for key, n in json.load(f).items():
                        counts[key] = counts.get(key, 0) + n
            except Exception:
                pass
        shutil.rmtree(_FROST_ROUTING_DIR, ignore_errors=True)
    if not counts:
        return
    total = sum(counts.values())
    frost_total = sum(n for key, n in counts.items() if key.startswith("frost:"))
    terminalreporter.section("FROST routing")
    terminalreporter.write_line(f"graphs on FROST engines: {frost_total}/{total} ({100.0 * frost_total / total:.1f}%) -- transition goal is all-FROST")
    for key in sorted(counts):
        terminalreporter.write_line(f"  {key}: {counts[key]}")
