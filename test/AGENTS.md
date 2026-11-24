# test — Agent Guide

Two suites: `test/cpp` (Catch2, C++ graph API) and `test/python` (pytest). Both need an NVIDIA GPU and a cuDNN 9.x backend at runtime. Build/install commands: [../AGENTS.md](../AGENTS.md).

- New numerical APIs and kernel paths: use the [determinism skill](../skills/determinism_tests/SKILL.md) to add/run L0 tests.
- Kernel determinism changes follow the [root rules](../AGENTS.md#kernel-determinism-and-batch-invariance) and the [coverage table](python/determinism_goldens/README.md). Determinism details are in [Determinism and batch invariance](#determinism-and-batch-invariance) below.

## C++ tests (`test/cpp`)

- Catch2 v3 binary, target `tests`, built by the default CMake build (`CUDNN_FRONTEND_BUILD_TESTS=ON`) into `build/bin/tests`.
- Run all: `./build/bin/tests`. List: `--list-tests`. One case: `./build/bin/tests "Validate conv node"`. Filter by tag: `./build/bin/tests "[serialize]"`.
- After changing shared graph validation, run the unfiltered `tests` and `samples` binaries; a feature-tagged run misses existing callers. For pointwise arity changes, inspect shipped positive calls of each overload and keep intentional negative tests (the INT8 convolution sample exercises unary IDENTITY).

### Plan failure diagnostics

Match each rejected plan's name and position in the same diagnostic fragment; independent searches pass when a reason is attached to the wrong plan. `test/cpp/plans.cpp` uses the actually offered engine names, independent of backend failure strings.

### Autotune failure contracts

- Warmup success is not a measured candidate. Test all-failed tuning with real invalid bindings, then correct the bindings and execute the preserved plan (`test/cpp/autotune.cpp`, which also covers unbuilt slots and retry).
- Inject a backend error after successful warmup and check that no winner is published. A CUDA event failure must not become a zero-time winner.
- After tuning, compare each surviving plan's engine/knob identity, name and notes with its pre-tune snapshot. Never assert a timing rank. Include a rejected prefix and an unbuilt suffix so pruning moves a valid plan to index zero; replay its identity, append/build another config, and tune a serialized/reloaded plan with no engine-config list.

### Header and runtime version guards

Keep header-version comparisons out of runtime `if` conditions: `CUDNN_VERSION < 92900` in a combined condition is MSVC C4127, fatal under `/WX`. Select the header case with `#if`/`#elif`, then check the loaded backend version inside the branch. Preserve older-header and newer-header/older-runtime behavior, including the inverse guard of an old-backend refusal test (the custom-reduction tests and samples follow this).

## Python tests (`test/python`)

### Running and selecting tests

Run from `test/python` so `pytest.ini` and `conftest.py` apply:

```bash
cd test/python
pytest                       # pytest.ini addopts default to -m L0 (smoke) --tb=short --no-header
pytest -m L1                 # levels L0..L4; higher = larger sweeps
pytest -n 4                  # pytest-xdist; mind marker gpu_exclusive for tests that need the GPU alone
pytest -m smoke sdpa/frost   # the SMOKE tier: one cell per code path, marked from tiers/smoke_<arch>.txt (tiers/README.md)
pytest -m "L0 and not nightly_only" sdpa/frost   # the FULL local tier: L0 minus tiers/nightly_only.txt; CI stays -m L0
pytest conv/graph/test_conv_fprop.py  # one file; the default -m L0 filter still applies
pytest gemm/cutedsl/                  # CuTe DSL kernel tests
```

- **Read pytest's own summary line; do not post-process the output.** `... | grep -c "passed"` counts a collection error as a pass, and `cmd | tail` reports tail's exit status.
- Requirements: `pip install -e .` plus `pytest pytest-xdist looseversion`. `cutedsl/` directories need an SM90/SM100-class GPU; unsupported arch/dtype/backend-version combos must skip, never fail.
- **Levels.** Mark tests `@pytest.mark.L0` ... `L4`. L0 stays fast (default CI smoke); big sweeps go higher. Check for a module-level `pytestmark` before adding per-test markers; duplicating it is noise.
- **Tiers are lists, not markers in test files.** `conftest.py` marks cells `smoke` / `nightly_only` at collection from the committed node-id lists in `tiers/` (`smoke_<arch>.txt` per compute capability, `nightly_only.txt`). `CUDNN_TEST_TIER_ARCH=cc107` applies another arch's list on any host. Rules for listed cells, the kept twin of every demoted cell, and run lines are in `tiers/README.md`. CI keeps `-m L0`; tiers are opt-in. `test_tiers.py` asserts every listed id still collects and the markers select exactly the listed cells: when you rename or re-parametrize a listed test, edit the list in the same commit.
- **Keep `sdpa/frost/` paths contiguous on the command line.** `sdpa/frost/conftest.py` sets `CUDNN_FRONTEND_ENABLE_FROST_ENGINES` through an autouse fixture (FROST manifest rows are opt-in). Any top-level path placed between two `sdpa/frost/` paths drops that fixture for everything after it: no FROST row is offered, engine pins fail loudly, and tests that `pytest.skip` on a missing python plan go falsely green. Detector, as its own plugin:

  ```python
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_call(item):
      print(item.name, os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES"), "_frost_opt_in" in item.fixturenames)
      yield
  ```

- **`test/python/sdpa/` is a mixed directory; the `test_` prefix is load-bearing.** `fp16.py`, `helpers.py`, `random_config.py` are harness modules. `pytest.ini` has no `python_files` override, so a test file without the prefix is never collected. After moving or adding a test, confirm the default sweep collects it:

  ```bash
  pytest --collect-only -q | grep -c sdpa/torch/test_torch_ops.py
  ```

- **CI lane selection.** Check the actual CI target and `-k` filters before relying on L0.
  - The general Python target excludes FROST cases: representative shared-API FROST tests also need collection under `sdpa/frost/` (`test_sdpa_ordered_bindings.py` reuses the shared ordered-binding smoke logic). Verify collection and execution on a supported GPU.
  - FROST regression filenames must match the CI `-k frost` selection; verify collection with that filter.
  - The architecture CI selector may omit a newly added standalone file. The SM107 lane selects `test_sdpa_fp8_sm107.py` explicitly: keep prepared FP8 cases in `TestPreparedSm107Fp8` there and mirror SM107 numerical regressions into that file (reuse the checks, keep parameter marks), or update the lane selector with the move.
  - An imported/re-exported test function does not inherit its source module's `pytestmark`. Mark the wrapper class L0 explicitly and verify collection with the lane's marker expression (`test_sdpa_fwd_mxfp8_sm100.py::TestStagedMxfp8`).
- Temporary skips, only when explicitly requested: limit them to known failing cases, use a `skip until fix` reason, keep the failure report, and remove them with the fix.

### conftest.py landmines — read before editing

- `PYTORCH_CUDA_ALLOC_CONF` is set at the very top, **before any torch import** (torch reads it once at allocator init). Do not move it or import torch in a plugin that loads earlier.
- `import transformer_engine` (in try/except) happens **before** `import cudnn`; the other order conflicts. Preserve it.
- `pytest_configure` asserts `torch.cuda.is_available()`; there is no CPU-only mode.
- Custom CLI options exist (`--dryrun`, `--repro`, `--seed`, `--perf`, per-op overrides like `--b/--s_q`, `--nsa-*`, `--dsa-*`); check `pytest_addoption` before adding one.
- **Crash isolation** (`# Crash isolation` block): `pytest_cmdline_main` injects `-n1 --max-worker-restart=100000`, so a segfault, poisoned CUDA context or hang kills one xdist worker, which is replaced. After every test `pytest_runtest_logfinish` probes the context with `torch.cuda.synchronize()`. A per-test `faulthandler.dump_traceback_later(exit=True)` deadline (`CUDNN_TEST_TIMEOUT`, default 1500 s, `0` disables) covers the probe; it is a C watchdog because a hung driver call holds the GIL. Not injected under `-n<N>`, `-s`, `--pdb`, `--collect-only`, or `CUDNN_TEST_NO_ISOLATION=1`; then a dead context stops the run via `pytest.exit`. Killing a worker does not stop a kernel it left running; the next worker can block behind it.
- **Several pytest processes on one tree or one GPU** (pins: `test_conftest_parallel.py`):
  1. FROST routing tallies (`# FROST routing summary` block) go to a per-run directory `test/python/.frost_routing_<host>_<run id>`. The controller mints the id in `pytest_configure` and exports `CUDNN_TEST_RUN_ID`; workers inherit it (fallback `PYTEST_XDIST_TESTRUNUID`); a child pytest spawned by a test is its own run (`pytest_cmdline_main` drops inherited `PYTEST_XDIST_*` / `CUDNN_TEST_RUN_ID` unless `config.workerinput`). `pytest_sessionstart` (`_sweep_stale_routing_dirs`) removes only this host's directories whose controller pid is dead; other hosts' and old-layout directories expire after `_ROUTING_LEFTOVER_TTL_S` (one day). The banner prints `Test run id: <id>`. Never revert to one shared directory.
  2. The GPU memory gate (`# GPU memory gate` block: wait for free memory before each test, `torch.cuda.empty_cache()` after, one retry on `torch.OutOfMemoryError`; `CUDNN_TEST_MEM_GATE_FRACTION`, default 0.2, `CUDNN_TEST_MEM_GATE_TIMEOUT`, default 30 s) arms only under several xdist workers. For K independent processes on one GPU, set `CUDNN_TEST_SHARED_GPU=1` in each (stderr shows `[mem-gate] armed by CUDNN_TEST_SHARED_GPU=1`).
  3. A record a **passing** test needs in the CI job log goes through `frost_routing.measured(key, text)`; the controller prints it under `measured`, aggregated across workers. Captured stdout of passed tests never reaches the log (lanes run without `-rA`). Pin: `test_conftest_parallel.py::test_a_measurement_reaches_the_controllers_summary`.
  4. A host-only test of version-gated behavior pins `cudnn.backend_version` with `monkeypatch` (e.g. `crashing_backend` in `sdpa/frost/test_sdpa_backend_guard.py`): local and CI backends differ, so an inherited version passes vacuously or fails in CI.
- **Handle fixture.** A session-scoped autouse `cudnn_handle` fixture binds one handle to a dedicated stream; use it instead of per-test handles. Tests that rebind it save `cudnn.get_stream(cudnn_handle)` before setup and restore that exact stream in `finally`, including on setup failure/skip; restoring `torch.cuda.current_stream()` leaks a different stream. Eager calls on that handle need producer-to-handle and handle-to-reference edges: wrap its stream in `torch.cuda.ExternalStream`, wait for current-stream input updates before execute, then wait for its output before the reference (`test_public_graph_geometry_rebind_replay`, SM80).
- **CUDA Graph test lifetimes.** Reset a test-owned CUDA Graph after replay verification (`finally` or context manager); a graph freed by GC inside a later test's capture invalidates that capture. `conftest.py` runs `gc.collect()` between tests and fails a test in teardown if the collector freed a captured, never-reset graph. Fix ownership; do not disable GC or treat a retry as validation.

### Streams, capture and sync-debug

- A side-stream test calls `stream.wait_stream(torch.cuda.current_stream())` after creating inputs and before switching streams; also after output initialization (including sentinel fills). Synchronizing the consumer afterward cannot repair a missing producer edge.
- Raw `pygraph.execute` launches on the cuDNN handle's stream. For capture, bind the handle to the capture stream and pass it to `execute`. A shared handle may name stream zero: create a non-default capture stream and restore the handle afterward. Treat the empty-CUDA-graph warning as an error (`test_forward_state_alias_after_graph_warmup`). Detectors: `test_cuda_graph_retention_option` with a default-stream handle; QSA's `test_entry_points_select_the_tensors_device_with_another_device_current`.
- When a CUDA-graph check can select a backend plan, supply an explicit handle bound to Torch's capture stream. An empty capture followed by NaN outputs is a harness failure; check the captured graph and changed-input replay before blaming the backend.
- **Scope sync-debug to the tested launch inside the capture context.** Enter `with torch.cuda.graph(...)` first, then `set_sync_debug_mode("error")` around the execute call, and restore the previous mode in `finally` before leaving the context. Torch capture setup can synchronize; guarding it tests Torch, and auditing `capture_begin()` can poison later tests. `test_native_d576_compact_thd_rebind_and_capture` is the example (rebinding, poisoned replay, empty inputs). A pass on a Torch build that accepts the old scope does not reproduce the CI failure.

### Layout

- `test/python/<operation>/<backend>/`: operation (`sdpa`, `gemm`, `conv`, `norm`, `rope`, `linear_attention`, ...), then backend/framework: `graph` (native cuDNN via `cudnn.pygraph`), `frost`, `cutedsl`, `torch`, `jax`, or engine names such as `linear_attention/cake`. Cross-backend tests sit at the operation root (`linear_attention/test_la.py`).
- `test/python/core/<backend>/`: infrastructure tests (dispatch, import boundaries, graph execution, FROST buffers, ...).
- `test/python/test_utils.py` holds shared helpers; shared SDPA references are in `test/python/sdpa/`.

### Confirm you are testing the code you edited

- Editable installs pin one checkout (see [../AGENTS.md](../AGENTS.md)). Check `python -c "import cudnn; print(cudnn.__file__)"` and the `conftest.py` banner (`cuDNN Frontend Path:`). To point the editable install at another tree for one run, patch the finder's `MAPPING` from a `sitecustomize.py` on `PYTHONPATH`:

  ```python
  # sitecustomize.py -- the finder module name embeds the installed version; discover it.
  import importlib, pkgutil

  name = next(
      m.name
      for m in pkgutil.iter_modules()
      if m.name.startswith("__editable___nvidia_cudnn_frontend_") and m.name.endswith("_finder")
  )
  importlib.import_module(name).MAPPING["cudnn"] = "/path/to/your/worktree/python/cudnn"
  ```

- `python/cudnn/frost/template_loader.py` loads templates by absolute path (`spec_from_file_location`). To find the template a test compiles, log `path` at the top of `load_template`; do not infer it from `_pick_flavor`.
- `linear_attention/conftest.py` prepends its own checkout's `python/cudnn` to `cudnn.__path__`, so `cudnn.__file__` does not prove which LA modules load. For original-versus-candidate (RED/GREEN) runs, use a separate test tree per source tree, copy only the identical test changes into the baseline worktree, and check the concrete operation module's `__file__` (and source hashes for benchmark records) after conftest setup.
- Read repository docs relative to the test checkout, not `cudnn.__file__`: CI can load a wheel (no Markdown files) while conftests overlay source (`gemm/frost/test_block_scale_smem_layout_sm107.py`).
- **Subprocesses.** Children inherit neither pytest's `sys.path` nor editable-finder changes.
  - Add the test helper root (`test/python`) and the module's own directory to the child's `sys.path`; check the child from a temporary working directory without that root on `PYTHONPATH`. The delayed-MMA FP8 dual-KV probe needs `test/python` and `sdpa/frost`.
  - Compiler subprocesses import the same `cudnn` as the parent: derive the package root from `cudnn.__file__` (CI may test an installed wheel while the checkout has no native extension). The native FP8 decode score-release probe exercises this.
  - Source-package copies shared by xdist workers must exclude `__pycache__`/bytecode (atomic `.pyc` renames break `copytree`).
  - For a timed subprocess test, arm a child-side `faulthandler` dump before the parent deadline and surface `TimeoutExpired.stdout` / `.stderr` in the failure (`prepared_bwd_cache_utils.run_artifact_child`). Do not classify a compilation stall as a GPU hang without child-stage evidence.
  - A subprocess-based GPU test checks architecture in the parent before spawning; a `pytest.skip` in a plain child exits nonzero.
  - Cross-architecture compile probes set the target before importing the DSL in a fresh subprocess, override an inherited `CUTE_DSL_ARCH`, and disable the compiled cache. Skip only the named missing-intrinsic error (`test_rubin_profile_traces_or_names_the_missing_dsl_intrinsic`; run with `CUTE_DSL_ARCH=sm_80` to check isolation).
- When renaming a shared test reference module, search for bare and package-qualified imports, then collect the affected family alongside the neighboring family that originally collided, in both orders.
- When moving Python modules, build the validation wheel with fresh build output (stale `build/lib.*` ships removed modules), inspect its module paths, then run the API inventory against it. `core/test_import_boundaries.py` checks removed imports do not resolve.

## Conventions for new tests

### General

- **A regression test must be seen RED.** Restore the old line, confirm the test fails, restore the fix.
- Gate on capability: skip via `check_support()` failures, `cudnn.backend_version()`, `torch.cuda.get_device_capability()`. Architecture-specific adjustments use device capability, not engine-name suffixes.
- **Family checks compare a range.** CI `sm120` lanes include GB10 (SM 12.1). In `gemm/frost`, branch on `active_family() == "sm120"` (or `120 <= arch < 130`). Detector: `git grep -nE '_current_arch\(\) *== *12[0-9]' test/python/gemm`. A helper that looks for one engine's plan must catch `cudnnGraphNotSupportedError` from `create_execution_plans` (raised when no engine proposes a plan).
- If an optional import's `except ImportError` ends before the test body, use `return pytest.skip(...)` (CodeQL otherwise reports the name as possibly uninitialized). Check the CodeQL alert result as well as the workflow.
- Compare against an independent reference (`*_ref.py` / `*_reference.py`) with dtype-appropriate tolerances.
- **Scale the tolerance to the tensor:** `TOL * max(|ref|.max(), 1.0)`. GQA dK/dV sum over `h_q/h_kv` heads, so absolute bounds break at larger group sizes.
- **Build the reference in fp64 when the bound is tighter than ~1e-3.** CI containers run fp32 matmul as TF32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`). Use `sdpa/fp8_ref.compute_ref(dtype=torch.float64)`; in a hand-rolled reference `.double()` the operands before the matmul.
- **Check reference GEMM rounding before relaxing a chained-kernel tolerance.** Compare the projection in FP64 before its BF16 cast; the serving pool tests use `acc_dtype=torch.float64` while keeping intermediate casts and tolerances.
- **Seed before you allocate.** For comparable runs, build inputs once and reuse them. Every randomized SDPA input, including block masks, uses `generator=rng_data_gen` (`test_block_mask_uses_the_per_test_data_generator`). Compare actual masks as well as Q/K/V before blaming an earlier engine for an order-dependent failure.
- **Heuristic tests must survive tuning changes.** Do not pin a workload's winning scheduler, packing, split count, candidate order/set, or performance threshold. Test knob admission/rejection, validity of proposed candidates, and transport of a mocked chooser's result (`test_propose_preserves_recommendations_and_places_one_marker`). Check geometry against an independent oracle, not private helper calls. Performance rankings belong in offline benchmarks.
- **SM-count-dependent dispatch is asserted against the live resolver.** Compute expectations from the resolver the API runs (`proj_gemm._auto_tile_name`, `sparse_decode_split_choice(..., sm_count=sm)`) and assert a pinned branch only under the SM count that selects it (#1754). A declaration-time flag derived from such a resolver is re-read where consumed (`capture_safe` at `compile()`).
- Test feature combinations through the support contract: assert acceptance/decline for concrete graph facts and requested knobs. Capability-matrix fixtures sample a shape that serves the compared feature; keep native-flavor rejection separate and do not widen eligibility for a stale fixture (`test_band_capabilities.py`, `test_sdpa_native_d64_sm90.py`).
- **Named-engine rejection has two paths.** All engines decline: assert the target's specific reason in the aggregate error. Another engine supplies plans: a strict selector reports no plan for the named engine. Backend version and strides (including Stats layout) affect eligibility.
- **Plan-specific xfails check the selected plan** via `get_engine_and_knobs_at_index()`, with `strict=True` and the expected exception type (`test_sdpa_ragged_decode_stats`). Shape-override windows and strict xfails measured on one native plan also gate on the selected plan; an xfail does not contain an out-of-bounds access.
- Shape-override tests cover a backend lowering decline as well as a lowered graph. Filter ragged-offset (backend-only) overrides against `_variant_pack_uids()` while requiring every Q/K/V/O, Stats and length operand (`test_thd_cache_shape_grid_tracks_runtime_capacity`).
- **When you remove a fallback, invert its counter assertion.** E.g. "`calls["bwd"]` increments and `bwd_cpp` does not".
- **Version-bounded backend workarounds ship with a re-measuring detector.** For a constant such as `sdpa/fwd/backend_guard.py::SQ1_MXFP8_PLANNING_CRASH_FIXED_IN`, `sdpa/graph/test_mhas_v2.py::test_sdpa_mxfp8_cc107_backend_planning_crash_guard_is_current_L0` runs the trigger matrix one interpreter per contract with the guard disabled in the child (crash = child rc 139), plus a control, and fails with the version to record once the backend plans cleanly or crashes again above a recorded version.
- Memo-key regression tests observe the actual lookup and selected plan, not tuple offsets (`test_block_scaled_memo_respects_overlap_margin`).
- Workspace device checks compare storage with the declared launch device (`core/cutedsl/test_workspace_device.py`).
- Source-routing regexes must not backtrack exponentially on long multiline non-matches (`gemm/cutedsl/test_sm110_amax_reduction.py::test_direct_redux_scan_handles_multiline_call`).
- Host-only compile-cache tests that replace `cute.compile` also mock `compile_options()` (`test_DSA_sparse_attention_backward_sm100_d576_compile_cache_separates_address_width`).
- Benchmark harness tests: labels describe the executed backend layout; apply route-specific constraints before deduplicating (`benchmark/attention_inference/tests/test_runner.py`). Summaries reject non-finite/nonpositive medians with explicit checks (not `assert`), serialize with `allow_nan=False`, and recheck numerical errors against tolerance (`benchmark/linear_attention/steep_decay/`). Performance comparisons use a numerically valid baseline.
- When appending an SDPA adapter constructor argument, update every signature-tail pin (API-contract, score-stats, SM107, bias-max suites); search `test/python/sdpa/frost` for the previous trailing parameter.
- Pinning append-only records: assert a frozen historical prefix written explicitly; distinguish positional operands from appended keywords with the AST (SDPA field-order and FP8 THD fold pins).
- When a template-loader signature gains an optional argument, update every monkeypatched loader spy to accept and forward it; exercise the spy with legacy and new calls.
- Keep launches and recorder consumption (`frames.pop()`) outside `assert` statements.
- Source-parity tests across architectures use exact, counted replacements for the declared delta; check that changing the delta or a shared barrier still fails.
- A graph-ranking helper that catches an unsupported-graph exception returns a verdict shape each caller can unpack, or lets the caller propagate the original exception.

### Public API index and bindings

- When adding a public Python export (including an enum member), regenerate `test/python/api_index/api_index.txt` from the built package, inspect the diff, and run the API-index suite with `CUDNN_API_INDEX_PACKAGE_ROOT` set (otherwise the runtime check skips). No GPU needed. The scanner inspects Enum descriptors without evaluating them (`api_index/test_api_index.py`). Signature tests accept `object` and `object | None` for `None` defaults.
- When changing direct CPython C API calls, build with `CUDNN_FRONTEND_PYTHON_STABLE_ABI=ON` and import the abi3 module (`PyFloat_AS_DOUBLE` is unavailable under `Py_LIMITED_API`; use `PyFloat_AsDouble`). To qualify a build flag, inspect compiler definitions; `CUDNN_FRONTEND_SKIP_JSON_LIB=ON` in a CMake cache alone proves nothing.
- For DLPack dtype regressions, assert the descriptor dtype right after `tensor_like`, and exercise the raw C++ binding separately from the wrapper (`core/graph/test_raw_tensor_like_dtype.py`).
- When patching a newly allocated pybind tuple, move its unique ownership; an extra owning cast makes `PyTuple_SetItem` reject it.

### Memory and parallel workers

- **Large-grid regressions budget verification tensors too.** Compare every element in bounded chunks; keep boundary shapes. Qualify under a per-process allocator cap that makes the old test fail, then under the CI worker count; a four-worker run on a larger GPU does not establish the CI budget. Reduce only contraction width when launch geometry is independent of it; keep true large-index detectors at their address spans (`test_sdpa_bwd_dot_grid_determinism.py`, the delta suite).
- **Bound dense references independently of kernel memory.** Compute heads separately (mapping query head to KV head), keep workload/masks/precision/tolerances, validate against the dense oracle on small MHA/GQA/MQA (`test_sdpa_fwd_split_kv_sm120.py`). Recurrent FP64 references: chunk heads, compare chunked gradients to the whole-head reference on small ragged/empty cases. A reference OOM after kernel comparisons passed is not a kernel allocation failure; say so in triage.
- Large physical-stride tests: catch `torch.OutOfMemoryError` only around the large storage allocation and skip; never around launch or assertions. `gpu_exclusive` alone does not serialize xdist: use the marker plus `@pytest.mark.xdist_group(name="gpu_exclusive")` with `--dist loadgroup` (see the time-slicing note in [python/cudnn/sdpa/AGENTS.md](../python/cudnn/sdpa/AGENTS.md)).
- Before releasing Torch's cache, synchronize every device with reserved memory (`test_empty_cache_waits_for_other_devices_default_stream`).
- An asynchronous CUDA failure in an SDPA reference is not proof of a kernel defect: reproduce the reference alone with its batch sizes, strides and per-GPU process count. On SM107, varied-shape FP32 strided-batched cuBLAS can fail or hang; the shared reference uses ordinary GEMMs per broadcast batch (`sdpa/graph/test_mhas_v2.py::test_sdpa_reference_matmul`).

### Reference cache (opt-in)

`sdpa/ref_cache.py::cached_reference(name, key, compute)` stores oracle outputs on disk. Off unless `CUDNN_TEST_REF_CACHE=<node-local dir>` (never a cross-site network file system).
- Key: content of `_REF_SOURCES` (closed under its own imports; `sdpa/test_ref_cache.py::test_the_reference_sources_are_closed_under_their_own_imports`), full content of every non-library file on the call stack (recipe module, wrappers, test module), device class (name, cc, SM count), torch version, and the caller's `key` (shape, seed, scale, mask, dtype: anything passed as a parameter). A helper the recipe has already returned from is covered only via `_REF_SOURCES`: list it there.
- An edit made during a cache-enabled run keys later entries by new content: rerun once with VERIFY or use a fresh directory.
- `CUDNN_TEST_REF_CACHE_VERIFY=1` recomputes on every hit and fails on drift.
- Entries load with torch's weights-only unpickler; keep the directory writable by the test account only.
- `sdpa/frost/test_sdpa_bwd_fp8_sm107.py::_run_fp8` also memoizes in-process (bypassed under VERIFY).

### Warm stress for shared-memory stage hand-back

A two-launch equality check is not a race test. For a kernel that hands a shared-memory stage back to its producer (Rule 11 in [python/cudnn/AGENTS.md](../python/cudnn/AGENTS.md)), use `warm_stress.assert_warm_stress(StressCase(...))` with a one-launch callable on the shipping launch path and every tensor the kernel writes (its own storage, never a copy). Example and control: `sdpa/frost/test_tile_dsl_consume_then_arrive.py`; report pin: `core/test_warm_stress.py`.
1. The shape refills the stage inside one launch: state `refills_per_launch` (zero refused).
2. State `thread_blocks`; prefer at least two thread blocks per SM.
3. A zero count passes only where `stress_validation()` saw the unsafe twin fail on this GPU; otherwise skip `warm stress not validated on <part>` and record via `frost_routing.measured`.
4. Byte equality only; non-bit-reproducible kernels cannot use it.
5. Give an independent `reference` whenever one exists. An FP64 oracle with tolerances is not a byte reference: validate both cold launches against it, keep byte comparisons for warm launches (`TestD64ThdPackedUnsplit.test_warm_refill`).
6. Default 3,000 + 3,000 launches (0.1 % rate bound); larger counts are local runs, never L0. The 60 s budget skips as incomplete on a busy GPU: mark `gpu_exclusive` with its xdist group; never raise a kernel timeout. Under `process_watchdog`, call `warm_stress.build_launchers()` and build the kernel before arming the watchdog.

### Poisoning, numerics and quantization

- **Poison caller workspace, including physical padding**: fill with both zero and `0xff`, poison outputs, compare eager and captured results (`test_delta_workspace_padding_does_not_feed_dkdv`). Keep the original cold initialization when adding warm/replay controls (`TestSM120EmptyQueryCompatibility`).
- **Poison unused attention storage** with NaN, infinities and large finite values; require unchanged valid gradients and zero unused gradients in eager and replay. Check `+inf` sinks against a finite dominant-sink control.
- Sink logits at both infinities with live Q and zero KV iterations: disabled sink gives O=0/LSE=-inf, dominating +inf sink gives O=0/LSE=+inf (`test_native_thd_paged_sink_endpoints`). A strided graph sink descriptor with a flat carrier passes the declared head pitch to the native THD host; validate sink presence/dtype/device/span before length writes (`test_native_thd_paged_sink_rebind_replay`).
- Pair very negative LSE with large finite dO; use an analytic reference.
- **FP8 references mirror intermediate P rounding**: use `sdpa.fp8_ref.compute_ref` with the kernel's P scale and rescale threshold (`test_paged_adapter_fp8_compile_key_canonicalizes_the_logical_kv_maximum`). Match softmax precision (HALF vs FLOAT) before budgeting FP8 rounding; for split-KV derive bounds per partition including the sink.
- **Every declared quantized output is asserted, `Amax_O` included.** `sdpa.fp8.assert_amax_o` checks Amax against the reference's fp32 pre-cast, pre-scale amax (within one P code step) and against the kernel's stored O (within half a code spacing). Prefill outputs with NaN, never zero. Backend-served graphs (`graph.selected_engine is None`) xfail with `BACKEND_AMAX_O_ISSUE`; FROST-served graphs fail. An empty live O requires a finite, exactly zero Amax_O. Pins: `sdpa/test_assert_amax_o.py`; `sdpa/graph/suites/context/test_context_mxfp8.py::test_context_mxfp8_thd` (all-Q-empty).
- Check amax outputs for finiteness before repeat comparisons; `sdpa.helpers.exact_equal` treats matching NaNs as equal (#2122).
- **An fp8 midpoint flip is proved from the reference's intermediates.** `assert_close_fp8_grad(..., operand=, flip_unit=, intermediates=, fp8_dtype=, out_dtype=)` lifts the `4 * atol` row cap only when a valid position of the row's own reduction sits within 1/32 code spacing of a midpoint and one flip reproduces the row (ordinary tolerance, per element, and least-squares flip count 1 within 1/2). Packed outputs and `h_k != h_v` keep the plain cap. Pin: `sdpa/test_fp8_flip_budget.py`. See [../AGENTS.md](../AGENTS.md) for the per-lane dataset check.
- MXFP8 backward: `assert_close_mxfp8_grad` caps the magnitude of every extra mismatch (`sdpa/test_mxfp8_grad_budget.py`). Emulating the exponent FFMA: round the multiplier and the Stats term before the fused rounding; pin in `sdpa/test_mxfp8_ref_quantize_ds.py`.
- Low-precision quantization needs exact signed midpoint tests with non-power-of-two block scales; compare the quantization stage before diagnosing attention gradients.
- Decode Stats are tested independently of training: request Stats explicitly, NaN-init every head, include padded-Stats, MHA and `s_q > 1` controls, pin a backend plan (`test_mhas_v2.py::test_sdpa_ragged_decode_stats`, NVBug 6783545; `--runxfail` on older backends).
- For FP8/MXFP8 batch changes, quantize each sequence once and keep payload/scale bytes. Columnwise F8_128x4 scale atoms span all B*H rows: unswizzle, rebatch, reswizzle without requantizing (`test_v_tail_scale_repack` in `test_sdpa_fp8_v_tail_sync_sm120.py`). Include a scale-only change; keep SM90 FP8 and SM100/SM103/SM107 MXFP8 results separate. An architecture skip is not a pass.
- Rebind scale buffers with different values, poison and rebind amax, change scales in place after capture (`test_prepared_fp8_rebind_scales_and_buffers`, `test_prepared_fp8_capture_replay_reads_current_scales`).
- A harness mirror of an engine's admission rule includes the engine's DSL target gate (`sdpa/mxfp8.py::block_scaled_o_draw`; `sdpa/test_mxfp8_block_scaled_fold.py::test_cc107_default_admission_needs_the_sm_107a_dsl_target`); a mismatch fails flag-less d128 draws of `test_sdpa_mxfp8_fwd_L0`. FP16/BF16 architecture extensions do not extend MXFP8 rows; SM110 must fold.
- For a GEMM+GLU failure, compare the output with the stored GEMM intermediate and an independent dot product; SwiGLU pairs alternate 32-column blocks (`test_swiglu_failure_diagnostics.py`).
- Synthetic LayerNorm/RMSNorm nodes keep saved statistics as (N,1); run `test_frost_norm_engines.py` and `test_frost_norm_engine_e2e.py` after tightening validation.
- Fused RoPE table preprocessing keeps FP32 conversion before sin/cos and full-range trigonometry (`test_rope_table_large_angles_and_special_values`).

### fp32 operands and TF32 references

Blackwell MMAs run fp32 operands as TF32. Compute the torch reference with `torch.backends.cuda.matmul.fp32_precision = "tf32"` set explicitly in the test; the default differs between NGC containers (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`) and pip torch.

### Clamp gradients at the boundary

torch 2.14 changed `clamp`'s backward exactly on the limit. State the kernel's convention with `torch.where(inside, x, x.detach().clamp(lo, hi))` (`test_grouped_dgeglu_autograd.py::clamp_closed`, #1522).

### Sanitizer diagnostic controls

Keep `--dump-kernel-launches` out of the pass/fail sanitizer run: some builds count dump records in `ERROR SUMMARY`. Verify with a valid-memory control and a known-bad canary. Run racecheck for scratch reuse (sparse metadata, cross-warp reductions, paged rings, V-scale tails).

## Determinism and batch invariance

- The shared comparator checks bytes, names, shapes and dtypes. Clone each result before the next launch. Keep these suites in L0, including stress cases.
- Name parameter IDs with direction, dtype and shape; report collected cases separately from replay executions. Cover tile/routing boundaries, grouped-head reductions, duplicate sequences, changed neighbors and isolated upstream-gradient sources.
- Backward batch checks slice fixed upstream gradients with the inputs; never a batch-mean loss. Validate with all gradients.
- GDP expands K/V/beta by `num_householder`; Q/G stay on the token axis. QSA packing shifts token and block metadata and keeps each sequence's selection mask.
- DSA: shift valid Top-K indices by the KV offset, keep invalid ones invalid; use unique active indices except in cases pinned to an SM100/SM103 conflict-rank dKV layout (assert that layout). dKV is not batch invariant once a batch spans several waves: compare forward, dQ and generic dSink and name the gap. H128 D576 has no public forward kernel; feed reference forward results. Isolate the target's upstream gradient for the shared sink gradient; check signed infinity sentinels before byte comparison.
- DSA indexer backward: restore mutated score buffers, keep `loss_coeff / query_count` and `grad_loss` fixed, compare all three gradients and scores, test BF16/FP32 dK separately, remap Top-K IDs and keep negative padding. Cover large TopK with more than twice as many query rows as producer planes; poison workspace/outputs; use `assert_bitwise_runs` (`test_DSA_indexer_backward_deterministic_topk2048_boundary`, L1).
- Split batch-invariance fixtures reserve total-Q capacity for inserted/duplicated sequences and keep `max_total_seq_len_q`; head-major Stats controls need storage for `H * head_stride`.
- API-by-dtype/GPU support goes in [the status page](../docs/operations/determinism.md); counts, commands, runtimes and diagnostics go in [the test report](python/determinism_goldens/README.md).

## SDPA and FROST test contracts

### Eligibility, ownership and routing

- When widening SDPA eligibility, also run `sdpa/frost/test_heuristic_geometry.py` and `sdpa/frost/test_sdpa_graph_analyzer.py` (some cases need a GPU and backend handle). Keep negative coverage at a still-unsupported boundary.
- Layout-only admission fixtures provide real `SdpaGraphFacts` geometry matching their tensor declarations, not a `SimpleNamespace`.
- Shared SDPA helpers accept public knob mappings and native knob dataclasses; normalize with `SdpaFwdKnobs.from_public` (`TestNativeD512HalfPaged::test_dense`, `TestNativeD576Half::test_paged`).
- Engine splits / owner migrations:
  - Update shared `test_mhas_v2` routing assertions and explicit plan pins (engine ID and physical cluster knob); keep concrete owner/body assertions so native fallback cannot pass.
  - Audit legacy-name prefix filters, helper defaults, behavior-note controls and compiled-module spies; negative controls select an eligible current owner.
  - Run `test_band_capabilities.py`, `test_sdpa_frontend_integration.py` and affected paged/native-binding cases. Band-matrix facts include current Q/K/V/O declarations. A partially migrated row is sampled for both retained and delegated masks (`test_band_capabilities.py::test_model_and_probe_agree_on_every_band_sample`).
  - Migrate non-exact fixture widths; use the implementation's domain predicate (SM107 D24 bias, D248/V224 split, D80 FP8 target tests). Scalar-tail, THD-interior and D512-origin fixtures resolve their covering physical shape.
  - When a native flavor replaces an envelope, update the exact template expectation in end-to-end routing tests (`test_sdpa_fwd_cc107_decode_tile_envelope_pinned_L0`). D64 with CGA1 selects decode or prefill by query envelope; straddle the boundary (`TestPagedD64PhysicalOrigins`).
  - When a fixture turns an unsplit record into a split record, select the implementation after computing final knobs (`test_paged_thd_split_capture_lengths_and_stats`, `test_native_paged_thd_capture_and_rebind`).
- Numerical tests of an explicit split-KV leg construct that plan with `create_execution_plan`; keep heuristic-offering tests separate (`test_packed_and_split_legs`).
- A fixed engine sharing a selector with a fallback family: reject fallback selections before recommending it; test eligibility, recommendation and standalone admission on both sides of each bound with metadata-only descriptors.
- A capability row's `sm_lo` is a lower bound; prepared admission respects the adapter's exact device support (`test_prepared_fp8_override_capability_envelope`).
- SM107 metadata/heuristic tests run on non-SM107 lanes: model compiler target availability explicitly; keep real target-decline tests separate.
- A plan compiles the schedule its support check validated (CSA ratio-128 backward pins `nb_total`).
- Plan selection respects current filters: build-all after an explicit selection excludes that selection, and a stale candidate must not turn all-rejected into success (`test_build_all_preserves_backend_selection_queries`, `test/cpp/plan_selection.cpp`). Support alone does not count as a built plan.
- Native candidate compilation: test each offered candidate before first-success and ALL policies (`sdpa/frost/test_sdpa_native_decode_candidates.py`, #1084).
- Restored-plan metadata: compare count, name, engine/knob record, notes and workspace; preserve every offered knob including implicit defaults and unknown newer types (never collapse to `NOT_SET`); keep deprecated knob identities valid. Run the C++ `[knob-vocabulary]` detector with older and current headers and `core/graph/test_restored_plan_metadata.py` (convolution, matmul, SDPA).
- A shared SDPA harness's declared-total switch: inspect the backward node before planning (`test_fp8_training_declared_totals`).
- Torch varlen providers: test each op's schema independently (`test_torch_provider_dispatch.py`); keep public-selector rejections separate from direct operator numerics. Check routing by recording the native operator call, not stderr logs.
- A compiled-helper test does not cover AOT backward: keep raw-pointer helpers behind a registered custom op with a fake implementation; run `torch.library.opcheck` (`sdpa/torch/test_torch_ops.py::TestOpContract::test_opcheck`).
- A saved Stats tensor can have `requires_grad=True` with grad mode disabled (`test_varlen_backward_uses_prepared_stats`).
- When an SDPA device helper gains operands, audit subclasses in other packages (BSA SM120; `test_sm120_blk128_fixed_topk`).

### Architecture reuse and retiring kernels

- A new architecture reuses the shared forward harness `sdpa/frost/test_sdpa_fwd_dsl_sm100.py` through its module-level `_ARCH` (fed to `engine_name(arch=...)`), not by copying it:

  ```python
  @pytest.fixture
  def sm100(monkeypatch):
      import test_sdpa_fwd_dsl_sm100 as module
      monkeypatch.setattr(module, "_ARCH", "sm90")
      return module
  ```

  The delegating file carries its own device gate and calls runners as functions (the source's `pytestmark = requires_blackwell` skips elsewhere). `test_sdpa_compiled_cache_gpu.py` keeps `_ARCH` inside its `_CHILD` source string: parameterize it. Detector: delete the arch's row from `ENGINE_SPECS`; every strict `select_engine` pin must fail with `no plan for engine ...`.
- Architecture modules that re-export shared checks must not import back from the architecture module; move shared helpers to a non-`test_` module (CodeQL cyclic import).
- Retiring/moving a kernel: audit dotted module names (`from package import old_module`, `find_spec`, `import_module` strings), embedded subprocess scripts, dynamically built fixture paths, standalone `_main()` callers, and dynamic module exports (`_host_thd_split`). Detectors: `test_dynamic_kernel_fixture_paths_exist`, `test_sdpa_native_dense_binding.py`, `test_sdpa_native_prefill_binding.py`, `test_sdpa_fwd_pointer_abi.py`, `test_sdpa_fwd_o_empty_backedge.py`, the MXFP8 credit-arrive inventory in `test_sdpa_fwd_mxfp8_sm100.py`, `native_fp8_decode_sm100.py`, `test_native_d576_compact_thd_rebind_and_capture[split2_unpacked]`.
- When splitting a template or body: keep module-load refusals (`test_per_tensor_body_refuses_mxfp8_only_mask_at_load`), pin each body's barrier sites separately, and identify bodies by owner directory (a `kernel.py` basename is generic).
- When retiring a fake-tensor builder, move negative guards to `cute.runtime.make_fake_tensor` / `make_fake_compact_tensor`; patching a deleted helper with `raising=False` proves nothing.
- After a kernel consolidation, trace each public wrapper to the kernel it compiles and run regressions through the public wrapper.
- PTX probes keep earlier arguments live so parameter positions do not renumber. Before refreshing a PTX hash record, render historical and current source/probe with the same compiler distribution and compare (`test_fork_rendering_ptx_md5_is_recorded`). A store-order optimization for one width keeps other widths' lowering (SM107 bias/off/streamed records). Compiled PTX entries are not launch counts; keep a launch-counter test.
- Unchanged device-function ASTs do not imply unchanged code; a unit-stride fast path also exercises nonunit strides (`test_d256_paged_host_rebinds_table_column_stride`).
- Compiled DSL call arity excludes `cutlass.Constexpr` parameters and the stream keyword (`test_every_combine_call_site_matches_the_compiled_arity`).
- A standalone FROST adapter compiles at plan time: every driver calls `compile()` before `execute()`. Detector: an `.execute(` without a preceding `.compile(` in the same driver (`sdpa/frost/test_sdpa_fwd_sparse_sm107.py`, `sdpa/frost/sparse_item_sequences.py`).
- A test that spies a trace-time call sets both `CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0` and `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` (`_bs_dq_arm` in `sdpa/frost/test_sdpa_bwd_mxfp8_sm107.py`). Disk-artifact hit tests clear the process memo before population and reload.

### Launch grids and wide addresses

- Test launch bounds against each actual scheduler: 65535 and 65536 independently for batch and heads, in graph admission and standalone checks, for every launch in a multi-kernel chain including auxiliary/delta preprocess launches (`sdpa/frost/test_sdpa_bwd_launch_grid.py`, `sdpa/frost/test_sdpa_bwd_d512_quantized_bounds.py`, `test_sdpa_bwd_d512_half_grid.py`, `test_sdpa_bwd_thd_fold_grid_determinism.py`). Cover each extent below its limit with the product above it. Keep runtime max-length arguments unspecialized.
- A flattened launch bounds the physical grid and iterates Int64 owners; keep live canaries past grid.x's limit and past 2^32. Skip only an explicit unsupported-target compiler diagnostic.
- Wide strides stay Int64 through every host and setup argument, including page tables, `lse_ext`, SF strides (16-byte units) and ragged divisors. Exercise physical strides above `2**32` with two live sequences and a narrower stride whose product overflows, with wrapped addresses poisoned inside allocated guard storage. Detectors: `test_graph_decode_prepared_keeps_int64_page_table_batch_stride`, `test_thd_output_row_stride_above_int32_reaches_device_descriptors`, `test_mhas_v2.py::test_sdpa_thd_output_stride_int64`, `test_sdpa_prepared_stats_int64.py`, `test_mxfp8_v_scale_plane_stride_multiplies_in_int64`, `test_prepared_mxfp8_sf_head_stride_above_int32_units` (L1), `test_sf_repack_steps_past_int32_with_allocated_guards`, `test_quantized_gate_batch_stride_above_int32`, `test_sdpa_native_ragged_decode_binding.py`, `sdpa/torch/test_varlen_metadata.py`, `test_sdpa_sm80_packed_init.py`, `test_sdpa_sm80_packed_metadata.py`. Cover every served arch/flavor.
- Compile-time stride fakes for Int64 slots use Int64 values; a plain Python `0` can infer Int32. A singleton axis with a stride above `2**31` catches narrowing at binding time without a large allocation.
- Default-provider passes alone cannot qualify an advertised native path; pin the native route.
- A CuTeDSL `cutlass.Array` scalar index is a flat element offset; form non-contiguous offsets in Int64. Use full multidimensional indexing for wide-stride global stores.
- Multistage descriptors check the end of the highest addressed stage against the address window (`TestNativeD576Half.test_admission`).
- When fixing grid bounds, audit auxiliary launches too.

### Native binders and prepared hosts

- Derive host frames from each actual template signature and compare native with Python binding; do not infer one architecture's ABI from another (`test_sdpa_native_arch_binding.py`, `test_sdpa_native_sm120_binding.py`, `test_sdpa_native_bwd_binding.py`, `test_sdpa_native_sm90_binding.py`, `test_sdpa_native_sm80_binding.py`, `test_two_by_two_host_slots_match_role_split`). AST frame fixtures keep unannotated arguments (`argument.annotation is None` before `ast.unparse`); only `Constexpr` removes a slot. Raw positional entries require every runtime slot, including `None` pointers. SM120 appended untyped origin arguments: `test_thd_fp8_actual_host_frames_and_current_pointers`.
- Partial dtypes follow the selected host: SM120 writes half partials via `o_ptr`; SM100 and split-capable SM107 per-tensor FP8 write FP32 via `o_partial_ptr`; SM100 D512 and SM107 MXFP8 write half; the D256 mixed-KV paged host writes FP32 (`test_sdpa_native_fp8_arch_binding.py`, `test_sdpa_native_mxfp8_binding.py`, `TestMixedFp8Kv.test_d256_dense_paged` for split/unsplit paged O/Stats, `test_d256_standalone` for the FP32 partial ABI).
- Separate QK and V widths through all geometry (`test_sdpa_native_prefill_binding.py`, `test_sdpa_sm80_thd_native_widths.py`, `test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_widths`).
- Keep observed storage separate from effective geometry; overrides cannot enlarge the allocation (`test_sdpa_native_thd_binding.py`, `test_sdpa_paged_binding.py`). Workspace capacity: `None` for unknown, reject every observed capacity below required scratch including zero (`core/frost/test_workspace_capacity.py`, `core/graph/test_execute_workspace.py::test_short_backend_workspace_rejects_before_any_write`).
- Validate every operand and every final output before the first launch, copy or write; poison workspace and outputs and verify sentinels survive rejection (`test_sdpa_native_split_binding.py`, `test_dense_native_rejects_invalid_operands_before_launch`, `test_native_auxiliary_outputs_validate_before_writes`, `test_sdpa_native_thd_fp8_binding.py`, `test_sdpa_native_padded_stats_binding.py`, `test_sdpa_native_block_output_binding.py`; short backing span: `test_sm80_revalidates_every_current_operand[short]`). Standalone quantized THD requires exact Q/KV carrier counts.
- Alias checks: carved scratch versus operand byte spans (packed FP4, strided views, optional outputs, pointer tables); for staged launches, check against original caller operands (`test_staged_amax_alias_checks_original_operand`, `test_staged_sf_alias_checks_original_operand`). Keep other operand ranges clear of the tested interval.
- Gradient overlap uses declared address sets and current pointers (`test_gradient_overlap_uses_declared_graph_addresses`); keep registered `BwdLaunchSpec.name` (`test_sdpa_bwd_thd_sm120.py`, `test_sdpa_bwd_config_sm100_2x2.py`).
- Stats packing: preserve the declared packing for ambiguous shapes (`test_ragged_square_stats_follow_declared_packing`, `test_standalone_thd_token_major_stats`, S=1 and T=H=1 cases).
- Physical origins: use `ExecConfig.ragged_origin_rows`; give O independent Int64 offsets when its stride changes; leave gaps between sequence origins (`test_mhas_v2.py::test_sdpa_physical_origins_L0`, `frost/test_sdpa_physical_origins_shared.py`, `test_sdpa_fwd_fp8_sm100.py::TestD128Fp8PhysicalOrigins` (prepared and standalone), `TestD256Mxfp8PhysicalOrigins`, `TestMidwidthPhysicalOrigins`, `TestPagedD64PhysicalOrigins`, `TestPagedD256PackedSplit`; `test_native_d256_paged_split_frames` compares both binders and rejects partial buffers overlapping origin metadata). Scaled ragged offsets keep d_qk for Q/K/dQ/dK and d_v for V/O/dO/dV (`sdpa/graph/test_ragged_offset_multiplier.py`).
- Bind tensor views in the graph's declared axis order; H == S hides a missing transpose (`test_sdpa_fp8_paged_equal_head_and_query_axes`).
- MXFP8 scale factors: THD uses `pack_thd_columnwise_sf` (`test_thd_columnwise_sf_matches_independent_atoms`; keep D512 multi-head); the split combine specializes its scale pointer to None (`test_prepared_mxfp8_capture_reads_current_scales`). MXFP8 input does not imply three SF buffers or one-byte V: the PV-BF16 specialization consumes only SF_Q/SF_K, keeps BF16 V/O, binds `sf_v_ptr=None` with a zero third SF tile count, keeps fixed role positions and rejects a supplied SF_V before launch (`test_sdpa_native_pv_bf16_binding.py`).
- Recorders that replace `spec.fn` rebuild `type(spec.native)`; oracle substitution also clears the retained native ordered entry.
- Keep `stream` last in private CuTe host signatures and the `_positional_order` guard.
- Prepared artifacts survive a disabled persistent cache and fresh processes: avoid dataclass `Constexpr` arguments, compute `template_key(globals(), locals(), ...)` before local imports, give shared kernels a source identity (`test_wrapper_capacity_reuses_artifact_without_disk_cache`, `test_block_output_artifact_reloads_in_fresh_process`).
- After moving a wrapper to a prepared host, forbid the old tensor launcher/JIT and check fresh bindings plus changed-input replay (`test_sdpa_sm80_thd_forward_prepared.py`, `test_sdpa_prepared_sm90.py`, `test_sdpa_bwd_native_layouts_sm100.py::test_native_rebind_stream_capture` (aligned pitches, no hidden layout copies; unaligned outer pitches are negative admission cases), `test_dense_native_replay_or_rope_decline`). Test both graph prepared admission and the standalone adapter's compiler (`test_fp8_paged_prepared_table_stride_admission`, `test_fp8_paged_distinct_table_strides_prepared`); run existing prepared-launch tests on binder migrations (`test_sdpa_prepared_thd.py::test_split_prepared_strided_output_rebind_and_capture`).
- Wrappers: caches key every input stride (`test_wrapper_cache_distinguishes_current_input_strides`); allocating wrappers pass per-call scratch on the launch stream (`test_wrapper_scratch_survives_explicit_stream_consumer`, `sdpa/frost/test_sdpa_fwd_dsl_sm120.py::TestStagedSm120Wrapper`); AMAX resets stay on the launch stream (`test_swiglu_wrapper_outputs_and_amax_use_launch_stream`); SM80 packed wrappers keep cumulative tensors as row offsets (`test_thd_wrapper_preserves_packed_row_origins`) and singleton-Stats pitches (`test_sdpa_sm80_native_wrappers.py`); metadata fusion keeps a cheap compact-sink path (`test_forward_metadata_compact_sink_needs_no_compiler`, `sdpa/torch/test_aux_metadata.py`).
- Direct split SDPA workspace: intercept allocation at `API.execute` too (`test_standalone_split_requires_caller_workspace_without_allocating`); exercise the actual standalone adapter (`test_shared_standalone_adapter_uses_current_native_storage`).
- Optional-output flags and sample descriptors agree at every host boundary (`test_pv_bf16_no_amax_flag_with_sample_descriptor`). A `None` compile sample may still occupy a positional FFI slot (`test_native_packed_physical_stride`).
- Gate layout admission shares the adapter's TMA predicate (`test_gate_batch_stride_alignment`). The sigmoid gate changes O only; Stats and Amax stay ungated.
- SM120 masks the rightmost KV tile for every mask mode (`test_sm120_prepared_bounded_geometry_override`). SM120 decode stage width can be 48; intersect with `config_sm120.decode_kv_stage`.
- Paged ring-loader checks use more K/V stages per split than ring slots. A raw-copy barrier phase invariant keeps exact chunk-count guards (`TestMixedFp8Kv.test_d512_admission`). The predecoded THD scheduler returns slot credit after all lanes' reads (CGA1 and CGA2).
- Sparse score-recompute support tests mock both `device_capability()` and `torch.cuda.get_device_capability()`; test non-power-of-two TMEM slot counts and the usable 227 KiB SMEM bound (`test_DSA_sparse_score_recompute.py`, `test_DSA_sparse_attention_score_recompute_uses_launchable_smem_budget`).
- Large-head backward keeps contraction width separate from output-column tile (`test_sdpa_bwd_d512_fp8.py`); unequal-width dK/dV reductions keep masked vectors in their interval (`test_sdpa_bwd_thd_d192_sm100.py`).
- TMA store stages are counted in committed groups (`test_gemm_swiglu_retained_outputs_replay`). Bulk MoE COMBINE: `gemm/frost/test_moe_combine_bulk.py`; run `test_two_cta_atomic_fallback` under compute-sanitizer on SM100 before widening normal 2-CTA bulk.
- Shared auxiliary kernels need direct boundary tests and caller checks (THD fold inputs FP32 or half; test the full output allocation).

### Devices, streams and contexts

- Direct standalone adapters select the plan device before entering a prepared host and restore the caller on failure (`test_direct_standalone_with_another_device_current`). `stream_context(None, device)` is a no-op. Staged multi-GPU copies resolve the stream on Q's device; check each operand device's architecture before launching on it.
- Torch/JAX DLPack: select the tensor's device during conversion and restore afterward; initialize the JAX backend inside an outer device guard. JAX target admission covers zero-argument JIT closures (`test_fully_captured_public_call_refuses_mixed_arch_under_jit`).
- Pass the operand architecture explicitly to the compiler and key artifacts by it; preserve a Torch fallback when the runtime entry is absent on heterogeneous hosts, while propagating ordinary errors.
- Automatic handle caches isolate device and thread (`core/graph/test_wrapper_graph.py`, `test_auto_handle_cleanup_uses_creation_device`, `test_auto_handle_cleanup_retries_failed_handle`). Indexed execution keeps each caller's plan (`core/test_dispatch.py::test_indexed_execution_keeps_each_callers_plan`, `core/graph/test_execute_overload.py`).
- A stream wait on a never-recorded event is a no-op. Delayed-consumer probes assert the completion event is still pending; use a bounded byte consumer for recycled-storage negative controls.
- CuTe DSL 4.7.0/4.7.1 can leak an init lock on cold concurrent first calls: initialize serially, then test concurrent frames (`test_native_thd_concurrent_streams_use_independent_frames`).
