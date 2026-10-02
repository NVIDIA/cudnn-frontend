# test — Agent Guide

Recurrent FP64 autograd references can retain a K x V state for every token
and head. When a production stress case scales heads with the GPU SM count,
bound the reference by independent head chunks while keeping the full kernel
workload and tolerances. Compare every chunked gradient with the original
whole-head reference on smaller cases, including ragged/empty sequences and
partial chunks; preserve grouped-head reductions unless explicitly validated.
An OOM in the reference after kernel comparisons passed is not evidence of a
kernel allocation failure. Keep that attribution explicit in CI triage.

Two suites: `test/cpp` (Catch2, C++ graph API) and `test/python` (pytest). Both need an NVIDIA GPU and a cuDNN 9.x backend at runtime. Build/install commands: [../AGENTS.md](../AGENTS.md).

## C++ tests (`test/cpp`)

- Catch2 v3 binary, target `tests`, built by the default CMake build (`CUDNN_FRONTEND_BUILD_TESTS=ON`) into `build/bin/tests`.
- Run all: `./build/bin/tests`. List: `--list-tests`. One case: `./build/bin/tests "Validate conv node"`. Filter by tag: `./build/bin/tests "[serialize]"`.

## Python tests (`test/python`)

Run from `test/python` so `pytest.ini` and `conftest.py` apply:

```bash
cd test/python
pytest                       # pytest.ini addopts default to -m L0 (smoke) --tb=short --no-header
pytest -m L1                 # levels L0..L4; higher = larger sweeps
pytest -n 4                  # pytest-xdist; mind marker gpu_exclusive for tests that need the GPU alone
pytest conv/graph/test_conv_fprop.py  # one file — note the default -m L0 filter still applies
pytest gemm/cutedsl/                  # CuTe DSL kernel tests
```

**Read pytest's own summary line; do not post-process the output.** `... | grep -c "passed"` counts a collection error as a pass, and `cmd | tail` reports *tail's* exit status, so a failed build or a failed suite behind a pipe looks like success. Both have produced confidently wrong "all green" reports here. Requirements: `pip install -e .` plus `pytest pytest-xdist looseversion`. The `cutedsl/` directories additionally require an SM90/SM100-class GPU; tests skip (or should skip) on unsupported arch/dtype/backend-version combos rather than fail.

### conftest.py landmines — read before editing

- Source-package subprocess probes must exclude `__pycache__` and bytecode
  when copying a tree shared by xdist workers. Python atomically renames its
  temporary `.pyc` files, so `copytree` can enumerate a file that disappears
  before it is copied (the causal-conv1d import-contract probe hit this in CI).

- `PYTORCH_CUDA_ALLOC_CONF` is set at the very top, **before any torch import** (torch reads it once at CUDA-allocator init). Don't move it, and don't import torch in a plugin that loads earlier.
- `import transformer_engine` happens (in try/except) **before** `import cudnn` — TE and cuDNN conflict if loaded in the other order. Preserve this ordering.
- Crash isolation (`# Crash isolation` block in `conftest.py`): `pytest_cmdline_main` injects `-n1 --max-worker-restart=100000`, so a segfault, a poisoned CUDA context, or a hang kills only one xdist worker, which the controller replaces before continuing. After every test `pytest_runtest_logfinish` probes the context with `torch.cuda.synchronize()`; a per-test `faulthandler.dump_traceback_later(exit=True)` deadline (`CUDNN_TEST_TIMEOUT`, default 1500 s, `0` disables) covers the probe too. It is faulthandler's C watchdog, not a Python thread or `SIGALRM`, because a hung CUDA driver call holds the GIL and parks the main thread. Not injected under `-n<N>`, `-s`, `--pdb`, `--collect-only`, or `CUDNN_TEST_NO_ISOLATION=1`; without a worker to restart, a dead context stops the run via `pytest.exit` and a hang still hard-exits. Killing a worker does **not** stop a kernel it left running -- the driver keeps that context until the kernel ends, and the next worker can block behind it.
- A side-stream test must order input production before consumption: call `stream.wait_stream(torch.cuda.current_stream())` after creating inputs on the current stream and before switching streams. Synchronizing the consumer afterward cannot repair a missing producer dependency; the GAT/GATv2 current-stream probes exposed this under concurrent CI load.
- A session-scoped autouse `cudnn_handle` fixture creates one handle bound to a dedicated stream; use it instead of creating handles per-test. Tests that rebind it must save `cudnn.get_stream(cudnn_handle)` before setup and restore that exact stream in `finally`, including setup failures/skips. Restoring `torch.cuda.current_stream()` instead leaks a different stream into later tests.
- `pytest_configure` asserts `torch.cuda.is_available()` — there is no CPU-only mode.
- Many custom CLI options exist (`--dryrun`, `--repro`, `--seed`, `--perf`, per-op dimension overrides like `--b/--s_q`, `--nsa-*`, `--dsa-*`); check `pytest_addoption` before adding new ones.
- **Keep `sdpa/frost/` paths contiguous on the command line.** `sdpa/frost/conftest.py` sets `CUDNN_FRONTEND_ENABLE_FROST_ENGINES` through an autouse fixture, because the FROST manifest rows are opt-in. Interleaving a top-level path between two `sdpa/frost/` paths — `pytest sdpa/frost/test_a.py test_dispatch.py sdpa/frost/test_b.py` — silently drops that fixture for everything after the top-level file: `_frost_opt_in` is absent from `item.fixturenames` and no FROST row is offered. Any top-level file does it, not just `test_dispatch.py`, and both contiguous orders are fine. Tests that *pin* an engine then fail loudly, but a test that `pytest.skip`s when it finds no python plan goes **falsely green**. Detector, as its own plugin so no test file changes:

  ```python
  @pytest.hookimpl(hookwrapper=True)
  def pytest_runtest_call(item):
      print(item.name, os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES"), "_frost_opt_in" in item.fixturenames)
      yield
  ```

### Layout

- `test/python/<operation>/<backend>/` — tests grouped by operation (`sdpa`, `gemm`, `conv`, `norm`, `rope`, `linear_attention`, ...), then by the backend or framework they exercise: `graph` (native cuDNN through `cudnn.pygraph`), `frost` (FROST engines), `cutedsl` (CuTe DSL frontend-only APIs), `torch` and `jax` (framework integrations), and engine names such as `linear_attention/cake`. Tests that span backends sit at the operation root (e.g. `linear_attention/test_la.py`).
- `test/python/core/<backend>/` — infrastructure tests not tied to one operation (dispatch, import boundaries, graph execution, FROST buffers, ...).
- `test/python/test_utils.py` holds shared helpers; shared SDPA references are in `test/python/sdpa/`.
- **`test/python/sdpa/` is a mixed directory and the `test_` prefix is load-bearing.** `fp16.py`, `helpers.py`, `random_config.py` are harness modules the tests import; `sdpa/test_*.py` (and `sdpa/frost/test_*.py`) are collected tests. `pytest.ini` sets no `python_files` override, so a test file dropped there **without** the prefix is silently treated as a helper — it is never collected, and the suite stays green while asserting nothing. After moving or adding a test, confirm it is picked up by the *default* sweep, not just when named directly:

  ```bash
  pytest --collect-only -q | grep -c sdpa/torch/test_torch_ops.py
  ```

### Conventions for new tests

- Mark with a level (`@pytest.mark.L0` ... `L4`): L0 must stay fast (default CI smoke); big parameter sweeps go to higher levels.
- **Default L0 coverage is not sufficient if the CI target excludes the provider.** Check the actual CI path and `-k` filters. The general Python target excludes FROST cases, so representative shared-API FROST tests also need collection under `sdpa/frost/`; `test_sdpa_ordered_bindings.py` reuses the shared ordered-binding smoke logic. Verify both target collection and execution on a supported GPU.
- **Check for a module-level `pytestmark` before adding per-test markers.** Many files apply a level or capability marker file-wide (`pytestmark = ...` near the top); duplicating it on each test is noise, and suggesting it in review wastes a round-trip (recurred on PRs #814, #811, #797).
- Gate on capability, don't assume it: skip via `check_support()` failures, `cudnn.backend_version()`, and `torch.cuda.get_device_capability()`.
- Large physical-stride tests must handle allocation-time memory pressure: another xdist worker can consume free memory after `mem_get_info()`. The `gpu_exclusive` marker alone does not serialize ordinary xdist scheduling. Catch `torch.OutOfMemoryError` only around the large test-storage allocation and skip for unavailable resources; never catch the launch or numerical assertions. Retain a successful physical run with sufficient memory.
- Compare against a reference implementation (see existing `*_ref.py` / `*_reference.py` patterns) with dtype-appropriate tolerances.
- **Scale the tolerance to the tensor, not to the dtype alone.** A fixed absolute bound quietly becomes wrong when magnitudes grow: GQA dK/dV sum over `h_q/h_kv` query heads, so at a group size of 4 the *relative* error stays ~0.5% while `|dv|` peaks near 9.6 and blows a bound that passed at `h_kv == h_q`. Compare against `TOL * max(|ref|.max(), 1.0)`, or the next GQA ratio someone adds will look like a correctness regression.
- Shape-override tests must cover a backend lowering decline as well as a lowered graph. Ragged-offset tensors are backend-only operands and can be absent from the Python-only layout; filter those auxiliary overrides against `_variant_pack_uids()` while requiring every Q/K/V/O, Stats and length operand. `test_thd_cache_shape_grid_tracks_runtime_capacity` exercises both layouts without weakening capture, launch-bound or replay checks.
- **Heuristic tests must survive legitimate tuning changes.** Do not pin a particular
  workload's winning scheduler, packing or split count, candidate order/exact set,
  or a performance threshold. Do not turn the current measured/unmeasured shape
  boundary into a correctness contract: another independent optimization may
  legitimately choose a different plan for the same control shape.
  Test explicit knob admission and rejection, validity of proposed candidates,
  and transport of a mocked chooser's result; see
  `test_propose_preserves_recommendations_and_places_one_marker`.
  Check geometry against an independent oracle, rather than spying on private
  arithmetic helper calls. Kernel tests explicitly select supported knobs and
  verify O/LSE, changed-input capture/replay, and storage bounds.
  Exact expectations belong to semantic/API contracts, with the invariant stated
  in the test. Performance rankings and tuning boundaries belong in reproducible
  offline benchmarks with source/hardware attribution, not CI golden assertions.
- **A regression test must be seen RED.** Before trusting one, run it against the unfixed code — restore the old line, confirm it fails, restore the fix. `test_dsl_sm100_thd_interleaved_kv_views` and `test_varlen_backward_does_not_sync` were both checked this way, and both were genuinely red beforehand; a test written for a bug and never seen to fail is asserting an unknown.
- **Poison unused attention storage.** Use independent indices; poison unused KV with NaN, infinities, and large finite values. Require unchanged valid gradients and zero unused gradients in eager execution and graph replay. Check `+inf` sinks against a finite dominant-sink control.
- **Pair very negative LSE with large finite dO.** Exponent clamps can still overflow in dS. Use an analytic reference and confirm the test rejects masking after the product.
- **Low-precision quantization needs exact midpoint tests.** Approximate reciprocal multiplication can move an exact E2M1 tie across its rounding boundary even when the native conversion uses round-to-nearest-even. Include signed midpoint values with non-power-of-two block scales, and compare the quantization stage itself before diagnosing amplified attention-gradient differences.
- **Build the reference in fp64 when the bound is tighter than ~1e-3.** The DLFW CI containers run fp32 matmul in TF32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`; recent torch also defaults `fp32_precision` to `tf32` on Blackwell+), a ~3e-4 relative error per `Q @ K^T` logit. A 1024-column log-sum-exp averages it down to ~2e-5, a causal row with ONE valid column keeps it whole: `test_fp8_stats_is_the_exact_softmax_lse[causal]` read max|dLSE| 1.3e-4 against its 1e-4 bound on the sm107 lane (2026-09-15) with an exact kernel, and passed on a 208-SM node whose draws happened to be kinder. `sdpa/fp8_ref.compute_ref(dtype=torch.float64)` is the existing knob; in a hand-rolled reference `.double()` the operands before the matmul, not just the `logsumexp`.
- **An fp8 midpoint flip is PROVED from the reference's intermediates, never inferred from the output's shape.** One flipped P/dS code moves one output d-row by `(c_alt - c_ref) x descale x operand_row` (Q for dK, K for dQ, dO for dV, V for O) -- but a power-of-two multiple of an operand row is not evidence of one: a masked key, another batch's or another head's row, two identical rows each one flip away, or a step no adjacent pair of codes has (8192 in e4m3) all fit that description (review on PR #1075). `assert_close_fp8_grad(..., operand=, flip_unit=, intermediates=, fp8_dtype=)` lifts the `4 * atol` row cap only when, at a VALID position of the row's own reduction (same batch, a q head of the same GQA group, unmasked), the reference's scaled fp32 intermediate -- `compute_ref(..., return_intermediates=)` / `compute_ref_backward(...)`, re-run on the bad rows only -- sits within 1/32 of a code spacing of the midpoint between its fp8 code and the adjacent one, and that single flip reproduces the row three ways: within the ordinary tolerance, per element within the output's own rounding (`atol` plus half an output code spacing from each side -- both sides are dequantized output codes; pass `out_dtype=torch_otype`), and as a whole (the least-squares number of flips fitted to the row is 1 within 1/2 -- two flips fit at 2, and on a row of large gradients `rtol * |expected|` is itself two flips wide, so the ordinary tolerance alone would take three flips for one). It prints the position, the two codes, the step and the three fits. Packed (ragged) outputs keep the plain cap, as does `h_k != h_v`. `sdpa/test_fp8_flip_budget.py` pins the accepted case and each of those rejections on a problem the reference itself built. The negative-score q rows have amplitude 8 at d192, so a legitimate flip moves a dK row by 0.5 -- the fixed cap (0.32) alone would call that a defect (sm107 212-SM lane, test310, 2026-09-15).
- **Decode Stats must be tested independently of training.** The random SDPA harness uses `generate_stats=cfg.is_train`, so its `s_q == 1` inference sweep checks O without checking LSE. For ragged GQA decode, request Stats explicitly, initialize every head to NaN, and compare every head against the reference. Include padded-Stats, MHA, and `s_q > 1` controls; pin a backend plan when testing native codegen so FROST cannot mask it. `test_mhas_v2.py::test_sdpa_ragged_decode_stats` is the detector (NVBug 6783545); run with `--runxfail` when checking an affected older backend.
- **Plan-specific xfails must check the selected plan.** A heuristic can start ranking a working engine first without fixing another engine's compiler bug. An architecture-only xfail then turns correct outputs into strict-XPASS failures. Inspect `get_engine_and_knobs_at_index()` for the selected plan and keep `strict=True` plus the expected exception type for the affected engine. `test_sdpa_ragged_decode_stats` distinguishes the affected native 10X/107 engines (10/18) from working engine 8.
- **Seed before you allocate.** `torch.manual_seed()` after constructing the inputs seeds nothing that matters. Two runs meant to be compared then differ by data, and the assertion fails (or worse, passes) for a reason unrelated to what is under test — if two runs must be comparable, build the inputs once and reuse them.
- **Bound dense attention references independently of kernel memory.** A full
  `[B, H, S_q, S_kv]` FP32 score tensor and GQA `repeat_interleave` can exhaust a
  small GPU when xdist workers share it, even after the tested kernel succeeds.
  Compute independent heads separately, mapping each query head to its KV head,
  while preserving the full workload, masks, precision and tolerances. Validate
  the refactor against the dense oracle on small MHA/GQA/MQA cases and run the
  original large kernel test under the same allocator cap before and after the
  fix. `test_sdpa_fwd_split_kv_sm120.py` covers this pattern.
- **Every randomized SDPA input uses the per-test generator.** A seeded Q/K/V tuple is not a reproducible case if its block mask comes from the process-global CUDA RNG. Pass `generator=rng_data_gen` to auxiliary draws too; `test_block_mask_uses_the_per_test_data_generator` perturbs global RNG while holding the case seed fixed. Before attributing an order-dependent failure to an earlier engine, compare the actual masks as well as Q/K/V.
- **Compiled DSL call arity excludes compile-time parameters.** A `cutlass.Constexpr` argument belongs to the compilation signature and disappears from the compiled runtime call. When checking positional launch sites against `_host`, exclude these annotations as well as the stream keyword; do not add a runtime argument to satisfy an unfiltered Python signature count. `test_every_combine_call_site_matches_the_compiled_arity` is the detector.
- **A new architecture reuses the shared forward harness through `_ARCH`, not by copying it.** `sdpa/frost/test_sdpa_fwd_dsl_sm100.py` selects its engine architecture through a module-level `_ARCH` string fed to `engine_name(arch=...)`. A new row reuses its runners and assertions from a file carrying its **own** device gate:

  ```python
  @pytest.fixture
  def sm100(monkeypatch):
      import test_sdpa_fwd_dsl_sm100 as module
      monkeypatch.setattr(module, "_ARCH", "sm90")
      return module
  ```

  Two traps. (1) That module's `pytestmark = requires_blackwell` skips all of its cases on a non-Blackwell card, so the delegating file must carry its own gate and call the runners **as functions** — widening the shared gate instead puts three other architectures at risk, and its parametrized head dims are the SM100 line's. (2) `test_sdpa_compiled_cache_gpu.py` keeps `_ARCH` inside its `_CHILD` subprocess source string, where `monkeypatch` cannot reach; parameterize the string. Detector for the pin itself: delete the arch's row from `ENGINE_SPECS` and every strict `select_engine` pin must fail with `no plan for engine ...` — if it doesn't, the pin wasn't strict.
- **A compiled-helper test does not cover AOT backward.** `torch.compiler.is_compiling()` can be false while AOT traces a custom op's backward with FakeTensor/FunctionalTensor inputs. Keep raw-pointer helpers behind a registered custom op with a fake implementation even in that context. Run the enclosing op's `torch.library.opcheck`, including dynamic AOT dispatch; `sdpa/torch/test_torch_ops.py::TestOpContract::test_opcheck` detects this for packed Stats preparation.
- **Pointer-ABI stride fakes must preserve Int64, including page tables.** Annotating a host stride as Int64 is insufficient if its compile-time fake uses a plain Python `0`, which can infer Int32. A singleton axis can legally have a stride above `2**31` without requiring a large allocation; use that layout to catch narrowing at binding time. `test_graph_decode_prepared_keeps_int64_page_table_batch_stride` is the decode detector.
- **A native binder must keep observed storage separate from effective geometry.** Graph declarations and overrides can enlarge logical shapes without enlarging the caller's allocation. Derive ragged capacity from the producer's observed byte span in the effective element width; for fixed-size length/Stats reads validate known observed spans as well as logical numel. Keep the bare-pointer unknown-span contract explicit. `test_sdpa_native_thd_binding.py` checks these rules against the Python binder, including misaligned int32 lengths and overrides that claim more storage than the producer owns.
- **Paged overrides retain the producer's storage bound.** Validate each pool's effective TMA byte strides and observed span, and page-table element alignment, before launch. A larger override does not enlarge the allocation. Use host-only malformed-fact probes instead of launching an invalid tensor; `test_sdpa_paged_binding.py` covers short pools, misaligned strides/tables, and valid wide-stride or unknown-span bindings.
- **Wide host strides must stay wide through device setup arguments.** An Int64 pointer host can still truncate a stride while launching a descriptor-setup kernel. Check casts at the launch site and the setup parameter, not just the host signature. Exercise two live sequences with a physical row stride above `2**32` and poisoned output; a singleton-axis or binder-only probe never steps the truncated address. `test_thd_output_row_stride_above_int32_reaches_device_descriptors` checks numerical output and replay. Cover every served arch/flavor: a pre-Rubin-only marker hid narrowing in all four SM107 half hosts. `test_mhas_v2.py::test_sdpa_thd_output_stride_int64` is collected by both CI arch selections and checks native and Python binding independently, including D192/D128.
- **Unchanged device-function ASTs do not imply unchanged generated code.** Replacing static layout constants with runtime strides can change device address calculations; compare GPU time for the affected cases. A unit-stride fast path must also exercise nonunit strides through the same compiled host; `test_d256_paged_host_rebinds_table_column_stride` checks this contract.
- **When you remove a fallback, invert its counter assertion — do not delete it.** Tests that asserted `calls["bwd_cpp"]` incremented had to become "`calls["bwd"]` increments **and** `bwd_cpp` does not", so a silent regression to the old path fails the suite instead of passing it.

### Confirm you are testing the code you edited

Bind tensor views in the graph's declared axis order. A BSHD allocation and
a BHSD declaration can have identical shapes when H == S; shape equality then
preserves the producer's strides and cannot infer the intended transpose.
Use a metadata-only transpose at the test binding boundary, keeping the
reference's allocation unchanged. `test_sdpa_fp8_paged_equal_head_and_query_axes`
covers this collision through the actual FP8 harness and prepared executor.

`pip install -e .` does **not** put the package on `sys.path`. It installs a
`sys.meta_path` finder (`__editable___nvidia_cudnn_frontend_*_finder.py`) whose
`MAPPING` hard-codes an absolute path to the checkout it was installed from.
Meta-path finders run *before* `sys.path`, so **`PYTHONPATH` cannot shadow it** —
if you edit a different clone or a git worktree, your changes are silently not
under test. Symptoms are indistinguishable from a real result: a probe that
should change the output leaves it bit-identical, and edits appear to do nothing.

Check first, every time you work outside the installed checkout:

```bash
python -c "import cudnn; print(cudnn.__file__)"   # must be YOUR tree
```

`conftest.py` prints the same path in its banner (`cuDNN Frontend Path:`) — read
it rather than assuming. To point the editable install at another tree for one
run, patch the finder's `MAPPING` from a `sitecustomize.py` on `PYTHONPATH`
(`site` imports it after processing `.pth` files, so the finder already exists):

```python
# sitecustomize.py -- the finder module name embeds the installed version, so
# discover it rather than hard-coding it (it changes when __version__ bumps).
import importlib, pkgutil

name = next(
    m.name
    for m in pkgutil.iter_modules()
    if m.name.startswith("__editable___nvidia_cudnn_frontend_") and m.name.endswith("_finder")
)
importlib.import_module(name).MAPPING["cudnn"] = "/path/to/your/worktree/python/cudnn"
```

The same trap hides *inside* a run: `python/cudnn/frost/template_loader.py`
loads kernel templates by absolute path via `spec_from_file_location`, so the
template that serves a config may come from elsewhere too. To find out which
template a test actually compiles, log `path` at the top of `load_template` —
do not infer it from `_pick_flavor` by reading the source.

Nested conftests can also change package lookup after the initial banner.
`linear_attention/conftest.py` inserts its own checkout's `python/cudnn` into
`cudnn.__path__`. Running a candidate LA test by absolute path with a baseline
package can therefore load candidate implementations and produce a false
GREEN baseline. For RED/GREEN checks, copy only the new test changes into an
isolated baseline worktree and collect there; verify the concrete operation
module's `__file__` after conftest setup, not only `cudnn.__file__`.

Import-regression subprocesses are separate interpreters: in-process
`sys.path` or editable-finder changes do not automatically propagate. Carry
the selected package/source setup into each child and verify its loaded path.
Children that import test helpers need both the helper directory and
`test/python` on their own `sys.path`; pytest's parent-process path setup is
not inherited. Run fresh-process cache probes from their temporary directory
so launching pytest from `test/python` cannot hide missing child imports.

### Pending-consumer lifetime probes

A CUDA stream wait on an event that has never been recorded is a no-op; recording
it later does not retroactively block the consumer. Do not use an unrecorded event
as a host-released latch. A bounded delayed-consumer probe must assert that its
completion event is still pending after the producer/churn work. If the delay
expires, fail the setup instead of accepting a lifetime result without overlap.
Use a bounded byte consumer for recycled-storage negative controls, never a real
kernel on a deliberately invalidated workspace.

### Caller workspace alias regressions

Check the carved scratch byte range against operand byte spans, including packed
uint8 FP4 storage, strided views, optional outputs, and device pointer tables.
Intercept the compiled consumer for negative tests so intentional aliasing never
reaches a kernel; prove RED before the fix. Also exercise disjoint slices of one
allocation so rejecting shared ownership does not substitute for checking overlap.
Device pointer-table contents remain a caller contract, not a reason for a D2H read.

### CUDA Graph test lifetimes

Explicitly reset a test-owned CUDA Graph after replay verification, using a
`finally` block or context manager. Python reference cycles can defer its CUDA
executable destruction until GC runs inside a later test's capture, invalidating
that capture. The SM120 FP8 prepared suite reproduced this when the D128
strided-input graph was collected during the D384x320 case; tracing
`CUDAGraph.__del__` identified both tests. Keep capture error mode unchanged and
fix ownership instead of disabling GC or treating a retry as validation.

### Prepared quantized launch probes

When testing a retained staging path, choose a declaration that the graph
validator accepts but the native pointer layout declines. A nonunit D stride
is rejected before SDPA lowering; unit D with an unaligned outer pitch reaches
the existing dense conversion path. `test_staged_rebind_stream_capture` in
`test_sdpa_bwd_staged_sm100.py` guards its row padding and forbids legacy tensor
compilation/DLPack while checking changed allocations and replay.

Optional gradients copied from accumulators after a prepared launch still need
presence, dtype and extent checks before any staging write. They can be absent
from the pointer ABI, so the common binder cannot validate them. The detector
is `test_staged_auxiliary_outputs_validate_before_writes` (SM80 backward):
forbid copy/zero/launch and pass malformed or uncompiled dBias/dSink outputs.

Test both graph prepared-plan admission and the standalone adapter's compiler
selection when retaining a tensor fallback. Declining the graph attachment alone
can still compile a prepared artifact inside the adapter and fail at execution.
`test_fp8_paged_prepared_table_stride_admission` checks both decisions for distinct
K/V page-table strides; `test_fp8_paged_distinct_table_strides_prepared`
checks both routes numerically with the tensor compiler forbidden, changed
allocations and CUDA Graph replay.

For descriptor stride products, inspect the traced multiplication intermediates,
not just the final cast or Python annotation. MXFP8 V scales use a separate
plane stride: `test_mxfp8_v_scale_plane_stride_multiplies_in_int64` checks the
real host expression before descriptor encoding and must fail before widening.

The SM107 CI lane selects `test_sdpa_fp8_sm107.py` explicitly. Keep its prepared
FP8 cases in `TestPreparedSm107Fp8` there, or update the lane selector together
with a move; a new sibling file alone is not exercised by that lane.
An imported test function does not inherit its source module's `pytestmark`.
When re-exporting L0 checks into another architecture file, mark the wrapper
class L0 explicitly and verify collection with the lane's marker expression.
`test_sdpa_fwd_mxfp8_sm100.py::TestStagedMxfp8` covers this boundary.

Rebind scale buffers with different values, not only cloned storage: identical
values let a stale pointer pass. Poison and rebind amax too, then change scales
in place after CUDA Graph capture and check outputs after replay.
`test_prepared_fp8_rebind_scales_and_buffers` and
`test_prepared_fp8_capture_replay_reads_current_scales` cover both lifecycles.

### Prepared SM120 dense launches

SM120 masks the rightmost KV tile for every mask mode, unlike SM100's
padding-dependent tail rule. A shared binder must preserve that distinction.
`test_sm120_prepared_bounded_geometry_override` shrinks an unpadded plan to
S_kv=113, reuses the artifact, and checks O and Stats. Keep head counts and
dimensions as compile-time constants: the same binder fixes those per plan.

### Sparse score-recompute resource boundaries

Exercise non-power-of-two TMEM slot counts and partial ring wraps across
persistent query tiles; a full-ring single-tile case misses slot aliasing and
per-slot barrier-phase drift. Keep the focused regressions in
`deepseek_sparse_attention/cutedsl/test_DSA_sparse_score_recompute.py`. For SM100 SMEM planning, test
a boundary that distinguishes the usable 227 KiB from the nominal 228 KiB;
`test_DSA_sparse_attention_score_recompute_uses_launchable_smem_budget` must
fail against the old planner before accepting the fix.

Sparse metadata and cross-warp reduction scratch also need explicit reader
completion before reuse. Run racecheck on both indexer and attention cases:
ordering Q/K MMA alone does not publish every metadata lane's stores, and a
max-to-sum reduction can overwrite shared scratch before all warps read it.

SM107 metadata/heuristic tests run on non-SM107 CI lanes too. Model compiler
target availability explicitly when testing a synthetic Rubin capability row;
do not require the worker's DSL wheel to expose `sm_107a`. Keep real installed
compiler-target decline tests separate, and retain the live target check for
actual SM107 kernel execution. The release-DSL SM80 lane exposed this split.

A capability row's `sm_lo` names its lower bound, not the actual device.
Prepared admission must respect the adapter's exact device support even when
the row spans later compute capabilities. Include future-cc rejection controls
alongside the supported device in `test_prepared_fp8_override_capability_envelope`.

### Concurrent prepared frames versus SDK initialization

CuTe DSL 4.7.0/4.7.1 can leak the runtime's process-global initialization lock
when two threads first call the same cold artifact; a later test then hangs in
`cuda_dialect_init_library_once` before its kernel launches (see the independent
[runtime reproduction](https://github.com/NVIDIA/cudnn-frontend/pull/1236#issuecomment-5854182558)).
For a test of concurrent per-call bindings, initialize the artifact serially,
retain that warm call's owners, then use distinct buffers, poisoned outputs,
fresh workspace and independent streams for the concurrent calls.
`test_native_thd_concurrent_streams_use_independent_frames` follows this recipe.
This tests frame independence; it does not establish that cold concurrent SDK
initialization is fixed. Keep that runtime reproduction and its result separate.

### MXFP8 prepared scale-factor bindings

Dense V scale factors for D > 128 are plane-major; THD factors are packed
per-sequence tiles per head. Reuse `_quantize_seq` when constructing THD test
inputs; reshaping a dense buffer does not produce the THD contract. Rebind
E8M0 exponent values as well as pointers and validate after graph replay.
The MXFP8 split combine has no per-tensor output scalar: specialize its
scale pointer to None and remove Amax unscaling, rather than handing a NULL
runtime address to the per-tensor unscale kernel. `test_prepared_mxfp8_capture_reads_current_scales`
covers every native flavor with requested Amax. Physical SF stride units are
16 bytes; widen tile-count products before multiplication. The L1
`test_prepared_mxfp8_sf_head_stride_above_int32_units` steps a physical 64-GiB
head stride and checks numerical output, with resource-only OOM skips.

Linear SF repacks need the same width audit: an Int32 block index can wrap
before the byte address is formed even without a wide declared stride.
`test_sf_repack_steps_past_int32_with_allocated_guards` crosses the signed
Int32 boundary for both backward SF layouts. Its source and destination
prefixes keep the old negative offsets inside allocated storage, so the old
implementation fails numerically rather than through an invalid access.

### Packed SM80 forward launch migration

After moving a wrapper to a prepared host, forbid the old tensor launcher
after warmup and check fresh bindings plus changed-input graph replay. Keep
physical stride and product-overflow probes on every native flavor, with
wrapped addresses poisoned inside allocated guard storage.
`test_sdpa_sm80_thd_forward_prepared.py` exercises both properties. Removing
THD tensor fakes must also preserve the dense off-flavor/RoPE fallback.


### SM80 staged pointer launches

A pointer host can follow existing layout staging without changing that
staging's numerical contract. Keep off-flavor and RoPE references, mutate
angle tables after capture, and forbid `cutlass.cute.runtime.from_dlpack`
after warmup so a return to tensor launch plumbing fails independently of
numerical output. `test_dense_staged_pointer_launch_and_rope_replay` was
RED on the old tensor launcher for both dtypes and both template families.

Trailing-dimension and contiguity checks do not prove that a broadcast input
has a live batch slice: `(0, H, SQ, SKV)` passes both for an empty bias. Reject
it before workspace writes or pointer binding; the `bias_tensor-empty_batch`
case in `test_dense_staged_rejects_invalid_operands_before_staging` is the
RED-then-green detector. A compiler-memo test must also rebuild the wrapper
on its second call: clear the wrapper cache while retaining the compiler memo,
then forbid JIT. Reusing the same adapter cannot test compiler reuse.

SM80's standalone packed wrapper preserves cumulative tensors as row offsets
into the supplied storage. Do not substitute graph API cumulative-length
normalization: that changes which Q/K/V rows the standalone call addresses.
`test_thd_wrapper_preserves_packed_row_origins` checks different Q/KV origins,
empty sequences and changed origins after capture, including zeroed output
capacity outside those origins. Keep baseline compatibility evidence.


Prepared THD artifact reuse must survive a disabled persistent cache and an
unknown environment manifest. Change packed capacities and launch bounds with
JIT forbidden, then check numerical output and graph replay. Keep workspace
sizing per plan and exclude tensors, pointers and streams from the artifact
memo. `test_wrapper_capacity_reuses_artifact_without_disk_cache` is the native
SM80 detector. When asserting disk-artifact hits, clear the process memo before
both cache population and reload: otherwise an earlier test can prevent the
temporary cache from being populated, or a memo hit can bypass the disk counter.

A `None` compile sample may still occupy a positional TVM-FFI argument slot.
When extending a prepared host with a staged-only operand, retain the native
entry signature and delegate internally; do not assume the absent operand is
removed from the exported call ABI. Exercise both native and staged routes,
including a fresh-process artifact reload. For staged THD output padding,
test the bounded cast/fold with physical wide output strides as well as the
input staging; `test_staged_packed_physical_stride` covers both stride and
index-product overflow with allocated guard storage.

Prepared host migrations must preserve persistent compiled artifacts as well as
warm execution. A dataclass passed as a `Constexpr` compile argument can prevent
artifact export even though its runtime slots disappear. Carry only the immutable
primitive configuration facts the host needs, or read the template module's
configuration internally. `test_block_output_artifact_reloads_in_fresh_process`
builds in one process, forbids JIT in another, and checks O/SF/Stats/Amax after
execution and poisoned-output CUDA Graph replay. A same-process cache hit alone
does not establish cross-process reuse.

Compute `template_key(globals(), locals(), ...)` before local imports or other
local assignments. Including a helper function in `locals()` makes the key
uncacheable even when the graph's specialization is fully static; all eight
SM100/SM107 MXFP8 prepared entries must obey this ordering.

### Prepared gate pointer regressions

Auxiliary TMA inputs need the same physical Int64 stride coverage as Q/O.
`test_quantized_gate_batch_stride_above_int32` steps two live BF16 gate batches
across an 8-GiB gap. A valid decoy island at the truncated offset makes its
Int32 negative control fail numerically without launching an invalid access.
After graph capture, swap the gates between sigmoid saturation at zero and one:
O must change, while Stats and requested Amax still describe ungated SDPA.

Optional-output flags and sample descriptors must agree at every host boundary.
Cover an explicit disabled flag with a retained sample descriptor as well as
an absent descriptor: compilation, argument validation, scratch initialization
and tensor-operand elision must use the same effective presence decision.
`test_pv_bf16_no_amax_flag_with_sample_descriptor` checks prepared and retained
tensor entries; inconsistent decisions caused a D192 `None.iterator` compile
failure and an output that was simultaneously required and forbidden.

For fixed-geometry staged pointer hosts, validate Q/K/V/O shape, dtype and device,
and all auxiliary bindings before workspace carving or any copy/fill. A tensor
compiler previously checked some of these at dispatch; raw addresses cannot.
The SM80 detector `test_dense_staged_rejects_invalid_operands_before_staging`
replaces the workspace carver with a tripwire so an invalid short buffer fails
safely before it can reach a GPU launch.

For a GEMM+GLU failure, compare the final output with both the stored GEMM
intermediate and an independent dot product before attributing it to GEMM.
SwiGLU pairs alternate 32-column input/gate blocks; the two operands are not
halves of the N dimension. `test_swiglu_failure_diagnostics.py` checks that mapping,
bounded failure output, and preservation of the original assertion.

Count TMA store stages in committed groups, not individual output subtiles.
If one group reads two AB12 slots and one C slot, four AB12 slots and two C
slots permit only two outstanding groups. Rotate each output ring by its
own consumed-slot count across persistent tiles. A replay that repeatedly
overwrites one output can hide an earlier corrupted store; retain distinct
outputs and check every launch. `test_gemm_swiglu_retained_outputs_replay`
covers multiple groups within a tile and persistent tile transitions, including
one-group tiles that must advance the C ring at every tile boundary.

Independent page tables need independent observed-span checks and Int64 stride
slots in the prepared host. Test distinct K/V page values and layouts, then
rebind allocations and mutate table values under capture replay. Preserve the
shared-stride constraint for a host whose ABI still has only one stride pair.

A shared kernel imported as an ordinary module has no template-loader digest.
Calling `template_key` there otherwise returns `None` and silently bypasses
persistent caching. Give the module a source identity and require fresh-process
reload of the whole chain, including split combine and both pointer/tensor
calling conventions; forbidding JIT only around the attention kernel misses it.

For staged pointer launches, validate aliases against the original caller operands
before replacing them with workspace views. The core binder only sees gathered
buffers and otherwise misses an Amax scalar or block-scale SF output aliasing
the original Q or O. `test_staged_amax_alias_checks_original_operand` and
`test_staged_sf_alias_checks_original_operand` intercept copies to detect these
aliases before any kernel is launched. Reuse the SF binder on original facts
so its full atom span and output/input alias rules remain consistent.

For staged copies on multiple GPUs, resolve an omitted launch stream on Q's
device and keep that device context active through gather, launch and scatter.
A nondefault stream on Q's device must remain authoritative even when another
CUDA device is current; restore the caller's device after execution.
In multi-GPU tests, check the operand device's architecture before allocating or
launching on it. A module-level marker only checks the initially current device;
it cannot admit a second target on a heterogeneous machine. Allow a different
architecture on the caller's current device when testing context restoration.

When retiring a fake-tensor builder, move negative guards to the live
`cute.runtime.make_fake_tensor` and `make_fake_compact_tensor` constructors.
Patching a deleted helper with `raising=False` proves nothing. Keep direct
SASS and split-partial tests on the production pointer entry, including
partial-output inspection before combine.

When retiring a compiler entry, audit its standalone `_main()` as well as
adapter and test callers. A leftover unqualified `compile(...)` silently
resolves to Python's builtin after the definition is deleted. Execute the
actual CLI with its replacement compiler intercepted and assert that the
prepared entry is called; import-only checks cannot catch this failure.

### Wrapper coverage after workspace migrations

When a prepared adapter starts requiring caller workspace for an existing
layout, test every public convenience wrapper that constructs it. Adapter
checks with explicit workspace cannot detect a wrapper that still omits it.
Exercise BHSD-contiguous and padded conversion inputs plus the compact control,
including plan-cache reuse and a non-default current stream. The allocating
wrapper must obtain `scratch_workspace_bytes()` and pass per-call scratch on
the input device and actual launch stream; a zero-workspace plan should keep its
allocation-free path. Test an explicit stream different from the ambient stream:
a non-default current stream alone cannot expose premature scratch reuse.
`test_wrapper_scratch_survives_explicit_stream_consumer` warms the real SM80
wrapper, intercepts execution with a bounded byte write, and checks live ambient-
stream allocations while that consumer is pending. Prewarm the churn allocator
pool too: a fresh `cudaMalloc` can synchronize away the intended overlap.
The SM120 detector is `TestStagedSm120Wrapper` in
`test/python/sdpa/frost/test_sdpa_fwd_dsl_sm120.py`.


A fixed-layout wrapper cache must include every input stride used by the prepared
plan. Same-shape calls can alternate compact, padded and permuted storage; a
shape-only cache reuses an incompatible native plan.
`test_wrapper_cache_distinguishes_current_input_strides` exercises three SM80
layouts and returns to the first one to verify both separation and reuse.
When padding Q/K to a vector width, retain the original attention scale and
check non-multiple-of-eight widths against an independent reference.

When a test is re-exported from another module, the source module's `pytestmark`
does not follow it. A subprocess-based GPU test must check architecture in the
parent before spawning; a `pytest.skip` in a plain Python child exits nonzero.
Keep genuine child failures failing on supported devices. The half SDPA artifact
reload detector is re-exported through `TestStagedHalf` and covers this boundary.

Gate layout admission must share the adapter's TMA predicate, including batch
stride alignment for B > 1. A valid head/sequence pitch cannot compensate for
an unaligned batch pitch. `test_gate_batch_stride_alignment` covers FP8, half
and float element widths and the non-stepped singleton-batch control.

Fusing standalone RoPE table preprocessing must preserve the FP32 conversion
before sin/cos and full-range trigonometry. Approximate PTX sin/cos is not a
replacement for Torch's large-angle range reduction. The exact-output detector
`test_rope_table_large_angles_and_special_values` includes large/small finite
angles, signed zero and nonfinite inputs. Keep real Int64 stride and product
overflow checks on the angle input, with wrapped addresses inside allocated
guard storage, plus changed-angle replay and fresh-process artifact reload.


A CuTeDSL `cutlass.Array` scalar index is a flat element offset; even a
one-element tuple takes that path. For non-contiguous rank-one prefixes,
form the element offset explicitly in Int64 before indexing. A contiguous-only
metadata test misses this. `sdpa/torch/test_varlen_metadata.py` checks strided
prefixes, changed prefixes under replay, physical prefix strides above `2**32`,
and smaller strides whose index product overflows. Its provider tests pin real
backend and FROST graph plans; SM80 standalone THD support is not evidence of
SM80 THD graph eligibility.

Packed wrapper initialization belongs in the existing compiled host: a separate
Python launch can erase the host savings from fusing device clears. Preserve the
wrapper's zeroed capacity holes/tails separately from direct graph outputs and
MHA dK/dV, whose unwritten storage must remain untouched. Check runtime word counts
above `2**32`, including half-tensor extent-to-byte conversion, with physical guard
storage and poisoned tail probes. `test_sdpa_sm80_packed_init.py` covers these cases.
A `stream_context(None, device)` is deliberately a no-op; it does not select the
operand device. Packed wrappers must guard that device for both compilation and
pointer launch and restore the caller. Test a foreign current device on both
operand GPUs, with default and explicit streams; validate outputs after the call.

Same-dtype `to(dtype)` preserves a sliced input's strides, and `reshape` can
preserve them too. When a wrapper declares compact lengths or sinks to the
graph, explicitly normalize both layout and pointer alignment before binding.
`sdpa/torch/test_aux_metadata.py` pins backend and FROST plans with strided
native metadata and checks changed-input replay; the old backend silently read
gap values while FROST rejected the inconsistent declaration. Performance
comparisons must use a numerically valid baseline, such as dtype-converting
inputs or an explicit compact-copy control, rather than time the wrong result.

Metadata fusion also needs a single-input timing control. A compact half sink
alone already requires one Torch cast; general prepared dispatch can improve
GPU time while increasing CPU enqueue cost. Pin backend and FROST separately
and measure that case alongside mixed conversions and native views.
`test_forward_metadata_compact_sink_needs_no_compiler` guards the cheap cast,
including unaligned source offsets and changed-input replay.

A saved Stats tensor can have `requires_grad=True` inside an ordinary provider
backward where grad mode is disabled. Route assertions must exercise that caller;
only an active differentiable helper call needs the Torch autograd fallback.
`test_varlen_backward_uses_prepared_stats` guards the real provider route.
Selecting an operand CUDA context does not necessarily change CuTe DSL's default
compiler target on heterogeneous hosts. Pass the operand architecture explicitly
to the compiler and include it in the artifact key; the metadata target detector
checks this alongside execution. For packed-to-padded conversion, include highly
uneven lengths and heavy padding in correctness and performance comparisons;
measure multiple calls per captured graph so host replay submission cannot hide
a device regression.

An explicit compiler target still does not guarantee a runnable execution entry.
Some DSL versions compare it with device zero's runtime architecture and leave
the entry absent on heterogeneous hosts. Optional shared producers must preserve
their Torch fallback in that case. Exercise the actual compiler-to-entry boundary
with a missing entry, repeat with changed inputs and graph replay, and verify that
ordinary compiler/launch errors still propagate rather than broadly catching them.

Preserving packed metadata storage requires both a native element type and a
wide address product. Test Int32/Int64 prefixes and FP16/BF16/FP32 sinks with
changed strides, broadcast views and fresh buffers; widen before multiplying
the sequence/head index by the element stride. A metadata value may retain
an Int32 sequence-length contract while its storage address requires Int64.
`test_sdpa_sm80_packed_metadata.py` includes physically wide strides/products,
changed-value replay and zero-copy token-major backward Stats.


### Prepared SM90 migration

A shared pointer binder must preserve the joining kernel's layout and scheduler
contract. SM90 D512 serves covering dense permutations beyond the SM100 token-major
predicate, but embeds its batch and dense extents in scheduler coordinates. Share
the exact layout predicate between graph admission and runtime binding; reject a
shrinking THD batch before a fixed-batch kernel reads its lengths. Keep the 128-byte
alignment of the embedded Hopper tensor maps.
`test_sdpa_prepared_sm90.py` checks those guards, forbids the old tensor/JIT path,
rebinds fresh buffers, captures and replays changed inputs, reloads exported
artifacts in a fresh interpreter, and writes physical rows beyond an Int32 stride.
Singleton stride spies must inspect the prepared pointer frame, not a tensor
launcher that the graph no longer calls.

Standalone THD Stats declarations can use BHS rank three while the shared
packed binder also recognizes rank-three TH1. Preserve the declared axis order
in buffer facts before binding; changing a tensor view during execute violates
the prepared contract. `test_standalone_thd_token_major_stats` covers declared
BHS and packed TH1/TH/flat storage with tensor-conversion methods forbidden.
Include S=1: BHS and TH1 can have identical shapes, so disambiguation must
also inspect their head/token strides.
