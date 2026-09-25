# test — Agent Guide

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
pytest test_conv_fprop.py    # one file — note the default -m L0 filter still applies
pytest fe_api/gemm/          # OSS kernel tests
```

**Read pytest's own summary line; do not post-process the output.** `... | grep -c "passed"` counts a collection error as a pass, and `cmd | tail` reports *tail's* exit status, so a failed build or a failed suite behind a pipe looks like success. Both have produced confidently wrong "all green" reports here. Requirements: `pip install -e .` plus `pytest pytest-xdist looseversion`. `fe_api/` additionally requires an SM90/SM100-class GPU; tests skip (or should skip) on unsupported arch/dtype/backend-version combos rather than fail.

### conftest.py landmines — read before editing

- `PYTORCH_CUDA_ALLOC_CONF` is set at the very top, **before any torch import** (torch reads it once at CUDA-allocator init). Don't move it, and don't import torch in a plugin that loads earlier.
- `import transformer_engine` happens (in try/except) **before** `import cudnn` — TE and cuDNN conflict if loaded in the other order. Preserve this ordering.
- Crash isolation (`# Crash isolation` block in `conftest.py`): `pytest_cmdline_main` injects `-n1 --max-worker-restart=100000`, so a segfault, a poisoned CUDA context, or a hang kills only one xdist worker, which the controller replaces before continuing. After every test `pytest_runtest_logfinish` probes the context with `torch.cuda.synchronize()`; a per-test `faulthandler.dump_traceback_later(exit=True)` deadline (`CUDNN_TEST_TIMEOUT`, default 1500 s, `0` disables) covers the probe too. It is faulthandler's C watchdog, not a Python thread or `SIGALRM`, because a hung CUDA driver call holds the GIL and parks the main thread. Not injected under `-n<N>`, `-s`, `--pdb`, `--collect-only`, or `CUDNN_TEST_NO_ISOLATION=1`; without a worker to restart, a dead context stops the run via `pytest.exit` and a hang still hard-exits. Killing a worker does **not** stop a kernel it left running -- the driver keeps that context until the kernel ends, and the next worker can block behind it.
- A session-scoped autouse `cudnn_handle` fixture creates one handle bound to a dedicated torch stream; use it instead of creating handles per-test.
- `pytest_configure` asserts `torch.cuda.is_available()` — there is no CPU-only mode.
- Many custom CLI options exist (`--dryrun`, `--repro`, `--seed`, `--perf`, per-op dimension overrides like `--b/--s_q`, `--nsa-*`, `--dsa-*`); check `pytest_addoption` before adding new ones.

### Layout

- `test/python/test_*.py` — core graph-API tests (conv, matmul, norms, SDPA `test_mhas*.py`, rope, kernel cache, OSS engine `test_sm100_rms_norm_silu_graph_api.py`, ...). Shared SDPA references in `test/python/sdpa/`.
- **`test/python/sdpa/` is a mixed directory and the `test_` prefix is load-bearing.** `fp16.py`, `helpers.py`, `random_config.py` are harness modules the tests import; `sdpa/test_*.py` (and `sdpa/frost/test_*.py`) are collected tests. `pytest.ini` sets no `python_files` override, so a test file dropped there **without** the prefix is silently treated as a helper — it is never collected, and the suite stays green while asserting nothing. After moving or adding a test, confirm it is picked up by the *default* sweep, not just when named directly:

  ```bash
  pytest --collect-only -q | grep -c sdpa/test_torch_ops.py
  ```
- `test/python/fe_api/<family>/` — one subdir per OSS kernel family (`gemm/`, `grouped_gemm/`, `bsa/`, `dsa/`, `nsa/`, `norm/`, `sdpa/`), each with `test_<op>.py` + utils/reference modules.

### Conventions for new tests

- Mark with a level (`@pytest.mark.L0` ... `L4`): L0 must stay fast (default CI smoke); big parameter sweeps go to higher levels.
- **Default L0 coverage is not sufficient if the CI target excludes the provider.** Check the actual CI path and `-k` filters. The general Python target excludes FROST cases, so representative shared-API FROST tests also need collection under `sdpa/frost/`; `test_sdpa_ordered_bindings.py` reuses the shared ordered-binding smoke logic. Verify both target collection and execution on a supported GPU.
- **Check for a module-level `pytestmark` before adding per-test markers.** Many files apply a level or capability marker file-wide (`pytestmark = ...` near the top); duplicating it on each test is noise, and suggesting it in review wastes a round-trip (recurred on PRs #814, #811, #797).
- Gate on capability, don't assume it: skip via `check_support()` failures, `cudnn.backend_version()`, and `torch.cuda.get_device_capability()`.
- Compare against a reference implementation (see existing `*_ref.py` / `*_reference.py` patterns) with dtype-appropriate tolerances.
- **Scale the tolerance to the tensor, not to the dtype alone.** A fixed absolute bound quietly becomes wrong when magnitudes grow: GQA dK/dV sum over `h_q/h_kv` query heads, so at a group size of 4 the *relative* error stays ~0.5% while `|dv|` peaks near 9.6 and blows a bound that passed at `h_kv == h_q`. Compare against `TOL * max(|ref|.max(), 1.0)`, or the next GQA ratio someone adds will look like a correctness regression.
- **Performance rankings belong in offline benchmark validation.** Public contract tests mock ranking decisions and verify eligibility, marker handling, and explicit overrides; see `test_propose_preserves_recommendations_and_places_one_marker`.
  Kernel correctness tests explicitly select the intended engine and knobs instead of asserting that performance heuristics rank that plan first.
- **A regression test must be seen RED.** Before trusting one, run it against the unfixed code — restore the old line, confirm it fails, restore the fix. `test_dsl_sm100_thd_interleaved_kv_views` and `test_varlen_backward_does_not_sync` were both checked this way, and both were genuinely red beforehand; a test written for a bug and never seen to fail is asserting an unknown.
- **Low-precision quantization needs exact midpoint tests.** Approximate reciprocal multiplication can move an exact E2M1 tie across its rounding boundary even when the native conversion uses round-to-nearest-even. Include signed midpoint values with non-power-of-two block scales, and compare the quantization stage itself before diagnosing amplified attention-gradient differences.
- **Build the reference in fp64 when the bound is tighter than ~1e-3.** The DLFW CI containers run fp32 matmul in TF32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`; recent torch also defaults `fp32_precision` to `tf32` on Blackwell+), a ~3e-4 relative error per `Q @ K^T` logit. A 1024-column log-sum-exp averages it down to ~2e-5, a causal row with ONE valid column keeps it whole: `test_fp8_stats_is_the_exact_softmax_lse[causal]` read max|dLSE| 1.3e-4 against its 1e-4 bound on the sm107 lane (2026-09-15) with an exact kernel, and passed on a 208-SM node whose draws happened to be kinder. `sdpa/fp8_ref.compute_ref(dtype=torch.float64)` is the existing knob; in a hand-rolled reference `.double()` the operands before the matmul, not just the `logsumexp`.
- **An fp8 midpoint flip is PROVED from the reference's intermediates, never inferred from the output's shape.** One flipped P/dS code moves one output d-row by `(c_alt - c_ref) x descale x operand_row` (Q for dK, K for dQ, dO for dV, V for O) -- but a power-of-two multiple of an operand row is not evidence of one: a masked key, another batch's or another head's row, two identical rows each one flip away, or a step no adjacent pair of codes has (8192 in e4m3) all fit that description (review on PR #1075). `assert_close_fp8_grad(..., operand=, flip_unit=, intermediates=, fp8_dtype=)` lifts the `4 * atol` row cap only when, at a VALID position of the row's own reduction (same batch, a q head of the same GQA group, unmasked), the reference's scaled fp32 intermediate -- `compute_ref(..., return_intermediates=)` / `compute_ref_backward(...)`, re-run on the bad rows only -- sits within 1/32 of a code spacing of the midpoint between its fp8 code and the adjacent one, and that single flip reproduces the row three ways: within the ordinary tolerance, per element within the output's own rounding (`atol` plus half an output code spacing from each side -- both sides are dequantized output codes; pass `out_dtype=torch_otype`), and as a whole (the least-squares number of flips fitted to the row is 1 within 1/2 -- two flips fit at 2, and on a row of large gradients `rtol * |expected|` is itself two flips wide, so the ordinary tolerance alone would take three flips for one). It prints the position, the two codes, the step and the three fits. Packed (ragged) outputs keep the plain cap, as does `h_k != h_v`. `sdpa/test_fp8_flip_budget.py` pins the accepted case and each of those rejections on a problem the reference itself built. The negative-score q rows have amplitude 8 at d192, so a legitimate flip moves a dK row by 0.5 -- the fixed cap (0.32) alone would call that a defect (sm107 212-SM lane, test310, 2026-09-15).
- **Decode Stats must be tested independently of training.** The random SDPA harness uses `generate_stats=cfg.is_train`, so its `s_q == 1` inference sweep checks O without checking LSE. For ragged GQA decode, request Stats explicitly, initialize every head to NaN, and compare every head against the reference. Include padded-Stats, MHA, and `s_q > 1` controls; pin a backend plan when testing native codegen so FROST cannot mask it. `test_mhas_v2.py::test_sdpa_ragged_decode_stats` is the detector (NVBug 6783545); run with `--runxfail` when checking an affected older backend.
- **Seed before you allocate.** `torch.manual_seed()` after constructing the inputs seeds nothing that matters. Two runs meant to be compared then differ by data, and the assertion fails (or worse, passes) for a reason unrelated to what is under test — if two runs must be comparable, build the inputs once and reuse them.
- **Every randomized SDPA input uses the per-test generator.** A seeded Q/K/V tuple is not a reproducible case if its block mask comes from the process-global CUDA RNG. Pass `generator=rng_data_gen` to auxiliary draws too; `test_block_mask_uses_the_per_test_data_generator` perturbs global RNG while holding the case seed fixed. Before attributing an order-dependent failure to an earlier engine, compare the actual masks as well as Q/K/V.
- **Compiled DSL call arity excludes compile-time parameters.** A `cutlass.Constexpr` argument belongs to the compilation signature and disappears from the compiled runtime call. When checking positional launch sites against `_host`, exclude these annotations as well as the stream keyword; do not add a runtime argument to satisfy an unfiltered Python signature count. `test_every_combine_call_site_matches_the_compiled_arity` is the detector.
- **Pointer-ABI stride fakes must preserve Int64, including page tables.** Annotating a host stride as Int64 is insufficient if its compile-time fake uses a plain Python `0`, which can infer Int32. A singleton axis can legally have a stride above `2**31` without requiring a large allocation; use that layout to catch narrowing at binding time. `test_graph_decode_prepared_keeps_int64_page_table_batch_stride` is the decode detector.
- **Unchanged device-function ASTs do not imply unchanged generated code.** Replacing static layout constants with runtime strides can change device address calculations; compare GPU time for the affected cases. A unit-stride fast path must also exercise nonunit strides through the same compiled host; `test_d256_paged_host_rebinds_table_column_stride` checks this contract.
- **When you remove a fallback, invert its counter assertion — do not delete it.** Tests that asserted `calls["bwd_cpp"]` incremented had to become "`calls["bwd"]` increments **and** `bwd_cpp` does not", so a silent regression to the old path fails the suite instead of passing it.

### Confirm you are testing the code you edited

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
