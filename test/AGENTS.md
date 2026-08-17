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

**Read pytest's own summary line; do not post-process the output.** `... | grep -c "passed"` counts a collection error as a pass, and `cmd | tail` reports *tail's* exit status, so a failed build or a failed suite behind a pipe looks like success. Both have produced confidently wrong "all green" reports here. Requirements: `pip install -e ".[cutedsl]"` plus `pytest pytest-xdist looseversion`. `fe_api/` additionally requires an SM90/SM100-class GPU; tests skip (or should skip) on unsupported arch/dtype/backend-version combos rather than fail.

### conftest.py landmines — read before editing

- `PYTORCH_CUDA_ALLOC_CONF` is set at the very top, **before any torch import** (torch reads it once at CUDA-allocator init). Don't move it, and don't import torch in a plugin that loads earlier.
- `import transformer_engine` happens (in try/except) **before** `import cudnn` — TE and cuDNN conflict if loaded in the other order. Preserve this ordering.
- Crash isolation (`# Crash isolation` block in `conftest.py`): `pytest_cmdline_main` injects `-n1 --max-worker-restart=100000`, so a segfault, a poisoned CUDA context, or a hang kills only one xdist worker, which the controller replaces before continuing. After every test `pytest_runtest_logfinish` probes the context with `torch.cuda.synchronize()`; a per-test `faulthandler.dump_traceback_later(exit=True)` deadline (`CUDNN_TEST_TIMEOUT`, default 900 s, `0` disables) covers the probe too. It is faulthandler's C watchdog, not a Python thread or `SIGALRM`, because a hung CUDA driver call holds the GIL and parks the main thread. Not injected under `-n<N>`, `-s`, `--pdb`, `--collect-only`, or `CUDNN_TEST_NO_ISOLATION=1`; without a worker to restart, a dead context stops the run via `pytest.exit` and a hang still hard-exits. Killing a worker does **not** stop a kernel it left running -- the driver keeps that context until the kernel ends, and the next worker can block behind it.
- A session-scoped autouse `cudnn_handle` fixture creates one handle bound to a dedicated torch stream; use it instead of creating handles per-test.
- `pytest_configure` asserts `torch.cuda.is_available()` — there is no CPU-only mode.
- Many custom CLI options exist (`--dryrun`, `--repro`, `--seed`, `--perf`, per-op dimension overrides like `--b/--s_q`, `--nsa-*`, `--dsa-*`); check `pytest_addoption` before adding new ones.

### Layout

- `test/python/test_*.py` — core graph-API tests (conv, matmul, norms, SDPA `test_mhas*.py`, rope, kernel cache, OSS engines `test_sm{90,100}_prefill_oss_engine.py`, ...). Shared SDPA references in `test/python/sdpa/`.
- **`test/python/sdpa/` is a mixed directory and the `test_` prefix is load-bearing.** `fp16.py`, `helpers.py`, `random_config.py` are harness modules the tests import; `sdpa/test_*.py` (and `sdpa/frost/test_*.py`) are collected tests. `pytest.ini` sets no `python_files` override, so a test file dropped there **without** the prefix is silently treated as a helper — it is never collected, and the suite stays green while asserting nothing. After moving or adding a test, confirm it is picked up by the *default* sweep, not just when named directly:

  ```bash
  pytest --collect-only -q | grep -c sdpa/test_torch_ops.py
  ```
- `test/python/fe_api/<family>/` — one subdir per OSS kernel family (`gemm/`, `grouped_gemm/`, `bsa/`, `dsa/`, `nsa/`, `norm/`, `sdpa/`), each with `test_<op>.py` + utils/reference modules.

### Conventions for new tests

- Mark with a level (`@pytest.mark.L0` ... `L4`): L0 must stay fast (default CI smoke); big parameter sweeps go to higher levels.
- **Check for a module-level `pytestmark` before adding per-test markers.** Many files apply a level or capability marker file-wide (`pytestmark = ...` near the top); duplicating it on each test is noise, and suggesting it in review wastes a round-trip (recurred on PRs #814, #811, #797).
- Gate on capability, don't assume it: skip via `check_support()` failures, `cudnn.backend_version()`, and `torch.cuda.get_device_capability()`.
- Compare against a reference implementation (see existing `*_ref.py` / `*_reference.py` patterns) with dtype-appropriate tolerances.
- **Scale the tolerance to the tensor, not to the dtype alone.** A fixed absolute bound quietly becomes wrong when magnitudes grow: GQA dK/dV sum over `h_q/h_kv` query heads, so at a group size of 4 the *relative* error stays ~0.5% while `|dv|` peaks near 9.6 and blows a bound that passed at `h_kv == h_q`. Compare against `TOL * max(|ref|.max(), 1.0)`, or the next GQA ratio someone adds will look like a correctness regression.
- **A regression test must be seen RED.** Before trusting one, run it against the unfixed code — restore the old line, confirm it fails, restore the fix. `test_dsl_sm100_thd_interleaved_kv_views` and `test_varlen_backward_does_not_sync` were both checked this way, and both were genuinely red beforehand; a test written for a bug and never seen to fail is asserting an unknown.
- **Seed before you allocate.** `torch.manual_seed()` after constructing the inputs seeds nothing that matters. Two runs meant to be compared then differ by data, and the assertion fails (or worse, passes) for a reason unrelated to what is under test — if two runs must be comparable, build the inputs once and reuse them.
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
