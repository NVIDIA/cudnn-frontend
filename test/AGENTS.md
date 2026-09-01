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
- Gate on capability, don't assume it: skip via `check_support()` failures, `cudnn.backend_version()`, and `torch.cuda.get_device_capability()`.
- Compare against a reference implementation (see existing `*_ref.py` / `*_reference.py` patterns) with dtype-appropriate tolerances.
- **Scale the tolerance to the tensor, not to the dtype alone.** A fixed absolute bound quietly becomes wrong when magnitudes grow: GQA dK/dV sum over `h_q/h_kv` query heads, so at a group size of 4 the *relative* error stays ~0.5% while `|dv|` peaks near 9.6 and blows a bound that passed at `h_kv == h_q`. Compare against `TOL * max(|ref|.max(), 1.0)`, or the next GQA ratio someone adds will look like a correctness regression.
- **A regression test must be seen RED.** Before trusting one, run it against the unfixed code — restore the old line, confirm it fails, restore the fix. `test_dsl_sm100_thd_interleaved_kv_views` and `test_varlen_backward_does_not_sync` were both checked this way, and both were genuinely red beforehand; a test written for a bug and never seen to fail is asserting an unknown.
- **Seed before you allocate.** `torch.manual_seed()` after constructing the inputs seeds nothing that matters. Two runs meant to be compared then differ by data, and the assertion fails (or worse, passes) for a reason unrelated to what is under test — if two runs must be comparable, build the inputs once and reuse them.
- **When you remove a fallback, invert its counter assertion — do not delete it.** Tests that asserted `calls["bwd_cpp"]` incremented had to become "`calls["bwd"]` increments **and** `bwd_cpp` does not", so a silent regression to the old path fails the suite instead of passing it.
