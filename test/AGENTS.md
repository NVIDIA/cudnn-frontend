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

Requirements: `pip install -e ".[cutedsl]"` plus `pytest pytest-xdist looseversion`. `fe_api/` additionally requires an SM90/SM100-class GPU; tests skip (or should skip) on unsupported arch/dtype/backend-version combos rather than fail.

### conftest.py landmines — read before editing

- `PYTORCH_CUDA_ALLOC_CONF` is set at the very top, **before any torch import** (torch reads it once at CUDA-allocator init). Don't move it, and don't import torch in a plugin that loads earlier.
- `import transformer_engine` happens (in try/except) **before** `import cudnn` — TE and cuDNN conflict if loaded in the other order. Preserve this ordering.
- `torch.cuda.synchronize` is monkeypatched to a guard that prints a filtered traceback and hard-exits (`os._exit`) on async CUDA errors; the original is kept as `torch.cuda.synchronize_unsafe`.
- A session-scoped autouse `cudnn_handle` fixture creates one handle bound to a dedicated torch stream; use it instead of creating handles per-test.
- `pytest_configure` asserts `torch.cuda.is_available()` — there is no CPU-only mode.
- Many custom CLI options exist (`--dryrun`, `--repro`, `--seed`, `--perf`, per-op dimension overrides like `--b/--s_q`, `--nsa-*`, `--dsa-*`); check `pytest_addoption` before adding new ones.

### Layout

- `test/python/test_*.py` — core graph-API tests (conv, matmul, norms, SDPA `test_mhas*.py`, rope, kernel cache, OSS engines `test_sm{90,100}_prefill_oss_engine.py`, ...). Shared SDPA references in `test/python/sdpa/`.
- `test/python/fe_api/<family>/` — one subdir per OSS kernel family (`gemm/`, `grouped_gemm/`, `bsa/`, `dsa/`, `nsa/`, `norm/`, `sdpa/`), each with `test_<op>.py` + utils/reference modules.

### Conventions for new tests

- Mark with a level (`@pytest.mark.L0` ... `L4`): L0 must stay fast (default CI smoke); big parameter sweeps go to higher levels.
- Gate on capability, don't assume it: skip via `check_support()` failures, `cudnn.backend_version()`, and `torch.cuda.get_device_capability()`.
- Compare against a reference implementation (see existing `*_ref.py` / `*_reference.py` patterns) with dtype-appropriate tolerances.
