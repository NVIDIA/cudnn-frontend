# cuDNN Frontend — Agent Guide

cuDNN Frontend (FE) is a **header-only C++ library** plus a **Python package** (`nvidia-cudnn-frontend`, imported as `cudnn`) that wraps the cuDNN Graph API, and a growing set of open-source CuTeDSL kernels (SDPA/Flash Attention, MoE grouped GEMM fusions, fused normalizations).

Directory-specific guides: [include/cudnn_frontend/AGENTS.md](include/cudnn_frontend/AGENTS.md) (C++ library), [python/cudnn/AGENTS.md](python/cudnn/AGENTS.md) (Python API + OSS kernels), [test/AGENTS.md](test/AGENTS.md) (running tests), [samples/AGENTS.md](samples/AGENTS.md).

## Repository map

| Path | Purpose |
|---|---|
| `include/` | The header-only C++ library (CMake INTERFACE target `cudnn_frontend`). C++17. |
| `python/` | pybind11 bindings (`python/*.cpp`, `python/pygraph/`) + pure-Python `python/cudnn/` package |
| `python/cudnn/<op>/` | Frontend-only OSS CuTeDSL kernels (GEMM fusions, grouped GEMM, BSA/DSA/NSA, SDPA) |
| `samples/` | C++ samples (Catch2 binaries `samples`, `legacy_samples`) and Python notebooks |
| `test/` | `test/cpp` (Catch2 binary `tests`) and `test/python` (pytest) |
| `benchmark/` | Standalone perf harnesses (SDPA training, norms, DSA, CuTeDSL fusions); each has a README |
| `tools/cudnn_repro/` | Standalone CLI that parses cuDNN logs into repro commands (own pyproject) |
| `docs/` | Markdown docs: `operations/` (graph-op reference), `fe-oss-apis/` (OSS kernel APIs), how-to guides |
| `cmake/cuDNN.cmake` | Locates the cuDNN backend library (or reuses existing `CUDNN::` targets) |
| `skills/` | Agent skills (see [Agent skills](#agent-skills)) |

## Environment requirements

- NVIDIA GPU required for essentially all tests and samples (SDPA/OSS kernels need Hopper SM90 or Blackwell SM100+).
- CUDA toolkit (`nvcc`), cuDNN **9.x** backend (headers + libs), CMake ≥ 3.23, a C++17 compiler.
- If cuDNN or CUDA are not in default system locations, set `CUDNN_PATH` and `CUDAToolkit_ROOT` (both honored by CMake and `setup.py`).
- Python ≥ 3.10. `cudnn.backend_version()` gates many features at runtime (integer, e.g. 9.12.0 → `91200`); tests skip on older backends.

## Build

C++ (builds samples + tests by default):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # add -DCUDNN_PATH=... -DCUDAToolkit_ROOT=... if not system-installed
cmake --build build -j $(nproc)
# artifacts: build/bin/{samples,legacy_samples,tests}
```

CMake options (defaults): `CUDNN_FRONTEND_BUILD_SAMPLES=ON`, `CUDNN_FRONTEND_BUILD_TESTS=ON`, `CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS=OFF`, `CUDNN_FRONTEND_SKIP_JSON_LIB=OFF`.

The C++ build uses `-Werror` (`/WX` on MSVC) with `-Wall -Wextra -Wpedantic` — new warnings break the build.

Python (editable; compiles the pybind11 extension via CMake):

```bash
pip install -e .              # graph API + the OSS CuTeDSL kernels (nvidia-cutlass-dsl, cuda-python, tvm-ffi; framework-neutral)
                              # ".[cutedsl]" still resolves -- it now holds only cuda-python, which the DSL pulls in anyway
pip install -e ".[cutile]"    # + the cuTile linear-attention engines (cuda-tile; needs a system tileiras)
pip install --group torch      # + torch for the CuTeDSL APIs (torch, torch-c-dlpack-ext)
pip install --group jax        # + jax for the CuTeDSL APIs (jax >= 0.5; XLA entry points via cutlass.jax)
```

`setup.py` honors env vars: `CUDNN_PATH`, `CUDA_PATH` / `CUDAToolkit_ROOT`, `DEBUG=1` (debug build), `CMAKE_BUILD_PARALLEL_LEVEL`, `CMAKE_GENERATOR`.

**Editable installs pin one checkout — a worktree or second clone is silently not under test.** `pip install -e .` installs a `sys.meta_path` finder whose `MAPPING` hard-codes the absolute path of the checkout it was installed from; meta-path finders run before `sys.path`, so `PYTHONPATH` cannot shadow it. Before trusting any measurement from a worktree or second clone, confirm the import actually resolves there: `python -c "import cudnn; print(cudnn.__file__)"`. If a deliberately destructive probe changes nothing, suspect the import path before the code.

## Test

```bash
# C++ (Catch2): list and run cases by name
./build/bin/tests --list-tests
./build/bin/tests "Validate conv node"

# Python: run from test/python so pytest.ini and conftest.py apply
cd test/python
pytest                        # default is -m L0 (smoke level) per pytest.ini
pytest -m L1                  # deeper levels: L0..L4
pytest test_conv_fprop.py     # one file (still filtered by -m L0 — pass -m "L0 or L1" to widen)
pytest fe_api/                # OSS kernel tests; require `--group torch` (and `--group jax` for the *_jax tests) + SM90/SM100 GPU
```

Read [test/AGENTS.md](test/AGENTS.md) before touching tests — `test/python/conftest.py` has import-order and env-var requirements that are easy to break.

## Format / lint

```bash
git add <changed files>
pre-commit run                # clang-format 21 (C++/CUDA) + black -l 160 (python + notebooks), staged files only
```

First invocation builds the hook environments and can take >5 minutes; later runs are fast. Run on the files you changed (staged files, or `pre-commit run --files <paths>`), not `--all-files` — some pre-existing files are not currently formatter-clean, and reformatting them would pollute your diff. C++ style is Google-based, 4-space indent, 120 columns (`.clang-format`); Python is black with line length 160.

## Conventions

- `include/` is header-only: no `.cpp` files, no new required dependencies. Vendored third-party code lives in `include/cudnn_frontend/thirdparty/`.
- Every new frontend-only Python API needs: `APIBase` subclass + wrapper, lazy export in `python/cudnn/__init__.py`, docs under `docs/fe-oss-apis/`, and pytest coverage under `test/python/fe_api/`. Full recipe: [python/cudnn/AGENTS.md](python/cudnn/AGENTS.md) and the `cutedsl-kernel-integration` skill.
- Frontend-only OSS APIs are experimental; keep the lazy-import boundary intact (no eager `torch`/`cutlass` imports at `cudnn` import time). CuTeDSL is a required dependency now, but a tensor framework is not, and `import cudnn` still has to stay cheap.
- The `pyproject.toml` floor on `nvidia-cutlass-dsl` (`>=4.6.2`) is the **downstream** floor (vLLM/SGLang inherit quack-kernels' `==4.6.2`), and it is **below** what the FROST-derived kernels need (`CUTEDSL_MIN_VERSION`, 4.7.0). Every backend/kernel gates the DSL version at runtime and declines with an error that names the version — never assume the installed DSL satisfies your kernel. [python/cudnn/AGENTS.md](python/cudnn/AGENTS.md) **Rule 7** is canonical; cite it in review.
- Version lives in three places that must stay in sync: `CMakeLists.txt` (`project(... VERSION ...)`), `include/cudnn_frontend_version.h`, `python/cudnn/__init__.py` (`__version__`).
- Runtime debugging: set `CUDNN_FRONTEND_LOG_INFO=1` and `CUDNN_FRONTEND_LOG_FILE=stderr` for FE logs; backend logs via `CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=stderr`.
- Public-API signatures evolve **append-only**: new parameters go at the end (with defaults), never inserted mid-signature — positional callers across C++, pybind, and Python wrappers break silently otherwise (review on PR #266).
- Never delete an existing log or diagnostic statement in a cleanup/refactor — several were added after repeated hard-to-repro failures and are the only tripwire for a recurrence (review on PR #280). If one looks redundant, ask before removing.
- Every new source file needs the repo's SPDX/license header (flagged in review on PR #747) — enforced by the `spdx-license-header` pre-commit hook.
- Changing any FROST SDPA `Capabilities` field that affects graph eligibility, or adding/retiring an `EngineSpec`, updates [python/cudnn/sdpa/frost/SUPPORT_MATRIX_TRACKER.md](python/cudnn/sdpa/frost/SUPPORT_MATRIX_TRACKER.md) in the same commit — it is maintained by hand and has no other tripwire. A change confined to knob domains (`tile_ms`, `sched_policies`, ...) is exempt. [python/cudnn/sdpa/AGENTS.md](python/cudnn/sdpa/AGENTS.md) **Rule S2** is canonical for the exact scope; cite it in review.

## Agent skills

Reusable task recipes live in `skills/` (auto-discovered by Claude Code via `.claude/skills`; other agents: read the relevant `skills/<name>/SKILL.md` before starting a matching task):

- `skills/cutedsl-kernel-integration/` — integrating a CuTeDSL kernel as a frontend-only Python API end to end (API class, wrapper, exports, docs, tests).

## Links

- Published documentation: <https://docs.nvidia.com/deeplearning/cudnn/latest/developer/overview.html>
- In-repo docs index: [llms.txt](llms.txt) · operation reference in [docs/operations/](docs/operations/) · OSS kernel APIs in [docs/fe-oss-apis/overview.md](docs/fe-oss-apis/overview.md)
- PyPI: <https://pypi.org/project/nvidia-cudnn-frontend/>


## PR labels

When opening a pull request, apply **at minimum one label from each group**: one `cat-*` (change type), one or more `area:*` / `op:*` (affected area), and one `orig-*` (originator). See the full label list at <https://github.com/NVIDIA/cudnn-frontend/labels>.

Leave `closed-*` and `open-*` labels for maintainers; Milestone/Projects sidebar fields are set by reviewers/maintainers, not authors.

## Reviewing a PR (human or agent)

Each Hard Rule is numbered so it can be cited by number in review comments. Before approving or requesting changes on a diff, check it against every Hard Rules section whose directory it touches: [python/cudnn/AGENTS.md](python/cudnn/AGENTS.md) (Rules 1-5, `execute()`/import-time), [python/cudnn/sdpa/AGENTS.md](python/cudnn/sdpa/AGENTS.md) (SDPA Rules S1+), [include/cudnn_frontend/AGENTS.md](include/cudnn_frontend/AGENTS.md) (header-only, warnings-as-errors), plus the conventions in [test/AGENTS.md](test/AGENTS.md) and [samples/AGENTS.md](samples/AGENTS.md). Where a rule names a detector (grep pattern, `set_sync_debug_mode` snippet, a `RED-then-green` test), prefer running or citing it over an eyeballed read. When a review surfaces a new concrete, checkable technique or trap that these guides don't already cover, land it in the relevant `AGENTS.md` in the same PR rather than leaving it in a review comment.
