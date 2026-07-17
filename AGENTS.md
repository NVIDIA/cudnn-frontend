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
- Python ≥ 3.9. `cudnn.backend_version()` gates many features at runtime (integer, e.g. 9.12.0 → `91200`); tests skip on older backends.

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
pip install -e .              # core graph API only
pip install -e ".[cutedsl]"   # + OSS CuTeDSL kernels (torch, nvidia-cutlass-dsl, cuda-python)
```

`setup.py` honors env vars: `CUDNN_PATH`, `CUDA_PATH` / `CUDAToolkit_ROOT`, `DEBUG=1` (debug build), `CMAKE_BUILD_PARALLEL_LEVEL`, `CMAKE_GENERATOR`.

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
pytest fe_api/                # OSS kernel tests; require ".[cutedsl]" install + SM90/SM100 GPU
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
- Frontend-only OSS APIs are experimental; keep the `[cutedsl]` optional-dependency boundary intact (no eager `torch`/`cutlass` imports at `cudnn` import time).
- Version lives in three places that must stay in sync: `CMakeLists.txt` (`project(... VERSION ...)`), `include/cudnn_frontend_version.h`, `python/cudnn/__init__.py` (`__version__`).
- Runtime debugging: set `CUDNN_FRONTEND_LOG_INFO=1` and `CUDNN_FRONTEND_LOG_FILE=stderr` for FE logs; backend logs via `CUDNN_LOGLEVEL_DBG=3 CUDNN_LOGDEST_DBG=stderr`.

## Agent skills

Reusable task recipes live in `skills/` (auto-discovered by Claude Code via `.claude/skills`; other agents: read the relevant `skills/<name>/SKILL.md` before starting a matching task):

- `skills/cutedsl-kernel-integration/` — integrating a CuTeDSL kernel as a frontend-only Python API end to end (API class, wrapper, exports, docs, tests).

## Links

- Published documentation: <https://docs.nvidia.com/deeplearning/cudnn/latest/developer/overview.html>
- In-repo docs index: [llms.txt](llms.txt) · operation reference in [docs/operations/](docs/operations/) · OSS kernel APIs in [docs/fe-oss-apis/overview.md](docs/fe-oss-apis/overview.md)
- PyPI: <https://pypi.org/project/nvidia-cudnn-frontend/>
