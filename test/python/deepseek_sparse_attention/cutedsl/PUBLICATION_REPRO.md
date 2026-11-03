# DSA per-head publication reproduction

Base: NVIDIA `develop`, `42200cd663d13e8a01ee076fe1e157949a35b071`.

## Setup

Use Linux with an SM100-family GPU (B200/GB200/GB300), a CUDA 13 toolkit and
compatible driver, cuDNN 9 headers/libraries, and a C++17 compiler. Put `nvcc`
and `compute-sanitizer` on `PATH`. As in [Building from Source](../../../../README.md#building-from-source),
set `CUDNN_PATH` and `CUDAToolkit_ROOT` if they are installed outside the default locations.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone --branch fix-dsa-publication-upstream https://github.com/MuuSeoTia/cudnn-frontend.git test_fix
cd test_fix
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
```

Run both sections below from this fresh checkout. The checker uses the installed
editable package, disables compiled caches, and runs six numerical regression
cases under racecheck. It writes `publication-results.xml` at the repository root.

## Before: without the fix

Replace only the kernel with the version from the pinned NVIDIA develop commit:

```bash
git restore --source=42200cd663d13e8a01ee076fe1e157949a35b071 -- \
  python/cudnn/deepseek_sparse_attention/score_recompute/dense_score_recompute_sm100.py
python test/python/deepseek_sparse_attention/cutedsl/reproduce_publication.py
```

Expected: **6 passed**, but racecheck reports shared-memory conflicts and the
checker exits **86**. Recorded on GB300: **120 racecheck errors**. Numerical
checks alone do not expose this race; hazard counts can vary by GPU and runtime.

## After: with the fix

Restore the patched kernel from this branch and run the same checker:

```bash
git restore --source=HEAD -- \
  python/cudnn/deepseek_sparse_attention/score_recompute/dense_score_recompute_sm100.py
python test/python/deepseek_sparse_attention/cutedsl/reproduce_publication.py
```

Expected: **6 passed, no skips, zero racecheck errors**, exit **0**. The editable
install picks up the Python kernel change without reinstalling.

## Independent changes

FP32 head weights and dtype-aware compile keys already exist on this base
([NVIDIA #1311](https://github.com/NVIDIA/cudnn-frontend/pull/1311)). This PR
adds only the publication fix.

The load warp's ordinary W/LSE stores need a handoff to epilogue readers;
Q/K tensor-core completion does not publish every load lane's stores. A
two-slot `PipelineAsync` uses all 32 load lanes as producers and all 128
epilogue lanes as consumers. Consumer release prevents premature slot reuse.
Public arguments, weight types, math and the two-slot data buffer are unchanged.
Separate MXFP8/MXFP4 implementations are outside this qualification.

## Recorded qualification

On 2026-11-03, the checker ran against the pinned develop base and this fix:
six passes per arm, racecheck **120 errors → 0**, exit **86 → 0**, no skips.
The regression covers 32-head partial/persistent tiles and a 64-head case
whose first causal tile has no K blocks to compute.

GB300 SM103 (152 SMs), Python 3.12, Torch 2.13.0, CuTeDSL 4.8.1,
cuda-bindings 13.4.4, TVM-FFI 0.1.12, cuDNN 9.22.0.52, pytest 8.4.2 and
Compute Sanitizer 2026.3.0.0. Source Python used the official Frontend 1.32.1
wheel's native extension.

The baseline received only the regression test; the fixed tree additionally
received the kernel change.

| Check | Unfixed | Fixed |
| --- | --- | --- |
| Existing FP32/BF16 support | 36 passed | Not selected |
| Publication numerics | 6 passed | 6 passed |
| Publication racecheck | 120 errors, 100 displayed hazards | 0 errors |
| Focused attention numerics | 2 passed | 2 passed |
| Focused attention racecheck | 8 errors | 0 errors |
| Synccheck, publication and focused attention | 0 errors | 0 errors |
| Shared-mainloop cases | Not selected | 39 passed, 44 deselected |
| Paged BSHD/THD and rejection contracts | Not selected | 6 passed; racecheck/synccheck clean |

Cold numerical runs of the six publication, two focused attention and six paged
cases also passed on each compiler combination below, with fresh processes and
compiled caches disabled. The loaded compiler flavor was checked explicitly.

| CuTeDSL | Compiler CUDA | Result |
| --- | --- | --- |
| 4.8.0 | 12.9 | 14 passed |
| 4.8.0 | 13.4 | 14 passed |
| 4.8.1 | 12.9 | 14 passed |
| 4.8.1 | 13.4 | 14 passed |

The CUDA 12 compiler runs used the same CUDA 13 framework/backend and driver
(driver API 13020); they qualify the DSL flavor, not a CUDA-12-only deployment.

No selected case skipped. These are kernel/API correctness results, not model
accuracy or performance measurements.
