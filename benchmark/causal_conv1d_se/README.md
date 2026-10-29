# CuTe DSL causal-convolution SE benchmarks

Compare the NCW, NWH, and fused projection/mixer kernels in `python/cudnn/causal_conv1d_se` with the cuDNN SE backend through the frontend bindings. Dense BF16 NWH K4 SiLU cases also compare the existing frontend bulk implementation. These are low-level candidates; public operator routing is unchanged.

The directory contains a fixed workload runner, its comparison/timing helpers, workload manifests, and result collectors. Correctness tests live in `test/python/causal_conv1d/cutedsl`. Historical performance and compilation measurements, optimization details, and remaining acceptance gaps are recorded in [PR #1945](https://github.com/NVIDIA/cudnn-frontend/pull/1945).

## Contracts

| Family | Input | Filters | Widths | Outputs |
| --- | --- | --- | --- | --- |
| NCW | `[B,D,L]` contiguous | `[D,K]` | `2 <= K <= 256` | forward; `dx`, `dw`, `db` |
| NWH / NWC | `[B,L,D]` contiguous | `[K,D]` | `2 <= K <= 128` | forward; `dx`, `dw`, `db` |
| Back-to-back | `[B,3*D,L]` contiguous | projection `[3*D,Kp]`, mixer `[D,Km]` | `2 <= Kp <= 32`, `2 <= Km <= 256` | ungated and gated forward; `dx`, `dwp`, `dwm`, `dskip` |

Inputs, filters, bias, and data outputs share one of FP16, BF16, FP32, or FP64. Parameter gradients use FP32 except for FP64 inputs, which use FP64 gradients. Bias is a required `[D]` tensor for NCW/NWH. These low-level kernels assume valid shapes, matching dtypes, dense storage, aligned vector accesses, and disjoint input/output buffers; they do not implement the frontend lifecycle's runtime validation.

The last filter tap multiplies the current token. NCW/NWH support identity and SiLU. The back-to-back channel order is interleaved `(x1, x2, v)` for each feature. Its projections have no activation:

```text
p1 = causal_conv(x1, wp1)
p2 = causal_conv(x2, wp2)
pv = causal_conv(v, wpv)
u = p2 * pv
y = causal_conv(u, wm) + skip * u
y_gated = p1 * y
```

Back-to-back backward receives the gradient of `y_gated` and uses the saved, input-dtype `y` for the gradient of `p1`, matching the backend ABI. It does not accept an independent gradient for the auxiliary `y` output. The mixer product and its previous-tile halo stay in the compute dtype, including non-vectorized sequence lengths.

Parameter-gradient accumulation uses atomics and is not bitwise deterministic. The backward launchers reset parameter outputs by default. Compile with the final `accumulate=True` argument to preserve existing parameter gradients and add to them, matching the backend ABI. Data gradients are overwritten in either mode. The fused back-to-back kernels need no global intermediate workspace. Standard backward retains a caller-owned compute-dtype `dz` workspace for the wide-filter NWH path; the fused NCW and short-filter NWH paths do not use that workspace.

The candidate launchers perform no allocation, tensor conversion, or synchronization. Compile outside capture and pass the active CUDA stream at execution. Full public API integration, including argument checks and architecture admission, is still required.

## Existing API contracts and replacement scope

The existing `causal_conv1d_bulk_sm100` package processes complete dense or packed sequences for prefill and training. Its autograd adapter supports input, filter, bias, and initial-state gradients; a returned final state can also receive a gradient. The semantic `cudnn.ops.causal_conv1d` API adapts public `[B,D,L]` tensors and `K-1` history states to the native package's layouts. Its native route also has mixed FP32-filter support, reusable compiled plans, and a deterministic bias-free backward mode.

The `causal_conv1d_update_sm100` package processes one new token per active sequence during decoding. It updates history in place, can route tokens to cache slots through `conv_state_indices`, and supports padding slots. It is inference-only and explicitly rejects autograd inputs. Preserving this decode path does not require implementing decode in the dense SE kernels. The folder suffixes do not define the complete admitted architecture set.

The existing generic NCW, NWH, and fused back-to-back functions in `python/cudnn/ops/causal_conv1d.py` already register autograd and FakeTensor implementations around their backend calls. The added SE launchers provide the underlying forward/backward computations but are not connected to these registrations. Replacing the backend should reuse that API layer and preserve parameter-gradient casts, optional-bias handling, saved fused outputs, stream behavior, `torch.compile` support, and allocation/cache ownership. Atomic SE parameter reductions do not provide the native bulk path's deterministic guarantee.

The dense SE comparisons do not establish parity for packed sequences, initial/final states, mixed filter dtypes, activation checkpointing, or prefill-to-decode handoff. Integration must retain the existing native routes for these contracts and run the semantic API, autograd, and prefill/decode seam tests. These are direct operator APIs; no cuDNN graph API integration is required. CUDA Graph capture/replay remains an execution contract tested separately.

## Run the fixed matrix

Use the active task worktree and a Python environment containing matching frontend bindings, PyTorch, CUDA Python, and the CuTe DSL version required by that frontend revision. Select the GPU with `CUDA_VISIBLE_DEVICES`. Set `CUDNN_SE_BACKEND_LIBRARY` to the exact `libcudnn_ext.so` under comparison and configure the runtime loader to use it. `CUDA_HOME` must supply the CUDA/CCCL headers for backend JIT compilation, and the NVRTC major version must match the cuDNN build. The harness verifies the loaded library's path and file identity and records its SHA-256, size, and frontend-reported backend version.

Run all four suites on an idle GPU:

```bash
cd "${CUDNN_FRONTEND_GITHUB_DIR:?}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SE_BENCH_RESULTS=/tmp/causal-se

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" benchmark/causal_conv1d_se/controlled_matrix.py \
  --backend-library "${CUDNN_SE_BACKEND_LIBRARY:?}" --output "$SE_BENCH_RESULTS/main"
"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" benchmark/causal_conv1d_se/controlled_matrix.py \
  --backend-library "$CUDNN_SE_BACKEND_LIBRARY" --accumulate --output "$SE_BENCH_RESULTS/additive"
"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" benchmark/causal_conv1d_se/controlled_matrix.py \
  --backend-library "$CUDNN_SE_BACKEND_LIBRARY" --supplement --output "$SE_BENCH_RESULTS/supplement"
"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" benchmark/causal_conv1d_se/controlled_matrix.py \
  --backend-library "$CUDNN_SE_BACKEND_LIBRARY" --supplement --accumulate --output "$SE_BENCH_RESULTS/supplement-additive"
```

`workloads.json` covers the named non-FFT backend cases and the frontend B=1,2,4,8 batch sweep. The main runner adds the common frontend workload, producing 77 direction-level comparisons; additive mode runs its 38 backward directions. `supplemental_workloads.json` adds 32 directions, including NCW K256, NWH K7/K64/K128, and fused Kp32/Km256, with eleven additive backward directions. Together these are 158 comparisons per GPU before backend exclusions. Backend-unsupported cases remain explicit in `summary.json`. Source hashes are checked after every subprocess; an edit during a sweep aborts it. Use `--kernel-dir` to select a frozen candidate source directory.

For one workload, reuse a generated `*.case.json`:

```bash
"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" benchmark/causal_conv1d_se/compare_variants.py \
  --case "$SE_BENCH_RESULTS/main/000-forward.case.json" --direction forward \
  --variant candidate=python/cudnn/causal_conv1d_se/kernel.py \
  --shared-output-timing --output "$SE_BENCH_RESULTS/single.json"
```

Select `backward_kernel.py` for standard backward or `b2b_kernel.py` for fused operations. Add `--frontend` only for BF16 NWH K4 SiLU overwrite comparisons. `--accumulate` selects additive backward, and `--profile` emits CUDA profiler/NVTX ranges. Repeated `--variant label=path` arguments compare source variants; use the separate `paired` and `frontend_paired` measurements for conclusions because adding timing arms can change cache behavior.

## Accuracy and timing

Before timing, each candidate must pass three backend comparisons. Data outputs (`y`, `yg`, `dx`) require equal raw bits and finite values. Only atomic parameter gradients (`dw`, `db`, `dwp`, `dwm`, `dskip`) use tolerances: FP32 `atol=max(1e-4, 2e-5*sqrt(B*L))`, `rtol=2e-4`; FP64 `atol=rtol=1e-10`. The backend is unchanged. Existing FE data outputs use `atol=rtol=0.03`, with raw-bit differences also retained; its parameter gradients use the same atomic-gradient contract.

Device timing uses preallocated CUDA Graphs on the active stream, a warmup period, 100 kernel executions per sample, and 21 samples in randomized order. Compilation and allocation are excluded. Backend and candidate use identical output addresses; a backend A/A control records sensitivity to output placement. Overwrite timings include parameter clearing in both arms. Additive timings exclude clearing in both arms and validate accumulation from nonzero buffers. Raw samples and round positions are retained. Bootstrap intervals describe within-run sampling, not repeatability across machines, clocks, or cache layouts.

Each result records candidate `cute.compile` wall time separately. This harness does not measure backend NVRTC compilation or the existing FE's full initialization/cache lifecycle; those require separate instrumentation. The historical three-way compiler study remains in the PR report.

## Collect results

Save the expected kernel hashes from the frozen source before starting a sweep:

```bash
"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
root = Path("python/cudnn/causal_conv1d_se")
output = Path(os.environ["SE_BENCH_RESULTS"])
output.mkdir(parents=True, exist_ok=True)
names = ("kernel.py", "backward_kernel.py", "b2b_kernel.py")
(output / "source-hashes.json").write_text(json.dumps({n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names}))
PYCODE

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" benchmark/causal_conv1d_se/summarize_controlled.py \
  --run "$SE_BENCH_RESULTS/main" --source-hashes "$SE_BENCH_RESULTS/source-hashes.json" \
  --output "$SE_BENCH_RESULTS/main-summary"
```

Repeat `--run` to combine GPUs running the same suite and frozen source. Collect the other suites separately with the matching `--accumulate` and/or `--supplement` flags. The collector checks completeness, source identity, correctness acceptance, and shared output addresses. It exports device comparisons, candidate compilation times, and a summary retaining every slower median. `--original-regressions <csv>` associates earlier losses with the new measurements.

## Run correctness tests

```bash
cd test/python
CUDNN_SE_BITWISE_RESULTS="$SE_BENCH_RESULTS/bitwise" \
  "${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" -m pytest -n 0 -m 'L0 or L1' \
  causal_conv1d/cutedsl/test_causal_conv1d_se.py \
  causal_conv1d/cutedsl/test_causal_conv1d_se_pipeline.py \
  causal_conv1d/cutedsl/test_causal_conv1d_se_tools.py \
  causal_conv1d/cutedsl/test_causal_conv1d_se_bitwise.py
```

The focused tests check independent numerical references, causal boundaries, output reset, accumulation, sequence/channel tails, changed inputs during CUDA Graph replay, and a nondefault stream. Pipeline tests cover asynchronous-buffer reuse, masked stages, FMA rounding profiles, and fused filter/sequence halos. The L1 bitwise suite reproduces all 74 named `levels.layer` cases through candidate and backend frontend bindings. `extract_levels.py --backend-root <checkout> --ref <revision> --output <manifest.json>` regenerates this pinned correctness fixture; it is distinct from the direction-grouped performance manifest.

From the repository root, `summarize_bitwise.py --input <bitwise-directory> --output <summary-directory>` exports per-output comparisons, missing cases, backend exclusions, and nondeterministic parameter reductions. Repeat `--input` for other GPUs. Its exit status is nonzero unless every expected case passes; inspect the exported statuses to distinguish backend exclusions from mismatches.

The current FP32 NWH rounding profiles target the verified cuDNN 9.29 backend with NVRTC 13.4–13.6 and need revalidation when the backend or compiler changes. The full seven-architecture performance matrix has not been rerun after the modern-compiler correction; six historical backend performance losses remain open. Public autograd/FakeTensor routing, validation, and cache/workspace ownership also remain integration work.
