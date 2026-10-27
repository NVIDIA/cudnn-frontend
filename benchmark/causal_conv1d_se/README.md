# CuTe DSL causal-convolution SE kernels

This directory compares CuTe DSL implementations in `python/cudnn/causal_conv1d_se` with the cuDNN SE implementation and the existing frontend kernels. These are **pre-integration candidates**. The public `cudnn.ops` routing has not changed. Architecture coverage and performance gates must pass before a default route is added.

## Validation status

The historical CuTe DSL 4.7.0 sweep, using backend NVRTC 13.6 and cuDNN 9.29.0.2, accepted 506/518 named manifest comparisons; 12/518 were backend-unsupported because FP64 NWH exceeded shared-memory admission on SM86/89/120. Separately, the focused suite passed 175/175 and the boundary suite passed 926/938, with 12 exact-bit data-check failures. Backend-unsupported manifest cases and boundary accuracy failures are different cases.

The current correction passes all 12 previously failing configurations in focused reruns with CuTe DSL 4.8.0. It updates the FP32 NWH first-pair FMA profiles for modern backend NVRTC and selects the correct profile for the second masked buffer stage. The backend and acceptance tolerances are unchanged. The historical totals above remain the original measurements; the full seven-architecture sweep has not been rerun on this source.

The historical device matrix retains six slower candidate medians against the backend among 1,088 supported comparisons. The candidate wins all 84 measured directions against existing FE. These aggregate results do not establish complete backend replacement or performance of the full rebased tree with CuTe DSL 4.8.0.

### Modern-compiler rounding correction

The original failures are FP32 NWH input-gradient mismatches at B=2, D=1032, L=1031: K2 identity, K3 identity, and K3 SiLU on SM80/86/89/90. The old candidate passed these cases against backend NVRTC 13.0 but failed them with both 13.5 and 13.6. The correction targets the verified 13.4–13.6 contraction profiles. It does not claim bitwise compatibility with the different 13.0 profiles; any future public route needs compiler/backend admission.

The focused check combines 42 existing FP16/BF16/FP32 K2–K8 rounding cases with 18 new FP32 tail cases covering each masked axis, both buffer stages, and small grids. Every case runs three trials. Data gradients must match raw bits; only atomic parameter gradients use the existing tolerances below.

| GPU / architecture | Backend NVRTC 13.5 | Backend NVRTC 13.6 |
| --- | --- | --- |
| A100 / SM80 | 60/60 | 60/60 |
| A40 / SM86 | 60/60 | 60/60 |
| L40S / SM89 | 60/60 | 60/60 |
| H100 / SM90 | 60/60 | 60/60 |

These 480 case/compiler/architecture checks use cuDNN 9.29.0.2 and CuTe DSL 4.8.0. A100/L40S 13.6 and A40 13.5/13.6 run the committed test functions against the final kernel source. The other runs use a frozen equivalent kernel whose Python AST matches the final kernel after removing docstrings, with the same cases, seeds, and three-trial backend comparison. Actual NVRTC calls are instrumented. Three preliminary runs stopped when the source-change guard detected an edit; the complete reruns above supersede those partial results.

A separate H100 control using the CI backend, cuDNN 9.27.0.42 debug, and verified NVRTC 13.4.92 also passes 60/60. The original CI log did not record the NVRTC patch level, so this control does not establish an identical original compiler binary. These local runs reuse a compatible Python 3.12 native binding and overlay the selected Python sources; they do not replace a full frontend rebuild or current-head CI.

A focused A100 device check of the corrected source uses cuDNN 9.29.0.2, NVRTC 13.6, and CuTe DSL 4.8.0. All rows are FP32 NWH backward at B=2, D=1032, L=1031. Shared output addresses, parameter clearing in both arms, and 21 randomized paired CUDA Graph samples follow the controlled harness. Compilation is excluded.

| Filter / activation | Backend device µs | Candidate device µs | Backend / candidate |
| --- | --- | --- | --- |
| K2 / identity | 25.170 | 14.070 | 1.79× |
| K3 / identity | 26.378 | 13.568 | 1.94× |
| K3 / SiLU | 46.356 | 19.139 | 2.42× |

This is one focused run on one GPU. Existing FE has no equivalent FP32 K2/K3 arm in this harness. The six historical performance regressions, the complete architecture matrix, and public API integration remain open.

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

## Run

Use the active task worktree and a Python environment containing the matching frontend bindings, PyTorch, CUDA Python, and the CuTe DSL version required by the selected frontend revision. Set `CUDNN_SE_BACKEND_LIBRARY` to the exact `libcudnn_ext.so` under comparison. Its NVRTC major version must match the cuDNN build, and `CUDA_HOME` must provide the CUDA/CCCL headers needed by backend JIT compilation.

```bash
cd "${CUDNN_FRONTEND_WORKTREE_DIR:?}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" \
  benchmark/causal_conv1d_se/probe.py \
  --layout nwc --shape 2,768,4096 --width 4 --dtype bfloat16 \
  --vector 4 --backend-library "${CUDNN_SE_BACKEND_LIBRARY:?}"

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" \
  benchmark/causal_conv1d_se/perf_matrix.py --fixed \
  --backend-library "${CUDNN_SE_BACKEND_LIBRARY:?}" --output /tmp/causal-se-forward

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" \
  benchmark/causal_conv1d_se/backward_matrix.py \
  --backend-library "${CUDNN_SE_BACKEND_LIBRARY:?}" --output /tmp/causal-se-backward

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" \
  benchmark/causal_conv1d_se/manifest_matrix.py \
  --backend-library "${CUDNN_SE_BACKEND_LIBRARY:?}" --output /tmp/causal-se-manifest

"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" -m pytest -n0 -q \
  test/python/causal_conv1d/cutedsl/test_causal_conv1d_se.py
```

For a three-way forward comparison, run `tune.py --frontend --schedules auto`. For backward, use `tune_bwd.py --frontend --rows 32`. Both require the common dense BF16, K4, SiLU workload in NWH layout. The existing frontend's stateful, packed, decode, and mixed-filter-dtype features have no equivalent in this SE ABI and require separate integration coverage.

`--schedules auto` uses a fixed, provisional policy in the harness. It does not select the fastest measured result for each case and is not a public frontend dispatch policy. Explicit `V:R` schedules remain available for tuning.

## Measurement and evidence

The original probe and tuning tools check an independent FP64 direct-convolution reference before timing. The controlled comparison tools require exact backend data bits and the approved parameter-gradient limits before timing. Backward checks every returned gradient. The tests also exercise causal boundaries, parameter-output reset, sequence/channel tails, changed inputs during CUDA graph replay, and a nondefault stream.

Timing uses preallocated CUDA graphs on the same stream, randomized interleaving of candidates and baselines, a warmup period, and 21 recorded samples. Compilation and allocation are excluded. Overwrite-mode timings include parameter-output initialization for every arm. Additive-mode timings omit initialization for every arm; their correctness checks start from nonzero parameter buffers. The default graphs repeat an operation 100 times on warm inputs. `tune.py --cold` and `tune_bwd.py --cold` instead evict with a 512 MiB buffer before each single-operation sample; eviction time is excluded. Retain raw samples and report cache policy with any speedup. Single-operation cold-cache samples have less timing precision; inspect their spread before drawing conclusions near parity.

`collect_results.py --input <result-directory> --output <report-directory>` exports a CSV and a per-GPU summary. Repeat `--input` to combine grids. For duplicate cases, the first supplied result wins; the exporter never chooses the fastest repeat. It rejects missing or mismatched kernel hashes and keeps backend rejections in `exceptions.json`.

`workloads.json` records 41 distinct configurations from every named non-FFT backend case and the complete frontend batch sweep (1, 2, 4, and 8). `manifest_matrix.py` executes 70 jobs, including both directions of the common frontend workload; back-to-back jobs validate and time both directions. This yields 77 direction-level comparisons before backend exclusions. The manifest retains the original source case names and revisions. Gradient references run in channel chunks to bound FP64 memory. Use `--reuse <result-directory>` only for previously completed fixed-policy comparisons; reuse requires matching GPU model, compute capability, kernel digest, backend path, and schedule. Keep the backend binary immutable while reusing results. Fused reuse requires positive, finite timings for both candidate and backend in both directions; an untimed correctness probe is rerun with timing enabled.

The supplementary forward grid has 29 cases from backend `scripts/levels.layer` and the frontend long-context benchmarks. It covers both layouts, short and long sequences, batch 1 through 64, FP16/BF16/FP32, and wide filters. FP64 and additional boundary cases belong to the correctness grid. `backward_matrix.py` adds 12 fixed standard-backward cases and four back-to-back cases, including the backend's sequence-tail and maximum-width workloads.

The local investigation pins frontend base `edc071fee` and backend source `637f900b84aa71a98745e6b455f5e748457cb0d7`. Its backend binary is cuDNN `9.29.0.2`, CUDA `13.5` packaging. All eight non-FFT kernel source files were found byte-for-byte inside that library; the local `backend-source-verification.json` records offsets and hashes. Keep equivalent provenance when changing the binary.

A backend rejection is not a numerical failure or a successful comparison. In particular, the legacy FP64 NWH backend exceeds shared-memory capacity on several consumer-class GPUs. Preserve its status separately from candidate correctness.

## Acceptance status

Do not infer all-architecture support from a successful compile on one GPU. Public DSL 4.7 rejects `sm_75` and `sm_107a` on real T4 and Rubin hardware in this investigation. The SE backend runs on T4, so the requested complete backend architecture coverage cannot be met by that public DSL runtime. An available internal DSL runtime executes a basic candidate on SM107; that is separate evidence, not public-runtime support.

The local result set includes earlier tuning experiments as well as later fixed-policy runs. Only compare measurements from the same source snapshot, input configuration, GPU, and cache policy. Remaining regressions and untested architectures must stay visible; a geomean or a selected win does not establish acceptance.

## Exact comparisons with backend frontend bindings

`test/python/causal_conv1d/cutedsl/test_causal_conv1d_se_bitwise.py` preserves all 74 named non-FFT cases from backend `scripts/levels.layer`: 36 NCW, 25 NWH, and 13 fused back-to-back cases. The fixture pins revision `189b7430650dba30f036afbcaa0f83eaef8a6632`; the 58 cases in the original active backend checkout are an exact subset. Repeated L1/L3 configurations remain separate test IDs and use independent seeds.

The test calls the six `cudnn.causal_conv1d_*` / `cudnn.b2b_causal_conv1d_*` frontend bindings and the private CuTe DSL launchers on identical inputs. It compares integer views of every output, including FP32 parameter gradients for FP16/BF16 inputs. Raw diagnostics preserve signed-zero differences and all parameter-gradient bit mismatches. Forward outputs and input gradients must match exactly. Only atomic parameter gradients (`dw`, `db`, `dwp`, `dwm`, `dskip`) may use numerical tolerances, for comparison with the unchanged backend. Nonfinite outputs fail. Fused backward receives the same saved backend `y` in both arms to isolate backward arithmetic; the forward tests check both `y` and `y_gated` separately.

Each case runs at least three times, or ten for the manifest's `N:10` cases. The results record cross-implementation equality and repeatability of each implementation. The test records strict bitwise equality separately from acceptance. Parameter gradients use `atol=max(1e-4, 2e-5*sqrt(B*L))`, `rtol=2e-4` for FP32 accumulation, and `atol=rtol=1e-10` for FP64 accumulation. These reduction tolerances follow the existing backward reference benchmark and do not depend on the input's half precision. Every element must satisfy `abs(candidate-backend) <= atol + rtol*abs(backend)`. Data-output bit mismatches and parameter errors beyond those limits fail. Status 3010 is reported separately as an unsupported backend comparison.

```bash
cd "${CUDNN_FRONTEND_WORKTREE_DIR:?}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
# LD_LIBRARY_PATH must select the exact cuDNN backend and matching NVRTC runtime.
export CUDNN_SE_BITWISE_RESULTS=/tmp/causal-se-bitwise
"${CUDNN_FRONTEND_PYTHON_VENV:?}/bin/python" -m pytest -n0 -m L1 -q \
  test/python/causal_conv1d/cutedsl/test_causal_conv1d_se_bitwise.py
```

The fixture can be regenerated with `benchmark/causal_conv1d_se/extract_levels.py --backend-root "$CUDNN_WORKTREE_ROOT" --ref <revision> --output test/python/causal_conv1d/cutedsl/se_levels_cases.json`. Add `--active-levels <path>` to verify another checkout's cases are an exact subset.

Raw-bit testing found and corrected short-filter NWH tap order, fast SiLU division/reciprocal, initial FP32 NWH input-gradient contraction, and an incorrect low-precision cast in the fused forward halo. Earlier tolerance-based results do not establish bitwise equality, and timings from earlier source hashes do not validate the corrected revision's performance. The NWH backward arithmetic also mirrors the backend's shared-memory admission, because its legacy fallback changes the tap order and SiLU calculation. `frontend_overlap.py --output <directory>` checks exact data bits and the same parameter-gradient limits for the common BF16 K4 SiLU case and the frontend's complete B=1,2,4,8 long-context sweep.

Parameter-gradient reduction order prevents full raw-bit parity; the approved acceptance contract allows only those atomic outputs to differ within the stated limits. The candidate remains private until accuracy, performance, architecture coverage, and public lifecycle integration pass.

## Regression, startup, and cache investigation

`investigate.py --source-csv <comparisons.csv> --output <directory>` replays every slower candidate/backend comparison for the current GPU architecture. Use the same GPU model and an idle or exclusive allocation. It retains raw latency samples, source hashes, compilation wall times, and backend first-dispatch time. The latter includes initialization and execution; it is not a pure compiler measurement. `--sweep` explores legal tile choices without changing the fixed baseline policy. `--profile` emits CUDA profiler ranges for Nsight Systems.

`summarize_regressions.py --baseline <comparisons.csv> --replay sm80=<directory> ... --output <directory>` accounts for every original regression, including losses below 5%. Its 95% bootstrap interval describes within-run latency sampling. It does not establish repeatability across machines, clocks, or cache layouts. The category column groups source-level investigation targets; it is not proof of an individual root cause.

`compare_variants.py --case <retained-timing.json> --direction backward --variant baseline=<kernel.py> --variant experiment=<modified-kernel.py> --output <result.json>` uses identical addresses and inputs for both variants. It requires raw-bit equality for data outputs and the approved parameter-gradient tolerances before timing. A `:rows` suffix selects a backward tile size. Use the `paired` results for conclusions: each variant is compared separately with the backend. The multi-variant `timing` field is supplementary because adding arms can change cache behavior. Compile outside capture; callbacks use the current capture stream.

`compile_cache_probe.py --output <cold.json>` tests the frontend's existing `cudnn.frost.compiled_cache` helper with all six kernel families/directions. Set `CUDNN_FRONTEND_COMPILED_CACHE` to a new, task-specific directory. Then run a fresh process with `--require-disk-hit --output <reload.json>` and the same directory. The second process forbids `cute.compile` on reuse, checks the hit counters, compares outputs, and replays CUDA graphs. This is a cache integration proof, not a public candidate lifecycle. A production key must identify every shape/stride, dtype, activation, schedule, target, and compile option that changes generated code.

`feature_contract_probe.py --output <result.json>` checks the integrated candidate's additive parameter-gradient contract with nonzero initial buffers and two backward calls. Pass the compile-time `accumulate=True` argument to `launch_bwd` or `launch_b2b_bwd` for the backend's additive behavior. The default `False` clears parameter gradients before execution. The probe also checks exact NCW forward output at batch 65,536; the candidate flattens this grid to avoid CUDA's grid-z limit. Neither this probe nor the 74-case manifest exhausts the backend's supported shape and filter-width domain.

The backward NWH pipeline stages input and output-gradient tiles with `cp.async`, retains parameter partials across the sequence chunk, and reduces them once per CTA. Its tile shape depends on architecture and workload size. Blackwell uses packed FP32 FMAs where the backend's arithmetic permits them. NCW and fused paths retain halos in registers and use a shared-memory transpose for sequence tails. These schedules target the backend's tap order and data-output rounding; the modern-compiler correction above closes the observed exceptions in focused checks.

For close performance comparisons, add `--shared-output-timing` to `compare_variants.py`. It records a backend A/A buffer-placement control and uses the same output buffers for backend and candidate timing. `--frontend` adds the existing BF16 NWH K4 SiLU forward or backward implementation, sharing the data output buffer; backward parameter-weight layout still requires a separate buffer. The `frontend_paired` field compares each candidate with that implementation. Existing frontend data outputs are checked numerically and their bit differences are retained; the candidate must still match backend data outputs bit for bit.

`manifest_matrix.py` includes the 68 manifest timing jobs and both directions of the common frontend workload, producing 77 direction-level comparisons before backend exclusions.

## Integrated scheduling and reuse

NCW uses flattened launch grids except for measured architecture-specific short-sequence schedules. Backward stages unaligned sequence tiles through warp-local shared memory to coalesce loads and stores. Temporal groups retain parameter-gradient partials until the final reduction. Large batches use flattened grids to avoid the CUDA z-grid limit.

The short-filter NWH backward pipeline double-buffers input and output-gradient tiles with asynchronous copies, retains weights and parameter-gradient partials across multiple sequence tiles, and reduces once per CTA. Blackwell schedules use packed FP32 FMA while preserving the per-element tap order. NWH forward retains a circular convolution window. Selected large workloads use three shared buffers, prefetch two tiles ahead, and carry the halo between buffers without reloading it from global memory. Other admitted workloads retain the two-buffer schedule. Streaming output stores avoid a measured L40S cache-policy regression.

Fused forward retains input, projection, and product halos across sequence tiles for both short and wide filters. Wide-filter weights are loaded cooperatively into distributed registers. Low-parallelism long half-precision workloads use shared filter weights on SM80/90, and for FP16 on SM86/89. SM100 and SM103 split these sequences across CTAs and reconstructs the preceding projection/product halo; the flattened grid avoids adding a sequence-dependent grid-z limit. SM120 uses 16 elements per thread for this workload group. Fused backward finishes each projection gradient before extending other live values, reuses coalesced tail storage, and retains parameter partials. Its wide-filter path reuses distributed register weights on every tested architecture. It explicitly preserves the backend's first-tap FMA order between the skip contribution and mixer filter. SM100 uses 16 elements per thread and four warps for the long, wide-filter half-precision path with at most 128 batch/channel pairs. The admitted long half-precision small-filter path uses 16 elements per thread and four warps on Ada, and eight elements per thread and eight warps on SM100. Short Blackwell cases use one warp. Independent parameter reductions are interleaved by shuffle stage, and the final atomics are distributed across lanes. The warp reduction tree and per-element data arithmetic are preserved; changing the temporal grouping can change parameter-gradient bits within the approved tolerance.

`test_causal_conv1d_se_pipeline.py` adds raw-bit backend comparisons for asynchronous-buffer reuse, partial sequence/channel tiles, FP16/BF16, identity/SiLU, and partial fused-backward warp groups. These supplement the unchanged 74-case manifest.

`controlled_matrix.py --backend-library <libcudnn_ext.so.9> --output <directory>` runs every fixed workload with identical candidate/backend output addresses. It verifies that the single loaded Linux SE library matches the requested file, records its resolved path, SHA-256, size, and frontend-reported backend version, and rejects missing or ambiguous library selections. It includes the common frontend case and the B=1,2,4,8 frontend sweep, requires three exact-data comparisons before timing, records paired timings and a backend A/A allocation control, and checks source hashes throughout. This produces 77 directions per GPU before backend rejections. Add `--accumulate` to run all 38 backward directions with parameter clearing excluded from both timed paths. The additive sweep is a separate acceptance gate, because the backend ABI adds to existing gradients. The candidate must not rely on cheaper clearing to hide a slower convolution kernel. Preserve the standard manifest timings alongside this controlled protocol; changing the allocator arrangement does not erase earlier slower medians.

Short NWH backward also reproduces the pinned cuDNN 9.29 compiler's first-pair FMA contractions. The backend contracts full and masked 64x64 tiles differently, with additional differences between its two buffer stages for FP32 K3 and K7/K8 identity. `_nwh_pair_masks` records these finite lane profiles; `_nwh_first_pair` selects the explicit FMA order without changing the mathematical convolution or parameter reductions. Profiles were checked against the actual backend binary on SM80/86/89/90/100/103/120. They require revalidation if the backend or its compiler changes. The 42 additional rounding cases exercise FP16/BF16/FP32, K2–K8, both activations, both buffer stages, and channel/sequence tails. The 18 modern-compiler masked-stage cases bring the full pipeline test file to 152 cases. The additional tests cover three-buffer reuse, wide fused filter tails, NCW wide-filter SiLU gradients, and NWH forward fallback admission. Four more cases check wide fused backward in FP32/FP64 with full and partial sequence tiles. Four cases check aligned split-CTA forward and backward halos with multiple batches. The forward fallback must match the backend's descending tap order and legacy SiLU calculation when the sequence-parallel backend kernel is not admitted.

`supplemental_workloads.json` adds 32 directions from the original tuning grids and the wide-filter boundary investigation, including NCW K256, NWH K7/K64/K128, medium-size NWH, and fused Kp32/Km256. Run `controlled_matrix.py --supplement` and then `--supplement --accumulate` for its eleven backward directions. Together with the main matrix, this gives 158 directions per GPU before backend exclusions. These supplementary cases are a separate gate; a pass on the 74 named manifest cases does not establish their correctness or performance.

`summarize_controlled.py --run <directory> --source-hashes <sha256.json> --output <audit-directory>` exports every paired comparison and compilation time. Repeat `--run` for the other GPUs. Add the matching `--accumulate` and/or `--supplement` flags for each suite. The collector checks the expected direction count, the frozen source hashes, correctness acceptance, and shared output addresses. It retains every slower median and its sampling interval. `--original-regressions <csv>` links the original losses to the new measurements. Use `controlled_matrix.py --kernel-dir <frozen-source-directory>` when reproducing an earlier revision.

For startup measurements, `compile_cache_probe.py --large-nwh` adds the staged NWH forward/backward schedules to the six family/direction checks. Report first-process initialization separately from later disk loads and in-process memo hits. The private launchers still require the caller to manage compilation, validated tensor contracts, device ownership, and workspaces; this probe does not turn them into a public cached API.


The final scheduling pass adds shared K128 filter storage with packed FP32 output pairs on SM100/103, one-warp short NCW K32 reductions on SM100/103, revised NWH pipeline geometry on SM100/120, and sequence splitting for wide fused backward on SM103/120 when batch/channel parallelism is small. Large half-precision K4 NWH identity forward on SM100 uses a three-buffer 32-row schedule. Each backward split recomputes its future data-gradient halo. Parameter sums include only its assigned sequence tile. The expanded boundary suite checks both pipeline geometries, full and partial shared-filter tiles, aligned and masked short reductions, and multiple batches with mixer widths 64, 128, and 256.
