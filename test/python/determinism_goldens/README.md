# Attention determinism

For API support by GPU and datatype, see the
[determinism status page](../../../docs/operations/determinism.md).
This report holds test coverage, measured results, and failure diagnostics.

### SM90 KDA FP32 summaries — 16-token staging, 2026-11-11

The final SM90 path shares a 16-token K/gate/decay/beta tile across 16 rows
in a 512-thread CTA. Reverse state CTAs also share scaled Q; transition CTAs
do not read Q, V/dO or the seed. Every row retains the direct FP32 recurrence
and reduction order. No BF16/TF32 state operands, prefix inverse, global
scratch, host length read, additional launch or execute-time compile is added.
This is shared preprocessing, not the experimental block triangular solve.

Rebased on `develop@5887e729f`. The complete summary suite passed **62 tests,
0 failures, 0 skips** on H20-3e with Torch 2.9.1+cu128, CUDA 12.8 and cuDNN
91002 (3.57 s). The previously committed FP32 goldens passed unchanged, with
recording disabled. The two physical wide-stride cases used an actual BF16
token pitch of `2**32 + 128` elements and poisoned wrapped addresses. These
cases skip only if less than the required approximately 9 GiB is free.
Existing Hopper apply controls passed **17 tests**, with **1 skipped** because
the optional cuTile runtime is not installed. The native NVRTC path does not
use CuTeDSL; CUDA 13 and other GPUs remain unverified for this addition.

Added cases cover 1/15/16/17/31/32/33/65-token boundaries, empty spans and
zero sequences, zero/near-zero/strong/mixed per-channel gates, variable beta,
1024-token summaries, large FP32 seeds, nondefault backward scales, and six
changed-gate graph iterations per direction (three FP64 checks followed by
three transition-independence replays). All previous strided, INT32/INT64,
reference, autograd, logical-CP64, concurrent-work, packing, and malformed
binding checks are retained. Tolerances and existing goldens are unchanged.

Compute Sanitizer memcheck and racecheck each passed **22 cases**, selecting
strided capture, tile/gate boundaries, changed-gate capture and optional-transition
empty cases. Memcheck reports 0 errors; racecheck reports 0 errors / 0 warnings.
Set **both** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` and
`PYTORCH_ALLOC_CONF=expandable_segments:False`: setting only the newer name
allowed conftest's legacy default to enable expandable allocation, causing
`cuMemCreate` permission error 800 under instrumentation before the kernels.
The successful runs took 21.88 s and 118.61 s. The already-recorded nanobind
shutdown reference warnings remain; no host-leak fix is claimed.

Reproduction from `test/python` (recording must stay unset):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest linear_attention/hopper/test_kda_summary_sm90.py -m L0 -q -s

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
PYTORCH_ALLOC_CONF=expandable_segments:False \
compute-sanitizer --tool racecheck --error-exitcode 78 \
python -m pytest linear_attention/hopper/test_kda_summary_sm90.py -m L0 -q -s \
  -k 'strided_rebind_capture or tiled_summary_boundaries or changed_gate_capture or state_only_empty'
```

Use `--tool memcheck --error-exitcode 77` for the corresponding memory check.

Self-review found a nonnull zero-length `DeviceView` was incorrectly rejected
as aliasing a live output. The new regression failed in both directions before
the guard fix, then passed. A Torch empty view alone did not expose this because
its exported address was normalized; the detector uses a real `DeviceView`.

The committed harness `benchmark/linear_attention/benchmark_sm90_summary.py`
times the public graph plan and optionally a frozen direct FP32 kernel from
`a136913d`. The frozen source SHA256 is
`f391eab1e6fb689e66dc52daddf73b08da911e1312986b16dd849b293957e332`.
All 14 measured forward/reverse cases compare raw output bytes before timing.
Inputs: CPU seed 713, H=8, K=V=128, normalized BF16 Q/K, BF16 V/dO,
FP32 g=-0.001 and beta=0.5. Times are median of five groups of 50 CUDA
Graph replays, in microseconds. Compile and allocation precede capture.
These are summary-only GPU times, not KDA apply, communication or training E2E.

| THD document lengths | Direct fwd | Staged fwd | Direct reverse | Staged reverse |
| --- | ---: | ---: | ---: | ---: |
| 64 | 80.90 | 38.32 | 92.32 | 47.65 |
| 256 | 314.07 | 143.22 | 358.97 | 181.17 |
| 1024 | 1246.39 | 563.54 | 1423.09 | 714.73 |
| 4096 | 6787.15 | 2393.22 | 7920.86 | 3303.29 |
| 16384 | 29668.12 | 10000.92 | 31626.50 | 13652.56 |
| 32 × 33 | 915.14 | 521.99 | 1030.05 | 626.81 |
| 20 × 17 + 1024 + 4096 | 7815.47 | 3202.44 | 8586.86 | 4110.96 |

Workspace is **zero bytes** for every case. Unlike the experimental triangular
solve, this staging path does not materialize sequence-sized coefficients or
expand the graph into a host loop over windows. The experimental 1024-token
window solve was faster on some long weak-decay cases but needed 13.31 MiB
at N=1/H=8 and did not yet meet the dynamic packed-layout contract; it is not
part of this PR. CTA sizes 128/256/512/1024 were compared; 512 was retained
for the measured balance across single-head, strong-gate and short-document
cases, without adding a public precision or tuning option.

NCU 2025.1 compared direct and staged forward at T=1024/H=8. Under profiler
replay, duration was 1.93 ms versus 662.66 us; these cold-cache instrumented
times are distinct from the warm graph table. Average warp cycles per issued
instruction fell 15.10→10.91. The original path reported 10.5 cycles of
L1TEX scoreboard waiting per issue (69.7% of its issue interval). Staging
uses 18.50 kB static shared memory per forward CTA and 48 registers/thread,
versus 0 shared bytes and 32 registers/thread. Grid/block geometry changed
512×128→128×512, retaining 65,536 total threads. This supports the shared
reuse rationale; it does not attribute a percentage of speedup to each change.

Fence audit: passed. Tile writers and readers are generic-proxy CUDA threads
in one CTA; the first `__syncthreads()` publishes every initialized slot and
the second protects reuse. The Q predicate is CTA-uniform because no CTA
straddles the H/M boundary. Every row has one writer, full-warp shuffles
exchange registers, empty spans bypass both barriers uniformly, and partial
tiles read only initialized entries. Global input/output ordering follows the
caller stream. No inter-CTA, async-proxy or device-wide fence is needed.

### SM90 KDA FP32 summaries — original pre-staging validation, 2026-11-11

`linear_attention/hopper/test_kda_summary_sm90.py` adds 35 L0 cases for
`kimi_delta_attention_summary` and `kimi_delta_attention_summary_bwd`, pinned
to `kda_summary_hopper_cuda`. BF16 Q/K/V/dO, FP32 gate/beta/state, K=V=128,
equal head counts and INT32/INT64 offsets are covered. Both FP32 outputs
(H/G and transition) are compared with an independent FP64 recurrence;
the reverse summary also matches FP64 autograd. The APIs are non-differentiable
summaries, not implementations of the ordinary dQ/dK/dV/dG/dBeta backward.

Two determinism cases compare all output bytes over four fixed-input calls
with fresh storage and concurrent GPU work, then move/duplicate the same
33-token sequence among empty and 17-token neighbors, including a 23-sequence
packed batch. `batch_invariant=True` is enabled. Inputs are unchanged and
the independent reference passes before recording two new runtime goldens.
No existing golden is replaced. Four CUDA Graph cases cover both directions
and offset dtypes, with two changed-input/changed-offset replays each, padded
outer strides, poisoned outputs, and fresh-address eager rebinding.
Separate tests check 64 logical spans composed in FP64 against a monolithic
FP64 recurrence; this isolates summary error and is **not** a 64-GPU E2E run.
Unsupported-feature and malformed-binding tests reject before launch.

On H20-3e (SM90), Torch 2.9.1+cu128, CUDA 12.8, driver 12080 and cuDNN 91002,
the full suite passed **35 tests, no skips**, both during separate golden
recording (2.58 s) and final strict checking with recording unset (2.51 s).
The installed DSL was 4.8.1, but this engine compiles CUDA C++ with NVRTC;
it does not use CuTeDSL. Imports and the freshly built extension resolved
to the validation checkout based on `develop@3ac68cf8f`.
Ten existing Hopper CUDA forward/default-routing/serving-dtype controls
also passed (1.27 s). Other GPU architectures and CUDA 13 are unverified
for this addition; existing FROST support is unchanged.

Compute Sanitizer memcheck passed 15 reference/empty/capture cases (9.22 s),
with `ERROR SUMMARY: 0 errors`. This run used
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`: the first instrumented
attempt encountered `CUDA_ERROR_NOT_PERMITTED` in Torch's expandable
allocator `cuMemCreate`, before the test kernels. The instrumented process
also reports nanobind shutdown reference warnings; an unchanged ordinary
Hopper forward control reproduces those warnings and passes with zero
memcheck errors. No memory-leak fix is claimed here.

From `test/python`, after preparing the two runtime goldens separately:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest linear_attention/hopper/test_kda_summary_sm90.py -m L0 -q -s
python -m pytest linear_attention/hopper/test_kda_sm90_cuda.py -m L0 -q -s \
  -k 'is_the_default_on_hopper or serving_dtypes or absent_initial_state_is_zeroed'
```

The new V-independence regression was observed RED on the unmodified base:
no SM90 summary plan was available. During development, direct FP32 `exp(g)`
factors still failed the unchanged 64-span FP64 tolerance under weak decay;
using `expm1(g)` with FMA in that regime made both directions pass.

For context, the following measurements use `make_case((T,), heads=8)`
(seed 713, normalized BF16 Q/K, BF16 V, g=-0.001, beta=0.5), the suite's
FP64 reference, and median CUDA-event time over five groups of 50 CUDA Graph
replays. The probe is two ordinary `kda_hopper_cuda` forwards, one with zero
state and one with identity, followed by state subtraction. Summary timing
includes H and M, not communication or the full KDA layer.

| Tokens | Probe M relative L2 error | FP32 summary M relative L2 error | Probe time (µs) | FP32 summary time (µs) |
| --- | --- | --- | --- | --- |
| 64 | 1.1082e-2 | 2.3770e-7 | 41.58 | 80.84 |
| 256 | 3.9914e-2 | 5.5151e-7 | 130.62 | 314.68 |
| 1024 | 8.6890e-1 | 1.1520e-6 | 608.54 | 1247.20 |

This is a precision/coverage addition with a measured latency cost, not a
performance improvement to the ordinary tensor-core apply kernels.

### Shared suite inventory

These suites check six attention families and the DSA indexer APIs. Run from `test/python`:

```bash
pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0
pytest compressed_sparse_attention/cutedsl/test_CSA_compressor.py -m L0 \
  -k 'batch_invariance or replay_determinism or determinism_golden'
pytest core/test_determinism_utils.py
```

| Suite | API | Forward results | Backward results |
| --- | --- | --- | --- |
| `TestKDA` | `kimi_delta_attention` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDP` | `gated_delta_product`, 1/2/3 Householder steps | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN` | `gated_delta_net` | output, final state | dQ, dK, dV, dG, dBeta, dInitialState |
| `TestGDN2` | `gated_delta_net_v2` | output, final state | dQ, dK, dV, dG, dBeta, dW, dInitialState |
| `TestQSA` | `qsa_forward`, `qsa_backward` | BF16 output, FP32 output, LSE | dQ, dK, dV |
| `TestQSABackwardEngine` | `qsa_backward`, `qsa_indexed_backward` | — | dQ, dK, dV |
| CSA compressor focused cases | `CSA.csa_compressor_forward_wrapper`, `CSA.csa_compressor_backward_wrapper` | BF16 pooled output, including exact-zero capacity rows | BF16 dKV/dScore, FP32 dAPE |
| `TestDSA` | `sparse_attention_forward_wrapper`, `sparse_attention_backward_wrapper` | output, max logits, LSE, optional indexer LSE | dQ, dKV, dSink |
| `TestDSADeterministicDKV` | `sparse_attention_backward_wrapper` dKV layouts (`deterministic_dkv_mode`, SM100/SM103) | output, max logits, LSE (H128 D576: reference inputs) | dQ, dKV, dSink |
| `TestDSAIndexerForward` | `indexer_forward_wrapper` | dense scores | — |
| `TestDSAIndexerBackward` | `indexer_backward_wrapper`, `dense_indexer_backward_wrapper` | — | dIndexerQ, dWeights, dIndexerK, mutated score buffers |
| `TestDSAIndexerFP8` | `indexer_forward_wrapper` | FP32 scores; SM90 FP32 LSE | Unsupported for FP8/MXFP8 inputs |
| `moe/frost/test_shared_expert_decode_determinism.py` | `SharedExpertDecode` (the shared-expert decode kernel pair, `Geometry(HIDDEN, INTERMEDIATE)`) | BF16 output `y`, the inter-launch BF16 `act` and FP32 `gate_part` scratch rows | — (inference-only step kernels; no backward) |
| `attention_gate/frost/test_gate_quant_decode_determinism.py` | `AttentionGateQuantDecode` (the attention gate + quantize decode step, rows 1 .. 160) | FP8 E4M3 output `out` | — (inference-only step kernel; no backward) |
| `gemm/frost/projection_decode/test_fp8_projection_decode_determinism.py` | `Fp8ProjectionDecode` (the FP8 split-K projection for decode rows, `ProjectionGeometry(OUT_FEATURES, IN_FEATURES)` at a fixed `split_k`) | BF16 output `out` (the entry's only result) | — (inference-only step kernel; no backward) |
| `test_gdn_decode_fused.py` / `test_gdn_verify_fused.py` (the `rank_geometry` cells) | `GdnDecodeFused`, `GdnVerifyFused` (the fused GDN decode / MTP-verify step kernels, per-rank `Geometry`) | FP8 output, BF16 output, SSM state slots, conv-state pool, reader tickets | — (inference-only step kernels; no backward) |
| `test_custom_reduction_determinism.py` | `pygraph.custom_reduction`: finite Torch-rule argmax/argmin and direct ADD body | INT32 index projection for FP16/BF16/FP32 values; FP32 sum output | — (no custom-reduction backward entry point) |

### SM100 D128 per-tensor FP8 decode — qualification, 2026-11-12 (UTC)


Review follow-up on `6536a9d4226ced931c8c41886e72c6c2ac9cd5cc`, B200 with
that same runtime: score-slot credit now consumes all eight TMEM score words
through the release-zero dependency; the existing wait/fence and arrival counts
stay in place. Capture cleanup covers errors, and sync-debug mode wraps execute
rather than PyTorch capture setup. Four production PTX renderings pin the full
eight-word dependency. The detector's full core suite passes 29 checks.

Jobs 5887863 and 5887949 together pass 30 native functional cases, eight existing
dual-stream cases, four PTX checks, four replay/batch cases, two warm/poisoned
stress cases and six strict **unchanged** pre-review goldens. All three sanitizer
tools pass four cases each with zero errors/hazards, using the same unfiltered
commands below. CUDA API error reporting remains disabled. No numerical
threshold or golden changed for the synchronization fix. New runtime evidence
is SM100-only; the unchanged SM103 route keeps its prior qualification.

The first runner passed 29 core and 38 functional checks but four PTX checks
could not create its nested pytest temporary directory. The second runner
creates the missing parent and runs only those four checks plus the remaining
phases; source bytes are identical. This setup failure remains separate from
the passing results. Its run root is
`/tmp/yanxu-fp8-native-review-b200-v2-5887949`; strict/sanitizer phases set
`CUDNN_DETERMINISM_GOLDENS=<run_root>/source/test/python/determinism_goldens`
and leave recording unset. The earlier source/record history below is retained.

`TestNativeFp8Decode` and `TestNativeFp8DecodeDeterminism` in
`sdpa/frost/test_sdpa_fwd_fp8_sm100.py` exercise the native five-stage KV ring
and eight-live-column swap-AB lowering through the actual architecture CI
entry. It serves dense Q with paged KV, CGA1, per-tensor E4M3/E5M2 and at most
eight packed query rows. FP16/BF16 and FP8 O, optional Amax/Stats, final log2
Stats, independent K/V maps, padded head dimensions, and strided Stats are
covered. There is no separate determinism switch and no new backward path.

The four replay/batch cases fix packing and split count, quantize each payload
once, and preserve scales across isolation, insertion, reordered/duplicated
sequences and B513. Fresh allocations, concurrent GPU work, retained-graph V
sign changes and exact restoration are checked. O/Stats are byte invariant;
whole-batch Amax is checked numerically after rebatching and bitwise on replay.
CPU FP64 numerical thresholds are unchanged: O 0.04 (or three times the FP8
quantization floor), Amax 0.03, Stats 0.0006. Padding canaries and input bytes
are checked independently.

B200 (SM100, 148 SMs), CUDA compiler and loaded NVRTC 13.4, CuTeDSL 4.8,
cuDNN 9.30.0.4, Torch 2.14.1+cu130:

- 30 functional cases pass across the incremental runs: 10 graph cases each
  make six reference/replay checks, plus feature, route, config, optional-output
  and direct-template Stats-base cases. No functional skips.
- Four new replay/batch cases and both affected older D128/CGA1 replay/batch
  cases pass. Two warm-refill cases each use 592 CTAs, five KV tiles and five
  ring refills; 256 warm and 256 poisoned launches have zero mismatches. The
  unsafe-ring control fails 200/200. The warm harness uses actual compact
  output allocations; separate replay tests retain padded/strided outputs.
- Four new golden records and two updated old D128/CGA1 records pass in a fresh
  strict process. The two old records retain identical input/runtime hashes;
  O, Stats and Amax hashes change with the new swap-AB reduction and FP8 P
  conversion. Independent numerical and batch checks pass before these updates.
  No other existing golden is changed.
- Unfiltered CUDA 13.4 memcheck, synccheck and racecheck each pass all four new
  goldens. Memcheck/synccheck report zero errors; racecheck reports zero hazards,
  errors and warnings. CUDA API error reporting is disabled.

Final architecture routing checks pass on both B200 and B300. Each also passes
the two affected D128/CGA1 strict goldens: updated SM100 records and unchanged
SM103 parent records. This verifies the final SM100-only selection after the
candidate's two-GPU qualification.

The earlier warm-stress run stopped at the harness's contiguous-output
precondition after all four replay/batch cases passed. The correction allocates
compact output storage before prepare; it does not copy a kernel result or
change the kernel. A separate old-template control with log2 split-partial
Stats fails both E4M3/E5M2 numerical checks, and the corrected natural-log
partial guard passes both. These expected RED failures are separate from the
passing qualification counts.

SM103 candidate qualification also passed the 30 functional cases, four new
replay/batch cases, two affected older replay/batch cases, both refill stress
cases and six fresh-process goldens. All six candidate input/output hashes
match SM100. However, its measured performance did not justify replacing the
existing SM103 schedule, so the final route selects this body only on SM100.
The SM103 candidate goldens are not committed; SM103's existing records stay
unchanged. No new SM103 sanitizer result is claimed.

Source: base `5c99b2c8b2e05b6b3f414cf9fbeb782954faa3cd` with the native FP8
changes in this PR. The frozen source/library manifests verify the actual
import root and native-extension inputs. The B200 V2 qualification ran from
`/tmp/yanxu-fp8-native-qual-b200-v2-5886831`. Final edits limit FP8 selection to
SM100 and correct comments; the qualified kernel's executable AST is unchanged.
The previous functional/batch checks used identical routed kernel arithmetic;
the only earlier AST correction guards final log2 Stats with `SPLIT_KV == 1`
(the normal adapter already cleared that flag on split partials).

Run from `test/python`, with `CUDNN_DETERMINISM_RECORD` unset:

```bash
python -m pytest sdpa/frost/test_sdpa_fwd_fp8_sm100.py \
  -k 'TestNativeFp8Decode or test_native_fp8_decode_architecture_route' -m L0 -q -rs
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py \
  -k 'd128 and cga1' -m L0 -q -rs
```

The sanitizer commands below use the CUDA 13.4 executable, qualification
Python, and frozen run root named above. There is no kernel-name or launch
filter. Each tool runs a separate process:

```bash
"$sanitizer134" --tool memcheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" memcheck
"$sanitizer134" --tool synccheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" synccheck
"$sanitizer134" --tool racecheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" racecheck
```

The wrapper verifies source/library hashes and runtime, then calls `pytest.main`
with a result-recording plugin and these arguments:

```text
sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestNativeFp8DecodeDeterminism::test_golden
-o addopts= -m L0 -q -rs --tb=short --junitxml=<run_root>/results/<tool>/pytest.xml
```

`CUDNN_DETERMINISM_RECORD` is unset; `CUDNN_DETERMINISM_GOLDENS=<run_root>/goldens`
selects the preceding record phase. The environment sets
`CUDNN_TEST_NO_ISOLATION=1`, `CUDNN_TEST_TIMEOUT=240`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`,
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`, preloads CUDA 13.4 NVRTC, and uses
the cuDNN 9.30.0.4 library directory with isolated source/cache paths.

### D128 paged live-column decode — qualification, 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fwd_decode_live_columns_determinism.py` covers FP16/BF16
O and FP32 Stats for the eight-live-row swap-AB lowering, with fixed packing
and split counts. Cases include Q1/G8, Q2/G4, split1/4 and D80/V120 inside
the D128 envelope. Replay uses fresh buffers and concurrent work; batch
changes include isolation, insertion, reordering, duplication and more than
one device wave. Retained graphs change V, query/KV lengths and independent
K/V page maps, then restore and compare bytes. Numerical checks use CPU FP64.
The refill stress uses four blocks per SM, warm/poisoned runs and the unsafe
ring control. Backward uses other kernels and is outside this suite.

B200 native qualification (CUDA compiler/NVRTC 13.4, CuTeDSL 4.8,
cuDNN 9.30.0.4, Torch 2.14.1+cu130): four replay/batch cases and two warm
stress cases pass. Four new golden records were created after the numerical
and replay checks, then all four pass in a separate strict process. Existing
goldens are unchanged; the new records baseline the swap-AB reduction order.
The stress uses 592 blocks (4 per SM), 256 warm and 256 poisoned launches per
dtype, with no mismatches; the unsafe control fails 200/200. No skipped cases.

Unfiltered Compute Sanitizer memcheck, synccheck and racecheck each pass the
four golden cases. Memcheck/synccheck report 0 errors; racecheck reports 0
hazards, errors or warnings. CUDA API error reporting is disabled. Under
instrumentation FE reports cuDNN 93000 and Torch 92400; the explicit standalone
FROST path executes no backend attention. No new SM103 sanitizer run is claimed.
The separate public graph suite passes 28 cases, including legacy D256
wide-tile controls. The source/import and runtime library hashes are verified.

The B200 run used base revision `a736a06c6d8dfde1b4322eee0430f93bae78178c`
with the frozen D128 overlays subsequently committed in `fd911361d886b60a083464b6b3e790ba302c7733`.
The three production modules have identical executable ASTs to PR revision
`73bcd50c6166a574fab360136a76c90f6f3ea012`; the determinism test is byte-identical.
The base revision alone does not contain this lowering. The frozen source
manifest also verifies 134 native-extension build inputs.

The actual sanitizer invocations below ran in separate processes. Here
`sanitizer134` denotes the installed CUDA 13.4 `compute-sanitizer` executable,
`python312` the qualification environment's Python, and `run_root` the frozen
run directory (`/tmp/yanxu-d128-determinism-5882703`). No kernel-name filter,
launch limit, or launch-dump option was used:

```bash
"$sanitizer134" --tool memcheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" memcheck
"$sanitizer134" --tool synccheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" synccheck
"$sanitizer134" --tool racecheck --error-exitcode 91 --report-api-errors no \
  "$python312" -u "$run_root/probe.py" racecheck
```

The wrapper verifies source/library hashes, the import path, GPU and runtime,
then calls `pytest.main` from the frozen `source/test/python` directory with
the following arguments (plus a result-recording plugin):

```text
sdpa/frost/test_sdpa_fwd_decode_live_columns_determinism.py::test_live_columns_golden
-m L0 -q -rs --tb=short --junitxml=<run_root>/results/<tool>/pytest.xml
```

For all three tools, `CUDNN_DETERMINISM_RECORD` was unset and
`CUDNN_DETERMINISM_GOLDENS=<run_root>/goldens` selected the four records from
the preceding record phase. The wrapper used `CUDNN_TEST_NO_ISOLATION=1`,
`CUDNN_TEST_TIMEOUT=240`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` and
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`, with CUDA 13.4 NVRTC preloaded,
the cuDNN 9.30.0.4 library directory, and isolated source/cache paths.

Native B300 SXM6 AC (cc10.3, 148 SMs, same compiler/runtime versions) passes 40
checks through `TestDecodeLiveColumns` in the actual SM100/SM103 CI entry,
including the older sink, routing, wide-tile and Int64 page-table-stride cases.
Its four new golden records pass a separate strict process. Both dtype stress
cases report 0 mismatches in 256 warm and 256 poisoned launches at 592 CTAs; the
unsafe-ring control fails 200/200. No skips. The B300 job's subsequent
performance phase failed during FlashInfer import before any measurements;
that harness failure does not change these completed qualification results.

Run from `test/python` with `CUDNN_DETERMINISM_RECORD` unset:

```bash
python -m pytest -m L0 -q sdpa/frost/test_sdpa_fwd_decode_live_columns_determinism.py
python -m pytest -m L0 -q sdpa/frost/test_sdpa_fwd_decode_live_columns_sm100.py
```


### Native SDPA service register release — validation, 2026-11-12 (UTC)

Twenty-six SM100/SM107 forward templates now release the unchanged service
register budget at one common warpgroup instruction before role dispatch.
`TestServiceRegisterRelease` in the SM100 DSL and SM107 FP8 CI entry files
compiles every changed template and checks the actual PTX, including both
entries of D512 MXFP8 and the separate correction-group release. Three old
template controls fail this detector; all 26 candidate templates pass without
skips. SM107 results in this compile run are cross-compilation only.

B200 (SM100, 148 SMs), Torch 2.14.1+cu130, cuDNN 9.30.0.4 and CuTe DSL 4.8.0
using actual CUDA 13.4 and NVRTC 13.4: 127 native cases pass. They cover 72
dense half/FP8/MXFP8 physical-stride permutations across D128/D192/D256/D512,
24 half storage/rebind cases, four D192 cases, five D512 half datapath cases,
one FP8 D512 multi-tile case, two MXFP8 D512 multi-wave cases, and 19 existing
paged FP8 determinism cases. The latter include nine strict unchanged-parent
goldens, nine repeatability/batch-invariance cases and one warm case. No
tracked golden or numerical tolerance changes. The warm case has only four
CTAs and is not a several-waves stress qualification.

Racecheck, memcheck and synccheck each pass 12 dense stride/replay cases
(three input families and four widths), with zero errors or hazards. Only
racecheck filters instrumentation to `kns=cudnn_kernel_`; memcheck and
synccheck are unfiltered. CUDA API error reporting is disabled, so these
results qualify the selected memory/synchronization tools only. Earlier
private harness failures are retained: simultaneous golden record/check
settings, counting two L1 cases before pytest marker deselection, and a
rejected sanitizer filter argument. None changed production bytes or relaxed
an assertion; completed numerical cases were retained, and only missing
phases were rerun. Native SM103 and performance are not measured here.
This forward-only change makes no new backward determinism claim.

Native SM107, Torch 2.14.0+cu130, cuDNN 9.30.0.4 and CuTe DSL 4.8.0 with
CUDA 13.4/NVRTC 13.4: 104 candidate cases pass without skips or failures.
They reuse the 72 dense stride permutations, 12 split1 half storage/rebind
cases, four D512 FP8 strict goldens and four repeatability/batch-invariance
cases, eight D576/V512 half nonpaged cases and four half determinism/golden
cases. Eight parent goldens are recorded in a separate process and checked
without recording by the candidate. The common half fixture's twelve
pre-existing SM107 split exclusions are deselected before execution; this
run does not qualify those split configurations. Native SM107 sanitizers
remain unmeasured. The initial private runner used a CPython 3.12 extension
in a 3.10 container and failed before GPU tests; the corrected run uses a
matching extension whose native build inputs equal the frozen source.

From `test/python`, the generated-instruction detector is:

```bash
python -m pytest -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestServiceRegisterRelease \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestServiceRegisterRelease -q -rs
```

The existing strict paged suite uses the normal golden workflow below;
record its nine golden cases from the unchanged parent into a separate
directory, then run the candidate with recording disabled. Dense sanitizer
selection is `test_sdpa_fwd_dsl_sm100.py::TestNativeDenseStrides -m L0 -k order0`.
The additional multi-wave selection is
`test_sdpa_fwd_mxfp8_sm100.py::test_mxfp8_d512_dense_multi_wave_alias -m L1`.
The frozen validation base is `7c6ea67c633a8509ba976cbf3b0bb0c32473b606`;
source and library hashes and the actual import path were checked.

### SM120 cooperative paged prefill — validation, 2026-11-12 (UTC)

The paged half-MMA prefill loader partitions the same vector copies across
all four existing service warps. Half K/V retains per-thread async-copy
completion before publication; FP8 K/V retains ordinary release publication
after widening stores. Full-barrier arrivals and named-barrier participation
include all 128 producer threads. CTA size, register budgets, shared-memory
layout and attention arithmetic are unchanged. Dense, decode and wide-head
producer counts retain their existing values.

The initial candidate on parent `b2c5270751eeab2213d85eb3103e0ef507937901`
passed 78 existing cases with zero failures/skips, including eight unchanged
strict parent O/Stats goldens, explicit-plan rebind/capture, reordered and
duplicated batches, changed neighbors, width/mixed-KV controls and half refill
stress. Racecheck, memcheck and synccheck each passed three selected cases
with zero errors: one affected small-page half prefill case and two unchanged
decode-ring controls. These runs used Compute Sanitizer 2026.4 (CUDA 13.5),
with API error reporting disabled; the zero-error result applies to the
selected memory/synchronization tools.

`test_dsl_sm120_paged_vector_loop_warm` additionally covers FP16/E4M3 and
BF16/E5M2 K/V, retaining both half cases. The M128/N128 shape has 512 live
CTAs plus 512 empty-sequence controls, rather than 1,024 active CTAs. Each
live CTA reads three KV tiles and refills twice. Independent numerical checks
precede byte stress. The initial half checks pass 256 warm and 256 poisoned
launches per dtype, with zero differing output-storage bytes; the unsafe
control fails 200/200 launches. This corrects the interpretation of the
total launch grid in the earlier records without changing those measurements.

The supplemental run passes all four half/FP8 warm cases with the same
256 warm and 256 poisoned launches and the 200/200 unsafe control, plus the
two FP8 small-page strict-golden cases under each sanitizer, with zero
errors. Together the runs cover 80 distinct cases and five cases per
sanitizer; the repeated half controls are not counted as new coverage.
An initial supplemental allocation exited during GPU inventory before
testing. The successful replacement preserves the one-GPU/architecture/idle
checks. Generated PTX confirms that service warps 8 through 11 reach one
common register-release instruction, with 128 full-barrier arrivals and
384 participants in the tested M128 tile's K/V reuse barriers.

Measured on RTX PRO 6000 Blackwell Server Edition, 188 SMs, Torch
2.14.1+cu130, cuDNN 9.30.0.4, CuTe DSL 4.8.0 with actual CUDA 13.4 and
NVRTC 13.4. Source/import locations and runtime library hashes were checked.
No golden, tolerance or skip changed. This is forward-only, fixed-runtime
qualification; SM121 and cross-runtime identity remain unverified.

From `test/python`, with recording disabled:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k 'small_page_cache_determinism or paged_vector_loop' -q -rs
```

### SM120 warp-cached page tables — validation, 2026-11-12 (UTC)

The common paged loader now loads a tile's live page IDs across a producer
warp and selects them with shuffles before the vector live predicate. Tiles
whose worst-case page span exceeds 32 retain the scalar lookup. The rolled
copy loop, payload operations, physical Int64 offsets, waits, arrivals and
arithmetic order are unchanged. This changes only forward kernels.

`test_dsl_sm120_small_page_cache_determinism` adds four L0 cases to the
selected architecture entry: FP16/BF16 Q/O with half K/V or E4M3/E5M2 K/V.
D128 M128/N128 packed plans use eight-token pages and KV lengths
`[257, 129, 0, 247, 1]`: sixteen-page full tiles, one-row tails, empty
neighbors and independent K/V page order. The existing helper checks CPU-built
fixed inputs, O/FP32 Stats bytes, numerical references, source/canary integrity,
fresh buffers, isolated/reordered/duplicated/changed-neighbor batches,
129-sequence scheduling stress and CUDA Graph replay.

The unchanged parent loader recorded eight goldens after reference/replay
checks; all eight passed strict comparison in a fresh candidate process.
Four existing records were byte-identical to the committed files, and only
four new `page8-blocks35` records are added under `c36345d9c3841d69`.
The targeted candidate selection passed **78 cases, zero skips/failures**:
the existing 14 vector-loop cases, four new small-page cases, eight native
rebind cases, 24 width cases, 20 mixed-KV cases, six partial-GQA decode cases
and two decode-ring cases. Memcheck and synccheck each passed three selected
page8/reused-ring cases with zero errors. Racecheck passed the page8 case
unfiltered and both ring cases with `--kernel-name kns=cudnn_kernel_`, with
zero hazards. The original unfiltered attempt exceeded its 32 GiB host-memory
limit; a 128 GiB retry passed page8 but hit the 180-second test watchdog in
the first ring case's Torch numerical comparison. Both ring cases then passed
with FROST-only instrumentation and unchanged numerical references (7.30s
and 4.81s). Those interrupted attempts are not counted as passing runs.
CUDA API error reporting was disabled for these sanitizer runs, so their
zero-error counts describe the selected memory/synchronization tools only.

The two warm cases retain 1,024 CTAs (5.45 per SM), two refills per launch,
256 warm and 256 shared-memory-poisoned launches per dtype with zero differing
output-storage bytes. Their unsafe control failed 200/200 launches. Each
256-launch block bounds per-launch failure below 1.16% at 95% confidence;
this is not the helper's 3,000-launch 0.1% bound.

Measured on RTX PRO 6000 Blackwell Server Edition, 188 SMs, Torch
2.14.1+cu130, cuDNN 9.30.0.4 and CuTe DSL 4.8.0 compiled with CUDA 13.4;
NVRTC 13.4 was loaded explicitly. Actual imports, source and library hashes
were checked. The frozen parent SM120 runtime source matches develop
`9be74f4d7f748ebbf14febe6a9c624f61e9355c4`. SM121 is unverified; this makes
no cross-runtime or backward determinism claim.

From `test/python`, with recording disabled:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k 'small_page_cache_determinism or paged_vector_loop' -q -rs
```

### SM120 paged vector-copy loop — validation, 2026-11-12 (UTC)

The selected architecture entry `sdpa/frost/test_sdpa_fwd_dsl_sm120.py`
adds 14 L0 cases under `paged_vector_loop`: eight explicit-plan rebind/capture
cases for half D128 prefill, D512, D128 decode and half-Q/FP8-KV prefill; four
D128 M128/N128 repeatability/batch-invariance cases for FP16/BF16 with packed
or unpacked GQA; and two sustained refill checks. The production change rolls
the page-copy vector loop without changing its register budget, synchronization
or arithmetic. It changes no backward path.

The four byte cases preserve O and FP32 Stats across replay, fresh buffers,
isolated/reordered/duplicated batches, 129-sequence scheduling stress and
changed neighbors under capture. Inputs are built on CPU and reused across
batch variants; numerical reference and canary checks are separate. Four
new runtime records are committed under `c36345d9c3841d69`; no existing golden
is replaced. Recording followed successful reference and byte checks, then a
fresh process passed all four with recording disabled. Parent/candidate A/B
measurements separately preserve identical FROST output bytes on all three
measured BF16 shapes; no numerical change is intended.

RTX PRO 6000 Blackwell Server Edition (SM120, 188 SMs), Torch 2.14.1+cu130,
cuDNN 9.30.0.4, CuTe DSL 4.8.0 using CUDA 13.4 and NVRTC 13.4:
**14 passed, 364 deselected, one warning in 72.47s**; fresh strict goldens:
**4 passed, 374 deselected, one warning in 24.26s**. The warning is Torch's
prototype synchronization-debug mode. The frozen source derives from
`b1034fa93283627d0619685bf7ef93fdaa0d09cd`; source/test hashes and the actual
import location were checked. SM121 is unverified. GPU-exclusive scheduling
markers were added after this serial qualification; test bodies are unchanged.

The warm cases use 1,024 CTAs (5.45 per SM), three KV tiles and two refills per
launch. Each dtype passed 256 warm and 256 shared-memory-poisoned launches
with zero differing output-storage bytes. The unsafe control failed 200/200
launches on this GPU. Each block bounds a per-launch failure rate below 1.16%
at 95% confidence; this is not the helper's 3,000-launch 0.1% bound. The stress
helper compares a cold byte reference, while an independent numerical check
runs before stress. This addition has no new sanitizer measurement.

From `test/python`, with recording disabled:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k paged_vector_loop -q -x
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  -k paged_vector_loop_determinism -q -x
```

### SM120 wide partial PackGQA — validation, 2026-11-11 (UTC)

The new L0 cases qualify explicit native forward plans for D512/V512 and
D576/V512 with GQA ratios 96 and 128, two KV heads, and FP16/BF16 queries.
There are 16 dense and 16 paged half cases across split counts 1 and 3,
plus 16 paged mixed-KV cases across E4M3/E5M2 storage at split count 3.
They preserve O and FP32 Stats bytes across repeat, isolated/reordered/
duplicated/changed-neighbor batches and CUDA Graph replay. The stress batch
has 33 sequences; numerical references, empty/tail/keyless rows, source
immutability and output canaries remain independent checks. KV quantization
is fixed across batch variants. This is forward-only coverage; no backward
API or arithmetic is changed.

Measured on RTX PRO 6000 Blackwell Server Edition (SM120, 188 SMs), driver
595.84.01, Torch 2.14.0+cu130, cuDNN backend 9.24.0.43, CuTe DSL 4.8.0 with
its CUDA 13.4 compiler, and the original image's cuda-bindings 13.0.3.
The frozen source derives from `3c6a485cd3b9`; all 2,547 source/binary files
per arm were hash-checked and the imported `cudnn` path was verified before
each run. The compatible extension's binding/build inputs are unchanged.
SM121 and other runtimes are unverified.

The immutable parent recorded 14 existing wide dense/THD/paged controls;
the candidate passed all 14 against those unchanged hashes. The 48 new
subgroup cases passed numerical/batch/replay checks before recording, then
all 48 passed a separate strict comparison with recording disabled. These
62 records are added under runtime fingerprint `812f98809d9324c5`; no existing
record is replaced. An additional 18 new functional cases and one admission
case pass. Memcheck, racecheck and synccheck each pass the same 24-case subset:
all 18 functional cases plus six representative D576 half/mixed subgroup
cases. Memcheck reports zero errors, racecheck zero errors/warnings, and
synccheck zero errors. Racecheck/synccheck filter to `cudnn_kernel_` symbols;
memcheck is unfiltered and CUDA API errors remain enabled. All allocations
were released.

From `test/python`, the strict new-golden selection is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -v -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm120.py \
  sdpa/frost/test_sdpa_fwd_d576_sm120.py sdpa/frost/test_sdpa_fwd_paged_sm120.py \
  -k wide_subgroup
```

The old controls select `test_d576_dense_thd_determinism` and
`test_d576_paged_determinism` in `test_sdpa_fwd_d576_sm120.py`. The functional
selection is `wide_partial_pack or pack_gqa_admits_any_ratio_up_to_the_tile`.
The sanitizer selection on the same three files is
`wide_partial_pack or (wide_subgroup and 3-96-dims1-dtype1) or (wide_subgroup_mixed and 96-dims1)`.
Host graph-analysis checks pass 629 cases, and four variants compile offline;
neither substitutes for the target results above.

Earlier attempts are retained as failures: the first runner stopped before
collection because xdist was absent; the second recorded 18 failures before
interruption (10 ordinary-generator division-by-zero errors, eight reference
cache permission errors) and 15 passes. After setting a writable Triton cache
and using explicit legal plans, the remaining 53 cases passed, followed by
the strict 48 and the three sanitizer passes. The ordinary candidate-generator
failure is a separate integration blocker; these explicit-plan results do
not claim it is fixed. Existing measurements below are retained, not rerun
as part of this addition.

### Prepared SM90 Indexer Forward + Top-K plan — validation, 2026-11-10

`test_DSA_indexer_compressed_sm90_determinism.py` now covers the public
`DSA.prepare_indexer_forward_top_k` / `IndexerForwardTopKPlan.execute` path for
BF16 and per-token E4M3 FP8. The direct plan cases compare indices, logits,
selected-logit softmax and LSE as raw bytes across fixed-input replays with
queued side-stream work, and compare each preserved sequence alone, packed
with empty/ragged neighbors, reordered and duplicated. They also run the
existing independent FP32 reference. Exact ties, unsupported specializations,
zero-candidate rows and caller-buffer sentinels are covered by adjacent L0
cases. The CUDA Graph case mutates packed prefixes, causal offsets and
candidate offsets at stable addresses, poisons outputs and workspace, and
compares repeated replay bytes with an independently invoked eager wrapper.

The plan adds no new arithmetic body: it retains the already-qualified SM90
compact scorer and deterministic radix Top-K launchables. Therefore no new
runtime golden was recorded; the direct raw-byte equality to the existing
public wrapper and its independent reference is the numerical A/B arm. This
is forward-only API coverage; there is no prepared backward entry point.
SM100/103 and SM107 are rejected by the plan contract.

Measured on an SM90 validation host with Torch 2.12.1+cu130, CUDA 13.0,
cuDNN backend 92000 and `nvidia-cutlass-dsl[cu13]` 4.8.1 (CUDA 13.4 DSL
compiler). Imports resolved to the synchronized validation checkout. The
focused SM90 run passed all prepared-plan, combined-SM90 and API-export cases;
the post-rebase run passed 23 collected cases with 134 deselected and no
failures in 13.59 s. Three of those pytest cases capture CUDA Graphs. They
contain six explicit `graph.replay()` call sites and issue seven replay
invocations in the focused run because the Megatron-like helper replays both
its initial and changed payloads. Fixed-input replay loops inside the other
cases are kernel invocations, not additional pytest cases. On the separate
SM100 validation host, the complete existing
compressed-indexer L0 file passed 96 cases with 16 marker-deselected cases;
the architecture-admission/API-export focus passed 2 cases. These SM100
results are compatibility evidence for the shared stage-2 host refactor, not
support for the SM90-only plan. The final SM90 command, run from
`test/python`, was:

```bash
CUDNN_TEST_TIMEOUT=1800 CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -q -s \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_compressed_sm90_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_forward.py \
  -k "prepared_plan or sm90_compressed or api_is_split_from_top_k"
```

### Custom reduction — SM100 validation, 2026-11-09 (UTC)

The new `test_custom_reduction_determinism.py` L0 suite has **seven cases**:
argmax and argmin index projections for FP16, BF16 and FP32, plus a direct
FP32 ADD body. These measurements are new; the earlier attention-suite
measurements and 330-case collection record below are retained separately
and were not rerun for this addition.

Each case checks its independent Torch CPU reference, device input bytes,
fresh-buffer replay with queued side-stream work, and a target row alone,
packed among 32 neighbors, reordered, duplicated, and with changed neighbor
data. Each also replays a large batch of `16 * SM_count + 1` rows: **2369 rows**
on the measured 148-SM device, with the target at row 1184. The reduction
extent is 257, exercising the nominal 256-element boundary. These are workload
shapes, not measurements of the backend's tile size or physical CTA waves.
There are 10 graph calls per case (70 per complete suite run), not 70 pytest
cases. Actual kernel overlap is not measured.

The suite orders the producing Torch stream into the actual cuDNN handle
stream, then orders outputs back before comparisons. It preserves the handle
stream and uses no device-wide prelaunch barrier to drain the queued noise.
Snapshots own their storage; result comparisons check names, shapes, dtypes,
finite values, and raw bytes without tolerance. Index inputs remain local to
each row. No packed offsets, upstream gradients, mutable state, or custom-
reduction backward entry point exist in these tested graphs.

The index-projection graphs consume only the INT32 index. The helper's value
stream remains internal and is not claimed as a tested output. The finite
integer-valued inputs include repeated extrema: the target argmax is **5**
and argmin is **7** under the Torch tie rule. The direct ADD body uses the
same bounded data divided by eight, making all partial sums exactly
representable in FP32; the target sum is **-41/8 (-5.125)**. This establishes
no blanket guarantee for arbitrary user callbacks, arbitrary floating-point
sum data, NaN/infinity policies, or unused tuple outputs. Each graph strictly
selects a publicly enumerated native cuDNN backend plan; absence fails rather
than falling back to another provider. No OSS CuTe DSL kernel path is qualified
by this run.

Source qualification: public parent `8202053a0eac9f224a9961c3f3150765ab3be98f`,
frozen test SHA256
`5e4d7d169a55829a7a4ba43e24c0c34c5f9aa991bc72c2b8be3ff930932db644`.
The test was overlaid onto an isolated validation artifact freshly built at
`a49f23d07c068ec8442da6241bbd755c51460624`; the compiled sources are equivalent
to the public parent. Import resolution was asserted against that artifact's
`/artifacts/python/cudnn`, and both the compiled-header and loaded-backend
versions were asserted to be **92900** before execution. This is native
artifact validation with the frozen source test, not a rebuilt module from
the later documentation/golden addition.

Runtime: GPU name reported as `NVIDIA Graphics Device`, SM100 (cc 10.0),
148 SMs; Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA 13.4, driver API 13030,
cuDNN 92900, nvidia-cutlass-dsl 4.7.1 and apache-tvm-ffi 0.1.13.post3.
CuTe DSL is runtime-key metadata here; the selected numerical path is native
cuDNN. The golden runtime key is `5592daabbefc7ee5`.

| Run | Passed | Failed | Errors | Skipped | Deselected | Pytest time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New baseline recording | 7 | 0 | 0 | 0 | 0 | 11.64 s |
| Strict check, goldens mounted read-only and recording unset | 7 | 0 | 0 | 0 | 0 | 11.89 s |
| Negative control: swap valid argmax/argmin factories | 0 | 6 intended reference failures | 0 | 0 | 1 | 2.38 s |
| Negative control: replace direct ADD with legal MUL, identity 1, and a distinct body key | 0 | 1 intended reference failure | 0 | 0 | 6 | 0.81 s |

Both normal runs collected all seven L0 cases, with two Torch JIT deprecation
warnings each. The negative controls reached native execution and failed at
the independent reference assertions, rather than compilation, support checks,
missing goldens, or harness exceptions. The mutations were confined to isolated
validation copies; the frozen source above was unchanged.

Executed pytest commands inside the validation container, from `/tmp/tests`
with the asserted artifact import active:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_TEST_NO_ISOLATION=1
export CUDNN_DETERMINISM_RECORD=/results/goldens
unset CUDNN_DETERMINISM_GOLDENS
python3 -m pytest -m L0 -o addopts='--tb=short --no-header' \
  -o cache_dir=/tmp/pytest_cache --junitxml=/results/result.xml \
  test_custom_reduction_determinism.py

# Separate strict run; /goldens is the reviewed baseline, mounted read-only.
unset CUDNN_DETERMINISM_RECORD
export CUDNN_DETERMINISM_GOLDENS=/goldens
python3 -m pytest -m L0 -o addopts='--tb=short --no-header' \
  -o cache_dir=/tmp/pytest_cache --junitxml=/results/result.xml \
  test_custom_reduction_determinism.py
```

The swapped-factory control used the same strict environment and pytest command
with `-p det_mutation -k arg_index`; its plugin interchanged the two public
modifier factories. The legal-MUL control used `-k direct_add` after replacing
only the copied body's ADD with MUL, its identity with 1, and its body key with
`test.python.custom_reduction.determinism.direct_multiply.v1`.

Seven JSON files under `5592daabbefc7ee5/` are new baselines for this previously
uncovered numerical API; no prior golden was replaced and no numerical change
is intended. Recording followed replay and references. Independent integer
arithmetic and IEEE byte encoding reconstructed all input, iota, and expected
output hashes (5, 7, and -41/8), and the seven complete file SHA256 values
remained unchanged after the strict run and when copied into this directory.
A normal checkout check uses:

```bash
unset CUDNN_DETERMINISM_RECORD
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest test_custom_reduction_determinism.py -m L0 -q -rs
```

The two custom-reduction support rows now cover the measured SM100
configurations only. SM103 remains explicitly unverified in the existing
combined SM100/103 column; SM90, SM107, other hardware, other data profiles,
and arbitrary reducers remain unverified.

The four linear-attention suites live in `linear_attention/frost/test_*_determinism.py`.
They pin FROST and set `batch_invariant=True`. They cover FP16 and BF16 with
nonzero initial state and fixed output/final-state gradients. QSA lives in
`QSA/cutedsl/test_qsa_determinism.py` and enables `deterministic=True` on
backward; the forward has no switch.
It covers BF16 with 12:1, 24:2, and 6:1 query/KV head counts.
DSA uses FP16/BF16 and enables `deterministic=True` on backward;
`TestDSADeterministicDKV` also pins each SM100/SM103 dKV layout.

CSA compressor coverage currently lives in
`compressed_sparse_attention/cutedsl/test_CSA_compressor.py`. It checks repeated
forward/dKV/dScore bytes, `deterministic=True` dAPE, strict torch deterministic mode,
poisoned caller workspace/output CUDA Graph replay, ragged and empty segments, and
exact-zero static-capacity rows for the generic and ratio-128 forward families. The
focused determinism set has 29 L0 cases: five replay cases; eighteen forward/deterministic-
backward batch cases spanning ratios 2, 4, and 128, both `coff` values, scalar and
wide generic layouts, and both supported ratio-128 head dimensions; and six
replay-plus-reference goldens for generic ratio-2, generic ratio-4 overlap, and the
dedicated ratio-128 path. Batch cases compare a fixed sequence alone, before and after
different-size neighbors, duplicated, and beside changed neighbor data. The ratio-128
case crosses its small/default forward schedule boundary. The ratio-128 `coff=2`,
head-dimension-512 forward boundary remains a non-strict xfail with its raw-byte
assertion intact because CI observed a batch-size-dependent byte; all other cases remain
ordinary assertions. Backward isolates the fixed sequence's upstream gradient and checks
BF16 dKV/dScore plus FP32 dAPE bytes.

At revision `dcb7bddaa358`, the new SM100 run (CUDA 13.0, cuDNN 9.24, PyTorch
2.14.1, CUTLASS DSL 4.8.1) reported 18/18 batch cases, 5/5 replay cases, and 6/6
golden/reference cases passed. The SM103 run (CUDA 13.0, cuDNN 9.20, PyTorch
2.12.1, CUTLASS DSL 4.7.1) reported the same 29 passes. The two committed runtime
baselines use keys `842e0ee8788d6de6` and `4e05d04e8e72ed1b`; identical inputs
produced identical output hashes across the two recorded runtimes. Current-head CI pipeline
80937667 later observed one of 1024 BF16 output bytes differ in
`r128-c2-d512-small-to-default-fprop` when the target moved from the standalone small
schedule into a default-schedule batch. A fresh local rerun passed, so the failure is
environment-sensitive but still disproves the broader forward batch-invariance claim; the
raw-byte assertion is retained as a non-strict xfail. Forward is Replay only in the combined
SM100/103 column. All nine deterministic-backward batch cases passed on both recorded
runtimes, so opt-in deterministic backward remains Supported. The earlier capacity-zero and
poisoned replay cases passed on SM90, but SM90 batch invariance and goldens remain
unverified; SM107 is also unverified. The default atomic dAPE path remains Not guaranteed
by contract.

At revision `bca6dcd50b81`, the requested `@cudnn-ci-bot run python/misc` pipeline
reported 147 passed / 71 skipped / 0 failed CSA cases on SM90, 218 passed / 0 skipped /
0 failed on SM100, and 218 passed / 0 skipped / 0 failed on SM107. The separate SM100
ratio-128 contract gate passed all 21 cases and every shipped forward/backward schedule.

Replay checks use fresh inputs and side-stream contention. Batch
checks reuse exact sequence data and upstream gradients while removing,
reordering, and inserting neighbors. They include nonaligned lengths, partial
tiles, and empty linear-attention sequences. QSA retains each sequence's mask
while shifting its packed token and block offsets. Stress cases cross the
8192-token linear-attention span boundary and QSA's dense/sparse forward
boundary, or create more sequence/head work items than GPU SMs.

Comparisons check tensor names, shapes, dtypes, and raw bytes. Signed zero
differences fail. Non-finite results fail unless the operation declares an
infinity sentinel: QSA uses negative-infinity LSE; DSA uses positive-infinity
LSE and negative-infinity max logits for empty rows. DSA also checks the exact
empty-row locations. Snapshots own their storage. Helper controls reject
changed signed zeros, NaN payloads, missing gradients, wrong infinity signs,
and golden mutations.

## D512 FP8 paged forward — SM100 validation, 2026-11-09 (UTC)

This addition covers `pygraph.sdpa_fp8` through the FROST SM100/SM103 D512
forward body, with dense Q and native paged K/V. It adds no backward path.
Numerics are unchanged from the existing D512 FP8 arithmetic; the new
eligibility and page addressing had no earlier golden. No existing golden
was replaced.

The eight L0 determinism cases in
`sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py` comprise four replay/batch
cases and four corresponding goldens: E4M3→BF16, E4M3→E4M3, E5M2→FP16 and
E5M2→E5M2, with split counts 1/4, NHD/HND pools and PackGQA. Each target's
quantized Q/K/V and FP32 scale bytes stay fixed across batches of 1, 3, 5
and 129 requests, reordered/duplicated targets and changed neighbors. Every
call binds fresh buffers and workspace. Repeated calls include queued work
on a separate stream; actual overlap is not measured. O and Stats are compared
per sequence; Amax is compared on repeated identical batches and checked
against the whole-batch numerical reference otherwise. Input bytes must remain
unchanged. The independent reference dequantizes the operands and computes
attention in Torch; it is separate from raw-byte comparisons.

The adjacent paging tests cover page sizes 8/16/32/64/128/256, query lengths
1/2/16/257, empty and partial KV pages, Q trimming, causal/window masks,
sinks with and without split, separate K/V tables and independently selected
pool layouts, D384/V320 head padding, runtime stride rebinding and changed
page/scalar CUDA Graph replay. Dense/THD controls exercise the existing
unpaged D512 implementation.

Measured on an SM100 development GPU with 148 SMs, driver API 13020,
Torch 2.14.1+cu130, cuDNN 9.28.0.38, CuTe DSL 4.8.0 (CUDA 12.9 compiler),
TVM FFI 0.1.14.post1, Python 3.10.12, and FE `868b3ca` plus this patch.
Imports resolved to the isolated checkout's `python/cudnn`. SM103 was
unavailable and remains unverified. SM90/SM107 do not admit this D512 FP8
paged path. Source qualification does not establish cross-GPU byte equality.

| Run | Passed | Failed | Skipped |
| --- | --- | --- | --- |
| Prepared D128/D256/D512 replay, four D512 batch cases, admission, comparator controls | 37 | 0 | 0 |
| New goldens recorded into an empty candidate directory after reference/replay | 4 | 0 | 0 |
| Strict golden check plus final page-boundary/envelope cases, recording unset | 16 | 0 | 0 |
| Existing dense/THD D512 capture and graph split controls | 40 | 0 | 0 |
| Compute Sanitizer memcheck, final page-boundary/envelope cases | 8 | 0 | 0 |
| Compute Sanitizer synccheck, same cases | 8 | 0 | 0 |
| Compute Sanitizer racecheck, page8 single-Q and page128 persistent/window cases | 2 | 0 | 0 |

These rows describe separate runs; shared cases are not additional coverage.
The first D512 prepared paging case fails on the unchanged parent because
there is no FROST candidate. During test development, shrinking a declared
split-plan batch was rejected by its existing contract; the batch tests now
build a plan for each batch size. Torch's absent FP8 `isfinite` implementation
also required the shared snapshot helper to promote only its finiteness
predicate, while preserving original FP8 snapshots and byte comparisons.
Two comparator controls cover payload preservation, signed zero and NaN
rejection. Neither issue was waived with a skip or weakened byte comparison.

Run from `test/python`, with `PYTHONPATH` pointing to the checkout:

```bash
python -m pytest sdpa/frost/test_sdpa_prepared_fp8_paged.py::test_prepared_fp8_paged_strides_and_capture \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_d512_paged_replay_and_batch_invariance \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_declines_off_contract \
  core/test_determinism_utils.py -x -q
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_boundaries \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_head_envelope -x -q
python -m pytest sdpa/frost/test_sdpa_prepared_fp8.py::test_prepared_fp8_capture_replay_reads_current_scales \
  sdpa/frost/test_sdpa_fp8_d512_graph_split.py -k 512 -x -q
```

Memcheck and synccheck each report **0 errors**; racecheck reports **0 hazards**
(0 errors, 0 warnings) for its two selected boundary cases. Their process uses isolated
`cuda-bindings==13.2.0`, matching driver API 13020, with all API diagnostics
enabled. An initial memcheck with the environment's 13.4.1 bindings reported
40 API-resolution errors (13.3/13.4 requests to the 13.2 driver), while its
8 numerical tests passed; it was not counted as a clean sanitizer run.
No kernel memory error was reported. Reproduce with the runtime above:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python -m pytest \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_boundaries \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_d512_head_envelope -x -q
# Repeat with --tool synccheck. For racecheck, add -k '8-1-1-True or 128-257'.
```

The paged kernel launch explicitly retains its register split. The emitted
SM100 SASS contains `USETMAXREG.TRY_ALLOC` with 240 registers and
`USETMAXREG.DEALLOC` with 40; the original launch without the occupancy hint
had no register-split instructions. Dense launch geometry is unchanged.

The four new records are under runtime key `ef27de8cf1fd96d2`. Recording used
`CUDNN_DETERMINISM_RECORD` with a new temporary directory and `-k golden`;
the separate strict run used the committed records with recording unset.

## How to read the test cases

A **suite** is one operation's test class. A **case** is one collected pytest
node after parameter expansion. For example,
`TestKDA::test_replay[bf16-bprop]` is one case: BF16 backward, eight executions.
Those eight executions are not eight reported tests. Backward cases also check
the forward outputs and saved state used to compute gradients.

The expanded set has **330 cases**: 324 operation cases and six harness
controls. All 330 cases are L0,
including stress and boundary cases, so the default marker selects the full set.
List the exact node IDs without executing the kernels:

```bash
pytest core/test_determinism_utils.py linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py deepseek_sparse_attention/cutedsl/test_DSA*determinism.py --collect-only -q
```

| Check | What stays fixed | What changes | Pass condition |
| --- | --- | --- | --- |
| Replay | Input bytes, state, masks, upstream gradients | Fresh buffers and concurrent GPU work | Every result byte matches the first execution |
| Batch invariance | Each selected sequence and its upstream gradients | Batch size, neighbors, order, packed offsets | Its results match the corresponding baseline slice |
| Neighbor isolation | Target sequence, mask, state, upstream gradients | Other sequences' data, state, gradients, and QSA masks | Target results do not change |
| Golden regression | Input hashes and complete runtime key | Frontend source revision | All result hashes match the committed baseline |
| Numerical reference | Mathematical inputs and gradient sources | GPU kernel versus independent reference | Existing operation tolerances pass |

The golden cases check replay and the numerical reference before reading the
saved baseline. The new head-layout, GDP step-count, isolated-loss, and QSA
mask-replay cases also check an independent reference. A reference pass alone
does not establish bitwise determinism.

### Linear attention: 28 cases per family, plus four GDP cases

The shared implementation is `linear_attention/determinism.py`. Each FROST
wrapper pins one operation. Except where listed, inputs use BF16, two Q/K/V
heads, K=V=64, and a nonzero FP32 initial state.

| Test method | Level | Parameters and purpose | Cases per family |
| --- | --- | --- | ---: |
| `test_replay` | L0 | FP16/BF16 × fprop/bprop; lengths 65, 0, 17, 1; eight executions; input mutation check | 4 |
| `test_batch_invariance` | L0 | FP16/BF16 × fprop/bprop; removal/reorder; empty state and state-gradient passthrough | 4 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes | 2 |
| `test_long_batch_invariance` | L0 | fprop/bprop; lengths 17001, 17; four executions then standalone comparison | 2 |
| `test_many_work_items_replay` | L0 | fprop/bprop; four sequences × enough heads to exceed the SM count; eight executions | 2 |
| `test_tile_boundaries` | L0 | FP16/BF16 × fprop/bprop; lengths 1, 15, 16, 17, 63, 64, 65 | 4 |
| `test_head_layouts` | L0 | fprop/bprop × three head/dimension layouts below; checks shared-head gradient reductions | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; fixed 17-token target, changed 65- and 3-token neighbors | 2 |
| `test_gradient_source` | L0 | bprop with output loss only or final-state loss only | 2 |
| GDP `test_householder_steps` | L0 | fprop/bprop × 1/2 steps; the other GDP tests use 3 steps | +4 for GDP |

The three head layouts are `(Q/K heads, V heads, K dimension, V dimension)`:
`(1, 4, 64, 128)`, `(4, 1, 128, 64)`, and `(2, 2, 128, 128)`.
Tile, head-layout, isolated-loss, and GDP step-count cases each run four
executions, then compare every sequence alone, the reversed batch, and a
duplicated first sequence. GDP slices K/V/Beta gradients on the expanded
Householder axis. State tensors use `[sequence, output heads, V, K]`.

### Fused GDN step kernels: per-geometry replay and batch invariance, no goldens

The fused decode / MTP-verify step entries (`cudnn.linear_attention.frost.gdn_decode_fused` / `gdn_verify_fused`) carry their determinism
cells inside their own modules (`linear_attention/frost/test_gdn_decode_fused.py`, `test_gdn_verify_fused.py`; the FlashInfer plan-API twin in
`test_flashinfer_gdn_backend.py`), at the default geometry (16 K / 48 V heads, hidden 5120) and at five per-rank geometries ((1, 8, 8192),
(2, 16, 8192), (4, 32, 8192), (4, 16, 4096) and, since 2026-11-06, (2, 8, 4096) -- 8-way tensor parallelism of a hidden-4096 model; the FlashInfer
plan-API twin covers the first four).  Every comparison is raw bytes: the FP8 and BF16 outputs, every FP32 state slot, the conv-state
pool and the reader tickets.

| Check | Cells | What it holds |
|---|---|---|
| Replay | `*_matches_the_torch_model_and_replays_bitwise` / `*_matches_the_torch_reference_and_replays_bitwise`, `test_deterministic_*`, the graph-replay cells | Two fresh identical operand sets (and a CUDA-graph replay over distinct layers) are byte-identical; the independent fp64 reference is a separate check at the operations' tolerances |
| Batch invariance | `test_rank_geometry_rows_are_batch_invariant` (decode), `test_rank_geometry_requests_are_batch_invariant` (verify) | A row / request alone writes the bytes it writes inside a 4-row / 4-request step (its outputs, its state slot(s), its conv slot).  The verify step holds this under ONE gated-norm association: the one-request pick takes the reference kernel's two-CTAs-per-head association (`ctas2`, 4-row chains), several requests `ctas1` (8-row chains) -- `norm_form_for(requests)`, a by-design dependence of the normed output's last bits on the request count, so the lone request runs under `ctas1` in the cell |
| Cross-form | `*_knob_arms_are_bitwise_*`, `test_pair_and_one_cta_forms_are_bitwise_at_one_request` | Every form family (cluster pair, register-capped, streamed, thread mappings, issue orders, ring depths) writes the bytes of the one-CTA form |
| Reference-kernel bytes | `test_bitwise_with_the_reference_verify_kernel_outputs[*]` (with `CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR`), the FlashInfer `frost` bitwise cells | The default geometry replays the saved reference-kernel output sets byte for byte; no committed goldens (the saved sets are the oracle) |

GPU coverage: cc 10.0 (a GB200; the per-rank cells 2026-11-05), cc 10.3 (verified on a GB300, 2026-11-05: the same cells at the default and the four per-rank geometries, the saved reference-kernel sets byte for byte, the three register-capped default-geometry forms rendering the same `sm_103a` cubin as before the geometry change) and cc 12.0 (an RTX 5090, 99 KB per-CTA opt-in, 2026-11-05: the default geometry's cells and the per-rank cells -- the replay, batch-invariance and cross-form cells there compare against the landing-slab one-CTA form, the one this part holds; the arms the part does not hold skip by their typed decline); cc 10.7 in the record of 2026-11-10 below.  The fifth geometry's cells, (2, 8, 4096), were verified
on cc 10.3 (a GB300, 152 SMs, 2026-11-06, inside a vLLM container image: torch 2.14.0+cu130, nvidia-cutlass-dsl 4.8.0, cuDNN backend 9.24.0;
`pytest linear_attention/frost/test_gdn_decode_fused.py -k k2v8h4096` 30 passed / 244 deselected, `pytest linear_attention/frost/test_gdn_verify_fused.py
-k k2v8h4096` 37 passed / 306 deselected) and on cc 10.0 (a GB200, 152 SMs, in the repository's CI: torch 2.14.0a0+b2c75dd062.nv26.09, CUDA 13.4,
driver API 13040, cuDNN 92700, nvidia-cutlass-dsl 4.8.0; 16 pytest workers; the CI run of `test/python/linear_attention` on 2026-11-06 (UTC)
at the head of the pull request that added the cells (#2319), and the scheduled run of 2026-11-09 on `develop` -- the CI log prints its selection, not a
pytest command line, so the counts are read from the junit report of each run and are the same in both: the `k2v8h4096` cells of
`test_gdn_decode_fused.py` 30 passed, of `test_gdn_verify_fused.py` 37 passed, none skipped or failed); on cc 12.0 those cells are unverified.

cc 10.7 record (2026-11-10 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API
13050, cuDNN 92700, nvidia-cutlass-dsl 4.8.0, tvm-ffi 0.1.14.post0; one in-process pytest run (`CUDNN_TEST_NO_ISOLATION=1`) on one visible device,
with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`, `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` and `CUDNN_FROST_GDN_VERIFY_REFERENCE_DIR` set to the 16
saved reference-kernel sets, from a clean tree at `868b3cae4` (#2369); no file under `python/cudnn/linear_attention` or
`test/python/linear_attention` differs between that commit and `8e31a5725`, the `develop` head this record was written on.  `cd test/python &&
pytest linear_attention/frost/test_gdn_decode_fused.py linear_attention/frost/test_gdn_verify_fused.py
linear_attention/frost/test_gdn_support_matrix_docs.py -v -ra --durations=15` -> 616 passed, 4 skipped, 0 failed in 590 s:
`test_gdn_decode_fused.py` 273 passed / 1 skipped, `test_gdn_verify_fused.py` 340 passed / 3 skipped, `test_gdn_support_matrix_docs.py` 3 passed.
Each of the five per-rank geometries: 30 passed (decode) and 37 passed (verify), none skipped.  Replay:
`test_rank_geometry_matches_the_torch_model_and_replays_bitwise` 25, `test_rank_geometry_matches_the_torch_reference_and_replays_bitwise` 50,
and at the default geometry `test_deterministic_and_matches_the_torch_model`, `test_graph_replay_over_distinct_layers_matches_eager` and
`test_deterministic_and_graph_replay_over_distinct_layers_matches_eager`, 1 each.  Batch invariance:
`test_rank_geometry_rows_are_batch_invariant` 5, `test_rank_geometry_requests_are_batch_invariant` 5 (the verify cells under one gated-norm
association, as on the other parts).  Cross-form: `test_knob_arms_the_part_holds_are_bitwise_with_the_default_pick_at_b4` 55 and
`test_rank_geometry_knob_arms_are_bitwise_with_the_one_cta_form` 95 (decode); `test_knob_arms_the_part_holds_are_bitwise_with_the_default_pick`
43, `test_rank_geometry_knob_arms_are_bitwise_with_the_one_cta_form` 115 and `test_pair_and_one_cta_forms_are_bitwise_at_one_request` 2
(verify).  Reference-kernel bytes: `test_bitwise_with_the_reference_verify_kernel_outputs` 14 passed, 2 skipped.  The 4 skips:
`test_operands_on_another_cuda_device_are_rejected_before_any_launch` in both modules ("needs two CUDA devices") -- rerun in the same runtime
with two devices visible and no reference directory, `pytest linear_attention/frost/test_gdn_decode_fused.py
linear_attention/frost/test_gdn_verify_fused.py -v -ra -k another_cuda_device` -> 2 passed, 600 deselected; and the saved sets `r16t6_pad` and
`r8t6_padskip` ("the reference kernel declined this set": their saved reference files hold a decline instead of outputs, so the two cells skip
wherever these sets are used; the torch reference is their oracle).  Not run on cc 10.7: the FlashInfer plan-API twin
(`test_flashinfer_gdn_backend.py`).

### Shared-expert decode pair: replay, batch invariance and goldens per geometry and form

`moe/frost/test_shared_expert_decode_determinism.py` pins `cudnn.SharedExpertDecode` (the bf16 shared expert next to a routed MoE,
1 .. 24 tokens, two launches) at both validated geometries (hidden 8192 with intermediate 256 and 128).  Every comparison is raw bytes
over the pair's output `y` and the live rows of its inter-launch scratch (`act` bf16, `gate_part` fp32); the independent fp64 reference of
the bf16 operands is a separate check at the harness band (rel-L2 <= 2e-2; `test_shared_expert_decode.py`).

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | geometry x T 1 / 6 / 24; eight executions on fresh buffers under side-stream contention; reference check | 6 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | two entries, two fresh identical operand sets | 2 |
| `test_batch_invariance` | L0 | geometry x inside T 6 / 24: every token alone (T 1, one m-tile, row 0) vs inside, and inside the reversed batch -- the T 24 cell crosses the two-m-tile form boundary and the fragment row position | 4 |
| `test_duplicated_token_and_neighbour_isolation` | L0 | a token duplicated (both copies = alone), neighbours redrawn around it in the two-tile form, a row-strided x view vs its contiguous copy | 2 |
| `test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified` | L0 | geometry x T 5 / 20; `x` and `out` as column slices of a slab 64 columns wider than the hidden size (a row stride that is not the hidden size): the live rows' bytes equal the contiguous run's, the slab's columns past the hidden size keep their sentinel, every input holds the bytes it held before the launches | 4 |
| `test_twenty_four_to_twenty_five_is_the_typed_decline` | L0 | the T ceiling is a `NotImplementedError`, never a provider switch | 1 |
| `test_golden` | L0 | geometry x form (T 3 / 19); three executions, the reference, the saved hashes (`shared-expert-decode-<geometry>-tiles<n>-ksplit8`: the default form -- eight K slices, x resident by bulk row copies, the gate leaves folded on the cluster leader) | 4 |
| `test_v1_arm_reproduces_the_v1_goldens` | L0 | geometry x form on the SAME operands, the first release's form (`v1_cfg`: K_SPLIT 2 / 4, the x ring, the launch-2 gate fold) against the `ksplit2` / `ksplit4` sets recorded with that release -- the shared code paths moved no first-release addition | 4 |

Golden sets: the `ksplit2` (I 256) / `ksplit4` (I 128) sets were recorded with the first release (2026-11-06, cc 10.3) and are kept as the
v1 arm's; the `ksplit8` sets are the one-wave default's -- a NEW rounding class (reason: the pinned `K_SPLIT` moved from 2 / 4 to 8, the eight
slices are added in cluster-rank order, and the gate logit's leaves are added per row-pair group on the cluster leader before launch 2's
lane fold), recorded on the same part after the replay, batch-invariance and reference cells passed.  The knob arms (`PDL`, `PDL_PAIR`,
`PDL_RELEASE`, `N_GROUPS`, `X_BULK_ROWS`, `X_STAGES`, `W_PHASES`, `ROWS_PER_CTA`, `KRED_SPLIT`) are bitwise with the default in
`test_shared_expert_decode.py` (`test_bit_neutral_knob_arms_write_the_default_bytes`, `test_kred_split_arm_is_bitwise_at_the_one_group_geometry`);
the `K_SPLIT` / `GATE_CLUSTER_FOLD` / v1 rounding-order arms are within the band there and each is pinned bitwise-replay (three executions
under side-stream contention) and batch-invariant (every token of a T 6 step alone vs inside) in
`test_order_arms_are_rounding_order_arms_within_the_band`; goldens exist for the default and the first release's form only; the
graph-capture replay is bitwise with eager there too.

GPU coverage: cc 10.3 and cc 10.7 (replay, batch invariance and goldens) and cc 10.0 (replay and batch invariance; the golden cells are not
covered there).  Run of record on cc 10.3 (2026-11-06): a GB300 (cc 10.3, 152 SMs) with torch 2.14.0+cu130, CUDA 13.0, driver 13030,
cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (the goldens' `runtime` block); `cd test/python && pytest
moe/frost/test_shared_expert_decode_determinism.py` -> 27 passed, 0 skipped, and `pytest moe/frost/test_shared_expert_decode.py` ->
191 passed, both from a clean tree at the commit that carries the `ksplit8` goldens.  Another runtime stack (torch / CUDA / cuDNN / DSL
version) hashes to another goldens directory, so the golden cells SKIP there -- a coverage gap, not a pass -- until a set is recorded for
it with a stated reason.

cc 10.0 record (2026-11-07 UTC, repeated 2026-11-09): a GB200 (cc 10.0, 152 SMs) in the repository's CI, with torch 2.14.0a0+b2c75dd062.nv26.09,
CUDA 13.4, driver API 13040, cuDNN 92700, nvidia-cutlass-dsl 4.8.0, tvm-ffi 0.1.14.post1; 16 pytest workers.  Two runs of
`test/python/moe/frost`: at the head of the pull request that added the suite (#2340), and the scheduled run of 2026-11-09 on `develop` (the CI log
prints its selection, not a pytest command line; the counts are read from the junit report of each run and are the same in both).
`test_shared_expert_decode.py`: 197 passed, 0 skipped.  `test_shared_expert_decode_determinism.py`: 19 passed, 8 skipped.  Passed:
`test_replay` 6, `test_replay_on_a_fresh_entry_and_fresh_operands` 2, `test_batch_invariance` 4,
`test_duplicated_token_and_neighbour_isolation` 2, `test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified` 4,
`test_twenty_four_to_twenty_five_is_the_typed_decline` 1.  Skipped: every case of `test_golden` (4) and of
`test_v1_arm_reproduces_the_v1_goldens` (4), each with "no golden for ..." -- no set is recorded for that runtime, so on cc 10.0 the golden
comparison is a coverage gap, not a pass.  A cc 10.0 part with 148 SMs (the host cells pin the same forms for 148, 152 and 212 SMs) has one
device run (2026-11-10 UTC; x86 host, Python 3.10, torch 2.13.0+cu130, CUDA 13.0, driver API 13010, cuDNN 92700, nvidia-cutlass-dsl 4.8.0,
tvm-ffi 0.1.13.post3; in-process with `CUDNN_TEST_NO_ISOLATION=1` and `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`; on the tree of the pull request
that added the cc 10.7 record below: the sources of `8e31a5725` with that pull request's comment, status-cell pin and documentation edits):
`cd test/python && pytest
moe/frost/test_shared_expert_decode.py moe/frost/test_shared_expert_decode_determinism.py linear_attention/frost/test_gdn_support_matrix_docs.py
-v -ra --durations=10` -> `test_shared_expert_decode.py` 197 passed, 0 skipped; `test_shared_expert_decode_determinism.py` 19 passed, 8
skipped, with the per-method counts of the 152-SM part (the 8 golden cases skip there too: no set for that runtime);
`test_gdn_support_matrix_docs.py` 3 passed.  cc 12.x declined by type.

cc 10.7 record (2026-11-10 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API
13050, cuDNN 92400, nvidia-cutlass-dsl 4.8.0, tvm-ffi 0.1.15.post1 (the `runtime` block of the goldens in `73840df1b366af2c/`); in-process
pytest runs (`CUDNN_TEST_NO_ISOLATION=1`), one fresh process per command, with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` and
`CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1`, from a clean tree at `868b3cae4` (#2369); between that commit and `8e31a5725`, the `develop` head
this record was written on, `python/cudnn/moe` and `test/python/moe` differ only in the comment and the status-cell pin of #2392.  From
`test/python`: `pytest moe/frost/test_shared_expert_decode.py -v -ra --durations=15` -> 197 passed, 0 skipped.  `pytest
moe/frost/test_shared_expert_decode_determinism.py -v -ra --durations=15` against the goldens committed at that commit -> 19 passed, 8 skipped:
`test_replay` 6, `test_replay_on_a_fresh_entry_and_fresh_operands` 2, `test_batch_invariance` 4,
`test_duplicated_token_and_neighbour_isolation` 2, `test_row_strided_views_write_the_contiguous_bytes_and_leave_the_inputs_unmodified` 4 and
`test_twenty_four_to_twenty_five_is_the_typed_decline` 1 passed; every case of `test_golden` (4) and of
`test_v1_arm_reproduces_the_v1_goldens` (4) skipped with "no golden for ..." (no set for this runtime key).  With the replay, batch-invariance
and reference cases passed, the set was recorded into a new directory, `CUDNN_DETERMINISM_RECORD=<new directory> pytest
moe/frost/test_shared_expert_decode_determinism.py -v -ra -k golden` -> 8 passed, 19 deselected, and checked strictly,
`CUDNN_DETERMINISM_GOLDENS=<that directory> pytest moe/frost/test_shared_expert_decode_determinism.py -v -ra` -> 27 passed, 0 skipped.  The 8
files are committed as `73840df1b366af2c/`.  Reason for the new set: a new runtime key (GPU name, compute capability, SM count and driver API
version differ from the cc 10.3 set's), not a numerical change -- the input and output hashes of all 8 records (`y`, `act`, `gate_part`;
`ksplit2` / `ksplit4` / `ksplit8`, one and two m-tiles, both geometries) equal those of the cc 10.3 set `6451c9d94a2d4210/`.  That equality is
an observation on these two runtimes, not a contract across architectures.  A cc 10.7 part that reports another GPU name, SM count or software
version hashes to another key, and the golden cells skip there.

### Attention gate + quantize decode step: replay, batch invariance and goldens

`attention_gate/frost/test_gate_quant_decode_determinism.py` pins `cudnn.AttentionGateQuantDecode` (the sigmoid output gate of an attention
block fused with the static E4M3 quantization of the following projection's input: bf16 `attn` / `gate` `[rows, width]`, one fp32 scale,
E4M3 `out`; rows 1 .. 160, one launch).  Every comparison is raw bytes over the one output `out`; the step has no backward, no state and no
scratch.  The independent reference is a separate check (`gate_quant_utils.assert_reference`: every byte is the code of one of the two bf16
values that bracket the fp64 gated value, and of the nearest one outside a 2.5-fp32-unit margin around the bracket's midpoint; then the
torch model of the rounding chain, bitwise) and gates the goldens.  The kernel has no reduction, so there is no split or association to
hold fixed across batch layouts; the cells build one operand set and slice or permute it.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | rows 1 / 17 / 160 x width 2048 / 6144, and 160 x 32768 (2,560 CTAs of 256 threads: several waves); eight executions into fresh buffers under side-stream contention; reference check | 7 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | width 2048 / 1032; two entries, two fresh identical operand sets | 2 |
| `test_launch_on_a_side_stream_writes_the_same_bytes` | L0 | the same call under a side stream (ordered after the operands' producer stream) vs the current stream | 1 |
| `test_batch_invariance` | L0 | width 2048 / 6144 / 1032 / 1536 x `THREADS` 128 / 256; one 160-row operand set: every row alone (a one-row call) vs inside the 160-row call, vs the first 2 / 3 / 8 / 16 / 17 / 24 / 32 / 64 rows as one call, vs the reversed batch.  At 2048 / 6144 a row is whole CTAs; at 1536 rows share CTAs, at 1032 also warps | 8 |
| `test_insertion_removal_and_reordering` | L0 | width 2048 / 1032 x `THREADS`; a 24-row call, then a row inserted at position 5, rows 3 / 11 / 23 removed, a permutation: every row keeps its bytes | 4 |
| `test_duplicated_row_and_neighbour_isolation` | L0 | width 2048 / 6144 / 1032 / 1536 x `THREADS`; a row twice in one call (both copies = alone); the row among 63 redrawn neighbours | 8 |
| `test_pick_function_moves_no_byte` | L0 | every served thread count vs the default pick; then a pick table that depends on the row count (128 threads up to 8 rows, 256 above): both configs are compiled at plan time, and every row alone writes the bytes it writes inside 2 .. 160 rows on either side of the pick boundary | 1 |
| `test_inputs_are_unmodified` | L0 | `attn`, `gate`, `scale` bytes before and after twelve launches (caller-owned and entry-owned output, the release arm included) | 1 |
| `test_one_sixty_to_one_sixty_one_and_the_empty_call_are_typed_declines` | L0 | 161 rows is a `NotImplementedError`, zero rows a `ValueError`; never a launch, never another provider | 1 |
| `test_golden` | L0 | width 2048 x 3 rows (scale 0.01) and width 6144 x 19 rows (scale 1.0): three executions, the reference, the three knob arms on the same operands, the saved hashes | 2 |

Golden sets: `attention-gate-quant-decode-w2048-rows3` and `attention-gate-quant-decode-w6144-rows19`, the first goldens of this API,
recorded after the replay, batch-invariance and reference cells passed -- in `6451c9d94a2d4210/` for the cc 10.3 runtime below and in
`d2a97fcefba4b867/` for the cc 10.7 runtime below.  The knob arms (`THREADS` 128, `PDL_RELEASE`, both) write the default's bytes inside
`test_golden` and, on more shapes, in `test_gate_quant_decode.py` (`test_bit_neutral_knob_arms_write_the_default_bytes`), so the one set
holds for every served config.  The release arm is compared with no dependent grid on the stream; a chain with a dependent launched under
the programmatic-stream-serialization attribute is not covered here.  The graph-capture replay (bitwise with eager, operands changed in
place) is in `test_gate_quant_decode.py`.  The input and output hashes of the two records are equal between the cc 10.3 and the cc 10.7
set: an observation on these two runtimes, not a contract across architectures.

GPU coverage: cc 10.3 and cc 10.7 (replay, batch invariance and goldens); cc 10.0 has not been run on a device (unverified); cc 9.0 and
cc 12.x are declined by type.  Every run below is an in-process pytest run (`CUDNN_TEST_NO_ISOLATION=1`), one fresh process per command
with empty CuTe DSL and driver compile caches, from `test/python` of a clean tree at the commit that carries the goldens.

cc 10.3 record (2026-11-11 UTC): a GB300 (cc 10.3, 152 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API 13030,
cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi 0.1.15.post1 (the `runtime` block of the goldens in
`6451c9d94a2d4210/`).  `pytest attention_gate/frost/test_gate_quant_decode.py core/frost/test_frost_kernel_name_prefix.py -q -ra` -> 147
passed, 0 skipped.  `CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" pytest attention_gate/frost/test_gate_quant_decode_determinism.py
-q -ra` -> 35 passed, 0 skipped (the case counts of the table).  `pytest -m L2 attention_gate/frost/test_gate_quant_decode_census.py -q -s`
-> 1 passed.  Before the set existed the same determinism command without the goldens variable read 33 passed, 2 skipped (both
`test_golden` cases, "no golden for ..."); the set was then recorded into a new directory, `CUDNN_DETERMINISM_RECORD=<new directory> pytest
attention_gate/frost/test_gate_quant_decode_determinism.py -q -ra -k golden` -> 2 passed, 33 deselected.

CUDA 12.9 compiler flavour on the same part and runtime (2026-11-11 UTC; the CuTe DSL 4.8.0 install without its CUDA 13 toolchain
libraries, so the DSL binds its CUDA 12 flavour: `CUDA_VERSION` 12.9; torch and the driver stay CUDA 13 -- this isolates DSL compatibility
and is not a deployment that uses CUDA 12 throughout): the same three commands -> 146 passed, 1 skipped (`test_rendering_pins_the_convention`
for `sm_107a`: that compiler rejects the target, and the cell skips with the entry's own decline); 35 passed, 0 skipped -- the golden cases
pass against the SAME set, whose runtime key does not name the compiler flavour, so the two flavours write the same bytes for those cases;
1 passed.

CuTe DSL 4.8.1 on the same part (2026-11-11 UTC; the 4.8.1 wheels put first on the path of the same runtime, compiling with CUDA 13.4):
the functional command -> 147 passed, 0 skipped; the determinism command WITHOUT the goldens variable -> 33 passed, 2 skipped (both
`test_golden` cases: the DSL version is part of the runtime key and no set is committed for that key, so the golden comparison is a
coverage gap there, not a pass); the census -> 1 passed.  A record into a new directory under that runtime (2 passed, 33 deselected) holds
the input and output hashes of the committed 4.8.0 set; it is not committed.  With only the CUDA 12 toolchain libraries of 4.8.1
(`CUDA_VERSION` 12.9): 146 passed, 1 skipped (the `sm_107a` rendering cell, as under 4.8.0); 33 passed, 2 skipped (the golden cases, same
reason); 1 passed; its record holds the same hashes (that run's tree differed from the commit in this report's text only).  Cold compile on an x86 host (fake tensors, empty caches): under 4.8.0 and under
4.8.1 the kernel's four arms render byte-identical cubins for `sm_100a`, `sm_103a` and `sm_107a` with the CUDA 13.4 compiler and for
`sm_100a` / `sm_103a` with the CUDA 12.9 compiler; `sm_107a` does not compile under CUDA 12.9 in either version (the entry's typed
decline).  Not run on a device: 4.8.1 on cc 10.7.

cc 10.7 record (2026-11-11 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver
API 13050, cuDNN 92400, nvidia-cutlass-dsl 4.8.0 (compiling with CUDA 13.4), tvm-ffi 0.1.15.post1 (the `runtime` block of the goldens in
`d2a97fcefba4b867/`).  The same three commands -> 147 passed, 0 skipped; 35 passed, 0 skipped; 1 passed.  Before its set existed the
determinism command read 33 passed, 2 skipped, and the set was recorded the same way (2 passed, 33 deselected).  Reason for the second set:
a new runtime key (GPU name, compute capability, SM count and driver API version differ), not a numerical change.  A part that reports
another GPU name, SM count or software version hashes to another key, and the golden cases skip there -- a coverage gap, not a pass.

Host-entry change, re-run record (2026-11-11 UTC).  A later commit binds the entry's slab operands with a runtime extent and scopes its
compile and its launch to the entry's device.  The kernel source is unchanged; the compiled kernel's parameter block changes (a pointer
and an extent per slab), and from that commit on the kernel the entry compiles on a device is the rendering the functional module pins,
byte for byte, whatever the first call's row count is.  The functional module gains three cases (147): the declared extents of the
entry's views, the entry's own launch arguments compiled against the pinned rendering, and an entry on a device that is not the caller's
current one (needs two CUDA devices).  The determinism suite and both golden sets are unchanged.  Re-run from a clean tree at that
commit with the three commands above, in the runtimes above:

- cc 10.3 (the GB300 runtime, one visible device), CuTe DSL 4.8.0 compiling with CUDA 13.4: 149 passed, 1 skipped (the two-device case:
  "needs two CUDA devices" -- not covered on cc 10.3); 35 passed, 0 skipped against the unchanged set in `6451c9d94a2d4210/`; 1 passed.
- the same part with the CUDA 12.9 compiler flavour of 4.8.0: 148 passed, 2 skipped (the `sm_107a` rendering case, as before, and the
  two-device case); 35 passed, 0 skipped against the same set; 1 passed.
- cc 10.7 (the VR200-class runtime) with TWO devices visible, CuTe DSL 4.8.0 compiling with CUDA 13.4: 150 passed, 0 skipped -- the
  two-device case ran there: first call, warm calls with both output owners and a side stream of the entry's device, the bytes of the
  same call made with that device current, the caller's device unchanged; 35 passed, 0 skipped against the unchanged set in
  `d2a97fcefba4b867/`; 1 passed.
- CuTe DSL 4.8.1 on the cc 10.3 part (the 4.8.1 wheels put first on the path of the same runtime, as in the 4.8.1 record above; these
  two runs are one commit later, which changes this report's text only).  Compiling with CUDA 13.4: 149 passed, 1 skipped (the
  two-device case); the determinism command WITHOUT the goldens variable 33 passed, 2 skipped (the golden cases: no set is committed for
  that runtime key -- a coverage gap, as before); 1 passed.  With only the CUDA 12 toolchain libraries (CUDA 12.9): 148 passed, 2 skipped
  (the `sm_107a` rendering case and the two-device case); 33 passed, 2 skipped; 1 passed.  A record into a new directory under each
  flavour (2 passed, 33 deselected) holds the input and output hashes of the committed 4.8.0 set; it is not committed.  Not run on a
  device: 4.8.1 on cc 10.7.
- Cold compile on an x86 host at that commit (a cc 8.0 device: every device case skips), CuTe DSL 4.8.0 and 4.8.1: 65 passed with CUDA
  13.4, 64 passed and the `sm_107a` rendering case skipped with CUDA 12.9, under either version; the three rendering cases and the
  launch-argument case are among them.
### FP8 projection for decode rows: replay and batch invariance at every row count, goldens per byte class and token tile

`gemm/frost/projection_decode/test_fp8_projection_decode_determinism.py` pins `cudnn.Fp8ProjectionDecode` (an FP8 E4M3 weight times FP8 E4M3
activations to bf16, 1 .. 32 rows, one launch) at its three served byte classes: `ProjectionGeometry(4608, 8192)` at `split_k` 4 and at
`split_k` 2, and `ProjectionGeometry(8192, 2048)` at `split_k` 2.  Forward only: the entry is an inference step with no backward entry
point, so there is no backward case.  Every comparison is raw bytes over the entry's one result (`out`, bf16); the independent fp64
reference of the E4M3 operands is a separate check under a per-element bound (half a bf16 unit plus the fp32 accumulation term; the
utilities module beside the suite), and operands whose partial sums are exact in fp32 are compared byte for byte with the host in
`test_fp8_projection_decode.py`.  The kernel is specialised on the row count -- one rendering per row count -- so replay and batch
invariance have one case per served row count, not a sampled set.  Operands come from a CPU generator, so an operand set has the same
bytes on every device.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | class x rows 1 .. 32; eight executions on fresh output buffers under side-stream contention; then the fp64 reference under its bound | 96 |
| `test_replay_on_a_fresh_entry_and_fresh_operands` | L0 | class x rows 3 / 12 / 20 (one per token tile); two entries, two identical operand sets drawn afresh | 9 |
| `test_batch_invariance` | L0 | class x R 2 .. 32; the R-row call replayed three times; every row equals the same row launched alone (one row: the 8-row token tile, accumulator column 0); the reversed batch and a shuffled batch | 93 |
| `test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` | L0 | class x 8 / 16 / 31 rows; one new row inserted at the head, in the middle and at the tail (8 -> 9 and 16 -> 17 change the token tile, 31 -> 32 reaches the ceiling): the original rows keep their bytes, the inserted row writes its alone bytes; read from the larger batch it is the removal | 9 |
| `test_duplicated_row_and_redrawn_neighbours` | L0 | class; a row twice in one batch (both copies = alone); the other rows redrawn around it at the head, middle and tail of 8-, 16- and 32-row batches | 3 |
| `test_pick_arms_cover_other_ring_depths_at_every_class` | L0 | host only: the arms of the next case hold at least two other ring depths per class, none names a split, each validates on the reference part | 1 |
| `test_pick_is_byte_neutral` | L0 | class; for every row count 1 .. 32, this part's pick against the form the pick returns on modelled parts (148, 212 and 64 SMs; 100 KB of shared memory per SM) and against every band of the pick's table, each run as the entry's `form` | 3 |
| `test_inputs_are_not_modified` | L0 | class; every operand holds its bytes after six launches (the row-alone form and both sides of each token-tile boundary) | 3 |
| `test_thirty_two_to_thirty_three_rows_is_the_typed_decline` | L0 | the row ceiling is a `NotImplementedError` at the launch and at the plan-time compile, never another provider | 1 |
| `test_golden` | L0 | class x rows 3 / 12 / 20; three executions on a fresh entry, the reference, the saved hashes (`fp8-projection-decode-n<OUT>-k<IN>-split<S>-tile<T>`) | 9 |

Collected: 227 cases (226 on a device, 1 host-only).  Replay executions: `test_replay` 96 x 8 = 768; `test_batch_invariance` 93 x 3 of the
batch, 93 x 2 of the permuted batches and 3 x 32 of a row alone; `test_golden` 9 x 3.

The remaining performance knobs (the 128-row weight tile, ring depths from two stages up, 64 / 128 tensor-memory columns, a token tile
wider than the rows need, the weight loader waiting for the predecessor, the late tensor-memory release, the dependent-launch attribute
and the release of the successor) are bitwise with the default in `test_fp8_projection_decode.py`
(`test_performance_arms_write_the_default_bytes`, 45 cases at 8 row counts each), which also holds the CUDA-graph replay against eager,
the warm-launch stress with byte references (2,016 launches per case, plainly and behind another kernel) and the cell that prints, without
asserting, how many elements differ between split 4 and split 2 -- two byte classes, each replayed bitwise.

Golden sets: 9 files, one per byte class and token tile, recorded on cc 10.3 after the replay, batch-invariance and reference cases
passed, into the directory of the runtime that already holds the shared-expert decode pair's cc 10.3 set (`6451c9d94a2d4210/`).  A new
set is owed with a stated reason when the split, the K tile (128 elements, four K-32 tensor-core instructions) or the cast changes.

GPU coverage: cc 10.3 (replay, batch invariance, goldens) and cc 10.7 (replay and batch invariance; no golden set is committed for a
cc 10.7 runtime, so the golden comparison is not covered there).  cc 10.0 was not run: the entry admits it and the status page reads
"SM100: Unverified".  Every run below is one in-process pytest run per command (`-n 0`, or an environment without pytest-xdist), one
visible device, with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`, `CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, from `test/python` of a clean tree: the two runs before and during recording on the tree before the
9 golden files and the cc 10.7 compiler gate were added, every other run on the tree that carries them; `<suite>` =
`gemm/frost/projection_decode/test_fp8_projection_decode_determinism.py`, `<module>` =
`gemm/frost/projection_decode/test_fp8_projection_decode.py`.

cc 10.3 record (2026-11-11 UTC): a GB300 (cc 10.3, 152 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0, driver API
13030, nvidia-cutlass-dsl 4.8.0 compiling with CUDA 13.4, in two environments.  (a) The goldens' runtime -- cuDNN 92400, tvm-ffi
0.1.15.post1 (the `runtime` block of the files in `6451c9d94a2d4210/`): `pytest <suite> -m L0 -q -rsP` before any set existed -> 218
passed, 9 skipped (the 9 `test_golden` cases, "no golden for ..."); `CUDNN_DETERMINISM_RECORD=<new directory> pytest <suite> -m L0 -k
golden` -> 9 passed, 218 deselected; the 9 records were inspected (the runtime block equals the directory's; shapes and dtypes; the
operand hashes equal those of the operands rebuilt on an x86 host from the suite's CPU generator in 9 of 9 files; the two splits of
`ProjectionGeometry(4608, 8192)` share their operand hashes and differ in the result hash at all three token tiles) and copied;
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest <suite> -m L0 -v -rs` -> 227 passed, 0 skipped: `test_replay` 96,
`test_replay_on_a_fresh_entry_and_fresh_operands` 9, `test_batch_invariance` 93,
`test_inserting_or_removing_a_row_keeps_the_other_rows_bytes` 9, `test_duplicated_row_and_redrawn_neighbours` 3,
`test_pick_arms_cover_other_ring_depths_at_every_class` 1, `test_pick_is_byte_neutral` 3, `test_inputs_are_not_modified` 3,
`test_thirty_two_to_thirty_three_rows_is_the_typed_decline` 1, `test_golden` 9.  `test_pick_is_byte_neutral` compared 80 / 40 / 40 (row
count, form) cells at the three classes -- ring depths 1, 2, 3, 4, 7 and 11 (one and two waves) at `(4608, 8192)` split 4, 5 and 7 at
split 2, 3, 4 and 5 at `(8192, 2048)` -- all bitwise with the part's pick.  (b) cuDNN 92700, tvm-ffi 0.1.14.post0 (another runtime key:
no golden set): `pytest <suite> -m L0 -q -rs -n 0` -> 218 passed, 9 skipped (the 9 golden cases); `pytest <module>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0 -k "not performance_arms"` -> 198 passed, 45 deselected, and `pytest <module>
-m L0 -q -rs -n 0 -k performance_arms` -> 45 passed, 194 deselected: the module 239 passed, 0 skipped, the symbol-prefix test 4 passed.

The other compiler flavor of the DSL on cc 10.3 (2026-11-11 UTC): nvidia-cutlass-dsl 4.8.0 bound to its CUDA 12.9 compiler flavor
(`cutlass.base_dsl.version_info.CUDA_VERSION` = 12.9) inside environment (a), under the same CUDA 13 driver, torch and cuDNN -- a
statement about the DSL's compiler flavor, not about a CUDA 12 deployment.  `CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens pytest
<suite> -m L0 -q -rs` -> 227 passed, 0 skipped: the kernels compiled by the CUDA 12.9 flavor write the bytes of the goldens recorded
under the CUDA 13.4 flavor.  `pytest <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs` -> 243 passed, 0 skipped (the
case that launches the kernels in a child interpreter, `test_launch_matrix_completes_in_a_child_process`, compiles there with the
default flavor).

cc 10.7 record (2026-11-11 UTC): a VR200-class part (cc 10.7, 212 SMs; aarch64 host, Python 3.12) with torch 2.14.0+cu130, CUDA 13.0,
driver API 13050, cuDNN 92700, nvidia-cutlass-dsl 4.8.0 compiling with CUDA 13.4, tvm-ffi 0.1.14.post0.  `pytest <suite> -m L0 -v -rs
-n 0` -> 218 passed, 9 skipped (the 9 golden cases, "no golden for ...": no set for this runtime); `pytest <module>
core/frost/test_frost_kernel_name_prefix.py -m L0 -q -rs -n 0 -k "not performance_arms"` -> 198 passed, 45 deselected (`<module>` 194,
the symbol-prefix test 4), and `pytest <module> -m L0 -q -rs -n 0 -k performance_arms` -> 45 passed, 194 deselected: the module 239
passed, 0 skipped.  With cuDNN 92400 and tvm-ffi 0.1.15.post1 on the same part, a candidate set was recorded into a new directory (9
passed, 218 deselected) and the suite was checked strictly against it (227 passed, 0 skipped); the operand and result hashes of its 9
records equal those of the cc 10.3 set.  That equality is an observation on these two runtimes, not a contract across architectures;
the candidate is not committed, so a cc 10.7 run of the suite skips its 9 golden cases.

The other compiler flavor of the DSL on cc 10.7 is a DECLINE, not a lane: the DSL's CUDA 12.9 compiler has no target for the part (the
DSL refuses it from inside the trace), so the entry declines it by name before any compile.  In the cuDNN 92400 environment bound to
that flavor, `pytest <suite> <module> core/frost/test_frost_kernel_name_prefix.py -m L0 -v -rs` -> 73 passed, 397 skipped: every case
that needs a compiled kernel skips with the entry's message ("... on compute capability 10.7 needs a CuTe DSL that compiles with CUDA
13.4 or newer ..."), `test_compile_follows_the_gate_on_this_device` passes on its declined branch (`compile` raises that message, nothing
is traced), the host-only cases and the symbol-prefix test pass.  Before the gate existed the same command failed 394 cases inside the
DSL; replay and batch invariance on cc 10.7 are therefore evidence for the CUDA 13.4 flavor only.

### QSA: 42 cases

QSA uses BF16 Q/K/V with head dimension 256. The saved FP32 output and LSE
must match even when the BF16 output rounds to the same value.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × Q:KV heads 12:1, 24:2, 6:1; eight executions; input mutation check | 6 |
| `test_batch_invariance` | L0 | Same six combinations; lengths 65, 3, 17; removal/reorder with masks preserved | 6 |
| `test_golden` | L0 | fprop/bprop; lengths 17, 3; three executions, reference, saved hashes; the backward runs its default path | 2 |
| `test_sparse_batch_invariance` | L0 | fprop/bprop; lengths 2057, 3; four executions then standalone comparison | 2 |
| `test_dispatch_boundary` | L0 | fprop/bprop × total tokens 512/513; a fixed 3-token target before/after a neighbor | 4 |
| `test_mask_replay` | L0 | fprop/bprop × all selected, every seventh row empty, all masked; reference and eight executions | 6 |
| `test_mask_batch_invariance` | L0 | Same six combinations; each sequence alone, reversed batch, duplicate sequence | 6 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring values, masks, and upstream gradients; the backward keeps the packed layout | 2 |
| `test_zero_upstream_gradient_bprop` | L0 | Eight executions with zero dO; all gradients must be zero | 1 |
| `TestQSABackwardEngine::test_replay` | L0 | bitmap and indexed entries with `deterministic=True`; multi-tile dense, sparse record windows, packed sparse; four executions and reference | 7 |

Head parameters have explicit IDs such as `q24-kv2`. A failure such as
`test_dispatch_boundary[total513-fprop]` identifies the routing boundary and
direction without a separate lookup table. A test stops at its first mismatch;
later checks in that case are not claimed as passes.

### DSA: 101 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_determinism.py`:
56 `TestDSA` cases and 45 `TestDSADeterministicDKV` cases.
Each `TestDSA` case checks all returned forward tensors. Backward uses deterministic
execution and checks dQ, the shared K/V latent gradient dKV, and dSink.
The three `(heads, QK dimension, Top-K)` layouts are `(64,512,64)`,
`(64,576,64)`, and `(128,512,128)`; `(96,576,64)` and `(128,576,64)` run
forward only. SM100 H96 forward launches the H64 kernel on heads [0, 64) and
[32, 96), and H128 D576 forward on [0, 64) and [64, 128); their backward is
not in this suite. The 576-wide path uses all QK dimensions
but only the first 512 KV dimensions for values.

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | L0 | fprop/bprop × FP16/BF16 × three layouts, and fprop × FP16/BF16 × H96 and H128 D576; eight executions, input mutation check, numerical reference | 16 |
| `test_batch_invariance` | L0 | Same sixteen combinations; each sequence alone, packed, reversed, and duplicated | 16 |
| `test_selection` | L0 | fprop/bprop × ragged lengths, invalid indices, empty rows, Top-K 65 tail, indexer prefix 512; replay, reference, batch checks | 10 |
| `test_query_boundary` | L0 | fprop/bprop × H64 with 128/129/257 queries and H128 with 74/75 queries; four replays then batch checks | 10 |
| `test_neighbor_isolation` | L0 | fprop/bprop; change neighboring Q/KV values and Top-K selection while keeping the target fixed | 2 |
| `test_golden` | L0 | fprop/bprop; query lengths 3, 2; Top-K 16; three executions, reference, saved hashes | 2 |

Packing shifts valid Top-K indices by the sequence's KV offset. Negative and
out-of-bounds indices stay invalid; they must not name a neighboring KV row.
`TestDSA` rows have unique valid indices, as the shard and relay layouts of
deterministic DSA backward require. Duplicating a sequence creates a separate
KV region.

dSink is one shared per-head vector, so it cannot be sliced by sequence.
Backward batch checks isolate one target's loss: its dO stays fixed and all
other sequences receive zero dO. The complete dSink vector is then comparable
to the target alone. Replay/reference cases also cover nonzero dO for every
sequence. Empty-row sentinels retain their actual float dtype and raw bits;
NaNs and non-finite outputs/gradients still fail.

### DSA deterministic dKV layouts: 45 cases

`TestDSADeterministicDKV` runs on SM100/SM103, where deterministic backward
stores dKV in conflict-rank storage, and skips elsewhere: SM107 plans keep the
shard layouts that `TestDSA` covers. Every case runs the public wrappers with
`deterministic=True` and `deterministic_dkv_mode`, and asserts the backend and
dKV layout each batch selects. Each layout packs two sequences; the packed
batch spans several waves, and the KV extents keep fewer dense planes than a
wave has queries, so shared rows also take overflow records.

| Layout ID | Heads, QK dim, Top-K, dtype | Query lengths | KV lengths | Mode | Backend/layout |
| --- | --- | --- | --- | --- | --- |
| `records-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-h64-d512-k64-fp16` | 64, 512, 64, FP16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-h64-d576-k64-bf16` | 64, 576, 64, BF16 | 17, 300 | 1000, 1000 | `records` | generic M64, records |
| `records-strided-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 1183 | 1000, 1000 | `records` | generic M64, records, strided waves |
| `chain-h64-d512-k128-bf16` | 64, 512, 128, BF16 | 17, 300 | 2048, 2048 | `chain` | generic M64, chain relay |
| `shards-h64-d512-k64-bf16` | 64, 512, 64, BF16 | 17, 300 | 1000, 1000 | `shards` | generic M64, shards |
| `rank-fused-h128-d512-k128-bf16` | 128, 512, 128, BF16 | 40, 60 | 8192, 16384 | `records` | H128 D512 two-CTA, fused conflict-rank |
| `rank-setup-h128-d512-k1024-bf16` | 128, 512, 1024, BF16 | 40, 60 | 8192, 16384 | `records` | H128 D512 two-CTA, per-wave-setup conflict-rank |
| `rank-h128-d576-k128-bf16` | 128, 576, 128, BF16 | 17, 60 | 8192, 16384 | `records` | H128 D576 two-CTA, one-round conflict-rank |
| `rank-sorted-h128-d576-k1024-bf16` | 128, 576, 1024, BF16 | 17, 60 | 2200, 2201 | `records` | H128 D576 two-CTA, one-round conflict-rank, sorted rows |

| Test method | Level | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_dkv_replay` | L0 | Every layout with full selections; the records BF16, chain, both H128 D512 and the D576 Top-K 128 layouts also with ragged and duplicate selections; shards also with ragged selections. Eight executions with fresh outputs, caller workspaces filled with 0x00, 0xA5 and 0xFF bytes, and side-stream work; input mutation check; numerical reference | 21 |
| `test_dkv_neighbor_isolation` | L0 | Every layout, ragged selections; the other sequence's values, Top-K selection and lengths change while the shapes stay; every result | 10 |
| `test_batch_invariance` | L0 | Every layout except the sorted D576 one, ragged selections; each sequence alone, packed, reversed and duplicated; forward results, dQ and, for the generic M64 layouts, dSink | 9 |
| `test_independent_of_dkv_storage` | L0 | Same inputs under two modes: records and shards (H64; forward results, dQ, dSink), chain and shards (forward results, dSink), conflict-rank and private planes (H128 D512 Top-K 128 and 1024, D576 Top-K 128; forward results and dQ) | 5 |

Ragged selections add per-query lengths with empty rows and negative and
out-of-range indices. Duplicate selections repeat a quarter of every row's slots
and make every query select its sequence's KV row 0 Top-K / 32 times; they
replay the larger sequence alone, so that row is shared by every query of a
wave. Duplicates are inside the contract of every conflict-rank layout; no
shard case uses them. H128 D576 has no public forward kernel: its sequences
carry reference forward results as fixed backward inputs, so these cases
compare no forward results.

Known gaps, stated rather than tolerated:

- dKV is not batch invariant once a batch spans several waves, on any layout.
  A conflict-rank layout ranks a KV row's writers by their wave position and
  sizes its dense planes from the batch's KV extent; the shard layouts take
  their shard count from the batch's query extent and fold shards in groups;
  the default mode picks the layout from the batch's extents. `develop`
  changes dKV in the same multi-wave comparisons (validation record below).
  `test_batch_invariance` therefore does not compare dKV.
- Generic M64 plans reduce dSink with one warp per head (lane `t` sums queries
  `t, t + 32, ...`, then an xor butterfly); a uniform query shift only rotates
  the lanes, so dSink is batch invariant under the measured packings and
  records and chain plans give the shard plans' bits. H128 two-CTA plans sum
  dSink in an order that follows query positions, in both layouts, as on
  `develop`; their batch cases do not compare dSink.
- H128 D576 plans with Top-K 1024 and wider sort each query's rows by ranges of
  the batch's KV extent and alternate the walk direction between waves, so
  their dQ is not batch invariant either and differs in bits from the
  private-plane plans; the sorted layout has replay and neighbor cases only.
- Not exercised: records plans with split pairs (their dQ adds two partials
  once the batch's KV extent selects them), chain plans whose rows are shared
  by every active query of a wave (their tile order follows those rows),
  H128 D576 one-round plans wider than Top-K 128 that stride their waves (the
  wave order follows the batch's indices), the few-plane H128 D512 variant,
  H96 D576 plans and the cluster relay. The L1 bitwise tests in
  `test_DSA_sparse_attention_backward.py` cover several of these paths.
- These layouts have no golden records; their bits depend on the plan, and
  for D576 on the SM count.

### DSA BF16 indexer: 46 cases

The suite lives in `deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py`.
Q/K are BF16. Forward weights use BF16 or FP32; scores are FP32. Backward
weights, dQ and dWeights are BF16; dK uses BF16 or FP32. Score operands,
normalizers and the loss gradient are FP32. This BF16-input suite is separate from the FP8/MXFP8 cases below.

| Test method | API path | Parameters and purpose | Cases |
| --- | --- | --- | ---: |
| `test_replay` | Forward | H32/H64 × BF16/FP32 weights × ratio 1/4; eight executions and FP64 reference | 8 |
| `test_batch_invariance` | Forward | Same eight combinations; standalone, packed, reversed, duplicate | 8 |
| `test_boundary` | Forward | Q/K lengths 127/128/129; all-masked rows; four executions, reference, batch checks | 4 |
| `test_ragged_batch` | Forward | BF16/FP32 weights; packed query/key lengths (3,7)/(17,129), reorder/duplicate, mask padding | 2 |
| `test_neighbor_isolation` | Forward | Fixed target with changed neighboring Q/K/weights | 1 |
| `test_golden` | Forward | BF16 weights, three queries; reference, three executions, saved hash | 1 |
| `test_replay` | Backward | Sparse H32/H64 and dense H64 × BF16/FP32 dK; reference then eight executions | 6 |
| `test_batch_invariance` | Backward | Same six combinations; standalone, packed, reversed, duplicate | 6 |
| `test_sparse_invalid_slots` | Sparse backward | Local/global indices with negative padding; reference, replay, batch checks | 2 |
| `test_sparse_persistent_rows` | Sparse backward | Top-K 896/2048, 193 queries over 96 producer planes; reference, four executions, batch checks | 2 |
| `test_zero_loss_gradient` | Both backward APIs | Zero loss gradient; eight executions and exact zero gradients | 2 |
| `test_neighbor_isolation` | Both backward APIs | Fixed target and loss scale with changed neighbors | 2 |
| `test_golden` | Both backward APIs | One query, FP32 dK; reference, three executions, saved hashes | 2 |

All 46 cases are L0. Sparse backward sets `deterministic=True`; dense backward
has no determinism switch. Each call restores mutable score inputs and checks
immutable inputs. Gradient buffers start at NaN; sparse deterministic workspace
starts at `0xA5`. Results include all three gradients and each mutated score
buffer. Numerical references use the existing autograd checks with cosine
similarity at least 0.99 and RMS relative error at most 0.02; bitwise checks
still allow no difference.

Backward sets `loss_coeff = B * S_q` and fixes `grad_loss = 0.75`, so adding or
removing a batch member does not change its gradient scale. Local Top-K IDs
stay local. Global IDs are rebuilt for each batch order; negative padding stays
invalid. The 193-row cases force a persistent producer to process a third row. Golden backward cases have one query; their passes do not
establish determinism for multi-query reductions.

See the [API-by-dtype results](#api-and-datatype-results) and
[SM100 indexer failures](#indexer-failures-sm100) below.

### DSA FP8/MXFP8 indexer: 34 cases

`deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py` adds
17 L0 cases for SM100/SM103/SM107 MXFP8 and the same 17 for SM90 FP8.
Each parameter ID names the architecture group and precision. The fixture
skips other architectures before launching a kernel and checks the DSL target.

| Method | Parameters | Cases per precision |
| --- | --- | ---: |
| `test_replay` | H32/H64 × ratio 1/4; eight executions, numerical reference | 4 |
| `test_batch_invariance` | Same combinations; standalone, packed, reordered, each sequence duplicated | 4 |
| `test_boundary` | Q/K lengths 127/128/129, all-masked rows; four executions, reference, batch checks | 4 |
| `test_ragged_batch` | H32/H64; lengths (3,7)/(17,129); replay, reference, reorder/duplicate with repacked scales | 2 |
| `test_neighbor_isolation` | Fixed target payload/scales with changed neighbors | 1 |
| `test_scale_change` | Fixed FP8 payload; doubled Q scales must change scores and still match the reference/replay | 1 |
| `test_golden` | Three queries; reference, three executions, payload/physical-scale/output hashes | 1 |

Both paths use E4M3 Q/K and BF16 weights. MXFP8 uses nonuniform E8M0 scales
per 32 values, packed with the independent scale-layout reference. SM90 uses
FP32 token/head descales. Quantization happens once per sequence using CPU-seeded
inputs; repacking never requantizes. Input mutation checks compare raw payload
and scale bytes. Both paths compare FP32 scores; SM90 also compares FP32 LSE.
The score reference uses dequantized inputs and FP64 computation. Existing
correctness tolerances apply only to references; replay and batch comparisons
remain byte-exact.

SM100/SM103/SM107 MXFP8 and SM90 FP8 each had **17 passed, 0 failed**,
including their own saved golden. Each device skips the other precision group.
The [API-by-dtype results](#api-and-datatype-results) keep these gaps separate
from measured results. FP8/MXFP8 backward is not
supported by these DSA attention/indexer APIs and has no positive test cases.

## Golden checks

Each golden records SHA-256 hashes of all input and result bytes, with their
shapes and dtypes. This retains full precision without large tensor files.
The runtime key includes the GPU, SM count, Torch, CUDA, driver API version,
cuDNN, CuTe DSL, and TVM FFI.
The frontend revision is deliberately not part of the key: the check must
detect changes across frontend revisions on the same runtime.

Golden cases first pass replay and an independent numerical reference.
Linear attention uses the existing FP64 recurrent references on the CPU.
QSA uses its existing FP64 attention and gradient reference. DSA uses the
existing FP32 gather-based forward and autograd gradient references. Reference
tolerances test mathematical correctness; golden and replay checks use no
tolerance. A first execution is only a replay baseline, never a saved golden.

A normal run reads this directory. An unrecorded runtime skips only the golden
comparison; replay and batch checks still run. For CI, require a complete
runtime baseline with an explicit directory. Missing goldens then fail:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0
```

Record candidates into a **new** directory on a known-good revision:

```bash
CUDNN_DETERMINISM_RECORD=/tmp/attention-goldens-candidate \
  pytest linear_attention/frost/test_*_determinism.py QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -m L0 -k golden
```

Recording refuses to overwrite an existing file. Check every golden supported on the target architecture, inspect
the input/output hashes, and explain any numerical change before replacing
committed files. Never enable recording in a PR check. A new runtime needs its
own reviewed baseline; matching results across architectures is not assumed.

## Agent and review rules

Follow [the root agent rules](../../../AGENTS.md#kernel-determinism-and-batch-invariance)
and [the test guide](../../AGENTS.md). The PR template requires an accompanying
determinism test change for kernel changes, or an explicit unsupported-path
explanation. Extend this table and the suites when adding an operation or a
new numerical path. Keep hardware/dependency skips specific; a missing FROST
plan on a supported runtime is a failure. Explicitly requested temporary skips
must name the defect, retain the recorded failures, and be removed with its fix.

The earlier suites cover FROST and public QSA/DSA APIs on SM100/SM103/SM107.
The quantized indexer suite adds SM90 FP8 and SM100/SM103/SM107 MXFP8.
The set excludes cuTile/CAKE, other Hopper paths, QSA indexed/two-pass APIs,
checkpoint/summary/decode paths, shared gate parameters, fused/separate Top-K,
paged/MXFP4 indexer forward, THD dense indexer backward, score recompute,
pooled DSA, and distributed recipes. Existing tests cover some paths separately.
Recipe CI and an automatic source-to-test gate remain integration work;
the checklist is a review safeguard.

## Temporary skips

The 21 cases below use `skip until fix` reasons, as requested. They remain L0.
The set is the union of failures recorded on SM100, SM103, and SM107. A pass
on one run does not clear a nondeterministic case that failed on another GPU.
Passing parameters, golden checks, and all FP8/MXFP8 cases remain active.

| Suite | Temporarily skipped parameters | Cases |
| --- | --- | ---: |
| QSA | Backward batch checks, all three head layouts | 3 |
| QSA | Sparse batch checks, both directions | 2 |
| QSA | Dispatch boundary at total513, both directions | 2 |
| QSA | Backward mask batch checks, all-selected and empty-rows | 2 |
| DSA attention | H128/D512/Top-K128 BF16 backward batch check | 1 |
| DSA attention | Backward query boundaries H64 Q129/Q257 and H128 Q74/Q75 | 4 |
| DSA indexer | Dense H64 replay/batch checks, BF16 and FP32 dK | 4 |
| DSA indexer | Sparse persistent rows, Top-K896/2048 | 2 |
| DSA indexer | Dense backward neighbor isolation | 1 |

These skips do not change API support status. The baseline measurements and
failure diagnostics below were recorded before these skips. Remove each skip
with the kernel fix and verify the affected cases on the supported GPUs.

### Skip validation: 2026-11-05 (UTC)

The three changed suites ran on SM100: **103 passed, 26 skipped, 0 failed**
in 131.55 seconds. Runtime key: `5f8a825586430e85`, as in the SM100 baseline
below. Imports resolved to this checkout. Golden recording was disabled and
all seven golden cases in these suites passed.

| Suite | Passed | Skipped |
| --- | ---: | ---: |
| QSA | 21 | 14 |
| DSA sparse attention | 43 | 5 |
| DSA indexer forward | 24 | 0 |
| DSA indexer backward | 15 | 7 |

The skipped case IDs exactly match the union of the recorded SM100/SM103/SM107
failures. All 285 original case IDs still collect with `-m L0`. Pre-commit and
`git diff --check` passed. Other suites and GPUs were not rerun for this change.

Run from `test/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py -m L0 -q -rs
```

## SM100 H128 windowed sparse attention forward validation: 2026-11-11 (UTC)

Change under test: `DSA.sparse_attention_forward_wrapper` accepts H128 D576,
and H128 D512 with `indexer_topk=2048`, on SM100 by launching the H64 kernel
on heads [0, 64) and [64, 128); `TestDSA` adds the forward-only
`h128-d576-k64` layout (four cases). Source: this change on the H96 change
above. Only the DSA suite was rerun.

| GPU | Runtime | Result |
| --- | --- | --- |
| B300, SM103 (cc 10.3), 148 SMs | Torch `2.14.1+cu130`, CUDA `13.0`, driver API `13030`, cuDNN `92700`, CuTe DSL `4.8.1`, TVM FFI `0.1.15.post0`; key `55097120e8121c53` | **94 passed, 2 failed, 5 skipped**: the four H128 D576 and four H96 cases passed; five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |

H128 D576 backward is outside this change and is not in this suite.

## SM100 H96 sparse attention forward validation: 2026-11-11 (UTC)

Change under test: `DSA.sparse_attention_forward_wrapper` accepts H96 with
D512/D576 on SM100 by launching the H64 kernel on heads [0, 64) and [32, 96),
and `TestDSA` adds the forward-only `h96-d576-k64` layout (four cases). Source:
this change on `develop` `17e10b09e`. `cudnn` resolved to a package whose
Python modules link to this checkout, with the installed pybind module. Only
the DSA suite was rerun.

| GPU | Runtime | Result |
| --- | --- | --- |
| B200, SM100, 148 SMs | Torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13040`, cuDNN `92600`, CuTe DSL `4.8.0` (CUDA 13.4 compiler), TVM FFI `0.1.14`; key `5c945a33b1699e7e` | **90 passed, 2 failed, 5 skipped** in 177 s: the four H96 cases passed; five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |
| B300, SM103 (cc 10.3), 148 SMs | Same software runtime; key `b28819ca225d3a0b` | **90 passed, 2 failed, 5 skipped** in 289 s: the four H96 cases passed; five temporary skips; both goldens failed with `no golden` (no baseline for this runtime) |
| SM107 (cc 10.7, `NVIDIA Graphics Device`), 208 SMs | Same software runtime; key `1c7a6611734bfd74` | **45 passed, 2 failed, 50 skipped** in 23 s: the four H96 cases passed; five temporary skips and 45 `TestDSADeterministicDKV` cases that run only on SM100/SM103; both goldens failed with `no golden` (no baseline for this runtime) |

H96 backward is outside this change and is not in this suite.

From `test/python`:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 -q -rs
```

## Deterministic dKV storage validation: 2026-11-06 (UTC)

Change under test: conflict-rank dKV storage for deterministic SM100/SM103 DSA
backward, the `deterministic_dkv_mode` option, conflict-rank plans that reduce
dSink like the shard plans, and `TestDSADeterministicDKV`. Source revision:
<RESULT revision>. The DSA suite was rerun; the other suites were not rerun
for this change and keep their earlier results.

The default plans of these `TestDSA` cases now select conflict-rank storage,
so their dKV bits change: `test_replay[h128-d512-k128-bf16-bprop]` and the
skipped H128 BF16 batch and query-boundary cases (fused H128 D512 layout), and
the packed batch of the skipped `test_query_boundary[h64-q257-bprop]`
(records). The golden cases (H64, D512, Top-K 16, query lengths 3 and 2) keep
the 32-shard layout; no golden record changes.

Run from `test/python`:

```bash
python -c 'import cudnn; print(cudnn.__file__)'
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 --collect-only -q
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_determinism.py -m L0 -q -rs
```

The B200 runs below omitted `CUDNN_DETERMINISM_GOLDENS` because their runtime
has no recorded baseline; the two golden cases then skip.

| GPU | Runtime | Result |
| --- | --- | --- |
| B200, SM100, 148 SMs | Torch `2.14.0a0+4fdf77b940.nv26.08`, CUDA `13.4`, driver API `13040`, cuDNN `92500`, CuTe DSL `4.8.0` (CUDA 13.4 compiler), TVM FFI `0.1.13.post3`; key `3c93e700c060c612` | **86 passed, 0 failed, 7 skipped** in 192 s: 45 of 45 `TestDSADeterministicDKV`, 41 `TestDSA`; five temporary skips, two goldens without a baseline |
| B200, SM100, 148 SMs | Same, CuTe DSL `4.8.0` with the CUDA 12.9 compiler | Earlier revision of this change, before the dSink reduction change: **85 passed, 0 failed, 7 skipped** in 186 s |
| SM100, recorded runtime `5f8a825586430e85` | <RESULT runtime> | <RESULT pytest summary, including both goldens> |
| B300, SM103, 148 SMs | <RESULT runtime> | <RESULT pytest summary> |

The new cases take about 140 s of a B200 run with empty compile caches,
mostly first-use compilation of their plans.

With the temporary skips removed locally (not committed), four of the five
`TestDSA` cases fail on B200 on dKV, as on `develop`; 42 passed and the two
goldens skipped. The H128 cases now fail at the first packed batch, because the
conflict-rank plane count follows the batch's KV extent.
`test_query_boundary[h64-q257-bprop]` passes on B200: the target alone selects
shards and the packed batch selects records, whose dQ and dSink match the shard
plans and whose dKV for this target adds the same contributions in the same
order. The case keeps its skip: SM107 plans keep the shard layouts, where it
failed on dKV ([SM107 validation](#sm107-validation)), and the skip set is the
union of the recorded failures.

| Skipped case | First mismatch with conflict-rank storage (B200) |
| --- | --- |
| `TestDSA::test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (0, 1), sequence 0 dkv: 8/164864 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | passes |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (0, 1), sequence 0 dkv: 8/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (0, 1), sequence 0 dkv: 9/164864 bytes differ |

#### Batch layouts on `develop` and with this change (B200)

The same comparisons ran on `develop` (no `deterministic_dkv_mode`) and with
this change under `'shards'`, `'records'` and `'chain'`: each sequence alone
against its slice of the packed, reversed and duplicated batches (six
comparisons per configuration, other sequences' dO zero), neighbor isolation,
and replay, on FP16 and BF16. Configurations: the `TestDSA` sequences (lengths
17 and 3) for H64 D512, H64 D576 and H128 D512; H64 D512 Top-K 64 with lengths
17 and 300 and KV 1000 + 1000 (three records waves), and lengths 17 and 1183
(strided records waves), full and ragged selections; the same with KV
2048 + 2048 (Top-K 128 and 64) for the chain relay. Counts are comparisons out
of six with any differing byte.

| Configuration | `develop` | `'shards'` | `'records'` | `'chain'` |
| --- | --- | --- | --- | --- |
| `TestDSA` sequences, H64 D512/D576 and FP16 H128 | none differ | none differ | none differ (shard layout selected) | none differ (shard layout selected) |
| `TestDSA` sequences, BF16 H128 D512 (two-CTA) | dKV 1, dSink 2 | dKV 1, dSink 2 | dKV 3, dSink 2 | rejected (two-CTA backend) |
| Three waves, FP16 and BF16, full / ragged | dKV 3 / 1 | dKV 3 / 1 | dKV 3 / 3 | dKV 5 / 2-3 |
| Strided waves, FP16 and BF16, full / ragged | dKV 3 / 1 | dKV 3 / 1 | dKV 4-5 / 3 | dKV 4-5 / 3 |
| KV 2048 + 2048, three and strided waves, FP16 and BF16, full / ragged | not run | dKV 1-3 / 1-2 | dKV 3 / 3 | dKV 3-5 / 2-3 |

Forward results, dQ, replay and neighbor isolation matched in every
configuration. Every generic configuration kept dSink under all three modes;
records and chain dSink is bitwise the shard plans' dSink for the same inputs.
`develop` and `'shards'` gave the same per-comparison counts. FP16 DSA backward
on SM100/SM103 is listed as Replay only because `develop`'s shard layouts
already change dKV at these multi-wave shapes; the earlier Supported status
reflected the single-wave `TestDSA` shapes only.

SM107 was not run for this change: its plans keep the shard layouts that
`TestDSA` measured on SM107, and `TestDSADeterministicDKV` skips there.

## Baseline validation: 2026-11-04

Kernel base: `da88207aa5b9197b0c84514a7321d5bc3d2683b4` (`develop`).
GPU: SM100, 148 SMs. Runtime: Torch `2.15.0.dev20260820+cu134`, CUDA `13.4`,
cuDNN `92500`, CuTe DSL `4.8.0.dev0`, TVM FFI `0.1.12`.
The complete runtime record is in [the goldens](5f8a825586430e85/dsa-fprop.json).
The test banners confirmed imports from this checkout.

Three disjoint default-L0 runs on SM100 covered all 285 cases. Run from `test/python`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  deepseek_sparse_attention/cutedsl/test_DSA_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA_indexer_determinism.py -q
python -m pytest linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py -q
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_fp8_determinism.py -q
```

- DSA attention, indexer, and harness: **89 passed, 11 failed, 0 skipped** in 128.77 seconds.
- Existing five operation suites: **137 passed, 14 failed, 0 skipped** in 489.44 seconds.
- New FP8/MXFP8 module: **17 passed, 0 failed, 17 skipped** in 20.17 seconds.
- Combined: **243 passed, 25 failed, 17 skipped**. Node IDs match the
  285 cases selected by default L0 collection (0.29 seconds). The five existing
  operation suites retain their earlier results; their code is unchanged by
  the indexer addition.

| Suite | fprop | bprop |
| --- | --- | --- |
| KDA | 13 passed | 15 passed |
| GDP | 15 passed | 17 passed |
| GDN | 13 passed | 15 passed |
| GDN-2 | 13 passed | 15 passed |
| QSA | 15 passed, 2 failed | 6 passed, 12 failed |
| DSA sparse attention | 24 passed | 19 passed, 5 failed |
| DSA BF16 indexer forward | 24 passed | — |
| DSA sparse indexer backward | — | 13 passed, 2 failed |
| DSA dense indexer backward | — | 3 passed, 4 failed |
| DSA MXFP8 indexer (SM100) | 17 passed | Unsupported |
| DSA FP8 indexer (SM90) | 17 passed on SM90; 17 skipped on SM100 | Unsupported |

All six harness controls and all sixteen goldens runnable on SM100 passed,
including replay and independent references. SM90 also passed its FP8 golden
on Hopper. MXFP8 and Hopper FP8 each add one record; the fifteen earlier
records are unchanged. Golden recording was disabled in validation. Sparse indexer large-Top-K cases pass replay and
reference, then fail dK after reorder. Dense indexer backward fails dK replay,
batch, or neighbor checks. The [API-by-dtype results](#api-and-datatype-results)
and [SM100 indexer failure list](#indexer-failures-sm100) record the measurements.

Pre-commit (Black, Black Jupyter, SPDX), Python syntax, and `git diff --check`
passed. Runtime warnings were Torch context/debug notices and existing CuTe
DSL scheduler/allocator deprecations. `fern check --warnings` found zero
errors; the remote redirect check was skipped after an FDR HTTP 403 response.

### API and datatype results

Each cell is **passed / failed / skipped** for the 2026-11-04 runs.
Counts are expanded pytest cases, each assigned once to its direction.
Backward cases also compare forward results where applicable. A case stops at
its first mismatch; later comparisons do not count as passes.

| API | Direction | Datatype | SM90 Hopper | SM100 | SM103 | SM107 Rubin |
| --- | --- | --- | --- | --- | --- | --- |
| `kimi_delta_attention` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `kimi_delta_attention` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `kimi_delta_attention` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `kimi_delta_attention` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_product` | fprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_product` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_product` | bprop | BF16 | 0 / 0 / 14 | 14 / 0 / 0 | 14 / 0 / 0 | 14 / 0 / 0 |
| `gated_delta_product` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `gated_delta_net` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_net` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net_v2` | fprop | BF16 | 0 / 0 / 10 | 10 / 0 / 0 | 10 / 0 / 0 | 10 / 0 / 0 |
| `gated_delta_net_v2` | fprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `gated_delta_net_v2` | bprop | BF16 | 0 / 0 / 12 | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 |
| `gated_delta_net_v2` | bprop | FP16 | 0 / 0 / 3 | 3 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |
| `qsa_forward` | fprop | BF16 | 0 / 0 / 17 | 15 / 2 / 0 | 15 / 2 / 0 | 15 / 2 / 0 |
| `qsa_backward` | bprop | BF16 | 0 / 0 / 18 | 6 / 12 / 0 | 7 / 11 / 0 | 6 / 12 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | BF16 | 0 / 0 / 18 | 18 / 0 / 0 | 18 / 0 / 0 | 18 / 0 / 0 |
| `DSA.sparse_attention_forward_wrapper` | fprop | FP16 | 0 / 0 / 6 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | BF16 | 0 / 0 / 18 | 13 / 5 / 0 | 13 / 5 / 0 | 14 / 4 / 0 |
| `DSA.sparse_attention_backward_wrapper` | bprop | FP16 | 0 / 0 / 6 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; BF16 weights | 0 / 0 / 15 | 15 / 0 / 0 | 15 / 0 / 0 | 15 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | BF16 Q/K; FP32 weights | 0 / 0 / 9 | 9 / 0 / 0 | 9 / 0 / 0 | 9 / 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 0 / 0 / 4 | 4 / 0 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| `DSA.indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 0 / 0 / 11 | 9 / 2 / 0 | 9 / 2 / 0 | 9 / 2 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; BF16 dK | 0 / 0 / 2 | 1 / 1 / 0 | 1 / 1 / 0 | 1 / 1 / 0 |
| `DSA.dense_indexer_backward_wrapper` | bprop | BF16 Q/K/W; FP32 dK | 0 / 0 / 5 | 2 / 3 / 0 | 2 / 3 / 0 | 2 / 3 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | MXFP8 E4M3 Q/K; E8M0 scales; BF16 weights | 0 / 0 / 17 | 17 / 0 / 0 | 17 / 0 / 0 | 17 / 0 / 0 |
| `DSA.indexer_forward_wrapper` | fprop | SM90 FP8 E4M3 Q/K; FP32 scales; BF16 weights | 17 / 0 / 0 | 0 / 0 / 17 | 0 / 0 / 17 | 0 / 0 / 17 |

The table covers 279 operation cases per GPU. Six harness controls also passed
on each GPU. Hopper attention and BF16 indexer skips are coverage gaps; they
do not establish that the APIs are unsupported. The other precision group's
FP8/MXFP8 cases skip on each architecture. Runtime details follow below.

Run the full selection from `test/python`, with imports from this checkout:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
export CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens"
python -m pytest core/test_determinism_utils.py \
  linear_attention/frost/test_*_determinism.py \
  QSA/cutedsl/test_qsa_determinism.py \
  deepseek_sparse_attention/cutedsl/test_DSA*determinism.py -q
```

Add `--collect-only` to list the 285 L0 cases. The explicit golden directory
makes missing runtime baselines fail. Keep golden recording disabled.

### SM90 Hopper validation

The same 285-case default-L0 selection had **23 passed, 0 failed, 262 skipped**
in 13.71 seconds. All 17 FP8 indexer cases and six harness controls passed.
The other cases skip at their architecture gates. This does not measure
attention or BF16 indexer behavior on Hopper.

Runtime: SM90, 124 SMs; Torch `2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`,
driver API `13040`, cuDNN `92700`, CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`.
The [Hopper golden](ffe6ff35e0022211/dsa-indexer-fp8-fprop.json) was recorded
only after replay and the FP64 reference passed, then checked again with
recording disabled. It includes E4M3 input bytes, FP32 scales, weights, offsets,
scores, and LSE. Test source came from commit
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.

### SM103 validation

All 285 L0 cases ran: **244 passed, 24 failed, 17 skipped** in 482.93 seconds.
The 17 skips are SM90-only FP8 cases. All 17 MXFP8 indexer cases, six harness
controls, and 16 operation goldens passed. Failures: 13 QSA, five DSA attention
backward, and six BF16 indexer backward cases.

GPU: `NVIDIA B300 SXM6 AC`, SM103, 148 SMs. Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13040`, cuDNN `92700`,
CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`. Source snapshot:
`8671c79c2a1e7073bf1efe0cce1939f093a0d840`; imports resolved to that snapshot.
The [SM103 goldens](7a727a3958771fbc) passed replay and independent references
before a separate run with recording disabled. The same full-suite command
listed under [API and datatype results](#api-and-datatype-results) selected all 285 cases.

These 16 records add a new GPU/runtime baseline. Existing records are unchanged.
Compared with SM100, linear-attention gate input hashes differ; the other input
hashes match. The numerical reference checks use each run's actual inputs.
Software versions differ between SM100 and SM103, so these are separate
measurements, not a hardware-only comparison.

| SM103 failed case | First mismatch |
| --- | --- |
| `TestQSA::test_replay[bprop-q12-kv1]` | replay 1 dk: 2/43520 bytes differ |
| `TestQSA::test_replay[bprop-q24-kv2]` | replay 1 dk: 3/87040 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q12-kv1]` | batch (1, 0), sequence 0 dq: 1/399360 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 2/66560 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q6-kv1]` | batch (1, 0), sequence 0 dq: 1/199680 bytes differ |
| `TestQSA::test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `TestQSA::test_sparse_batch_invariance[bprop]` | replay 1 dk: 142/1054720 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_mask_replay[all-selected-bprop]` | replay 1 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_replay[empty-rows-bprop]` | replay 3 dk: 1/43520 bytes differ |
| `TestQSA::test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dv: 2/33280 bytes differ |
| `TestQSA::test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestDSA::test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (1, 0), sequence 0 dkv: 4/164864 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]` | replay 5 d_index_k: 1/131072 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 2038/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 989/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132237/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202412/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 513/131072 bytes differ |

### SM107 validation

All 285 L0 cases ran: **244 passed, 24 failed, 17 skipped** in 399.55 seconds.
The 17 skips are SM90-only FP8 cases. All 17 MXFP8 indexer cases, six harness
controls, and 16 operation goldens passed. Failures: 14 QSA, four DSA attention
backward, and six BF16 indexer backward cases.

GPU: `NVIDIA VR NVL72`, SM107, 212 SMs, on ARM64. Torch
`2.14.0a0+b2c75dd062.nv26.09`, CUDA `13.4`, driver API `13050`, cuDNN `92700`,
CuTe DSL `4.8.0`, TVM FFI `0.1.14.post1`. Source snapshot:
`35cf3082e5f3e505b066097f29bf232a848ea269`; imports resolved to that snapshot.
Test and kernel source match the SM103 run. The [SM107 goldens](72e27159dc117014)
passed replay and independent references before a separate full-suite run
with recording disabled. The same command selected all 285 L0 cases.

These 16 records add a new GPU/runtime baseline. Existing records are unchanged.
Some CPU-generated input hashes differ from the x86 runs; references use the
actual inputs. No cross-device byte equality is inferred from separate passes.

| SM107 failed case | First mismatch |
| --- | --- |
| `TestQSA::test_replay[bprop-q12-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `TestQSA::test_replay[bprop-q24-kv2]` | replay 1 dk: 1/87040 bytes differ |
| `TestQSA::test_replay[bprop-q6-kv1]` | replay 1 dv: 2/43520 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q12-kv1]` | batch (0,), sequence 0 dk: 2/33280 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 1/66560 bytes differ |
| `TestQSA::test_batch_invariance[bprop-q6-kv1]` | batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestQSA::test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `TestQSA::test_sparse_batch_invariance[bprop]` | replay 1 dk: 140/1054720 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `TestQSA::test_mask_replay[all-selected-bprop]` | replay 4 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_replay[empty-rows-bprop]` | replay 1 dv: 1/43520 bytes differ |
| `TestQSA::test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestQSA::test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `TestDSA::test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `TestDSA::test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 9/164864 bytes differ |
| `TestDSA::test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 12/164864 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-bf16-dense-h64]` | replay 1 d_index_k: 1/131072 bytes differ |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 1843/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 1083/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132358/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202644/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 1035/131072 bytes differ |

### Indexer failures (SM100)

**DSA BF16 indexer: 40 passed, 6 failed, 0 skipped.** The first mismatch in each failed case is dK.
Sparse large-Top-K cases pass replay and reference, then fail after batch
reorder. Dense backward also has replay and neighbor differences. Sparse dK
reduction assigns query rows to fixed producers; changing batch position can
change the grouping even when replay is stable. Dense backward uses atomic
dK accumulation. These facts explain why a replay pass alone cannot establish
batch invariance. Exact mismatch counts may vary between runs.

| Indexer case | First mismatch |
| --- | --- |
| `TestDSAIndexerBackward::test_replay[dk-fp32-dense-h64]` | replay 1 d_index_k: 2166/262144 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-bf16-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 1/65536 bytes differ |
| `TestDSAIndexerBackward::test_batch_invariance[dk-fp32-dense-h64]` | batch (0, 1), sequence 0 d_index_k: 793/131072 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk896]` | batch (1, 0), sequence 0 d_index_k: 132237/1048576 bytes differ |
| `TestDSAIndexerBackward::test_sparse_persistent_rows[topk2048]` | batch (1, 0), sequence 0 d_index_k: 202412/1048576 bytes differ |
| `TestDSAIndexerBackward::test_neighbor_isolation[dense]` | changed neighbor d_index_k: 1095/131072 bytes differ |

The backward golden cases use one query. Their passes do not establish
bitwise determinism for the multi-query reductions that fail above.

### DSA failures (SM100)

All 24 fprop cases passed. Bprop had 19 passes and five failures, all in batch
invariance. Every replay and numerical-reference check reached in this suite
passed. The first mismatch in each failed case was dKV; forward results and
dQ matched. dSink comparison was not reached after those failures.

| Test | First mismatch |
| --- | --- |
| `test_batch_invariance[h128-d512-k128-bf16-bprop]` | batch (1, 0), sequence 0 dkv: 4/164864 bytes differ |
| `test_query_boundary[h64-q129-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `test_query_boundary[h64-q257-bprop]` | batch (0, 1), sequence 0 dkv: 2/99328 bytes differ |
| `test_query_boundary[h128-q74-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |
| `test_query_boundary[h128-q75-bprop]` | batch (1, 0), sequence 0 dkv: 6/164864 bytes differ |

The H64 failures cross the documented 32/64/128-shard query-count thresholds.
The H128 BF16 failures move the target within the two-CTA path. Same-shape
replay passed: `deterministic=True` did not establish batch invariance for
these cases. No kernel changes, xfails, or relaxed comparisons were added.

### QSA failures (SM100)

The same 14 case IDs failed in the earlier run. First mismatch counts and
tensors can vary with the order of atomic reductions.

| Test | First mismatch |
| --- | --- |
| `test_replay[bprop-q12-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `test_replay[bprop-q24-kv2]` | replay 1 dk: 1/87040 bytes differ |
| `test_replay[bprop-q6-kv1]` | replay 1 dk: 1/43520 bytes differ |
| `test_batch_invariance[bprop-q12-kv1]` | batch (0,), sequence 0 dk: 3/33280 bytes differ |
| `test_batch_invariance[bprop-q24-kv2]` | batch (0,), sequence 0 dk: 1/66560 bytes differ |
| `test_batch_invariance[bprop-q6-kv1]` | batch (0,), sequence 0 dk: 1/33280 bytes differ |
| `test_sparse_batch_invariance[fprop]` | sparse sequence 1 out_fp32: 237/36864 bytes differ |
| `test_sparse_batch_invariance[bprop]` | replay 1 dk: 136/1054720 bytes differ |
| `test_dispatch_boundary[total513-fprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `test_dispatch_boundary[total513-bprop]` | total 513, target position 0 out_fp32: 196/36864 bytes differ |
| `test_mask_replay[all-selected-bprop]` | replay 1 dv: 3/43520 bytes differ |
| `test_mask_replay[empty-rows-bprop]` | replay 1 dk: 1/43520 bytes differ |
| `test_mask_batch_invariance[all-selected-bprop]` | mask all-selected, batch (0,), sequence 0 dv: 3/33280 bytes differ |
| `test_mask_batch_invariance[empty-rows-bprop]` | mask empty-rows, batch (0,), sequence 0 dk: 1/33280 bytes differ |

QSA backward [documents floating-point atomic reductions](../../../docs/operations/sparse_attention/qsa.md).
The dK/dV replay changes are consistent with changing reduction order. Sparse
bprop fails replay before the batch comparison. Both 513-token cases first
fail on saved FP32 output, before gradient comparison; BF16 output matches.
Both 512-token controls pass. All-masked rows, changed-neighbor isolation,
and zero-upstream-gradient QSA cases pass.

### SM90 compact IndexerForward + Top-K validation (2026-11-09)

The new packed-THD ratio-4, D128, 64Q/1KV combined path ran **9 passed, 0
failed, 0 skipped** for BF16 and E4M3 FP8 on SM90 with Torch `2.12.1+cu130`,
CUDA `13.0`, cuDNN `92000`, and CuTe DSL `4.8.1`. Imports resolved to the
tested checkout. The suite covers raw-byte replay under side-stream pressure,
batch reorder/duplication and changed-neighbor isolation, ragged/empty/short
segments, tail rows, exact ties, compile counts, buffer-alias rejection,
nonzero causal offsets, explicit-side-stream internal sizing, and changed-input
CUDA Graph replay.
Independent FP32 references validate selected indices, logits, softmax, and
LSE, including caller-owned LSE-only output. The two existing SM90 Indexer
Forward suites plus this suite ran **41 passed, 66 skipped**. Backward is
outside this combined forward API.

The focused suite command was:

```bash
cd test/python
pytest -q -s deepseek_sparse_attention/cutedsl/test_DSA_indexer_compressed_sm90_determinism.py
```

### SM100 IndexerForward regression (2026-11-09)

The existing compressed IndexerForward + Top-K suite ran **96 passed, 0
failed** on SM100, including BF16/block-scaled numerics, deterministic ties,
compile-cache reuse, preallocated buffers, and CUDA Graph cases. The general
IndexerForward suite ran **95 passed, 3 skipped**. The tested checkout used
Torch `2.14.1+cu130`, CUDA `13.0`, cuDNN `92400`, and CuTe DSL `4.8.1`.

## Method

- [Megatron module tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/unit_tests/determinism): fixed inputs, restored state, outputs and gradients.
- [Megatron kernel replay PR](https://github.com/NVIDIA/Megatron-LM/pull/7148): raw-byte comparison, side-stream contention, negative controls, and coverage review.
- [Nemotron functional tests](https://github.com/NVIDIA/Megatron-LM/tree/main/tests/functional_tests/test_cases/nemotron): deterministic execution and saved regression values.
- Internal recipe CI (not linked from this public repository): follow-up integration target; its contents were not accessible during this change.

### Hopper compact H32 sparse indexer backward

Verified after the rebase on 2026-11-09 with NVIDIA L20X SM90 (132 SMs),
PyTorch 2.12.1+cu130, cuDNN 9.20.0, and CuTe DSL 4.8.0 compiling with the
CUDA 13.4 flavor. The checkout was the exact pushed commit `f4540d88` on
develop parent `7a68feff`. `cudnn`, the API module, and the SM90 kernel all
resolved from `/tmp/cudnn-fe-h32-f4540d88/python`, reporting frontend version
1.33.0; the host's installed frontend 1.31.0 was not used.

The L0 `test_sm90_deterministic_h32_topk512` cases cover BF16 and FP32 dK,
sparse negative IDs, replay, short five-query batch reordering with global
IDs, and an independent gradient reference. Strict replay against the
committed runtime key `70c200b13182bbc2` passed 2/2 cases in 7.63 seconds. The
L0 support-envelope cells passed 24/24 in 0.38 seconds. Four H32 L0 stress
cases passed in 15.25 seconds, covering Top-K 384/512/2048, both deterministic
dW folds, BF16 and FP32 dK, local and global IDs, and 194 query rows over 96
producer planes. Those stress cases check fixed-layout replay and the
independent numerical reference; they do not compare batch reordering across
the 96-plane producer boundary. Because the producer-plane assignment depends
on the flattened batch row after that boundary, SM90 H32 is documented as
Replay only rather than batch-invariant. The complete indexer-backward L0 file
finished with 43 passed, 20 expected SM100-only skips, and 63 deselected in
32.64 seconds.

The RED control used parent `7a68feff` with the new test file but without the
implementation. Both H32 core cases failed at admission with
`SM90 IndexerBackward requires H=64, got H=32`, confirming that the green run
exercised this change. Commands run with `PYTHONPATH` pinned to the checkout:

```bash
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k sm90_deterministic_h32_topk512 -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k check_support_envelope -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -k 'deterministic_reproducible_and_correct and h32' -q -rs
python -m pytest deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py \
  -m L0 -q -rs
```

The CUDA 12 compiler flavor was unavailable and remains unverified.

### 2026-11-09: native SM107 per-tensor FP8 paged D512

`TestNativeD512Fp8Paged` in `sdpa/frost/test_sdpa_fp8_sm107.py` exercises
the native SM107 role-split kernel and the shared prepared binding. The
implementation was based on `fc4490c88557` (the SM100/SM103 paging addition),
with matching FE 1.33.0 bindings rebuilt for Python 3.12.12. Runtime:
Torch 2.14.1+cu130, cuDNN 9.30.0.2, CuTe DSL 4.8.0 (CUDA 13.4 compiler).
The imported package was the isolated qualification source, not an editable
installation of another checkout.

Coverage includes E4M3/E5M2 inputs, BF16/FP16/scalar-FP8 outputs, FLOAT and
HALF softmax, page8/16/32/64/128/256, independent K/V page tables and pool
layouts, runtime strides, fresh buffers, changed-page/scale capture replay,
empty and partial pages, causal/windows/sinks, split1/4, whole-group packing,
and the aligned D384/V320 envelope with poisoned storage padding. Capability
checks retain the smaller FP8 flavors', half/MXFP8 D512 and THD-Q declines.

Six golden records in `0972294c7bd4a660/` cover four FLOAT and two HALF
softmax configurations. Quantized payloads and scale bytes stay fixed across
sequences alone, reordered, duplicated and mixed with changed neighbors,
including 129-request multi-wave batches. O/Stats are compared as raw bytes
per sequence; whole-batch Amax is compared on identical repeated batches and
against the independent numerical reference, not misclassified as a per-row
output. Inputs and prepared owners are checked for unintended mutation.

Commands from the isolated `test/python` directory:

```bash
# After independent-reference/replay qualification, record into a fresh directory:
CUDNN_DETERMINISM_RECORD=/path/to/new-records python -m pytest \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Paged -k golden -q
# Inspect/copy the six JSON records, then leave RECORD unset:
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" python -m pytest \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD512Fp8Paged -q
```

Measured: 31 feature/strict-golden cases passed, with no failures or skips.
The existing native D512 Pack/Split suites additionally passed 32 selected
FLOAT-softmax controls (`-k 'not rebind_and_replay and not half_adapter'`).
One earlier HALF-adapter control reached its reference's FP64 `torch.exp2`
and failed because the installed NVRTC rejected the SM107 target; this
reference-environment failure is retained, not counted as a kernel pass.
The new HALF paging, replay and batch-invariance checks pass independently.

Compute Sanitizer 13.4.92: eight boundary/head-envelope cases under memcheck,
the same eight under synccheck, and two page8/page128 persistent cases under
racecheck; no memory/synchronization errors or race hazards. The older
13.2 sanitizer reported an unsupported device and is not a qualifying run.
No backward capability or other architecture's status is inferred from this
SM107 execution. No heuristics ranking changed.

The paged launch's generated SM107 cubin was also checked with nvdisasm
13.5.37: exactly one `USETMAXREG.DEALLOC` at 40 registers and one
`USETMAXREG.TRY_ALLOC` at 240. All four service warps release before role
branching. Dense launch/register behavior is unchanged.

### Native SM107 half D128 dense-Q paging (2026-11-10)

The architecture entry `test_sdpa_fwd_native_paged_sm107.py::TestNativeD128DensePaged`
collects the shared `native_d128_dense_paged_determinism.py` checks. Six
standalone cases pin native D128, CGA1/2, N64/N128 and split1/4, with FP16/BF16
Q/O and matching half or E4M3/E5M2 K/V. Each case checks fixed CPU-seeded inputs,
fresh buffers, queued independent work, reordered/duplicated batches and129
requests, then records O and FP32 Stats only after the independent CPU-double
reference and byte comparisons pass. Changed-input CUDA capture mutates Q,
Q/KV lengths and independent K/V table contents. Poisoned output/workspace,
output-padding canaries and input-byte preservation are checked. K/V physical
layouts differ; table strides match but are non-contiguous. Equal pool
capacities and matching table strides retain the existing half binder contract.

Graph tests separately cover partial GQA, page8–1024, short and prefill-shaped
queries, all-empty KV, head envelopes, sinks/windows and changed-input replay.
The new mixed-KV graph checks use the same independent double reference on CPU:
this runtime's Torch eager outer-product specialization invokes an older Triton
ptxas that does not recognize SM107. The native kernel remains on the GPU; no
kernel output, test tolerance or admission is substituted to accommodate it.

Runtime: SM107216SM, driver630.10, Python3.12.12, Torch2.14.1+cu130,
cuDNN9.30.0.2, CuTeDSL4.8.0/CUDA13.4 compiler, matching rebuilt FE1.33 bindings.
Final strict suite:61 passed, zero failures/skips, including all six goldens.
Related controls:23 passed (metadata and existing THD N64), plus24 L1 native
dense packing/split replay cases. Compute Sanitizer13.4.92: memcheck7,
synccheck7 and racecheck4 passed, zero diagnostics. The final published tests
also pass the smallest D32 head envelope boundary separately (1 case).
Other GPU results are not inferred from this native SM107 qualification.

### Native SM107 half D64 dense/THD paging (2026-11-10)

`test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged` qualifies physical D64
operands/MMA/output on the native D128-family kernel file. Parent D64 dense
paging selects the D128 envelope; parent THD D64 paging is declined. The new
route supports matching FP16/BF16 Q/K/V/O, CGA1/2, packed/unpacked execution,
N64/N128 and dense split-KV. Paged THD remains unsplit; mixed FP8 KV stays D128.

Eight fixed-CPU-payload goldens cover four dense and four compact-THD plans.
The dense checks reuse the D128 fixture at D64; THD additionally changes the
target sequence's packed origin across unrelated neighbors. Both verify
independent CPU-double references, O/Stats byte identity with fresh buffers,
queued stream work, repeated/reordered/duplicated batches through129 requests,
input preservation and untouched capacity/padding. Separate numerical graph
checks change page tables, lengths and tensor bindings under CUDA capture.

The current-parent suite passes105 cases without failures/skips:56 new D64
checks (32 dense,8 THD origins,8 goldens,7 boundaries,1 admission),46 D128
controls with unchanged golden bytes and3 architecture metadata controls.
Pages8–1024, Q1/2/17/129/257, partial head envelopes, empty/tail KV, asymmetric
pool layouts, partial GQA, windows and sinks are included. The existing THD
stage assertion is corrected for D64's fixed four-stage K/V ring.

Runtime: SM107216SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
DSL4.8.0/CUDA13.4 and matching FE1.33 bindings. Goldens use runtime key
`0972294c7bd4a660`. No qualification of another architecture is implied.

Compute Sanitizer13.4.92:7 page-boundary memcheck cases,8 THD synccheck cases
and8 THD racecheck cases pass with zero diagnostics. No backward path changes.

Additional nonpaged D64 controls pass30 cases:22 dense split/replay and8
independent THD-origin cases.


### Native SM107 half D512 dense-Q paging (2026-11-10)

`test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged` pins the native SM107
2x2 kernel file, two-CTA MMA, twin-multicast K/V column shares and descriptor
version1. The new page-box loader preserves three-stage K/V rings and pair
transaction credits. Dense FP16/BF16 Q/K/V/O supports partial PackGQA and
split-KV; THD paging and mixed FP8 KV remain declined.

Four goldens cover FP16/BF16 and split1/4 at fixed CGA2/N128. The shared paging
fixture at D512 verifies independent CPU-double O/Stats references, raw bytes
under replay and queued stream work, fresh buffers, reordered/duplicated
batches through129 requests, input preservation, output-padding canaries and
changed Q/length/table contents under capture. Separate graph checks use
independent K/V table contents and HND/NHD layouts, including mixed pools.

The strict feature suite passes21 cases with no failures/skips:8 numerical
rebind/capture cases,8 boundaries,4 goldens and1 graph/config admission check.
Boundaries include page8–1024, Q1/2/4/17/129/257, empty/tail KV, windows/sinks,
partial GQA and D504/V480 and D264/V256 envelopes. The first boundary attempt
requested LPT for a plan whose existing domain is NATURAL-only; the final
suite pins the supported scheduler without widening its domain.

Runtime: SM107216SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
DSL4.8.0/CUDA13.4 and matching FE1.33 bindings. Goldens use runtime key
`0972294c7bd4a660`. No new backward or other-GPU qualification is implied.

Existing nonpaged native 2x2 controls pass16 cases:8 dense graph cases,4
sink/Stats cases and4 THD cases. Compute Sanitizer13.4.92 passes8 page-boundary
memcheck cases,8 dense synccheck cases and8 dense racecheck cases, with zero
errors/hazards. Commands from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged -q -x`; sanitizer
runs select `test_boundaries` (memcheck) or `test_dense` (synccheck/racecheck)
from the same class with `--error-exitcode 99`.


### Native SM107 half D576/V512 dense-Q paging (2026-11-10)

`test_sdpa_fwd_native_paged_sm107.py::TestNativeD576Half` pins the native
2x2 body with three192-column QK chunks, one CTA pair and three-stage rings.
The complete P ring requires descriptor version1 even though its first slot
starts below256KiB. Exact FP16/BF16 D576/V512 serves dense or paged K/V,
partial PackGQA and dense split-KV; THD and mixed FP8 KV remain declined.

Four new strict goldens cover paged FP16/BF16 and split1/4 at CGA2/N128.
The shared fixture gains an independent V width; prior defaults and golden
names are unchanged. Independent CPU-double O/Stats references, fresh-buffer
raw-byte replay with queued work, changed table/length/Q graph replay,
reordered and duplicated batches through129 requests, input preservation and
output-padding canaries are checked. Nonpaged coverage separately checks
pre-folded scale, sink/window semantics and changed-input capture.

Runtime: SM107208SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
CuTeDSL4.8.0/CUDA13.4, matching FE1.33 bindings. Runtime key2916acaa3402817b.
The initial28-case suite passes with no failures/skips, alongside39 existing
2x2 structural checks and16 nonpaged D512 dense/THD/sink controls. Additional
shared-record MLA checks and final sanitizer results are recorded below.

Initial qualification caught a too-narrow descriptor guard checking only
the P-ring start; it now agrees with the config's full-ring bound. Two harness
errors were corrected without loosening numerical/byte comparators: record
and comparison directories were initially set together, and the D576 metadata
probe initially omitted the required operand descriptors. Original failures
remain in the private qualification report. No backward or other-GPU claim.


Final qualification passes31 cases, including three actual absorbed-MLA
shared-record checks (V aliases the first512 columns of each576-wide K row),
G96/G128 and ordinary Q1/Q2/Q16 candidates. Compute Sanitizer13.4.92 reports
zero diagnostics:7 page-boundary memcheck cases,3 shared-record memcheck cases,
8 paging synccheck cases and8 paging racecheck cases. Run from `test/python`:
`python -m pytest sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD576Half -q -x`.
Sanitizers select `test_boundaries`, `test_shared_mla_record` or `test_paged`
from that same class with `--error-exitcode 99`. Pre-commit passes; Fern has
zero errors and its existing unauthenticated-redirect warning.

### D512 per-tensor FP8 backward — 2026-11-10

`sdpa/frost/test_sdpa_bwd_d512_fp8.py` covers graph numerical references and
raw-byte replay/batch invariance for E4M3, FP16 and BF16 gradients. Dense
BSHD E4M3 inputs use independent Q/K/V/O/dO tensor scales, left windows,
GQA and tail tiles. Three golden cases per runtime preserve the quantized
inputs while running alone, reordered, duplicated and in batches through129.
All gradients and four Amax outputs are compared; changed-input capture and
sync-debug execution run on the same plans. Nine numerical cases and three
recording cases pass on SM100(148SM, cuDNN9.28.0.38, Torch2.14.1+cu130,
DSL4.8.0/CUDA12.9) and SM107(208SM, cuDNN9.30.0.2, same Torch,
DSL4.8.0/CUDA13.4). The 19-case strict suite passes on each runtime,
including all six committed goldens. SM100 additionally passes ten existing
D192 controls with the shared delta/prepared host. On both SM100 and SM107,
all nine numerical cases pass memcheck, synccheck and racecheck with zero
errors or hazards. SM103 compiles all three gradient types but is unverified
at execution. The existing SM107 CI entry exposes the same shared checks as
`test_sdpa_fp8_sm107.py::TestD512Fp8Backward`; the new D512 entry runs on
SM100/SM103 only, so directory sweeps do not duplicate the Rubin cases. Original16k-SWA geometry has a full structured oracle
on SM107; it is not a full random-matrix oracle or a performance result.

Review closeout: the SM107 D512 FP8 class moved into the actual selected
`test_sdpa_fp8_sm107.py` entry; direct collection of the former backward file
was not CI inclusion. The new kernel now carries its required cuDNN prefix.

### D512 MXFP8 backward — 2026-11-10

`TestD512Mxfp8Backward` is collected by the SM100 MXFP8 entry and the
explicitly selected SM107 `test_sdpa_fp8_sm107.py` entry. The shared `d512_mxfp8_bwd_checks.py` has six independent
reference cases, seven graph declines and two raw-byte golden/batch cases.
CPU-seeded inputs are quantized once; SF bytes are reordered in their canonical
rowwise/columnwise layout with the corresponding payloads. Singletons,
reordered/duplicated batches through129, fresh buffers, concurrent work,
repeat launches, sync-debug execution and captured dO/SF mutations are checked.
Neither the MXFP8 reference nor its cosine tolerance is changed.

Both SM100 and SM107 pass all15 recording cases. Two goldens per runtime cover
FP16/BF16 gradients under the same runtime keys as the D512 per-tensor FP8
qualification above. SM103 execution is unverified. ExactD512 MHA/GQA/SWA uses
only FP32 row-delta scratch and reads canonical scales without repacking.

Final strict checks:15 pass on SM107;39 pass on SM100 comprising15 MXFP8,
19 per-tensor D512 FP8 (all prior golden bytes unchanged) and5 existing
D128/D192 MXFP8 controls. On each architecture, all6 new numerical cases
pass memcheck, synccheck and racecheck without errors or hazards. SM103
compiles both gradient types; no execution result is inferred. Original
B2/H64:1/S16384/window128 passes a full structured output oracle and changed-
input capture on SM107 with8,388,848bytes of scratch. This is not a full
random-matrix oracle or a performance result.

Review closeout on SM107: the selected `test_sdpa_fp8_sm107.py` entry passes
all 15 MXFP8 and 19 FP8 strict cases, plus the three kernel-prefix guards
(37 total). Both new backward kernels have cuDNN symbol prefixes. Arithmetic
and all existing golden output bytes remain unchanged.

### SM107 D64 mixed FP8 K/V — 2026-11-10

`test_sdpa_fp8_sm107.py::TestMixedFp8Kv` now exposes the exact-D64 mixed
storage checks in the selected architecture entry. FP16/BF16 Q/O use E4M3/E5M2
K/V directly. The native physical-64 prefill specialization serves dense and
paged KV, CGA1/2, dense splits and unsplit THD; mixed THD splits decline.

On SM107 with cuDNN 9.30.0.2, Torch 2.14.1+cu130 and DSL 4.8.0/CUDA 13.4,
90 strict feature/resource/control cases pass, including eight new dense/THD
runtime goldens and 16 existing D128 mixed-paged controls. Eight additional
Q=1 dense/paged graph cases pass for all four Q/KV dtype combinations. Memcheck,
synccheck and racecheck each pass 13 selected cases with zero errors/hazards.
The public architecture entry collects all new cases. Other architectures keep
their existing mixed-storage support; no additional GPU execution is claimed.

CPU-fixed, once-quantized payloads are retained across alone/reordered/duplicated
batches through 129 requests. Each case checks all O/Stats bytes, fresh-buffer
replays with queued concurrent work, input preservation and an independent
CPU-f64 numerical reference. THD changes packed origins and checks untouched
capacity tails. Graph tests separately check independent physical offsets,
poisoned gaps, independent HND/NHD tables and changed-input capture. Dense
standalone tests also exercise the native split path. The initial admission
fixture retained dense Stats while synthesizing THD facts and therefore hit an
earlier padded-Stats decline; correcting that fixture required no kernel or
numerical-tolerance change. All eight goldens pass strict comparison.

The shared standalone driver now forwards and asserts its requested split count.
All eight D64 standalone split cases pass with the actual split3 plan; the earlier
run had exercised unsplit standalone execution despite its parameter labels.
Dense graph splits and the eight standalone replay goldens were already passed
with their requested split counts and remain unchanged.


CI-route correction: the pinned SM107 pipeline selects `test_sdpa_fp8_sm107.py`,
not `test_sdpa_fwd_dsl_sm107.py`. The shared mixed-KV class now lives in the
selected file, with explicit L0/DSL/SM107 guards and no duplicate collector.
The GPU counts above are direct qualification, not completed CI results.

### SM80 native THD sink forward

The native adapter passes sink pointers and head strides through the existing
THD host for dense and paged K/V. `test_sdpa_fwd_paged_sm80.py` covers FP16/BF16,
strided sink declarations with flat graph carriers, HND/NHD independent pools,
new storage, dynamic lengths/origins and changed-sink CUDA Graph replay. Invalid
sink bindings are checked before output or metadata writes.

The determinism cases use CPU-generated fixed operands and an independent CPU
FP64 sink reference, repeat under concurrent scheduling work, compare target
sequences alone, with neighbors, reordered and duplicated, and mutate neighbor
pages under capture. On the A100 qualification environment (exact runtime
identifiers retained in private evidence),
the original96cases pass:64existing controls,24finite-sink cases covering
D64/D256/D576x512 and eight strict dense/paged runtime goldens. An additional12endpoint cases cover disabled(-inf) and dominating(+inf) sinks
with live Q but zero KV iterations, graph/standalone and captured replay. The
new regression fails before the kernel fix; all108cases pass afterward, with
unchanged finite-input golden files. The four initial paged and eight expanded
dense/paged golden-recording cases passed before the original strict run. These are SM80 execution results only; no other GPU is claimed.

The initial finite-sink qualification passed eight D64 cases per sanitizer.
After the endpoint fix, memcheck, racecheck and synccheck each pass four
endpoint cases across D64/D256/D576 and graph/standalone with zero findings. These runs use
driver-compatible task-local CUDA bindings. The initial environment
reported API-version query errors before attention; an
exact-source create/destroy-handle control reproduced them. That initial run
is not counted as a sanitizer pass. No suppression is used in the clean runs.

Commands from `test/python`: `python -m pytest sdpa/frost/test_sdpa_fwd_paged_sm80.py -m L0 -q -x`
with `CUDNN_DETERMINISM_GOLDENS` pointing to the committed golden directory.
The endpoint sanitizer selection is `sink_endpoints and dtype0 and (False or dims0)`
on the same file, with `--error-exitcode 91`. Pre-commit passes; Fern has zero errors and
one existing redirect warning.

### SM120 small-head paged forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_width_determinism`
records eight FP16/BF16 goldens for D64/V64 and D120/V88, packed and unpacked.
CPU-seeded operands and a separate CPU-double oracle cover O and Stats, repeated
raw bytes with queued work, reordered/duplicated batches through129 requests,
and changed-neighbor CUDA Graph replay. Padded head columns and pool rows are
poisoned; output gaps and strided Stats retain their canaries.

Runtime key42742425a3d29d7c identifies the RTX5080(SM120,84SM)
qualification environment; exact toolchain and driver details are retained
in private qualification evidence.
The full paging file passes143 cases with strict goldens; the pre-existing
40GiB physical-wide-pool case is skipped on this16GiB GPU. This does not
qualify wide-address execution, SM121 or RTXPRO6000 performance.

Twelve cases spanning D24/V40,D64/V64,D120/V128, both half types, Q1,
packed GQA, sinks and windows pass each of memcheck, racecheck and synccheck
with zero findings. These runs use driver-compatible task-local CUDA bindings.
The original environment and a handle-only control report the same14
API-query errors; their exact compatibility diagnosis is retained privately.
Those initial runs are not counted as sanitizer passes.
Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fwd_paged_sm120.py -m L0 -q -x`; the sanitizer selection
is `-k 'widths and (dims6 or dims2 or dims10) and (decode or packed_sink)'`.

### SM120 paged decode and split-KV — 2026-11-10

Eight new records at key42742425a3d29d7c cover FP16/BF16 D64/V64 on M16
and D120/V88 on M32, packed, split1/3. The preceding small-head paging runtime
applies. CPU-seeded inputs, a CPU-double O/Stats oracle, raw-byte repeats with
concurrent work, reordered/duplicated batches through129 requests and
changed-neighbor capture all pass before recording. All eight compare strictly
alongside the eight original prefill records; the existing records are unchanged.
The split count and tile shape remain fixed for each batch-invariance check.

Feature coverage includes direct K/V pages, all16/32/48/64-key stage widths,
D8 through128, independent HND/NHD layouts, poison/canaries, GQA6 partial
packing, sink endpoints, device Q/KV lengths and strided natural/log2 Stats.
Twenty existing nonpaged numerical/capture controls pass, including the newly
admitted48-key dense stage for unequal QK/V widths. Target execution is SM120;
SM121 remains unverified. Sanitizer and multi-wrap completion are recorded below.

Final qualification:54 feature cases,17 strict original/new golden and scope
checks,20 nonpaged numerical/capture controls and7 ring/domain checks pass.
Each sanitizer reports zero findings on8 compact cases plus2 cases whose
per-split stage count exceeds the physical ring size. The latter force slot
reuse for both M16 and M32 consumer groups; four larger numerical cases also
change page contents/lengths under capture after several ring wraps.
Original GPT-OSS Q1/KV131072/D64 full and SWA128 graphs execute through ordinary
OPENSOURCE candidates (M16/N32, split8 and split1 respectively), without a
candidate or ranking patch. The bounded timing run is not performance qualification.

### Native SM120 wide paged decode — 2026-11-10

`test_sm120_frost_paged_wide_decode_determinism` adds eight records for
D256/V256 on M16 and D192/V128 on M32, both half dtypes and split1/3.
The existing independent CPU-double oracle, CPU-seeded inputs, eager/captured
byte checks, queued work and reordered/duplicated batches through129 all pass
before recording. Runtime key42742425a3d29d7c identifies the same RTX5080
qualification environment as the small-head records; exact runtime details
remain in private evidence. Existing goldens are unchanged.

Fifty-six feature cases cover independent eight-aligned QK/V widths through256,
including D136/V8 and D8/V192, plus asymmetric D248/V256 and D256/V248 tails.
Two compact multi-wrap cases verify that every producer reuses ring slots;
one host admission detector retains the prefill128 and decode256 boundaries.
The wide tests explicitly request legal native plans because ordinary
candidate generation does not propose decode tiles for every longer Q.
This qualifies native capability, not that candidate ranking or performance.
SM121 remains unexecuted. Memcheck, racecheck and synccheck each pass10cases
with zero findings, including the two multi-wrap cases. Reproduce with
`-k 'wide_decode_ring_pipeline or (wide_decode and not determinism and
(dims0 or dims6) and dtype0)'` on `test_sdpa_fwd_paged_sm120.py`.

### Native SM107 D256 mixed FP8 KV — 2026-11-10

`sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv` exposes D256 checks
through the explicitly selected architecture CI entry. FP16/BF16 Q/O with
E4M3/E5M2 KV use the native two-CTA D256 prefill pipeline, dense Q and no
packing. Dense/paged KV, Q=1 and split1/3/4 are covered. Four committed
goldens preserve CPU-fixed quantized inputs across fresh-buffer replay,
queued work, reordered/duplicated batches through129, independent HND/NHD
pools, zero-length sequences, sinks, bottom-right/window masks and changed
capture replay. Independent CPU-double O/Stats references and input/output
padding canaries accompany raw-byte comparisons.

SM107208SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
CuTeDSL4.8.0/CUDA13.4, runtime key2916acaa3402817b:44 feature/resource/golden
checks pass strictly, with no failures/skips; six existing uniform-half D256
gate/THD controls also pass. Compute Sanitizer13.4.92 memcheck, racecheck and
synccheck each pass eight dense/paged split/unsplit cases with zero diagnostics.
Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv -m L0 -q -x
-k 'd256 or native_resources'`. Sanitizer runs select
`test_d256_dense_paged and (qdtype0 or qdtype3)` with `--error-exitcode 91`.
Two additional host admission checks cover unsupported combinations.

Qualification found a paged split ABI mismatch: the adapter excluded paging
from its FP32-partial predicate while the native host wrote FP32. The fix
retains the native partial format for both layouts; no numerical comparator
or tolerance changed. Earlier failing probes remain in the private report.
Forward only; other GPU mixed-D256 routes are not admitted.


### Native SM107 D512 mixed FP8 KV — 2026-11-10

The selected `test_sdpa_fp8_sm107.py::TestMixedFp8Kv` entry exposes exact
D512/V512 mixed-storage checks. FP16/BF16 Q/O and E4M3/E5M2 K/V use one
native CTA pair, with dense/paged KV, packed/unpacked groups and splits.
Four fixed-input goldens cover both Q and KV types, split1/4 and packing;
raw O/Stats bytes are compared under fresh-buffer replay, queued work,
reordered/duplicated batches through129 and changed-input graph replay.
Independent CPU-double references, empty rows, bottom-right/window/sink
semantics, independent HND/NHD pools and capacity canaries are retained.
Q1/Q16 at H128:1 additionally checks partial packing of a full MQA group.

SM107216SM, driver630.10, cuDNN9.30.0.2, Torch2.14.1+cu130,
CuTeDSL4.8.0/CUDA13.4, runtime key0972294c7bd4a660:46 feature/admission/golden
recording cases pass, with no failures/skips. The initial probe stopped at a
configuration guard because the generic record retained d_flavor128; the
loader now normalizes exact mixed D512 before selecting its native2x2 body.
No numerical comparator changed. Strict comparison, original uniform-half
controls and sanitizer results are recorded below after completion.

Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv -m L0 -q -x -k d512`.
Forward only; mixed D512 admission on other GPUs is unchanged.


Final qualification repeats46 strict cases against all four recorded goldens.
Twenty existing uniform-half D512/D576 paging and shared-MLA controls pass.
The structural suite passes40 cases; its one SM100-only config case is skipped
by that file's architecture marker and checked separately as host-only assertions.
Compute Sanitizer13.4.92 memcheck, racecheck and synccheck each pass eight
representative dense/paged split/unsplit cases with zero errors/hazards.
Sanitizer selection is `test_d512_dense_paged and (qdtype0 or qdtype3)` with
`--error-exitcode 91`. No SM100 kernel execution or mixed-D576 support is claimed.


### Native SM80 split-KV forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_split_sm80_determinism.py` qualifies FP16/BF16
D128 with split 3 and D512 with split 17 on A100. Four new fixed-input goldens
compare O and FP32 Stats byte for byte under fresh-buffer replay with concurrent
work, reordered/duplicated batches through 129, empty neighbors and
changed-neighbor CUDA Graph replay. Independent CPU-double references and
input-preservation checks pass before recording. Split count is fixed within
each comparison; equality across split counts is not claimed.

Runtime key `a29ca37b88513cb6` identifies the A100 qualification environment;
complete runtime details are retained in private evidence. Four recording
cases pass, followed by 112 strict cases (the 108 existing SM80 paging/sink
cases plus the four new goldens). The final source then passes 36 numerical
split cases plus the four strict goldens, with no failures or skips. These
cover dense/paged KV, both half types, D64 through D576/V512, graph and
standalone entry points, strided outputs, sink endpoints, padding, optional
Stats, both Stats log bases, poisoned partials and changed bindings. Two
additional split cases reload compiled artifacts with compilation forbidden.
No existing golden was regenerated.

Compute Sanitizer memcheck, racecheck and synccheck each pass eight cases
with zero errors/hazards: paged D64/D256/D576 in graph and standalone modes,
and dense D512 with absent or base-two Stats. The selection is
`-k '(paged_split_rebind and dtype0 and (dims0 or dims3 or dims5)) or
(dense_split_partial_initialization and dtype1 and dims1)'`, with
`--error-exitcode 91`. The unchanged common combine separately passes 12
infinite-sink endpoint controls on SM100; its production source is unchanged.

Run from `test/python`: `CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens python -m pytest
sdpa/frost/test_sdpa_fwd_paged_sm80.py
sdpa/frost/test_sdpa_fwd_split_sm80.py
sdpa/frost/test_sdpa_fwd_split_sm80_determinism.py -m L0 -n 0 -q`.
Forward only; THD split and packing remain unsupported. This does not
qualify another GPU, default plan ranking, or serving performance.


### Native SM80 small query/KV tiles — 2026-11-10

`sdpa/frost/test_sdpa_fwd_small_tiles_sm80.py` exercises explicit M16/M32
query tiles and N16/N32 KV stages without changing default candidate policy.
Eight new runtime goldens cover D64 M16/N16, D128 M16/N32 and M32/N16,
and D256 M16/N16 in both half types, all at split 3. The reused split
qualification preserves CPU-fixed inputs through fresh-buffer replay with
concurrent work, reordered/duplicated batches through 129 and changed-neighbor
CUDA Graph replay. O and FP32 Stats pass independent CPU-double references
before their raw bytes are recorded. No old golden is regenerated.

A100, runtime key `a29ca37b88513cb6`: 64 numerical feature cases and eight
recording cases pass. Features include dense/THD queries, independent paged
K/V, dense-Q splitting, eight-aligned head envelopes, sinks including infinity
endpoints, padded/empty rows, strided outputs, pointer rebinding and capture.
The host geometry suite passes 96 cases while leaving default candidate
choices implicit. These are forward-only results on this GPU/runtime.

The new D192/M128/N16 case exposed a copy-wave divisibility restriction.
The SM80 loader now chooses a whole subset of CTA threads for each operand,
so dense and paged vector copies still fill complete tiles. Existing tile
geometries retain all their copy threads. The first 64-case attempt stopped
after five passes at this compile-time rejection. A subsequent DSL-control-flow
error was resolved by keeping the static configuration diagnostic outside
the dynamic lane guard; both failed attempts remain in private evidence.
The final 64 cases pass after that repair.

Run from `test/python`: `python -m pytest
sdpa/frost/test_sdpa_fwd_small_tiles_sm80.py -m L0 -n 0 -q` with
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` for strict comparison.
The standard golden workflow remains required on a different runtime.


The strict follow-up passes 120 cases: 108 existing paging/sink cases,
four original split goldens and eight new small-tile goldens. Memcheck,
racecheck and synccheck each pass 12 cases with zero errors/hazards,
covering standalone/graph and dense/THD query forms over paged KV at
D48/V32 M16/N16, D192/V128 M128/N16 and D256 M16/N16.
Sanitizer selection is `-k 'paged_rebind and dtype0 and
(48-32-16-16 or 192-128-128-16 or 256-256-16-16)'` with
`--error-exitcode 91`. Runtime details and initial failures remain in
private evidence; no unsupported GPU or default performance claim is made.

Two additional dense D192/V128 M16/N16 and M128/N16 graph cases pass
numerical, fresh-binding and replay checks. Each also passes memcheck,
racecheck and synccheck with zero diagnostics, bringing the sanitizer scope
to 14 cases per tool. These controls exercise the nonpaged copy path with
all CTA threads and with the reduced copy-thread subset.


### Native SM80 PackGQA forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_pack_gqa_sm80.py` qualifies native FP16/BF16
packed-head forward on A100. Ten new CPU-seeded goldens cover D64/group3
unsplit M16/N16, D128/group12 split3 M32/N32, D128/group128 split3 M16/N16,
D256/group8 unsplit M16/N32 and D512/group3 split17 M32/N16. These include
head groups crossing tile boundaries. Independent CPU FP64 references, fresh
buffers and concurrent-work replay pass before recording. O and FP32 Stats
are byte-compared alone, in reordered/duplicated batches through129 and under
changed-neighbor CUDA Graph replay. Runtime key `a29ca37b88513cb6` identifies
the qualification environment; exact runtime details remain in private evidence.
No old golden was regenerated. This is forward-only, with fixed knobs and
runtime; THD PackGQA remains unsupported.

The native feature run passes64paged graph/standalone cases across both half
types, split/unsplit, independent HND/NHD pools, masks, sink endpoints,
strided canaries, fresh pointers and changed device lengths under capture.
The route assertion observes the packed head count in the compiled template.
Eight dense split partial-initialization cases and six dense bias/sink replay
cases pass. The first launch attempt lacked the isolated worktree's compiled
Python extension and stopped during import; no kernel executed in that attempt.

From `test/python`, run `python -m pytest -n 0 -m L0 -q
sdpa/frost/test_sdpa_fwd_pack_gqa_sm80.py`, with
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` for strict comparisons. Other runtimes
require the usual separate golden-recording workflow.


The strict follow-up passes132cases:108existing paging/sink cases,
four original split goldens, eight small-tile goldens, ten new PackGQA goldens,
and two packed split artifact-reload cases with compilation forbidden on the
second plan. All108host geometry/admission cases pass, plus the direct THD
PackGQA rejection. Memcheck passes18cases with zero errors. The initial unfiltered racecheck was interrupted after five passes because
instrumenting the independent Torch reference dominated runtime; its log is
retained. Final racecheck and synccheck each pass22cases with zero errors or
warnings, retaining all18original cases and adding four G128 single-query
cases. These runs use `--kernel-name kns=cudnn_kernel_` to check every native
producer/combine and exclude reference kernels. A full launched-name audit
confirms the producer and combine are included; the first audit hit the
100-message printing limit, so the completed audit uses `--print-limit 0`.
The standalone audit passes one additional graph/capture case. No sanitizer
finding was suppressed or converted into a pass.

### SM90 D512 dense split and partial PackGQA forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_split_sm90_determinism.py` adds eight L0
forward-only cases: FP16/BF16, split3 unpacked and split17 full PackGQA,
plus split3 partial groups G12/G128. CPU-seeded inputs retain an independent
FP64 numerical reference. Fixed-plan checks cover fresh-buffer replay,
concurrent work, B129 reorder/duplicates/empty neighbors and changed-neighbor
CUDA Graph replay. Eight H100 records are appended under runtime key
`6521230c64ce3d35`; existing goldens are unchanged. No equality across split
counts, packing choices or runtimes is claimed.

The final combined run covers 449 cases: all 57 split/partial-packing
features, eight strict goldens, existing D512 paths and the newly landed D64
route. Initially 434 pass; one old error-message expectation and 14 child
compilation probes fail. The probes imported an unrelated editable install;
they now remove that finder, bind the parent's package directory and assert
the source path. Twelve D64 cross-compilation probes pass on A100 (compiler
checks only), then two D512 probes plus twelve template-validation cases pass
on H100. Thus all 15 initial failures are addressed without repeating the
unchanged native numerical suite. Eighteen existing singleton-stride tests
also exposed a recorder/API/spec reference cycle; retaining only frame
metadata removes all eighteen shutdown leak diagnostics, with eighteen
focused passes before and after the test-only correction.

On H100, unfiltered memcheck and native-producer/combine-filtered racecheck
and synccheck each pass 21 cases with zero diagnostics. The selection is
`(dtype0 and (multiple_kv or (dense_split_partial and dims0) or
partial_pack_multiple)) or partial_pack_graph`, with `--error-exitcode 91`.
The latter tools use `--kernel-name kns=cudnn_kernel_`; independent framework
reference kernels are excluded, while the attention producer and shared
combine are included. An earlier racecheck failed on both the original
unsplit parent and split prototype. Direct per-thread arrivals with matching
counts now order the shared alpha/inverse-factor handoff; intermediate
named-barrier experiments were discarded. Numerical success alone was not
used to waive the initial sanitizer diagnostics. This is fixed-GPU/runtime
forward qualification, not backward or default-dispatch performance.


### SM120 D512 paged forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_d512_determinism`
adds eight L0 forward cases: FP16/BF16, unpacked/PackGQA and split1/3 at
D512/V512, M64/N32. CPU-seeded inputs and an independent FP64 O/Stats reference
precede byte checks and golden recording. Tests include fresh storage,
concurrent work, B129 reorder/duplicates/empty neighbors, changed-neighbor
CUDA Graph replay, input immutability and output canaries. Split and packing
choices stay fixed within each comparison; cross-configuration identity is
not claimed.

RTX5080 (84 SMs), CUDA driver13.5, Torch2.14.1+cu130, cuDNN9.30.0.2,
CuTe DSL4.8.0 and tvm-ffi0.1.14.post1 use runtime key `1858cfcbf6876f83`.
Eight new records are appended without replacing existing baselines.
Eight existing D64/D120-V88 paging cases also pass and gain records for this
runtime: their input and output hashes exactly match the older driver13.4
records under `42742425a3d29d7c`. This observed equality does not establish
a cross-runtime guarantee. SM121 and backward remain unverified by these runs.

From `test/python`, run `python -m pytest -n 0 -m L0 -q
sdpa/frost/test_sdpa_fwd_paged_sm120.py -k "d512_determinism or paged_width_determinism"`
with `CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Recording used separate new directories;
normal comparison leaves `CUDNN_DETERMINISM_RECORD` unset.

The complete paging regression passes 144 cases, including all sixteen strict
goldens on this runtime. Final D512 qualification passes 49 cases: the 25 new
feature/admission/golden cases, existing dense/THD controls and two new dense
infinite-sink endpoint checks. After the last paged-only score-buffer barrier
fix, all 25 new cases pass again (279 deselected, no failures or skips).
Ten BF16 feature/endpoint cases pass each of Compute Sanitizer 13.4 memcheck,
racecheck and synccheck with zero reported errors; racecheck also reports zero
warnings. Memcheck is unfiltered; racecheck/synccheck select the native
`cudnn_kernel_` kernels. These are forward-path checks, not backward coverage.

Retained controls distinguish numerical success from synchronization safety.
The redundant paged V-tail stores passed numerics but produced 288 racecheck
errors; skipping those stores on paged input makes the identical case clean.
A broader ten-case run then passed numerically while reporting 80 grouped
score-exchange races. An explicit warp-pair read-completion barrier fixes the
single-buffer reuse: its failing split/packed control and the full ten-case
racecheck now report zero errors. The dense TMA path retains its prior handoff.

### SM120 half Q/O with paged FP8 K/V — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm120.py::test_sm120_frost_paged_mixed_fp8_determinism`
adds eight L0 forward goldens on RTX5080 (84 SMs): D64 prefill and D128/D256
paged decode plus D512, each with FP16/E4M3 and BF16/E5M2 Q/KV types. Packing
is fixed; split is one for D64 and three for the other shapes. Raw quantized
cache payloads stay fixed across fresh buffers, eager/captured replay,
reordered/duplicated batches up to B129, empty neighbors and changed neighbor
contents. Independent FP64 O/Stats references and input/output canaries are
checked before byte comparisons. This is plain `sdpa` cache storage, with no
scalar descales or quantized Q/O contract.

Torch2.14.1+cu130, cuDNN9.30.0.2, CuTeDSL4.8.0 (CUDA12.9 compiler),
CUDA driver13.5 and tvm-ffi0.1.14.post1 use runtime key `1858cfcbf6876f83`.
Eight records were appended from a new directory after successful recording;
the separate strict run passes all eight (331 deselected, no failures/skips).
No old golden was regenerated. The 27 feature/conversion/candidate checks
also pass: both half types and both FP8 formats, independent HND/NHD pools,
head envelopes, masked dead page/head tails, pointer/length rebinding, packed
splits, persistent reuse, normal graph candidates and all 256 FP8 encodings.
Finite conversion results, including signed zeros, match their references
byte-for-byte. Direct packed FP8-to-BF16 conversion failed with the CUDA12.9
compiler; the global loader now widens through FP16 exactly, then converts to
BF16. Existing shared-memory FP8 conversion code generation is preserved.

Compute Sanitizer13.4 runs ten E5M2 FP16/BF16 feature cases each under
memcheck, racecheck and synccheck: all pass with zero errors (racecheck zero
warnings). Memcheck is unfiltered; racecheck/synccheck select native
`cudnn_kernel_` kernels. Full host graph analyzer:628 passed. Earlier failures
were retained: one incorrect test page-size argument; one BF16 assembly
failure after a FP16 pass; three longer-Q cases blocked because the explicit
plan helper first required a heuristics proposal. The helper now reaches the
explicit plan directly; three separate ordinary-candidate cases remain active.

From `test/python`, run `python -m pytest -n0 -m L0 -q
sdpa/frost/test_sdpa_fwd_paged_sm120.py -k mixed` with
`CUDNN_DETERMINISM_GOLDENS=$PWD/determinism_goldens` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; leave recording unset. The sanitizer
selection is `-k "mixed_fp8 and not determinism and kv_dtype1"`.
SM121, other runtimes and backward remain unverified; mixed dense KV and
THD queries remain unsupported on this new SM120 path. Cross-split or
cross-packing byte identity is not claimed.

Final shared-loader regression:36 passed,303 deselected, no failures/skips in
168.73s, including24 strict goldens (16 existing half plus8 mixed), native half
rebinding/replay and decode ring reuse. The old half baselines are unchanged.
The owned SM120 allocation completed normally after these checks.

## SM120 THD D256/GQA backward qualification

`test_sdpa_bwd_thd_sm120_gqa.py` compares dQ/dK/dV with independent CPU FP64
autograd, including per-port origins, GQA, bottom-right quiet rows and empty
sequences. Its L0 single-KV-tile byte checks preserve fixed inputs/upstream
gradients across fresh storage, concurrent work, B1/B3/B129, duplicates,
reordering and changed-neighbor capture. On GB10/SM121 (driver 580.142,
Torch 2.9.0a0+50eac811a6.nv25.09/cu130, cuDNN 9.13.0, DSL 4.8.0/compiler 12.9),
both FP16/BF16 cases pass recording and fresh-process strict checks, using
runtime key `d2d1094fc872ddba`. All three gradients match their independent
reference. SM120 remains unverified.
Final combined execution passes 73 cases (40 new and 33 existing D128),
including the two strict goldens; 123 host checks pass. Five cases each pass
memcheck, racecheck and synccheck using CUDA 13.4 sanitizer 2026.3.0 with
API-query reporting disabled (CUDA Python 13.2 probes newer APIs than the
CUDA 13.0 driver). Racecheck reports zero errors and zero warnings.
General deterministic backward remains unsupported because multiple KV tiles
accumulate dQ with unordered FP32 atomics; admission tests retain that decline.

SM120-family wide half backward (`sdpa-sm120-wide-bwd-d512-gqa-swa-*`):
D512, GQA 6:2, Q17/KV19 crossing two native KV tiles, bottom-right SWA,
FP16/BF16. The deterministic relay is exercised under fresh allocations,
repeated calls, CUDA Graph replay, changed neighbors, duplicate/reordered
sequences and batches 1/3/129. dQ/dK/dV are checked against independent CPU
FP64 autograd before strict raw-byte comparison. Recorded on SM121 GB10;
this fingerprint does not qualify SM120 hardware or low-precision backward.

### SM120 native per-tensor FP8 backward

`sdpa/frost/test_sdpa_bwd_fp8_sm120.py::test_replay_batch_and_golden` adds six
SM121/GB10 runtime-keyed records for D192/D512 and E4M3/FP16/BF16 gradients.
Each checks dQ/dK/dV and amax_dQ/dK/dV/dP with fixed scalar scales and payloads
across batch reorder/duplication, fresh pointers, changed-dO capture replay,
and repeated calls. The independent FP8 numerical oracle is checked first.
SM120 hardware and cross-runtime bit identity are not covered.


### SM120 native MXFP8 backward

`test_sdpa_bwd_mxfp8_sm120.py::test_replay_batch_and_golden` adds four
SM121/GB10 records for D128/D512 with FP16/BF16 gradients. Input payloads
and all seven canonical SF tensors are reordered together, including the
D-plane-major transposed atoms. The checks cover B1/3/4/129, fresh storage,
repeated calls, changed-payload capture and live SF mutation. The independent
MXFP8 numerical oracle passes before byte comparisons. SM120 hardware and
cross-runtime identity remain unverified.

Qualification on 2026-11-10: GB10 SM121, driver 580.126.20, Torch
2.9.0a0+50eac811a6.nv25.09/cu130, cuDNN 9.13, DSL 4.8/compiler 12.9.
The full new suite passes **55 tests, zero skips**, including the four goldens
in a fresh strict-check process; the existing per-tensor FP8 suite passes
**79 tests**, with all six committed hashes unchanged. The numerical oracle
also independently requires finite dQ/dK/dV and relative L2 error below 0.05.
Eleven representative cases pass each of memcheck, racecheck and synccheck.
From `test/python`, with the isolated checkout verified:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest -q -n0 sdpa/frost/test_sdpa_bwd_mxfp8_sm120.py
```

This change adds no forward kernel; existing forward coverage is unchanged.

#### Optional MXFP8 gradient amax outputs — 2026-11-10

`test_sdpa_bwd_mxfp8_sm120_amax.py` adds fourteen L0 cases: all eight
subsets of amax_dQ/dK/dV, two standalone binding cases, and four D128/D512
FP16/BF16 batch/replay goldens. Each requested scalar is the global maximum
absolute FP32 gradient accumulator before conversion to the gradient dtype.
The independent MXFP8 numerical reference passes before byte comparisons;
the amax comparison bounds only the final FP16/BF16 rounding.

The four records under runtime key `d2d1094fc872ddba` preserve all parent
MXFP8 input and gradient hashes, adding the three FP32 maxima. Coverage
includes B1/3/4/129, reorder/duplication, fresh pointers, NaN/Inf output
poisoning, workspace poisoning and changed-dO capture replay. Global amax
must compose as the maximum of singleton values; it is not a per-sequence
output invariant to adding arbitrary neighbors. Standalone checks also
require invalid optional bindings to leave outputs and workspace untouched,
followed by successful execution on the same plan.

Recorded on GB10 SM121, driver 580.126.20, Torch
2.9.0a0+50eac811a6.nv25.09/cu130, cuDNN9.13, DSL4.8/compiler12.9 and
tvm-ffi0.1.14.post1. SM120 hardware and cross-runtime bit identity remain
unverified. This extension changes no forward kernel or existing golden.
Fresh-process strict qualification passes all 148 cases: 14 in this extension,
55 existing MXFP8 cases and 79 existing per-tensor FP8 cases, with no skips.
All four new records retain the parent's input and gradient hashes. Host
admission/specialization checks pass 95 cases; 12 native SM100 D128/D192/D512
controls also pass with the shared prepared ABI. The amax-enabled D128 and
D512 kernels cross-compile for SM120; this is not hardware qualification.
Twelve representative cases pass each of memcheck, racecheck and synccheck,
with zero device errors or race warnings. Compute Sanitizer 2026.3.0 instruments
all kernels; API-query reporting is disabled for the recorded CUDA Python
13.2/driver13.0 startup mismatch. The first standalone fixture incorrectly
equated graph `left_bound=9` with `window_size_left=9`; it now uses eight
positions to the left plus the diagonal. No kernel change or golden rerecording
was needed for that fixture correction.

From `test/python`, after verifying the isolated import path:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest -q -n0 sdpa/frost/test_sdpa_bwd_mxfp8_sm120_amax.py
```

### SM120-family D576/V512 forward — SM121 qualification, 2026-11-10

`sdpa/frost/test_sdpa_fwd_d576_sm120.py` adds eighteen L0 replay/batch cases:
eight uniform-half paged, four mixed-FP8 paged, and six dense/split/THD.
They hold split and packing fixed, preserve CPU-seeded inputs, compare O/Stats
against FP64, and check duplicated/reordered batches up to B129 and changed
neighbors under capture. Compact THD repacks inputs and declared offsets as
a neighbor changes between empty and live; independent physical gaps are not
part of this existing path's contract. Backward is unsupported by this addition.

GB10 (SM121,48SMs), driver580.142/CUDA13.0, Torch2.9.0a0+50eac811a6.nv25.09,
cuDNN9.13.0, CuTeDSL4.8.0/compiler12.9, tvm-ffi0.1.14.post1, matching native
FE extension and verified isolated import: **18 passed,35 deselected** when
recording into a fresh directory (178.53s), followed by **18 passed,35
deselected** in a new strict-check process (5.15s). The18 new JSON baselines
live under runtime key `d2d1094fc872ddba`; all16 hashes recorded before the THD
fixture correction are unchanged. The independent reference and batch checks
run before each baseline is recorded. Command from `test/python`:

```bash
CUDNN_TEST_NO_ISOLATION=1 CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest -q -n0 sdpa/frost/test_sdpa_fwd_d576_sm120.py -k determinism
```

The first recording passed16 and failed2 THD cases because the fixture changed
lengths while leaving later sequences at old physical addresses. That failure
is retained separately from the successful compact replay. An earlier expanded
feature run stopped at4pass/3fail because the native binder rejected asymmetric
widths before launch; after its envelope and extension-gate correction, all
**35 feature cases pass** (275.33s). The full host analyzer passes629checks.
SM120 D576 hardware execution, other runtimes and performance remain unverified.
Targeted sanitizer and existing D512 regression results are recorded separately
when complete; no completed result is implied by this golden qualification.


### SM90 native D128/D256 forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_native_heads_sm90.py` adds eight L0 forward
bytewise cases at D128 and D256, FP16/BF16, unsplit unpacked and split3
PackGQA. A fixed CPU payload and FP64 O/Stats reference precede recording.
The checks cover fresh storage, concurrent work, B1 versus B129, duplicated
and reordered sequences, changed-neighbor isolation, empty neighbors and
CUDA Graph replay, with padding, a bottom-right window and sinks. Eight new
JSON records under `6521230c64ce3d35` were recorded into a new directory,
then copied and strictly checked with `CUDNN_DETERMINISM_RECORD` unset.

H100 NVL cc9.0, driver595.71.05, Torch2.14.1+cu130, cuDNN9.28.0.38 and
DSL4.8/compiler12.9: all 88 new feature/strict cases pass without skips.
The initial 64 cover numerical/mask/layout/split contracts; the final 24
add the eight strict goldens, partial groups G12/G128, all three schedulers,
independent THD offsets with both packed Stats layouts, and offset replay.
Forty existing controls pass, including all eight unchanged D512 goldens,
D64 prepared execution and standalone D64 feature envelopes.

Memcheck, native-attention/combine racecheck and synccheck each pass the same
22 cases with zero errors or race warnings. The selection is
`thd_origin_replay or thd_physical_origins or partial_pack_replay or
multiple_stages or (split_replay and bf16)`, under Compute Sanitizer13.4.
Memcheck is unfiltered; the other tools use `--kernel-name kns=cudnn_kernel_`.
The first memcheck run passed all22 tests but reported40 startup API-query
errors: CUDA Python13.3 requested newer symbols than the13.2 driver API.
The clean run uses cached CUDA Python13.2 with API-error reporting retained;
no device diagnostic was suppressed. This is H100 forward qualification,
not a backward, H200 performance, cross-runtime or cross-plan byte guarantee.

Commands run from `test/python` with the isolated checkout verified:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -n0 sdpa/frost/test_sdpa_fwd_native_heads_sm90.py -m L0
```

The measured execution split this command into the initial64 feature cases
and the final24 cases; together they cover the complete88-case collection.

### Native SM90 paged KV forward — 2026-11-10

`sdpa/frost/test_sdpa_fwd_paged_sm90.py` adds 31 L0 cases for dense-query
FP16/BF16 paging through the native D128/D256/D512 bodies. Coverage includes
page sizes 8/16/32/64/128/192/256, independent K/V pool layouts and table
pitches, fresh pointers, changed-table CUDA Graph replay, poisoned padding,
base-2 Stats, sinks, split3, all three schedulers and partial GQA groups
G12/G128. Standalone head boundaries include D8/V512 and D512/V8; graph
head envelopes retain their existing limits. THD paging and Hopper backward
remain unsupported.

Six new golden records under `6521230c64ce3d35` cover D128, D192/V128 and
D512 in both dtypes. Each compares O and Stats with an independent FP64
reference, then checks raw bytes across fresh storage, concurrent work,
B1/B129, reordered/duplicated sequences, changed neighbors and captured
replay. Recording into a new directory passed all six cases; the subsequent
strict run left recording unset and used the committed files.

H100 NVL cc9.0, driver 595.71.05, Torch 2.14.1+cu130, cuDNN 9.28.0.38 and
DSL 4.8/compiler 12.9: the strict run passed 172 cases, 68 deselected,
zero failures/skips, in 266.05 seconds. These are 23 new paging cases and
149 existing native binding/head controls. The six standalone head-boundary
cases and two partial-PackGQA cases also passed under each sanitizer,
completing all 31 new cases.

Memcheck, racecheck and synccheck each passed 24 selected cases with zero
errors or hazards: eight BF16 page-replay cases, six standalone boundaries,
two partial groups and eight existing D64 graph controls. The runs were
split into 22 cases and the final two partial groups. Memcheck was unfiltered;
racecheck and synccheck used `--kernel-name kns=cudnn_kernel_`. CUDA Python
13.2 matched the driver API and API-error reporting stayed enabled.

The first two control runs exposed a prototype nonpaged host-ABI mismatch:
the common frame binds K table strides as `(0, 0)` and V table strides as
`None`. The final host specialization preserves that distinction. A first
recording command incorrectly combined recording and checking modes and
stopped before writing; the clean recording and strict run above supersede it.
These results qualify H100 forward correctness and per-plan determinism;
H200 execution, other runtimes and performance remain unmeasured.

Reproduction commands from `test/python` (the measured runs split the suite
between strict and sanitizer selections):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -n0 sdpa/frost/test_sdpa_fwd_paged_sm90.py -m L0

compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest -q -n0 \
    sdpa/frost/test_sdpa_fwd_paged_sm90.py \
    sdpa/frost/test_sdpa_native_d64_sm90.py \
    -k '(paged_replay and bf16) or standalone_page_head_boundaries or paged_partial_pack_groups or graph_uses_prepared_native_body'
```

### SM107 single-sequence physical-origin setup — 2026-11-10

`sdpa/frost/test_sdpa_single_sequence_setup_sm107.py` adds14 L0 cases for
uniform FP16/BF16 D64/D128, nonpaged and unsplit. Eight physical-origin cases
exercise both CTA group sizes, packing, independent INT32/INT64 element
origins and multipliers, token/head-major Stats, changed token strides,
rebound addresses, retained CUDA Graph replay, keyless rows and empty work.
Four byte tests use fixed CPU payloads and independent CPU FP64 O/Stats
references before recording. They compare fresh storage/concurrent work,
poisoned-workspace replay, B1 against B3/B5/B129, duplicate/reordered/empty
sequences and changed-neighbor isolation. Each byte test now reuses one
capacity-4096 plan through public shape overrides, then returns to B1 after
multi-sequence bindings. Two additional tests use runtime B4 and check the
generic setup branch without forcing a specialization.
Stats negative infinity is permitted only as the keyless-row sentinel; the
reference and canaries still check its location and all other output values.

SM107 / driver620.62 / Torch2.14.1+cu130 / cuDNN9.30.0.2 /
DSL4.8 compiler13.4:14new cases passed, four new runtime goldens under
`aa97de087283f297` were recorded, then all42 strict/control cases passed
(14new+20dead-subtile+8existing D64 physical-origin controls), without skips.
The strict run had recording unset and used the copied golden directory.
The final capacity-4096 qualification repeated all42strict/control cases
(63.83s). Memcheck/racecheck/synccheck each passed all14new cases
with zero errors or hazards. Memcheck was unfiltered; race/sync used
`--kernel-name kns=cudnn_kernel_`; API-error reporting remained enabled.
The generic multi-sequence setup body is AST-identical to the prior source.
This qualifies the setup path, not new arithmetic, other GPUs, split/paged
specializations or cross-runtime/cross-plan byte invariance.

From `test/python`, after asserting the isolated import path:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q sdpa/frost/test_sdpa_single_sequence_setup_sm107.py \
  sdpa/frost/test_sdpa_thd_dead_subtile_sm107.py \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD64Half::test_independent_origins
```

The initial collection attempt used an unavailable xdist option and performed
no tests. The first byte-test attempt rejected the legitimate keyless Stats
sentinel under the name `stats`; explicitly declaring its allowed negative
infinity fixed the harness, with no kernel or tolerance change. All14 cases
then completed and were checked again in the42-case strict run.

## Simplified sparse attention — 2026-11-09

The new suite is
`simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py`.
It tests the public `cudnn.ops.simplified_sparse_attention` call. There is no
mode that guarantees determinism. All cases are L0 and use BF16 input,
BF16 output/gradients, and FP32 loss. Tests check output, indexer loss,
dQ, dK, dV, dIndexerQ, and dIndexerK. They do not expose or compare saved
internal statistics. There are no gate or recurrent-state gradients.

The initial SM100 run expanded to 16 cases: **9 passed, 7 failed, 0 skipped**.
Forward passed all eight cases. Backward passed one small golden case and
failed the other seven cases. Each failure stopped at dK, so it does not
establish the status of later gradient comparisons. No backward golden is
approved from that run. The three forward baselines passed the independent
FP64 reference and eight fresh-allocation replays before recording. Golden
names include the loaded DSL compiler's CUDA flavor.

| Check | Shapes / variation | Result |
| --- | --- | --- |
| Replay | Packed lengths 65/3/17; Hq16/top-k512, Hq32/top-k1024, Hq96/top-k2048 | Forward passes; backward dK differs by 3, 1, and 1 bytes in 43,520 bytes on replay 1. |
| Scheduling boundary | 297 queries, Hq16, top-k512; 148 SMs | Forward passes; backward dK differs by 13 of 152,064 bytes. |
| Batch changes | Hq16/top-k512; target length65; neighbor length17; insertion, removal, reordering, duplication, changed neighbor | Forward passes; backward stops at first insertion, with 3 of 33,280 dK bytes changed. Later layouts are unmeasured for backward. |
| Golden | Length17; same three Hq/top-k pairs | All forward cases pass. Hq16 backward passes locally; Hq32/Hq96 dK each differ by 1 of 8,704 bytes during replay. |

Inputs and upstream gradients are generated once on CPU, then packed without
new draws. Invalid rows include empty key intervals. The scalar loss uses a
fixed denominator of 257, and batch checks isolate the target's loss and
upstream gradient. Full empty Q/K streams are unsupported. Concurrent GPU
work and fresh allocations vary scheduling without changing the inputs.

Runtime: SM100, 148 SMs, driver 610.11 (CUDA driver API 13.3), PyTorch
2.13.0+cu130, cuDNN 9.27.0, CuTe DSL 4.8.0 with the CUDA 13.4 native compiler,
TVM FFI 0.1.13.post2. The integration starts from Frontend revision
`cb5f9e54bbb1f64b9b3af2992dc5dbe95f831888`. The printed import path was the
integration checkout, verified before testing. SM103 and long filtered-path
determinism are unverified.

Initial measurement, from `test/python`, with the runtime library path set:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_RECORD=/tmp/ssa-new-goldens \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py -m L0 -q -s
```

Use a new recording directory. Inspect hashes and reference results before
copying baselines. Normal checks leave `CUDNN_DETERMINISM_RECORD` unset:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py -m L0 -q
```

The expanded suite has 20 cases. Adding Hq32/top-k1024 and Hq96/top-k2048
batch checks exposed two forward failures: each changes 1 of 4 scalar-loss
bytes when the target moves to packed slot 1. The output bytes match before
the loss comparison fails. Loss accumulation groups physical rows into warps,
so moving a sequence can change FP32 rounding. The expanded strict forward
run was **8 passed, 2 failed, 10 deselected**.

At the requester's direction, retain kernel behavior and scope expected
failures on SM100 to the measured dK byte mismatches and those two packed-loss
cases. Each replay passes its numerical reference before byte comparison. Shape/dtype changes,
non-finite results, other tensors, and other loss mismatches still fail.
Backward golden cases remain expected failures until replay is stable and
a baseline can be approved; an intermittent replay pass is insufficient.
Remove these exceptions with the corresponding reduction fixes.

The final installed-wheel L0 run on the same SM100 runtime reported
**106 passed, 12 xfailed, 4 deselected, 0 failed, 0 skipped**. Of these,
the determinism suite accounts for **8 passed and 12 xfailed**. Nine xfails
stop at a dK byte mismatch, two at the packed scalar loss, and one at the
unapproved backward golden after its small replay passes. Later tensor and
layout comparisons after each mismatch remain unmeasured. Forward status
is **Replay only**; backward is **Not guaranteed**. Expected failures do
not improve those support claims.

A final 20-case rerun with the FP64 reference checked on every replay also
reported **8 passed, 12 xfailed**. Eight cases stopped at dK, two at packed
loss, and two at unapproved backward goldens. Hq32's small backward replay
passed in this run despite failing earlier; no new baseline was recorded.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

The installed wheel contains all 32 Python operation modules and its source
notice. Separate CuTe DSL 4.8.0 CUDA 12.9 compiler testing passed 17 public API
cases, including all nine Hq/top-k forward/backward reference combinations,
selected-support numerics, tails, invalid configurations, and capture rejection.
This used PyTorch/cuDNN built for CUDA 13; it qualifies the DSL compiler
flavor, not a deployment using CUDA 12 throughout. The loaded native libraries
were the cu12 and cu13 compiler extensions under `cutlass/_mlir/_mlir_libs/`
in separate environments. SM103, long-path CUDA 12 compilation, and newer DSL
versions remain unverified. The actual DSL 4.7.1 runtime rejects this API before
kernel import. A missing PyTorch conditional-graph entry also rejects before
filtered-state allocation; bypassing that guard makes its regression test fail.

The installed-wheel L1 run passed **4 cases, 118 deselected, 0 skipped**:
cold exact repair through the public API, changed addresses/bounds/extents,
reserved capacity at its maxima, sampler diagnostics, and a 16,384-query
structured-score stress case. These are numerical and storage-reuse checks,
not bytewise determinism qualification of the long path.

```sh
python -m pytest simplified_sparse_attention/cutedsl -m L1 -q -s
```

Compute Sanitizer `synccheck` passed all three Hq16/top-k512,
Hq32/top-k1024, and Hq96/top-k2048 public forward/backward reference cases:
**3 passed, 15 deselected; 0 sanitizer errors**. This covers those short
packed numerical paths, not the long filtered repair graph.

```sh
compute-sanitizer --tool synccheck --error-exitcode 86 python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py -m L0 -q -s \
  -k 'public_forward_backward_reference and ((k512 and bf16-h16) or (k1024 and bf16-h32) or (k2048 and bf16-h96))'
```

## Simplified sparse attention JAX — 2026-11-09

The JAX suite is
`simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism_jax.py`.
It exercises the public native-FFI API under `jax.jit` and `jax.vjp`, using
JAX/JAXlib 0.11.0 on the SM100 runtime above. The source base remains
`cb5f9e54bbb1f64b9b3af2992dc5dbe95f831888`, with this integration applied.

| Suite | Forward results | Backward results | Cases |
| --- | --- | --- | --- |
| PyTorch determinism | BF16 output, FP32 indexer loss | BF16 dQ/dK/dV/dIndexerQ/dIndexerK | 20 |
| JAX determinism | BF16 output, FP32 indexer loss | BF16 dQ/dK/dV/dIndexerQ/dIndexerK | 20 |

Both suites check the independent FP64 reference on every replay, then compare
raw bytes with fresh result buffers and concurrent GPU work. The JAX suite
uses the same fixed packed data, ragged tails, insertion/removal/reordering,
duplication, changed neighbors, and 297-query scheduling boundary. Internal
LSE, teacher distributions, and long filtered-path determinism remain unverified.

The JAX recording run reported **8 passed, 12 xfailed, 0 skipped**. Nine
cases stopped at atomic dK, two at the known packed loss, and one at the
unapproved backward golden. The three new forward JSON records passed their
references and replay checks. All input and output hashes match the existing
PyTorch goldens. Their filenames also pin JAX 0.11.0 and compiler CUDA 13.4.
No backward baseline was recorded.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_RECORD=/tmp/ssa-jax-new-goldens \
  python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism_jax.py -m L0 -q -s -rxX
```

A combined installed-wheel run reached three additional byte mismatches after
dK happened to match: PyTorch H96/top-k2048 packed backward dV (1/33280 bytes),
its backward golden replay dV (1/8704 bytes), and JAX H32/top-k1024 replay dV
(2/43520 bytes). Each passed its independent numerical reference first. These
are the same already-excepted backward cases; both dK and dV use FP32 atomic
reductions. The scoped SM100 backward exception now names both gradients.
Other tensors, non-finite values, shapes, dtypes, and reference failures still
fail. The interrupted diagnostic run reported **104 passed, 3 failed,
13 xfailed, 6 deselected**; it is not a qualification pass.

The two JAX L1 numerical cases passed at top-k512 and top-k2048 with 129
queries and 131,329 keys. All 129 tied rows force two exact-repair chunks;
a second call changes indexer inputs and key bounds and includes invalid
rows. The tests check selected scores against full FP32 top-k and compare
output, loss, and all five gradients with an independent FP64 reference
using the selection from that same public call. Tied selections may differ
between calls, so a separate selector execution cannot supply its reference.

Near-uniform KL is about 2.5e-6 in the tied top-k2048 case. A pure 1% relative
loss check exposed about 2.6e-8 absolute error. A shared native attention
probe also showed FP32 teacher-mass rounding (up to 5.3e-7), contributing
about 1.5e-8 of loss error. The new long-path loss check uses 1% relative
error plus an absolute budget of `16 * FP32_epsilon * loss_coeff * valid_rows /
loss_denominator`; output and gradient tolerances remain unchanged. This
budget applies equally to both compiler flavors. No kernel arithmetic changed.

The final combined installed-wheel L0 run reported **135 passed, 24 xfailed,
6 deselected, 0 failed, 0 skipped**. Each framework's determinism suite
contributes 8 passes and 12 expected failures. Across both suites, 16 cases
stop at atomic gradient differences, four at packed scalar loss, and four at
unapproved backward goldens after small replays pass. Later comparisons after
a mismatch remain unmeasured. The 21 JAX public API cases cover all nine
head/top-k forward/backward combinations, selected support, tails, triangular
metadata, zero loss coefficient, input ownership, invalid inputs, captured-input
JIT, rejection of mixed GPU architectures, and the dense workspace index limit.

```sh
CUTE_DSL_CACHE_DIR=/tmp/ssa-cu13-cache CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

In a separate installed-wheel environment, CuTe DSL 4.8.0's CUDA 12.9 compiler
passed **23 JAX cases, 0 failed, 0 skipped**: all 21 public cases plus both long
Hq16 filtered-repair cases. The loaded native libraries under `cutlass/_mlir/_mlir_libs/` used the cu12
and cu13 flavors, respectively. Each flavor used a fresh
process and its own initially empty DSL cache. PyTorch, JAX, and cuDNN still
use CUDA 13 packages, so this is compiler compatibility, not a full CUDA 12
deployment. CUDA 12 bytewise determinism and SM103 execution remain unverified.

```sh
CUTE_DSL_CACHE_DIR=/tmp/ssa-cu12-cache python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_jax.py -m 'L0 or L1' -q -s
```

The final wheel includes 34 Python operation modules, its Apache-2.0 notice,
and `contributors.md`. The inventory was compared byte for byte with the
checkout. The final rebuild changes only the JAX docstring's description of
the XLA repair loop relative to the full L0 run; the CUDA 12 run uses that
rebuild. Source copyright notices remain in the released files.

Compute Sanitizer **memcheck and synccheck each passed 1 JAX L1 case,
1 deselected, with 0 reported errors** on the final rebuilt wheel. The case
covers top-k512 cold repair in two chunks, changed inputs/bounds, selected
scores, and all five gradients. This does not qualify every head/top-k route.

```sh
compute-sanitizer --tool memcheck --report-api-errors explicit \
  --suppressions /tmp/cuda-binding-probe-suppression.xml --error-exitcode 86 \
  python -m pytest simplified_sparse_attention/cutedsl/test_filtered_jax.py -m L1 -q -s -k 512
# Repeat with --tool synccheck.
```

The initial memcheck run completed its numerical test and reported only seven
CUDA-binding feature-probe failures, plus seven extended diagnostics: the
bindings request CUDA API version 13041 against driver API 13030. A separate
lazy-import test reproduced the API errors without running these kernels.
The final run uses explicit API-error reporting and suppresses only
`cuGetProcAddress_v2` returning `CUDA_ERROR_INVALID_VALUE` (result 1) from
`cuda/bindings/_internal/driver*.so`. The XML record came from that import
control and retains its binding-module stack constraint. All memory-access
checks and other explicit API errors remain enabled; no device-access or
barrier error was suppressed.

JAX installed-wheel benchmarks (three warmups, 20 timed calls, synchronized
wall time) measured 2.951 ms forward and 3.441 ms training at
Tq1024/Tk4096/H16/top-k512, and 6.914 ms filtered forward at
Tq1024/Tk131072/H32/top-k2048. These shared-GPU samples include host dispatch
and completion waits; they are not direct speed comparisons with the earlier
PyTorch CUDA-event measurements.

### CodeQL follow-up — 2026-11-09

The follow-up to `40d2377` adds explicit lazy-export declarations, rejects
invalid GEMM layouts before allocation, and initializes top-k settings in
the base constructor. Removing unused names preserves the Hq16 kernel's
shared-memory reservation and arithmetic. No golden or expected-failure
rule changed.

The source-checkout L0 run on SM100 reported **148 passed, 24 xfailed,
6 deselected, 0 failed, 0 skipped**. This includes 13 new host cases for
invalid layouts and explicit/automatic top-k settings. The lazy-import test
also checks all four package exports against `cudnn.ops`. Both frameworks'
forward/backward reference, replay, batch-invariance, and golden checks ran
with the existing scoped exceptions. Support classifications remain unchanged;
SM103 and long-path bytewise determinism remain unverified.

Runtime: Python 3.12, PyTorch 2.13.0+cu130, JAX 0.11.0, cuDNN 9.27,
and CuTe DSL 4.8.0 with its CUDA 13.4 compiler. Imports resolved to the
edited checkout. From `test/python`:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m L0 -q -s -rxX
```

CodeQL 2.27.4 with `codeql/python-queries@1.9.0` scanned all 34 operation
modules using `python-security-and-quality.qls`. None of the 14 inline
findings on PR #2370 remains. The scan still reports 239 other package
diagnostics; no query or path was excluded by this fix. Pre-commit passed.
Fern reported zero errors and one existing warning.

The subsequent L1 run reported **6 passed, 172 deselected, 0 failed,
0 skipped**. It covers PyTorch repair-state reuse, reservation, sampling,
and diagnostics, plus both JAX long-key repair cases and their gradients.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest simplified_sparse_attention/cutedsl -m L1 -q -s
```

### Host concurrency and export follow-up — 2026-11-09

The follow-up to `80de1aa` serializes each stream's shared-scratch enqueue
sequence for dense indexing, filtered indexing/reservation, and H16 backward.
Filtered-state retirement also protects its shared host list. No kernel
arithmetic, golden, or expected-failure rule changed.

The six new public-API cases pause a host thread after writing shared scores
or filtered inputs. Another thread then enters with opposite indexer queries.
They cover default, shared side, and separate streams at Tq8/Hq16/top-k512,
with Tk1024 and Tk131072. The independent expected attention outputs are all
ones and all zeros. Concurrent output and scalar-loss bytes must match the
sequential results; separate-stream calls must complete while the first is
paused.

With the original `80de1aa` kernels, these cases reported **4 failed,
2 passed, 31 deselected**: both shared-stream modes corrupted all 32,768
output elements in the first call, on both indexer paths. With the fix,
the six cases and the Torch-only import check reported **7 passed,
30 deselected, 0 failed, 0 skipped**. The import check blocks JAX in a fresh
interpreter; the JAX export assertion now lives in the optional JAX suite.

```sh
python -m pytest simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py \
  -m L0 -q -s -k 'host_threads_share_stream or lazy_imports'
```

The rebuilt `nvidia_cudnn_frontend-1.33.0-cp312-cp312-linux_x86_64.whl`
has SHA256 `55db69c2370f8bdc0fdd294c7a8352790011c0d54cd8ee1c71c6ada153a8eade`.
All 35 operation Python modules, `NOTICE`, and `contributors.md` match the
edited checkout byte for byte. The installed-wheel API-index check reported
**11 passed**, including the runtime baseline with six intentional additions
and no removals.

The installed-wheel L0 and L1 run reported **161 passed, 24 xfailed,
0 failed, 0 skipped, 0 deselected** in 750.39 seconds. It includes all six
concurrency cases, both frameworks' numerical and determinism suites, and
all six long-sequence cases. Existing expected failures and goldens are
unchanged. SM100 forward remains replay-only and backward remains not
guaranteed; SM103, long-path batch invariance, and saved internal statistics
remain unverified.

Runtime: SM100 (148 SMs), driver 610.11, Python 3.12.12,
PyTorch 2.13.0+cu130, JAX/JAXlib 0.11.0, cuDNN 9.27.0, CuTe DSL 4.8.0
with the CUDA 13.4 compiler, and TVM FFI 0.1.13.post2. Imports resolved to
the installed wheel. From `test/python`:

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest simplified_sparse_attention/cutedsl -m 'L0 or L1' -q -s -rxX
```

## Phase 1 dense indexer KL public exports (2026-11-09)

Revision `b1a9c4a`, based on `55b9180`. The installed wheel exposes
`dense_indexer_kl` through both `cudnn.ops` and
`cudnn.simplified_sparse_attention`. The API retains the scalar loss and
indexer-only autograd contract. `use_kf=False` selects chunked PyTorch;
`use_kf=True` selects the supplied-LSE CuTe kernel. Both reject external
CUDA graph capture. No kernel arithmetic or deterministic mode was added.

The new suites collect 64 L0 cases: 32 numerical/contract cases and 32
forward/backward determinism cases. Numerical coverage uses an independent
FP64 full-softmax teacher and autograd for the student. It checks all three
head counts, query/key tails (3/17, 65/257), and split-key work (33/4097).
The loss uses coefficient 0.17 and fixed denominator 71; backward uses a
fixed upstream gradient of 0.7. Loss tolerance is `rtol=2e-4, atol=2e-7`;
BF16 gradient tolerance is `rtol=0.02, atol=2e-6`.

Replay compares raw bytes of the loss, BF16 indexer gradients, and saved
FP32 gradients with fresh allocations and concurrent GPU work. The stress
case has `32 * (SM_count + 1)` query rows (4,768 here), crossing CTA scheduling
waves. Packed checks preserve inputs, scaling, and upstream gradients across
neighbor insertion/removal, reordering, duplication, and changed neighbors.
Only the target contributes to the scalar loss. Inputs are checked for
mutation, outputs for finiteness, and teacher Q/K for absent gradients.
Contract tests cover invalid metadata, chunk sizes, fused dtype/head/scale/LSE
requirements, old-DSL rejection before kernel import, capture rejection, and
side-stream forward/backward without host synchronization.

Before exceptions, the initial runs reported **40 passed, 10 failed,
8 deselected**, followed by **6 passed, 2 failed, 24 deselected** for stress
and golden recording. All numerical checks passed. The 12 bytewise failures
were approved as scoped expected failures:

- Fused H32 tail forward/backward replay: scalar loss differs by one byte.
- Fused H96 split-key backward replay: saved FP32 indexer gradients differ.
- Fused scheduling stress, both directions: scalar loss differs by one byte.
- Packed PyTorch backward at H16/H32/H96 and fused backward at H32/H96:
  loss or saved gradient bytes differ.
- Packed H96 forward, both paths: scalar loss differs by one byte.

Exceptions match only these named cases on SM100 and raw-byte mismatch
messages for loss/indexer gradients. Numerical, non-finite, mutation,
shape/dtype, and unrelated failures remain strict. A case stops at its first
byte mismatch; later comparisons in that case are unmeasured. Remove each
exception when its replay and batch checks are fixed. Intermittent passes do
not establish determinism.

Runtime: SM100, 148 SMs, driver 610.11 / CUDA driver API 13.3; Python 3.12.12,
frontend 1.33.0, PyTorch 2.13.0+cu130, cuDNN 9.27.0, CuTe DSL 4.8.0,
TVM FFI 0.1.13.post2. Separate environments loaded native compiler extensions from
`cutlass/_mlir/_mlir_libs/`: the cu13 flavor reported compiler CUDA 13.4
and the cu12 flavor reported compiler CUDA 12.9.
Each installed-wheel lane used a fresh process and separate empty DSL/driver
cache directories. Imports resolved to the corresponding environment's
`site-packages/cudnn`, not the editable checkout.

| Check | Result |
| --- | --- |
| Installed wheel, CUDA 13.4, Phase 1 plus nine Phase 2 numerical cases and both framework export checks | 63 passed, 12 xfailed; 0 failed/skipped/deselected, 102.49 s |
| Installed wheel, CUDA 12.9, all Phase 1 cases | 54 passed, 10 xfailed; 0 failed/skipped/deselected, 43.82 s |
| Editable source, Phase 1, lazy exports, and three kernel-prefix checks | 56 passed, 12 xfailed; 0 failed/skipped/deselected, 54.12 s |
| Installed-wheel API inventory | 11 passed; one added name, no removals |
| Wheel contents | All 35 operation Python modules, NOTICE, and contributors.md match the checkout bytes |
| Formatting / docs | Pre-commit passed; Fern: 0 errors, 1 existing warning |

The two H32 fused replay cases passed in the CUDA 12 run; their known
exceptions remain. Four one-row forward/backward goldens were recorded in a
new directory for each compiler flavor after reference and replay checks.
The eight committed files have identical input and output hashes across
flavors for the same backend/direction. No existing golden was replaced.
These tiny cases cover the scalar oracle and PyTorch path; the fused fast
path has no stable golden because larger reductions fail replay.

Wheel: `nvidia_cudnn_frontend-1.33.0-cp312-cp312-linux_x86_64.whl`.
SHA256: `4a3bb8627750be8121c8436647dd4ff5d9b8dd10803f6603cf6fe3e745871208`.
The CUDA 12 compiler lane uses CUDA 13 framework/backend packages; it does
not qualify a full CUDA 12 deployment. SM103, other GPUs, newer DSL releases,
and multi-device execution remain unverified. Phase 1 has no JAX entry point.
Phase 2's earlier full-suite results above remain historical measurements.

Commands from `test/python`, with the installed cuDNN library directory on
`LD_LIBRARY_PATH` and separate fresh `CUTE_DSL_CACHE_DIR`/`CUDA_CACHE_PATH`
values per compiler lane:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl.py \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl_determinism.py \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_lazy_imports \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_public_forward_backward_reference \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py::test_public_jax_export \
  -m L0 -q -s -rxX --tb=short

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl.py \
  simplified_sparse_attention/cutedsl/test_dense_indexer_kl_determinism.py \
  -m L0 -q -s -rxX --tb=short
```

## Simplified sparse attention: offset and CI regressions (2026-11-10)

The triangular score producer and top-k consumer now widen operands before
workspace products. The tile decoder derives its upper bound from the query
extent. `test_triangular_offsets.py` extracts the production expressions and
checks them with fixed-width host scalars and native CuTe probes. Rows span
0 through 262143, including the 64K overflow boundary; decoder cases cover
1, 512, 1024, 1025 and 2048 blocks. The probes write offsets to small output
arrays and never dereference the large addresses.

The unfixed code produced **13 failed, 12 passed** in the 25-case regression
selection. The fixed code produced **25 passed** on SM100 with the CUDA 13.4
compiler. This selection also checks that public JAX exports remain inspectable
with a modeled DSL 4.7.1 while execution still rejects before kernel import,
and that all Torch/JAX test transfers enter the tensor's device context.
Three public triangular cases additionally passed numerical forward/backward
checks and forward replay across two tiles, for H16/K512, H32/K1024 and
H96/K2048. These checks do not establish backward determinism.

CI also measured a one-byte scalar-loss mismatch for packed H16/K512 forward
on SM100 with PyTorch 2.14.0a0 and CUDA 13.4. It passed numerical checks before
failing byte comparison. The earlier local H16 pass does not establish batch
invariance; the support page retains the replay-only status.

The installed-wheel API-index suite passed all 11 tests with actual DSL 4.7.1
and all 11 with DSL 4.8.0. Direct execution under 4.7.1 still raises the
4.8 requirement before loading SSA kernel modules. No API baseline names were
removed. The guard-word scan passed all 214 rules on the tracked source tree.
The rebuilt wheel contains all 35 operation Python modules, NOTICE and
contributors.md with bytes identical to the checkout.

The full installed-wheel CUDA 13.4 run reported **242 passed, 1 failed,
1 skipped, 34 xfailed** in 833.79 s. Its sole failure was a test-side stream
race in `test_reserved_state_at_its_maxima`: the side stream consumed zero
queries before the caller stream finished producing them. The test now waits
on the producer and delays that producer to expose a missing dependency.
It passed in 23.69 s; removing only the wait failed again in 24.52 s. No
kernel, tolerance or expected-failure rule changed for this test repair.
The skipped case needs two GPUs; its device-context control passed on this
single-GPU host. All existing goldens remain unchanged. The H16 CI packed-loss
case remained strict in this run; the approved exception is recorded below.

Runtime: SM100, 148 SMs, driver 610.11, frontend 1.33.0, Python 3.12.12,
PyTorch 2.13.0+cu130, JAX/JAXlib 0.11.0, cuDNN 9.27.0, CuTe DSL 4.8.0
and TVM FFI 0.1.13.post2. Imports resolved to the installed wheel. Its SHA256
is `52b1d3f198e09e6ece095ceb454b9469666643c870249becf01f676137c03a86`.
SM103 and actual multi-GPU execution remain unverified. External CUDA graph
capture remains unsupported; the existing rejection tests still pass.

Code revision: `a8300fe`. The separate installed-wheel CUDA 12.9 compiler run
passed **62 tests, with 1 two-GPU skip**, in 336.19 s: all address/decoder
probes, nine public Torch numerical cases, three triangular numerical/replay
cases, the JAX version guard, all public JAX tests, both long-key JAX cases,
and the delayed-producer reservation test. This lane uses CUDA 13 framework
and backend packages; it does not qualify a full CUDA 12 deployment or CUDA
12 bytewise batch invariance. Pre-commit passed. Fern reported 0 errors and
1 existing warning.

Commands from `test/python`, with the installed wheel environment active,
`LD_LIBRARY_PATH` pointing to cuDNN 9.27, and separate DSL/driver cache paths:

```sh
# CUDA 13.4: full suite, before correcting the test's stream dependency.
CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
python -m pytest simplified_sparse_attention/cutedsl \
  -m 'L0 or L1' -q -s -rxX --tb=short

# CUDA 13.4: corrected delayed-producer test and its missing-wait control.
python -m pytest \
  simplified_sparse_attention/cutedsl/test_filtered_contract_cuda.py::test_reserved_state_at_its_maxima \
  -m L1 -q -s --tb=short

# CUDA 12.9: changed paths and numerical compatibility.
CUBLAS_WORKSPACE_CONFIG=:4096:8 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m pytest \
  simplified_sparse_attention/cutedsl/test_triangular_offsets.py \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_public_forward_backward_reference \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_triangular_tiles_reference_and_replay \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention.py::test_jax_old_dsl_allows_inspection_but_rejects_execution \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_jax.py \
  simplified_sparse_attention/cutedsl/test_filtered_contract_cuda.py::test_reserved_state_at_its_maxima \
  -m 'L0 or L1' -q -s --tb=short
```

From the repository root, in each DSL 4.7.1/4.8.0 environment, with
`CUDNN_API_INDEX_PACKAGE_ROOT` set to the installed `cudnn` directory:

```sh
CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s test/python/api_index -p test_api_index.py -v
```

### Approved H16 packed-loss exception (2026-11-10)

Extend the existing byte-mismatch exception to PyTorch BF16 forward at
H16/top-k512 on SM100, as approved. It matches only
`packed slot 1 indexer_loss: 1/4 bytes differ`. Numerical checks still run
before byte comparison. H16 backward loss, replay, other tensors, other byte
counts, SM103, and the JAX H16 case remain outside this new exception. Remove
it when the scalar-loss reduction is batch invariant. The replay-only support
status is unchanged.

On the same installed wheel and CUDA 13.4 runtime recorded above, the six
Torch batch cases reported **1 passed, 5 xfailed** in 33.32 s. H16 forward
passed locally; its CI failure remains recorded above. Seven injected-failure
controls confirmed the previous code rejects the observed diagnostic, the
approved forward case reports an expected failure, and backward loss, SM103,
two-byte loss, other-output, and numerical failures still raise assertions.
No kernel, numerical tolerance or golden changed.

```sh
# From test/python, with the installed wheel environment active.
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest \
  simplified_sparse_attention/cutedsl/test_simplified_sparse_attention_determinism.py::test_batch_invariance \
  -m L0 -q -s -rxX --tb=short
```

### SM107 D512/D576 FP32 split partial stores — 2026-11-10

Aligned, D-contiguous partial rows now use 16-byte global stores. The scalar
fallback retains per-element bounds and dead-row zero selection. Two added
L0 paged boundary cases exercise V480 and empty V264 inside the D512 envelope.
The independent references, rebinding and changed-input replay stay unchanged.

SM107,216SM, driver620.62, Torch2.14.1+cu130, cuDNN9.30.0.3, DSL4.8.0
with compiler13.4:98 native cases passed in158.35s, without skips. These cover
dense/paged KV, FP16/BF16, D512/D576, packed/unpacked splits, masks/sinks,
optional/natural/log2 Stats, physical layouts and D512 mixed E4M3/E5M2 KV.
The12 half/mixed runtime goldens under `d634113bbc56110d` were recorded from
the unchanged parent after its independent references and byte checks passed
(12cases,31.06s). The candidate passed strict comparison against all12 with
recording unset. No existing golden was replaced. An initial candidate run
passed34 cases before stopping on the missing runtime-specific golden; it
reported no numerical mismatch. Other GPU/runtime execution is unverified.

From `test/python`, after asserting the isolated import path:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m 'L0 or L1' \
  sdpa/frost/test_sdpa_d512_pack_split_sm107.py \
  sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged \
  sdpa/frost/test_sdpa_fwd_native_paged_sm107.py::TestNativeD576Half \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv::test_d512_dense_paged \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv::test_d512_standalone \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestMixedFp8Kv::test_d512_replay_batch_golden
```

### SM120 half THD physical origins — 2026-11-10

`test_sdpa_fwd_thd_origins_sm120.py` adds 24 L0 functional cases covering
FP16/BF16, token/head Stats, D64/D128/D192-V128/D256/D512/D576-V512.
Each compares against an independent FP64 reference, protects padding
canaries and checks mixed-width/multiplier origins, fresh pitches/pointers,
empty and keyless transitions under retained capture. Exact-byte checks
cover repeat execution, batch reorder/duplication, a sequence alone and
changed neighbors. Twelve separate golden cases finish those checks before
recording. No prior golden or tolerance is replaced.

RTX PRO 6000 SM120 (188 SMs), driver 595.84.01, Torch 2.14.0+cu130,
cuDNN 9.24.0.43, DSL 4.8.0 / compiler 13.4, cuda-bindings 13.0.3:
51 functional cases passed, including6 native/reference adapter-fallback,
3 bounded-batch and18 dense half/per-tensor-FP8 host controls. All 12 new
goldens under `812f98809d9324c5` recorded and strictly checked. Six memcheck
cases passed with 0 errors. Racecheck then reported 628 errors in V-tail
sanitization; synccheck did not run. The follow-up gives one warp ownership and synchronizes the compute group.
Complete compact D64/D256 controls report 88/226 race errors on the original
source and zero on the repair; their reference and retained replay checks
pass in both arms. The final run passes 73 functional/golden cases in 183.14 s
and 15 memcheck cases in 6.97 s with zero errors, preserving all 12 golden
hashes. Racecheck passes the same 15 cases in 62.93 s with zero errors/warnings;
synccheck passes 15 in 3.79 s with zero errors. The original FlashInfer issue
regression also passes all three default FROST cases and all three backend
controls after the synchronization repair. Earlier script quoting, fixture
multiplier, cache-directory and node-local-reuse failures happened outside
this successful final run; they are not counted as GPU passes.
SM121 and other runtimes remain unverified for independent physical gaps.

```bash
# From test/python after checking the isolated import path.
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -m L0 sdpa/frost/test_sdpa_fwd_thd_origins_sm120.py
```


### SM107 coalesced packed split partials — 2026-11-11

`test_coalesced_split_replay_batch_golden` adds four L0 cases for FP16/BF16
and packing groups32/64 at split4. It checks standalone/reordered/duplicated
and changed-neighbor batches through33members, repeated raw bytes, numerical
references and changed-input CUDA Graph replay. Existing graph/standalone
boundary coverage now includes groups16and32, alongside64/128, partial groups,
keyless rows, masks and sinks.

On SM107 with driver620.62, Torch2.14.1+cu130, cuDNN9.24.0.43 and
DSL4.8.0/compiler13.4,106native functional/strict-golden cases passed without
skips.20cases each passed Compute Sanitizer2026.3 memcheck, racecheck and
synccheck, with zero errors and zero race warnings. Memcheck was unfiltered;
race/sync used the attention-kernel name filter with API errors enabled.
All16records under `de01ba4cf30a4c9b` were recorded from unchanged parent
`8e31a5725` after its references and replay/batch checks passed, then checked
strictly by the candidate. No previous baseline or tolerance changed.

From `test/python`, use the preceding D512/D576 qualification command and add
`sdpa/frost/test_sdpa_fp8_sm107.py::test_coalesced_split_replay_batch_golden`.
The four coalesced cases live in that explicit SM107 CI entry, so the architecture
lane retains the new coverage without duplicate collection in directory sweeps.
The sanitizer subset is
`test_sdpa_fwd_native_paged_sm107.py::TestNativeD512HalfPaged::test_boundaries`
plus `test_sdpa_d512_pack_split_sm107.py`, selected with
`test_boundaries or g64_split or g128_split or g32_split or g16_split or keyless`.
Earlier runs stopped before tests on a missing interpreter target and a DSL
package path; the final execution used the explicitly verified imports above.

### SM120 FP8/MXFP8 shared-memory synchronization — 2026-11-11

`test_sdpa_fp8_v_tail_sync_sm120.py` adds eight replay/batch cases, eight
runtime goldens and four CPU scale-layout controls. Coverage is per-tensor
FP8 and MXFP8, D128/D512, E4M3/E5M2, BF16 output, Q33 with KV capacity80
and live length17, and poisoned padding. Exact split1/unpacked plans retain
quantized payloads and logical scales across standalone, reordered, duplicated
and changed-neighbor batches. O/Stats are compared bytewise; global Amax is
checked for identical-input replay and against each whole-batch reference.

RTX PRO 6000 Blackwell Server Edition (SM120,188SM), driver595.84.01,
Torch2.14.0+cu130, cuDNN9.24.0.43, DSL4.8.0/compiler13.4 and
cuda-bindings13.0.3: **45 functional/strict checks passed** in76.38s without
skips. The existing dense/THD cases include multiple KV tiles and persistent
work units. **19 cases each passed memcheck, racecheck and synccheck**
in22.48s/16.95s/4.05s, with zero errors and zero race warnings. Memcheck
was unfiltered; race/sync used the `cudnn_kernel_` name filter with API errors
enabled, under Compute Sanitizer2026.3.0.

All eight goldens under `812f98809d9324c5` were recorded from unchanged parent
`8e31a5725` after independent references and byte checks passed, then checked
strictly by the candidate with recording unset. No existing baseline or
numerical tolerance changed. The parent passes four selected numerical cases
but reports924race errors; this negative control prevents numerical success
from hiding the synchronization defect. Earlier fixture and sanitizer failures
were retained during development. The supplemental performance script stopped
after qualification on a missing test-helper import; its failure did not
produce a performance result or bypass any qualification check.

From `test/python`, after checking that `cudnn.__file__` names the intended
checkout, the functional qualification command is:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -v -m 'L0 or L1' \
  sdpa/frost/test_sdpa_fp8_v_tail_sync_sm120.py \
  sdpa/frost/test_sdpa_fwd_fp8_sm120.py \
  sdpa/frost/test_sdpa_fwd_mxfp8_sm120.py \
  -k 'v_tail or thd_poisoned_pad or d512_thd or thd_multi_unit_per_cta or padded_kv_poisoned_tail or d512_persistent_units or thd_head_dims'
```

For the sanitizer subset, replace the selection with
`(replay_and_batch and e4m3) or thd_poisoned_pad or d512_thd or thd_multi_unit_per_cta`.
SM121 execution, cross-runtime identity and independent physical THD origins
for quantized inputs remain unverified by this addition.

### SM120 per-tensor FP8 THD physical origins — 2026-11-11

`test_sdpa_fwd_thd_origins_fp8_sm120.py` adds 28 L0 functional cases and
eight golden cases. E4M3/E5M2 inputs and BF16 output cover independent
Q/K/V/O/Stats physical origins, mixed INT32/INT64 offsets and multipliers,
token/head-major Stats, changed pitches/pointers, empty/keyless transitions,
and retained capture with changed lengths, origins and scalar scales. Head
shapes are D64, D128, D192/V128, D256, D512 and D384/V512. Numerical
checks use an independent CPU FP64 reference; raw-byte checks cover repeat
execution, batch reorder/duplication, a sequence alone and changed neighbors.
The prepared and standalone fallback routes are checked within their existing
contracts. The hot-path check rejects host synchronization and CUDA allocations.

RTX 5090 SM120 (170 SMs), driver 620.62, Torch 2.14.0+cu130, cuDNN
9.24.0.43, DSL 4.8.0/compiler 13.4: **30 functional cases passed**, including
the existing D512 dense-split and interleaved standalone THD controls. The
candidate strictly preserves **12 unchanged-parent goldens** (eight FP8/MXFP8
V-tail and four FP16/BF16 origin controls). **Eight new origin goldens** under
`dee7499a23a43ab6` were recorded only after reference and byte checks passed,
then strictly checked with recording unset. No existing baseline was replaced.
The qualified runtime/test source is `b971af997`; the subsequent commit adds
only these records and documentation.

The unchanged combined parent `5196c9c35` fails all four D128/D512 E4M3/E5M2
poisoned-gap detectors while the candidate passes; all eight compact controls
pass. A separate E5M2 attribution run compares parent compact, candidate
compact and candidate gapped layouts: all 48 input/output/reference/error
records match exactly across the three arms, including nine identical failures
of the initially proposed 0.03 budget (maximum absolute error 0.05463). The new
E5M2 cases therefore use the existing SM120 E5M2 absolute 0.1 budget, with
zero relative tolerance, accounting for the two-mantissa-bit P cast. Existing
tests, E4M3/Stats tolerances and strict byte checks are unchanged.

**Four cases each passed memcheck, racecheck and synccheck**, with zero
errors and zero race warnings under Compute Sanitizer 2026.3.0. No kernel
filter was used. API reporting was `--report-api-errors explicit`, retaining
errors returned by explicitly called CUDA APIs. The default extended mode
first exited 91 after a `Selective device code recompilation in progress`
API warning; an independent `torch.full()` control, without importing FE,
reproduced that same warning and passed under explicit reporting. The original
warning is retained as a tool/runtime diagnostic, not counted as a clean run.
Earlier missing-dependency, fallback-override, unsupported-packing and
provisional E5M2-budget failures were resolved before these passing results.

From `test/python`, after checking the isolated import path:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -v -m 'L0 or L1' \
  sdpa/frost/test_sdpa_fwd_thd_origins_fp8_sm120.py \
  sdpa/frost/test_sdpa_fwd_fp8_sm120.py::test_fp8_sm120_d512_split_kv \
  sdpa/frost/test_sdpa_fwd_fp8_sm120.py::test_fp8_sm120_thd_fused_record_slices
```

The sanitizer subset selects independent-origin token/E4M3/D128 and
head/E5M2/D512, plus omitted-total-bound E5M2/D128 and E4M3/D512. Each
uses fixed split1/unpacked execution. SM121, MXFP8 physical gaps, backward
and cross-runtime byte identity remain outside this qualification.

### SM120 MXFP8 THD physical origins — 2026-11-11

`test_sdpa_fwd_thd_origins_mxfp8_sm120.py` adds28 L0 functional cases and
eight golden cases. E4M3/E5M2 inputs with BF16 output cover independent
Q/K/V/O/Stats origins, mixed INT32/INT64 offsets and multipliers,
token/head-major Stats, fresh pointers/pitches, empty/keyless transitions
and retained capture. Head shapes are D64,D128,D192/V128,D256,D512 and
D384/V512, with fixed split1/unpacked plans. An independent CPU FP64
reference checks values; exact-byte comparisons cover replay, batch
reordering/duplication, a sequence alone and changed neighbors. Batch
changes preserve raw payloads and repack existing logical scale bytes.
The fallback-adapter and hot-path checks retain their existing contracts.

RTX5090 SM120 (170SMs), driver620.62, Torch2.14.0+cu130, cuDNN9.24.0.43,
DSL4.8.0/compiler13.4:28 functional cases and12 unchanged-parent strict
controls pass on source288043149. Runtime sources are unchanged since
49c628af6. Four shared default MXFP8 cases and four native-route twins
also pass; the FP8-fixed/MXFP8-old parent fails all four origin detectors.
The combined shared suite passes10 native and11 public cases, including
retained half/FP8 and explicit BF16 backend controls.

The shared D512 V-scale fixture previously reshaped plane-major scales.
Both fixtures now use a byte permutation checked against independent
per-head/128-token quantization:8 controls pass, with4 old D512 failures
and4 old D128 passes. The production scale-load arithmetic is unchanged.
An earlier replay test doubled V without scaling its absolute reference
budget and failed15 cases with1–2 mismatches each. The final test first
checks the exact replay inputs at the original scale, then requires doubled
O and unchanged Stats/padding bytewise. It compares the doubled result in
original V units using the same reference error constants. All28 cases
pass; no existing tolerance or golden is replaced.

All eight new records under `dee7499a23a43ab6` were recorded after the
reference/replay/batch checks, then strictly verified with recording unset.
Four cases each pass memcheck, racecheck and synccheck in20.98s/285.20s/13.63s,
with zero errors and zero race warnings. These include D128/E4M3 and
D512/E5M2 full replay plus D128/E5M2 and D512/E4M3 without total hints.
No kernel-name filter is used. API diagnostics use `--report-api-errors explicit`;
the extended-mode runtime warning was independently reproduced by a Torch-only
control in the preceding per-tensor qualification. The first recording attempt
set both record/check variables and failed their mutual-exclusion precondition;
it produced no records and was corrected before the successful strict run.
SM121, backward, other runtimes and performance remain unverified here.

```bash
# From test/python with the isolated import path verified.
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -m L0 \
  sdpa/frost/test_sdpa_fwd_thd_origins_mxfp8_sm120.py \
  -k 'not golden'
# Record only after reference/replay checks; then run with RECORD unset.
CUDNN_DETERMINISM_RECORD=/tmp/new-mxfp8-goldens pytest -m L0 \
  sdpa/frost/test_sdpa_fwd_thd_origins_mxfp8_sm120.py -k golden
CUDNN_DETERMINISM_GOLDENS=/tmp/new-mxfp8-goldens pytest -m L0 \
  sdpa/frost/test_sdpa_fwd_thd_origins_mxfp8_sm120.py -k golden
```

### SM100 FP8 D128 THD physical origins — 2026-11-11

`test_sdpa_fwd_fp8_sm100.py::TestD128Fp8PhysicalOrigins` collects the
shared `test_mhas_v2` E4M3/E5M2 D128 cases with a strict native engine pin,
four token/head-major reference/rebind/replay cases, two no-total-hint
cases, two half D128/D256 forwarding controls and four new strict goldens.
All use the existing unsplit, unpacked CGA2 nonpaged THD contract.

The four replay cases check mixed INT32/INT64 offsets and multipliers,
independent Q/K/V/O/Stats gaps, NaN-filled input padding, output canaries,
changed pitches and storage, retained captures with changed live lengths
and offsets, and all-empty queries. Exact-byte checks cover repetitions,
batch permutations/duplicates, a sequence alone and changed neighbors.
The prepared graph and standalone graph routes produce the same bytes;
execute retains its allocation-free, asynchronous behavior. The numerical
reference and tolerances are unchanged.

Measured on SM100 (148 SMs), driver 595.71.05, Torch 2.14.1+cu130,
cuDNN 9.24.0.43, DSL 4.8.0 / compiler CUDA 12.9, tvm-ffi 0.1.14.post1:
the 10 functional cases and four strict goldens pass. New hashes live under
`361d6da48de8fd55`; no earlier golden is replaced. Twelve existing compact
D128 / paged D128-D256 FP8 controls also pass. The original shared D128
FP8 cases fail on the parent and pass on the repair. An exploratory CGA1
selection was rejected by the existing graph contract before kernel launch;
it is outside this unsplit nonpaged test domain.

Four cases each pass memcheck, racecheck and synccheck with zero errors
and zero race warnings. The initial memcheck run reported 20
`cuGetProcAddress_v2` API errors; a Torch-initialization-only control with
no FE import reproduced all 20. Initializing cuda-python before Torch in
the private sanitizer launcher eliminates those diagnostics while retaining
`--report-api-errors explicit`, all kernel instrumentation and error exits.
SM103 E4M3/E5M2 physical-origin lowerings compile offline; GPU execution
there is unverified. Wider heads, MXFP8, SM107 and backward are not qualified
by these D128 results.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestD128Fp8PhysicalOrigins
```

### SM100 MXFP8 D128/D512 THD physical origins — 2026-11-11

`test_sdpa_fwd_mxfp8_sm100.py::TestPhysicalOrigins` collects 24 L0 cases:
4 shared `test_mhas_v2` regressions, 8 reference/rebind/replay/batch cases,
4 no-total-hint cases and 8 new goldens. Both E4M3/E5M2 types run the
existing public THD configurations: D128 CGA2 and D512 CGA1. The initial
expanded probe also requested D128 CGA1 and D512 CGA2; those twelve cases
were existing planning declines, not numerical failures. The suite now
states the admitted pairs explicitly, with no capability or ranking change.
Every expanded case asserts its actual native kernel template.

The shared fixture retains quantized payload and scale bytes across batch
changes, poisons physical data gaps with NaNs, and checks output canaries.
It covers independent mixed-width offsets and multipliers, FP64 references,
new pointers/strides, live/empty retained replay, scale-only replay changes,
batch reorder/duplication/isolation and prepared/standalone graph parity.
The D512 maps use token dimension 1 for Q/K/V and dimension 2 for O;
D128 maps retain dimension 2 for every port. Scale map coordinates continue
to use compact logical prefixes.

Current measured runtime: SM100, 148 SMs; driver 595.71.05;
Torch 2.14.1+cu130; cuDNN 9.24.0.43; DSL 4.8.0 with its CUDA 13.4
compiler; cuda-bindings 13.4.4; tvm-ffi 0.1.14.post1. The runner asserts
`cutlass.CUDA_VERSION`, uses a fresh compiler cache, and checks the imported
checkout and native-extension hashes. The native C++ extension is rebuilt
against toolkit 13.4.92.

On this runtime all four shared cases fail on the immediate parent and pass
after repair. The full strict matrix passes all 24 cases, including eight
unchanged goldens; 30 neighboring MXFP8 cases also pass. No case is skipped.
The new baselines use `361d6da48de8fd55`; older goldens and all tolerances
are unchanged. The historical CUDA 12.9 run passed the 16 functional cases
and eight new golden-record cases with independent references and exact-byte
checks before recording. Its later strict run also passed all 24 cases.
Compiler-version equality is asserted separately because the current golden
runtime key records the DSL package version, not its CUDA compiler flavor.

CUDA 13.4 memcheck additionally reproduced pre-existing single-CTA THD
scheduler errors on the unchanged D256/D512 parent. Local publication fixes
those errors; D256 broadcasts all predecoded fields before an elected warp
returns slot credit, while D512 retains its per-lane credits and matching count.
The six directed memcheck cases then pass with zero errors.

Racecheck also reproduced the existing D128 CGA2 V-scale tail helper's access
to an exited peer: a compact causal-tail parent control reports 2400 errors.
The tail path now keeps both CTAs alive through a cluster exit barrier. Both
origin replay dtype cases pass racecheck with zero hazards, errors or warnings;
the remaining four D256/D512 racecheck cases also pass in a separate run,
and the six-case synccheck suite passes with zero errors. The final repaired
source passes strict24 with unchanged goldens and all30 MXFP8 neighbors. The earlier failed
sanitizer runs remain in the private qualification report.

All six SM103 D128/D256/D512 x E4M3/E5M2 lowerings compile with CUDA 13.4.
The shared origin helper also compiles in the SM107 half D128 consumer,
including its single-batch fast path. This is compile coverage only;
SM103 execution and SM107 MXFP8 execution remain unverified.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_mxfp8_sm100.py::TestPhysicalOrigins
```


### SM100 D192/D256 quantized THD origins — 2026-11-11

`TestMidwidthPhysicalOrigins` in both SM100 FP8/MXFP8 architecture suites
collects60 L0 cases. It pins the native D192/V128 body at CGA1/CGA2 and the
D256 body at CGA1; both E4M3/E5M2 inputs write BF16 O and token/head-major
Stats. Tests cover independent mixed-width/multiplier offsets, poisoned gaps,
changed pointers/pitches, omitted total hints, empty/keyless transitions,
retained changed-input capture, raw-byte replay during concurrent work,
batch reordering/duplication/neighbor isolation and graph fallback parity.
Two persistent clusters force sequence and scheduler-slot reuse. MXFP8 data
moves independently of the original packed logical scale tiles.

The recorded runtime is SM100/148SM, driver595.71.05, Torch2.14.1+cu130,
cuDNN9.24.0.43, DSL4.8.0 with actual CUDA13.4 compiler, bindings13.4.4 and
toolkit13.4.92. Import root and native-extension hash are asserted. The
unchanged parent extension is reused; this change does not edit its ABI.
There are24 new baselines under `361d6da48de8fd55`; the earlier16 remain
byte-identical after adding CGA1 and concurrent-work coverage. Existing
repository goldens and numerical tolerances are unchanged.

The immediate parent fails four independent-origin reference controls.
Sanitizers also reproduce its dedicated D192 CGA1 schedulers' invalid DSMEM
publication and its CGA2 MXFP8 V-scale peer lifetime. Persistent compact
D192 FP8 CGA2 additionally reports scheduler field reuse hazards while its
numerical reference passes. The repair publishes locally at CGA1, broadcasts
every decoded field before returning slot credit, and retains the MXFP8 peer
until V-scale tail accesses finish. Earlier positive logs are preserved.

The final source passes all 60 cases with strict golden comparison and 38
existing neighboring-path controls (98 passed, no skips). Directed memcheck,
racecheck and synccheck each pass all 12 family/width/CGA/dtype cases, with
zero sanitizer errors; racecheck also reports zero hazards or warnings.
These sanitizer cases retain reference, rebinding and capture/replay checks;
concurrent work and batch invariance are covered by the full 60-case suite.
All 12 final SM103 lowerings compile with CUDA 13.4. SM103 execution, SM107
and backward remain unverified by this suite.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestMidwidthPhysicalOrigins \
  sdpa/frost/test_sdpa_fwd_mxfp8_sm100.py::TestMidwidthPhysicalOrigins
```

### Native SM107 D128 FP8 THD origins — 2026-11-11

`test_sdpa_fp8_sm107.py::TestD128Fp8PhysicalOrigins` collects 12 L0 cases:
two shared public-graph regressions, four independent-origin replay cases,
two missing-total-hint cases and four golden cases. Both E4M3/E5M2 inputs
write BF16 O and token/head-major Stats. The expanded cases pin two persistent
clusters, the native SM107 module and the prepared binder, then check references,
independent address/stride rebinding, retained live/empty capture, batch
reordering/duplication/neighbor isolation and raw-byte replay with concurrent work.

All ten offline CUDA 13.4 variants compile: two encodings with token, head,
padded or omitted Stats under physical origins, plus their dense controls.
Native SM107 under actual CuTe DSL CUDA13.4, DSL4.8.0, Torch2.14.1+cu130
and cuDNN9.24.0.43 passes all12 strict cases with zero failures/skips and
four new byte goldens. Functional checks pass before recording; recording
and strict comparison run in separate processes. Directed memcheck4,
racecheck4 and synccheck4 pass with zero errors/hazards/warnings. The native
binder is rebuilt with CUDA13.4.92 and Python3.12.12. Production source is
identical to the tested snapshot; the parent stack's later test-only optional
reference initialization does not change numerical operations.

The two shared regressions fail against the unfixed parent at the Stats
comparison, then pass with this change. The four existing `test_fp8_thd`
neighbors report three passes and one existing causal E5M2 numerical failure
on this SM107 part (`0.043079853 > 0.04`). A same-device parent/candidate
comparison records identical quantized input payloads/scales, O/reference
bytes and Amax for all four cases, including that failure. Its tolerance and
existing device-specific edge marker are unchanged; it is not counted as a
passing case. Other native SM107 widths and backward remain unverified here.

```bash
cd test/python
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 sdpa/frost/test_sdpa_fp8_sm107.py::TestD128Fp8PhysicalOrigins
```

#### Refresh after the #2220 develop merge

The integrated native D128 implementation passes12 strict cases plus all four
compact THD neighbors, with zero failures/skips. Its memcheck4, racecheck4
and synccheck4 runs are clean; all ten offline CUDA13.4 variants still compile.
The comparison control contains the same develop schedule with only this
D128 physical-origin change reverted. Its four compact neighbor cases also
pass, with byte-identical inputs/scales, O/reference and Amax.

The incoming #2220 `_E11` schedule normalizes O with the unquantized FP32
row sum instead of the earlier quantized-P ones-MMA denominator. Replaying
the original source reproduces its four old golden files exactly; comparing
against the integrated source preserves input/runtime and Stats hashes while
changing O. Maximum O changes are0.01171875 for E4M3 and0.0234375 for E5M2,
identical for token/head Stats layouts. Independent references, byte replay,
concurrent work and batch invariance pass before recording the four updated
O hashes, then strict comparison passes in another process. No tolerance or
skip was changed. The earlier pre-#2220 E5M2 edge and old golden mismatch
remain historical evidence; neither is a failure on this integrated head.

### Native SM107 D192/D256/D512 FP8 THD origins — 2026-11-11

The selected architecture entry `test_sdpa_fp8_sm107.py` exposes
`TestD192Fp8PhysicalOrigins` and `TestWideFp8PhysicalOrigins`: 30 L0 cases
cover E4M3/E5M2 inputs with BF16 O and token/head-major Stats. The native
D192/V128 CGA2, D256 CGA1 and D512 CTA-pair role-split bodies use independent
Q/K/V/O/Stats physical offsets. Tests pin two persistent clusters, require
the native template and prepared binder, compare independent references,
and check retained live/empty captures, fresh addresses and strides, missing
total hints, batch reorder/duplication/neighbor isolation and concurrent-work
raw-byte replay. Twelve golden cases preserve fixed quantized inputs/scales.

Initial D192 qualification under native SM107, actual CUDA13.4, DSL4.8.0,
Torch2.14.1+cu130 and cuDNN9.24.0.43 passed10 strict cases,8 existing THD
neighbors and four cases under each of memcheck/racecheck/synccheck. Two
old-source controls fail because the prepared plan omits physical origins;
this detector stops before GPU execution, so it is not an old-source numerical
failure. All eight compact controls pass on that parent.

Final D192/D256/D512 qualification, including the Q acquire fold, passes
30 strict new cases plus22 existing compact THD neighbors,0fail/skip. The
parent passes the same22 compact controls; six new-origin controls fail at
the missing-binding assertion before GPU execution. Memcheck12, racecheck12
and synccheck12 are clean with zero errors/hazards/warnings. All30 CUDA13.4
offline variants compile: three widths, two encodings and token/head/padded/
omitted Stats plus dense controls. Twelve new golden files pass a separate
strict process; the four D192 files exactly match the initial qualification.
Other GPUs, datatypes, backward, split, packing and paged routes are outside
this added coverage.

```bash
cd test/python
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestD192Fp8PhysicalOrigins \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestWideFp8PhysicalOrigins
```

### Native SM107 D256 MXFP8 THD origins — 2026-11-11

`test_sdpa_fp8_sm107.py::TestD256Mxfp8PhysicalOrigins` exposes10 L0 cases
for E4M3/E5M2 inputs, E8M0 block scales, BF16 O and token/head-major Stats.
CGA1 nonpaged, unsplit, unpacked plans bind independent Q/K/V/O/Stats
origins. The data maps select each physical sequence while scale descriptors
retain packed batch zero and logical per-sequence tile prefixes. Fixed
quantized values and scale tiles survive batch reorder/duplication/isolation,
changed addresses/pitches and retained live/empty CUDA Graph capture. The
suite independently checks numerical references and concurrent-work raw bytes.

Native SM107 with actual CUDA13.4, DSL4.8.0, Torch2.14.1+cu130 and
cuDNN9.24.0.43 passes all10 strict cases,0fail/skip, after four golden cases
pass recording and separate strict processes. The two old-source controls
stop at the missing-origin-binding assertion before GPU execution. All ten
offline variants compile: both formats with token/head/padded/omitted Stats
and dense controls. All24 compact prepared neighbors pass on both parent
and candidate, including changed scale bindings, all four output dtypes,
optional Stats/Amax, SF carriers and no-allocation/no-sync checks.
Memcheck4, racecheck4 and synccheck4 pass with zero errors/hazards/warnings.
Other widths, GPUs, backward, split, packing and paging remain outside this
new suite; their existing admission is unchanged.

```bash
cd test/python
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 sdpa/frost/test_sdpa_fp8_sm107.py::TestD256Mxfp8PhysicalOrigins
```

### SM100 half / per-tensor FP8 D512 THD physical origins — 2026-11-11

`test_sdpa_fwd_fp8_sm100.py::TestD512PhysicalOrigins` collects 42 L0 cases:
4 shared `test_mhas_v2` regressions, 12 reference/rebind/replay/batch cases,
6 no-total-hint cases, 8 half head-envelope cases and 12 goldens. FP16/BF16
each run both the role-split and 2x2 half implementations; E4M3/E5M2 run
the FP8 implementation. Every
expanded case asserts the actual kernel template, so a sibling implementation
cannot satisfy its check. All plans retain the existing unsplit/unpacked
contract and cover token/head-major Stats.

Independent Q/K/V/O/Stats origins use mixed INT32/INT64 buffers and element
multipliers. The fixture poisons input gaps with NaNs and checks output
canaries, independent FP64 references, fresh pointer/stride bindings and
retained capture after changing live lengths, origins and scalar scales.
All-empty query work must leave output padding untouched. Exact-byte checks
cover repeats, batch reorder/duplication, an isolated sequence and changed
neighbors. Prepared and standalone graph execution must agree. D512 uses
16-element Q/K multipliers so its +16 head-padding variant still represents
whole physical token rows exactly. Eight additional half cases cover
D384/V448 and D264/V504 in both implementations and dtypes; the latter
uses an 8-element origin unit that divides its declared token stride.

Measured runtime: SM100, 148 SMs; driver 595.71.05; Torch 2.14.1+cu130;
cuDNN 9.24.0.43; DSL 4.8.0/compiler CUDA 12.9; tvm-ffi 0.1.14.post1.
The final suite passes all 42 cases in 52.62 seconds, with zero failures or
skips, including strict checks of twelve new goldens under
`361d6da48de8fd55`. These were recorded after their references and byte
comparisons passed; earlier goldens and tolerances are unchanged. The
paired-host ABI detector passes separately. All 23 neighbor controls pass:
compact D512 FP8 THD, dense output/amax, existing half dense/THD and D576
physical-origin packed/split paths.
The immediate parent fails all three shared D512 dtype cases. The first
expanded candidate exposed seven 2x2 declines caused by an obsolete D576-only
origin compile guard; removing it enables the existing D512 origin body.
All six SM103 lowerings compile offline; execution there remains unverified.

Compute Sanitizer memcheck, racecheck and synccheck each pass eight cases:
the six token-major replay legs and both BF16 D264/V504 implementations.
All report zero errors; racecheck reports zero hazards and warnings. Runtimes
are 6.64, 341.42 and 4.67 seconds respectively. The private launcher initializes
cuda-python before Torch for the procedure-lookup issue documented above;
API reporting remains explicit, with all kernels instrumented and error exit
code 86. No sanitizer diagnostics are suppressed.

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestD512PhysicalOrigins
```

### SM100 D256 paged THD packed split — 2026-11-11

`sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestPagedD256PackedSplit` is collected
by the existing SM100 CI entry, with all six cases marked L0. Four public graph
cases cover FP16/BF16, CGA2, split2/3/5/9, unpacked and packed heads (including
partial GQA6), HND/NHD/mixed page pools, page8/16/128/256, independent physical
Q/O/Stats offsets and token/head Stats. They compare an independent CPU FP64
reference, preserve poisoned gaps and inputs, rebind fresh storage, and replay
changed data/offsets/tables/lengths through retained CUDA Graphs. Route assertions
pin the D256 native body and native binder.

The two standalone dtype cases hold split5 and packing fixed while comparing
raw O/FP32 Stats bytes across fresh-buffer replay with concurrent work, sequence
alone/batched/reordered/duplicated, changed-neighbor Q and129 requests. Lengths
include zero and fully masked query rows. Independent references check both
outputs and signed infinity sentinels before two small runtime-keyed goldens
are recorded. These tests cover forward only; no backward capability is added.

New measurement: six recording cases pass; the subsequent strict run passes all
50 cases (six public/golden cases,36 native binding cases, eight existing
paged controls), zero failures/skips. Both goldens match unchanged. The four
public origin cases pass memcheck, racecheck and synccheck with zero sanitizer
errors/hazards/warnings. Four SM103 half dtype/packing variants compile offline;
this is not execution evidence. Runtime:
SM100/148SM, driver595.71.05, cuDNN9.24.0.43, Torch2.14.1+cu130,
CuTe DSL4.8.0 with actual compiler CUDA13.4, cuda-bindings13.4.4 and toolkit13.4.92.
The Torch wheel's CUDA13.0 tag is separate from the asserted DSL compiler.
The two JSON records were inspected after reference/replay passed. Strict
comparison must leave `CUDNN_DETERMINISM_RECORD` unset. SM103 execution is
unverified; this evidence does not establish cross-runtime bitwise stability.

```bash
cd test/python
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -m L0 sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestPagedD256PackedSplit
```

### SM100 D512 TMEM reader lifetime — 2026-11-12

`TestD512PhysicalOrigins.test_tmem_release_waits_for_all_readers` adds three
L0 cases (BF16, E4M3, E5M2). A temporary template delays one output warp and
records completion of each warp's final TMEM load and the deallocation.
It uses the existing synthetic physical-origin fixture, including reference,
batch-invariance, changed-binding and retained live/empty graph checks. The
assertion compares event order; it has no duration threshold. No production
debug argument is added.

Before the terminal reader join, all three cases fail the lifetime assertion
in 10.96 seconds, although their numerical checks pass. With the join, the
full D512 class passes all 48 cases in 94.30 seconds, with no failures/skips.
This includes the twelve unchanged committed goldens and the existing Stats
reader checks. Runtime: SM100/148SM, Torch 2.14.1+cu130, cuDNN 9.24.0.43,
DSL 4.8.0 with actual compiler CUDA 13.4; all three artifact caches disabled.
The same native extension is reused because no native source changed.

The three new cases also pass Compute Sanitizer 13.4 memcheck and synccheck
(12.46 and 12.29 seconds), with zero errors. The retained four-case selection
passes racecheck 13.2 in 179.45 seconds, with zero hazards/errors/warnings;
kernel compilation remains CUDA 13.4. These are local candidate results.

This establishes and repairs a TMEM lifetime violation. It does not by itself
attribute or clear the separately retained instrumented numerical failure.
SM103 execution remains unverified; no golden or tolerance is changed.

```bash
cd test/python
python -m pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::TestD512PhysicalOrigins
```

### D512 paired-CTA reader completion — 2026-11-12 follow-up

The terminal four-warp join above still allowed an MMA warp to proceed
before its own CTA's readers finished: its barrier counted only the peer
CTA's completion. The role-split half/FP8 paths now send one elected local
credit and one elected peer credit after each four-warp join. Each MMA warp
waits for both reader groups. This retains the existing 2x2 path's
local-plus-peer completion requirement and adds no mainloop synchronization.

The lifetime regression now has nine cases: BF16/E4M3/E5M2 crossed with a
delay in both output CTAs, only the even CTA, or only the odd CTA. It checks
both CTAs' final TMEM loads against each deallocation. On unchanged
`b70a558b4ea2`, the three symmetric cases pass but all six one-sided cases
fail the event-order assertion (28.34s); output numerics still pass there.
With the two-credit protocol, all nine pass (29.28s). A separate synthetic
racecheck control removes only the local credit and its count from the
candidate's temporary template: the even-CTA BF16 case fails the same
ordering assertion, with zero reported hazards. That control is not a
reproduction of the reviewer's numerical failure.

The current-head reviewer still observed numerical/repeatability failures on
`b70a558b4ea2` after the earlier join. Those results remain valid; this new
local evidence does not by itself clear the reviewer's original setup.
No seed, reference tolerance or golden is changed.

The expanded full D512 class passes 54 cases, zero failures/skips, in 101.85s
on the same SM100/148SM runtime, including all 12 unchanged committed
goldens. All artifact caches remain disabled. Four changed SM103 variants
(FP16/BF16 half and E4M3/E5M2 per-tensor FP8) compile offline with CUDA 13.4;
SM103 execution remains unverified. The 2x2 and native binding sources are
unchanged.

CUDA 13.4 memcheck and synccheck each pass all nine lifetime variants with
zero errors (34.04s and 30.95s). The separate racecheck runs pass the
original four-case selection/order (182.94s) and an even-CTA BF16 lifetime
case (56.87s), with zero hazards/errors/warnings. Kernel compilation remains
CUDA 13.4 when using the available 13.2 racecheck tool. These local passes
do not replace requalification in the original reviewer environment.

### SM100 single-sequence physical-origin setup — 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestSingleSequenceSetup` collects
18 new L0 cases through the existing CI entry. The helper is intentionally
not named `test_*`, so a directory sweep does not collect a second copy.
The four dense D128 physical-origin cases cover FP16/BF16, packed/unpacked
CGA2, independent element offsets, changed addresses/strides, token/head
Stats, retained capture and empty/keyless replay. Four paged B1 cases cover
D64/D128, CGA1/CGA2, HND/NHD/mixed KV layouts and independent Q/O/Stats
origins. Ten FP16/BF16 byte cases cover dense D128 packed/unpacked, paged
D64 CGA1/CGA2 and paged packed D128 CGA2, all with split count one.

Each byte case uses fixed CPU-generated sequence payloads and independent
CPU FP64 O/Stats references. It compares B1 with reordered, duplicated and
changed-neighbor B3/B5/B129 bindings through one capacity-4096 plan, then
returns to B1. Fresh buffers, concurrent GPU work, poisoned workspace,
empty/keyless sequences and retained CUDA Graph replay remain checked.
Paged payloads are copied once into finite-poisoned pools; unused page tails
cannot change the reference. Input bytes, O bytes and FP32 Stats bytes are
checked separately. Negative-infinity Stats is accepted only at keyless rows.
There is no backward setup change or new backward qualification.

On local SM100 (148 SM), actual CUDA 13.4 compiler/sanitizer, DSL4.8,
Torch2.14.1+cu130, cuDNN9.24.0.43, driver API13020 and TVM FFI0.1.14.post1,
the unchanged kernel parent `cb4c4e759873` passed all25 selected L0 cases
(18 new, six existing D64 paged-origin controls, one compact B4 control),
with one L1 case deselected, in68.41s. Ten new runtime goldens were recorded
under `361d6da48de8fd55`; no existing golden was changed. The final candidate
passed the same25 cases, zero failures/skips and one deselection, in70.12s
with recording unset and every new golden checked against that baseline.
Actual imports and native compatibility were asserted; all three compiled
cache controls were disabled. The generic multi-sequence setup body and
existing native source are unchanged.

CUDA13.4 memcheck and synccheck each passed the packed BF16 dense-origin
case and D64 CGA1 paged-origin case, with zero errors (10.93s and8.38s).
Offline compilation passed six SM103 half variants (FP16/BF16 times paged
D64 CGA1, dense D128 CGA2 and packed paged D128 CGA2), plus two SM107 dense
D128 CGA2 controls of the shared helper's unchanged default. GPU execution
on SM103 and new SM107 execution remain unverified; compilation is not a
runtime pass. This setup-only optimization changes neither the attention
arithmetic nor eligibility/ranking, and does not promise cross-plan bytes.

From `test/python`, after checking that `cudnn` imports this checkout:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1 CUTE_DSL_NO_CACHE=1 CUTE_EXPERIMENTAL_DSL_NO_CACHE=1 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 \
  sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestSingleSequenceSetup \
  sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestPagedD64PhysicalOrigins \
  sdpa/frost/test_sdpa_fwd_thd_origins.py::test_compact_thd_origins_use_single_load_maps
```

The first baseline attempt copied unsupported SM107 configurations and
failed10 cases before candidate testing: dense D64 origins, dense D128 GQA
CGA1 and unsplit packed D64 are outside these SM100 routes. The final suite
uses supported dense D128 and paged D64/D128 routes; reference tolerances
and byte assertions were not weakened. The earlier failed run is not counted
as successful coverage.

### SM100 unified D64/D128 split setup — 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestUnifiedSplitSetup` adds18 L0
cases through the existing SM100/SM103 CI entry, with a helper that is not
independently collected. Eight B1 paged-origin cases cover D64/D128,
FP16/BF16, packed split5 and unpacked split3, HND/NHD pools and token/head
Stats. Two nonpaged D128 split3 cases cover both half types. Two D128
mixed-KV cases cover FP16/E4M3 paged and BF16/E5M2 nonpaged split3.
They retain independent origins, changed storage/strides, numerical O/Stats
checks and changed-input capture replay, including empty/keyless queries.

Six split5 byte cases cover packed paged D64/D128 and unpacked nonpaged
D128 for FP16/BF16. One declared-capacity256 plan alternates through runtime
B1/B3/B5/B129 and back to B1, including reordered/duplicated sequences,
changed neighbors, concurrent work, independent physical origins, poisoned
workspace and CUDA Graph replay. Fixed CPU-generated payloads and CPU FP64
references remain unchanged from the unsplit fixture. Split count and packing
stay fixed; there is no assertion of cross-plan equality. The appended
`splits=1` helper default preserves the existing unsplit tests and golden names.

On local SM100 (148 SM), actual CUDA13.4 compiler, DSL4.8,
Torch2.14.1+cu130 and cuDNN9.24.0.43, parent `21abdcc482a1` passed all18
cases in50.89s and recorded six new goldens under `361d6da48de8fd55`.
The candidate passed the same18 in50.09s, zero failures/skips, with recording
unset and all six O/Stats hashes checked against the parent. No existing
golden changed; the new input hashes also match the existing unsplit fixtures.
All three compiled cache controls were disabled and source imports asserted.
Earlier candidate qualification on local SM100 (68 SM), with the same compiler
and library versions, passed13 existing physical-origin cases in35.75s and
24 mixed-KV cases in70.21s, including unsplit and multisequence controls.

CUDA13.4 memcheck and synccheck each passed three focused cases in12.43s
and10.83s, with zero errors: packed B1 BF16 D64 paging, B1 FP16/E4M3 D128
paging, and BF16 nonpaged D128 B1/B3/B5/B129 byte comparisons. Offline
SM103 compilation passed ten split-host variants: both half types times
paged D64, paged/nonpaged D128 and paged E4M3/nonpaged E5M2 KV.
This is compilation evidence only; SM103 execution remains unverified.
The change affects setup only and adds no backward qualification.
Four existing unsplit controls also passed in13.60s, retaining their original
goldens. CI-style collection with `-m L0 -k frost` finds all18 new cases.

From `test/python`, after checking the actual imported checkout:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1 CUTE_DSL_NO_CACHE=1 CUTE_EXPERIMENTAL_DSL_NO_CACHE=1 \
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 -k frost \
  sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestUnifiedSplitSetup
```

The first parent-only attempt failed15 cases before kernel launch: the paged
B1 fixture retained a packed-total hint above `B*S_q`, and the shape-override
byte fixture omitted its required split packed-total bound. The corrected
fixtures declare valid bounds; no numerical tolerance or byte assertion was
relaxed. Those failures are retained separately from passing qualification.

### Native SM107 D64 paged THD split forward — 2026-11-12

`TestNativeD64Paged` in the CI-selected `test_sdpa_fwd_dsl_sm107.py` covers
matching FP16/BF16 paged THD at CGA1/N128,
with fixed split counts and packed or unpacked GQA. The shared physical-D64
single-Q body is compiled for SM107 and writes FP32 partials; the final
combine retains independent O/Stats origins. Mixed FP8 KV, split sinks and
padded Stats retain their existing declines. No backward path is added.

On SM107, cuDNN 9.30.0.2, CuTe DSL 4.8.0 compiled with CUDA 13.4 and
Torch 2.14.0+cu130, 20 distinct focused cases pass: eight split origin/replay
cases, eight existing unsplit controls, one admission case, one partial-GQA
multi-wrap case and two replay/batch/golden cases. The two new goldens also
pass strict comparison in a fresh process. No existing golden is replaced.

The batch cases preserve fixed Q/K/V payloads across sequence insertion,
reordering and duplication, including empty sequences, fresh output/workspace
allocations and concurrent GPU work. Both O and Stats retain identical raw
bytes under the same plan; independent FP64 checks remain separate. The graph
cases check HND/NHD/mixed pools, physical origins and strides, changed-input
capture replay and empty replay. The six-head sharing group exercises partial
packing with more KV iterations than ring slots. The initial three-head-group
fixture was correctly declined by the existing packing contract; the final
case uses a supported group without widening that contract.

```bash
cd test/python
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 -k frost \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_admission \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_thd \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_thd_split_partial_pack \
  sdpa/frost/test_sdpa_fwd_dsl_sm107.py::TestNativeD64Paged::test_thd_split_determinism
```

These new measurements qualify the SM107 extension only. Existing SM100/103
paged-split support is unchanged and receives no new hardware coverage here.

### SM107 D128 per-tensor FP8 single-CTA dense splits — 2026-11-12 (UTC)

`test_sdpa_fp8_sm107.py::TestNativeD128Fp8Cga1Split` exposes12 L0 cases
through the selected Rubin CI entry: one graph-domain check, six functional
rebind/capture cases, two replay/batch cases, two golden cases and one
refilled-ring stress. E4M3/E5M2 input, BF16/FP16/E4M3/E5M2 output, packed
and unpacked groups, splits2/3/9, dense/causal/bottom-right/SWA/sink,
optional Stats and D112/V96 plus D80/V112 envelopes are exercised.
Every functional check uses the shared FP64 FP8 probability-cast reference
and checks Amax_O separately. Native template, single-CTA geometry,
four-stage KV ring, K64 MMA and FP32 combine are asserted.

On SM107216SMs, driver620.62, fullcuDNN9.30.0.4, CuTeDSL4.8.0 compiled
with CUDA13.4 and Torch2.14.0+cu130, all12 cases passed in25.88s.
The two new goldens also passed strict comparison in a fresh process
(2 cases,3.21s). No previous golden was changed. The stress recorded zero
wrong bytes in512 warm and512 poisoned launches; its unsafe control failed
200/200 launches. The stress shape launches512 blocks and refills its ring
four times per launch; numerical correctness is checked separately first.

Fixed-payload batch checks compare the target alone, reordered, duplicated,
with changed neighbors and among129 sequences. O/Stats are per-sequence;
Amax_O is a whole-batch maximum, checked numerically after rebatching and
bitwise during replay. New byte guarantees cover E4M3/BF16 packed split3
and E5M2/FP16 unpacked split2. Other GPU lines are unmeasured by this suite;
this change adds no backward or THD path.

From `test/python`, with the correct checkout imported:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -q -m L0 -k frost \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestNativeD128Fp8Cga1Split
```

### SM100/103 FP8 coalesced paged loads — validation, 2026-11-12 (UTC)

`sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py` now also covers D64/D128
paging, preserving its existing D512 defaults. Five new configurations each
have a repeatability/batch-invariance test and a strict golden test:

| D | Page | CGA | Split | Layout | Input / output |
| --- | --- | --- | --- | --- | --- |
| 64 | 8 | 1 | 1 | NHD | E4M3 / BF16 |
| 128 | 16 | 1 | 2 | HND | E4M3 / E4M3 |
| 128 | 8 | 2 | 1 | HND | E5M2 / FP16 |
| 64 | 64 | 1 | 4 | NHD | E5M2 / E5M2 |
| 128 | 128 | 1 | 4 | NHD | E4M3 / BF16 |

The suite fixes quantized payloads and scales, uses independent K/V page
orders, and checks O, FP32 Stats and Amax numerically. Replay compares all
three outputs bytewise with fresh buffers and concurrent work. The per-row
O/Stats checks cover isolated, reordered, duplicated and changed-neighbor
sequences plus a 129-sequence batch; Amax is a whole-batch reduction.

Both B200 (SM100, 148 SMs) and B300 SXM6 AC (SM103, 148 SMs) pass
**37 cases, 16 deselected, no failures or skips**: the ten new byte cases,
one refilled-ring stress, twelve existing page-boundary cases extended to
page8, five two-KV-stream paged cases, eight D128 prepared-stride/capture
cases and one existing D512 helper control. Runtime was 100.58s on B200 and
127.37s on B300. The single warning is Torch's synchronization-debug mode.

Five goldens per GPU were recorded from the unchanged parent kernel only
after independent reference/replay checks, then strictly matched by the
candidate in a fresh process with recording disabled. The ten new JSON
records live under `c8aefc970ef3af15` and `98a778ce33bb9402`; no existing
record is replaced. The initial fixture requested an unsupported D64/CGA2
combination and was corrected to the existing CGA1 domain before this run;
no production admission or numerical tolerance was changed.

The stress case traverses eight KV tiles and refills its four-stage ring,
checking 256 warm and 256 shared-memory-poisoned launches. Both GPUs record
zero wrong bytes; their unsafe controls fail 200/200 launches. The tested
kernel grid has only four blocks (0.03 per SM), so this is a low-occupancy
rate bound, not a saturated-scheduling race qualification. An independent
numerical check precedes the stability stress. GPU-exclusive scheduling
markers were added after serial qualification; test bodies are identical.

B200 also passes Compute Sanitizer 2026.3 memcheck and synccheck on the
page8/HND/E4M3 mixed-length case, one case each, with 0 errors. No new racecheck
measurement is claimed. The sanitizer processes report FE cuDNN 93000 while
Torch's cuDNN query reports 92400; this explicit FROST case does not execute
backend attention. The normal qualification uses full cuDNN 9.30.0.4,
Torch 2.14.1+cu130, CuTe DSL 4.8.0 and actual compiler/NVRTC 13.4; driver API 13040.
Source/import hashes were verified against parent
`32d4554e93f2564be0a005d305e458e72551e1c9` plus the loader/test patch.
The relevant kernel/test files are unchanged on subsequent develop
`f013a9d1e`. SM90/SM107/SM120 are unmeasured by this suite. This modifies
forward paging only; no paged-KV backward API is added.

From `test/python`, with the qualified checkout imported and recording unset:

```bash
CUDNN_DETERMINISM_GOLDENS="$PWD/determinism_goldens" \
  python -m pytest -n0 -m L0 -q -rs \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py \
  -k 'small_paged or paged_loader_warm_stress'
python -m pytest -n0 -m L0 -q \
  sdpa/frost/test_sdpa_fwd_paged_sm100.py::test_paged_graph_fp8_matches_reference \
  sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py::test_paged_two_kv_streams_match_the_reference
python -m pytest -n0 -m L0 -q \
  sdpa/frost/test_sdpa_prepared_fp8_paged.py::test_prepared_fp8_paged_strides_and_capture -k128
python -m pytest -n0 -m L0 -q \
  'sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_d512_paged_replay_and_batch_invariance[fwd-d512-e4m3-bf16-split1-NHD]'
```


### SM100 FP8 interleaved two-KV-stream schedule — validation, 2026-11-12 (UTC)

The SMEM-P dual-stream loop issues QK(A), PV(previous A), QK(B), then
PV(previous B). Each stream retains its previous arithmetic order, all waits,
arrivals, credits and ring phases. SM103 uses the unchanged TMEM-P branch.
No eligibility, tuning domain, ranking, or backward path changes.

B200 and B300 SXM6 AC (148 SMs each) both pass **62 cases, 16 deselected,
no failures or skips**. This reuses the preceding coalesced-page qualification
and runs the entire 30-case `test_sdpa_fwd_fp8_dual_kv_sm100.py` suite in place
of its five paged cases. Coverage includes dense and paged KV, odd tails,
empty splits, persistent scheduler waves, deliberately delayed MMA,
independent K/V strides, changed-input capture, replay and batch invariance.
Candidate runtimes were 132.38s on B200 and 95.96s on B300.

Each GPU also passes five parent golden checks (38.64s and 39.82s), recorded
in an isolated directory from the coalesced parent and strictly matched by
the candidate in a fresh process with recording disabled. All ten parent
JSON records are byte-identical to the committed baselines; none is updated.
The existing refilled-ring stress reports zero mismatches in 256 warm and
256 poisoned launches on each GPU, with unsafe controls failing 200/200.
Its four-block geometry remains a limited-occupancy measurement; separate
numerical tests provide the reference checks.

B200 Compute Sanitizer 2026.3 synccheck and racecheck each pass the existing
`page16-unaligned` dual-stream test: two sequences, Q1/H64:8/D128, live KV
lengths 1024/777, eight/seven KV tiles, and repeated ring reuse. Synccheck
reports 0 errors; racecheck reports 0 hazards (0 errors, 0 warnings). Each
sanitizer invocation runs one case, four deselected. The sanitizer processes
report FE cuDNN 93000 and Torch cuDNN 92400; the tested explicit FROST path does
not execute backend attention. Normal tests use full cuDNN 9.30.0.4.

Both GPUs use Torch 2.14.1+cu130, CuTe DSL 4.8.0 and actual compiler/NVRTC 13.4.
Imports and source/library hashes are checked. The runner disables the
editable-install finder and gives child interpreters the same source path.
The shared runner has no pytest-xdist; tests run serially with
`CUDNN_TEST_NO_ISOLATION=1`. Its two warnings are the unavailable
`xdist_group` marker and Torch synchronization-debug mode.
The kernel and test files on landed parent `9be74f4d7` match those qualified
here. After qualification, only the phantom-step explanatory docstrings
changed; the executable kernel AST is identical. SM90/SM107/SM120 are not
measured by this follow-up.

Reproduce the preceding qualification commands, replacing the selected
`test_paged_two_kv_streams_match_the_reference` node with the complete
`sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py` file. Omit `-n0` when xdist is
not installed. Run the sanitizer on the following case, then repeat with
`--tool racecheck`:

```bash
compute-sanitizer --tool synccheck --error-exitcode 91 --report-api-errors no \
  python -m pytest -m L0 -q \
  'sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py::test_paged_two_kv_streams_match_the_reference[page16-unaligned]'
```


### SM100 paged FP8 separate Q/K and V producers — validation, 2026-11-12 (UTC)

The existing SM100 D128 paged dual-stream body gives V production to the
O-store warp while the loader produces Q/K. V retains its own ring index and
phase across persistent work items, using the existing MMA empty credits.
Arithmetic, barrier counts, plan knobs and graph eligibility are unchanged.
SM103 retains the existing producer path; no backward path changes.

B200 and B300 SXM6 AC (148 SMs each) pass **43 cases, six deselected, no
failures or skips**. Two D128/CGA1 parent golden cases per GPU are recorded
into a fresh directory and strictly compared by the candidate with recording
disabled. All four parent records equal the committed baselines. Two existing
fixed-plan batch-invariance cases retain O/Stats bytes through reordered,
duplicated, changed-neighbor and multiple-wave batches.

The suite also runs all 30 existing dual-stream cases and eight new capture
cases with an assertion on the actual producer flag (enabled on SM100,
disabled on SM103). The new cases cover HND/NHD, independent V page tables,
packing on/off, split 1/4, scheduling policies 0/1/2, causal bounds, odd tails,
empty splits and persistent work. Each retained graph replays shorter, empty
and restored live lengths 16 times each; restored O and Amax compare raw bytes
with snapshots from the original execution. Independent numerical checks keep
the original quantization bounds. No committed golden or tolerance changes.

The existing four-block refilled-ring stress is included, with 256 warm and
256 poisoned launches and the unsafe control. Both GPUs report zero mismatches
and their unsafe controls fail 200/200. Its four-block geometry remains a
limited-occupancy measurement, not a full-device stress claim.

B200 Compute Sanitizer 2026.3 racecheck, memcheck and synccheck each pass all
eight new capture cases. Racecheck reports zero hazards/errors/warnings;
memcheck and synccheck report zero errors. Racecheck filters
`kns=cudnn_kernel_`; the other tools are unfiltered. CUDA API error reporting
is disabled. These B200 processes report FE cuDNN 93000 and Torch cuDNN 92400;
the explicit FROST attention path does not execute backend attention.
B300 has no new sanitizer result in this follow-up.

Actual compiler/NVRTC 13.4 and CuTe DSL 4.8 are checked separately from the
Torch 2.14.1+cu130 build label. Explicit FROST plans are required; source/import
and library hashes are checked. Other architectures are unmeasured by this
follow-up. The only postqualification kernel change removes experiment
wording from a comment; the executable AST is identical.

Reproduce from `test/python`, with the candidate checking the isolated parent
golden directory through `CUDNN_DETERMINISM_GOLDENS` and recording disabled:

```bash
python -m pytest -m L0 -q sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py
python -m pytest -m L0 -q -k 'd128 and cga1' \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_small_paged_golden \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_small_paged_replay_and_batch_invariance
python -m pytest -m L0 -q \
  sdpa/frost/test_sdpa_fp8_d512_paged_determinism.py::test_fp8_paged_loader_warm_stress
compute-sanitizer --tool memcheck --error-exitcode 91 --report-api-errors no \
  python -m pytest -m L0 -q \
  sdpa/frost/test_sdpa_fwd_fp8_dual_kv_sm100.py::test_paged_split_producer_lengths_replay
```

Repeat the final command with `--tool synccheck`, and with
`--tool racecheck --kernel-name kns=cudnn_kernel_`.

### 2026-11-12 — service-register CI follow-up and FP8 scale lifetime

On parent `e2fbd4f69`, converging the service register releases exposed an
extra output-scale spill in native SM107 D128 FP8. The follow-up computes
that scale in the correction role, retaining the multiply order and the
common service release. Existing spill ceilings and numerical tolerances
are unchanged. No output golden is replaced.

Native SM107, CUDA compiler/NVRTC 13.4, CuTeDSL 4.8.0, cuDNN 9.30.0.4:
the unchanged parent records four E4M3/E5M2 token/head-Stats origin goldens
after their references and replay checks pass (**4 passed, 36 deselected**).
The candidate then passes **40 cases, no skips**, including those four strict
parent records, all D128 origin checks, D64/D128 masks, D64 GPT-OSS shapes
and the D128 output-dtype cases. O/Stats batch invariance and changed-storage
replay retain their existing scope; Amax_O is checked numerically by the
dense cases. Backward, new sanitizer coverage and performance are unmeasured
by this follow-up.

Run the same selected L0 suites with `CUBLAS_WORKSPACE_CONFIG=:4096:8`:

```bash
python -m pytest -m L0 -q \
  sdpa/frost/test_sdpa_fp8_sm107.py::TestD128Fp8PhysicalOrigins \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_masks \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_d64_masks \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_d64_gptoss_shape \
  sdpa/frost/test_sdpa_fwd_fp8_sm100.py::test_fp8_output_dtypes
```

For the parent phase add `-k origin_golden` and record into a fresh directory.
The candidate uses that directory through `CUDNN_DETERMINISM_GOLDENS`, with
recording disabled. Both phases load Python and matching native bindings
from the isolated source package; the reused bindings have identical native
build inputs to this source.

B200 with CUDA compiler/NVRTC 13.4 and CuTeDSL 4.8.0 passes **61 cases, no
skips**: the complete dual-KV suite (including the delayed-MMA subprocess),
six D576 SASS cases, the THD/paged gate cubin records, both masked-softmax
pins, all four epilogue/scheduler pins, six score-off records, both score-on
store checks and the SM107 D128 service-release guard. Cross-compiling
SM103/SM107 instructions on B200 is not native execution on those targets.
The fresh child receives no inherited test-root path; an isolated CPU
control fails its import before the fix and passes afterward.

All 19 old cubin records are reproduced with the pre-convergence kernel
sources and the same DSL 4.8.0/CUDA 13.4 compiler. Sixteen change with the
common-release fix and are refreshed; the three SM107 score-off records stay
byte-identical. PTX and SASS are retained for attribution, and the existing
spill and score-store detectors remain active. CUDA 13.5 disassembles the
SM107 cubins. Native SM103 execution is unverified by this follow-up.

## SM100/SM103 D64 packed THD prefill (2026-11-13)

The new L0 `TestD64ThdPackedUnsplit` entry checks exact D64 with whole-group
PackGQA, N128, CGA1/2 and SPLIT_KV=1. Both graph and standalone calls assert
the direct native prefill body, native binding, independent origin metadata
and absent split workspace. Coverage includes FP16/BF16, HND/NHD/mixed pools,
GQA2/4/8/16, pages8/16/64/256, short-query routing, window/sink/Stats handling,
fresh strides, changed device lengths/tables, empty replay and tail canaries.
The graph batch tests hold sequence payloads fixed across reorder, duplication,
changed neighbors and B1/B3/B5/B129 under one capacity4096 plan. The refill
checks validate both cold O/Stats baselines with CPU FP64 before stressing
the four-stage KV ring with exact byte comparisons.
No backward kernel is changed or qualified by these tests.

```bash
cd test/python
python -m pytest sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestD64ThdPackedUnsplit \
  -o addopts= -m L0 -q -rs
```

B200 (148 SMs) and B300 (148 SMs), with CuTeDSL4.8.0, compiler/NVRTC13.4,
FE backend9.30.0 and Torch2.14.1+cu130, each pass all **22 unique cases**:
16 contract/functional cases, four bytewise batch/golden cases and two warm
refill cases. Each run additionally passes four strict golden rechecks and
two cases under each of memcheck, synccheck and racecheck: **32 successful
test executions per GPU**, no failures or skips. Sanitizers report zero
errors/hazards. Complete allocations took 2m52s and 2m53s respectively.
Four new golden records per GPU are committed; existing records are unchanged.
Both precisions pass256 warm and256 poisoned launches; the unsafe ring
control fails200/200. The actual persistent grid has one CTA per SM, below
the control's three-CTA geometry, so the rate bound is specific to that grid.
Forty-two existing host config/capability regressions and eight new contract
checks also pass locally. Changed-file pre-commit and fern check pass.

The original qualification snapshot's production and test bytes match
`e4e1e46c53567673b1c728e388e890f6f0e04083`, based on
`fed09b0fa691273435e993676f4bfb28c28a0461`. The review follow-up adds the FP64
checks to the two actual cold launches; it changes no production code or
golden bytes. Its two updated stress cases pass on each B200 and B300, with
no skips, using the same runtime and unchanged 256 warm/256 poisoned counts.
The qualified helper SHA256 is `ceef49f454fc3ff386680d00eb728b78a862e7b3b0320ab149c688ea9713847f`.
The phase commands are below (from `test/python`, using the runtime above). The recording test writes only after its numerical and
bytewise batch checks pass. Strict rechecks always disable recording.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_TEST_NO_ISOLATION=1
entry=sdpa/frost/test_sdpa_fwd_dsl_sm100.py::TestD64ThdPackedUnsplit
golden_dir=$(mktemp -d)
env -u CUDNN_DETERMINISM_RECORD -u CUDNN_DETERMINISM_GOLDENS \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short \
  -k 'not batch_and_golden and not warm_refill'
env -u CUDNN_DETERMINISM_GOLDENS CUDNN_DETERMINISM_RECORD="$golden_dir" \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short -k batch_and_golden
env -u CUDNN_DETERMINISM_RECORD CUDNN_DETERMINISM_GOLDENS="$golden_dir" \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short -k batch_and_golden
env -u CUDNN_DETERMINISM_RECORD -u CUDNN_DETERMINISM_GOLDENS \
  python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short -k warm_refill
for sanitizer in memcheck synccheck racecheck; do
  env -u CUDNN_DETERMINISM_RECORD -u CUDNN_DETERMINISM_GOLDENS \
    compute-sanitizer --tool "$sanitizer" --error-exitcode 91 --report-api-errors no \
    python -m pytest "$entry" -o addopts= -m L0 -q -rs -x --tb=short \
    -k 'test_paged_origins and (g2 or g4)'
done
```

The retained development smoke passed two N128 paged origin/replay cases;
it also exposed the existing N64 dense-only factory boundary, which is now
rejected by capability and standalone checks before compilation. An earlier
config admission failure and a synthetic fixture missing its padding flag
are retained as development failures; neither is a successful GPU test.
