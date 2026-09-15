# Same-device SM120 three-path comparison

Date: September 15, 2026. These are new local measurements, not ratios of historical runs on different GPUs.

## Setup and fairness

- One RTX PRO 6000 Blackwell Server Edition, SM120, 188 SMs.
- BF16 Q/K/V BHSD `[1, 8, 142720, 128]`; PyTorch `2.13.0+cu130`; seed `20260915`.
- Current native blk128: branch revision `97102524`, kernel checkpoint `e0602027`.
- Native blk64: the checkout's original SM120 CuTe BSA kernel. Its source and BSA shared utilities are unchanged from PR #1010.
- Closed PR #1010: exact archived API, interface, and metadata source at `1d5ed9d5`, loaded as an isolated package with the same installed dependencies. The archived files are byte-checked against Git before timing.
- All three paths use the same Q/K/V and exactly the same selected KV tokens. For 15% density this is 167/1115 KV128 blocks or 334/2230 KV64 blocks (actual 14.9776%); at 20% it is 223/1115 or 446/2230.
- Native64 metadata is expanded once **outside** timing. PR #1010 expands metadata **inside every timed API call**, as its public implementation does. No Q/K/V reorder or quantization is applied.
- Each process uses 12 warmups and 102 timed calls per path per case. All six permutations of the three paths are repeated equally, so each occupies each timing position equally often.
- CUDA-event timings cover warmed public-wrapper execution. Compilation, mask preparation for native64, and validation are excluded. No concurrent GPU benchmark, profiler, or clock adjustment was used.
- This is the open-source cuDNN BSA path, not an opaque dense cuDNN backend engine. It is an equal-mask comparison, not a sweep over independently selected blk64 masks.

## Independent run 1

| Density | Pattern | Native blk64 (ms) | PR #1010 (ms) | Current native blk128 (ms) | vs blk64 | vs PR #1010 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 14.9776% | strided | 35.0322 | 35.0517 | 31.2788 | 1.1200x | 1.1206x |
| 14.9776% | local | 35.2330 | 35.2657 | 31.3648 | 1.1233x | 1.1244x |
| 20.0000% | strided | 47.2286 | 47.2497 | 42.1145 | 1.1214x | 1.1219x |
| 20.0000% | local | 47.6372 | 47.6697 | 42.4035 | 1.1234x | 1.1242x |

## Independent run 2

| Density | Pattern | Native blk64 (ms) | PR #1010 (ms) | Current native blk128 (ms) | vs blk64 | vs PR #1010 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 14.9776% | strided | 35.4504 | 35.4771 | 31.6873 | 1.1188x | 1.1196x |
| 14.9776% | local | 35.6599 | 35.6702 | 31.7743 | 1.1223x | 1.1226x |
| 20.0000% | strided | 47.6678 | 47.6842 | 42.4655 | 1.1225x | 1.1229x |
| 20.0000% | local | 49.3349 | 49.3656 | 43.7443 | 1.1278x | 1.1285x |

Across the two runs, current native blk128 reaches 1.1188x-1.1278x versus native blk64 and 1.1196x-1.1285x versus PR #1010. This is approximately 12%-13% more throughput or 10.6%-11.4% lower latency. Absolute timings drift; each ratio compares paths interleaved in the same process on the same device.

The measured PR #1010 versus native64 median difference is about 0.010-0.033 ms, while current native128 saves about 5.1-5.6 ms at 20% density. Most of this improvement is therefore associated with native128 kernel execution, not elimination of metadata conversion. The small difference also includes wrapper/launch differences and measurement variation; it is not a separately isolated conversion-kernel timer.

The earlier 0.8%-1.1% gain used the already optimized FA4-style native128 `9869b9b6` as its baseline. It is not the baseline used here, and these results do not claim that the additional 1.05x goal against `9869b9b6` has been achieved.

## Validation

- After promotion to the portable public harness, the complete BSA suite passed
  **65 tests with 19 skips**, with all 12 new harness cases enabled. These
  include archive integrity, output preservation, argument validation, full
  small FP32 references, and an end-to-end CLI smoke run.
- Five local harness tests passed: balanced launch order, timing arithmetic, pinned-source checks, and full FP32-reference checks for all three paths on small BF16 top-k 1/3 strided/local workloads.
- Before and after timing, all four target cases passed sampled FP32-reference checks for all three paths.
- PR #1010 and native64 complete O/LSE tensors were bitwise identical in every target case.
- Native128 and native64 are **not** bitwise identical: KV64 and KV128 group online softmax differently. Full target comparisons passed `O: atol=3e-4, rtol=3e-2` and `LSE: atol=3e-6, rtol=1e-6`. The maximum complete-tensor differences were `2.44140625e-4` for O and `3.814697265625e-6` for LSE in both runs.
- Small full-reference tests use the existing BSA suite's `O: atol=rtol=3e-2`, `LSE: atol=rtol=2e-3` bounds. An initial temporary harness incorrectly applied the long-output cross-path absolute O bound to top-k=1/3 small cases; four checks failed at near-zero outputs. The harness was corrected to use the established small-reference contract, then all five tests passed. No production kernel or production test tolerance was changed.

## Artifacts and reproduction

- [Comparison harness](benchmark_sm120_three_paths.py)
- [Run 1 raw event samples and source hashes](results/sm120_three_paths_20260915_run1.json)
- [Run 2 raw event samples and source hashes](results/sm120_three_paths_20260915_run2.json)
- [Harness tests](../../test/python/fe_api/bsa/test_sm120_three_paths_benchmark.py)

The portable harness requires an explicitly prepared, pinned legacy source
directory. It never downloads code or changes branches. Install this checkout
first and verify the imported kernel path as described in
[the review summary](SM120_REVIEW_SUMMARY.md#reproduce).

From the repository root, prepare the legacy archive in an ignored directory:

```bash
git fetch https://github.com/tiffany940107/cudnn-frontend.git bsa-sm120-blk128-bf16
mkdir -p agent/agent_space agent/agent_benchmark
VSA_LEGACY_ROOT="$(mktemp -d -p agent/agent_space pr1010.XXXXXX)"
git archive 1d5ed9d596d51087bcc2f81cc44a1f7ea260189f python/cudnn/block_sparse_attention | tar -x -C "$VSA_LEGACY_ROOT"
export SM120_PR1010_SOURCE_DIR="$(realpath "$VSA_LEGACY_ROOT/python/cudnn/block_sparse_attention")"
CUDA_VISIBLE_DEVICES=0 python benchmark/bsa/benchmark_sm120_three_paths.py \
  --legacy-source-dir "$SM120_PR1010_SOURCE_DIR" \
  --warmup 12 --repeats 102 \
  --json agent/agent_benchmark/three_paths_repeat.json
cd test/python
CUDA_VISIBLE_DEVICES=0 python -m pytest -q fe_api/bsa/test_sm120_three_paths_benchmark.py
```

Choose a new JSON filename for each repeat; the harness refuses to overwrite
an existing output file. The five archived-legacy GPU tests and the archive
integrity test skip unless `SM120_PR1010_SOURCE_DIR` is set; order, summary,
and argument-validation tests do not require the archive. Unsupported GPU/DSL
configurations are skipped.

The tables above retain the original measurement revisions and samples.
The public harness removes local checkout assumptions, adds explicit input
validation, and restores the temporary legacy Python package after each run;
it retains the same mask construction, six-permutation timing, and accuracy
checks. The legacy metadata adapter is loaded only as a benchmark baseline,
never added to the native128 production dispatch.

Only sanitized timing JSON, source hashes, and the portable harness are
published. Legacy archives, machine-specific launch settings, and profiler
reports remain ignored. Shared artifacts contain no machine IP, hostname,
GPU UUID, or private absolute path.
