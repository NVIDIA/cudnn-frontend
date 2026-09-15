# SM120 native BF16 blk128 review summary

This branch implements dedicated native KV128 forward kernels. It does not
include the logical-blk128-to-blk64 adapter proposed in PR #1010. Both the
general path and the optimized fixed-top-k path consume blk128 metadata
directly, without a blk64-kernel call or metadata expansion.

## Adopted changes

| Change | Purpose and scope |
| --- | --- |
| General native blk128 kernel | Direct sparse KV128 addressing, including variable counts, explicit valid block sizes, partial Q/KV, and MHA/GQA. |
| FA4-style fixed-top-k specialization | Register-resident Q, a dedicated TMA K/V load warp, separate K/V synchronization, early sparse-address preparation, and K/V shared-memory reuse for the output epilogue. |
| Two-fold, then four-fold sparse-loop unrolling | Reduce loop overhead in both producer and compute paths; top-k remains a runtime value. |
| Warpgroup-uniform register donation | All four warps in a donating warpgroup execute the same register-adjustment instruction. This is correctness hardening, not a separate speedup claim. |
| Underfilled final-wave balancing | Split only eligible final Q128 work into two Q64 CTAs, each still loading KV128. A single kernel launch uses the original sparse metadata; SM-count planning runs during JIT tracing, not cache hits. |
| Reusable paired benchmark and regression tests | Alternating A/B order, source hashes, raw CUDA-event samples, FP32-reference checks, and an explicit performance gate. |

"FA4-style" describes the SM120 scheduling and storage strategy, not a call
to the upstream FlashAttention-4 package. Q/K/V remain BF16 for the target
workload, with FP32 accumulators and BF16 probabilities. No FP8/INT8
quantization, approximate polynomial exponential, thresholded normalization,
or accuracy-tolerance relaxation is enabled.

For the target's 8,920 logical Q blocks on the measured 188-SM device, the
last 84 blocks become 168 smaller Q work units; the preceding 8,836 blocks
are unchanged. This does **not** split a selected 128-token KV block into
two 64-token KV blocks.

## Performance evidence

For the direct comparison requested during review, see the
[same-device native128/native64/PR #1010 report](SM120_THREE_PATHS.md).
Two independent 102-sample runs measured 1.1188x-1.1278x versus native64 and
1.1196x-1.1285x versus the closed PR's adapter. All paths use the same Q/K/V
and selected tokens; native64 preprocessing is excluded and the adapter's
per-call conversion is included. These are approximately 12%-13% throughput
gains (10.6%-11.4% lower latency), not gains against the already optimized
`9869b9b6` baseline below. The report includes both tables, raw samples,
accuracy checks, and reproducible commands.

The current kernel checkpoint is `e0602027`. The additional-speedup baseline
is `9869b9b6`, which already includes FA4-style execution and two-fold
unrolling. A 101-pair run with ten warmup pairs per case and seed `20260914`
used one RTX PRO 6000 Blackwell **Server Edition**, SM120, with PyTorch
`2.13.0+cu130`. Compilation and correctness checks are outside timing.

Q/K/V are BF16 BHSD `[1, 8, 142720, 128]`. There are 1,115 KV blocks;
requested 15% density selects 167 blocks (14.9776%), and 20% selects 223.
`strided` spreads the selected block addresses; `local` selects consecutive
blocks with cyclic wraparound. Neither pattern requires a tensor reorder.

| Density | Pattern | Baseline `9869b9b6` | Current `e0602027` | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 31.5949 ms | 31.3348 ms | 1.0083x |
| 14.9776% | local | 31.7997 ms | 31.5128 ms | 1.0091x |
| 20.0000% | strided | 43.1518 ms | 42.8118 ms | 1.0079x |
| 20.0000% | local | 44.2680 ms | 43.7693 ms | 1.0114x |

**The additional 1.05x target has not been met in any of these four cases.**
These are cumulative ratios against `9869b9b6`, not gains on top of the
already improved checkout. A separate 101-pair comparison against `69360b3e`
isolates final-wave scheduling at 1.0033x-1.0050x. Do not add or multiply
independently measured checkpoint gains to construct an unmeasured result.

Earlier Workstation Edition measurements in the [benchmark README](README.md)
showed a 3.77% latency reduction from the original native kernel to the initial
FA4-style specialization at 20% density. Their absolute latencies and dense
baseline must not be combined with the Server Edition measurements above.

## Validation and limits

- The final publication refresh, including the portable three-path harness,
  archive-integrity checks, and DSL-version guard, passed **66 tests with
  19 skips** across the entire BSA directory, with the pinned legacy archive
  enabled.
- The September 15 MR refresh reran the entire `test/python/fe_api/bsa`
  directory: **53 passed, 19 skipped**, with no failures. The narrower forward
  and paired-harness checkpoint previously passed 33 tests with nine skips.
  The CuTe BSA path was exercised on SM120; unsupported configurations were
  skipped. The local frontend backend-version probe was unavailable, so this
  is not a claim that backend Graph API or other GPU-architecture CI passed.
- `pre-commit run --from-ref upstream/develop --to-ref HEAD` passed the Black,
  Black Jupyter, and SPDX hooks; no changed C++ files required clang-format.
- FP16/BF16 tests cover top-k 1-5, partial Q, GQA, noncontiguous and
  query-dependent KV selections, all-tail/full/mixed/unsplit scheduling, and
  no repeated SM-count query on compile-cache hits.
- Separate full-size checks at both densities and both patterns found the
  entire O and LSE tensors bitwise identical to the baseline. The public
  paired benchmark checks representative FP32-reference rows; it is not a
  full-tensor bitwise test.
- A stricter exploratory FP32-reference test has a documented pre-existing
  two-element BF16 bound exceedance reproduced by the unchanged kernel;
  candidate and control are bitwise identical. This is not reported as a
  passing strict-reference test. See the
  [accuracy note](SM120_OPTIMIZATION.md#final-wave-scheduling-checkpoint).
- The optimized specialization requires fixed top-k, `q2k_block_nums=None`,
  `block_sizes=None`, full physical KV blocks, and QK/V dimensions 128.
  It requires CuTe DSL 4.7.0 or newer and rejects an older public DSL before
  importing the specialized kernel. The import-guard regression was observed
  RED before the fix and GREEN afterward; GPU kernel arithmetic is unchanged.
  Other supported blk128 inputs use the general native kernel. Existing
  SM120 default blk64 behavior and the public function signature are unchanged.
- These are single-GPU forward results. The earlier multi-GPU dual-stream
  and Q-block communication/compute-overlap prototypes remain in the separate
  Block-Sparse-Attention repository and are not part of this cuDNN change.

## Reproduce

Install this checkout and its test dependencies as described in the repository
root guide. Confirm that the kernel module resolves to the checkout being
reviewed, particularly when using an editable install or multiple worktrees:

```bash
python -c 'import cudnn.block_sparse_attention.csrc.fwd.sm120_blk128.bsa_fwd_sm120_fa4 as kernel; print(kernel.__file__)'
cd test/python
CUDA_VISIBLE_DEVICES=0 python -m pytest -q fe_api/bsa
cd ../..
```

Save the kernel file from commit `9869b9b6` to an ignored local workspace,
then run on an otherwise idle SM120 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/bsa/benchmark_sm120_blk128_pair.py \
  --baseline-source /path/to/agent/agent_space/baseline_9869b9b6.py \
  --sequence 142720 --heads 8 --densities 0.15 0.20 \
  --patterns strided local --warmup 10 --repeats 101 --seed 20260914 \
  --min-speedup 1.05 --fail-below-target \
  --json /path/to/agent/agent_benchmark/paired.json
```

The gate is expected to fail for the recorded checkpoint; a nonzero exit is
not a correctness failure when only the speedup threshold is missed. Repeat
in separate processes before accepting a small gain. For a fresh dense
comparison on the same device, run
`python benchmark/bsa/benchmark_sm120_blk128.py --fail-below-target`.

## Investigated but not enabled

Extra V buffers, dedicated softmax warps, lookahead QK/softmax, register-budget
changes, all-Q64 scheduling, L2 prefetch, and source-level SFU/MMA interleaving
did not show a stable additional benefit. The measured pre-scheduling
checkpoint already reached about 93.7%-93.8% Tensor FP pipeline utilization
at 20% density. Generated code already overlaps some exponential and PV MMA
instructions; source-level fusion alone is not evidence of new overlap.

The [screening log](SM120_OPTIMIZATION.md) records the negative results and
profiling context. A normalization-removal diagnostic is explicitly **not
attention** and is not counted toward the performance target. Experimental
sources, raw profiler captures, and machine-specific launch files remain
ignored; shared documentation contains no machine IPs, hostnames, or GPU UUIDs.
