# SM120 blk128 optimization screening

## Baseline and acceptance criteria

The September 14, 2026 screening uses commit `9869b9b6` as the kernel
baseline. The target is at least **1.05x** baseline throughput at both 15%
and 20% density, for BF16 BHSD `[1, 8, 142720, 128]`, with native blk128
metadata and complete physical KV blocks. Both strided and local address
patterns must be checked before adopting a candidate.

These measurements use one RTX PRO 6000 Blackwell **Server Edition**.
Do not compare their absolute times with the earlier Workstation Edition
measurements in the README. Baseline and candidate run in the same process,
on the same inputs and device, with alternating A/B and B/A launch order.
JIT compilation and correctness checks are outside CUDA-event timing.
Screening uses 10 warmup pairs and 31 timed pairs; it is not final acceptance.

## First screening: 20% strided

| Candidate | Baseline / candidate time | Decision |
| --- | ---: | --- |
| Delay denominator thread-quad sum until epilogue | 0.9904x | Reject |
| FTZ maximum reduction | 1.0001x | No demonstrated benefit |
| Four independent softmax partial sums | 1.0000x | No demonstrated benefit |
| Four-fold main-loop unroll | 1.0045x | Small validated gain; see checkpoint below |
| Eight-fold main-loop unroll | 0.9981x | Reject |
| Stagger the initial QK of the two compute warp groups | 0.9998x | No demonstrated benefit |
| Fifth-degree FP32 exp2 polynomial on 25% / 50% / 100% of pairs | 0.9795x / 0.9588x / 0.9181x | Reject |
| One K buffer and two V buffers | 0.9943x | Reject |
| Combined warp vote for exact maximum updates | 0.9985x | Reject |
| Defer maximum updates by an eight-log2-unit threshold | 1.0042x | Insufficient; changes rounding |
| Internal Q64 CTA with complete K128 tiles, unroll 2 / 4 | 0.8987x / 0.9071x | Reject |
| Prefetch four K / V / both matrix fragments | 1.0007x / 1.0005x / 1.0004x | No demonstrated benefit |
| Prefetch eight K and V matrix fragments | 1.0006x | No demonstrated benefit |
| Four compute warps, 32 Q rows per warp, streamed Q fragments | 0.8659x | Reject |
| Stagger compute warp groups on every KV iteration | 0.9790x | Reject |
| Compute register budget 224 / 232 instead of 240 | 0.9789x / 0.9901x | Reject |
| Native NVVM MMA intrinsic instead of inline PTX MMA | 1.0004x | No demonstrated benefit |
| Normalize two selected blk128 blocks together | 0.7550x | Reject |
| Two-block normalization, half Q staged in SMEM | 0.8570x | Reject |
| Two-block normalization, O staged in SMEM | 0.7001x | Reject |
| QK D-outer loop / groups of two / groups of four column pairs | 1.0010x / 1.0010x / 1.0005x | No demonstrated benefit |
| Head-interleaved CTA work | 0.9345x | Reject |
| 64-byte / 32-byte K/V TMA swizzle | 0.9926x / 0.9810x | Reject |
| Initial / per-iteration warp staggering plus delayed V wait | 0.9998x / 0.8986x | Reject |
| Two V buffers plus per-iteration staggering and delayed V wait | 0.8653x | Reject |
| Four / two dedicated softmax warps, S/P exchanged through SMEM | 0.8396x / 0.8580x | Reject |
| Two softmax warps with SMEM row state, compute budget 224 / 216 | 0.8558x / 0.8743x | Reject |
| Two S/P exchange buffers and quad-shared row state | 0.8629x | Reject |

Except for the initial denominator-only experiment, the candidates above
passed BF16 FP32-reference tests with top-k counts 1, 2, 3, and 223, partial Q
tiles, and GQA. The first
denominator experiment passed the existing six FP16/BF16 fixed-top-k
tests and a full-size baseline comparison. Some experiments change FP32
reduction order; passing a tolerance
check does not imply bitwise equivalence. No polynomial exp2 or thresholded
normalization is enabled in the production kernel.

An intrusive clock-instrumented baseline gives approximately 42% QK, 44% PV,
12% softmax, and 3% explicit K/V waits (rounded). Instrumentation changes
register lifetimes and instruction scheduling: these figures guide further
investigation, not a precise bottleneck attribution or performance claim.

The full-shape baseline and grouped-fragment-prefetch cubins both report zero
stack/local memory. Their static register reservation is 168 per thread;
compute warps dynamically request 240 via register donation. The static
reservation is not the compute warp's register limit.

The joint-two-block candidate reports a 208-byte stack with local load/store
instructions. Staging O in shared memory reduced that stack to 96 bytes but
did not recover performance. Another internal-Q64/full-K128 prototype reused
one shared buffer for K and V to allow two resident CTAs. Although its small
reference tests passed, one full-size comparison exceeded the output tolerance
(two elements); a repeat did not reproduce that failure. It is rejected
without an accuracy or performance claim.

The dedicated-softmax prototypes keep FP32 S and row state, BF16 P, and
native K128 loads. Their 20% strided full-shape O/LSE comparisons were
bit-exact. The four-warp and initial two-warp versions report 64-byte and
40-byte stacks. Moving row state into SMEM and using a 216-register compute
budget eliminated stack/spill code, but remained slower. Removing a spill
does not by itself establish a net latency improvement.
Doubling the exchange buffers and sharing row state within each thread quad
also remained spill-free and bit-exact in the 20% strided full-size check,
but did not improve performance.

**The additional 5% target has not been achieved in this screening.** Only the
four-fold-unroll checkpoint below is adopted. Other experimental sources,
tests, and profiles remain in the local ignored agent workspace; no
infrastructure addresses, GPU identifiers, or host names are recorded here.

## Four-fold-unroll checkpoint

Both producer and compute loops now use four-fold instead of two-fold
dynamic unrolling. The sparse traversal, MMA arithmetic, BF16 storage, and
physical K/V block size are unchanged. The top-k count stays a runtime value;
there is no external blk64 expansion or quantization.

The reusable paired harness measured the following with 101 timed pairs and
10 warmup pairs per case, using the same device and seed as the control.

| Density | Pattern | Baseline `9869b9b6` | Four-fold unroll | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 31.4372 ms | 31.3286 ms | 1.0035x |
| 14.9776% | local | 31.6646 ms | 31.5479 ms | 1.0037x |
| 20.0000% | strided | 43.0268 ms | 42.8518 ms | 1.0041x |
| 20.0000% | local | 45.2612 ms | 45.1096 ms | 1.0034x |

Two other independent 101-pair runs were positive in all four cases, spanning
1.0018x–1.0043x. One of those runs checked the **entire O and LSE tensors**
bit-for-bit against the saved baseline for all four cases and passed. This is
distinct from the sampled FP32-reference checks in the public paired harness.
The FP16/BF16 unit tests now exercise top-k counts 1–5, covering all four-way
loop tails and one complete unrolled group, with partial Q, GQA, and
noncontiguous selected KV blocks.
The related BSA forward and paired-harness regression run passed 24 tests;
nine tests for unsupported configurations were skipped on this device.

These measurements justify a small checkpoint, not a 5% success claim. All
four cases still fail the 1.05x acceptance gate.

## Paired-harness control

The reusable `benchmark_sm120_blk128_pair.py` harness was checked against the
unchanged `9869b9b6` kernel source, using 101 timed pairs per case. The expected
result is no speedup. All four cases correctly fail the 1.05x performance gate;
FP32-reference sample checks pass before and after timing.

| Density | Pattern | Saved baseline | Unchanged checkout | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 31.4823 ms | 31.4761 ms | 1.0002x |
| 14.9776% | local | 31.7752 ms | 31.7486 ms | 1.0008x |
| 20.0000% | strided | 43.1200 ms | 43.1068 ms | 1.0003x |
| 20.0000% | local | 47.0327 ms | 47.0687 ms | 0.9992x |

Absolute times drift during sustained runs; the ratios compare adjacent
launches on the same GPU. Raw samples and source hashes are saved locally by
the harness. These are control measurements, not an optimized-kernel result.

## Hardware-counter follow-up

Nsight Compute 2026.3 profiled the `1759da44` checkpoint on the same Server
Edition GPU at 20% density. The kernel was compiled and warmed up before
collection. Strided used 22 replay passes; local used 13. Neither GPU clocks
nor caches were forcibly controlled, and the profiler warns that measurements
can vary. These are diagnostic profiles, **not** acceptance timings.

| Counter | Strided | Local |
| --- | ---: | ---: |
| Tensor FP pipeline utilization, elapsed-cycle basis | 93.70% | 93.83% |
| L1/TEX throughput | 48.53% | 48.59% |
| L2 throughput | 38.39% | 38.41% |
| L2 hit rate | 97.05% | 99.25% |
| DRAM throughput | 6.75% | 1.91% |
| Local-memory spill requests | 0 | 0 |

The profiler identifies Tensor FP as the limiting pipeline. In the strided
profile, the leading warp stalls are fixed-latency execution dependencies
(36.6% of cycles between issues) and math-pipeline contention (35.6%).
Low issue-slot utilization alone therefore does not imply that the tensor
pipeline is idle. See NVIDIA's
[metric definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder).

At unchanged clocks and tensor instruction count, raising throughput by 1.05x
from 93.7% utilization would require approximately 98.4% utilization. This is
a directional estimate from the measured profile, not a proof that another
5% is achievable or impossible. It explains why extra memory buffering and
source-level MMA reordering have not delivered large gains.

The spill-free two-softmax-warp prototype was also profiled with 22 replay
passes. Tensor FP utilization fell to 80.32%, while L1/TEX throughput rose to
77.39%; local-memory spill requests remained zero. CTA-barrier waits became
the leading stall reason (41.8% of cycles between issues). These observations
are consistent with the extra S/P exchange and synchronization offsetting the
benefit of moving softmax out of the MMA warps.

To inspect the current checkout with the existing benchmark, use a private,
ignored output directory and a supported local Nsight Compute installation:

```bash
VSA_PROFILE_DIR=/path/to/ignored/agent/agent_profiles
mkdir -p "$VSA_PROFILE_DIR"
ncu --clock-control none --cache-control none \
    --kernel-name 'regex:.*BlockSparseAttnForwardSm120Blk128Fa4.*' \
    --launch-skip 10 --launch-count 1 \
    --section SpeedOfLight --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --export "$VSA_PROFILE_DIR/blk128_checkpoint" \
    python benchmark/bsa/benchmark_sm120_blk128_pair.py \
    --baseline-source python/cudnn/block_sparse_attention/csrc/fwd/sm120_blk128/bsa_fwd_sm120_fa4.py \
    --densities 0.20 --patterns strided --warmup 10 --repeats 3
```

Here both variants use the same checkout; the kernel-name filter and launch
skip select a warmed native blk128 invocation. Ignore timing ratios printed
while the profiler is attached. Run the paired acceptance benchmark separately
without profiling. Raw profiler reports can contain machine details and must
not be committed or shared without sanitization.

## Normalization-removal diagnostic

This experiment is **not an attention implementation and is never an
acceptance candidate**. It replaces probabilities with BF16-rounded raw QK
scores, retains the complete QK/PV tensor work and sparse traversal, and sets
the final normalization to one. Separate tests verify this linear-algebra
result for top-k 1, 2, 3, and 223 and explicitly confirm that it differs from
attention. No production kernel or attention accuracy check is weakened.

With 101 paired samples per case, against `9869b9b6`:

| Density | Pattern | Attention baseline | Linear diagnostic | Baseline / diagnostic |
| ---: | --- | ---: | ---: | ---: |
| 14.9776% | strided | 31.3471 ms | 29.8411 ms | 1.0505x |
| 14.9776% | local | 31.5826 ms | 30.0435 ms | 1.0512x |
| 20.0000% | strided | 42.9921 ms | 40.9665 ms | 1.0494x |
| 20.0000% | local | 46.6794 ms | 43.8664 ms | 1.0641x |

A separate 20% strided Nsight Compute capture counted exactly
4,073,799,680 warp-level tensor instructions, matching
`1115 Q blocks * 8 heads * 8 compute warps * 223 KV blocks * 256 MMAs`.
The QK computation was therefore not dead-code-eliminated. Tensor FP
utilization was 97.26% in that capture.

This is a directional estimate of normalization overhead for this pipeline,
not a universal performance bound. Removing normalization also changes
register allocation, scheduling, operand values, and potentially clocks.
In particular, the diagnostic's approximately 1.05x ratio must **not** be
reported as achieving the requested attention speedup. It motivates focusing
on overlapping normalization with tensor work rather than expecting small
softmax instruction rewrites alone to deliver another 5%.
