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
| Four-fold main-loop unroll | 1.0045x | Insufficient; needs repeatability check |
| Eight-fold main-loop unroll | 0.9981x | Reject |
| Stagger the initial QK of the two compute warp groups | 0.9998x | No demonstrated benefit |
| Fifth-degree FP32 exp2 polynomial on 25% / 50% / 100% of pairs | 0.9795x / 0.9588x / 0.9181x | Reject |
| One K buffer and two V buffers | 0.9943x | Reject |
| Combined warp vote for exact maximum updates | 0.9985x | Reject |
| Defer maximum updates by an eight-log2-unit threshold | 1.0042x | Insufficient; changes rounding |
| Internal Q64 CTA with complete K128 tiles, unroll 2 / 4 | 0.8987x / 0.9071x | Reject |
| Prefetch four K / V / both matrix fragments | 1.0007x / 1.0005x / 1.0004x | No demonstrated benefit |
| Prefetch eight K and V matrix fragments | 1.0006x | No demonstrated benefit |

All successfully compiled candidates above passed BF16 FP32-reference tests
with top-k counts 1, 2, 3, and 223, partial Q tiles, and GQA. The first
denominator experiment also passed the existing six FP16/BF16 fixed-top-k
tests. Some experiments change FP32 reduction order; passing a tolerance
check does not imply bitwise equivalence. No polynomial exp2 or thresholded
normalization is enabled in the production kernel.

An intrusive clock-instrumented baseline gives approximately 42% QK, 44% PV,
12% softmax, and 3% explicit K/V waits (rounded). Instrumentation changes
register lifetimes and instruction scheduling: these figures guide further
investigation, not a precise bottleneck attribution or performance claim.

**The additional 5% target has not been achieved in this screening.** None of
these experiments replaces the committed kernel. Experimental sources,
tests, and profiles remain in the local ignored agent workspace; no
infrastructure addresses, GPU identifiers, or host names are recorded here.
