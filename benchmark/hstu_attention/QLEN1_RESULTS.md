# HSTU single-query optimization

## Target workload

- BF16 HSTU attention, with causal and bounded local masks
- `seqlen_q = 1` for every sequence
- Head dimensions 64, 128, and 256 with matching Q/K/V head counts
- Variable `seqlen_kv`; the broad sweep covers average lengths 128 to 4096
- Batch sizes 16 to 2048 and 1, 2, 4, or 8 heads

`benchmark_hstu_qlen1.py` uses 10 warmup iterations and reports the median of
seven groups of 30 executions. The baseline is commit `8baf903`, which adds
only the benchmark on top of the original HSTU kernel.

## Broad-sweep final policy

The final selection is deliberately global rather than a lookup table of
individual shapes. For every GPU and direction, 36 batch/head/KV cases are
run for each combination of D64/D128/D256 and causal/local masking: 216 cases
per direction and GPU. The sweep covers BS=16..2048, H=1/2/4/8, average
KV=128..4096, and opposing low-grid/long-KV and high-grid/short-KV corners.
Additional local sweeps cover windows 63, 255, and 2047. Candidates are
rotated within the same allocation and every case receives equal weight.

Forward has no BS, head-count, KV-length, packed-token, window-size, or
causal/local tuning branch. Backward makes one cheap host-side choice between
the unsplit kernel and the architecture default; it reads tensor metadata only
and does not benchmark, synchronize the device, or inspect tensor values:

| GPU | Direction | Global BF16 qlen=1 schedule |
| --- | --- | --- |
| B300 (SM10.3) | Forward | Tensor Core M64/N128, 16dp SiLU, tail-only masking, unsplit; 5 KV stages for D64/D128 and 3 for D256 |
| B300 (SM10.3) | Backward | CUDA-core direct-pair, 4/2/1 KV rows per warp for D64/D128/D256; adaptive unsplit or split8 |
| Rubin (SM10.7) | Forward | Tensor Core M64/N128, 16dp SiLU, tail-only masking, unsplit; 5 KV stages for D64/D128 and 3 for D256 |
| Rubin (SM10.7) | Backward | CUDA-core direct-pair, 4/2/1 KV rows per warp for D64/D128/D256; adaptive unsplit or split13 |

D256 uses three stages only because five D256 stages exceed the shared-memory
capacity; it is the same forward kernel family, not a workload heuristic.

The backward selector defines normalized work as
`average_KV * D / 128` and base parallelism as `BS * H`. It uses the unsplit
kernel when normalized work is below 256; D256 does so only through 256 base
CTAs because it packs fewer rows per warp. It also uses unsplit when at least
4096 base CTAs already exist and normalized work is below 768. All remaining
cases use split8 on B300 or split13 on Rubin. Causal and local masks share this
rule: local backward avoids arithmetic outside the window but must still write
zero dK/dV rows across the complete packed KV extent.

A fresh two-candidate validation reran all 216 backward cases per GPU after
adding the selector (two warmups, three interleaved groups of six executions):

| GPU | Unsplit selections | Adaptive / per-case oracle, geometric mean | p95 | Worst | Adaptive speedup over always split |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 (SM10.3) | 50 / 216 | 1.006x | 1.069x | 1.156x | 1.043x |
| Rubin (SM10.7) | 50 / 216 | 1.003x | 1.013x | 1.103x | 1.035x |

Here the oracle chooses only between the two retained schedules, unsplit and
the architecture default, for each measured case. The selector therefore
recovers nearly all of the useful crossover without a per-shape lookup table.

The specialized policy applies to BF16 qlen=1 with causal or bounded-local
masks, D=64/128/256, matching Q/K/V heads, non-paged KV, no arbitrary mask,
and layouts accepted by the direct path. FP16 and other mask/layout modes keep
the existing general path.

For scale, the earlier D128 causal standalone comparison at H=4 and average
KV about 2048 was:

| GPU | Batch | Forward original -> fixed (ms) | Speedup | Backward original -> fixed (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.0808 -> 0.0513 | 1.58x | 0.2348 -> 0.1136 | 2.07x |
| B300 | 512 | 0.4174 -> 0.3132 | 1.33x | 1.7751 -> 0.7698 | 2.31x |
| B300 | 1024 | 0.8335 -> 0.6137 | 1.36x | 3.5462 -> 1.5301 | 2.32x |
| Rubin | 64 | 0.0443 -> 0.0378 | 1.17x | 0.1476 -> 0.0696 | 2.12x |
| Rubin | 512 | 0.2068 -> 0.1726 | 1.20x | 1.0972 -> 0.4431 | 2.48x |
| Rubin | 1024 | 0.4106 -> 0.3320 | 1.24x | 2.2278 -> 0.8909 | 2.50x |

These D128 numbers are not presented as an all-dimension average; local-mask
speedups below are likewise comparisons with an older path that performed
substantial work outside the requested window.

## Local-window qlen=1 extension

This section records the earlier D128-only, mask-specific tuning that built
the local path. The all-dimension policy above supersedes its B300 3-stage and
both GPUs' local split4 choices in favor of one kernel family and the same
lightweight backward selector for causal and local attention.

Packed HSTU aligns its single query to KV row `seqlen_k - 1`, so a bounded
local mask selects one contiguous suffix of KV. Forward reuses the M64/N128
tensor-core kernel and masks the partial block at the start of that suffix in
the score-to-P conversion. Backward vector-zeros dK/dV rows before the suffix,
performs QK and dO.V math only on attended rows, and interleaves those rows
across split CTAs so a short suffix does not leave all but the last CTA idle.
Every dK/dV row is still overwritten exactly once; only dQ uses the existing
packed BF16 atomic reduction.

The tuning sweep covers 18 shapes per GPU: batch sizes 64/512/1024 crossed
with average KV lengths 128/256/512/1024/2048/4096, always H=4 and D=128.
Each shape is measured at left windows 63, 255, and 2047, for 54 points per
candidate. The raw interleaved measurements are in
[`results-local-window/`](results-local-window/). These measurements predate
the lightweight unsplit/split selector described above; their candidate
timings remain the evidence used to check its local-window behavior.

For forward, the table reports the speedup of five KV stages over three in
that narrower D128 local sweep. B300 favored three stages there, while Rubin
favored five. Once causal cases and D64 are weighted together, five stages is
the better architecture-level D64/D128 choice on both GPUs.

| GPU | Fixed local schedule | Window 63 | Window 255 | Window 2047 | Worst five-stage / three-stage case |
| --- | --- | ---: | ---: | ---: | ---: |
| B300 (SM10.3) | M64/N128, 3-stage, unsplit | 1.003x | 0.996x | 1.006x | 0.953x |
| Rubin (SM10.7) | M64/N128, 5-stage, unsplit | 0.994x | 1.010x | 1.062x | 0.983x |

For backward, split4 is the best setting across these 54 local D128 points on
both targets. The B300 sweep labels this identical 128-thread configuration
`direct-pair-t128-split4`; normal automatic dispatch obtains 128 threads from
the split rule and does not retain the experimental override. The final global
policy instead chooses between unsplit and B300 split8 or Rubin split13 across
both masks and all three dimensions.

| GPU | Fixed local schedule | Window 63 | Window 255 | Window 2047 | All 54 | Wins vs unsplit | Worst vs unsplit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B300 (SM10.3) | 2 rows/warp, split4 | 1.020x | 1.029x | 1.104x | 1.050x | 43 / 54 | 0.870x |
| Rubin (SM10.7) | 2 rows/warp, split4 | 1.107x | 1.163x | 1.249x | 1.172x | 44 / 54 | 0.898x |

Those ratios compare two already-specialized direct kernels. Against the
original general local-window path at H=4 and average KV about 2K, the
then-selected D128 forward measurements are:

| GPU | Window | BS=64 original -> candidate | BS=512 original -> candidate | BS=1024 original -> candidate |
| --- | ---: | ---: | ---: | ---: |
| B300 | 63 | 0.04379 -> 0.01877 ms (2.33x) | 0.26268 -> 0.03446 ms (7.62x) | 0.51281 -> 0.06341 ms (8.09x) |
| B300 | 255 | 0.05439 -> 0.01902 ms (2.86x) | 0.33525 -> 0.04739 ms (7.07x) | 0.65737 -> 0.08642 ms (7.61x) |
| B300 | 2047 | 0.20727 -> 0.04411 ms (4.70x) | 1.23427 -> 0.27207 ms (4.54x) | 2.41245 -> 0.53369 ms (4.52x) |
| Rubin | 63 | 0.03006 -> 0.01541 ms (1.95x) | 0.12620 -> 0.03289 ms (3.84x) | 0.23257 -> 0.05363 ms (4.34x) |
| Rubin | 255 | 0.03338 -> 0.01522 ms (2.19x) | 0.14768 -> 0.04026 ms (3.67x) | 0.27299 -> 0.06762 ms (4.04x) |
| Rubin | 2047 | 0.08841 -> 0.03312 ms (2.67x) | 0.42414 -> 0.17583 ms (2.41x) | 0.81377 -> 0.34024 ms (2.39x) |

The corresponding backward measurements are below. The original B300 kernel
illegally accesses memory at BS=1024 and average KV about 2K, while the
then-selected D128 kernel completes at 0.6993/0.7678/1.3780 ms for windows
63/255/2047.

| GPU | Window | BS=64 original -> candidate | BS=512 original -> candidate |
| --- | ---: | ---: | ---: |
| B300 | 63 | 0.28815 -> 0.05970 ms (4.83x) | 2.19736 -> 0.35609 ms (6.17x) |
| B300 | 255 | 0.28781 -> 0.06361 ms (4.52x) | 2.19834 -> 0.38885 ms (5.65x) |
| B300 | 2047 | 0.28797 -> 0.10287 ms (2.80x) | 2.20050 -> 0.70203 ms (3.13x) |
| Rubin | 63 | 0.26541 -> 0.03905 ms (6.80x) | 2.00885 -> 0.24095 ms (8.34x) |
| Rubin | 255 | 0.26598 -> 0.04244 ms (6.27x) | 2.00794 -> 0.26689 ms (7.52x) |
| Rubin | 2047 | 0.20213 -> 0.07813 ms (2.59x) | 1.48741 -> 0.48011 ms (3.10x) |

### Local-window counter evidence

Nsight Compute 2026.2.1 profiled the original and then-selected D128 B300 paths
on the same BS=512, H=4, average-KV=2046.5, window-255 input. Metric-replay
durations are not CUDA-event timings, so only the paired comparison is used.

| Forward metric | Original | Measured M64 local | Change |
| --- | ---: | ---: | ---: |
| Kernel duration | 344.192 us | 52.224 us | 6.59x faster |
| DRAM read | 537.54 MB | 269.08 MB | 2.00x less |
| Executed instructions | 134.74 M | 9.35 M | 14.41x less |
| DRAM throughput | 20.50% | 68.01% | +47.51 pp |
| Shared memory / CTA | 231,424 B | 133,120 B | 42.5% less |
| Tensor-pipe active | 17.13% | 28.33% | +11.20 pp |

The launch grid remains 2,048 CTAs in both cases. The final kernel reads almost
exactly one 256-token K/V window per batch/head and eliminates work on masked
score columns rather than merely making the original full-tile work faster.

| Backward metric | Original three-launch path | Measured two-launch path | Change |
| --- | ---: | ---: | ---: |
| Total duration | 2,198.688 us | 388.960 us | 5.65x faster |
| DRAM read | 412.88 MB | 274.83 MB | 1.50x less |
| DRAM write | 2,304.32 MB | 2,093.85 MB | 1.10x less |
| Executed instructions | 240.53 M | 88.75 M | 2.71x less |
| Main-kernel grid | 49,152 CTAs | 8,192 CTAs | 6.00x fewer |
| Main-kernel shared memory / CTA | 231,424 B | 3,072 B | 75.3x less |
| Main-kernel DRAM throughput | 15.83% | 80.19% | +64.36 pp |

The original backward clears a 75 MB float workspace, runs the tensor-core
kernel, then converts dQ. The measured D128 path only clears the small BF16 dQ
output before its split4 kernel. dK/dV still account for about 2.1 GB of writes
because the API requires explicit zeros outside the attended suffix.

### Dynamic packed extents

The qlen=1 forward and direct-backward compile descriptors use symbolic
packed-Q, packed-KV, and batch extents. Head count and dimension remain
plan-time metadata, while maximum sequence lengths are runtime `Int32`
arguments. This prevents a compiled TMA descriptor or launch schedule from
being reused with a static extent from the first batch, without turning
continuous batching into one compilation per token total. Same-process
BS=2/KV=128 -> BS=3/KV=192 regression tests cover D64/D128/D256, the automatic
paths, and the generic Tensor Core fallback; all executions reuse the intended
artifact and match the CPU oracle.

## Initial forward and direct-path results

These controlled baseline/candidate measurements predate the small-MMA
backward path. The final backward dispatch is reported below.

### B300 (SM10.3)

| Batch | Forward baseline (ms) | Forward optimized (ms) | Speedup | Backward baseline (ms) | Backward optimized (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0798 | 0.0569 | 1.40x | 0.2338 | 0.2108 | 1.11x |
| 512 | 0.4235 | 0.3220 | 1.32x | 1.7760 | 1.0318 | 1.72x |
| 1024 | 1.0102 | 0.6482 | 1.56x | 3.5453 | 1.8740 | 1.89x |

### Rubin (SM10.7)

| Batch | Forward baseline (ms) | Forward optimized (ms) | Speedup | Backward baseline (ms) | Backward optimized (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0442 | 0.0441 | 1.00x | 0.1445 | 0.1293 | 1.12x |
| 512 | 0.2285 | 0.2249 | 1.02x | 1.0261 | 0.8069 | 1.27x |
| 1024 | 0.4557 | 0.4322 | 1.05x | 2.0375 | 1.4264 | 1.43x |

## Historical per-shape split-KV forward experiment

The qlen=1 HSTU output is additive across KV partitions: each partition can
compute `silu(alpha * QK^T) @ V / scaling_seqlen` independently, without the
softmax renormalization needed by standard attention. The split path exposes
2 or 4 virtual heads per real batch/head tile, gives each CTA a contiguous KV
block range, and atomically combines packed BF16 output pairs. Output zeroing
is included in every timing below.

The CUDA-core forward experiment was correct but lost decisively to the
tensor-core kernel. Its B300 times were 0.2728, 0.5855, and 1.0120 ms for
BS=64/512/1024; Rubin took 0.2525, 0.5355, and 0.9471 ms. The losing kernel is
not retained.

The final tensor-core comparison uses 10 warmups and the median of seven
groups of 30 executions:

| GPU | Batch | Unsplit TC (ms) | Selected split (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.0556 | 0.0488 (`split4`) | 1.14x |
| B300 | 512 | 0.3197 | 0.3154 (`split2`) | 1.01x |
| B300 | 1024 | 0.6925 | 0.6925 (unsplit) | 1.00x |
| Rubin | 64 | 0.0440 | 0.0387 (`split4`) | 1.14x |
| Rubin | 512 | 0.2238 | 0.2170 (`split2`) | 1.03x |
| Rubin | 1024 | 0.4317 | 0.4224 (`split2`) | 1.02x |

These early per-shape boundaries were useful for proving the split-KV idea,
but are superseded by the fixed broad-sweep policy above. The later M64
five-stage kernel is fast enough that production now uses unsplit TC on both
architectures; fixed split2 regresses short-KV and already-large-grid cases.

## Forward tile-shape experiment

The smallest useful one-CTA MMA M dimension on these devices is 64, so a
separate experiment replaced the M128/N128 QK and PV tiles with M64/N128. It
also tested an N64 KV tile and a qlen=1 epilogue in which only the warp owning
output row zero participates. M64 needs the native 32-data-path TMEM mapping;
reusing the M128 software fragment interpretation is numerically incorrect.

All M64 variants pass the boundary oracle on B300 and Rubin. Unsplit M64 is
bitwise identical to M128 for KV lengths 1, 2, 63, 64, 65, 127, 128, 129, 255,
256, 257, 2048, 2049, and 3072. The stable interleaved Rubin comparison was:

| Batch | Selected M128 (ms) | Best M64/N128 (ms) | M64 delta | Best M64/N64 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.0385 (`split4`) | 0.0411 (`split2`) | +6.7% | 0.0434 (`split4`) |
| 512 | 0.2169 (`split2`) | 0.2221 (`split2`) | +2.4% | 0.2548 (`split4`) |
| 1024 | 0.4222 (`split2`) | 0.4313 (`split2`) | +2.2% | 0.4978 (`split4`) |

On B300, M128+split4 remains about 4% faster than the best M64 schedule at
BS=64, and BS=512 slightly favors M128. At BS=1024, forward/reverse
interleaved runs changed which M tile won as shared-node frequency moved, so
there is no reproducible M64 speedup. N64 is consistently slower because it
doubles the roughly 2K-long KV loop and its TMA/barrier overhead even though
every KV token is useful. Restricting the epilogue to one warp is correct but
does not materially change latency.

### M64 critical-path counter check

A follow-up B300 NCU run forced unsplit M128/N128 and M64/N128 on identical
inputs. This isolates the M dimension while keeping the KV tile, launch grid,
and input bytes fixed. NCU metric-replay durations are higher than CUDA-event
timings, so only paired relative values are used here.

| Batch | M tile | Duration (us) | DRAM read | DRAM peak | BF16 tensor ops | UTCMMA instructions | Tensor-pipe active |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | 77.82 | 267.04 MB | 45.59% | 34.16 G | 65.2 K | 35.44% |
| 64 | 64 | 81.41 | 267.05 MB | 43.38% | 17.08 G | 65.2 K | 33.14% |
| 512 | 128 | 364.90 | 2.15 GB | 76.85% | 274.68 G | 523.9 K | 58.46% |
| 512 | 64 | 374.24 | 2.15 GB | 74.91% | 137.34 G | 523.9 K | 56.30% |
| 1024 | 128 | 696.67 | 4.30 GB | 80.55% | 550.23 G | 1049.5 K | 60.30% |
| 1024 | 64 | 712.51 | 4.30 GB | 78.76% | 275.11 G | 1049.5 K | 58.75% |

M64 exactly halves the tensor math operations, so the smaller instruction is
doing what was intended. It does not, however, reduce the number of issued
UTCMMA instructions. Tensor-pipe activity falls by only 1.6--2.3 percentage
points; at BS=1024 its activity multiplied by duration is essentially
unchanged (about 420 us versus 419 us). K/V traffic is unchanged and DRAM is
the more heavily utilized pipeline. Kernel duration increases by 4.6%, 2.6%, and 2.3% at
BS=64/512/1024. At BS=1024 the M64 fragment/TMEM path also raises total
executed instructions from 118.66 M to 178.72 M.

This M64 experiment alone cannot establish that doing less MMA work is
intrinsically slower, and aggregate utilization counters cannot prove temporal
TMA/MMA overlap. M64 changes the instruction shape and software mapping at the
same time: it halves nominal operations, leaves the UTCMMA instruction count
unchanged, and adds scalar instructions. It therefore establishes only that
this particular M64 implementation loses.

### Controlled MMA-work and KV-pipeline check

A stricter B300 experiment kept the M128/N128 tile, TMA loads, KV bytes, grid,
and output unchanged, but reissued each QK MMA 2, 4, or 8 times with zero-init
to the same TMEM destination. Every variant produced exactly the same output.
Because PV is unchanged, these variants execute 1.5x, 2.5x, and 4.5x the
baseline tensor work. CUDA-event medians below use 20 warmups and 11 groups of
80 executions.

| Batch | Baseline (ms) | QK 2x (ms) | QK 4x (ms) | QK 8x (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.05494 | 0.05842 (+6.3%) | 0.07418 (+35.0%) | 0.12501 (+127.5%) |
| 512 | 0.34364 | 0.39536 (+15.1%) | 0.56371 (+64.0%) | 0.86811 (+152.6%) |
| 1024 | 0.69955 | 0.81857 (+17.0%) | 1.12344 (+60.6%) | 1.75064 (+150.3%) |

NCU at BS=1024 confirms that this is the intended controlled variable:

| Variant | Duration | DRAM read | BF16 tensor ops | UTCMMA instructions |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 698.18 us | 4.30 GB | 550.23 G | 1.049 M |
| QK 2x | 763.55 us | 4.30 GB | 825.34 G | 1.574 M |
| QK 4x | 1.14 ms | 4.30 GB | 1375.56 G | 2.624 M |
| QK 8x | 2.02 ms | 4.30 GB | 2476.02 G | 4.723 M |

The immediate 6--17% slowdown from QK 2x proves that MMA is already on the
critical path; it is not completely hidden by data movement. The slowdown is
far below the 50% increase in total tensor work, so the pipeline still has
overlap or utilization headroom, but this experiment alone does not identify
TMA as the source of that headroom.

To test the KV pipeline directly, another diagnostic forced its shared-memory
ring from three stages to two or one. With one slot, the TMA producer cannot
issue the next K or V transfer until the UTCMMA consumer releases the current
slot. This removes the producer's lookahead while preserving the output,
tensor work, KV bytes, grid, and register count.

| Batch | Three KV stages (ms) | Two KV stages (ms) | One KV stage (ms) | One-stage slowdown |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.05494 | 0.06648 | 0.11419 | 2.08x |
| 512 | 0.34364 | 0.38043 | 0.57506 | 1.67x |
| 1024 | 0.69955 | 0.77335 | 1.11120 | 1.59x |

At BS=1024, NCU reports the same 4.30 GB DRAM reads, 550.23 G tensor ops,
1.049 M UTCMMA instructions, 168 registers per thread, 27.68 launch waves,
and one resident CTA for both schedules. The one-stage version lowers dynamic
shared memory rather than occupancy. NCU duration changes monotonically from
698.18 us to 881.47 us and 1.59 ms for three, two, and one stage; measured DRAM
throughput falls from 80.36% to 63.65% and 35.26%. Thus the multistage
TMA/UTCMMA producer-consumer pipeline has a material benefit. This comparison
includes both temporal overlap and the extra outstanding-TMA depth; it does
not assign an exact cycle count to TMA-over-MMA overlap. A literal overlap
percentage would require an in-kernel timestamp trace.

The combined conclusion is that TMA/MMA pipelining matters, but MMA is only
partially hidden. The M64 path loses because it fails to reduce issued MMA
instructions and adds mapping overhead, not because reducing arithmetic can
never help.

### Final M64 five-stage follow-up

The first M64 result above used the inherited three-stage KV ring and paid the
new TMEM mapping overhead on every block. The final follow-up keeps N128 but
makes two independent changes: full blocks skip per-element tail predicates,
and the smaller M64 Q/O footprint is reinvested in a five-stage KV ring. On a
B300 BS=1024 NCU comparison, increasing M64 from three to five stages lowers
duration from 677.6 to 629.4 us, lowers long-scoreboard cycles per issued
instruction from 8.67 to 7.87, and raises measured memory throughput from
6.35 to 6.84 TB/s. Registers and occupancy stay unchanged. Six stages do not
help B300 and leave too little shared-memory margin, so five is the fixed
cross-architecture choice.

Across the interleaved 36-case sweep, the final M64/five-stage path is 1.075x
faster than M128 on B300 and 1.159x faster on Rubin in geometric mean. The
single worst paired ratios are 0.998x and 0.992x, within run-to-run noise; the
standalone original-kernel comparison is positive in all 36 cases on both
architectures. M64/N128 with five KV stages and no split is therefore selected
automatically for the supported workload. N64 and the wider-TMEM experiments
remain rejected because their extra loop or register cost outweighs reduced
nominal work.

## Small-MMA Q-major backward

The benchmark can force the four base backward implementations with
`--backward-impl legacy|tc|tc-small|direct`; later sections also use the
`direct-split*` variants. `legacy` is the original tensor-core schedule, `tc`
is the first Q-major experiment, `tc-small` is the new small-MMA path, and
`direct` is the specialized CUDA-core path. The tables below use 10 warmups
and the median of five groups of 30 executions.

### B300 (SM10.3)

| Batch | Q-major TC (ms) | Small-MMA TC (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.2575 | 0.1846 | 1.39x |
| 512 | 1.4395 | 1.0736 | 1.34x |
| 1024 | 2.8185 | 2.1168 | 1.33x |

### Rubin (SM10.7)

| Batch | Q-major TC (ms) | Small-MMA TC (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 64 | 0.1835 | 0.1352 | 1.36x |
| 512 | 0.9274 | 0.7761 | 1.19x |
| 1024 | 1.7963 | 1.5186 | 1.18x |

The new path keeps the FA2 `bwd_loop_opt` loop order: one CTA owns one
batch/head pair, keeps its single Q fixed, and walks the KV tiles. The one
valid dQ row remains local across all KV tiles and is written once, while the
CTA writes its disjoint dK/dV rows directly. There are no dQ atomics, global
FP32 dQ workspace, workspace-zeroing launch, or conversion launch.

The computation is still tensor-core based. S, dP, dK, and dV use an
M128N16K128 tile instead of reserving 128 Q rows. dQ uses the transposed
`K^T @ dS^T` form with M128N8K128, which is the smallest useful N shape for
this SM100-family UMMA path. K is loaded once and reinterpreted as the
transposed dQ operand in shared memory. Four compute warps replace eight for
the reduced Q tile.

## Historical pre-split backward automatic dispatch

Before adding backward split-KV, the crossover was measured at extra batch
sizes rather than inferred from the three target points:

- B300: `tc-small` below BS=64, `direct` through BS=79, `tc-small` through
  BS=127, and vectorized `direct` from BS=128. The two small-grid regions are
  retained because CTA-wave boundaries make the crossover non-monotonic.
- Rubin: `legacy` below BS=128, `tc-small` through BS=191, and vectorized
  `direct` from BS=192.
- Other SM100-family devices retain the previous policy.

For the requested batches, those pre-split automatic choices and timings were:

| GPU | BS=64 | BS=512 | BS=1024 |
| --- | ---: | ---: | ---: |
| B300 | 0.1697 ms (`direct`) | 0.9163 ms (`direct`) | 1.7364 ms (`direct`) |
| Rubin | 0.1272 ms (`legacy`) | 0.6266 ms (`direct`) | 1.1806 ms (`direct`) |

Relative to the pre-vectorization dispatch, this final pass lowers B300 by
7.9%, 2.5%, and 2.5% at BS=64/512/1024. Rubin BS=64 keeps the faster legacy
path, while BS=512/1024 improve by 13.3% and 12.3%.

## NCU pipeline and work analysis

`profile_hstu_qlen1.py` compiles and warms up outside the CUDA profiler range,
then exposes exactly one selected execution to Nsight Compute. The B300 runs
used the command-line `LaunchStats`, occupancy, compute, memory, scheduler,
warp-state, and instruction metrics.

### Useful work and compulsory traffic

Ignoring the small activation cost, forward performs one QK dot product and
one weighted-V accumulation per KV row, about `4 * D = 512` FLOPs per
KV/head. Backward recomputes QK and adds dP, dQ, dK, and dV work, about
`8 * D = 1024` FLOPs per KV/head. Its compulsory traffic is one K/V read and
one dK/dV write. The requested workload is therefore approximately one useful
FLOP per compulsory byte in both directions:

| Batch | Total KV | Forward useful FLOPs / K+V bytes | Backward useful FLOPs / read+write bytes |
| ---: | ---: | ---: | ---: |
| 64 | 130,304 | 0.267 GFLOP / 0.267 GB | 0.534 GFLOP / 0.534 GB |
| 512 | 1,047,808 | 2.146 GFLOP / 2.146 GB | 4.292 GFLOP / 4.292 GB |
| 1024 | 2,098,944 | 4.299 GFLOP / 4.299 GB | 8.597 GFLOP / 8.597 GB |

The forward M128 tensor-core tile has only one valid Q row, so its hardware
MMA work is about 128 times the useful Q-row work, plus KV-tail padding. That
does not make the target workload compute-bound: at BS=1024 NCU reports the
expected 4.30 GB of K/V reads, 88.3% of peak DRAM throughput, and 40.7% tensor
pipe activity. The redundant tensor-core work is cheaper than replacing the
three-stage TMA/MMA pipeline with scalar work.

### Why the CUDA-core forward loses

A second B300 experiment kept the direct kernel's scalar K/V row loads and
loop structure but removed QK, SiLU, and the weighted-V accumulation. The
loaded values were accumulated into a live result so the compiler could not
delete the traffic. With 10 warmups and five groups of 30 executions, full
direct, load-only, and tensor-core timings were:

| Batch | Direct full (ms) | Direct load-only (ms) | Tensor core (ms) |
| ---: | ---: | ---: | ---: |
| 64 | 0.2734 | 0.2174 | 0.0566 |
| 512 | 0.5864 | 0.5671 | 0.3201 |
| 1024 | 1.0123 | 0.9875 | 0.6616 |

At BS=1024, deleting nearly all arithmetic saves only 0.025 ms, while the
load-only path remains 1.49x slower than the tensor-core path. NCU confirms
that this is a copy/pipeline problem rather than useful CUDA-core math. All
three kernels read about 4.30 GB from DRAM, but load-only reaches 57.2% of
peak DRAM bandwidth versus 87.8% for the TMA/tensor-core kernel. It executes
67.30 million scalar global-load instructions and 348.80 million total
instructions; the TMA path reports 0.246 million global-load instructions
and 118.73 million total instructions. Long-scoreboard stall ratio is 26.2
for load-only versus 13.0 for TMA. The diagnostic direct kernels were removed
after profiling; production keeps the TMA/tensor-core implementation.

At BS=64, split4 changes the forward launch from 256 to 1024 CTAs while keeping
the same 267 MB K/V read volume. In NCU it raises DRAM throughput from 59.6%
to 69.4%, tensor-core activity from 26.6% to 33.4%, and launch waves from 1.73
to 6.92; the main kernel falls from 59.3 to 50.9 us. Its included output-zero
launch costs about 4.2 us. This is why split-KV helps the small batch but does
little once the unsplit kernel already saturates memory.

### Vectorized direct backward and CTA tuning

The original direct kernel launched 256 threads (8 warps) per batch/head CTA.
Each warp serially loaded one KV row, reduced QK and dO-V, wrote dK/dV, and
then advanced by eight rows. NCU identified long-scoreboard stalls as the
dominant bubble, with no excess DRAM traffic and only 34 registers per thread.

The first optimization widened SM103 and SM107 to 512 threads (16 warps).
The math, Q-major loop order, and one final dQ write stayed unchanged, while
twice as many independent KV rows hid more memory latency. The dQ shared
reduction grew from about 5.1 to 9.2 KB per CTA, which is not the occupancy
limiter.

| B300 NCU metric | BS64, 8 warps | BS64, 16 warps | BS1024, 8 warps | BS1024, 16 warps |
| --- | ---: | ---: | ---: | ---: |
| Kernel duration | 383.8 us | 210.9 us | 1.86 ms | 1.78 ms |
| DRAM throughput | 16.9% | 30.6% | 59.9% | 62.8% |
| Active warps | 18.6% | 37.5% | 69.6% | 73.0% |
| Eligible warps / scheduler cycle | 0.21 | 0.50 | 0.97 | 1.04 |
| Issue-active | 18.3% | 33.9% | 47.3% | 49.2% |

Instruction profiling then exposed another bottleneck. The scalar kernel
issued eight coalesced global loads and eight stores per KV/head row. Assigning
four adjacent BF16 values to each lane allows one 64-bit K load, V load, dK
store, and dV store. At BS=1024 this reduces executed global-load instructions
from 67.95 million to 17.38 million and stores from 67.18 million to 16.81
million. The total executed instruction count falls from 999.34 million to
863.02 million without changing DRAM bytes or arithmetic.

The vector fragments increase register pressure. A second CTA sweep therefore
keeps 16 warps for small B300 grids, but uses 12 warps from BS=448 so five CTAs
can reside per SM. Rubin consistently prefers 16 warps. Final B300 BS=1024 NCU
reports 44 registers/thread, 7.2 KB shared memory, 64.4% peak DRAM throughput,
54.8% active warps, 0.74 eligible warps per scheduler cycle, and 43.5%
issue-active. Although issue occupancy is lower than the scalar 16-warp
kernel, the 13.6% smaller instruction stream and higher DRAM utilization lower
kernel time from about 1.78 to 1.73 ms.

At BS=64 the final 16-warp vector kernel keeps active warps essentially flat
at 37.7%, but raises eligible warps from 0.50 to 0.57, issue-active from 33.9%
to 35.6%, and DRAM throughput from 30.6% to 37.7%. NCU kernel time falls from
210.9 to 170.6 us.

The complete direct-path progression is:

| GPU | Batch | 8-warp scalar (ms) | Wider scalar (ms) | Final vector (ms) | Final vs scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.3851 | 0.2121 | 0.1697 (16 warps) | -55.9% |
| B300 | 512 | 1.0303 | 0.9396 | 0.9163 (12 warps) | -11.1% |
| B300 | 1024 | 1.8698 | 1.7810 | 1.7364 (12 warps) | -7.1% |
| Rubin | 64 | 0.3346 | 0.1847 | 0.1512 (16 warps) | -54.8% |
| Rubin | 512 | 0.8042 | 0.7227 | 0.6266 (16 warps) | -22.1% |
| Rubin | 1024 | 1.4247 | 1.3462 | 1.1806 (16 warps) | -17.1% |

Several rejected variants confirm the resource tradeoff. Before vectorization,
a 384-thread CTA was slower than 512 threads; after vectorization changed the
register footprint, retuning made 384 threads best for large B300 grids. A
1024-thread CTA reduces CTA residency and remains much slower at large batch
sizes. Vector loads with scalar stores were also rejected: the scalar stores
raise final B300 BS=1024 from 1.74 to 2.09 ms.

### Historical per-shape split-KV direct backward

The direct qlen=1 backward is also additive over KV partitions for dQ. Each
split CTA owns a contiguous KV range, writes its disjoint dK/dV rows normally,
and reduces its local dQ contribution in shared memory. Only the final 128
dQ values need cross-CTA combination; 64 threads issue packed BF16x2 atomic
adds after a small dQ zero-fill. There are no dK/dV atomics and no partial
gradient workspace.

The split count and CTA size were swept in the same allocation. Splitting by
2 or 4 is too little for the small grid, while excessive splitting repeats
Q/dO loads, metadata, CTA setup, local dQ reduction, and atomics. The measured
choices target roughly 32K short CTAs, cap the split at 64, and retain at least
about 32 average KV rows per CTA. B300 accepts split8 as its last useful
crossover; Rubin requires at least split16 because its unsplit direct kernel
is already faster at large grids.

The following early automatic choices and same-allocation comparisons used 10
warmups and the median of five groups of 30 executions. They were first
superseded by fixed split22/split26, then by the D128-only two-row
split13/split16 policy from its 36-case sweep:

| GPU | Batch | Previous dispatch (ms) | Split selection | Final (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| B300 | 64 | 0.1727 (`direct`) | 64 | 0.1178 | 1.47x |
| B300 | 512 | 0.9204 (`direct`) | 16 | 0.8444 | 1.09x |
| B300 | 1024 | 1.7438 (`direct`) | 8 | 1.6658 | 1.05x |
| Rubin | 64 | 0.1310 (`legacy`) | 64 | 0.0904 | 1.45x |
| Rubin | 512 | 0.6653 (`direct`) | 16 | 0.6457 | 1.03x |
| Rubin | 1024 | 1.2435 (`direct`) | 1 | 1.2435 | 1.00x |

The B300 BS=64 NCU comparison makes the tradeoff explicit. The main kernel
falls from 177.7 to 114.2 us; the included dQ zero-fill is 10.3 us. Total
K/V and gradient DRAM traffic stays near 492 MB, while active warps rise from
37.6% to 58.7%, eligible warps per scheduler cycle from 0.50 to 0.85, issue
activity from 34.1% to 47.1%, and DRAM throughput from 36.1% to 56.1%.
Executed instructions increase from 51.63 to 60.58 million, so the speedup
comes from latency hiding and higher memory utilization despite extra work.

Forced split8, split16, and split64 pass the boundary-length CPU oracle on
B300 and Rubin. dK/dV maximum absolute errors remain below 1e-6. Packed BF16
dQ atomics introduce order-dependent rounding; observed dQ maximum absolute
error is 2e-5 to 6e-5. Deterministic HSTU backward was already unsupported.

The early Rubin node runs normal kernels and CUDA-event benchmarks, but both
Nsight Compute 2025.3.1 and 2026.2.1 fail to initialize its hardware counter
library (`Failed to initialize LOP`, `LibraryNotLoaded`) with driver 615.12.
Rubin therefore uses the same kernel-structure diagnosis from B300 plus direct
before/after timing and correctness measurements on SM107.

### Two KV rows per warp at D128

The D128 backward pass divides each warp into two 16-lane groups. Each group
owns one KV row, loads/stores eight adjacent BF16 values per lane with one
128-bit transaction, performs width-16 QK and dO-V reductions, and writes its
disjoint dK/dV row directly. The two groups combine their accumulated dQ in
registers before the existing CTA reduction. The high-level loop order is
unchanged; split-KV still introduces only the packed dQ atomic reduction.

A same-split B300 NCU comparison isolates this change at BS=1024:

| Metric | 1 KV row/warp, split22 | 2 KV rows/warp, split22 |
| --- | ---: | ---: |
| Kernel duration | 2.132 ms | 1.918 ms |
| Executed instructions | 952.96 M | 752.86 M |
| Estimated DRAM bytes | 8.648 GB | 8.648 GB |
| Long-scoreboard cycles / issued instruction | 7.76 | 6.07 |
| Registers / thread | 44 | 72 |
| Grid CTAs | 90,112 | 90,112 |

Thus pairing itself is 1.112x faster with identical grid, split count, and
memory volume. Retuning the fixed split from 22 to 13 lowers the same NCU run
further to 1.848 ms. The D128 36-case interleaved sweep improves over the
previous optimized automatic path by 1.064x on B300 with fixed split13 and
1.208x on Rubin with fixed split16; all 36 cases improve on both architectures.

Four D128 KV rows per warp was also implemented with two 128-bit transactions
per lane. It helps the low-grid BS=64 target by roughly 5--7%, but is slower at
BS=512/1024 and increases the boundary-oracle error by about two orders of
magnitude. It is rejected rather than adding another shape-dependent branch.
D64 can pack four rows while retaining one aligned 128-bit transaction per
lane, so it does not have the same tradeoff.

### Rejected forward residency experiment

The qlen=1 Q and output shared-memory lifetimes can be made disjoint. Reusing
that storage and reducing the KV pipeline from three stages to two lowers the
CTA allocation from about 166 KB to 96 KB and permits two resident CTAs per
SM. It remains numerically correct, but the shallower pipeline slows B300 by
about 10% at BS=64/512 and 15% at BS=1024. A driver-level asynchronous memset
also has no measurable advantage over the existing output zeroing. Neither
experiment is retained.

## Design

The forward kernel uses one Q stage for a one-token query, a native M64 TMEM
mapping, 16dp SiLU, and tail-only masking. D64 and D128 use the same unsplit
five-stage M64/N128 schedule for causal and local attention on both GPUs.
D256 uses the same kernel family with its capacity-limited three-stage pipeline.

Backward always uses the vectorized Q-major CUDA-core `hstu_bwd_q1.py` for the
supported target. A warp packs four D64 rows, two D128 rows, or one D256 row
while retaining aligned 128-bit lane transfers. The metadata-only selector
uses unsplit for short work or an already-large base grid; otherwise it uses
split8 on B300 and split13 on Rubin. Large-grid unsplit launches use 128-thread
CTAs, while smaller unsplit grids retain the wider architecture-level block.
Causal and local attention share the selector and kernel source. Each CTA
writes disjoint dK/dV rows directly; only split-KV dQ is atomically combined.
Unsupported layouts, full/arbitrary masks, paged KV, and FP16 keep the existing
path.

This is the same high-level loop-order lesson as the FA2 `bwd_loop_opt`
branch, specialized further for the exact one-query case.

## Correctness

The benchmark checks D64/D128/D256 and KV lengths `1, 127, 128, 2049, 3072`
against a float32 PyTorch oracle. The oracle runs on CPU so it also works on
early Rubin systems whose PyTorch device-code toolchain does not recognize
SM10.7. Forced and automatic forward/backward paths pass on B300 and Rubin;
observed forward absolute error stays below `1.6e-5`, dQ below `5e-5`, and
dK/dV below `1e-6`. BF16 dQ atomic ordering accounts for its larger error.

The D64/D128/D256 local-window oracle covers left/right boundary pairs
`(0, 0)`, `(1, 0)`, `(63, 0)`, `(64, 8)`, `(128, 0)`, and `(255, 0)` over KV
lengths 1, 63, 64, 127, 128, 129, and 257. It passes on B300 and Rubin.
