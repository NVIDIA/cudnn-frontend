# SM90 KDA FP32 P: H20 precision and cost

Updated **2026-11-12** with the optimized implementation. Compare upstream
`8348887bd580770ec2dd2339ec919f06dbaf8702`, the initial FP32-P implementation
`719e5111a8eb9122785f0a0900522aba7323a8cb`, and the current packed/prefetched
FP32 implementation. Source hashes and samples are in
[the JSON records](results/kda/h20_fp32_p/performance-optimized.json).
The original `performance-final.json` remains as the initial PR measurement.

H20-3e, driver 570.133.20, PyTorch 2.9.1+cu128, CUDA Toolkit 12.8,
cuDNN 9.10.2, CuTe DSL 4.8.1. Inputs are generated on CPU from fixed seeds.
SM100 is unchanged and was not retested.

## What changes

The reverse recurrence adds `P = Q_g^T dO` into an FP32 state gradient. Its
Tensor Core accumulator was FP32, but storing P in BF16 discarded low bits
before that addition. P now stays FP32 through storage, loading and addition.
BF16 Tensor Core input operands, the inverse, saved state operands and other
BF16 intermediates retain their existing precision.

The optimized implementation packs P as `[8 key tiles][16 value tiles][16][8]`.
Each warp writes contiguous 8x8 halves of its MMA tiles directly from FP32
registers. The reverse scan copies only its 16- or 32-column value slice into
its existing three-stage asynchronous prefetch pipeline, then reads contiguous
FP32 pairs before the final projection. There is no repacking kernel and no
change to arithmetic order, public APIs, compile specializations or global
workspace capacity.

Each P stage reserves 16 KiB, sufficient for the widest 128x32 FP32 slice.
Dynamic scan shared memory is **198,144 bytes per CTA**, up from 148,992 in the
initial FP32 PR and equal to upstream BF16's padded staging allocation.
The unchanged 1 KiB runtime/static allocation is separate. Occupancy remains
one CTA per SM. The initial PR's 48 KiB shared-memory saving is superseded by
this faster pipeline. The initial seed-consumption fix is retained: the seed
feeds its existing BF16 MMA-operand staging before aliased storage is reused,
while the recurrent accumulator stays FP32.

## Precision

The focused case has unit keys on one coordinate, queries on an orthogonal coordinate, `g=0`, and exactly representable BF16 upstream gradients. It produces `P=1+2^-10` and an incoming FP32 gradient of `-1`. The correct residual is `2^-10 = 0.0009765625`. Baseline returns zero; FP32 P returns the exact residual. All 10 shape/scan variants fail on the baseline and pass after the change.

The following are relative L2 errors against an independent FP64 token recurrence, in percent; each cell is **baseline → FP32 P**. H=2 and K=V=128. All eight tensors are finite and byte-identical across fresh-allocation replays.

| Input | O | Final state | dQ | dK | dV | dG | dBeta | dInitialState |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ragged | 0.4164 → 0.4164 | 0.3213 → 0.3213 | 0.4011 → 0.4011 | 0.4076 → 0.4075 | 0.3630 → 0.3627 | 0.3577 → 0.3572 | 0.3355 → 0.3371 | 0.1235 → 0.1028 |
| weak | 0.4221 → 0.4221 | 0.3245 → 0.3245 | 0.4010 → 0.4010 | 0.4753 → 0.4623 | 0.3665 → 0.3505 | 0.4360 → 0.4202 | 0.3584 → 0.3217 | 0.3027 → 0.2568 |
| steep | 0.4178 → 0.4178 | 0.2029 → 0.2029 | 0.2193 → 0.2193 | 0.2517 → 0.2517 | 0.3181 → 0.3181 | 0.6891 → 0.6891 | 0.2225 → 0.2225 | 0.2394 → 0.1646 |
| correlated | 0.6296 → 0.6296 | 0.4497 → 0.4497 | 1.1711 → 1.1711 | 0.3322 → 0.3322 | 1.0410 → 1.0410 | 4.3598 → 4.3598 | 1.1784 → 1.1784 | 0.1417 → 0.0957 |
| long | 0.4495 → 0.4495 | 0.3577 → 0.3577 | 0.4298 → 0.4298 | 0.4678 → 0.4536 | 0.4271 → 0.4116 | 0.4675 → 0.4553 | 0.4111 → 0.3966 | 0.2865 → 0.2244 |

Ragged lengths are `[0,1,15,16,17,63,64,65]`; weak decay is T=257 with g=-1e-6; steep decay is T=65 with g=-5; correlated keys use T=65 with two repeated nonzero coordinates and g=0; the long random case has T=1024. These measurements isolate one rounding removal. The correlated case still has a larger dG error: this PR does not claim to fix all SM90 precision differences, establish loss parity, or replace a distributed training test.

The optimized implementation reproduces every error record in this table
exactly. All six gradients also match the initial FP32 PR **byte for byte** in
each of the performance cases below, including 128K/H32 and THD22. The strict
forward/backward golden is unchanged. No tolerance was relaxed.

## Performance and memory

CUDA-event latency for prepared launches captured in CUDA graphs; compilation,
allocations, references and Python autograd dispatch are outside timing.
Median of 9 alternating-order groups, 20 replays/group. All three implementations
use the same inputs/output addresses and separate scratch. F+B includes the
unchanged forward kernel and the complete backward chain. These are kernel-chain
measurements, not full-model training or distributed CP timings.

| THD workload | Upstream BF16 ms | Initial FP32 ms | Optimized FP32 ms | BWD reduction vs initial FP32 |
| --- | ---: | ---: | ---: | ---: |
| 1k-h8 | 0.1221 | 0.1319 | 0.1240 | 6.02% |
| 16k-h8 | 1.5703 | 1.7142 | 1.5970 | 6.84% |
| 128k-h8 | 12.1964 | 13.3750 | 12.3857 | 7.40% |
| 128k-h32 | 43.4178 | 47.6848 | 43.8784 | 7.98% |
| 128k-thd22-h8 | 11.5486 | 13.2340 | 11.5839 | 12.47% |
| ragged-h8 | 0.4581 | 0.5146 | 0.4600 | 10.61% |

| THD workload | Upstream BF16 ms | Initial FP32 ms | Optimized FP32 ms | Remaining F+B overhead vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| 1k-h8 | 0.4279 | 0.4377 | 0.4291 | +0.30% |
| 16k-h8 | 6.4606 | 6.6033 | 6.4859 | +0.39% |
| 128k-h8 | 52.5640 | 53.7411 | 52.7525 | +0.36% |
| 128k-h32 | 86.4734 | 90.7348 | 86.9478 | +0.55% |
| 128k-thd22-h8 | 21.4761 | 23.1608 | 21.5110 | +0.16% |
| ragged-h8 | 1.1868 | 1.2535 | 1.1987 | +1.00% |

128K is 131,072 tokens. THD22 is **20 x 256 + 2 x 62,976** tokens. The small
ragged case is `[0,1,65,257,4097]`.

## Why it is faster

A five-replay CUDA profiler sample at 16K/H8 measures prep **509.22 → 438.04 us**
and scan **598.42 → 554.39 us**, initial FP32 → optimized. The final backward
kernel remains unchanged within noise, **609.97 → 610.51 us**.

Nsight Compute on the same 16K/H8 geometry supports the memory-scheduling
explanation: prep's LG-throttle stall ratio drops **2.028 → 0.204**, and scan's
long-scoreboard ratio drops **0.323 → 0.252** (reported per issued instruction).
Prep uses 114 rather than 116 registers/thread; this scan specialization uses
128 rather than 130. These are not universal register reductions: the final
BV32 variants use 133/132/143 registers and still have no local spills.
The same global P bytes are transferred; the improvement comes from contiguous
warp accesses and overlapping P copies with scan computation, not a smaller
workspace. Raw selected counters and source hashes are in
[`ncu-summary.json`](results/kda/h20_fp32_p/ncu-summary.json).

An initial tile-layout-only experiment was faster than row-major P; adding
asynchronous staging improved it further. A four-float vector experiment was
slower in prep and was not retained. No additional algorithm dispatch or
precision option is introduced.

## Memory

| THD workload | Upstream workspace MiB | FP32 workspace MiB | Public F+B peak GiB, upstream → FP32 |
| --- | ---: | ---: | ---: |
| 1k-h8 | 25.06 | 29.06 | — |
| 16k-h8 | 401.00 | 465.00 | 0.928 → 0.990 |
| 128k-h8 | 3208.02 | 3720.02 | 7.397 → 7.897 |
| 128k-h32 | 12832.02 | 14880.02 | 29.588 → 31.588 |
| 128k-thd22-h8 | 3225.16 | 3742.41 | 7.466 → 7.970 |
| ragged-h8 | 112.21 | 130.71 | — |

Public peaks were rerun for the optimized implementation and exactly match the
initial FP32 PR. Upstream peaks are retained from the original fresh-process
measurement. The FP32 workspace increment remains
`max(ceil(T/64)+N-1,1) * H * 128 * 128 * 2` bytes: **512 MiB at 128K/H8/N1**,
**2 GiB at H32**, and **517.25 MiB at THD22/H8**. Global memory cost has not
been eliminated.

Public peak allocation uses `torch.cuda.max_memory_allocated()` around warmed
public forward + autograd in a separate process. It includes fixed inputs,
cloned differentiable leaves, outputs, gradients and transient workspace;
it excludes driver memory, allocator-reserved free blocks and CUDA graph pools.
It is not a full-model memory figure.

## Reproduce

Use unchanged checkouts for upstream (`$BASELINE`) and the initial FP32 commit
(`$PREVIOUS_FP32`). Set `PYTHONPATH` to this branch with its built `cudnn`
binding and verify `cudnn.__file__` before running.

```bash
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode baseline --output baseline-precision.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode fp32 --output optimized-precision.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --previous-fp32 "$PREVIOUS_FP32" --mode performance --output performance-optimized.json
python benchmark/linear_attention/benchmark_sm90_kda_precision.py --baseline "$BASELINE" --mode fp32 --memory --output optimized-memory.json
```

The sanitizer harness uses the shipped prepared backward launcher, five
geometries covering every scan specialization, exact-sized scratch surrounded
by 256-byte canaries, all six gradients and ten graph replays per geometry:

```bash
for check in memcheck racecheck synccheck; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False compute-sanitizer --tool "$check" --error-exitcode 99 \
    python benchmark/linear_attention/check_sm90_kda_workspace.py
done
```

Public autograd tests remain separate: CUDA 12.8 sanitizer reports handled
`cuCtxGetDevice` probes on autograd worker threads as API errors, and its default
expandable allocator fails `cuMemCreate` on this host. No API-error suppression
is used in the prepared-launch run.

## Review and coverage

The optimized source passes **47 tests, 1 optional-cuTile skip**, including all
23 precision/determinism cases, the unchanged strict golden and **6,000 clean
warm/poisoned replays**. The unsafe control fails 200/200. See the
[dated determinism record](../../test/python/determinism_goldens/README.md#sm90-cuda-kda-fp32-p--h20-validation-2026-11-11)
for commands, scope and retained original regression evidence. The optimized
prepared launcher passes **memcheck: 0 errors; racecheck: 0 errors/0 warnings;
synccheck: 0 errors** across all five scan variants.

Compatibility audit: passed for the changed packed BF16 CUDA path, both scan
widths, full/partial chunks, empty documents, nonzero boundary states/gradients,
stream routing, graph replay and existing serving/layout cases. No API or
dispatch gate changed. Docstring audit: passed for the KDA wrapper, workspace
sizing and launcher. Other GPUs and distributed training remain unverified.

Memory ordering: prep has one writer per P element and completes on the scan's
stream. P joins the existing cp.async commit/wait and CTA publication/reuse
protocol. Each packed slice is consumed before its stage can be refilled.
Generated SASS for all five scan specializations preserves all 16 seed-word
uses before seed release; all 16 loaded P words feed FP32 additions before any
branch or release. On loop continuation, accumulator staging precedes the
barrier permitting refill. Source ownership review, sanitizer and warm stress
supplement these concrete generated-code checks; the site record is
[`optimized-sass-audit.json`](results/kda/h20_fp32_p/optimized-sass-audit.json).
