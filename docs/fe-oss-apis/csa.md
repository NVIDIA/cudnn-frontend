# CSA Fused Compressor

**This is an experimental API and subject to change.**

## Overview

The CSA module hosts CuTe-DSL kernels for the CSA/HCA experimental attention variants
(the components that are not shared with the [DSA module](dsa.md)). Its first operation
is the **fused Compressor**: one forward and one backward kernel for the `Compressor`
gated-softmax pooling region (THD packed layout) used by CSA/HCA in Megatron-LM.

The kernels were ported from Megatron-LM at the maintainers' request
([Megatron-LM PR #5984](https://github.com/NVIDIA/Megatron-LM/pull/5984); measurements
and numerics in
[Megatron-LM issue #5968](https://github.com/NVIDIA/Megatron-LM/issues/5968)). The eager
region they replace decomposes into ~39 forward and ~51 backward kernel launches per
call (at `compress_ratio = 4`, `coff = 2`) and materializes `(total_comp, 2*ratio, 1, head_dim)`
window intermediates; the fused path is 1 + 1 kernels (plus one fp32 `dAPE`-buffer
zero-fill in backward — the backward kernel writes `dKV`/`dScore` in full, including
exact zeros to never-consumed positions, so those buffers need no fills).

### Semantics

For each THD segment `s` (`cu_seqlens[s]..cu_seqlens[s+1]`) and each output block `b` of
`ratio` tokens, with the overlapping window form (`coff == 2`, window size `2 * ratio`):

- `k in [0, ratio)`: previous block's token, first-half projection column, APE row `k`
  — invalid for the segment's first block (score `-inf`, kv `0`);
- `k in [ratio, 2*ratio)`: own block's token, second-half projection column, APE row
  `k - ratio`.

The own-block window form (`coff == 1`, window size `ratio`) drops the overlap: every
`k in [0, ratio)` is the block's own token on projection column `j` with APE row `k`, and
every window is fully valid (no first-block exception).

```text
out[b, j] = sum_k kv[w(b,k), c(k,j)] * softmax_k(score[w(b,k), c(k,j)] + ape[k % ratio, c(k,j)])
```

computed in fp32 with a single final bf16 rounding. Per-segment tail tokens
(`seqlen % ratio`) are dropped, as in the eager code. Output rows beyond
`cu_seqlens_comp[-1]` (a static CUDA-graph capacity) are computed with first-in-segment
semantics from token 0, exactly like the eager gather; the backward ignores incoming
gradients on such padding rows.

### Numerics

All arithmetic is fp32 with one final bf16 rounding; `mul.rn.f32` / `fma.rn.f32` are
pinned in PTX so results do not depend on compiler FMA contraction. The numerics
contract is **per ratio family** (the two families intentionally differ — do not
assume the ratio=4 guarantees at ratio=128):

- **`ratio == 4`** (production, unchanged): against an fp32-intermediate eager
  reference (same op order, fp32 throughout), `dKV`/`dScore` are **bit-identical**
  and the forward matches within one bf16 rounding step on a tiny fraction of
  elements.
- **`ratio == 128`** (deterministic tolerance contract): the kernels are
  **deterministic and faithful to the fp32-intermediate eager reference** (the
  same eager region computed with fp32 intermediates and one final bf16 rounding —
  the comparator every number below is measured against):

  1. **Determinism** — forward `out` and backward `dKV`/`dScore` are bitwise
     run-to-run deterministic on all supported inputs (see below; NaN-prefill
     replay tested, including bit-stable replay of the NaN pattern on the gate's
     overflow-intermediate case),
  2. **Same values within final-bf16 rounding on the gate tolerances** —
     `out`/`dKV`/`dScore` match the eager reference within differing elements
     `<= max(1, 0.1%)` of the tensor and `max_abs <= 1.6e-2`, thresholds
     **calibrated on the gate's documented input distribution** (`kv`, `grad_out`
     ~ N(0, 1) bf16, `score` ~ N(0, 1.5²) bf16, `ape` ~ N(0, 0.25²) fp32).
     Absolute bf16 deviations are NOT magnitude-free — bf16's grid is relative, so
     scaling `kv`/`grad_out` by `2^k` scales every deviation exactly `2^k`, with
     differing-element counts and fp64 parity unchanged for as long as the scaled
     inputs keep every fp32 intermediate finite (see 3.). The gate's ×64
     scaled-input case commits that as evidence (deviations ×64 forward/`dKV`,
     ×4096 `dScore`; counts and parity identical to its unit-scale twin), and
  3. **The eager reference's non-finite propagation** — both sides compute the
     window math in fp32, so finite inputs that overflow the eager reference's
     fp32 intermediates (a `score + ape` sum or a backward `kv · grad_out` product
     beyond fp32 range — elementwise ops both sides evaluate identically) poison
     the kernels' outputs as they poison the reference's, instead of the kernels
     returning clean-looking finite values. The gate's overflow-intermediate case
     commits this: `score` = bf16 max with `ape` = fp32 max drives fp32
     `score + ape` to +Inf, every fused output is NaN with the NaN mask equal to
     the eager reference's on all four outputs, and the pattern replays
     bit-stable; the fp64 comparator is explicitly skipped there because the fp64
     oracle stays finite (a NaN-vs-finite distance gates nothing). The mirror
     caveat: the fused evaluation order saturates EARLIER than the eager one near
     fp32 max — its forward chunk partials are un-normalized (up to `2 · ratio ×`
     the reference's normalized weighted sums), so e.g. uniform-score inputs with
     `|kv|` within a factor `~2 · ratio` of fp32 max return ±Inf where the eager
     reference is still finite.

  On inputs whose fp32 intermediates stay finite in BOTH evaluation orders — the
  eager reference's and the fused kernels' (which saturate earlier, see 3.) —
  `out`/`dKV`/`dScore` additionally carry **fp64-oracle parity**: per case and per
  tensor they are **at least as close to an fp64 oracle as the fp32-intermediate
  eager reference itself** (`err_fused <= err_eager * (1 + 1e-6) + 1e-4`, asserted
  in the tests and on every finite-intermediate case of the committed gate
  script): the fused reorders and fast-exp buckets do not lose accuracy against
  the eager reference on the tested envelope.

  Worst observed on the gate distribution (19 unit-scale cases = 15 union — the
  128k-token d=512 packs select the large-bucket schedules — + 4 padding, B200;
  reproduce with `benchmark/csa/gate_csa_compressor_r128.py`, which also asserts
  the case list selects every shipped schedule): forward 0.0275% differing elements
  (9/32,768), `dKV` 0.0041% (2,770/67,108,864), `dScore` 0.0113% (104/921,856);
  worst `max_abs` 3.91e-3 for forward/`dKV` (the 131k-token d=512 cases) and
  1.95e-3 for `dScore` (bf16-rounding magnitudes). The deviations come from
  measured-win reduction reorders (forward chunked-softmax merges; the backward's
  fused `den`/`S` partial merge with a hoisted `1/den`) and an `ex2.approx`-based
  fast exp enabled on specific schedule buckets in BOTH directions (forward:
  per-bucket table entries; backward: the default outside the d=128 vec=1
  small-pack buckets). `dAPE` (fp32, atomic accumulation) is gated at
  `max_abs <= 1e-3` vs the fp32 eager reference on the gate distribution — observed
  worst 2.0e-6, run-to-run replay deltas of the same magnitude — and scales the
  same way (observed 2.0e-3 under the ×64 case). `dAPE`'s distance from the fp64
  oracle fluctuates run to run around the eager reference's own (within ~1.5×
  either way — atomic reduction-order noise), so its parity check is asserted on
  the unit-scale cases, where the formula's absolute term dominates, and recorded
  on scaled inputs.

In BOTH families the forward, `dKV` and `dScore` are **bitwise run-to-run
deterministic** (fixed chunk boundaries and merge orders, no atomics in those
outputs; NaN-prefill replay is part of the test suite). `dAPE` is reduced with one
fp32 atomic per `(k, dim)` per CTA in both families and is **not** bitwise
run-to-run deterministic; the backward APIs raise under
`torch.use_deterministic_algorithms(True)` (warn-only mode warns and runs) — callers
that need a fully deterministic backward must use an eager implementation.

### Support surface (`check_support`)

- Compute capability **10.0** (the only validated architecture so far; the kernels use
  no arch-specific features, wider enablement is possible after validation)
- `ratio == 4`, `coff in {1, 2}` (`coff == 2` is the production CSA/HCA configuration,
  `coff == 1` the own-block window form) — served by the generic kernels, which are
  generic over `(ratio, head_dim, coff in {1, 2})` but keep the whole pooling window in
  registers (register-bound beyond `ratio = 32`)
- `ratio == 128`, `coff in {1, 2}`, `head_dim in {128, 512}` — served by dedicated
  kernels (see [ratio=128](#ratio128) below); the wrappers and class APIs
  route by `ratio` transparently, and the gate can be widened per configuration once
  validated. **The ratio=128 numerics contract differs** (see [Numerics](#numerics))
- BF16 `kv` / `score` / `out`, FP32 `ape`, int32 `cu_seqlens` / `cu_seqlens_comp`
- int32 flat offsets: `total_tokens * coff * head_dim < 2**31` and
  `total_comp * head_dim < 2**31`
- `total_comp > 0` requires `total_tokens >= ratio` (each compressed row gathers a
  window of `ratio` tokens)
- `head_dim <= 8388480` (forward launch `gridDim.y` bound)
- contiguous tensors on one CUDA device, with 16-byte-aligned base pointers (4-byte
  for the int32 cu_seqlens) — contiguity does not imply base alignment for
  storage-offset views, so this is checked per call

## Installation

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## API Usage

### High-level wrappers

```python
from cudnn import CSA

# forward: (total_tokens, coff*head_dim) BF16 kv/score, (ratio, coff*head_dim) FP32 ape,
# (B+1,) int32 cu_seqlens / cu_seqlens_comp
result = CSA.csa_compressor_forward_wrapper(
    kv, score, ape, cu_seqlens, cu_seqlens_comp,
    ratio=4, head_dim=128, coff=2,
    total_comp=None,  # defaults to cu_seqlens_comp[-1] (synchronizes); pass a static
                      # capacity explicitly to stay CUDA-graph capture-safe
    stream=None,
)
out = result["out"]  # (total_comp, head_dim) BF16

grads = CSA.csa_compressor_backward_wrapper(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out,
    ratio=4, head_dim=128, coff=2, stream=None,
)
grad_kv, grad_score, grad_ape = grads  # BF16, BF16, FP32
```

Set `coff=1` for the `ratio`-token own-block window (and use `kv` / `score` tensors whose
packed width is `head_dim`); `coff=2` selects the `2 * ratio` overlapping window shown
above.

The wrappers cache compiled API instances; the underlying JIT is shared per
`(ratio, head_dim, coff, device)`, so runtime shape changes never recompile.

### Class API

```python
from cudnn import CSACompressorForward, CSACompressorBackward

op = CSACompressorForward(
    sample_kv, sample_score, sample_ape, sample_cu_seqlens, sample_cu_seqlens_comp,
    sample_out, ratio=4, coff=2,
)
op.check_support()
op.compile()
op.execute(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, current_stream=None)
```

`CSACompressorBackward.execute` additionally takes `grad_out` and the
`grad_kv` / `grad_score` / `grad_ape` buffers. `grad_kv` / `grad_score` may be
**uninitialized**: the kernel writes every position (disjoint, atomic-free stores;
never-consumed positions — segment tails, the last block's first-half columns
(`coff == 2` only; `coff == 1` has no first-half columns), segments shorter than `ratio`,
token-capacity padding beyond `cu_seqlens[-1]` — get exact zeros
from their unique owning CTA, matching autograd). When `total_comp == 0` the kernel is
not launched and the buffers are left untouched (zero them yourself if you need
autograd's exact zeros; the high-level wrapper does). `grad_ape` must be
**zero-initialized before every `execute` call and before every CUDA-graph replay
that reuses the buffer** — the kernel only accumulates into it (fp32 atomics) and
never clears it. (The high-level wrapper allocates a fresh zeroed `grad_ape` per
call; the zero-fill is captured together with the kernel, so wrapper graph replays
re-zero automatically.)

### CUDA graphs

The launch path is capture-compatible once the kernels for a `(ratio, head_dim, coff)`
configuration are compiled: run one warmup call (or `compile()`) per configuration
before capturing, and pass `total_comp` explicitly. A call that would JIT under capture
raises a `RuntimeError` instead of corrupting the capture.

### Environment variables

- `CUDNNFE_CSA_COMPRESSOR_FAST_LAUNCH=0` — disable the cached-launch host optimization
  (a per-config snapshot of the CuTe-DSL launch state, replayed with in-place argument
  mutation; it removes tens of microseconds of per-call host overhead for these
  microsecond-scale kernels). The snapshot construction introspects
  private-but-stable DSL internals; on any structural mismatch (e.g. a future
  `nvidia-cutlass-dsl` upgrade) it falls back to the regular launch path automatically.

## Performance

Measured on 1x B200 (CC 10.0, driver 590.48.01); BF16 `kv`/`score`, FP32 `ape`; `ratio = 4`, `coff = 2`;
THD packs of 8192-token sequences; eager baseline = the exact replaced region of
Megatron-LM `Compressor._forward_thd` on identical inputs.

*Isolated GPU kernel time* (nsys, sum of kernel durations per iteration, 50 iterations
after 20 warmup; no launch/host overhead; backward includes its `dAPE` zero-fill):

| THD pack | head_dim | eager fwd | fused fwd | fwd | eager bwd | fused bwd | bwd |
|---|---|---|---|---|---|---|---|
| 1 x 8192 | 128 | 117.8 us | 4.5 us | **26.5x** | 187.2 us | 12.8 us | **14.6x** |
| 3 x 8192 | 128 | 229.8 us | 10.0 us | **23.0x** | 352.7 us | 22.2 us | **15.9x** |
| 1 x 8192 | 512 | 263.3 us | 12.4 us | **21.2x** | 425.0 us | 22.8 us | **18.6x** |
| 3 x 8192 | 512 | 664.3 us | 35.0 us | **19.0x** | 1155.8 us | 66.0 us | **17.5x** |

*End-to-end wall clock of the same region* (CUDA events, median of 100 after 30 warmup;
includes launch overhead; eager backward goes through torch autograd with the forward
outside the timed region, fused backward is the explicit backward wrapper call — not
comparable to the kernel-time numbers above):

| THD pack | head_dim | eager fwd | fused fwd | fwd | eager bwd | fused bwd | bwd |
|---|---|---|---|---|---|---|---|
| 1 x 8192 | 128 | 343.7 us | 37.4 us | **9.2x** | 558.7 us | 51.9 us | **10.8x** |
| 3 x 8192 | 128 | 389.3 us | 38.2 us | **10.2x** | 585.8 us | 62.4 us | **9.4x** |
| 1 x 8192 | 512 | 423.3 us | 39.9 us | **10.6x** | 666.5 us | 62.8 us | **10.6x** |
| 3 x 8192 | 512 | 831.7 us | 62.7 us | **13.3x** | 1503.6 us | 108.3 us | **13.9x** |

The previously published per-call wall clock (commit `b3ceb7c`) was `333.7 / 37.8 / 506.7
/ 47.2 us` at `1 x 8192 / 128` (and analogously for the other packs). The re-run above
reproduces the forward columns within ~±4% and the fused/backward columns within +2-14%:
the fused per-call wall clock carries the cached-launcher host optimization, whose
snapshot replay jitters run-to-run (the published fused-forward is even non-monotonic in
pack size, `37.8 -> 35.1 us`), and eager backward drifts with GPU boost-clock state. The
CUDA-graph replay numbers below collapse that host jitter into a single replay.

*CUDA-graph replay of the same region, both implementations captured symmetrically as a
forward-only graph and a forward+backward graph* (median of 100 after 30 warmup, replay
timing via `benchmark/csa/bench_csa_compressor.py`; capturing each side into a graph
collapses its per-operation launches into a single replay, so this is the fairest
wall-clock basis for launch-bound shapes):

| THD pack | head_dim | eager fwd | fused fwd | fwd | eager total | fused total | total |
|---|---|---|---|---|---|---|---|
| 1 x 8192 | 128 | 119.2 us | 10.8 us | **11.0x** | 364.6 us | 22.8 us | **15.9x** |
| 3 x 8192 | 128 | 235.3 us | 14.8 us | **15.8x** | 668.4 us | 38.6 us | **17.3x** |
| 1 x 8192 | 512 | 269.8 us | 17.1 us | **15.7x** | 802.5 us | 40.0 us | **20.0x** |
| 3 x 8192 | 512 | 673.1 us | 39.2 us | **17.1x** | 2057.6 us | 107.8 us | **19.0x** |

`eager total` / `fused total` capture forward + backward together. The eager total graph
captures the autograd backward of the captured forward against stable, pre-allocated zero
`.grad` buffers (the captured region zeros them in place, then runs forward + backward, so
every replay accumulates into a zeroed buffer — numerically identical to a single fresh
backward, verified per shape in the harness); the fused total graph captures the forward
wrapper immediately followed by the backward wrapper. The backward replay alone is not a
separately captured quantity — it is approximately `total - fwd` and is reported only as
that reference, never as a measured column. Graph speedups are `eager / fused` of the
displayed µs, truncated to one decimal.

Capturing each side into a graph narrows the absolute eager-vs-fused forward gap most on
the smallest, launch-bound shape (about 2.8x at `1 x 8192 / 128` forward) and barely on
the largest, less launch-bound shape (1.2x at `3 x 8192 / 512`); the speedup ratio is
similar to or larger than per-call because a graph replay collapses each side's
per-operation launches into one.

Environment: driver 590.48.01, PyTorch 2.13.0 (CUDA 13.3), `nvidia-cutlass-dsl` 4.6.1.
Measurement basis: identical inputs over exactly the replaced region for both
implementations; per-call eager backward = the torch autograd backward with the forward
outside the timed region; graph eager backward = the autograd backward of the captured
forward, captured together with it against stable zero `.grad` buffers; fused backward =
the backward wrapper (kernel + the fp32 `dAPE` zero-fill + host validation, no autograd
engine). Both wall-clock tables above (per-call and graph) are from a single run of
`benchmark/csa/bench_csa_compressor.py`; the kernel-time table is from nsys.

An ncu hardware-ceiling audit of the **ported kernels (prior to the two optimization
commits)** — cache-flushed, `--set full`, all four benchmark shapes — showed measured
DRAM read volume matching the algorithmically necessary bytes within 1% (the THD gather
adds no over-read; stores fully coalesced at 32/32 bytes per sector, loads 29-30/32),
with neither L2 (<27% of peak) nor DRAM (<33%) close to saturation at these
microsecond-scale sizes: the gap to a pure DRAM-floor time was a mix of sub-wave grid
width / occupancy, memory latency, and (at the largest shape) issue pressure — not
wasted traffic. The two optimizations that audit identified are folded in here: 32-bit
vectorized forward accesses and backward kernel-side zero-writes replacing the two bf16
grad-buffer fills. They do not change the bytes the kernels must read, so the
traffic-optimality conclusion carries over; the utilization percentages above predate
them.

## ratio=128

`ratio = 128` is served by dedicated kernels (`compressor_sm100_r128.py`): the generic
kernels keep the whole `coff * ratio` window in per-thread registers, which spills
kilobytes of local memory per thread at `ratio = 128` (255-register cap). The dedicated
forward streams the window with a chunked softmax (one CTA per output row, the window
split over `threadIdx.y` chunk-rows, per-column `(max, denom, acc)` triples, one
fixed-order smem merge); its launch schedule is **bucketed by the output-row count**
(small / default / large, all precompiled for CUDA-graph safety) and per bucket selects
the online-rescale or two-phase accumulation form and, where it measured faster, an
`ex2.approx`-based fast exp. The dedicated backward stages each row's window into
shared memory chunk-parallel, computes `exp` and per-chunk partial `den`/`S` sums in
the same pass, merges the partials per column in a fixed chunk order, and stores the
gradients with a hoisted `1/den` multiply — plus kernel-side zero-writes to
never-consumed slots and fp32-atomic `dAPE`, exactly as at ratio=4. The backward
defaults to the same fast exp outside the d=128 small-pack (vec=1) buckets.

**The ratio=128 numerics contract is the deterministic tolerance contract described in
[Numerics](#numerics), NOT the ratio=4 bitwise-`dKV`/`dScore` contract.** The
reduction orders (forward chunk merge, backward `den`/`S` partial merge), the
backward's hoisted reciprocal, and the fast-exp buckets all differ from the eager op
order by design, each adopted on a measured same-GPU win and gated on tolerance +
fp64-oracle parity (worst observed deviations in [Numerics](#numerics)). Everything
stays bitwise run-to-run: fixed chunk boundaries and merge orders, no atomics in
forward/`dKV`/`dScore`.

Both kernels are register-flat in the window length (ptxas sm_100a: forward 32-51
registers, backward 48-128 registers, 0 spill / 0 stack across every shipped
(config, schedule) kernel — 16 kernels total; reproduce the per-kernel table with
`benchmark/csa/reg_probe_csa_compressor_r128.py`) and JIT in ~0.4-1.0 s per
configuration.
The backward picks its `rows_per_cta` at launch time (a runtime argument — no
recompile) so the grid fits one resident wave; the static row capacity fixes it under
CUDA-graph capture. Small packs (`nb_total <= 192`, d=128) switch the backward to a
vec=1 schedule bucket (also precompiled) for grid-fill.

Measured on 1x B200 (CC 10.0, driver 590.48.01), torch 2.13.0 / CUDA 13.3 /
`nvidia-cutlass-dsl` 4.6.1; eager baseline = the fp32-intermediate reference region of
the test suite on identical inputs (the numerics-contract reference; the upstream eager
region differs only by a bf16 weight-rounding cast at ~equal cost). Packs are
single-sequence THD unless noted.

*Isolated GPU kernel time* (nsys, sum of kernel durations inside a
`cudaProfilerApi`-gated 100-iteration unsynced loop / 100; forward and backward each
measured in their own loop; the fwd+bwd total is one combined loop per training step —
forward + backward + the ~1 us fp32 `dAPE` zero-fill — and is not the sum of the
standalone columns, because the interleaved backward changes L2 residency for the
forward at long context):

| config | tokens | eager fwd | fused fwd | fwd | eager bwd | fused bwd | bwd | fused fwd+bwd total |
|---|---|---|---|---|---|---|---|---|
| coff 1, d 128 | 8192 | 84.7 us | 5.7 us | **15.0x** | 127.3 us | 8.4 us | **15.2x** | 15.1 us |
| coff 1, d 128 | 131072 | 451.8 us | 14.9 us | **30.2x** | 471.9 us | 63.7 us | **7.4x** | 82.5 us |
| coff 2, d 128 | 8192 | 187.1 us | 6.4 us | **29.1x** | 246.2 us | 11.2 us | **21.9x** | 18.9 us |
| coff 2, d 128 | 65536 | 726.3 us | 16.4 us | **44.4x** | 960.0 us | 61.2 us | **15.7x** | 76.7 us |
| coff 1, d 512 | 65536 | 641.1 us | 48.4 us | **13.2x** | 795.0 us | 104.2 us | **7.6x** | 150.0 us |
| coff 2, d 512 | 65536 | 2315.2 us | 78.8 us | **29.4x** | 3261.9 us | 205.8 us | **15.9x** | 287.7 us |
| coff 2, d 512 | 131072 | 4526.5 us | 117.6 us | **38.5x** | 6304.3 us | 457.6 us | **13.8x** | 580.7 us |

*End-to-end wall clock per call* (CUDA events, median of 100 after 30 warmup; fused
backward includes a `grad_ape.zero_()` per call; not comparable to the kernel-time
numbers above — at small packs the ~9 us python launch path dominates the 5.7-11 us
kernels):

| config | tokens | fused fwd | fused bwd |
|---|---|---|---|
| coff 1, d 128 | 8192 | 14.1 us | 24.3 us |
| coff 1, d 128 | 131072 | 24.4 us | 82.1 us |
| coff 2, d 128 | 8192 | 15.2 us | 26.3 us |
| coff 2, d 128 | 65536 | 25.5 us | 75.6 us |
| coff 1, d 512 | 65536 | 56.3 us | 118.9 us |
| coff 2, d 512 | 65536 | 88.1 us | 220.8 us |

The ratio=128 numerics contract (21-case gate: 15 union + 1 scaled-input + 4 padding
cases, each running the determinism, tolerance and fp64-parity gates, plus 1
overflow-intermediate case that gates bit-stable NaN replay and NaN-pattern equality
vs the eager fp32 reference with the fp64 comparator explicitly skipped, with shipped
schedule coverage asserted) and the per-kernel ptxas table are reproducible from the
committed scripts `benchmark/csa/gate_csa_compressor_r128.py` and
`benchmark/csa/reg_probe_csa_compressor_r128.py`.

## Testing

```bash
(cd test/python && pytest fe_api/csa/test_CSA_compressor.py)
```

The tests validate numerics against an fp32-intermediate eager reference (bitwise
`dKV`/`dScore` at ratio=4; the deterministic tolerance contract at ratio=128, with
fp64-oracle parity on finite-intermediate inputs), the upstream eager numerics, plus ragged packs, static-capacity
padding, kernel-side zero-writes into uninitialized gradient buffers (NaN-canary with
exact-zero assertions on every never-consumed slot class, and the `total_comp == 0`
zeros fallback), run-to-run determinism, `grad_ape` zeroing ownership, the ratio=128
dispatch envelope (schedule selection at every `nb_total` bucket boundary at L0, and
one execution of every shipped (config, schedule) kernel against the full contract —
tolerance, fp64-oracle parity, determinism — at L1; run
`pytest -m "L0 or L1"`), CUDA-graph capture/replay, and `check_support` boundaries.
