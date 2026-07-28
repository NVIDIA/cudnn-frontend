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
pinned in PTX so results do not depend on compiler FMA contraction. Against an
fp32-intermediate eager reference (same op order, fp32 throughout), `dKV`/`dScore` are
**bit-identical** and the forward matches within one bf16 rounding step on a tiny
fraction of elements. Forward, `dKV` and `dScore` are bitwise run-to-run deterministic.
`dAPE` is reduced with one fp32 atomic per `(k, dim)` per CTA and is **not** bitwise
run-to-run deterministic; the backward APIs raise under
`torch.use_deterministic_algorithms(True)`.

### Support surface (`check_support`)

- Compute capability **10.0** (the only validated architecture so far; the kernels use
  no arch-specific features, wider enablement is possible after validation)
- `ratio == 4`, `coff in {1, 2}` (`coff == 2` is the production CSA/HCA configuration,
  `coff == 1` the own-block window form; the kernels are generic over `ratio` and
  `head_dim` too and the ratio gate can be lifted once validated)
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
**zero-initialized** (fp32 atomic accumulation).

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

## Testing

```bash
(cd test/python && pytest fe_api/csa/test_CSA_compressor.py)
```

The tests validate numerics against an fp32-intermediate eager reference (bitwise
`dKV`/`dScore`), the upstream eager numerics, and an fp64 oracle, plus ragged packs,
static-capacity padding, kernel-side zero-writes into uninitialized gradient buffers
(NaN-canary, including the `total_comp == 0` zeros fallback), run-to-run determinism,
CUDA-graph capture/replay, and `check_support` boundaries.
