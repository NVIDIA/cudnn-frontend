# Shared-Expert Decode Pair (SM100-class)

**Experimental.** `cudnn.SharedExpertDecode` (also `cudnn.moe.frost.SharedExpertDecode`) is the bf16 shared
expert that sits next to a routed MoE, at decode-time token counts, as TWO CuTe DSL launches on one stream:

```text
h    = x @ W_gate_up^T                 W_gate_up [2 I, H] bf16 row-major: gate rows 0 .. I-1, up rows I .. 2I-1
act  = silu(h[:, :I]) * h[:, I:]       rounded once to bf16 between the launches
y    = (act @ W_down^T) * sigmoid(x @ W_gate^T)     W_down [H, I] bf16 row-major, W_gate [H] bf16; y [T, H] bf16
```

`x` is `[T, H]` bf16 with `T` in 1 .. 24 tokens (rows contiguous, any 16-byte-multiple row stride, so a view into a
taller slab serves), `y` is written into a caller-owned `[>= T, H]` bf16 buffer (or the entry's own, sliced to `T`).
The add into the routed output stays with the caller. Validated geometries: hidden 8192 with intermediate 256 (8-way
tensor parallelism of a 2048-wide shared expert) or 128 (16-way); compute capability 10.x; a CuTe DSL >= 4.8.0.
Everything else is declined by name (`TypeError` for a dtype, `ValueError` for a shape / stride / alignment / device,
`NotImplementedError` for the token count, the geometry, the architecture or the DSL version) before any DSL import.

Launch 1 (`gate_up`, `I / 4 x K_SPLIT` CTAs in clusters of `K_SPLIT` along K, 256 threads each) streams `W_gate_up`
once by `cp.async.bulk` with the row pairs `(j, I + j)` resident in one CTA, so the SiLU-mul sits in its epilogue;
`x` streams through a `cp.async` ring; the K-split clusters fold their partial sums through DSMEM (`st.async`); warp 1
of every CTA computes its slice of the gate logit `x @ W_gate^T`. Launch 2 (`down`, `H / 64` CTAs) prefetches its
`W_down` rows ahead of `griddepcontrol.wait`, folds the gate partials into `sigmoid` and writes `y`. The math path is
`mma.sync.m16n8k16` bf16 -> fp32 in both launches with the tokens on M (zero-padded to 16 or 32 rows), so every
output element is accumulated in a FIXED order that depends on the geometry alone: the result is bitwise
repeatable and batch-invariant (a token's bytes alone equal its bytes inside a batch of up to 24; see
[determinism](/operations/determinism)). Against an fp64 reference of the bf16 operands the output sits within the
usual bf16 band (rel-L2 <= 2e-2).

## Usage

```python
import torch
import cudnn

entry = cudnn.SharedExpertDecode(torch.device("cuda", 0))          # Geometry(8192, 256) by default
# entry = cudnn.SharedExpertDecode(dev, geometry=cudnn.moe.frost.Geometry(8192, 128))   # the 16-way shard
y = entry(x, w_gate_up, w_down, w_gate, out)                        # out: [>= T, 8192] bf16, caller-owned
```

The entry compiles once per (geometry, token-tile form: 1 .. 16 or 17 .. 24 tokens, x / out row strides) and
launches both kernels on torch's current stream; `T` is a runtime argument, so one compiled form serves every
token count of its band (a captured CUDA graph replays its own `T`). The entry owns the inter-launch scratch
(`act` [32, I] bf16, `gate_part` [32, K_SPLIT x I / 4] fp32), allocated once per form at the first compile; build
one entry per concurrently running stream.

Knobs (`cudnn.moe.frost.kernel.shared_expert_decode_config.Cfg`, passed as `cfg=`; every one bit-neutral except
`K_SPLIT`, a rounding-order axis pinned per geometry): `X_STAGES`, `W_PHASES`, `ROWS_PER_CTA`, `MIN_CTAS_PER_SM`.
Launch knobs `pdl` / `pdl_pair` / `pdl_release` set the programmatic-dependent-launch attributes of launch 1 / launch 2
and the early release; all default off. Under `pdl=True` the launch's immediate stream predecessor must not write
`w_gate_up` / `w_gate` (they load before the kernel's `griddepcontrol.wait`); the serving chain's static weights
satisfy it.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x (validated on cc 10.3; cc 10.0 / 10.7 run the same admission, unverified); 12.x declined (no launch-1 form fits a 99 KB per-CTA opt-in) |
| geometry | `Geometry(8192, 256)` and `Geometry(8192, 128)`; others declined by name |
| tokens | 1 .. 24 per call (runtime argument); 25 and up declined by name |
| dtypes | bf16 in / out; fp32 accumulation; `act` rounded once to bf16 |
| layouts | `x` / `out` rows contiguous with a 16-byte-multiple row stride; contiguous weights; 16-byte-aligned operands |
| determinism | bitwise replay and batch invariance within a call's token band (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0 |

Tests: `test/python/moe/frost/test_shared_expert_decode.py` (host declines, config pins, the device cells) and
`test_shared_expert_decode_determinism.py` (replay, batch invariance, goldens). Source: `python/cudnn/moe/frost/`.
