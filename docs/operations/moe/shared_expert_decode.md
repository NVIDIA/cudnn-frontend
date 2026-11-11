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

Launch 1 (`gate_up`, `K_SPLIT = 8` CTAs per cluster along K; 256 CTAs of 256 threads at either geometry, one wave on every
SM100-class part: two row-pair groups per CTA at `Geometry(8192, 256)` -- 32 KB of `W_gate_up` each, two CTAs per SM -- and
one at `Geometry(8192, 128)` -- 16 KB, four per SM at up to 16 tokens) streams its slice of `W_gate_up` once by
`cp.async.bulk` (one copy per row) with the row pairs `(j, I + j)` resident in one CTA, so the SiLU-mul sits in its
epilogue; the `T` live rows of `x`'s K slice land by one bulk copy each (resident, no ring, at every `T`); the eight K-slice
CTAs of a cluster fold their partial sums through DSMEM (`st.async`) in a fixed rank order; a warp per row-pair group
computes its slice of the gate logit `x @ W_gate^T`, folded on the cluster leader the same way. Launch 2 (`down`, `H / 64`
CTAs) prefetches its `W_down` rows ahead of `griddepcontrol.wait`, folds the per-group gate partials into `sigmoid` and
writes `y`. The math path is `mma.sync.m16n8k16` bf16 -> fp32 in both launches with the tokens on M (zero-padded to 16 or
32 rows), so every output element is accumulated in a FIXED order that depends on the geometry alone: the result is
bitwise repeatable and batch-invariant (a token's bytes alone equal its bytes inside a batch of up to 24; see
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
(`act` [32, I] bf16, `gate_part` [32, I / 4] fp32), allocated once per form at the first compile; build
one entry per concurrently running stream.

Knobs (`cudnn.moe.frost.kernel.shared_expert_decode_config.Cfg`, passed as `cfg=`): bit-neutral `N_GROUPS` (row-pair
groups per CTA; 0 = the one-wave rule), `X_BULK_ROWS` (x resident by bulk row copies; off = the first release's `cp.async`
ring), `X_STAGES`, `W_PHASES`, `ROWS_PER_CTA`, `MIN_CTAS_PER_SM`, `KRED_SPLIT` (the cluster-fold barrier split between the
accumulator ships and the gate leaves; an A/B arm); rounding-order axes `K_SPLIT` (pinned per geometry -- a function of the
geometry alone, never of the part: a part too small for the pinned form is declined by type) and `GATE_CLUSTER_FOLD` (the
gate tree's shape). A config the entry admits serves the whole 1 .. 24 band: both token-tile forms are validated at
construction, so a decline is raised there (`NotImplementedError` for a part fact, `ValueError` for a knob combination),
never at a launch. `shared_expert_decode_config.v1_cfg(geometry)` is the first release's form
(K_SPLIT 2 / 4, 128 CTAs, the ring, the launch-2 gate fold), kept as an A/B arm with its own goldens.
Launch knobs `pdl` / `pdl_pair` / `pdl_release` set the programmatic-dependent-launch attributes of launch 1 / launch 2
and the early release; all default off. Under `pdl=True` the launch's immediate stream predecessor must not write
`w_gate_up` / `w_gate` (they load before the kernel's `griddepcontrol.wait`); the serving chain's static weights
satisfy it. `pdl_release=True` creates an obligation for the consumer, not for the pair: a kernel launched after
launch 2 with the programmatic-stream-serialization attribute (e.g. a fused add of `y` into the routed output) may
start while launch 2 runs and must issue its own `griddepcontrol.wait` before it reads `y`; a consumer launched
without the attribute (the stock add) is unaffected. Measured on a GB300 (cc 10.3), the early release is a loss in every
form and stays off.

## Support matrix

| question | answer |
| --- | --- |
| device | compute capability 10.x (validated on cc 10.0, cc 10.3 and cc 10.7); 12.x declined (no launch-1 form fits a 99 KB per-CTA opt-in) |
| geometry | `Geometry(8192, 256)` and `Geometry(8192, 128)`; others declined by name |
| tokens | 1 .. 24 per call (runtime argument); 25 and up declined by name |
| dtypes | bf16 in / out; fp32 accumulation; `act` rounded once to bf16 |
| layouts | `x` / `out` rows contiguous with a 16-byte-multiple row stride; contiguous weights; 16-byte-aligned operands |
| determinism | bitwise replay and batch invariance within a call's token band (the [status page](/operations/determinism)) |
| DSL | nvidia-cutlass-dsl >= 4.8.0 |

Tests: `test/python/moe/frost/test_shared_expert_decode.py` (host declines, config pins, the device cells) and
`test_shared_expert_decode_determinism.py` (replay, batch invariance, goldens). Source: `python/cudnn/moe/frost/`.
