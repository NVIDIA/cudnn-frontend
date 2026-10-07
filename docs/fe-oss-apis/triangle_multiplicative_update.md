# Triangle Multiplicative Update (SM107)

**This is an experimental API and subject to change.**

## Overview

`cudnn.triangle_multiplicative_update` is the AlphaFold-class triangle multiplicative update ("outgoing" / "incoming"
triangle multiplication, AlphaFold 3 Algorithms 12 and 13) as three FROST CuTe-DSL launches on one stream, behind the
signature of `cuequivariance_ops_torch.triangle_multiplicative_update` (pinned to cuequivariance-ops-torch **0.12.0**:
the same parameter names, order and defaults, so a model switches providers by changing one import).

```text
x [B, N, N, c_z] bf16   (the pair representation; the caller adds the residual to the result)
 |
 |  K1  LayerNorm_in over the channel, the gated dual projections (value || gate) for a and b, sigmoid * mul, the pair mask
 |      -> a, b   channel-major planes [c, B*N, n_pad] bf16                      LayerNorm-prologue dual GEMM (FROST)
 |  K2  the per-channel contraction   y[c][i, j] = sum_k a[c][i, k] b[c][j, k]   (outgoing)   /   sum_k a[c][k, i] b[c][k, j]   (incoming)
 |      -> y      channel-major [c, B*N, n_pad] bf16                             ONE batched GEMM (batch = c * B) on the FROST GEMM engine
 |  K3  LayerNorm_out over the channel of y, LayerNorm_in of x recomputed, output projection * sigmoid(output gate)
 v      -> out    pair-major [B, N, N, c_z] bf16
```

Every stage is a FROST kernel; there are no cuBLAS or cuDNN-backend call-outs. K1 and K3 are the two arms of one
LayerNorm-prologue dual-GEMM body (`cudnn.gemm.frost.ln_prologue`), K2 is the shipped FROST GEMM template compiled for the
contraction's tile (chosen per N from a measured prototype). The intermediate planes and `y` live in one workspace the
caller provides (class API) or the function allocates (drop-in).

**Target: NVIDIA Rubin (SM107, compute capability 10.7) only.** Every other architecture is declined with a typed error
(`NotImplementedError`), never served slowly.

## Mathematics and numerics

For every pair `(i, j)` of a batch entry, with `c_z` the pair width, `c` the hidden width, `m` the pair mask (`None` = all valid):

```text
z_in = LayerNorm_{c_z}(x; norm_in_weight, norm_in_bias, eps)
a    = sigmoid(z_in . g_in_weight[:c]^T + g_in_bias[:c]) * (z_in . p_in_weight[:c]^T + p_in_bias[:c]) * m
b    = sigmoid(z_in . g_in_weight[c:]^T + g_in_bias[c:]) * (z_in . p_in_weight[c:]^T + p_in_bias[c:]) * m
y    = einsum("bikc,bjkc->bijc", a, b)   (outgoing)      /      einsum("bkic,bkjc->bijc", a, b)   (incoming)
u    = LayerNorm_c(y; norm_out_weight, norm_out_bias, eps)
out  = sigmoid(z_in . g_out_weight^T + g_out_bias) * (u . p_out_weight^T + p_out_bias)
```

Parameter conventions are cuEquivariance's: `p_in_weight`, `g_in_weight` are `[2 c, c_z]` with rows `[a; b]`; `p_out_weight`
is `[c_z, c]`; `g_out_weight` is `[c_z, c_z]`; the LayerNorm vectors are `[c_z]` (in) and `[c]` (out); every bias is optional
(`None` = zero). The output is not masked (the mask acts on `a` and `b`), and the residual add is the caller's.

Rounding points (all arithmetic fp32 with bf16 storage): LayerNorm statistics in fp32 (two-pass), the LayerNormed operand rounded
once to bf16 as the MMA operand; `a` and `b` rounded once to bf16 **after** the mask multiplies in fp32 (a zero mask row is exactly
`+0.0`; cuEquivariance multiplies a bf16-rounded product -- identical for `{0, 1}` and power-of-two masks, one bf16 ulp apart on
fractional weights); `y` accumulated in fp32 and rounded once to bf16; the gate sigmoid through `tanh.approx`; the output rounded
once to bf16. `cudnn.triangle_multiplicative_update.triangle_multiplicative_update_reference` is the fp64 reference with these
rounding points inserted (`rounding="kernel"`) or without them (`rounding="none"`, the pure formula). Measured on the test grid
(N in 64..768, both directions, masks all / padded / random, biases on and off): relative RMS to the kernel-rounding reference
1.68e-3 (= the output's bf16 rounding alone); per cell `|diff| <= 4e-3 + |ref| / 64` on all but 0.7..7.6 per million cells, every
cell within `2e-2 + |ref| / 64` (the tail of bf16 rounding flips propagating through the contraction); relative RMS to the
unrounded fp64 formula below 2e-2.

## Served today, and the typed declines

| what | served | otherwise |
|---|---|---|
| architecture | compute capability 10.7 (Rubin) | `NotImplementedError` |
| dtype | bf16 pair representation, bf16 parameters (`precision='bfloat16'` casts wider inputs once per call, as cuEquivariance does) | `NotImplementedError` |
| pair width `c_z` / hidden width `c` | `c_z == c` in `{128, 256}` (`c_z % 32 != 0` is a `ValueError`: cuEquivariance's own rule); 384 is withdrawn until its renderings stop spilling and carry an oracle cell. 256 is served correctly but not yet competitively (12-42 % behind cuEquivariance's generic route: the measured-performance section) | `NotImplementedError` |
| N | `N >= 33`, any N for the outgoing direction; the incoming direction needs `N % 8 == 0` | `NotImplementedError` |
| direction | `"outgoing"`, `"incoming"` | `ValueError` |
| mask | `None`, or `[B, N, N]` (`[N, N]` for B = 1) of any real / bool dtype (cast once per call to bf16 when not already bf16 contiguous) | `ValueError` on a shape mismatch |
| `precision` | `None` / `'default'` / `'none'` / `'bfloat16'` (`'bf16'`) | `'mxfp8'`, `'ieee'`, `'tf32'`: `NotImplementedError` |
| parameters | the eight weights are required (no implicit initialisation); biases optional | `ValueError` |
| layout | contiguous `x` (and `out`) | `ValueError` (never a hidden copy) |
| tensor extent | `B*N*N*c_z` and `2*c*B*N*n_pad` below 2^31 (32-bit element offsets in the kernels) | `NotImplementedError` |
| device | every `execute` tensor on the device the plan was declared on | `ValueError` |
| performance knobs | `k2_tile` / `knobs=TriMulKnobs(...)` within the measured domains of `cudnn.triangle_multiplicative_update.heuristics` (contraction tile, projection tile order, output-stage prologue form, scheduler ring depth) -- every value computes the same function | `ValueError` on an unknown name; `NotImplementedError` when the contraction tile cannot serve the geometry on this device |
| autograd | not registered yet (forward only); under `torch.no_grad()` or on detached tensors the call is served | `NotImplementedError` when grad mode is on and an input or parameter `requires_grad` (a tensor without `grad_fn` would silently drop the gradient) |

The `c != c_z` geometries (Protenix-v2 style), `c_z` outside the two widths, N below 33 and the incoming direction at `N % 8 != 0`
are the current kernel envelope, not the operation's; each is a typed decline naming the rule.

## Drop-in function

```python
import cudnn
from cudnn import triangle_multiplicative_update   # or: from cuequivariance_torch import triangle_multiplicative_update

out = triangle_multiplicative_update(
    x, "outgoing", mask,
    norm_in_weight, norm_in_bias, p_in_weight, p_in_bias, g_in_weight, g_in_bias,
    norm_out_weight, norm_out_bias, p_out_weight, p_out_bias, g_out_weight, g_out_bias,
    eps=1e-5, precision=None,
)
z = z + out                                        # the residual is the caller's, as with cuEquivariance
```

Signature (cuEquivariance 0.12.0, verbatim; the two keyword-only additions are appended):

```python
triangle_multiplicative_update(x, direction="outgoing", mask=None,
                               norm_in_weight=None, norm_in_bias=None, p_in_weight=None, p_in_bias=None,
                               g_in_weight=None, g_in_bias=None, norm_out_weight=None, norm_out_bias=None,
                               p_out_weight=None, p_out_bias=None, g_out_weight=None, g_out_bias=None,
                               eps=1e-5, precision=None, *, out=None, stream=None) -> torch.Tensor
```

`out` writes into a caller-provided bf16 tensor (stable addresses for CUDA-graph users); `stream` is a raw CUDA stream handle the
allocations and launches are ordered on (`None` = torch's current stream). The function caches one compiled plan per
`(shape, direction, mask present, biases present, eps)` and one prepared parameter set per parameter identity (address, shape,
strides, dtype, device, in-place version counter; a bounded LRU), allocates the output and the workspace per call, and launches.
Warm a shape once before capturing a `torch.cuda.graph` around the call; the captured graph replays the three kernels and follows
the inputs' contents.

## Prepared class API

```python
from cudnn.triangle_multiplicative_update import TriangleMultiplicativeUpdateFwd, prepare_weights

weights = prepare_weights(norm_in_weight, norm_in_bias, p_in_weight, p_in_bias, g_in_weight, g_in_bias,
                          norm_out_weight, norm_out_bias, p_out_weight, p_out_bias, g_out_weight, g_out_bias)   # once per parameter set
plan = TriangleMultiplicativeUpdateFwd(x, "outgoing", c_hidden=128, has_mask=True, has_in_bias=True, has_out_bias=True)
plan.check_support()                       # typed declines
plan.compile()                             # K1, K2 (per-N tile), K3 -- cached per parameter set across instances
workspace = torch.empty(plan.get_workspace_size(), dtype=torch.uint8, device=x.device)
out = torch.empty_like(x)
plan.execute(x, mask, weights, out, workspace, current_stream=None)   # validates and launches; no allocation, no sync
graph = plan.make_cuda_graph(x, mask, weights, out, workspace)        # optional: one graph launch per call for these buffers
```

`prepare_weights` turns cuEquivariance's parameters into the kernels' layouts once (the value / gate rows interleaved per
128-channel block, the biases stacked; a bias given on one side only is paired with zeros here, never per call). The constructor
accepts a tensor or a `cudnn.api_base.TensorDesc`; the declaration's flags are compile-time specialisations the `execute`
arguments must match exactly (a mask passed to a plan compiled without one raises, and vice versa). `get_workspace_size()` is the
two channel-major planes plus `y` (bf16, the inner axis padded to a multiple of 8, 256-byte aligned) plus a small readable slack the
outgoing contraction over-reads at `N % 8 != 0`. The performance knobs come from `cudnn.triangle_multiplicative_update.heuristics`:
`recommend(N, c, B, direction)` returns the measured default `TriMulKnobs` -- a bucket table keyed by (hidden width, N bucket) carries the
complete knob set, and a short refinement table overrides one axis per (batch bucket, direction) where a measured cell beat the bucket's pick
decisively in every pass that measured it. The bucket picks: the contraction tile (the 1-CTA 128x128 tile at N <= 128 and at 384, the 2-CTA
pair tile at 129-256 and 768, with the A operand multicast across 2 CTAs at 512 and across 4 from 1024), the projection stage's tile order
(`interleaved` at c = 128 below N = 512, `major` elsewhere), the output stage's prologue form (`transpose` at c = 128, `mn_major` at c = 256)
and the scheduler ring depth (1 from N >= 384 at c = 128 and from N >= 129 at c = 256, else 2). The refinements: at c = 128 the ring is 1 deep
at N <= 128 for the incoming direction with B = 1, 1 deep with the pass-major projection order at 256 <= N < 384 from B >= 4, and 4 deep at
384 <= N < 512 with B = 1; at c = 256 the ring is 1 deep at N <= 128 from B >= 4. Batch sizes
other than 1 and 4 take the nearest measured batch bucket below (B >= 4 -> the B = 4 picks, else the B = 1 picks), an N between two buckets
the first bucket at or below it. Every pick is the best of a three-pass measured sweep on one GPU of a 212-SM Rubin perf node (both directions,
B = 1 and 4, a control pair per cell; the second and third passes re-measured every alternative against the table they shipped). The
constructor's `k2_tile=` (one axis) or `knobs=` (a `TriMulKnobs` whose set axes override the pick; an axis left `None` keeps the table's value)
override it, `check_support` validates the vocabulary and probes the contraction tile's eligibility for the geometry without compiling. Every
value of every domain computes the same function (bitwise; the tests pin it).

## Traffic model and measured performance

Compulsory HBM traffic per call: `18 N^2 C` bytes per batch entry (K1 reads `x` 2 B per pair-channel and writes `a`, `b` 4 B; K2
reads 4 B and writes `y` 2 B; K3 reads `y` and `x` 4 B and writes 2 B) plus the mask (2 B per pair) --
`TriangleMultiplicativeUpdateFwd.traffic_bytes()`. The figures below are the shipped heuristics (`recommend`'s pick per cell, no override)
measured as the base of the final sweep pass on ONE GPU of a Rubin node (cc 10.7, 212 SMs, SM clock locked at 2376 MHz and sampled per
cell) against that node's HBM pin of a read + write copy kernel (10 160 GB/s); the competitor is cuEquivariance 0.12.0's generic route (its
only route on this architecture) measured on the SAME GPU within the hour with the same protocol (graph replay vs graph replay; 600-iteration
competitor windows so the clock sampler sees the small cells). Speed-ups are `t_cuEq / t_FROST - 1` (positive = FROST faster); the all-valid
mask, biases present, `c = c_z`. `C = 128`, `B = 1`:

| N | direction | FROST, graph replay (us) | K1 / K2 / K3 alone (us) | GB/s (% of the 10160 GB/s pin) | TFLOPS | cuEquivariance 0.12.0, same GPU (us) | speed-up vs cuEq | host floor per eager call (us) | clock |
|---|---|---|---|---|---|---|---|---|---|
| 128 | outgoing | 29.6 | 29.0 / 24.9 / 32.2 | 1276 (12.6 %) | 127 | 36.2 | +22.2 % | 86.4 | 2364 MHz, CLEAN |
| 128 | incoming | 29.8 | 29.2 / 24.9 / 31.4 | 1268 (12.5 %) | 126 | 39.6 | +33.0 % | 87.1 | 2364.0 MHz, CLEAN |
| 217 | outgoing | 53.8 | 29.3 / 26.0 / 32.3 | 2019 (19.9 %) | 221 | 80.5 | +49.7 % | 89.5 | 2364 MHz, CLEAN |
| 256 | outgoing | 62.9 | 31.7 / 24.5 / 30.6 | 2401 (23.6 %) | 273 | 67.3 | +7.0 % | 86.8 | 2352.0 MHz, CLOCK-DEGRADED |
| 256 | incoming | 63.1 | 31.7 / 24.4 / 31.3 | 2396 (23.6 %) | 272 | 67.6 | +7.2 % | 88.8 | 2352 MHz, CLOCK-DEGRADED |
| 384 | outgoing | 145.5 | 76.8 / 23.8 / 59.6 | 2337 (23.0 %) | 299 | 115.7 | -20.5 % | 88.7 | 2364.0 MHz, CLEAN |
| 384 | incoming | 145.8 | 76.8 / 23.5 / 58.9 | 2333 (23.0 %) | 298 | 115.3 | -20.9 % | 90.7 | 2364 MHz, CLEAN |
| 512 | outgoing | 187.6 | 99.4 / 28.8 / 71.7 | 3222 (31.7 %) | 458 | 197.2 | +5.1 % | 91.3 | 2352.0 MHz, CLOCK-DEGRADED |
| 512 | incoming | 188.1 | 99.2 / 28.9 / 71.8 | 3214 (31.6 %) | 457 | 197.8 | +5.2 % | 90.1 | 2364 MHz, CLEAN |
| 768 | outgoing | 400.6 | 209.7 / 56.9 / 144.3 | 3395 (33.4 %) | 579 | 424.0 | +5.8 % | 90.4 | 2364 MHz, CLEAN |
| 768 | incoming | 401.3 | 209.7 / 56.6 / 144.3 | 3390 (33.4 %) | 578 | 424.3 | +5.7 % | 91.1 | 2364 MHz, CLEAN |
| 1024 | outgoing | 695.4 | 364.9 / 94.8 / 245.8 | 3477 (34.2 %) | 692 | 743.5 | +6.9 % | 90.7 | 2364.0 MHz, CLEAN |
| 1024 | incoming | 695.5 | 364.6 / 94.5 / 245.7 | 3477 (34.2 %) | 692 | 741.8 | +6.7 % | 92.1 | 2364 MHz, CLEAN |
| 2048 | outgoing | 2920.2 | 1427.8 / 578.2 / 1007.0 | 3312 (32.6 %) | 1035 | 3108.1 | +6.4 % | 94.2 | 2364.0 MHz, CLEAN |
| 2048 | incoming | 2922.5 | 1428.2 / 578.3 / 964.6 | 3310 (32.6 %) | 1035 | 3111.3 | +6.5 % | 94.7 | 2364 MHz, CLEAN |

Per group of the 52-cell grid (N in {128, 217, 256, 384, 512, 768, 1024, 2048}, both directions where `N % 8 == 0`; the batch-4 geometries
above `2 c B N n_pad >= 2^31` are declined):

- `C = 128, B = 1` (15 cells): 12.5-34.2 % of the pin; vs cuEquivariance +22 / +33 % at N = 128, +50 % at 217, +7 % at 256, +5.1-6.9 % from
  512 to 2048, and **-20.5 / -20.9 % at N = 384** (the projection kernels' 256-row tiles waste a quarter of their rows at `N % 256 == 128`).
- `C = 128, B = 4` (13 cells, N <= 1024): 24.0-35.3 % of the pin; vs cuEquivariance +7.5 / +7.9 % at 128, +38 % at 217, +5.2-7.2 % at 256 and
  from 512 to 1024, -19.7 % at 384.
- `C = 256, B = 1` (13 cells, N <= 1024): 7.4-13.8 % of the pin; **slower than cuEquivariance by 18-42 %** at N >= 217 (-31 % at 217, -38 % at
  256, -42 % at 384, -22 % at 512, -19.5 % at 768, -18 % at 1024); +106 % at N = 128, where cuEquivariance's own route takes 207 us.
- `C = 256, B = 4` (11 cells, N <= 768): 10.4-13.9 % of the pin; slower than cuEquivariance by 12-39 % (-37 % at 128, -12 % at 217, -22 % at
  256, -39 % at 384, -19 % at 512, -18 % at 768).

At `C = 128` the two LayerNorm-prologue GEMMs (K1 + K3) are 80-88 % of every call and the contraction (K2) 10-20 %; the call reaches
24-35 % of the HBM pin on the 18 N^2 C model from N >= 256. At `C = 256` the K = 256 renderings of K1 and K3 are register-bound (the
LayerNorm row held in registers between the two statistics passes spills 240 / 499 words per lane at a flat 168 registers; K1 alone runs at
5.8 % of the pin, 17x its byte bound), so the call sits at 7-14 % of the pin and behind cuEquivariance's generic route -- the width is served
correctly (the oracle cells pass) but is not a performance claim until the LayerNorm pass stops holding the row in registers. The heuristics
table is worth +1.9..+8.3 % (`C = 128, B = 1`), +4.4..+9.0 % (`C = 128, B = 4`), +0.1..+1.3 % (`C = 256, B = 1`) and +1.0..+3.0 % (`C = 256,
B = 4`) over the pre-sweep defaults (the prototype's per-N contraction tile, the per-width projection order, the MN-major output prologue, a
2-deep scheduler ring), measured in the same sweep. Absolute times of a pass depend on what else the node runs: the final pass ran while two
other jobs occupied neighbouring GPUs of the node and every kernel launch read 0.5-1.5 us slower than in the quiet pass before it (the
28 us cells 4.6-5.1 % slower, the 700 us cells 0.2 %, the 3 ms cells 0.04 %; clocks and power identical) -- the per-cell margins between
arms are unaffected, the small-N absolute figures above are the pessimistic ones. The N = 256 and 512 outgoing cells and the N = 2048 cells
of the competitor ran at a 2352 MHz busy clock (one step under the lock, -1.0 %) and are marked CLOCK-DEGRADED by the protocol. cuEquivariance's
eager host floor on this GPU is 200-245 us per call; the FROST class API's is 86-95 us -- at N <= 256 an eager call is host-bound (device
30-63 us), which the captured CUDA graph removes.

Byte count (Nsight Compute `dram__bytes.sum` over the three kernels, warm L2 between launches, measured on the pre-heuristics defaults):
2414 MB against 2418 MB modelled at N = 1024 (ratio 0.998: K1 794 MB = 0.98x its model with `x` read once, K2 822 MB = 1.02x, K3 798 MB =
0.99x); at N = 512 the warm L2 keeps part of the planes between kernels and the chain reads 0.93x the model. With the pass-major K1 tile order
the N = 1024 chain read 1.11x the model (K1 re-read `x` from HBM for its second projection pass); the heuristics nevertheless pick the pass-major
order there together with the 1-deep scheduler ring (+1.4 % on the whole call in the sweep): the kernel runs at 20 % of the HBM pin, so the
extra bytes cost less than the ring-depth interaction gains, and the interleaved order stays the pick at `c = 128` below N = 512.

Launch glue: every kernel of the chain carries programmatic dependent launch (PDL) -- it is rendered with `USE_PDL = True`
(`griddepcontrol wait` before its first global read, `launch_dependents` once its mainloop no longer needs the predecessor) and launched
with the CuTe-DSL kernel launch's `use_pdl` attribute, so K2's prologue overlaps K1's tail and K3's overlaps K2's. Measured with the
pre-heuristics defaults on another GPU of the same node class: against the same three kernels rendered without PDL, the eager chain is
+0.2..+2.5 % faster with it at every N except 384 (-1.2 %, where the eager PDL chain pays more than it overlaps; the graph replay of the PDL-on
chain is still the fastest form there); the captured CUDA graph keeps the PDL edges and adds +1.1..+3.5 % over the eager PDL-on chain at
N <= 384, +0.3..+0.5 % at 512 and nothing above (vs the PDL-off eager chain: +3.9 % at 256 down to +0.3 % at 2048). An earlier revision of this
page stated that programmatic dependent launch was "not an arm of this API" because the launch path carried no such attribute; that was
wrong -- the chain had PDL on all along, and the PDL-off column was the baseline it was missing.

## Tests

`test/python/triangle_multiplicative_update/cutedsl/test_api.py` (the lazy export in every import order, the signature pin, the
typed declines, the padded contraction graph, the workspace and traffic models, the fp64 reference against the formula; on Rubin: the
end-to-end oracle, the API bitwise equal to the hand-chained kernels, the wrapper bitwise equal to the class API with a plan-cache
hit, `execute` allocating nothing and never synchronising, CUDA-graph capture and replay, bool / fp32 masks and batchless inputs,
`precision='bfloat16'`, the live-device declines, the stock-caller invocation shape hitting both caches, a plan declared on a
non-current device, the autograd decline). The three kernels have their own modules beside it.
