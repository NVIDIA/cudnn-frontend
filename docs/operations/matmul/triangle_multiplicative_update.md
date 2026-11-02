# Triangle Multiplicative Update (SM100 / SM103 / SM107)

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

**Target: NVIDIA Rubin (SM107, compute capability 10.7) and NVIDIA Blackwell (SM100 / SM103, compute capability 10.0 / 10.3).**
The three kernels are the same code on both: the LayerNorm-prologue dual-GEMM body is rendered once per architecture family at the
part's geometry (Rubin: a 9-deep operand ring in the 327 KiB carveout, 576 TMEM columns; Blackwell: a 6-deep ring -- 4 in the output
stage -- in the 227 KiB opt-in carveout, 512 TMEM columns, so `c_z = 256` fills the output stage's ring exactly), and the contraction's
FROST GEMM template sizes its pipeline from the live device. Every other architecture is declined with a typed error
(`NotImplementedError`), never served slowly. The measured performance below is Rubin's; the Blackwell column of the heuristics is its own
measured table (every knob swept on a GB200, 152 SMs at a 1830 MHz lock, by the same protocol; the `N < 256` buckets carry the Rubin picks
until measured there).

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
| architecture | compute capability 10.7 (Rubin), 10.0 / 10.3 (Blackwell SM100 / SM103) | `NotImplementedError` |
| dtype | bf16 pair representation, bf16 parameters (`precision='bfloat16'` casts wider inputs once per call, as cuEquivariance does) | `NotImplementedError` |
| pair width `c_z` / hidden width `c` | `c_z == c` in `{128, 256}` (`c_z % 32 != 0` is a `ValueError`: cuEquivariance's own rule); 384 is withdrawn until its renderings stop spilling and carry an oracle cell. 256 is served correctly but not yet competitively (13-44 % behind cuEquivariance's generic route at N >= 217: the measured-performance section) | `NotImplementedError` |
| N | `N >= 33`, any N for the outgoing direction; the incoming direction needs `N % 8 == 0` | `NotImplementedError` |
| direction | `"outgoing"`, `"incoming"` | `ValueError` |
| mask | `None`, or `[B, N, N]` (`[N, N]` for B = 1) of any real / bool dtype (cast once per call to bf16 when not already bf16 contiguous) | `ValueError` on a shape mismatch |
| `precision` | `None` / `'default'` / `'none'` / `'bfloat16'` (`'bf16'`) | `'mxfp8'`, `'ieee'`, `'tf32'`: `NotImplementedError` |
| parameters | the eight weights are required (no implicit initialisation); biases optional | `ValueError` |
| layout | contiguous `x` (and `out`) | `ValueError` (never a hidden copy) |
| aliasing | `out` and the workspace disjoint, as byte ranges, from every tensor the kernels read (`x`, the mask, the parameters) and from each other -- two views over one allocation whose ranges intersect are declined, pointer inequality is not enough (the output stage reads `x` while it writes `out`) | `ValueError` naming the two operands, before any launch |
| tensor extent | `B*N*N*c_z` and `2*c*B*N*n_pad` below 2^31 (32-bit element offsets in the kernels) | `NotImplementedError` |
| device | every `execute` tensor on the device the plan was declared on | `ValueError` |
| performance knobs | `k2_tile` / `knobs=TriMulKnobs(...)` within the measured domains of `cudnn.triangle_multiplicative_update.heuristics` (contraction tile, projection tile order, output-stage prologue form, scheduler ring depth, row tile, LayerNorm pass form, epilogue arithmetic form, the two stages' `.minnctapersm` register-split hint) -- every value computes the same function | `ValueError` on an unknown name; `NotImplementedError` when the contraction tile cannot serve the geometry on this device |
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

`out` writes into a caller-provided bf16 tensor (stable addresses for CUDA-graph users): `[B, N, N, c_z]`, or `[N, N, c_z]` at `B = 1`,
contiguous, its bytes disjoint from `x`, the mask, every parameter and the workspace (an intersecting byte range is a typed
`ValueError` before any launch); it is returned as given, whatever its rank -- only an output the function allocated follows `x`'s rank.
`stream` is a raw CUDA stream handle the
allocations and launches are ordered on (`None` = torch's current stream). The function caches one compiled plan per
`(shape, direction, mask present, biases present, eps)` and one prepared parameter set per parameter identity (address, shape,
strides, dtype, device, in-place version counter; a bounded LRU), allocates the output and the workspace per call, and launches.
Parameters created under `torch.inference_mode()` carry no version counter and bypass that cache: they are prepared on every call
(an in-place update of such a parameter could not be tracked) -- create them outside inference mode, or prepare them once with
`prepare_weights` and use the class API, to pay the interleave once.
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
decisively in every pass that measured it. The bucket picks: the contraction tile (the 1-CTA 128x128 tile at N &lt;= 128 and at 384, the 2-CTA
pair tile at 129-256 and 768, with the A operand multicast across 2 CTAs at 512 and across 4 from 1024), the projection stage's tile order
(`interleaved` at c = 128 below N = 512, `major` elsewhere), the output stage's prologue form (`transpose` at c = 128, `mn_major` at c = 256)
and the scheduler ring depth (1 from N >= 384 at c = 128 and from N >= 129 at c = 256, else 2). The refinements: at c = 128 the ring is 1 deep
at N &lt;= 128 for the incoming direction with B = 1, 1 deep with the pass-major projection order at 256 &lt;= N < 384 from B >= 4, and 4 deep at
384 &lt;= N < 512 with B = 1; at c = 256 the ring is 1 deep at N &lt;= 128 from B >= 4. Batch sizes
other than 1 and 4 take the nearest measured batch bucket below (B >= 4 -> the B = 4 picks, else the B = 1 picks), an N between two buckets
the first bucket at or below it. Every pick is the best of a three-pass measured sweep on one GPU of a 212-SM Rubin perf node (both directions,
B = 1 and 4, a control pair per cell; the second and third passes re-measured every alternative against the table they shipped). The
constructor's `k2_tile=` (one axis) or `knobs=` (a `TriMulKnobs` whose set axes override the pick; an axis left `None` keeps the table's value)
override it, `check_support` validates the vocabulary and probes the contraction tile's eligibility for the geometry without compiling. Every
value of every domain computes the same function (bitwise; the tests pin it). One axis is not a geometry pick: `min_blocks_per_mp`, the two
LayerNorm-prologue stages' `.minnctapersm` launch hint (0 = the rendering as shipped, whose register split ptxas drops; 1 = the hint under
which ptxas honours it), is resolved per RENDERING by `heuristics.register_split_hint(stage, c, has_bias, ln_pass_form, epi_math, pass_order,
ln_once)` from the measured rows -- whether a rendering spills at the flat allocation depends on its bias arm, its forms, the projection stage's
tile order and the LayerNorm-once work item, not on N or B.  Every measured row names the per-tile item (`ln_once=""`); under the `"block"`
item the table ships from N = 256 at c = 128 the hinted renderings spill and the hint measured a loss of 0.7..1.3 % on the whole call without
the biases (neutral with them, on the 212-SM Rubin class), so those plans run unhinted -- and a `knobs=` value overrides it for both stages.
The epilogue arithmetic form has a bias-path member, `epi_math="packed_biasfold"` (the gate bias staged
halved and folded into the tanh argument's fused multiply-add: 16 fewer packed adds per 32-channel subtile per lane; output-bitwise the
`packed` form, and byte-identical to it when the stage has no biases), which the table picks where the projection stage runs the register
LayerNorm form at N >= 256 (+1.0..+1.7 % on the whole call with the biases on the 212-SM Rubin class; a rolled-form projection stage loses
~0.7 % under it, so those cells keep `packed`); `"packed_hoist"` (the bias loads issued ahead of the TMEM loads) is admitted as a knob and
measured neutral.

## Traffic model and measured performance

Compulsory HBM traffic per call: `18 N^2 C` bytes per batch entry (K1 reads `x` 2 B per pair-channel and writes `a`, `b` 4 B; K2
reads 4 B and writes `y` 2 B; K3 reads `y` and `x` 4 B and writes 2 B) plus the mask (2 B per pair) --
`TriangleMultiplicativeUpdateFwd.traffic_bytes()`. The figures below are the shipped heuristics (`recommend`'s pick per cell, no override)
measured as the base of a sweep pass on ONE GPU of a Rubin node (cc 10.7, 212 SMs, SM clock locked at 2376 MHz and sampled per cell; every
window CLEAN) against that node's HBM pin of a read + write copy kernel (10 237 GB/s); the competitor is cuEquivariance 0.12.0's generic route
(its only route on this architecture) measured on the SAME GPU in the same reserved session with the same protocol (graph replay vs graph
replay; 600-iteration competitor windows so the clock sampler sees the small cells -- the two N = 128 cells at 6000 iterations with the
sampler's busy threshold at 20 % GPU utilization, because a 36-40 us call keeps the GPU at ~30 % busy and would otherwise leave no clock
sample; and every competitor cell whose call is under ~200 us (N &lt;= 512 here) in its OWN process at 1200 iterations, because a profiler
session earlier in the same process leaves CUPTI attached and adds 3.5-4.5 us of launch latency to every later graph replay, which a
short call cannot hide -- see the paragraph below the table).
Speed-ups are `t_cuEq / t_FROST - 1` (positive = FROST faster); the all-valid mask, biases present, `c = c_z`. Three columns, three
measurement states, never mixed: the FROST and cuEquivariance columns are graph replays of the whole call; the per-kernel column is the
CUPTI device time of each kernel inside the eager chain -- the median over 10 calls with its min-max; under programmatic dependent launch
the three overlap, so their sum can exceed the chain's span -- and the K1 + K3 share below is read from it. `C = 128`, `B = 1`:

| N | direction | FROST, graph replay (us) | K1 / K2 / K3 device time (us, CUPTI: median (min-max) over n = 10 eager calls of the chain) | GB/s (% of the 10237 GB/s pin) | TFLOPS | cuEquivariance 0.12.0, same GPU (us) | speed-up vs cuEq | host floor per eager call (us) | clock |
|---|---|---|---|---|---|---|---|---|---|
| 128 | outgoing | 29.3 | 11.1 (10.8-12.3) / 3.7 (3.6-3.8) / 10.4 (10.2-10.7) | 1288 (12.6 %) | 128 | 36.3 | +23.8 % | 96.7 | 2364.0 MHz, CLEAN |
| 128 | incoming | 29.7 | 11.0 (10.8-11.2) / 3.7 (3.7-3.8) / 10.2 (10.1-10.5) | 1272 (12.4 %) | 126 | 39.7 | +33.7 % | 96.8 | 2364 MHz, CLEAN |
| 217 | outgoing | 53.2 | 24.6 (24.5-24.8) / 7.0 (6.6-7.1) / 19.1 (18.6-19.6) | 2042 (19.9 %) | 223 | 76.5 | +43.9 % | 97.6 | 2364 MHz, CLEAN |
| 256 | outgoing | 63.4 | 27.3 (26.9-27.5) / 7.2 (7.0-7.8) / 26.3 (25.5-27.6) | 2383 (23.3 %) | 271 | 64.0 | +0.9 % | 95.9 | 2364.0 MHz, CLEAN |
| 256 | incoming | 63.4 | 27.4 (27.2-27.8) / 7.2 (6.8-7.8) / 26.2 (25.7-28.2) | 2384 (23.3 %) | 271 | 64.2 | +1.3 % | 96.9 | 2364 MHz, CLEAN |
| 384 | outgoing | 146.9 | 71.5 (71.2-72.0) / 18.7 (18.5-19.2) / 60.7 (59.9-60.9) | 2314 (22.6 %) | 296 | 115.1 | -21.7 % | 97.8 | 2364.0 MHz, CLEAN |
| 384 | incoming | 147.7 | 71.7 (71.5-72.0) / 19.3 (18.7-20.0) / 59.9 (57.8-61.5) | 2302 (22.5 %) | 294 | 115.0 | -22.1 % | 98.8 | 2364.0 MHz, CLEAN |
| 512 | outgoing | 187.8 | 93.1 (92.7-93.5) / 24.1 (23.5-24.8) / 72.1 (71.5-72.6) | 3218 (31.4 %) | 457 | 197.8 | +5.3 % | 99.5 | 2364 MHz, CLEAN |
| 512 | incoming | 188.1 | 93.3 (92.6-94.4) / 24.3 (23.9-24.9) / 72.0 (71.7-72.3) | 3213 (31.4 %) | 457 | 197.8 | +5.1 % | 99.1 | 2364.0 MHz, CLEAN |
| 768 | outgoing | 395.5 | 202.9 (201.7-203.2) / 51.2 (50.5-52.5) / 143.3 (143.2-143.7) | 3439 (33.6 %) | 586 | 423.5 | +7.1 % | 98.1 | 2364 MHz, CLEAN |
| 768 | incoming | 396.7 | 202.8 (202.4-203.5) / 51.9 (51.3-53.1) / 143.6 (143.1-144.4) | 3428 (33.5 %) | 585 | 423.6 | +6.8 % | 99.7 | 2364.0 MHz, CLEAN |
| 1024 | outgoing | 690.4 | 356.7 (356.0-357.6) / 89.6 (89.2-90.1) / 245.7 (245.3-246.1) | 3502 (34.2 %) | 697 | 743.6 | +7.7 % | 98.9 | 2364 MHz, CLEAN |
| 1024 | incoming | 691.3 | 356.5 (355.0-357.4) / 90.4 (89.9-91.3) / 245.7 (245.2-246.2) | 3498 (34.2 %) | 696 | 742.9 | +7.5 % | 100.6 | 2364 MHz, CLEAN |
| 2048 | outgoing | 2905.4 | 1410.9 (1410.4-1412.2) / 554.3 (553.3-555.2) / 941.8 (941.5-942.2) | 3329 (32.5 %) | 1041 | 3111.5 | +7.1 % | 101.3 | 2364.0 MHz, CLEAN |
| 2048 | incoming | 2907.3 | 1410.7 (1409.6-1412.7) / 555.8 (553.0-557.7) / 941.9 (941.3-942.8) | 3327 (32.5 %) | 1040 | 3115.6 | +7.2 % | 101.2 | 2364.0 MHz, CLEAN |

Per group of the 52-cell grid (N in {128, 217, 256, 384, 512, 768, 1024, 2048}, both directions where `N % 8 == 0`; the eight cells -- four geometries, both
directions -- with `2 c B N n_pad >= 2^31` are declined with the typed `NotImplementedError` of the extent row above -- `C = 256, B = 4` from N = 1024, `C = 128, B = 4`
at N = 2048 and `C = 256, B = 1` at N = 2048, where `2 * 256 * 2048 * 2048` is exactly 2^31):

- `C = 128, B = 1` (15 cells): 12.4-34.2 % of the pin; vs cuEquivariance +24 / +34 % at N = 128, +44 % at 217, +0.9 / +1.3 % at 256,
  +5.1-7.7 % from 512 to 2048, and **-21.7 / -22.1 % at N = 384** (the projection kernels' 256-row tiles waste a quarter of their rows at
  `N % 256 == 128`).
- `C = 128, B = 4` (13 cells, N &lt;= 1024): 23.6-35.3 % of the pin; vs cuEquivariance +1.2 / +1.0 % at 128, +39 % at 217, +5.3 / +5.3 % at 256,
  +6.2-7.9 % from 512 to 1024, -19.6 / -19.8 % at 384.
- `C = 256, B = 1` (13 cells, N &lt;= 1024): 7.0-13.7 % of the pin; **slower than cuEquivariance by 20-44 %** at N >= 217 (-32 % at 217, -38 % at
  256, -44 % at 384, -24 % at 512, -22 % at 768, -20 % at 1024); +96 % at N = 128, where cuEquivariance's own route takes 207 us.
- `C = 256, B = 4` (11 cells, N &lt;= 768): 10.4-13.8 % of the pin; slower than cuEquivariance by 13-40 % (-38 % at 128, -13 % at 217, -24 % at
  256, -40 % at 384, -21 % at 512, -20 % at 768).

At `C = 128` the two LayerNorm-prologue GEMMs (K1 + K3) are 81-90 % of the chain's device time (85 % at N = 128, 86-88 % at 217-1024, 81 % at
N = 2048 -- the CUPTI column above) and the contraction (K2) the rest; the call reaches 23-35 % of the HBM pin on the 18 N^2 C model from
N >= 256. At `C = 256` the K = 256 renderings of K1 and K3 are register-bound (the LayerNorm row held in registers between the two statistics
passes spills 240 / 499 words per lane at a flat 168 registers; K1 alone runs at 5.8 % of the pin, 17x its byte bound) and K1 + K3 are 93-96 %
of the device time, so the call sits at 7-14 % of the pin and behind cuEquivariance's generic route -- the width is served correctly (the
oracle cells pass) but is not a performance claim until the LayerNorm pass stops holding the row in registers. The heuristics table is worth
+1.9..+8.3 % (`C = 128, B = 1`), +4.4..+9.0 % (`C = 128, B = 4`), +0.1..+1.3 % (`C = 256, B = 1`) and +1.0..+3.0 % (`C = 256, B = 4`) over the
pre-sweep defaults (the prototype's per-N contraction tile, the per-width projection order, the MN-major output prologue, a 2-deep scheduler
ring), measured in the sweep that derived it on another GPU of the same node class. Absolute times reproduce across GPUs and nodes of the
class: against that derivation pass the figures above read within +-1.3 % on 50 of 52 cells (median +0.02 %); the `c = 256`, N = 128 cells are
the exception at 5 % slower here (105.8 vs 100.6 us). Of the competitor windows in the table above only the N = 2048 outgoing cell ran at a
2352 MHz busy clock (one step under the lock, -1.0 %; CLOCK-DEGRADED by the protocol) -- its incoming twin read CLEAN at 2364 MHz; across the
52-cell grid the competitor's degraded windows are five N = 2048 cells (`C = 128, B = 1` outgoing, `C = 128, B = 4` and `C = 256, B = 4` in
both directions). Every FROST window is CLEAN; every competitor window of the table holds 30 or more busy clock samples (the FROST windows
22-140). The competitor figures at N &lt;= 512 come from one process per cell: measured in one process after another cell's profiler session
(this table's first pass) the same calls read 3.9-4.6 us slower -- 68.0 us at N = 256 for the 64.0 here, 80.6 for the 76.5 at N = 217,
67.4 / 67.7 for the 63.2 / 63.1 at N = 128, B = 4 -- which had turned +0.9 / +1.3 % into +7.2 / +7.5 % at N = 256 and +1.2 / +1.0 % into
+7.9 / +8.4 % at N = 128, B = 4; the cells above ~120 us per call were unchanged within 1 %. The FROST column is not exposed to it: the
sweep interleaves twenty slots per iteration, so a graph replay never starts on an idle device, and the derivation pass (no profiler
session) agrees with these figures within the +-1.3 % stated above. cuEquivariance's eager host floor on this GPU is 217-289 us
per call; the FROST class API's is 96-101 us at `C = 128`, `B = 1` (86-95 us on the derivation node) -- at N &lt;= 256 an eager call is
host-bound (device 29-63 us), which the captured CUDA graph removes.

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
N &lt;= 384, +0.3..+0.5 % at 512 and nothing above (vs the PDL-off eager chain: +3.9 % at 256 down to +0.3 % at 2048). An earlier revision of this
page stated that programmatic dependent launch was "not an arm of this API" because the launch path carried no such attribute; that was
wrong -- the chain had PDL on all along, and the PDL-off column was the baseline it was missing.

## Tests

`test/python/triangle_multiplicative_update/cutedsl/test_trimul_api.py` (the lazy export in every import order, the signature pin, the
typed declines, the workspace and traffic models, the fp64 reference against the formula; on Rubin: the
end-to-end oracle, the API bitwise equal to the hand-chained kernels, the wrapper bitwise equal to the class API with a plan-cache
hit, `execute` allocating nothing and never synchronising, CUDA-graph capture and replay, bool / fp32 masks and batchless inputs,
`precision='bfloat16'`, the live-device declines, the stock-caller invocation shape hitting both caches, a plan declared on a
non-current device, the autograd decline). The three kernels have their own modules beside it; the padded contraction graph is
`test_contraction_graph.py`'s.
