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
| pair width `c_z` / hidden width `c` | `c_z == c` in `{128, 256}` (`c_z % 32 != 0` is a `ValueError`: cuEquivariance's own rule); 384 is withdrawn until its renderings stop spilling and carry an oracle cell | `NotImplementedError` |
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
`recommend(N, c, B, direction)` returns the measured default `TriMulKnobs` per (hidden width, N bucket) -- the contraction tile (the 1-CTA
128x128 tile at N <= 128 and at 384, the 2-CTA pair tile at 129-256 and 768, with the A operand multicast across 2 CTAs at 512 and across 4
from 1024), the projection stage's tile order (`interleaved` at c = 128 below N = 512, `major` elsewhere), the output stage's prologue form
(`transpose` at c = 128, `mn_major` at c = 256) and the scheduler ring depth (1 from N >= 384 at c = 128 and from N >= 129 at c = 256, else 2) --
each pick the best of a two-pass measured sweep on a 212-SM Rubin perf GPU (both directions, B = 1 and 4, a control pair per cell; the second
pass re-measured every alternative against the table). The constructor's
`k2_tile=` (one axis) or `knobs=` (the whole record) override it, `check_support` validates the vocabulary and probes the contraction tile's
eligibility for the geometry without compiling. Every value of every domain computes the same function (bitwise; the tests pin it).

## Traffic model and measured performance

Compulsory HBM traffic per call: `18 N^2 C` bytes per batch entry (K1 reads `x` 2 B per pair-channel and writes `a`, `b` 4 B; K2
reads 4 B and writes `y` 2 B; K3 reads `y` and `x` 4 B and writes 2 B) plus the mask (2 B per pair) --
`TriangleMultiplicativeUpdateFwd.traffic_bytes()`. The measured figures below were taken on ONE GPU of a Rubin node (cc 10.7, 212 SMs,
SM clock locked at 2376 MHz and sampled per cell; no other process on the node) with that node's HBM pin of a read + write copy kernel
(10 067 GB/s); the competitor is cuEquivariance 0.12.0's generic route (its only route on this architecture), measured with the same
protocol on the SAME GPU in the same hour (graph replay vs graph replay; the cuEq cells at N <= 384 from a 600-iteration window so the
clock sampler sees them). Speed-ups are `t_cuEq / t_FROST - 1` (positive = FROST faster); `D = c = 128`, `B = 1`, the all-valid mask,
biases present. Three launch-glue arms: `eager, PDL on` = the production chain (every kernel rendered with programmatic dependent
launch), `eager, PDL off` = the same three kernels rendered without it, `graph replay` = the captured CUDA graph of the PDL-on chain.

| N | direction | FROST API, graph replay (us) | eager, PDL on (us) | eager, PDL off (us) | PDL on vs off | K1 + K3 / K2 device time (us, CUPTI) | GB/s (% of the 10 067 GB/s pin) | TFLOPS | cuEq 0.12.0 forward, same GPU (us) | speed-up vs cuEq | host floor per call: class API / drop-in (us) | clock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | outgoing | 65.3 | 66.1 | 67.8 | +2.5 % | 55.1 / 7.1 | 2314 (23.0 %) | 263 | 65.1 | -0.3 % | 81 / 100 | 2364 MHz, CLEAN |
| 256 | incoming | 65.5 | 66.2 | 67.5 | +2.0 % | 55.1 / 6.9 | 2308 (22.9 %) | 262 | 69.5 | +6.1 % | 88 / 110 | 2364 MHz, CLEAN |
| 384 | outgoing | 152.2 | 157.5 | 155.6 | -1.3 % | 141.7 / 19.2 | 2234 (22.2 %) | 286 | 117.1 | -23.1 % | 90 / 110 | 2364 MHz, CLEAN |
| 384 | incoming | 152.5 | 157.4 | 155.6 | -1.2 % | 141.4 / 19.8 | 2230 (22.1 %) | 285 | 117.5 | -23.0 % | 90 / 110 | 2364 MHz, CLEAN |
| 512 | outgoing | 198.5 | 199.5 | 202.4 | +1.4 % | 175.4 / 25.8 | 3046 (30.3 %) | 433 | 199.3 | +0.4 % | 90 / 109 | 2364 MHz, CLEAN |
| 512 | incoming | 199.7 | 200.3 | 202.9 | +1.3 % | 175.6 / 26.5 | 3026 (30.1 %) | 430 | 199.5 | -0.1 % | 92 / 110 | 2364 MHz, CLEAN |
| 768 | outgoing | 425.9 | 426.2 | 430.7 | +1.0 % | 373.2 / 54.7 | 3194 (31.7 %) | 545 | 425.3 | -0.1 % | 90 / 112 | 2364 MHz, CLEAN |
| 768 | incoming | 426.4 | 427.3 | 431.1 | +0.9 % | 373.1 / 55.5 | 3190 (31.7 %) | 544 | 425.4 | -0.2 % | 92 / 112 | 2364 MHz, CLEAN |
| 1024 | outgoing | 741.6 | 742.7 | 747.0 | +0.6 % | 650.8 / 93.4 | 3260 (32.4 %) | 649 | 744.9 | +0.4 % | 92 / 112 | 2364 MHz, CLEAN |
| 1024 | incoming | 742.0 | 743.2 | 747.1 | +0.5 % | 651.0 / 93.9 | 3259 (32.4 %) | 648 | 743.3 | +0.2 % | 92 / 113 | 2364 MHz, CLEAN |
| 2048 | outgoing | 3116.8 | 3117.8 | 3124.9 | +0.2 % | 2569.2 / 575.4 | 3103 (30.8 %) | 970 | 3112.9 | -0.1 % | 91 / 111 | 2352 MHz, CLOCK-DEGRADED |
| 2048 | incoming | 3116.1 | 3117.5 | 3124.7 | +0.2 % | 2557.4 / 574.2 | 3104 (30.8 %) | 970 | 3117.0 | +0.0 % | 97 / 114 | 2352 MHz, CLOCK-DEGRADED |

The FROST API matches cuEquivariance's generic route within +0.4..-0.2 % at N >= 512 on the same GPU, trails it at N = 384 (-23 %: the
projection kernels' 256-row tiles waste a quarter of their rows at N % 256 == 128) and is within the measurement's own spread at N = 256
(-0.3 % / +6.1 % for outgoing / incoming on the same GPU; cuEquivariance's 5-kernel graph replay at N = 256 itself varied by 2.5 / 8 %
between two nodes of this class, so no N = 256 figure below +-8 % is a ranking). The two LayerNorm-prologue GEMMs (K1 + K3) are 82-89 % of
every call; the contraction (K2) runs at 78-86 % of the HBM pin on its own and is 11-18 %. The whole call reaches 22-32 % of the HBM pin on
the 18 N^2 C model -- the headroom is in K1 / K3 (their LayerNorm warps are latency-bound on serial fp32 reductions), not in the
contraction. cuEquivariance's eager host floor on the same GPU was 222-244 us per call; the FROST drop-in's is 100-114 us, the class API's
81-97 us -- at N = 256 an eager call is host-bound (device 65 us), which the captured CUDA graph removes. The N = 2048 cells ran at a
2352 MHz busy clock (one step under the lock, power 1.7 kW) and are marked CLOCK-DEGRADED by the protocol (cuEquivariance's N = 2048 cells
likewise, 2340-2352 MHz at 2.1 kW).

Byte count (Nsight Compute `dram__bytes.sum` over the three kernels, warm L2 between launches): 2414 MB measured against 2418 MB modelled at N = 1024 (ratio 0.998: K1 794 MB = 0.98x its model with `x` read once, K2 822 MB = 1.02x, K3 798 MB = 0.99x); at N = 512 the warm L2 keeps part of the planes between kernels and the chain reads 0.93x the model. With the pass-major K1 tile order the N = 1024 chain read 1.11x the model (K1 re-read `x` from HBM for its second projection pass), which is why the API fixes the interleaved order.

Launch glue (the table's three arms): every kernel of the chain carries programmatic dependent launch (PDL) -- it is rendered with
`USE_PDL = True` (`griddepcontrol wait` before its first global read, `launch_dependents` once its mainloop no longer needs the
predecessor) and launched with the CuTe-DSL kernel launch's `use_pdl` attribute, so K2's prologue overlaps K1's tail and K3's overlaps
K2's. Against the same three kernels rendered without PDL, the eager chain is +0.2..+2.5 % faster with it at every N except 384 (-1.2 %,
where the eager PDL chain pays more than it overlaps; the graph replay of the PDL-on chain is still the fastest form there); the captured
CUDA graph keeps the PDL edges and adds +1.1..+3.5 % over the eager PDL-on chain at N <= 384, +0.3..+0.5 % at 512 and nothing above (vs
the PDL-off eager chain: +3.9 % at 256 down to +0.3 % at 2048). The host floor per eager call is 81-97 us for the class API and
100-114 us for the drop-in function, against 65-200 us of device time at N = 256..512. An earlier revision of this page stated that
programmatic dependent launch was "not an arm of this API" because the launch path carried no such attribute; that was wrong -- the
chain had PDL on all along, and the PDL-off column above is the baseline it was missing.

## Tests

`test/python/triangle_multiplicative_update/cutedsl/test_api.py` (the lazy export in every import order, the signature pin, the
typed declines, the padded contraction graph, the workspace and traffic models, the fp64 reference against the formula; on Rubin: the
end-to-end oracle, the API bitwise equal to the hand-chained kernels, the wrapper bitwise equal to the class API with a plan-cache
hit, `execute` allocating nothing and never synchronising, CUDA-graph capture and replay, bool / fp32 masks and batchless inputs,
`precision='bfloat16'`, the live-device declines, the stock-caller invocation shape hitting both caches, a plan declared on a
non-current device, the autograd decline). The three kernels have their own modules beside it.
