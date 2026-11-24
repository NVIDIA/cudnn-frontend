# MoE Grouped Matmul

## Overview

The MoE Grouped Matmul operation computes a grouped matrix multiplication across experts, as used in Mixture-of-Experts (MoE) layers. Each expert has its own weight matrix, and tokens are routed to experts via `first_token_offset`.

Four routing modes are defined; COMBINE currently requires the FROST `frost_gemm` engine (see FROST COMBINE support below):

**None mode** (tokens already routed per expert):

$$\text{Output}[1,\ S \times \text{topK},\ N] = \text{Token}[1,\ S \times \text{topK},\ K]\ \times\ \text{Weight}[E,\ K,\ N]$$

**Gather mode** (gather tokens from unrouted layout before matmul):

$$\text{Output}[1,\ S \times \text{topK},\ N] = \text{Token}[1,\ S,\ K]\ \times\ \text{Weight}[E,\ K,\ N]$$

**Scatter mode** (scatter output back to token order after matmul):

First compute the grouped matmul for each routed row, then scatter each result back to the expanded token/top-k output slot using `TokenIndex` and `TokenKs`:

$$\text{GroupedResult}[r,\ N] = \text{Token}[1,\ r,\ K]\ \times\ \text{Weight}[\text{expert}(r),\ K,\ N]$$

$$\text{Output}[1,\ \text{TokenIndex}[r] \times \text{topK} + \text{TokenKs}[r],\ N] = \text{GroupedResult}[r,\ N]$$

**Combine mode** performs FC2, routing-score multiplication, and summation across selected experts, producing `[1, S, N]`. See the FROST COMBINE contract below.

where $E$ = number of experts, $S$ = number of tokens, $K$ = hidden size, $N$ = output (weight) size.

### Tensor Roles by Mode

| Tensor | Shape | Modes |
|---|---|---|
| `Token` | `[1, S*topK, K]` (None/Scatter/Combine) or `[1, S, K]` (Gather) | All |
| `Weight` | `[E, K, N]` | All |
| `FirstTokenOffset` | `[B*E+1, 1, 1]` explicit boundaries; INT32/INT64 | All |
| `TokenIndex` | `[1, S*topK, 1]`, INT32 | Gather, Scatter, Combine |
| `TokenKs` | `[1, S*topK, 1]`, INT32 | Scatter, Combine |
| `TopK` | scalar int32 | Scatter, Combine |
| `TopKScores` | `[1, S, topK]`, FP32 | Combine only |

## Support Matrix

The support matrix is based on the latest cuDNN backend.

| Operation | Minimum cuDNN | Minimum cublasLt | Datatype (I/O) | Compute type | Fusion pattern |
|---|---|---|---|---|---|
| MoE Grouped Matmul (forward) | 9.18.0 | — | int8, fp8, fp16, bf16, fp32, nvfp4/mxfp8 (9.21.0) | FLOAT | 9.21.0: SwiGLU, AMAX, etc. |
| MoE Grouped Matmul Bwd | 9.22.0 | 13.5 | fp16, bf16 | FLOAT | - |

### Important Notes

1. Let `G=B*E`. Use `G+1` nondecreasing boundaries starting at zero. Group g spans `[offset[g], offset[g+1])` and uses expert `g % E`. The final boundary may be smaller than the routed capacity (`TokenIndex.shape[1]` in Gather mode, otherwise `Token.shape[1]`); trailing routed rows are unused. In None/Gather mode their output rows remain untouched; in Scatter mode only destinations of used routed rows are written.
   **E=1 requires at least two boundaries.** FROST supports only explicit boundaries; native backend explicit-offset support requires cuDNN 9.28.0 or newer.
2. In **Scatter** mode, both `TokenIndex` and `TokenKs` are required, and `top_k` must be explicitly provided.
3. In **Gather** mode, `TokenIndex` is required.

### Deprecated implicit offsets

The length-G format, which infers the final endpoint from routed capacity, is
**deprecated**. It remains supported by native engines for E>1 for compatibility;
FROST declines it. For E>1, length modulo E selects the format: 1 for explicit
boundaries, 0 for deprecated implicit offsets. No mode parameter is needed.

To migrate, append the old routed-capacity endpoint to the G starts and declare
`FirstTokenOffset` as `[G+1,1,1]`. This preserves the old group ranges. Use a smaller
final endpoint when trailing capacity is intentionally unused. This applies to
both forward and backward APIs.

### FROST GATHER support

The opt-in `frost_gemm` engine supports ordinary and block-scaled GATHER on
SM100 and SM120, in both orientations: token-by-weight and weight-by-token
(`SWAP_AB=1`). Enable it with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` before
planning.

Let `T` be the source token count and `R` the routed row count. Supply token
`[1,T,K]`, weight `[E,K,N]`, INT32 `token_index` `[1,R,1]` with contiguous rows,
and INT32 or INT64 `first_token_offset` `[G+1,1,1]`. The result is `[1,R,N]`.
Offsets are nondecreasing boundaries beginning at zero, with final endpoint <= R;
group `g` uses expert `g % E`, and empty groups are supported. Each index is a
source row in `[0,T)`, so repeated source tokens are allowed.

For block-scaled GATHER, token scale factors (SFA) use `reorder=NONE` and
logical shape `[1,T,K/block]`, with contiguous K scales. Each scale occupies
one byte; the row stride must be at least `round_up(K/block,16)` bytes and
a multiple of 16 bytes, and the base pointer must be 16-byte aligned. SFA
uses source-token order and is gathered with the same `token_index` as the
token data. Token SFA in `F8_128x4` is not supported for GATHER. Weight scale
factors (SFB) retain the per-expert `F8_128x4` layout. These public operand
roles and layout requirements also apply with `SWAP_AB=1`.

For all block-scaled FROST MoE modes, bind weight SF as a rank-3 byte buffer
with E as its outermost axis. Each expert plane must be a dense packed
`F8_128x4` blob containing at least
`512 * ceil(N/128) * ceil((K/block)/4)` bytes. Its two inner axes may be
permuted. The expert stride must be a multiple of 16 bytes and at least the
full inner-plane byte span; padding between experts is supported. The base
pointer must also be 16-byte aligned. Views that interleave experts, overlap
planes, or expose insufficient plane capacity raise `ValueError` before launch.

```python
fc1 = graph.moe_grouped_matmul(
    token, weight, first_token_offset,
    token_index=token_index,
    mode=cudnn.moe_grouped_matmul_mode.GATHER,
)
activated = graph.swish(input=fc1)
activated.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
```

Routing logits, top-k selection, offsets, and indices are computed by the
caller. FROST uses TMA `tile::gather4` to load four indexed source rows directly
into shared memory, then runs grouped matmul and the supported epilogue in the
same kernel. No globally materialized permutation of the token matrix is
needed. With `SWAP_AB=1`, tokens are the internal B operand and GATHER4 loads
the token columns of the transposed matmul; the public tensors and routed
output order stay the same. SM120 uses the `shared::cta` GATHER4 instruction
with its existing warp-MMA mainloop and STG epilogue; its workspace remains
the 128-byte scheduler counter. Both SM100 and SM120 support parallel GEMMs,
including the gate/up pair for SwiGLU, sharing the same offsets and indices,
in both orientations and for ordinary and block-scaled inputs.
Output rows and row-wise epilogue auxiliaries use routed order.
The existing dtype, layout, alignment, and fusion support gates still apply.

### FROST linear-scale operands: gated FC1 and FP8 block-128 output

Operands whose scales are linear tensors run on the SM100 `frost_gemm` general
block-scale kernels: for example DeepSeek FP8, E4M3 data with one FP32 scale
per 128 K values of a token row (`block_size=[1,128]`) and one per 128x128
weight block (`block_size=[128,128]`). They serve NONE, GATHER, SCATTER and
COMBINE in both orientations, and the dense `matmul` of the same operands.

A weight scale tensor holds one matrix per expert, `[E, K/128, N/128]` for
the blocks above, or one that every expert shares, `[1, K/128, N/128]` (GATHER
takes one per expert). The plan is built for the count the graph declares, and
a call binds a scale tensor of that count: the other count is refused at the
call, before anything is written. A call binds the declared view, or any
buffer whose shape settles the axis order: `[E|1, K/128, N/128]` is read in
the graph's order through its own strides, `[E|1, N/128, K/128]` in the
kernel's order (for example a checkpoint's scale tensor bound without a
transpose). When the weight has as many N blocks as K blocks, strides cannot
tell the two orders apart, and only the declared view (its shape and strides)
is accepted.

Two fusions let FC1 produce FC2's input in one kernel, under NONE and GATHER
(and for the dense `matmul`):

- **Gated pair.** Two grouped matmuls that share the token operand, the
  offsets and the indices and differ in the weight, joined by a pointwise
  epilogue such as `mul(swish(gate), up)` (SwiGLU). The kernel runs every tile
  twice, once per weight, and the epilogue reads both results together. The
  weights are K-major. The tile config is a 1-CTA MMA tile of at most 128
  accumulator columns (two accumulator stages); split-K is not supported.
  The two weights' scale tensors declare the same shape (both one matrix per
  expert, or both shared); a pair whose scale shapes differ is refused when
  the plan is built.
  The pair's SiLU (`swish` without `swish_beta`, or with 1) is evaluated as
  `x * (1/2 + tanh(x/2) / 2)` through the approximate `tanh.approx.f32`, one
  MUFU operation per value. On SM100 it is within `4.0e-6 * |x|` of
  `x * sigmoid(x)`: under `4.4e-6` relative for `x >= -1`, more where
  `sigmoid(x)` approaches 0 (`1e-4` at `x = -6`, `1.4e-2` at `x = -12`), and
  `-0` for `x <= -16`. Every other FROST GEMM epilogue, a single GEMM of these
  kernels included, keeps `x / (1 + 2^(-x * log2(e)))` (with `swish_beta`,
  `beta * x` in the exponent) through approximate `exp2` and reciprocal
  (within `1.7e-7 * |x|` for SiLU). Against that form, a fused
  FC1 into FP8 block-128 changes 0.003-0.005% of its codes and its scales by
  at most `7e-6` relative; a BF16 output changes 8-14% of its codes on the
  gated test shapes, but only 0.07-0.13% of the values above 1% of the
  output's RMS. The error against an FP32 reference does not change.
- **FP8 block-128 output.** `block_scale_quantize(block_size=128)` whose scale
  output is `FLOAT`: E4M3 data `[1,R,N]` and one FP32 scale per 128
  output columns of a row, `[1,R,N/128]`. With `amax` the block's largest
  magnitude, NaN values ignored (0 when the block holds nothing else), the
  stored scale is `RN(amax * RN(1/448))`, raised to `2^-126` (the smallest
  normal FP32) when `amax > 0`, and a value `x` is stored as
  `cvt.rn.satfinite.e4m3(RN(x * min(rcp.approx.ftz(scale), FLT_MAX)))`. Above
  that floor the scale is the correctly rounded `amax / 448` or a neighbouring
  FP32 value. A block of zeros stores scale 0 and data 0; a NaN value stores a
  NaN code and leaves its block's scale unchanged; a block whose `amax` is
  below `448 * 2^-126` (about `5.3e-36`) stores codes below 448, because of
  the floor. `block_size=[1,128]` is accepted as the same block. `N` is a
  multiple of 128 and the scale tensor is 4-byte aligned, unreordered. It must
  be the graph's only output, without auxiliary epilogue inputs.
  Token-by-weight tiles are 128 columns wide; weight-by-token (`SWAP_AB=1`)
  tiles take any token width. Rows past the final offset are not written, in
  the data or in the scales. The scale output keeps the graph's shape, so a
  plan with this output runs only at the extents it was built for (R and N; M
  and N for the dense `matmul`): a call at other extents is refused before
  anything is written.

```python
gate = graph.moe_grouped_matmul(token, gate_weight, first_token_offset, token_index=token_index, mode=cudnn.moe_grouped_matmul_mode.GATHER)
up = graph.moe_grouped_matmul(token, up_weight, first_token_offset, token_index=token_index, mode=cudnn.moe_grouped_matmul_mode.GATHER)
hidden = graph.mul(a=graph.swish(input=gate), b=up)
data, scale = graph.block_scale_quantize(input=hidden, block_size=128)
data.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
scale.set_output(True).set_data_type(cudnn.data_type.FLOAT)
```

`token`, `gate_weight` and `up_weight` above are `block_scale_dequantize`
results over the FP8 operands and their FP32 scales. The two weights may be
views of one `[E, 2N, K]` buffer: declare each with the buffer's expert
stride. The quantized data and its scales are exactly the token operand and
`[1,128]` scale of a following FC2 graph.

The general block-scale grouped kernels follow these rules for the routing,
the launch and the token scales:

- **`alignment_value` of `first_token_offset` is a promise, not a checked
  input.** Declared with `graph.tensor(..., alignment_value=a)` or
  `set_alignment_value(a)`, it promises that every offset is a multiple of
  `a`. It sizes the weight-by-token epilogue's vector stores along the tokens
  and the runs of token columns that share a token scale (below), and when
  `a` is a multiple of 128 and of the cluster's token tile, the TMA-store
  epilogue writes every group through one output map. Offsets that break the
  promise let a tile write rows of the neighbouring group and rows past the
  final offset, and can stop the launch with a CUDA error. A weight-by-token
  plan runs token counts that are multiples of the greatest common divisor of
  the graph's token count, its epilogue vector and `a`: declare an `a` that
  divides every token count the plan runs.
- **Routed groups.** A plan is built for the group count its graph declares
  (`first_token_offset` `[G+1,1,1]`). A call that routes fewer groups runs; a
  call that routes more can be refused, before anything is written, so
  declare the largest group count the plan serves. Token rows after the final
  offset may hold anything: the kernels may read them, but nothing they hold
  reaches an output.
- **Empty calls.** A call without routed rows launches no GEMM (a COMBINE
  output is still zeroed). A call with K = 0, or a GATHER from a token buffer
  without rows, is refused before anything is written. K need not be a
  multiple of the scale block: the kernels read a partial last block.
- **Programmatic dependent launch.** The grouped kernels launch as
  programmatic dependents. Every role, the tile scheduler included, waits for
  the kernel ahead of it (`griddepcontrol.wait`) before it touches global
  memory, so a routing kernel may release its dependents before it writes the
  offsets. The kernels release their own dependents before their epilogue
  finishes: a kernel launched behind the call with programmatic stream
  serialization waits (`griddepcontrol.wait` or
  `cudaGridDependencySynchronize()`) before it reads the outputs.
- **Token scales under `SWAP_AB=1`.** A weight-by-token tile takes the tokens
  as its columns, from its group's first routed row. A token scale block of
  several rows (`[r,128]` with r > 1) is shared by a run of columns no longer
  than the greatest common divisor of r, a promotion thread's columns and
  `alignment_value`, and by no two columns under GATHER, whose columns are
  any token rows. An `alignment_value` that is a multiple of r keeps the runs
  long. One scale per token row (`[1,128]`, as in DeepSeek FP8) is read per
  column either way.

**Default tiles.** Without knobs, the engine plans a grouped graph of fixed
extents from r, the routed rows per group (`ceil(R / G)`), and K in bytes per
token row:

| Graph | Weight-by-token tile (`SWAP_AB=1`) |
|---|---|
| Gated pair, NONE or GATHER | 128x16 up to r = 16 (only at r = 1 from K = 4096 bytes), 128x32 up to r = 32, 128x64 up to r = 56 |
| One GEMM, NONE or GATHER | 128x16 up to r = 8 (only at r = 1 from K = 2560 bytes), 128x32 up to r = 16 |
| COMBINE | 128x16 at r = 1, or up to r = 4 with K of at most 768 bytes; past that, with K of at most 768 bytes and r * K of at most 8192 bytes, the 256x16 CTA pair (2x1 cluster) |

Past those rows, and for SCATTER, the plan is token-by-weight: 128x128 1-CTA
tiles in 1x2 clusters (one token tile multicast to two weight tiles) for
SCATTER, and for NONE from r = 128 with K of at least 1536 bytes, when the
weight's N tiles pair up and two-CTA clusters cover every SM; in 1x1 clusters
otherwise, and always for GATHER and COMBINE. A graph with override or
dynamic shapes is never swapped: GATHER takes 1x1, the other modes 1x2 under
the same conditions. Where the engine declines the default plan, it falls
back to 128x128 1x1. `create_execution_plans` also lists the other
weight-by-token widths, the 256x16 and 256x32 CTA pairs and 128x128 tiles in
1x1, 1x2 and 1x4 clusters for autotuning. These rules were measured on one
68-SM SM100 part.

A dense `matmul` of linear-scale operands takes 1-CTA tiles of 128 rows and
128 columns, in the widest of the 1x4, 1x2 and 1x1 clusters that divides its N
tiles. With fixed extents of at most 256 rows whose tiles fill at most half
the SMs, a K of at most 1024 bytes per row takes 64-column tiles (not with an
FP8 block-128 output or an N-major weight), and from K = 2048 bytes, where the
graph can split K, K splits into the largest power of two of at most 8 slices
that keeps the CTAs within one wave and at least two 128-byte K tiles per
slice: two slices only from K = 3072 bytes (6144 when the unsplit grid fills
more than 2/5 of the SMs), and four or eight slices that stop at two K tiles
per slice become 64-column tiles at half the slices. Past 256 rows at the
extents the graph declares, one GEMM of K-major operands with batch 1 and no
output quantize takes the 2x1 CTA pair (a 256x128 tile over two CTAs) from
512 rows, with a K of at least 2048 bytes, at least 16 N tiles, an even count
of 128-row tiles and more tiles than SMs; failing that, from a K of 1536
bytes, the 1x2 cluster when its N tiles pair up. An FP8 block-128 output, a
gated pair, an N-major weight and a batched graph keep the 1-CTA plan. The
unsplit 128-column plans (1-CTA in any cluster, or a CTA pair) write the same
bytes; a split-K plan adds its slices in another order. A graph with override or
dynamic shapes takes the plan that does not depend on M: 1-CTA 128-column
tiles, no split and no workspace.

**Host overhead of eager calls.** On SM100, a grouped `frost_gemm` plan runs
its full call-time validation the first two times a buffer signature occurs:
the bound buffers' types, shapes, strides, dtypes, devices, address alignment
(modulo 256 bytes) and which of them share an address. Later calls with that
signature, through `graph.execute` or the plan, replay the checked launch,
its output fills included: about 32-35 us of host time per launch instead of
124-141 us (FP8 block-128 FC1 and FC2 at one token, 68-SM SM100 part). The
workspace, the stream, the current device and the launch epoch are still
checked or drawn per call, and CUDA graph capture is unaffected. A call whose
buffers differ in any of those facts is validated in full.

### FROST SCATTER support

FROST supports SCATTER for ordinary and block-scaled MoE on SM100 and SM120,
in both orientations (`SWAP_AB=0/1`). The input and
its block scales keep their expert-grouped layout. The epilogue writes each
routed row `r` to the expanded token slot:

```text
destination = token_index[r] * top_k + token_ks[r]
output[0, destination, :] = grouped_result[r, :]
```

For `R = S * top_k`, supply token `[1,R,K]`, output `[1,R,N]`, and contiguous
INT32 `token_index` and `token_ks`, both `[1,R,1]`. The first index is in
`[0,S)`; `token_ks` is the position within the token's top-k list, in
`[0,top_k)`. Destinations of active routed tokens must be unique; unused output slots are left untouched.
`1 <= top_k <= E`; offsets retain the GATHER/NONE group convention with
G+1 explicit boundaries and support empty groups. Routing metadata is supplied
by the caller and may change between executions, including graph replay.

```python
fc2 = graph.moe_grouped_matmul(
    token, weight, first_token_offset,
    token_index=token_index, token_ks=token_ks, top_k=top_k,
    mode=cudnn.moe_grouped_matmul_mode.SCATTER,
)
fc2.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
```

SM100 uses TMA `tile::scatter4` for eligible N-major outputs, including
swapAB, and indexed STG for other supported layouts. SM120 uses indexed STG.
Pointwise epilogues and multiple materialized outputs share this store logic;
token-indexed auxiliary tensors and `gen_index` use the scattered output
coordinates. Input block scales retain their existing segmented SFA/per-expert
SFB contract. Output block quantization, reductions, and packed sub-byte
outputs are currently declined for SCATTER.

SCATTER preserves each top-k contribution separately. Multiplication by routing
weights and reduction across top-k require a subsequent operation to produce
the final `[1,S,N]` result.

---


### FROST COMBINE support

`cudnn.moe_grouped_matmul_mode.COMBINE` fuses FC2 with weighted scatter-add.
It is supported only by the opt-in `frost_gemm` engine, whose SM100 and SM120
source families both run it for ordinary and block-scaled inputs and both
`SWAP_AB=0/1` orientations. Native cuDNN engines decline this mode.

Let `R` be routed input capacity, `T` the original token count, and `k=top_k`.
Supply token `[1,R,K]`, weight `[E,K,N]`, contiguous INT32 `token_index` and
`token_ks` of shape `[1,R,1]`, and compact FP32 `top_k_scores` `[1,T,k]`.
The scores remain in original token/top-k-slot order. `1 <= k <= E`.
Offsets retain the explicit `[G+1,1,1]` contract; `R` may include unused
capacity and need not equal `T*k`.

For each active routed row `r` in group `g`, the operation computes:

```text
t = token_index[0, r, 0]
j = token_ks[0, r, 0]
output[0, t, :] += (token[0, r, :] @ weight[g % E, :, :]) * top_k_scores[0, t, j]
```

The output is `[1,T,N]`; every execution starts from zero, including empty
groups and repeated CUDA Graph replays. Out-of-range token/slot indices are
ignored. Repeated destinations accumulate; destinations without contributions
are zero. N-major and M-major output layouts, including padding, are supported.

For NVFP4 on SM107, the non-swap, single-CTA/singleton-cluster FROST path can
read compact token rows through immutable tensor maps. Expert token counts do
not need additional padding: COMBINE masks group-tail rows before reading
routing metadata or accumulating outputs. Token scales retain their existing
per-group 128-row blocked layout and use independent scale-block offsets. The
scale tensor map is bounded by the declared segmented-capacity contract,
including when the last TMA box extends beyond that capacity. Other layouts
and configurations retain their existing descriptor path.

The SM107 NVFP4 swapped `256x256x128`, `128x256x64` MMA, cluster `2x1`,
2-CTA tile also reuses immutable B/SFB maps with BF16 feature-contiguous output.
It transposes weighted fragments into a 128-token by 256-feature shared tile,
then bulk-reduces each token's contiguous feature segment. Warp-shared routing
loads and 64-column TMEM loads limit metadata traffic and live registers. The
compiler reserves this staging space before sizing the AB pipeline. Partial
feature segments or unaligned destinations use scalar reductions. This path
does not require expert token counts or output widths to match the tile.

```python
output = graph.moe_grouped_matmul(
    fc2_input, fc2_weight, first_token_offset,
    token_index=token_index, token_ks=token_ks,
    top_k=top_k, top_k_scores=top_k_scores,
    mode=cudnn.moe_grouped_matmul_mode.COMBINE,
)
output.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
```

GEMM accumulation and score multiplication use FP32. Each weighted contribution
is converted to the declared output dtype (BF16, FP16, or FP32), and atomic
addition accumulates in that dtype. Atomic order is unspecified, so results
are not bitwise deterministic and can differ from an FP32 top-k reduction
followed by a single output cast. Choose FP32 output when FP32 combining is
required. Scores are neither normalized nor modified by COMBINE.

The first implementation accepts one grouped GEMM, with optional input
block-scale dequantization, and one final output. Post-COMBINE pointwise,
reduction, quantization, and parallel-GEMM fusion are declined: applying those
operations before summing experts would change the graph's meaning.

The FC2 epilogue issues weighted atomic additions directly to the final output;
it does not materialize `[T*k,N]` results or launch a separate finalizer.
For token-by-weight GEMM with contiguous output features, each lane packs up to
eight BF16/FP16 or four FP32 contributions into a 16-byte vector atomic add.
The destination address selects narrower accesses when the output pointer or
row stride is unaligned. Single-CTA SM100-family block-scaled token-by-weight
kernels with 128-row MMA slices instead stage weighted feature tiles in shared memory
and issue `cp.reduce.async.bulk` once per row. Unaligned destinations and partial
feature tiles retain vector/scalar atomics. Shared-memory staging is included
in the GEMM pipeline budget. Normal 2-CTA configurations retain the existing
atomic path pending memory-safety qualification. Outside the SM107 swapped bulk
path described above, weight-by-token (`SWAP_AB`) and M-major outputs retain
scalar atomics because adjacent registers do not address adjacent features of
the same token. All paths preserve per-element atomicity and output-dtype
accumulation.

Bulk reduction trades shared-memory staging and synchronization for fewer
global atomics; it does not guarantee a speedup for every workload. A
[review spot check](https://github.com/NVIDIA/cudnn-frontend/pull/1942#pullrequestreview-5671615964)
on SM100 with NVFP4, two experts, top-k 2 and a single-CTA 128x256
tile measured BF16 COMBINE at 18.96 us versus 18.33 us for forced atomics at
S/N/K = 258/256/256, and 59.57 us versus 66.37 us at 8192/2048/2048.
These warmed CUDA Graph measurements include output zeroing and compare fixed
configurations, rather than end-to-end serving performance.
Output zeroing uses `cudaMemsetAsync` for contiguous storage and
`cudaMemset2DAsync` for padded row/column-major storage, preserving padding.
On SM100, a COMBINE plan of the general block-scale grouped kernels (linear
scale tensors, see "FROST linear-scale operands" above) zeroes its output with
a small FROST kernel instead, when the output is whole 16-byte vectors from a
16-byte aligned start: a contiguous output whose size is a multiple of 16
bytes, or padded rows whose width and pitch are multiples of 16 bytes. Other
layouts keep the memset. The kernel is launched with programmatic stream
serialization: it lets the grouped GEMM launch at once and waits for the
preceding stream work before its first store, which saved up to about 2.5 us
per MoE layer at 1-512 tokens on a 68-SM SM100 part. A kernel launched behind
the call with programmatic stream serialization must wait
(`griddepcontrol.wait` or `cudaGridDependencySynchronize()`) before it reads
the output, as it must behind the grouped GEMM itself, which releases its
dependents before its epilogue finishes.
The execute-time stream orders the output zeroing, the scheduler-counter
initialization kernel of the templates that have one (the general block-scale
grouped kernels have none), and the grouped GEMM. Routing metadata and scores
may change on graph replay, and runtime token/output extents remain
independent of routed capacity.

## MoE Grouped Matmul Forward

### C++ API

```cpp
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul(std::shared_ptr<Tensor_attributes> token,
                   std::shared_ptr<Tensor_attributes> weight,
                   std::shared_ptr<Tensor_attributes> first_token_offset,
                   std::shared_ptr<Tensor_attributes> token_index,   // optional, pass nullptr for None mode
                   std::shared_ptr<Tensor_attributes> token_ks,      // optional, pass nullptr unless Scatter mode
                   Moe_grouped_matmul_attributes options);
```

`Moe_grouped_matmul_attributes` is a lightweight structure with setters:

```cpp
Moe_grouped_matmul& set_name(std::string const&);

// Required: selects the routing mode
Moe_grouped_matmul& set_mode(MoeGroupedMatmulMode_t mode);
// MoeGroupedMatmulMode_t::NONE    — tokens already routed
// MoeGroupedMatmulMode_t::GATHER  — gather before matmul
// MoeGroupedMatmulMode_t::SCATTER — scatter after matmul

// Required for SCATTER mode
Moe_grouped_matmul& set_top_k(int32_t top_k_value);

Moe_grouped_matmul& set_compute_data_type(DataType_t value);
```

### Python API

#### Low-level graph API (`cudnn.pygraph`)

```python
output = graph.moe_grouped_matmul(
    token,                                # Token tensor
    weight,                               # Weight tensor
    first_token_offset,                   # Expert routing offsets
    token_index=None,                     # Required for Gather/Scatter modes
    token_ks=None,                        # Required for Scatter mode
    mode=cudnn.moe_grouped_matmul_mode.NONE,  # NONE, GATHER, or SCATTER
    top_k=1,                              # Top-k value; required for Scatter mode
    compute_data_type=cudnn.data_type.NOT_SET,
    name=None,
)
```

**Args:**
- `token` (cudnn_tensor): Token data.
  - None/Scatter modes: shape `(1, S*topK, K)`
  - Gather mode: shape `(1, S, K)`
- `weight` (cudnn_tensor): Expert weight data with shape `(E, K, N)`.
- `first_token_offset` (cudnn_tensor): INT32/INT64 tensor of shape `(B*E+1, 1, 1)` including the final endpoint. See [deprecated implicit offsets](#deprecated-implicit-offsets) for compatibility.
- `token_index` (Optional[cudnn_tensor]): INT32 tensor of shape `(1, S*topK, 1)`. Maps each routed slot to a source token index. Required for Gather and Scatter modes.
- `token_ks` (Optional[cudnn_tensor]): INT32 tensor of shape `(1, S*topK, 1)`. The top-k slot index in `[0, top_k)` for each routed token. Required for Scatter mode.
- `mode` (cudnn.moe_grouped_matmul_mode): Routing mode — `NONE`, `GATHER`, or `SCATTER`.
- `top_k` (int): Top-k routing value. Must be provided for Scatter mode.
- `compute_data_type` (Optional[cudnn.data_type]): Data type for internal computation. Defaults to FLOAT.
- `name` (Optional[str]): Name for the operation.

**Returns:**
- `output` (cudnn_tensor): Output tensor of shape `(1, M_out, N)`, where `M_out = token_index.shape[1]` for Gather mode, otherwise `token.shape[1]`.

#### High-level experimental API

```python
from cudnn.experimental.ops import moe_grouped_matmul

output = moe_grouped_matmul(
    token,              # (1, M, K) torch.Tensor, fp16 or bf16
    weight,             # (E, K, N) torch.Tensor, column-major inner dims
    first_token_offset, # (B*E+1, 1, 1) torch.Tensor, INT32
    token_index=None,   # (1, S*topK, 1) INT32; required for gather/scatter
    token_ks=None,      # (1, S*topK, 1) INT32; required for scatter
    mode="none",        # "none", "gather", or "scatter"
    top_k=1,            # required for scatter mode
)
```

The high-level API handles cuDNN handle management and graph caching automatically. cuDNN graphs are built once per unique (shape, dtype, mode, top_k) configuration and reused across subsequent calls.

### Configurable Options

- **Mode** (`mode`): Controls how tokens are routed to and from expert weight matrices.
  - `NONE`: Tokens are already ordered by expert (pre-routed). Direct grouped matmul with no reordering.
  - `GATHER`: Tokens are in original (un-routed) order. `TokenIndex` specifies which source token each expert slot reads from.
  - `SCATTER`: Tokens are pre-routed, but the output is scattered back to the original token order. Requires both `TokenIndex` and `TokenKs`.

- **TopK** (`top_k`): The number of experts each token is routed to. Required in Scatter mode for the scatter-back computation.

- **Compute data type** (`compute_data_type`): Sets the precision for internal accumulation. Defaults to FLOAT for fp16/bf16 I/O.

### Example (Python)

```python
import torch
import cudnn

num_experts = 8
token_num   = 1024   # S
hidden_size = 512    # K
weight_size = 256    # N

graph = cudnn.pygraph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# Token: [1, S, K], bf16, row-major
token_t = graph.tensor(
    name="token",
    dim=[1, token_num, hidden_size],
    stride=[token_num * hidden_size, hidden_size, 1],
    data_type=cudnn.data_type.BFLOAT16,
)

# Weight: [E, K, N], bf16, column-major inner dims (stride[1] == 1)
weight_t = graph.tensor(
    name="weight",
    dim=[num_experts, hidden_size, weight_size],
    stride=[hidden_size * weight_size, 1, hidden_size],
    data_type=cudnn.data_type.BFLOAT16,
)

# FirstTokenOffset: [E+1, 1, 1], INT32, including the final endpoint
fto_t = graph.tensor(
    name="first_token_offset",
    dim=[num_experts + 1, 1, 1],
    stride=[1, 1, 1],
    data_type=cudnn.data_type.INT32,
)

output_t = graph.moe_grouped_matmul(
    token_t, weight_t, fto_t,
    mode=cudnn.moe_grouped_matmul_mode.NONE,
    compute_data_type=cudnn.data_type.FLOAT,
    name="moe_fwd",
)
output_t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
graph.check_support()
graph.build_plans()
```

---

## MoE Grouped Matmul Backward

The backward operation computes the weight gradient $d\text{Weight}$ given the upstream gradient $d\text{Output}$ and the forward token activations:

$$d\text{Weight}[E,\ K,\ N] = \text{Token}^T[1,\ S,\ K]\ \times\ d\text{Output}[1,\ S,\ N]$$

per expert, where the per-expert token slices are determined by `FirstTokenOffset`.
Declare the `dweight` dimensions explicitly when using G+1 boundaries. Backward
has no weight input from which to recover E; omitted output dimensions retain
the deprecated implicit format's inference from the offset length.

### C++ API

```cpp
std::shared_ptr<Tensor_attributes>
moe_grouped_matmul_bwd(std::shared_ptr<Tensor_attributes> doutput,
                       std::shared_ptr<Tensor_attributes> token,
                       std::shared_ptr<Tensor_attributes> first_token_offset,
                       Moe_grouped_matmul_bwd_attributes options);
```

`Moe_grouped_matmul_bwd_attributes` is a lightweight structure with setters:

```cpp
Moe_grouped_matmul_bwd& set_name(std::string const&);

Moe_grouped_matmul_bwd& set_compute_data_type(DataType_t value);
```

### Python API

#### Low-level graph API (`cudnn.pygraph`)

```python
dweight = graph.moe_grouped_matmul_bwd(
    doutput,                              # Upstream gradient tensor
    token,                                # Forward token activations
    first_token_offset,                   # Expert routing offsets
    compute_data_type=cudnn.data_type.NOT_SET,
    name=None,
)
```

**Args:**
- `doutput` (cudnn_tensor): Upstream gradient with shape `(1, S, N)`, same layout as the forward output.
- `token` (cudnn_tensor): Forward token activations with shape `(1, S, K)`.
- `first_token_offset` (cudnn_tensor): INT32/INT64 tensor of shape `(B*E+1, 1, 1)`, same as used in the forward pass. See [deprecated implicit offsets](#deprecated-implicit-offsets) for compatibility.
- `compute_data_type` (Optional[cudnn.data_type]): Data type for internal accumulation.
- `name` (Optional[str]): Name for the operation.

**Returns:**
- `dweight` (cudnn_tensor): Weight gradient with shape `(E, K, N)`.

### Example (Python)

```python
import torch
import cudnn

num_experts = 8
token_num   = 1024
hidden_size = 512
weight_size = 256

graph = cudnn.pygraph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

# dOutput: [1, S, N], bf16, row-major
doutput_t = graph.tensor(
    name="doutput",
    dim=[1, token_num, weight_size],
    stride=[token_num * weight_size, weight_size, 1],
    data_type=cudnn.data_type.BFLOAT16,
)

# Token: [1, S, K], bf16, row-major
token_t = graph.tensor(
    name="token",
    dim=[1, token_num, hidden_size],
    stride=[token_num * hidden_size, hidden_size, 1],
    data_type=cudnn.data_type.BFLOAT16,
)

# FirstTokenOffset: [E+1, 1, 1], INT32, including the final endpoint
fto_t = graph.tensor(
    name="first_token_offset",
    dim=[num_experts + 1, 1, 1],
    stride=[1, 1, 1],
    data_type=cudnn.data_type.INT32,
)

dweight_t = graph.moe_grouped_matmul_bwd(
    doutput_t, token_t, fto_t,
    compute_data_type=cudnn.data_type.FLOAT,
    name="moe_bwd",
)
# dweight: [E, K, N]
dweight_t.set_dim([num_experts, hidden_size, weight_size]).set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A])
graph.check_support()
graph.build_plans()
```
