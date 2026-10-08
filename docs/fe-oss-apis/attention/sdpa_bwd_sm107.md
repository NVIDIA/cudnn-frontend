# SDPA Backward, d = 256 and d in (256, 512] (SM107 / Rubin; SM100 / SM103 via the 2x2-datapath body)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** pass for head dimension 256 on the NVIDIA Rubin line
(`SM107`, cc 10.7 through 11.9), implemented with CuTe DSL primitives:
tcgen05 MMAs with the accumulator resident in TMEM, TMA loads and stores, a
2-CTA cluster per KV block and a warp-specialized schedule. It consumes the
forward activations (`Q/K/V/O`), the loss gradient `dO` and the forward `Stats`
(natural-log LSE) and produces `dQ/dK/dV`.

Three FROST engines serve the d = 256 pass on cc 10.7 (`cudnn.sdpa.bwd.engines`), and two
more rows share the chain (the SM100 / SM103 d = 256 row and the cc 10.7 d in (256, 512]
row, both described further down); all five are `opt_in` — set
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` before `import cudnn`, and pin the
engine from the ranked plan list (`graph.plans` / `graph.select_plan(i)`) when
validating or measuring, because the bf16 d256 graph also has a native backend
plan:

* `sdpa_bwd_sm107` — bf16 / fp16 `sdpa_backward()` graphs;
* `sdpa_bwd_sm107_fp8` — per-tensor FP8 E4M3 `sdpa_fp8_backward()` graphs
  (cuDNN's contract: scalar descales in, FP8 or half gradients plus the
  `amax_dQ/dK/dV/dP` outputs);
* `sdpa_bwd_sm107_mxfp8` — block-scale MXFP8 `sdpa_mxfp8_backward()` graphs
  (E4M3 payloads with F8_128x4 E8M0 scale factors, bf16 `o_f16 / dO_f16`, bf16
  gradients, **no `amax_*` outputs** — a graph requesting them is declined; see
  [MXFP8](#mxfp8-numerics-sdpa_bwd_sm107_mxfp8)). The sole provider of that
  graph on Rubin: cuDNN 9.27 has no MXFP8 d = 256 backward kernel there.

A fourth engine, `sdpa_bwd_sm100_d256` (bf16 / fp16, SM100 / SM103, cc 10.0-10.6,
`opt_in`), runs the SAME chain on the Blackwell line over the 2x2-datapath main
kernel (see "The 2x2-datapath body" below); pin it the same way
(`startswith("sdpa_bwd_sm100_d256")`), because the bf16 d256 graph also has a
native backend plan there (cuDNN engine 5 on B200 / 9.26).

A fifth engine, `sdpa_bwd_sm107_d512` (bf16 / fp16, cc 10.7 - 11.9, `opt_in`,
2026-10-01), serves **d in (256, 512]** (multiples of 8, envelope-served on
512-wide tiles) on the same line -- see "The d in (256, 512] row" below.  It is
the only FROST d > 256 backward on cc 10.7, and the only backward at all for that
band there: cuDNN 9.26 builds no d > 256 backward plan on cc 10.7, so a pin by
`startswith("sdpa_bwd_sm107_d512")` is still the honest way to measure it.

There is no standalone wrapper for this pass yet; the graph API is the surface, plus the
adapters' own `execute` for the plan facts no graph declares (per-batch KV lengths, an
external `delta`, THD on the fp8 and MXFP8 rows — see Sequence lengths and THD below).

The kernels live in `python/cudnn/sdpa/bwd/kernels/sm107/`
(`bprop_d256_f16.py`, `bprop_d256_fp8.py`, `bprop_d256_mxfp8.py`; config
`bwd/config_sm107.py`), the adapters in `python/cudnn/sdpa/bwd/api_dsl_sm107.py`,
and the rest of the launch chain is shared with the other Blackwell-line backward
engines (`kernels/bprop_matmul_blackwell.py`, `kernels/bprop_chain_common.py`).

## Requirements

The CuTeDSL runtime (`nvidia-cutlass-dsl >= 4.8.0`, which knows `sm_107a` and
the 576-column TMEM allocation Rubin needs, plus `apache-tvm-ffi`), a CUDA 13.4+
driver (the kernels use Rubin's 327 KiB oversized shared-memory carveout), and
an SM107-line device.

## API usage

```python
import cudnn, torch

# BSHD-physical storage, logical (B, H, S, D) as cuDNN declares it
def bshd(b, h, s, d):
    return [s * h * d, d, h * d, 1]

g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16,
                  intermediate_data_type=cudnn.data_type.FLOAT,
                  compute_data_type=cudnn.data_type.FLOAT)
q  = g.tensor(name="q",  dim=[b, h_q,  s_q,  256], stride=bshd(h_q,  s_q,  256))
k  = g.tensor(name="k",  dim=[b, h_kv, s_kv, 256], stride=bshd(h_kv, s_kv, 256))
v  = g.tensor(name="v",  dim=[b, h_kv, s_kv, 256], stride=bshd(h_kv, s_kv, 256))
o  = g.tensor(name="o",  dim=[b, h_q,  s_q,  256], stride=bshd(h_q,  s_q,  256))
do = g.tensor(name="dO", dim=[b, h_q,  s_q,  256], stride=bshd(h_q,  s_q,  256))
stats = g.tensor(name="stats", dim=[b, h_q, s_q, 1], stride=[h_q * s_q, s_q, 1, 1],
                 data_type=cudnn.data_type.FLOAT)   # the forward's natural-log LSE
dq, dk, dv = g.sdpa_backward(name="bwd", q=q, k=k, v=v, o=o, dO=do, stats=stats,
                             attn_scale=1.0 / 16.0, use_causal_mask=True)
for t, (h, s) in ((dq, (h_q, s_q)), (dk, (h_kv, s_kv)), (dv, (h_kv, s_kv))):
    t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim([b, h, s, 256]).set_stride(bshd(h, s, 256))
g.validate(); g.build_operation_graph(); g.create_execution_plans([cudnn.heur_mode.A])
# pin the FROST row (the bf16 d256 graph has a native backend plan too)
idx = next(i for i in range(len(g.plans)) if g.get_plan_name_at_index(i).startswith("sdpa_bwd_sm107"))
g.select_plan(idx); g.check_support(); g.build_plans()
ws = torch.empty(g.get_workspace_size(), dtype=torch.uint8, device="cuda")
g.execute({q: q_t, k: k_t, v: v_t, o: o_t, do: do_t, stats: stats_t, dq: dq_t, dk: dk_t, dv: dv_t}, ws)
```

The FP8 row is the same graph through `g.sdpa_fp8_backward(...)` with E4M3
`q/k/v/o/dO`, the twelve scalar `descale_* / scale_*` inputs as `(1, 1, 1, 1)`
fp32 tensors, `dQ/dK/dV` declared E4M3 (or bf16 / fp16), and any of
`amax_dQ/dK/dV/dP` marked `set_output(True)`.

## The kernel chain

One backward call is a fixed sequence of launches on the caller's stream, every
scratch buffer carved from the caller's workspace (`graph.get_workspace_size()`
is a build-time function of the shape).  Both rows are **prepared launches**
(`bwd/prepared_sm107.py`, `kernels/sm107/prepared_host.py`): the plan compiles
ONE pointer-host artifact that runs the whole chain from device pointers, and
every stage below -- the padding copies, the `seq_kv` fill and the (rare) dS zero-fill
included, plus the fp8 row's fold + quantize passes -- is a kernel of that
artifact.  No torch op runs on the execute path; the graph binds its variant pack
straight into the artifact, follows the handle's stream and captures into a CUDA
graph.

```text
dot     delta = rowsum(dO ∘ O)                        (fp8: × descale_o · descale_dO)
main    per (kv block, head, batch), one 2-CTA cluster: walks the q tiles that
        attend the block; dV accumulates in TMEM and is stored per Q head;
        dS = attn_scale · P ∘ (dP − delta) is written to a kv-major
        [B, H_chunk, S_kv, S_q] GMEM workspace (fp8: e4m3 dS_q = e4m3(dS · scale_dP),
        amax_dP over the fp32 dS before the scale)
mm_dk   dK = dS · Q          batched GEMM over the workspace (bprop_matmul_blackwell,
                             the d = 256 cluster tile: 2x1, 256 × 256 per pair, no N padding;
                             fp8: the K64 fp8 arm over the e4m3 dS and the e4m3 Q payload,
                             epilogue · descale_dP · descale_q — then · scale_dK → e4m3 dK +
                             amax_dK at MHA, or fp32 true-unit per-Q-head partials under GQA;
                             MXFP8: the block-scale arm, EPI_NONE -- the caller's bf16 dK at MHA,
                             fp32 true-unit per-Q-head partials under GQA)
mm_dq   dQ = dSᵀ · K         same GEMM, the other operand major (fp8: · descale_dP · descale_k,
                             amax_dQ, · scale_dQ → the gradient dtype, straight into dQ)
fold    GQA only (half row): dK/dV = fixed-order sum of each KV head's group of
        per-Q-head partials.  fp8 row: dV always (fold + amax_dV + scale_dV + cast);
        dK under GQA, in the same launch (the partials are fp32 under GQA and summed
        BEFORE the amax, scale and cast, so the gradient is rounded once -- like the
        reference; a bf16 partial would be rounded a second time).  MXFP8 row: dK / dV
        under GQA through dkv_reduce -- the dK partials fp32 (rounded once), the dV
        partials bf16 (one rounding per group member: the main kernel stores them from
        its epilogue and an fp32 staging does not fit its shared memory)
```

The workspace is head-chunked (and batch-chunked on the half row) to one 8 GiB
budget shared by all three rows (at B=1 H=128 S=8K: 64-head chunks on the half row,
one 128-head launch on the fp8 row, 32-head chunks on the MXFP8 row; at 16K every
row still chunks); the artifact's host loops over the chunks with `head_base` /
`batch_base`, so one compiled artifact serves every launch of a plan.  The budget is
a chunking constant: `get_workspace_size()` reports the chunk's whole carve and the
caller allocates it -- a caller that cannot hold it bounds the plan with
`deselect_workspace_greater_than(...)`, a typed decline before any launch.  Measured on
Rubin (cc 10.7, 212 SMs, SM clock 2376 MHz) at that shape, whole row: 8 GiB over 4 GiB
is +3.7 % causal / +0.8 % dense on the bf16 row (4 -> 2 launches), +0.9 % dense on the
fp8 row (2 -> 1), and +9.3 % causal / +1.4 % dense on the MXFP8 row (8 -> 4; the control
twin's value -- the arm read +11.6 % against a slot whose control pair spread 2 %); every
extra chunk launch ends in a scheduler tail the next launch cannot fill.

The two GEMMs render the shared template at its **d = 256 cluster tile**
(`MatmulTemplateParams.cgrp_tile_mn = (256, 256)`: cluster 2x1, one 256-row ×
256-column tile per 2-CTA pair, six 32 KiB operand stages, a double-buffered
256-column TMEM accumulator so one tile's epilogue overlaps the next tile's
mainloop).  The SM100 d512 chain's (512, 512) tile would put the N-rank pair of
every cluster on columns 256..511 that a d = 256 gradient does not have — half of
every cluster's MMA work as TMA-OOB zero loads and clipped stores.  The selection
is the sm107 adapter's alone (Rubin line, d = 256); the SM100 chain's renderings
are unchanged, and a bitwise pin (`test_stage3_d256_rendering_is_bitwise_the_padded_one`)
holds the two tiles' dQ / dK / dV to identical bits (same k-tile walk, same
256x256x16 instruction, same fp32 accumulation order).

### Main kernel

One cga2 pair (2 CTAs, 12 warps each) owns a 256-row KV block of one (batch,
head) and iterates over the 128-row q tiles that attend it (lane = kv row, the
transpose of the forward). Per q tile the MMA warp issues `S = K·Qᵀ`,
`dP = V·dOᵀ` (both into TMEM) and `dV += P·dO` (TMEM accumulator, resident for
the whole block); eight softmax warps recompute `P = exp2(S·scale·log2e − LSE·log2e)`
from the forward's LSE, form `dS`, and stream it through a SMEM ring to the
workspace by TMA store. The bf16/fp16 body splits the `K·Qᵀ` contraction so half
of K rides TMEM (a UTCCP per block); the fp8 body keeps a lookahead MMA order
and a two-slot fp8 P ring in TMEM; the MXFP8 body picks its S issue order per
mask arm at load time (the lookahead on masked graphs, `K·Qᵀ` at the top of the
q iteration on dense ones — the same MMAs, bitwise identical, measured faster
each way). Rubin's 576 TMEM columns and 327 KiB SMEM
carveout are what let dV stay resident at d = 256 — the SM100 d512 backward is
a different, three-stage shape.

### The 2x2-datapath body (`kernels/bprop_d256_2x2_f16.py`; the SM100 row, and the Rubin twin)

The same chain has a second main kernel on the **2x2 tcgen05 datapath**:
`tcgen05.mma.cta_group::2` with the collective M = 128, i.e. 64 kv rows per CTA
per sub-block (the 4x1 body above is M = 256, 128 rows per CTA). A 64 x N fp32
accumulator then lands as row m -> TMEM lane `m + 64 * (n // (N/2))`, column
`n % (N/2)`: S and dP take 64 columns each, dV 128, so S / dP are double-buffered
and everything fits 512 non-exclusive TMEM columns, and every MMA operand is an
SMEM SS operand (K / V / P as 64-row 128-B-swizzled K-major slabs, Q / dO N-split,
dO_dv in the transposed BT form) -- no UTCCP K-split, no TMEM P alias. P is
lane-written into a 2-deep SMEM ring (each lane a 64-B half row at the swizzled
address) and published to the leader CTA's MMA with `fence.proxy.async` plus a
`.release.cta` arrive; every slot reuse (S, dP, P) is an explicit mbarrier. The
body has two profiles selected by `TemplateParams.datapath_2x2_profile`
(`bwd/config_d256_2x2.py`): profile 1 (one sub-block per CTA, a 128-row kv block
per pair, 210 KiB of SMEM, descriptor version 0) is the SM100 / SM103 row
`sdpa_bwd_sm100_d256` (`bwd/api_dsl_sm100_d256.py`), the footprint that fits
227 KiB / 512 columns; profile 2 (two sub-blocks per CTA, a 256-row block, 322 KiB,
descriptor version 1) is the Rubin twin behind the module constants
`api_dsl_sm107.BWD_D256_2X2` (default `False`: the shipped 4x1 rendering is
unchanged) and `BWD_D256_2X2_PROFILE` (2; 1 runs the SM100 body on Rubin, the A/B
arm). The register split is per profile: 176 / 152 on profile 1, 224 / 56 on
profile 2 (its 64-column compute lanes spill at 176). Both keep the 256-row kv WRITE PAIR: a 128-row block derives its q
range from the pair it belongs to, so the stage-3 GEMMs' K-trim and the
no-zero-fill contract are exactly the 4x1 chain's. The MMA issue order is a
config constant (`CfgBwdD256x2.MMA_LOOKAHEAD`): profile 1 ships the NATURAL order
(S(i), dP(i), BMM2(i) per q tile) -- on B200 it measured stage 2 at 3781 us against
4525 us for the lookahead order (S(i+1) between dP(i) and BMM2(i)) on the dense
B=1 H_q=32 H_kv=2 S=8192 bf16 shape, 2020 vs 1974 us causal (A/B/A x3 with a
control pair: the causal leg stays 1.4 % slower under NATURAL, the dense leg
11.8 % faster); profile 2 keeps the lookahead. Whole backward on that shape: 5673
us dense / 3256 us causal against the backend's engine 5 at 7265 / 3939 us (see
the tracker footnote). On the Rubin board (2026-10-01) the twin traced and ran on
both profiles -- fp64-oracle accepts, poisoned-workspace cases, two-launch bitwise,
and dQ / dK / dV BITWISE the 4x1 body's -- but measured slower than the 4x1 body
(profile 2 at 1.20x / 1.09x (causal stage 2 / whole) and 1.22x / 1.12x (dense); CUDA events, the 4x1 body as the in-session control), so
`BWD_D256_2X2` stays `False`.

### The d in (256, 512] row (`sdpa_bwd_sm107_d512`; `bwd/api_dsl_sm107_d512.py`)

Head dims above 256 are a different chain on this line too: the SM100
large-head-dim backward (`bwd/api_dsl.py::SdpaBwdDslSm100`, see the SM100 ᵇ
footnote of the support tracker) -- `delta = rowsum(dO·O)` -> a stage-2 kernel
that writes `S` and `dS` to `[B, H_chunk, S_q, S_kv]` GMEM workspaces (heads
chunked to a 4 GiB budget) -> `dV = Sᵀ·dO`, `dK = dSᵀ·Q`, `dQ = dS·K` as the
`bprop_matmul_blackwell` GEMMs at the (512, 512) cluster tile -> the GQA fold --
with stage 2 ALWAYS the **2x2-datapath** body `kernels/sm107/bprop_d512_f16_2x2.py`
at the cc 10.7 ring arm: two independent `tcgen05.mma.cta_group::2` pairs per
(4,1,1) cluster, 64 q rows per CTA, both BMMs (`Q·Kᵀ`, `dO·Vᵀ`) as SMEM SS
operands with d streamed in 64-column chunks, an **8-stage K/V chunk ring** (the
SM100 body runs 4) and **two cast stages** (SM100: 1) filling 320 of the line's
325 KiB usable SMEM, 256 TMEM columns, tcgen05 descriptor version 0 at zero
margin.  The file is the SM100 twin's sibling (`diff sm100/ sm107/` is the review
surface; its rendering at these parameters is PTX-identical, pinned by a
committed md5 record -- a board-only pin: the public 4.7.0 DSL has no sm_107a
target, so the md5 and SASS cases skip in public CI and the fork is held there by
its code-diff allowlist, its source pins and its import-time `_require`s), and the
row is its own `EngineSpec` rather than a widened
`sdpa_bwd_sm100` because the 4x1 role split that row renders by default never ran
on this line.  Served: d in (256, 512] in multiples of 8 (envelope-served on
512-wide tiles: d = 264 pays d = 512's MMA; the floor is exclusive at 256, which the
d256 rows above own), any S_q / S_kv (padded to 256 / 128 and masked), dense,
top-left and bottom-right causal, right-band widening, sliding window (left),
MHA / GQA / MQA, BSHD-physical io, contiguous fp32 Stats, bf16 / fp16.  Declined
on day one (each asserted by `test_sdpa_bwd_d512_sm107.py` and flipped only with
a board-run accept + tracker line): THD / ragged, `dense_flex` layouts, dense
padding masks, sink / dSink, bias / dBias, deterministic, decode shapes.  Pin it by
`startswith("sdpa_bwd_sm107_d512")`; there is no backend d > 256 backward plan on
cc 10.7 to fall back to.

### Masks

The main kernel bounds WHICH q tiles a KV block attends (causal: from the
diagonal; sliding window: up to the window), rounds that range outward to a
256-row q pair, and zeroes P on masked cells, so dV, dS, dK and dQ inherit the
mask. The stage-3 GEMMs render a TWO-SIDED K-trim over the same band (dK starts
at the block's first attended q pair and ends after the window; dQ starts at the
window and ends after the last attended kv block), so each GEMM multiplies only
the band's tiles — under a sliding window that is `~(W + 256) / S` of the dense
K range instead of all of it. The pair rounding is what makes the trim
self-contained: every tile a GEMM reads was written by the kernel (masked cells
as stored zeros), so the workspace needs no zero-fill under any mask; the one
exception the adapter still fills for is a top-left window with
`S_q > roundup(S_kv + W, 256)` (q rows past every block's window, written by no
block). Poisoned-workspace tests pin this per mask, and a bitwise pin against the
untrimmed rendering keeps the trim numerically inert.

### Sequence lengths

The kernels compile with every extent concrete (`S_q % 128 == 0`,
`S_kv % 256 == 0`). Any other length is served by padding: Q / dO and the LSE
(with `+inf`, so `P = 0`) are staged into zero-filled padded copies, K / V
likewise, and a padded S_kv selects the kernels' padded-mask specialization at
the uniform real length, so every padded kv row's dS / dV is exactly zero. The
GEMMs read real-extent slices and write the caller's tensors directly.

Per-batch KV lengths are served on the **standalone surface** of all three rows:
construct the adapter (`SdpaBwdDslSm107`, `SdpaBwdDslSm107Fp8` or
`SdpaBwdDslSm107Mxfp8`) with `seq_kv_lens_present=True` and pass `seq_kv_lens` (a
contiguous `[B]` int32 device tensor) to `execute`. The same padded-mask specialization
then reads `seq_kv_lens[b]` in place of the uniform length: every kv row at or past its
batch's length is select-dead (dS = dV = 0 exactly), a length of 0 is a dead batch whose
dQ / dK / dV are exact zeros whatever its Stats rows hold (0 or `-inf`) — and on the fp8
row the amax row gate follows the same per-batch length, so a dead or shortened batch
folds nothing into `amax_dV` / `amax_dP` — and under bottom-right causal the diagonal is
per batch (`seq_kv_lens[b] − S_q`) while the GEMMs' K-trim is computed from the uniform
`S_kv − S_q` — so for that arm the chain keeps the dS workspace zero-fill (ahead of every
batch / head chunk on the batch-chunked bf16 row, where a chunk's workspace slot may hold
the previous batch's dS in tiles the next batch's narrower band does not write; once per
execute on the fp8 / MXFP8 rows, which walk the whole batch in-grid) and drops a sliding
window from the stage-3 trim (a window edge anchored on the uniform diagonal would skip
live tiles of a shorter batch): the GEMMs read the plain bottom-right band there. Two
contract points are device data the host does not validate: every entry must satisfy
`0 <= seq_kv_lens[b] <= S_kv`, and **the K / V rows at or past a batch's length must hold
finite data** (on the MXFP8 row their scale-factor atoms too: an E8M0 NaN byte past the
length is a NaN dP) — the kernels select P = 0 on them, but dS is `(dP − delta) ∘ P` and
`NaN × 0 = NaN`; the dense rows get finite pads from the adapter's zero-filled staging
copies, the per-batch arm reads the caller's buffers as they are (finite garbage past
the length is fine, a NaN is not). The **graph** padding mask stays declined on every
row: a padded `sdpa_backward` graph carries `seq_len_q` as well as `seq_len_kv` (the
frontend requires both) and no body threads per-batch Q lengths, so serving the graph
form would mean ignoring the q lengths.

An externally computed `delta` is the other plan fact of that standalone surface, on
every row, dense and THD: `external_delta=True` at construction declares that the caller
computes stage 1's `delta = rowsum(dO ∘ O)` and hands it to `execute(..., delta_tensor=)`
— an fp32 contiguous tensor of `external_delta_shape`, 16-byte aligned, on the plan's
device, holding the raw row dot (`attn_scale` is applied in the main kernel): on a dense
plan `(B, H_q, S_q_pad)` (`S_q_pad` = `S_q` rounded up to the 128-row q tile, **zeros past
`S_q`**); on a THD plan the PACKED head-major `(1, H_q, ceil128(T_q))` the packed chain
reads at the packed token index (`T_q` = the plan's token capacity, the declared
`max_total_seq_len_q` tightened to `B · S_max`; **zeros past `T_q`**) — the dense layout
at `B = 1, S = T_q`, so one producer serves both forms. The chain
then launches no `dot` and reads O once less, and the workspace carve has no `delta`
region (`scratch_workspace_bytes()` shrinks by exactly it); the operand is checked before
any bind — dtype, shape, strides, device, alignment, each a typed `ValueError` — and a
plan built without the flag refuses a delta. The producer this exists for is the gated
attention block's sigmoid-gate backward kernel, which forms the delta over the `dO` it
stores in `dot`'s own reduction order, so the fused and the unfused block backward are
bitwise equal (`fuse_gate_bwd` in [gated_attention_block.md](../gated_attention_block.md)).
The flag is independent of `seq_kv_lens_present`: each decides its own appended operand
(the lengths, then the delta) and a plan built with both takes both at `execute`. It is a
plan fact, not an eligibility fact — `Capabilities` and the graph path are unchanged (no
graph declares a delta, so a graph plan keeps the chain's own `dot` launch and its
region). The units are the row's: the bf16 / fp16 and MXFP8 rows read the raw
half-precision dot (the MXFP8 row's own pre-pass is `dot` over its `o_f16` / `dO_f16`
ports, so a producer forming it in that order is bitwise the chain's own); **the fp8 row
reads `delta` in true units, unscaled** — its own pre-pass is the row sum of the e4m3
payload codes times `descale_o · descale_dO`, nobody applies those descales to a caller's
delta, so a delta formed from the bf16 O / dO binds as is and is not bitwise the row's
own pre-pass (its tests compare against an oracle fed the same delta). The pad rows
`[S_q, S_q_pad)` must be finite zeros on every row: the kernels read them, and under the
MXFP8 row's block-scaled dS a 32-element q block straddling the pad folds them into the
real columns' E8M0 scale. The `o` / `descale_o` (fp8) and `o_f16` / `dO_f16` (MXFP8)
operands stay required under the flag and are read by nothing. Under THD the same
contract holds in the packed form above: the chain reads the caller's tensor where it
read its own `delta` region, launches no `dot` and carves no region (the carve shrinks by
exactly `H_q · ceil128(T_q)` fp32 values); a producer that forms the delta in `dot`'s
order over the packed O / dO is bitwise the chain's own on the bf16 / fp16 and MXFP8 rows,
and the fp8 row again reads it in true units.

Bottom-right causal at a ragged `S_q` is served on all three rows: every body takes the
real lengths (`seqlen_q_real` / `seqlen_kv_real`) and derives the diagonal `S_kv − S_q`
and the q-tile trim from them, never from the padded compile extent
(`Capabilities.bottom_right_s_q_multiple = 1` on every row).

### THD / ragged (packed varlen)

`sdpa_bwd_sm107` (bf16 / fp16), `sdpa_bwd_sm107_fp8` (per-tensor E4M3) and
`sdpa_bwd_sm107_mxfp8` (block-scale MXFP8) serve a
**ragged** backward: Q/K/V/O/dO and dQ/dK/dV declared as the envelope `(B, H, S_max, D)`
with a per-tensor ragged offset over PACKED storage (`[1, T, H, D]` rows: element stride
1, head stride D, token stride >= H·D and a multiple of 8 elements), `use_padding_mask=True`
with the per-sequence `seq_len_q` / `seq_len_kv` as `(B,)` int32 tensors, and BOTH
`max_total_seq_len_q` / `max_total_seq_len_kv` declared (the packed workspace is sized
from them at build time; a graph without them is declined at plan creation — on the
quantized rows the `sdpa_fp8_backward` / `sdpa_mxfp8_backward` nodes and their bindings
carry the two attributes as trailing keywords, and a pybind extension built before them
cannot declare them, so a ragged fp8 / MXFP8 graph through such an extension is that typed
decline while the standalone surfaces serve). Stats is the forward's packed Stats in
either layout the forward emits -- token-major `(T, H)` or head-major
`(1, H, head_stride)` with `head_stride >= T`.
The standalone surface is `SdpaBwdDslSm107(..., thd=True, max_total_seq_len_q=..,
max_total_seq_len_kv=.., thd_stats_token_major=.., thd_stats_head_stride=..)` -- or
`SdpaBwdDslSm107Fp8(...)` with the same keywords plus `amax_requested`, or
`SdpaBwdDslSm107Mxfp8(...)` with the same keywords plus its MXFP8 sample operands -- with
`execute(seq_q_lens=.., seq_kv_lens=..)` taking `(B,)` lengths or `(B+1,)` prefix sums
per side (the fp8 row's twelve scalars and requested amax tensors, the MXFP8 row's payloads
and scale-factor tensors, as on their dense surfaces); `thd_stats_head_stride` (head-major only) is required when the Stats buffer's
head stride is not exactly the packed capacity -- the FROST forwards emit
`(1, H, ceil64(T))` -- and must cover the packed total (the graph path infers it from the
ragged strides).

How it runs: one setup launch builds the metadata on device (no host cumsum), the dS
workspace is blocked over packed KV tokens (each sequence owns a 256-row-aligned block),
the main kernel claims (sequence, kv block, head) units from a device counter, reads the
packed operands through descriptors clamped to the live packed totals (a NaN in the
declared-but-unused capacity tail cannot reach an MMA), masks each sequence's kv tail and
q pad columns from its own lengths (bottom-right's diagonal is `S_kv[b] − S_q[b]` per
sequence) and stores dV through per-sequence clipped descriptors; the gradient GEMMs run
over the blocked rows with per-sequence output descriptors and the same two-sided K-trim
as the dense path, PER SEQUENCE (every bound derived from the sequence's own lengths and its
own diagonal, so a GEMM reads only dS tiles the kernel wrote for that sequence and a tile
whose band is empty -- a kv block no query attends, a q pair with no key -- is stored as
exact zeros without a read), so the workspace needs no zero-fill under any mask; under GQA
the dQ GEMM runs once per head chunk over the packed K heads, as on the dense path. Nothing past the
packed totals is written into the caller's gradients: dQ / dK / dV stop at the
per-sequence clipped output descriptors and the GQA fold stops at the live kv total on
device (the rows up to the declared capacity keep whatever the caller left there). A unit
without query rows -- an empty-Q sequence, a spare unit of the occupancy-sized grid --
loads every operand past the clamped extent (zero-filled), so even an all-NaN Q / dO
capacity with no live query row yields exact-zero dK / dV. Served under THD:
none / causal / bottom-right / sliding window, GQA / MQA, empty sequences on either side
(their gradients are exact zeros), an external `delta` in its packed head-major form
(see Sequence lengths above). Declined under THD: right-band widening, bias.

On the fp8 row the same mechanism runs in e4m3: packed e4m3 payloads through the
packed-total-clamped descriptors, a kv-blocked **e4m3** dS workspace (`dS_q = e4m3(dS ·
scale_dP)`; bf16 on the `FP8_DS_DTYPE = DTYPE_BF16` twin, which needs compact packed rows —
token stride exactly H·D — for its exact e4m3 → bf16 upcast of Q / K), the fp8 K64 gradient
GEMMs trimmed per sequence with their descale / quantize epilogue, and the fold + quantize
passes bounded on device at the live totals (`cu_k[B]` for dV / dK, `cu_q[B]` for the twin's
dQ), so nothing past the packed totals is written into the caller's gradients. **Every amax
is the max over the packed live region** — dead units, kv pad rows, q pad columns and the
capacity tail are excluded at all four fold sites (the main kernel's row gate and fold
values, the GEMM epilogue's per-row gate, the bounded fold passes) — and there is one
`scale_dP` per packed batch, the forward's one-scalar-per-operand convention over packed
tokens.

On the MXFP8 row the same mechanism runs over the packed e4m3 payloads (`q / k / v / dO`
and the transposed-quantization `q_T / k_T / dO_T`, all packed `[1, T, H, D]` rows) with
one contract the other rows do not have: **the seven scale-factor tensors travel PACKED
per-sequence-TILE-padded**, the forward's convention. Per head, every sequence's
`ceil(s_b / 128)` F8_128x4 tiles follow each other in cu_seqlens order — sequence `b`'s
tiles start at `cu_sf[b] = Σ_{i<b} ceil(s_i / 128)`, *not* at `cu[b] / 128` — 1024 bytes
per (head, tile) at d = 256: rowwise (`descale_q / k / v / dO`) the tile's 128 rows x 8
groups; columnwise (`descale_q_T / k_T / dO_T`) BOTH D planes of the (head, tile)
contiguous (plane stride one 512-byte atom, tile stride the whole slab — the dense
tensors are D-plane-major over the whole tensor, and reading a packed tensor through that
layout fetches plane 1 from the wrong place by an S-dependent offset). The packed tile
count is a **per-call** fact derived from the bound buffer's byte size: whole
`H x 1024`-byte tile rows, one count per side (`descale_q / q_T / dO / dO_T` and
`descale_k / k_T / v` must each agree), at least `Σ_b ceil(s_b / 128)` tiles per head (the
scale-factor maps' tile extent is the live total; a shorter buffer is read past its end —
device data the host cannot check) and at most the plan's capacity, the larger of
`ceil(max_total_seq_len / 128) + B` tiles per head and the declared sample's own count — the
whole-row, one-count-per-side and capacity rules each a typed `ValueError` at bind; so the
graph may declare any dims with the right byte total, the dense capacity
`B × ceil(S_max / 128)` included (the forward's graph layout), and bind a buffer of exactly
those bytes. **The producer's pad
bytes may hold anything** (a `0xFF` is an E8M0 NaN): the chain re-stages the five scale
tensors whose pad positions are read — `descale_v / dO / dO_T` by the main kernel's dP / dV
operands and, under P-b, `descale_q_T / k_T` by the block-scale gradient GEMMs (whole atoms)
— into packed staging copies with every byte scaling a position at or past its sequence's
length zeroed, per execute, from the device prefixes; `descale_q / k` pads are harmless
(an S NaN is select-dead) and bind as they are. Both dS policies serve THD: P-c runs the
bf16 THD gradient GEMMs over the packed `q_T / k_T` dequantized exactly to bf16 per token
(no pad byte is read), P-b the block-scale arm's THD leg (the kv-blocked payloads + atoms,
B's scale factors through the per-sequence SF tile prefixes) with dQ once per head chunk
under GQA, as on the dense P-b chain: the dQ record's `b_head_group` is the GQA group, so
B and its scale factors are indexed by `h // group` (the SF tile prefix is a token-side
term) and one launch covers the whole head chunk, bitwise the per-member launches. No
amax (the row's contract); Stats comes from the caller — no Rubin MXFP8 THD forward row
feeds it yet.

### FP8 numerics (`sdpa_bwd_sm107_fp8`)

Every scalar is a 1-element fp32 device tensor read in-kernel, never a host
fold. `S = S_acc · descale_q · descale_k`; `P` is recomputed from the forward's
exact fp32 Stats and quantized to E4M3 with `scale_s` for the dV MMA;
`dV = dV_acc · descale_s · descale_dO`; `dP = dP_acc · descale_v · descale_dO`;
`dS = attn_scale · P ∘ (dP − delta)` in fp32 (`amax_dP` is its amax before the
scale and the cast, as the C++ node reduces it), then `dS_q = e4m3(dS · scale_dP)`
into an **E4M3** workspace — the cuDNN recipe (delayed scaling: feed the previous
step's `amax_dP`). The gradient GEMMs run Rubin's dense-FP8 K64 MMA over the
e4m3 dS and the e4m3 Q / K payloads and undo both scalings in their epilogue:
`acc · descale_dP · descale_k` (dQ) / `· descale_q` (dK), then `amax_dQ` / `amax_dK`
over that true-unit value, `· scale_dQ` / `scale_dK` and the cast to the graph's
gradient dtype — written straight into dQ, and into dK at MHA. Under GQA the dK
partials leave the GEMM in **fp32** (the true-unit value, `EPI_DESCALE`) and the
main kernel stores its per-Q-head dV partials in fp32 too (`dtype_o = FP32`); one
fold launch sums each KV head's group in fixed order and only then folds `amax_dK`
/ `amax_dV`, applies `scale_dK` / `scale_dV` and casts — the gradient is rounded
once, like the reference (a bf16 partial would round it a second time). dV always
takes that fold launch (bf16 partials at MHA, where the fold is a copy + amax,
scale and cast); dK joins it under GQA, so the e4m3 chain runs ONE fold + quantize
launch and no fold at all for dQ (quantized in its GEMM epilogue), as the kernel
chain above lists it. `api_dsl_sm107.FP8_DS_DTYPE = DTYPE_BF16` selects the
pre-quantized twin used for A/B and oracle work: bf16 dS, bf16 GEMMs over exact
E4M3 → bf16 upcasts of Q / K, bf16 partials, its three gradients folded + quantized
in two launches (dV + dK in one, then dQ), `descale_dP` / `scale_dP` bound and
unused.

### MXFP8 numerics (`sdpa_bwd_sm107_mxfp8`)

The block-scale row is the same graph through `g.sdpa_mxfp8_backward(...)` —
E4M3 `q / k / v / dO` plus the transposed-quantization payloads `q_T / k_T / dO_T`
(each contraction axis needs its own 1x32 quantization), bf16 `o_f16 / dO_f16`,
fp32 `stats`, and the seven `F8_128x4`-reordered E8M0 scale tensors
(`descale_q / q_T / k / k_T / v / dO / dO_T`; `descale_v` is the **rowwise** V
scale, as in the C++ node's own reference math) — with `dQ / dK / dV` declared
bf16. **One kernel plus scale-factor plumbing** (`sm107/bprop_d256_mxfp8.py`, the
fp8 body's pipeline): the scale factors ride their operands' TMA barriers into
TMEM and dequantize inside every tcgen05 block-scale MMA, so `S` and `dP` are in
TRUE units; `P` is recomputed from the exact fp32 Stats and quantized to E4M3
with the fixed `2^8` scale (byte 119, cuDNN's MXFP8 convention; `p_scale_log2` is
pinned to 8) into the fp8 body's two-slot TMEM P ring (the scale factors of each q
iteration ride along in the slot that is dead that iteration) for `dV += P · dO_T`; `dS = attn_scale · P ∘ (dP −
delta)` is formed from the **fp32** P (never the e4m3 P — pinned by
`test_ds_is_computed_from_the_fp32_p_not_the_e4m3_p`).

**dS policy: P-b ships** (`config_sm107.DS_SF_POLICY_DEFAULT = DS_SF_P_B`, read by the
adapter constant `api_dsl_sm107.MXFP8_DS_SF_POLICY` when the adapter is built — a
module constant, never a knob): dS is quantized to e4m3 per 32-element block in **both**
orientations — the kernel writes two payloads (`ds_dk` scaled per 32-q block of a kv row,
`ds_dq` per 32-kv block of a q column) plus their F8_128x4 E8M0 atoms, and the gradient
GEMMs render `MatmulTemplateParams.block_scale` over them and the columnwise
`q_T / k_T` scale factors, dequantizing in the MMA with no dequant pass (Rubin-line
only). Accept cells run the oracle at `quantize_ds=True` (its own 1x32 quantization of
the fp32 dS) under the fp8 recipe on dV, dK and dQ alike: both sides round dS to e4m3
per block from fp32 values that agree to ~1e-6, so a midpoint flip of one dS cell moves
one gradient row by one e4m3 step at the block's scale — the fp8 row's dS-flip class.
P-c — dS through a **bf16** workspace into the bf16 renderings over the *exactly*
dequantized bf16 `q_T / k_T` (`prepared_host.dequant_mxfp8_to_bf16_host`: `e4m3 ×
2^(e−127)` is a bf16 value), so dQ / dK carry no dS quantization at all, more accurate
than cuDNN's 1x32-quantized dS and the oracle twin (`sdpa.mxfp8_ref.compute_ref_backward(...,
quantize_ds=False)`) — stays built and selectable through
`api_dsl_sm107.MXFP8_DS_SF_POLICY = DS_SF_P_C`, with the bf16 row's recipe on dK / dQ.
The flip was a numerics change (the accept matrix re-run on Rubin, the support-matrix
cell re-written), decided on the measured A/B: on Rubin (cc 10.7, 212 SMs, SM clock
2376 MHz) at B=1 H=128/128 S=8192 the whole backward is +22.0 % (dense) / +17.1 %
(causal) faster than the bf16-dS chain; the block-scaled chain chunks its stage-2
workspace against the rows' shared 8 GiB budget (32-head chunks at 8K H=128) for
another +1.4 % / +9.3 % over a 4 GiB budget's 16-head chunks (the control twin's
value; the arm read +11.6 %).

Padding on this row has one obligation the other rows do not: the producer's
scale-factor tensors cover `ceil128(S)` rows / groups and their pad bytes are
undefined, while the kernel **reads** the pad positions (a `0xFF` there is an
E8M0 NaN → `0 × NaN` in `S` on the Q side, NaN in the dead rows' dS on the kv
side). So the artifact re-stages `descale_q / dO / dO_T` (when `S_q % 128 != 0`)
and `descale_k / v` (when `S_kv % 256 != 0`, grown to the kernel's 256-row pad)
with every pad byte zeroed, next to the zero-padded payload copies.
Poisoned-pad RED-then-green tests pin it. Under THD the same obligation is per
SEQUENCE: the scale tensors are packed per-sequence-tile-padded, and the chain re-stages
`descale_v / dO / dO_T` (and `descale_q_T / k_T` under P-b) with every byte past each
sequence's length zeroed from the device prefixes (see THD above).

Under GQA the MXFP8 SDPA backward folds its per-Q-head dK partials in fp32 and
rounds the sum once, like the reference, while its per-Q-head dV partials are bf16
(the kernel stores them from its epilogue; fp32 ones do not fit its 327 KiB
shared-memory budget), so dV carries one bf16 rounding per group member where a
once-rounded reference carries one in total (relative RMS about 3e-3 at a group of
4, the geometry the tests run, measured on the per-tensor fp8 row before it moved to
fp32 partials); the modelled oracle folds dV the same way and the distance to a
once-rounded fold is reported per cell. The dK fold is pinned bitwise: the row's dK
`torch.equal`s the fixed-order fp32 sum of its own fp32 partials rounded once
(`test_mxfp8_gqa_dk_is_the_once_rounded_fold_of_its_fp32_partials`); MHA is
untouched (the GEMM writes the caller's bf16 dK, the same kernels and bits).

Not produced: the `amax_dQ / dK / dV` outputs — a graph that marks them real
(the backend's canonical MXFP8 backward shape) is declined, typed. This row is
the sole provider of the d = 256 MXFP8 backward on Rubin (cuDNN 9.27 has no such
kernel on smVersion 1070), so that graph gets `cudnnGraphNotSupportedError` at
plan creation.

## Support surface and constraints

- SM107-line devices (cc 10.7 – 11.9); `sdpa_bwd_sm100_d256` on SM100 / SM103 (cc
  10.0 – 10.6, bf16 / fp16 only, the 2x2-datapath body)
- Head dims: `d_qk = d_v = 256` exactly (no envelope)
- Dtypes: bf16 / fp16 (`sdpa_bwd_sm107`); E4M3 payloads with E4M3, bf16 or fp16
  gradients (`sdpa_bwd_sm107_fp8`; E5M2 is declined); E4M3 payloads with
  F8_128x4 E8M0 scales and **bf16** `o_f16 / dO_f16 / dQ / dK / dV`
  (`sdpa_bwd_sm107_mxfp8`; fp16 gradients are declined — the bf16 GEMM stores its
  io dtype, an fp16 cast pass is a follow-up). Stats fp32, contiguous
  `(B, H_q, S_q, 1)`
- Layout: BSHD-physical Q/K/V/O/dO/dQ/dK/dV (and `q_T / k_T / dO_T / dO_f16`;
  stride order 3,1,2,0); packed BSHD rows under THD on every row (the MXFP8 row's
  scale-factor tensors packed per-sequence-tile-padded)
- Masks: none, causal (top-left or bottom-right), sliding window (left,
  with or without causal); any S_q / S_kv on every row
- GQA/MQA: any `H_kv` dividing `H_q`
- Declined (asserted by tests): graph padding masks (`seq_len_q/kv` — a padded
  graph carries both lengths and no body threads per-batch Q lengths; per-batch KV
  lengths are served on every row's standalone adapter, see Sequence lengths; a RAGGED
  padded graph is THD and served on `sdpa_bwd_sm107`, `sdpa_bwd_sm107_fp8` and
  `sdpa_bwd_sm107_mxfp8`), sink / dSink, bias / dBias, right-band widening,
  `dense_flex` layouts,
  decode shapes (`S_q == 1`), `use_deterministic_algorithm` (the chains have no atomics;
  the claim waits on the bring-up sweep), dropout / ALiBi / softcap; on the MXFP8
  row also the `amax_dQ / dK / dV` outputs, fp16 gradients and any
  `p_scale_log2 != 8`
- Workspace (carved from the caller's buffer): fp32 `delta` (not carved under the
  standalone adapters' `external_delta`), one head/batch
  chunk of the dS workspace (`B_chunk · H_chunk · S_kv · S_q` bytes at e4m3 on
  the fp8 row, `· 2` on the half and MXFP8 rows), padded staging copies when
  S_q / S_kv are not tile multiples, per-Q-head dK/dV partials under GQA; the
  fp8 row adds the per-Q-head dV partials (fp32 under GQA, bf16 at MHA) and,
  under GQA, the fp32 dK partials, plus an amax scratch; the MXFP8 row adds the
  block-scaled dS chain's second e4m3 payload and two scale-factor atom tensors
  (or, on its bf16-dS twin, the two dequantized bf16 `q_T / k_T` slabs) and the
  zero-filled scale-factor pad slabs, and carves its GQA dK partials fp32 (its dV
  partials bf16; under THD: the packed scale-factor staging copies at the plan's
  tile capacity and the per-sequence SF tile prefixes). Use
  `graph.get_workspace_size()`.
