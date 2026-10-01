# SDPA Backward, d = 256 (SM107 / Rubin)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** pass for head dimension 256 on the NVIDIA Rubin line
(`SM107`, cc 10.7 through 11.9), implemented with CuTe DSL primitives:
tcgen05 MMAs with the accumulator resident in TMEM, TMA loads and stores, a
2-CTA cluster per KV block and a warp-specialized schedule. It consumes the
forward activations (`Q/K/V/O`), the loss gradient `dO` and the forward `Stats`
(natural-log LSE) and produces `dQ/dK/dV`.

Two FROST engines serve it (`cudnn.sdpa.bwd.engines`), both `opt_in` — set
`CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1` before `import cudnn`, and pin the
engine from the ranked plan list (`graph.plans` / `graph.select_plan(i)`) when
validating or measuring, because the bf16 d256 graph also has a native backend
plan:

* `sdpa_bwd_sm107` — bf16 / fp16 `sdpa_backward()` graphs;
* `sdpa_bwd_sm107_fp8` — per-tensor FP8 E4M3 `sdpa_fp8_backward()` graphs
  (cuDNN's contract: scalar descales in, FP8 or half gradients plus the
  `amax_dQ/dK/dV/dP` outputs).

There is no standalone wrapper for this pass yet; the graph API is the surface.

The kernels live in `python/cudnn/sdpa/bwd/kernels/sm107/`
(`bprop_d256_f16.py`, `bprop_d256_fp8.py`; config `bwd/config_sm107.py`), the
adapters in `python/cudnn/sdpa/bwd/api_dsl_sm107.py`, and the rest of the launch
chain is shared with the other Blackwell-line backward engines
(`kernels/bprop_matmul_blackwell.py`, `kernels/bprop_chain_common.py`).

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
                             amax_dK at MHA, or bf16 true-unit per-Q-head partials under GQA)
mm_dq   dQ = dSᵀ · K         same GEMM, the other operand major (fp8: · descale_dP · descale_k,
                             amax_dQ, · scale_dQ → the gradient dtype, straight into dQ)
fold    GQA only (half row): dK/dV = fixed-order sum of each KV head's group of
        per-Q-head partials.  fp8 row: dV always (fold + amax_dV + scale_dV + cast);
        dK under GQA (the bf16 partials are summed BEFORE the amax, scale and cast)
```

The workspace is head-chunked (and batch-chunked on the half row) to a 4 GiB
budget; the artifact's host loops over the chunks with `head_base` / `batch_base`,
so one compiled artifact serves every launch of a plan.

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
and a two-slot fp8 P ring in TMEM. Rubin's 576 TMEM columns and 327 KiB SMEM
carveout are what let dV stay resident at d = 256 — the SM100 d512 backward is
a different, three-stage shape.

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

One exception on the fp8 row: **bottom-right causal needs `S_q % 128 == 0`**
(declined otherwise, at plan build, as not supported). The bottom-right
diagonal is `S_kv − S_q` in real rows; the f16 kernel takes the real lengths,
the fp8 kernel derives the diagonal from its padded q extent (its kv term is
the real length), so a ragged S_q would shift it. A ragged S_kv under
bottom-right is served on both rows.

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
gradient dtype — written straight into dQ, and into dK at MHA; under GQA the dK
partials leave the GEMM in bf16 (true units) and the fold pass sums them in
fixed order before it folds `amax_dK`, applies `scale_dK` and casts. dV always
takes the fold pass (`amax_dV`, `scale_dV`, cast). `api_dsl_sm107.FP8_DS_DTYPE =
DTYPE_BF16` selects the pre-quantized twin used for A/B and oracle work: bf16 dS,
bf16 GEMMs over exact E4M3 → bf16 upcasts of Q / K, three fold + quantize passes,
`descale_dP` / `scale_dP` bound and unused.

## Support surface and constraints

- SM107-line devices (cc 10.7 – 11.9)
- Head dims: `d_qk = d_v = 256` exactly (no envelope)
- Dtypes: bf16 / fp16 (`sdpa_bwd_sm107`); E4M3 payloads with E4M3, bf16 or fp16
  gradients (`sdpa_bwd_sm107_fp8`; E5M2 is declined). Stats fp32, contiguous
  `(B, H_q, S_q, 1)`
- Layout: BSHD-physical Q/K/V/O/dO/dQ/dK/dV (stride order 3,1,2,0)
- Masks: none, causal (top-left or bottom-right), sliding window (left,
  with or without causal); any S_q / S_kv — except bottom-right on
  `sdpa_bwd_sm107_fp8`, which needs `S_q % 128 == 0` (see above)
- GQA/MQA: any `H_kv` dividing `H_q`
- Declined (asserted by tests): dense padding masks (`seq_len_q/kv`), sink /
  dSink, bias / dBias, right-band widening, THD, `dense_flex` layouts, decode
  shapes (`S_q == 1`), `use_deterministic_algorithm` (the chain has no atomics;
  the claim waits on the bring-up sweep), dropout / ALiBi / softcap
- Workspace (carved from the caller's buffer): fp32 `delta`, one head/batch
  chunk of the dS workspace (`B_chunk · H_chunk · S_kv · S_q` bytes at e4m3 on
  the fp8 row, `· 2` on the half row), padded staging copies when S_q / S_kv are
  not tile multiples, per-Q-head dK/dV partials under GQA; the fp8 row adds the
  bf16 dV partials (and dK partials under GQA) and an amax scratch. Use
  `graph.get_workspace_size()`.
