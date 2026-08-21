# SDPA Backward (SM120)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** pass for the NVIDIA Blackwell GeForce line (`SM120` /
`SM121`: RTX 50-series, RTX PRO 6000 Blackwell, DGX Spark), implemented with
CuTe DSL primitives using TMA loads and a warp-specialized producer/consumer
schedule. Consumes the forward activations (`Q/K/V/O`), the loss gradient
`dO`, and the forward `LSE`; produces `dQ/dK/dV`.

Two integration surfaces are provided:

* a standalone wrapper (documented below), `cudnn.sdpa_bwd_wrapper_dsl_sm120`, and
* a FROST engine (`sdpa_bwd_sm120`, see `cudnn.sdpa.bwd.engines`) that serves
  single-node `sdpa_backward()` graphs built with `cudnn.pygraph` when
  selected from the ranked plan list (`graph.plans` /
  `graph.select_plan(i)`) with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`.

The kernel lives at `python/cudnn/sdpa/bwd/kernels/bprop_f16_sm120.py`
(one fused five-GEMM main kernel, plus a `dot` preprocess and a `cvt`
dQ-finalize kernel per call; GQA/MQA adds a fourth `reduce` kernel that sums
the per-q-head dK/dV partials).

## Requirements

The `cutedsl` optional dependency (`nvidia-cutlass-dsl` + `apache-tvm-ffi`)
and an SM120 or SM121 device.

## API Usage

```python
from cudnn.sdpa.bwd import sdpa_bwd_wrapper_dsl_sm120

grads = sdpa_bwd_wrapper_dsl_sm120(
    q_tensor=q, k_tensor=k, v_tensor=v,
    o_tensor=o, do_tensor=do, stats_tensor=stats,  # from the forward pass
    is_causal=True,
    causal_bottom_right=False,
    window_size_left=None,   # W: keys with k < q + diag - W are masked
    window_size_right=None,  # R: widen the causal diagonal right by R keys
                             # (keep k <= q + diag + R
    deterministic=False,     # ordered dQ KV-tile reduction (bitwise-reproducible)
    scale_softmax=None,      # None -> 1/sqrt(D)
    seq_q_lens=None,         # (B,) int32 per-batch Q lengths (padding mask)
    seq_kv_lens=None,        # (B,) int32 per-batch KV lengths (padding mask)
    sink_token=None,         # fp32 (1, H_q, 1, 1) sink logits; adds dsink_tensor
                             # to the result
)
dq, dk, dv = grads["dq_tensor"], grads["dk_tensor"], grads["dv_tensor"]
```

Tensors are logical `(B, H, S, D)`. Any dense layout with the head dim
innermost-contiguous is accepted (`dense_flex`) and addressed in place: the
declared strides bake into the kernel. TMA sets the limits — batch/seq/head
strides must be 16-byte multiples and each base address 16-byte aligned;
anything else is declined. Head dims that pad to the next supported size
are also served in place: the TMA descriptors declare the actual extents
and reads past them zero-fill in hardware. `stats` is the natural-log forward LSE, fp32
`(B, H, S_q, 1)`; any non-broadcast layout serves, its strides baked in the
same way (scalar loads, so no 16-byte rule). GQA/MQA is expressed through
the head counts: `H_kv` may be any divisor of `H_q` (K/V and dK/dV carry
`H_kv` heads).

Through the graph API, per-plan sequence-tile-width knobs can be requested via
`SdpaBwdKnobs`: `tile_m` controls the Q tile (`q_tile`), and `tile_n` controls
the KV tile (`kv_tile`).

## Determinism

By default the dQ accumulation across KV tiles uses fp32 atomics, so the result
can be bitwise non-deterministic when a q-tile receives contributions from
multiple KV-tile CTAs. `deterministic=True` serializes the per-`(batch, head,
q_tile)` additions in ascending KV-tile order through a GMEM turn-counter
array (the FlashAttention-style ordered-reduction relay). dK/dV are
deterministic in both modes — including under GQA, where the group reduction
runs in a fixed q-head order. This maps to the graph API's
`use_deterministic_algorithm` and costs a shape-dependent slowdown of the main
kernel while keeping the workspace linear in sequence length.

## Kernel design and optimizations

### The kernel chain

One backward call is three launches (four under GQA), overlapped with
programmatic dependent launch (PDL) so each kernel's prologue runs under its
predecessor's tail:

```text
dot     delta = rowsum(O ∘ dO); zeroes dq_accum (and, when deterministic, the relay counters)
main    the fused five-GEMM pass; writes dK/dV into dk_ws/dv_ws (aliased to the dk/dv
        outputs for MHA, per-q-head partial buffers for GQA); accumulates dQ into dq_accum
reduce  GQA only: dK/dV = fixed-order sum of each KV head's group of q-head partials
cvt     dq_accum (fp32, scrambled) -> dQ (io dtype), applying attn_scale
dsink   dSink_token graphs only, summing over every batch b and query row q:
        dsink[h] = -sum_{b,q} exp(sink[h] - LSE[b,h,q]) * delta[b,h,q]
```

### Main-kernel pipeline: KV-stationary, five chained GEMMs

Grid is `(num_kv_tiles, H_q, B)` — one CTA owns one KV tile of one **query**
head (its KV head is `q_head // group`), loads K/V **once**, and walks every
q-tile of its (batch, head) in descending order. Per q-tile iteration:

```text
GEMM1  S  = Q · Kᵀ            (K streamed from SMEM)
       P  = exp2((scale·S − LSE) · log2(e))   replay from natural-log LSE
GEMM2  dP = dO · Vᵀ           (V resident in registers after one ldmatrix pass)
       dS = P ∘ (dP − delta)  in fp32 accumulators; I/O-dtype copy -> SMEM
GEMM3  dV += Pᵀ · dO          (P read back transposed via ldmatrix.trans)
GEMM4  dQ  = dS · K           -> fp32 atomic scatter into dq_accum
GEMM5  dK += dSᵀ · Q          (the iteration's last sQ reader)
```

dK/dV live in registers across the whole pass (CTA-private KV rows — no
atomics) and are written once in the epilogue through SMEM buffers that alias
the dead sK/sV regions. dQ is the transposed case — every KV tile contributes
to every q-tile row — hence the cross-CTA atomic workspace.

Key register/SMEM economies: P stays in fp32 accumulator registers for the
dS pointwise (no SMEM round trip for the P→dS chain); V is register-resident
so `sdS` aliases `sV`. The `CONFIG` table selects three 2-D partitions of the
8 math warps per `(D, q_tile, kv_tile)`: GEMM1/2 share the S/dP partition,
GEMM3/5 share the dK/dV partition, and GEMM4 uses the dQ partition. This keeps
MMA fragments `ldmatrix`-legal and balances the accumulators.

### Warp specialization

384 threads = 12 warps: **8 math warps** (`setmaxregister` up to 240), **1 TMA
producer warp** (down to 24 registers), and 3 register-donor warps (down to
24; they exist only to hand their registers to the math warps). The producer
prefetches the tensormaps, issues the one-time K/V TMA loads, then streams
Q/dO tiles through an mbarrier `expect_tx` ring — double-buffered
(`Q_STAGES == 2`) where SMEM allows, so the next tile's Q is in flight while
the current one computes. Producer and consumers rendezvous on 288-thread
named barriers (loop-top ready/consumed, post-GEMM3 dO release, and the
single-buffer Q refill); math-only synchronization uses separate 256-thread
barriers that exclude the producer.

### Head-size support and tile configs

Each head dim selects a sweep-tuned default `(q_tile, kv_tile)` and
warp-partition triple:

| D | q_tile × kv_tile | note |
|---|---|---|
| 32 | 128 × 64 | |
| 64 | 64 × 128 | wide KV tile: fewer CTAs, halves Q/dO re-reads |
| 128 | 64 × 64 | |
| 192 | 32 × 64 | double-buffered Q at the default config |
| 256 | 32 × 64 | SMEM-bound: single-buffered Q at the default config |

SMEM per CTA is `Q_STAGES·tile_q·d_qk + tile_q·d_v + tile_kv·d_qk +
max(tile_kv·d_v, 2·tile_q·tile_kv)` elements against the ~99 KB SM120 cap.
The constructor tries `Q_STAGES = 2` and falls back to a **single Q buffer**
when it does not fit; among the default configs, this occurs at D=256. In
the single-buffer branch the iteration reorders GEMM5 *before* GEMM4 (GEMM5
is sQ's last reader), so the Q refill for the next tile hides behind GEMM4
and the dQ scatter instead of stalling the loop. Head dims that are multiples
of 8 but are not native sizes compute on the next native size: the TMA
descriptors declare the actual extents, reads past them zero-fill in
hardware (pad columns contribute nothing anywhere in the chain), and the
dQ/dK/dV writers guard their stores at the actual widths. Explicit `tile_m`/`tile_n` knobs
override the Q/KV tile defaults; off-table combinations derive their warp
partitions from a largest-valid rule.

The Q/K head dim may exceed the V head dim (MLA: DeepSeek-V3 and Kimi-K2.6
train at 192/128). `d_qk` sizes Q/K/dQ/dK and `d_v` sizes V/O/dO/dV: GEMM1
contracts over `d_qk`, GEMM2 over `d_v`, and dK/dV share one warp partition
with per-side column slices. Tile defaults come from `d_qk`. When the
kernel-facing padded sizes differ, both must be multiples of 64 (one smem
page/swizzle); the adapter pads each side to its own native kernel size and
raises the VO side to at least 64 when they differ — the actual head dims
need not be (e.g. 96/40 computes as 128/64).

### dQ scatter and the scrambled workspace

Naive per-element atomics from the MMA fragment layout produce scattered
addresses. Instead `dq_accum` uses a fragment-order ("scrambled") layout in
which the 32 lanes of a warp each reduce an adjacent fp32 pair, covering 64
consecutive floats per `red.global.add.v2.f32` invocation. The coalesced layout
reduces dQ atomic traffic, and the `cvt` kernel un-scrambles it while converting
to the I/O dtype. `dot` pre-zeroes the workspace (fused with the delta
reduction; PDL orders it before `main`'s first add).

### GQA/MQA group reduction

Under GQA the chain rule sums each KV head's gradient over its group of
`group = H_q / H_kv` query heads: `dK[kv_head] = Σ dK-contribution[q_head]`
(same for dV); dQ is unaffected. The grid stays per query head — shrinking it
by `group` would starve the GPU at small `B·H_kv`, and the kernel is
compute-bound, so K/V-load reuse from walking the group in one CTA is not
worth that trade. Instead each CTA writes its q head's dK/dV epilogue tiles to
`dk_ws`/`dv_ws`, io-dtype buffers with an `H_q`-sized head axis (one slot per
query head, so the group's partials coexist), and a lightweight `reduce`
kernel then produces dK/dV: one thread per 16-byte output vector, accumulating
the group's slices in fp32 in fixed q-head order — bandwidth-bound and bitwise
deterministic by construction. For MHA (`H_q == H_kv`) the buffers alias the
`dk`/`dv` outputs themselves — the same epilogue writes the results directly,
nothing extra is carved, and no `reduce` kernel is launched.

### Causal, sliding-window, and padding masks

Masking is applied twice, cheaply:

* **Loop bounds** do the heavy lifting: causal clamps the first q-tile
  (`q_block_min`, bottom-right via `diag_off = S_kv − S_q`, a right band via
  the compile-time widening `q_block_min = (kv_base − diag_off − R) / tile_q`),
  a left window clamps the last (`q_block_max`) — fully-masked tiles are never
  visited, so square causal attention runs roughly half as many tile
  iterations.
* **In-register score masking** runs only on tiles that straddle a mask edge
  (`do_mask_causal` / `do_mask_window` / `do_mask_pad` gates); interior tiles
  skip it.

The softmax replay guards fully-masked rows (forward `LSE = −inf`) by
substituting `+inf`, reconstructing `P = 0` instead of NaN. Non-tile-multiple
sequence tails are handled with load clamps and store row-gates.

**Padding mask** (per-batch `seq_kv_lens`, optionally `seq_q_lens`) reuses
both layers: `q_block_max` trims to `ceil(seq_q_lens[b] / tile_q)` and a KV
tile fully inside the pad drains without loads or compute, while boundary
tiles mask scores at `seq_kv_lens[b]`. With bottom-right alignment the
diagonal anchors at the **actual** lengths (`diag_off = seq_kv_lens[b] −
seq_q_lens[b]`), matching the SM120 forward kernel and cuDNN padded-graph
semantics. Q rows at or past `seq_q_lens[b]` ride on the forward's
`LSE = −inf` convention (`P = 0`), so `dQ` rows past `seq_q_lens[b]` and
`dK`/`dV` rows past `seq_kv_lens[b]` come out exactly zero. The length
tensors are `None`-specialized kernel parameters: a specialization built
without them carries neither the parameters nor any padding code, and
needs no extra workspace either way.

### Deterministic vs. non-deterministic dQ

The default path's relaxed atomics make the fp32 add order — hence the
bitwise result — scheduling-dependent. Deterministic mode serializes each
(batch, head, q-tile)'s adds in ascending KV-tile order through an int32
turn-counter array: one elected lane spins on an acquire load until the
counter equals the CTA's turn, a math-warps-only barrier releases the scatter,
and after a second barrier a single `st.release.gpu` publishes `turn + 1`.
The release store's own fence drains the adds; in the double-buffered branch,
the following GEMM5 overlaps that drain.
Correctness rests on CTAs dispatching in ascending `blockIdx.x` (so the
awaited predecessor is always resident or done) and on the mask loop bounds
making each q-tile's visitors contiguous in KV index — under a sliding window
the turn subtracts the first visitor,
`kv_lo = max((q_block·tile_q + diag − W) // tile_kv, 0)`.
The relay instructions fold out under `const_expr` when determinism is off;
only the unused relay operand remains in the kernel ABI.

## Support surface and constraints

- SM120 and SM121, e.g. RTX 5090, RTX PRO 6000 Blackwell, and DGX Spark
- Dtypes: FP16 / BF16 (LSE fp32)
- Head dims: 32/64/128/192/256 natively; any other multiple of 8 up to 256
  computes on the next supported size, in place through the TMA zero-fill
  envelope. Rectangular `d_qk >= d_v` (MLA, e.g. 192/128) is supported.
- Masks: none, causal (top-left or bottom-right), right-band-widened causal
  (`diagonal_band_right_bound` > 0, the causal diagonal shifted right by a
  compile-time R), sliding window (left-window offset, with or without
  causal), padding (per-batch `seq_kv_len` required, `seq_q_len` optional;
  composes with the other masks)
- GQA/MQA: any `H_kv` dividing `H_q` (including `H_kv == 1`)
- Sink tokens: sink logits input and optional `dSink_token` output. dQ/dK/dV
  need no sink code (the forward LSE already folds the sink into the softmax
  denominator); the `dSink_token` output adds one tiny `dsink` reduce kernel
  (`dsink[h] = -sum p_sink * delta`, fixed order — bitwise deterministic in
  both modes)
- No dropout / bias / ALiBi / softcap / THD
- Workspace (carved from the caller's buffer): fp32 `delta` and `dq_accum`
  scratch plus int32 relay-counter storage (reserved in both modes); GQA adds
  the io-dtype `dk_ws`/`dv_ws` partials buffers (`B·S_kv·H_q·d_qk_padded` and
  `B·S_kv·H_q·d_v_padded` elements, where `d_*_padded` are the adapter's
  zero-padded head dimensions); use `scratch_workspace_bytes()` for the exact
  total
