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
dQ-finalize kernel per call).

## Requirements

The `cutedsl` optional dependency (`nvidia-cutlass-dsl` + `apache-tvm-ffi`)
and an SM120 or SM121 device.

## API Usage

```python
from cudnn.sdpa import sdpa_bwd_wrapper_dsl_sm120

grads = sdpa_bwd_wrapper_dsl_sm120(
    q_tensor=q, k_tensor=k, v_tensor=v,
    o_tensor=o, do_tensor=do, stats_tensor=stats,  # from the forward pass
    is_causal=True,
    causal_bottom_right=False,
    window_size_left=None,   # W: keys with k < q + diag - W are masked
    deterministic=False,     # ordered dQ KV-tile reduction (bitwise-reproducible)
    scale_softmax=None,      # None -> 1/sqrt(D)
)
dq, dk, dv = grads["dq_tensor"], grads["dk_tensor"], grads["dv_tensor"]
```

Tensors are logical `(B, H, S, D)`; any dense layout with the head dim
innermost-contiguous is accepted (`dense_flex`) — non-BSHD-compact operands
are staged through workspace copies. `stats` is the natural-log forward LSE,
fp32 `(B, H, S_q, 1)` contiguous.

Through the graph API, per-plan sequence-tile-width knobs can be requested via
`SdpaBwdKnobs`: `tile_m` controls the Q tile (`q_tile`), and `tile_n` controls
the KV tile (`kv_tile`).

## Determinism

By default the dQ accumulation across KV tiles uses fp32 atomics, so the result
can be bitwise non-deterministic when a q-tile receives contributions from
multiple KV-tile CTAs. `deterministic=True` serializes the per-`(batch, head,
q_tile)` additions in ascending KV-tile order through a GMEM turn-counter
array (the FlashAttention-style ordered-reduction relay). dK/dV are
deterministic in both modes. This maps to the graph API's
`use_deterministic_algorithm` and costs a shape-dependent slowdown of the main
kernel while keeping the workspace linear in sequence length.

## Kernel design and optimizations

### The three-kernel chain

One backward call is three launches, overlapped with programmatic dependent
launch (PDL) so each kernel's prologue runs under its predecessor's tail:

```
dot   delta = rowsum(O ∘ dO); zeroes dq_accum (and, when deterministic, the relay counters)
main  the fused five-GEMM pass; writes dK/dV, accumulates dQ into dq_accum
cvt   dq_accum (fp32, scrambled) -> dQ (io dtype), applying attn_scale
```

### Main-kernel pipeline: KV-stationary, five chained GEMMs

Grid is `(num_kv_tiles, H, B)` — one CTA owns one KV tile, loads K/V **once**,
and walks every q-tile of its (batch, head) in descending order. Per q-tile
iteration:

```
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

SMEM per CTA is `(Q_STAGES + 1)·M·D + N·D + max(N·D, 2·M·N)` elements against
the ~99 KB SM120 cap. The constructor tries `Q_STAGES = 2` and falls back to a
**single Q buffer** when it doesn't fit; among the default configs, this occurs
at D=256. In the single-buffer branch the iteration reorders GEMM5 *before*
GEMM4 (GEMM5 is sQ's last reader), so the Q refill for the next tile hides
behind GEMM4 and the dQ scatter instead of stalling the loop. Head dims that
are multiples of 8 but are not native sizes are zero-padded by the adapter
(pad columns contribute nothing anywhere in the chain). Explicit
`tile_m`/`tile_n` knobs override the Q/KV tile defaults; off-table combinations
derive their warp partitions from a largest-valid rule.

### dQ scatter and the scrambled workspace

Naive per-element atomics from the MMA fragment layout produce scattered
addresses. Instead `dq_accum` uses a fragment-order ("scrambled") layout in
which the 32 lanes of a warp each reduce an adjacent fp32 pair, covering 64
consecutive floats per `red.global.add.v2.f32` invocation. The coalesced layout
reduces dQ atomic traffic, and the `cvt` kernel un-scrambles it while converting
to the I/O dtype. `dot` pre-zeroes the workspace (fused with the delta
reduction; PDL orders it before `main`'s first add).

### Causal and sliding-window masks

Masking is applied twice, cheaply:

* **Loop bounds** do the heavy lifting: causal clamps the first q-tile
  (`q_block_min`, bottom-right via `diag_off = S_kv − S_q`), a left window
  clamps the last (`q_block_max`) — fully-masked tiles are never visited, so
  square causal attention runs roughly half as many tile iterations.
* **In-register score masking** runs only on tiles that straddle a mask edge
  (`do_mask_causal` / `do_mask_window` gates); interior tiles skip it.

The softmax replay guards fully-masked rows (forward `LSE = −inf`) by
substituting `+inf`, reconstructing `P = 0` instead of NaN. Non-tile-multiple
sequence tails are handled with load clamps and store row-gates.

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
- Head dims: 32/64/128/192/256 natively; any other multiple of 8 up to 256 is
  served by zero-padding D to the next supported size (`d_qk == d_v`)
- Masks: none, causal (top-left or bottom-right), sliding window
  (left-window offset, with or without causal)
- Equal Q/KV head counts (no GQA/MQA); no dropout / bias / ALiBi / sinks /
  softcap / THD
- Workspace (carved from the caller's buffer): fp32 `delta` and `dq_accum`
  scratch plus int32 relay-counter storage (reserved in both modes); padded-D
  and non-compact layouts add staging copies
