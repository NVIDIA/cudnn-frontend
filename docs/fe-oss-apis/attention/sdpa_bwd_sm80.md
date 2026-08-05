# SDPA Backward (SM80)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** pass for NVIDIA Ampere A100 (`SM80`), implemented with the
CuTe DSL.  Consumes the forward activations (`Q/K/V/O`), the loss gradient
`dO`, and the forward `LSE`; produces `dQ/dK/dV` (+ `dBias` when an additive
bias is present, + `dSink` when learned sinks are present).

Two integration surfaces are provided:

* a standalone API (documented below) under `cudnn.sdpa`, and
* a FROST engine (`sdpa_bwd_sm80`, see `cudnn.sdpa.bwd.engines`) that serves
  single-node `sdpa_backward()` graphs built with `cudnn.pygraph` when
  selected from the ranked plan list (`graph.plans` /
  `graph.select_plan(i)`) with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`.

The kernel lives at
`python/cudnn/sdpa/bwd/kernels/bprop_f16_sm80.py` (generic,
fully parameterized on `d_qk`/`d_v`); `bprop_d64_f16_sm80.py` is a d=64
perf variant that is not yet routed.  Provenance:
`python/cudnn/sdpa/bwd/kernels/__init__.py`.

## Requirements

Same as the forward: the `cutedsl` optional dependency plus the `ctm` DSL
package (internal distribution; not on PyPI), and

## API Usage

```python
from cudnn.sdpa import sdpa_bwd_wrapper_sm80

grads = sdpa_bwd_wrapper_sm80(
    q_tensor=q, k_tensor=k, v_tensor=v,
    o_tensor=o, do_tensor=do, lse_tensor=lse,   # from the forward pass
    is_causal=True,
    window_size=(-1, -1),
    scale_softmax=None,
    causal_bottom_right=False,
    seq_kv_lens=None,
    seq_len_q=None,
    bias_tensor=None,       # returns dbias_tensor (fp32, bias-shaped)
    alibi=False,
    sinks=None,             # returns dsink_tensor (fp32 [H])
    block_mask=None,
    deterministic=False,    # ordered dQ KV-tile reduction (bitwise-reproducible)
)
dq, dk, dv = grads["dq_tensor"], grads["dk_tensor"], grads["dv_tensor"]
```

Packed THD / varlen: pass `[1, T, H, D]`-packed tensors, packed `[1, H, T_q]`
LSE, and `cum_seqlen_q_tensor` / `cum_seqlen_k_tensor`.

## Determinism

By default the dQ accumulation across KV tiles uses `atomicAdd`, which is
bitwise non-deterministic whenever a sequence spans more than one KV tile.
`deterministic=True` orders the per-`(seq, head, q_tile)` additions by
KV-tile via a gmem semaphore, making dQ bitwise-reproducible.  This maps to
the graph API's `use_deterministic_algorithm`.

## Support surface and constraints

- SM80 (A100) exactly
- Dtypes: FP16 / BF16 (LSE fp32; dBias/dSink fp32)
- Head dims: `(D_QK, D_V)` inside the `(256, 256)` envelope with
  `D_QK >= D_V` (zero-padded up to a kernel flavor, as in the forward)
- GQA/MQA: `H_q % H_kv == 0` (dK/dV are head-reduced to `H_kv`)
- Masks: none / causal / SWA / bottom-right (incl. BR+SWA) / per-batch
  padding / causal right-band widening
- Not supported: dropout, paged attention, `score_mod`, tensor-valued
  `attn_scale`, FP8; bias/RoPE/block_mask are dense-only (mutually
  exclusive with THD)
- `current_stream` accepts a `cuda.CUstream` (or raw stream int); the
  kernels dispatch onto it and are CUDA-graph-capturable
