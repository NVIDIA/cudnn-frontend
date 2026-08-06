# SDPA Forward (SM80)

**This is an experimental API and subject to change.**

## Overview

**SDPA forward** pass for NVIDIA Ampere A100 (`SM80`), implemented with the
CuTe DSL.  Computes the attention output `O`, log-sum-exp statistics `LSE`,
and (optionally) the raw score statistics `score_max` / `score_sum_exp`.

Two integration surfaces are provided:

* a standalone API (documented below) under `cudnn.sdpa`, and
* a FROST engine (`sdpa_fwd_prefill_sm80`, see `cudnn.sdpa.fwd.engines`) that serves
  single-node `sdpa()` graphs built with `cudnn.pygraph` when selectable from the ranked plan list (`graph.plans` /
  `graph.select_plan(i)`) with `CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1`.

The kernels live under `python/cudnn/sdpa/fwd/kernels/`
(`prefill_f16_sm80.py` generic, `prefill_d256_f16_sm80.py` d=256), building
on the shared FROST tile library at `python/cudnn/frost/tile_dsl/`
(provenance: `python/cudnn/sdpa/fwd/kernels/__init__.py`).

## Requirements

The `cutedsl` optional dependency (`pip install
nvidia-cudnn-frontend[cutedsl]`, i.e. `nvidia-cutlass-dsl` +
`apache-tvm-ffi`) and an SM80 (A100) device. There is no other kernel
dependency; the CuTe-DSL JIT runs on the first execute and the compiled
kernels are cached per shape.

## API Usage

### High-level wrapper

```python
import torch
from cudnn.sdpa import sdpa_fwd_wrapper_sm80

# BHSD-logical tensors with BSHD-physical stride order (3, 1, 2, 0).
def bshd(b, h, s, d):
    return torch.randn(b, s, h, d, dtype=torch.float16, device="cuda").permute(0, 2, 1, 3)

q, k, v = bshd(2, 8, 1024, 128), bshd(2, 2, 1024, 128), bshd(2, 2, 1024, 128)

result = sdpa_fwd_wrapper_sm80(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    is_causal=True,
    window_size=(-1, -1),        # (left, right); right > 0 widens the causal band
    scale_softmax=None,          # defaults to 1/sqrt(D_QK)
    causal_bottom_right=False,
    seq_kv_lens=None,            # per-batch KV lengths (padding mask), int32 [B]
    seq_len_q=None,              # per-batch Q lengths (with the padding mask)
    bias_tensor=None,            # additive bias [1, H, S_q, S_kv]
    alibi=False,
    sinks=None,                  # learned attention sinks, fp32 [H]
    return_score_stats=False,    # also return score_max / score_sum_exp
    rope_freqs=None,             # fused RoPE angles [max_s, 1, 1, D_QK]
    block_mask=None,             # bit-packed uint8 128x128 block sparsity
    scheduler="auto",            # auto / default / lpt / lpt_l2
)
o_tensor, lse_tensor = result
```

Packed THD / varlen: pass `[1, T, H, D]`-packed `q/k/v` plus
`cum_seqlen_q_tensor` / `cum_seqlen_k_tensor` (`[B+1]`, int32) and `max_s_q`;
the wrapper routes to the kernel's THD path.

### Class API

`SdpafwdSm80` implements the `APIBase` contract
(`check_support()` / `compile()` / `execute()`); `compile()` is a no-op
because the kernel modules own a per-shape JIT cache.

## Support surface and constraints

- SM80 (A100) exactly; `s_q == 1` (decode) is out of scope (prefill kernels)
- Dtypes: FP16 / BF16 (LSE and score stats are FP32)
- Head dims: any `(D_QK, D_V)` inside the `(256, 256)` envelope — inputs are
  zero-padded up to the nearest kernel flavor: gptoss `(64, 64)`, llama
  `(128, 128)`, dsv3 `(192, 128)`, qwen `(256, 256)`
- GQA/MQA: `H_q % H_kv == 0`
- Layout: BHSD-logical with BSHD-physical stride order `(3, 1, 2, 0)`
  (size-1 dims wildcarded)
- Masks: none / causal / SWA (including non-causal) / bottom-right
  alignment (incl. BR+SWA) / per-batch padding / causal right-band widening
- Not supported: dropout, paged attention, `score_mod`, tensor-valued
  `attn_scale`, `rng_dump`, FP8; bias/RoPE/block_mask are dense-only
  (mutually exclusive with THD); `scale_output != 1.0`
- `current_stream` accepts a `cuda.CUstream` (or raw stream int); the
  kernels dispatch onto it and are CUDA-graph-capturable

## Wrapper return values

`TupleDict` with keys `o_tensor`, `lse_tensor`
(+ `score_max`, `score_sum_exp` when `return_score_stats=True`).
