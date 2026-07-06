# SDPA Forward (SM100, D=256)

**This is an experimental API and subject to change.**

## Overview

**SDPA forward** pass for `D=256` on NVIDIA Blackwell GPUs (`SM100+`).
Computes the attention output `O` and log-sum-exp statistics `LSE` using a CUTE DSL implementation.

This is available through a standalone API (documented below) and is also used by the experimental [SDPA PyTorch custom operator](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html#scaled-dot-product-attention-pytorch-op) for supported `D=256` BHSD cases.

## API Usage

### High-level wrapper

```python
import torch
from cudnn import sdpa_fwd_wrapper_sm100_d256

result = sdpa_fwd_wrapper_sm100_d256(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    cum_seqlen_q_tensor=cum_seqlen_q,  # None for B,H,S,D layout
    cum_seqlen_k_tensor=cum_seqlen_k,  # None for B,H,S,D layout
    max_s_q=1024,  # Optional; inferred from cum_seqlen_q_tensor when None
    max_s_k=1024,  # Optional; inferred from cum_seqlen_k_tensor when None
    qk_acc_dtype=torch.float32,
    pv_acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,  # Defaults to 1/sqrt(D)
    scale_output=1.0,
    current_stream=None,
)

o_tensor, lse_tensor = result
# Key access: result["o_tensor"], result["lse_tensor"]
```

### Class API

```python
import torch
from cudnn import SdpafwdSm100D256

sdpa_fwd = SdpafwdSm100D256(
    sample_q=q,
    sample_k=k,
    sample_v=v,
    sample_o=o,
    sample_lse=lse,
    sample_cum_seqlen_q=cum_seqlen_q,  # None for B,H,S,D layout
    sample_cum_seqlen_k=cum_seqlen_k,  # None for B,H,S,D layout
    max_s_q=1024,
    max_s_k=1024,
    qk_acc_dtype=torch.float32,
    pv_acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,
    scale_output=1.0,
)

assert sdpa_fwd.check_support()
sdpa_fwd.compile()
sdpa_fwd.execute(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    lse_tensor=lse,
    cum_seqlen_q_tensor=cum_seqlen_q,
    cum_seqlen_k_tensor=cum_seqlen_k,
    scale_softmax=None,
    scale_output=1.0,
    current_stream=None,
)
```

## Parameters

### Input / output tensors

- `Q`
  - B,H,S,D layout: `(B, H_q, S_q, D)`
  - T,H,D layout: `(T_q, H_q, D)` or `(1, T_q, H_q, D)`
  - Dtype: `torch.float16` or `torch.bfloat16`
- `K`
  - B,H,S,D layout: `(B, H_k, S_k, D)`
  - T,H,D layout: `(T_k, H_k, D)` or `(1, T_k, H_k, D)`
  - Dtype: must match `Q`
- `V`
  - Same layout family and dtype constraints as `K`
- `O`
  - Same layout family and dtype constraints as `Q`
- `LSE`
  - B,H,S,D layout: `(B, H_q, S_q)` and must be contiguous
  - T,H,D layout: `(T_q, H_q)` or `(T_q, H_q, 1)`
  - Dtype: `torch.float32`
- Varlen cumulative lengths (T,H,D layout only)
  - `cum_seqlen_q_tensor`, `cum_seqlen_k_tensor`
  - Shape: `(batch_size + 1,)`
  - Dtype: `torch.int32`

### Common parameters

- `qk_acc_dtype`, `pv_acc_dtype`
  - Supported: `torch.float32` only
- `mma_tiler_mn`
  - Supported: `(128, 128)` only
- `is_causal`
  - If `True`, the right window bound is forced to `0`
- `window_size`
  - Sliding-window bounds `(left, right)`
  - Default: `(-1, -1)` (full window)
- `scale_softmax`
  - Defaults to `1 / sqrt(D)` when omitted or set to `0.0`
- `scale_output`
  - Output scaling applied after the attention reduction
  - Default: `1.0`

## Wrapper return values

Returns a `TupleDict` with keys:

- `o_tensor`
- `lse_tensor`

Tuple unpacking order is `(o_tensor, lse_tensor)`.

## Support surface and constraints

- `head_dim` must be exactly `256`
- `H_q` must be divisible by `H_k`
- `D_qk` must equal `D_v`
- Requires NVIDIA Blackwell (`SM100+`)
- `window_size_left < s_k_max - 1`
- `window_size_right < s_q_max - 1`
- Non-causal mode currently supports only full-window configuration:
  - `window_size == (-1, -1)`
- The standalone API supports both BHSD and THD-style varlen inputs
- The experimental SDPA PyTorch custom op currently routes only the plain BHSD `D=256` cases through this OSS forward kernel
