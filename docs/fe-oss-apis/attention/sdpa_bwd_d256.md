# SDPA Backward (SM100, D=256)

**This is an experimental API and subject to change.**

## Overview

**SDPA backward** pass for D=256 on NVIDIA Blackwell GPUs (SM100). 
Computes attention gradients (`dQ`, `dK`, `dV`) for scaled dot-product attention using a 2-kernel CUTE DSL implementation.

Given forward tensors and statistics (`Q`, `K`, `V`, `O`, `dO`, `LSE`), this API returns:

- `dQ`: gradient w.r.t. query
- `dK`: gradient w.r.t. key
- `dV`: gradient w.r.t. value

This is available through a standalone API (detailed below) or as part of the experimental [SDPA Pytorch custom operator](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html#scaled-dot-product-attention-pytorch-op).

## Acknowledgments

This kernel was jointly developed by Shengbin Di, Yuxi Chi, and Linfeng Zheng in close collaboration with Alibaba. We would like to extend special thanks to the core contributors from Alibaba: Siyu Wang, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, and Wei Lin for their significant contributions to this work.

### Equations

For attention probabilities

$$
P = \operatorname{softmax}(S), \quad S = \frac{QK^T}{\sqrt{D}} + \text{mask}
$$

the backward pass follows:

$$
dV = P^T dO,\quad
dP = dO V^T,\quad
dS = P \odot \left(dP - \sum(dP \odot P)\right),\quad
dQ = dS K,\quad
dK = dS^T Q
$$

Here, the `sum(dP \odot P)` term is reduced row-wise over the key/softmax dimension for each query row.

---

## API Usage

### High-level wrapper

```python
import torch
from cudnn import sdpa_bwd_wrapper_sm100_d256

result = sdpa_bwd_wrapper_sm100_d256(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    do_tensor=do,
    lse_tensor=lse,
    cum_seqlen_q_tensor=cum_seqlen_q,  # None for B,H,S,D layout
    cum_seqlen_k_tensor=cum_seqlen_k,  # None for B,H,S,D layout
    max_s_q=1024,  # Optional; inferred from cum_seqlen_q_tensor when None
    max_s_k=1024,  # Optional; inferred from cum_seqlen_k_tensor when None
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    dkdv_mma_tiler_mn=(128, 64),
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,  # Defaults to 1/sqrt(D)
    current_stream=None,
)

dq, dk, dv = result
# Key access: result["dq_tensor"], result["dk_tensor"], result["dv_tensor"]
```

### Class API

```python
import torch
from cudnn import SdpabwdSm100D256

sdpa_bwd = SdpabwdSm100D256(
    sample_q=q,
    sample_k=k,
    sample_v=v,
    sample_o=o,
    sample_do=do,
    sample_lse=lse,
    sample_dq=dq,
    sample_dk=dk,
    sample_dv=dv,
    sample_cum_seqlen_q=cum_seqlen_q,  # None for B,H,S,D layout
    sample_cum_seqlen_k=cum_seqlen_k,  # None for B,H,S,D layout
    max_s_q=1024,  # Optional; inferred from sample_cum_seqlen_q when None
    max_s_k=1024,  # Optional; inferred from sample_cum_seqlen_k when None
    acc_dtype=torch.float32,
    mma_tiler_mn=(128, 128),
    dkdv_mma_tiler_mn=(128, 64),
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,
)

assert sdpa_bwd.check_support()
sdpa_bwd.compile()
sdpa_bwd.execute(
    q_tensor=q,
    k_tensor=k,
    v_tensor=v,
    o_tensor=o,
    do_tensor=do,
    lse_tensor=lse,
    dq_tensor=dq,
    dk_tensor=dk,
    dv_tensor=dv,
    cum_seqlen_q_tensor=cum_seqlen_q,  # None for B,H,S,D layout
    cum_seqlen_k_tensor=cum_seqlen_k,  # None for B,H,S,D layout
    scale_softmax=None,
    current_stream=None,
)
```

---

## Parameters

### Input/Output tensors

- Input tensor **Q**: `q_tensor` (wrapper) or `sample_q`/`q_tensor` (class)
  - B,H,S,D layout: shape `(B, H_q, S_q, D)`
  - T,H,D layout: shape `(T_q, H_q, D)` (or `(1, T_q, H_q, D)`)
  - Dtype: `{torch.float16, torch.bfloat16}`
- Input tensor **K**: `k_tensor` (wrapper) or `sample_k`/`k_tensor` (class)
  - B,H,S,D layout: shape `(B, H_k, S_k, D)`
  - T,H,D layout: shape `(T_k, H_k, D)` (or `(1, T_k, H_k, D)`)
  - Dtype: must match `Q`
- Input tensor **V**: `v_tensor` (wrapper) or `sample_v`/`v_tensor` (class)
  - B,H,S,D layout: shape `(B, H_k, S_k, D)`
  - T,H,D layout: shape `(T_k, H_k, D)` (or `(1, T_k, H_k, D)`)
  - Dtype: must match `Q`
- Input tensor **O**: `o_tensor` (wrapper) or `sample_o`/`o_tensor` (class)
  - Same layout and shape family as `Q`
  - Dtype: must match `Q`
- Input tensor **dO**: `do_tensor` (wrapper) or `sample_do`/`do_tensor` (class)
  - Same layout and shape family as `Q`
  - Dtype: must match `Q`
- Input tensor **LSE**: `lse_tensor` (wrapper) or `sample_lse`/`lse_tensor` (class)
  - B,H,S,D layout: shape `(B, H_q, S_q)` (must be contiguous)
  - T,H,D layout: shape `(T_q, H_q)` (or `(T_q, H_q, 1)`)
  - Dtype: `torch.float32`
- Output tensor **dQ**
  - Wrapper: `result["dq_tensor"]`
  - Class: `sample_dq`/`dq_tensor`
  - Shape family: same as `Q`
  - Dtype: must match `Q`
- Output tensor **dK**
  - Wrapper: `result["dk_tensor"]`
  - Class: `sample_dk`/`dk_tensor`
  - Shape family: same as `K`
  - Dtype: must match `Q`
- Output tensor **dV**
  - Wrapper: `result["dv_tensor"]`
  - Class: `sample_dv`/`dv_tensor`
  - Shape family: same as `V`
  - Dtype: must match `Q`
- Varlen cumulative lengths (T,H,D layout only)
  - `cum_seqlen_q_tensor`, `cum_seqlen_k_tensor`
  - Shape: `(batch_size + 1,)`
  - Dtype: `torch.int32`

### Common parameters

- `acc_dtype: torch.dtype`
  - Accumulator dtype
  - Supported: `torch.float32` only
- `mma_tiler_mn: Tuple[int, int]`
  - dQ kernel MMA tile
  - Supported: `(128, 128)` only
- `dkdv_mma_tiler_mn: Tuple[int, int]`
  - dK/dV kernel MMA tile
  - Supported: `(128, 64)` only
- `is_causal: bool`
  - Enables causal masking
  - If `True`, the right window bound is forced to `0`
- `window_size: Tuple[int, int]`
  - Sliding-window bounds `(left, right)`
  - Default: `(-1, -1)` (full window)
- `scale_softmax: Optional[float]`
  - Softmax scale
  - Default: `1/sqrt(D)` when omitted or set to `0.0`
- CUDA stream (`current_stream` in both wrapper and class API)
  - Default: `None` (uses default stream)
  - When provided, both workspace initialization and kernel launch are enqueued on that stream

### Wrapper-specific parameters: `sdpa_bwd_wrapper_sm100_d256`

- `q_tensor`, `k_tensor`, `v_tensor`, `o_tensor`, `do_tensor`, `lse_tensor`
- `cum_seqlen_q_tensor`, `cum_seqlen_k_tensor`
- `max_s_q`, `max_s_k` (`Optional[int]`, default `None`; for varlen mode these are inferred from cumulative lengths when omitted)
- Common parameters listed above

### Wrapper return values

Returns a `TupleDict` with keys:

- `dq_tensor`
- `dk_tensor`
- `dv_tensor`

Tuple unpacking order is: `(dq_tensor, dk_tensor, dv_tensor)`.

### Class-specific parameters: `SdpabwdSm100D256`

#### `SdpabwdSm100D256` (constructor)

- `sample_q`, `sample_k`, `sample_v`, `sample_o`, `sample_do`, `sample_lse`
- `sample_dq`, `sample_dk`, `sample_dv`
- `sample_cum_seqlen_q`, `sample_cum_seqlen_k`
- `max_s_q`, `max_s_k` (`Optional[int]`, default `None`; for varlen mode these are inferred from sample cumulative lengths when omitted)
- Common parameters listed above

#### `SdpabwdSm100D256.execute`

- Runtime tensors for all inputs/outputs listed above
- `scale_softmax`, `current_stream`

---

## Support Surface and Constraints

### Shapes and head geometry

- `D_qk` must equal `D_v`
- `head_dim` (`D`) must be exactly `256`
- `H_q` must be divisible by `H_k` (supports GQA/MQA-style head grouping)

### Layout and stride constraints

- B,H,S,D mode (`Q/K/V/O/dO/dQ/dK/dV`):
  - both 'cum_seqlen_q_tensor' and 'cum_seqlen_k_tensor' must be 'None'
  - Tensors must be rank-4 with `(B, H, S, D)` shape semantics
  - Stride order must match `s,h,d,b` (as produced by views like `(B, S, H, D).transpose(1, 2)`)
  - `LSE` must be contiguous
- T,H,D mode (`Q/K/V/O/dO/dQ/dK/dV`):
  - both 'cum_seqlen_q_tensor' and 'cum_seqlen_k_tensor' must be provided
  - Tensors must be rank-3 `(T, H, D)` or rank-4 `(1, T, H, D)`
  - If rank-4 is used, batch dimension must be `1`
  - If constructor `max_s_q`/`max_s_k` are provided, they must be greater than or equal to the maxima implied by sample cumulative lengths

### Dtypes

- `Q`, `K`, `V`, `O`, `dO`, `dQ`, `dK`, `dV` must have the same dtype. Supported dtypes:
  - `torch.float16`
  - `torch.bfloat16`
- `LSE` and `acc_dtype` must be `torch.float32`
- `cum_seqlen_q_tensor` and `cum_seqlen_k_tensor` must be `torch.int32` and have matching shape

### Window/mask behavior

- `window_size_left < s_k_max - 1`
- `window_size_right < s_q_max - 1`
- Non-causal mode currently supports only full-window configuration:
  - `window_size == (-1, -1)`

### Hardware requirements

- CUDA must be available
- All tensors must be CUDA tensors on the same device (including `cum_seqlen_q_tensor` / `cum_seqlen_k_tensor` when provided)
- Requires SM100+ compute capability
- SM103 is not supported

---

## Usage Examples

For runnable examples and reference-comparison checks, see:

- `test/python/fe_api/test_sdpa_bwd.py`
- `test/python/fe_api/test_sdpa_bwd_utils.py`

