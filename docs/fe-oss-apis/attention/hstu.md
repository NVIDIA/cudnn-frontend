# HSTU Attention (SM100)

**This is an experimental API and subject to change.**

## Overview

Hierarchical Sequential Transduction Unit (HSTU) attention is an attention
variant for generative recommender models. For each packed sequence and
attention head $h$, let $Q_h, K_h \in \mathbb{R}^{L \times d}$ and
$V_h \in \mathbb{R}^{L \times d_v}$. The operation is

$$
S_h = \alpha Q_h K_h^T + R_h,
$$

$$
A_h = \frac{1}{L_{\mathrm{scale}}}
      \left(M \odot \operatorname{SiLU}(S_h)\right),
$$

$$
O_h = A_h V_h,
$$

$$
\operatorname{HSTU\text{-}MHA}(Q,K,V)
  = \operatorname{Concat}(O_1,O_2,\ldots,O_H).
$$

Here, $\alpha$ is the QK score scale, $M$ is the attention mask, and
$L_{\mathrm{scale}}$ is exposed as `scaling_seqlen`. If
`scaling_seqlen=None`, it defaults to `max_seqlen_q`. Unlike standard
Transformer attention, HSTU applies SiLU to the scores and does **not** perform
row-wise softmax normalization. This matches the requested operation in
[issue #369](https://github.com/NVIDIA/cudnn-frontend/issues/369).

The current SM100 implementation does not support the optional relative
attention bias $R_h$; callers must use `rab=None`. The equation above states
the complete HSTU operator definition, while the support tables below describe
the currently implemented subset.

## Installation

Install cuDNN Frontend with its CuTe DSL optional dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

The implementation is available from `cudnn.hstu_attention` and through the
following lazy top-level exports:

```python
from cudnn import (
    HSTUFwdSm100,
    HSTUBwdSm100,
    hstu_attention_forward,
    hstu_attention_backward,
)
```

## Tensor layout

HSTU uses packed variable-length, or THD, tensors:

| Tensor | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `q` | `(T_q, H, D)` | FP16 or BF16 | Packed queries |
| `k` | `(T_k, H, D)` | same as `q` | Packed keys |
| `v` | `(T_k, H, D)` | same as `q` | Packed values |
| `do` | `(T_q, H, D)` | same as `q` | Upstream gradient for backward |
| `cu_seqlens_q` | `(B + 1,)` | `torch.int32` | Cumulative query lengths |
| `cu_seqlens_k` | `(B + 1,)` | `torch.int32` | Cumulative key/value lengths |
| `out` | `(T_q, H, D)` | same as `q` | Packed attention output |
| `dq`, `dk`, `dv` | corresponding input shape | same as `q` | Backward gradients |

`T_q` and `T_k` are the total query and key/value token counts across the
batch. Cumulative-length tensors start with zero and delimit each packed
sequence; `max_seqlen_q` and `max_seqlen_k` give the maximum sequence lengths
for the batch. Q, K, and V use the same number of heads, so this implementation
currently exposes multi-head attention rather than GQA or MQA.

The API validates tensor metadata (rank, shape, dtype, device, and layout) but
trusts the values stored in CUDA metadata tensors such as `cu_seqlens_q`,
`cu_seqlens_k`, `num_targets`, `page_ids`, and `page_indptrs`. It does not copy
those values to the host before launch. Callers provide `max_seqlen_q` and
`max_seqlen_k` explicitly; `scaling_seqlen=None` then uses the supplied
`max_seqlen_q` without inspecting `cu_seqlens_q` on the host.

Tensors must have non-overlapping storage and a 16-byte-aligned base pointer.
The wrapper can adapt some otherwise non-contiguous packed views, but naturally
aligned THD tensors with a contiguous last dimension avoid an internal layout
copy. Paged-KV storage itself must be contiguous. Preallocated output or
gradient tensors must not overlap any input or one another.

## High-level functions

`hstu_attention_forward` allocates and returns the packed output. A basic causal
call has the following form:

```python
result = hstu_attention_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    scaling_seqlen=None,       # Defaults to max_seqlen_q
    window_size=(-1, 0),       # Causal mask
    alpha=1.0,
)
out = result["o_tensor"]
```

`hstu_attention_backward` computes the Q, K, and V gradients explicitly:

```python
grads = hstu_attention_backward(
    do,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    scaling_seqlen=None,
    window_size=(-1, 0),
    alpha=1.0,
)
dq = grads["dq_tensor"]
dk = grads["dk_tensor"]
dv = grads["dv_tensor"]
```

Forward and backward must use the same `alpha`, `scaling_seqlen`, mask
configuration, and sequence metadata. `scaling_seqlen` must be positive. It is
a runtime normalization factor rather than the number of valid tokens in each
individual packed sequence.

The first call for a new static kernel configuration JIT-compiles a CuTe DSL
kernel; subsequent calls reuse the in-process compile cache. Execution follows
the current PyTorch CUDA stream through TVM FFI.

## Class APIs

`HSTUFwdSm100` and `HSTUBwdSm100` provide the explicit FE OSS lifecycle for
applications that want to compile once and execute repeatedly with
preallocated outputs:

1. Construct the class with sample input and output tensors plus the static
   mask configuration.
2. Call `check_support()` to validate dtype, shape, layout, architecture, and
   feature combinations.
3. Call `compile()` before a latency-sensitive region or CUDA Graph capture.
4. Call `execute()` with runtime tensors that match the compiled descriptors.

The allocating functions above are the recommended entry point for common
usage.

## Mask support

`window_size=(left, right)` controls the standard token-window mask:

| Mode | Configuration | Support |
| --- | --- | --- |
| Full attention | `window_size=(-1, -1)` | Forward and backward |
| Causal | `window_size=(-1, 0)` | Forward and backward |
| Local/sliding window | finite left and/or right bound | Forward and backward |
| Target-group mask | `num_targets` with a causal window | Forward and backward |
| Arbitrary mask | CUDA `int32` `func` metadata | Forward and backward |
| Context mask | `num_contexts` | Not supported |

Target-group masking is defined only with causal attention. Arbitrary-mask
metadata is a specialized kernel contract and cannot be combined with causal,
local, target, context, or paged-KV modes.

`func_tensor` has shape `(1, N, L)` and dtype `torch.int32`, where `N` is
positive and odd and `L >= T_q + 256`. For every packed query row, its `N`
entries encode alternating masked interval boundaries followed by the
valid-key upper bound. The first dimension is currently fixed to one; the 256
extra columns are kernel padding. Arbitrary-mask forward requires
`max_seqlen_k <= 65536`, and backward requires `max_seqlen_q <= 32768`, matching
the kernels' fixed valid-block scratch capacity. The API trusts the
device-resident boundary values and does not validate their ordering or range
on the host.

Forward also has a causal paged-KV path using `paged_kv`, `page_ids`, and
`page_indptrs`. It requires a page size of 128 and cannot be combined with
local, context, or arbitrary masking. Paged KV is not supported by backward.
`paged_kv` has shape `(num_pages, 2, 128, H, D)` with `num_pages > 0`; K/V is
selected by the second dimension. `page_ids` is a contiguous one-dimensional
`torch.int32` array of physical page indices. `page_indptrs` is a contiguous
`torch.int32` array of shape `(B + 1,)` delimiting each sequence's slice in
`page_ids`. Their values must already be valid for `paged_kv`; they are consumed
on the GPU without host-side range or monotonicity checks.

## Support matrix

| Direction | Architecture | Dtype | Head dimension | Attention |
| --- | --- | --- | --- | --- |
| Forward | SM100/SM10x | FP16, BF16 | 64, 128, 256 | MHA |
| Backward | SM100/SM10x | FP16, BF16 | 64, 128 | MHA |

## Current limitations

- Relative attention bias is not implemented: `rab` must be `None` and dRAB
  is unavailable.
- Context masking through `num_contexts` is not implemented.
- Backward is nondeterministic; requesting deterministic backward raises
  `NotImplementedError`.
- Forward head dimension 256 does not have a corresponding backward kernel.
- GQA and MQA are not supported.
- Padded BHSD/BSHD inputs and separate `seqused_q`/`seqused_k` valid-length
  tensors are not supported; use packed THD tensors and cumulative lengths.
- The implementation requires NVIDIA Blackwell compute capability SM10x.
