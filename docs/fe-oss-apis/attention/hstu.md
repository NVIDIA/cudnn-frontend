# HSTU Attention (Blackwell SM100/SM103)

**This is an experimental API and subject to change.**

## Overview

Hierarchical Sequential Transduction Unit (HSTU) attention is an attention
variant for generative recommender models. For each packed sequence and
attention head $h$, let $Q_h, K_h \in \mathbb{R}^{L \times d}$ and
$V_h \in \mathbb{R}^{L \times d_v}$. The operation is

$$
S_h = \alpha Q_h K_h^T,
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

The HSTU kernels and cuDNN Frontend integration were developed by NVIDIA. The
forward and backward kernels and public API use the Apache License 2.0. Some
low-level attention utility files build on FlashAttention and NVIDIA
CUTLASS/CuTe DSL work and are distributed under the MIT License while retaining
their original author copyright notices; see the repository's [licensing
guide](../../../LICENSING.md) and [third-party notices](../../../THIRD_PARTY_LICENSES.txt).

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
for the batch. The kernel requires `max_seqlen_q <= max_seqlen_k`. Q, K, and V
use the same number of heads, so this implementation currently exposes
multi-head attention rather than GQA or MQA.

The API validates tensor metadata (rank, shape, dtype, device, and layout) but
trusts the values stored in CUDA metadata tensors such as `cu_seqlens_q`,
`cu_seqlens_k`, `page_ids`, and `page_indptrs`. It does not copy those values
to the host before launch. Callers provide `max_seqlen_q` and `max_seqlen_k`
explicitly; `scaling_seqlen=None` then uses the supplied `max_seqlen_q` without
inspecting `cu_seqlens_q` on the host.

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

The optional `dq_tensor`, `dk_tensor`, and `dv_tensor` arguments provide
caller-owned gradient output buffers. They can be supplied independently; the
function allocates any omitted output. Supplied buffers are overwritten and
returned in the result dictionary.

Forward and backward must use the same `alpha`, `scaling_seqlen`, mask
configuration, and sequence metadata. `scaling_seqlen` must be positive. It is
a runtime normalization factor rather than the number of valid tokens in each
individual packed sequence.

The first call for a new static kernel configuration JIT-compiles a CuTe DSL
kernel; subsequent calls reuse the in-process compile cache. Execution follows
the current PyTorch CUDA stream through TVM FFI.

Head dimension 256 backward uses dedicated two-CTA kernels, launching the dQ
kernel before the dK/dV kernel. Inputs with non-compact strides are materialized
into compact temporary buffers for this path, and preallocated gradient views
are copied back without changing the public tensor-layout contract.

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
| Arbitrary mask | CUDA `int32` `func` metadata | Forward and backward |

Arbitrary-mask metadata is a specialized kernel contract and cannot be
combined with causal, local, or paged-KV modes.

`func_tensor` has shape `(1, N, L)` and dtype `torch.int32`, where `N` is
positive and odd and `L >= T_q + 256`. For every packed query row, endpoints
`F0, F1, ...` encode the valid-key union
`[0, F0) ∪ [F1, F2) ∪ [F3, F4) ∪ ...`. Intervals are interpreted
independently, so they may overlap and their endpoints need not be globally
ordered; sequence-length bounds still exclude keys outside the current packed
sequence. The first dimension is currently fixed to one; the 256 extra columns
are kernel padding.

For native FP16 and BF16 arbitrary-mask forward and backward, the interface
automatically builds private block metadata from `func_tensor` on every
execution. Forward uses Q-to-K metadata; fused D64/D128 backward uses K-to-Q
metadata. The D256 two-kernel backward builds Q-to-K and K-to-Q views together
from one Q256-by-K128 classification. The device-only builder and attention
kernels run in order on the caller's current CUDA stream; metadata is not
exposed through either public API. Empty blocks are skipped, partially valid
blocks retain the exact token predicate, and fully valid blocks avoid reading
`func_tensor` (sequence-tail blocks still apply packed-length bounds).
Rebuilding on every execution means an in-place change to `func_tensor` is
visible, including during CUDA Graph replay.

Both dtypes use device-built metadata for D64, D128, and D256. The API trusts
device-resident boundary values and does not validate their ordering or range
on the host.

When both the mask and full counts of a metadata row are zero, the owning
attention-kernel tile writes zero directly to its real output rows. D256
kernels retain the required paired-CTA, cluster, and TMEM lifetime protocol
around this zero epilogue. No whole-output initialization is required from the
interface, and the behavior is preserved during CUDA Graph replay.

Forward also has a causal paged-KV path using `paged_kv`, `page_ids`, and
`page_indptrs`. It requires a page size of 128 and cannot be combined with
local or arbitrary masking. Paged KV is not supported by backward.
`paged_kv` has shape `(num_pages, 2, 128, H, D)` with `num_pages > 0`; K/V is
selected by the second dimension. `page_ids` is a contiguous one-dimensional
`torch.int32` array of physical page indices. `page_indptrs` is a contiguous
`torch.int32` array of shape `(B + 1,)` delimiting each sequence's slice in
`page_ids`. Their values must already be valid for `paged_kv`; they are consumed
on the GPU without host-side range or monotonicity checks.

## Support matrix

| Direction | Architecture | Dtype | Head dimension | Attention |
| --- | --- | --- | --- | --- |
| Forward | Blackwell SM100/SM103 | FP16, BF16 | 64, 128, 256 | MHA |
| Backward | Blackwell SM100/SM103 | FP16, BF16 | 64, 128, 256 | MHA |

## Current limitations

- Backward is nondeterministic; requesting deterministic backward raises
  `NotImplementedError`.
- GQA and MQA are not supported.
- Padded BHSD/BSHD inputs and separate `seqused_q`/`seqused_k` valid-length
  tensors are not supported; use packed THD tensors and cumulative lengths.
- The implementation requires NVIDIA Blackwell SM100/SM103.
- `max_seqlen_q` must be less than or equal to `max_seqlen_k`.
