# Block Sparse Attention (BSA)

**This is an experimental API and subject to change.**

## Overview

Block Sparse Attention computes non-causal scaled dot-product attention over a
block-level sparse pattern. For query token `i`, let `m = floor(i / block_size)`
be its query-block index and let `K_m` be the union of the key/value blocks
listed for that query block. The operation is

$$
O_i = \operatorname{softmax}_{j \in K_m}
\left(\frac{Q_i K_j^T}{\sqrt{D}}\right)V_j.
$$

The softmax is normalized once over all valid tokens in the selected blocks.
`block_sizes` can shorten individual key/value blocks so that padded tokens do
not participate in the softmax.

Unlike [NSA Selection Attention](nsa.md#1-selection-attention), BSA supplies one
list of key/value blocks per **query block**. NSA Selection supplies routing
metadata at query-token granularity and is one component of the larger NSA
pipeline.

This package contains only Python CuTe DSL/JIT kernels. The C++/CUTLASS AOT
extension from the source repository is not included.

## Installation

Install the CuTe DSL optional dependencies:

```bash
pip install nvidia-cudnn-frontend[cutedsl]
```

## Forward

```python
import torch
from cudnn import BSA

q = torch.randn(1, 8, 1024, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)

# For the SM90 blk64 path, this example has 16 Q blocks and 32 KV blocks.
# Every Q block attends to the first four KV blocks.
q2k_block_index = torch.arange(4, device="cuda", dtype=torch.int32)
q2k_block_index = q2k_block_index.view(1, 1, 1, 4).expand(1, 8, 16, 4).contiguous()
block_sizes = torch.full((32,), 64, device="cuda", dtype=torch.int32)

result = BSA.block_sparse_attention_forward(
    q,
    k,
    v,
    q2k_block_index,
    block_sparse_num=4,
    block_sizes=block_sizes,
    sparse_block_size=64,
)
o, lse = result
# The same values are available as result["o_tensor"] and result["lse_tensor"].
```

The default layout is `BHSD`. Pass `layout="bshd"` for tensors in `BSHD`
layout. The output follows the input layout, while `lse` is always the FP32
natural-log log-sum-exp with shape `(B, H_q, S_q)`.

### Sparse metadata

| Argument | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `q2k_block_index` | `(B, H_q, N_q, K_max)` | `int32` | KV-block IDs; only the valid prefix of each last dimension is read |
| `block_sparse_num` | scalar | Python `int` | Fixed valid-prefix length for every query block |
| `q2k_block_nums` | `(B, H_q, N_q)` | `int32` | Optional per-query-block valid-prefix lengths; overrides `block_sparse_num` |
| `block_sizes` | `(N_kv,)` or backend-supported batched form | `int32` | Number of valid tokens in each KV block |

Here `N_q = ceil(S_q / sparse_block_size)` and
`N_kv = ceil(S_kv / sparse_block_size)`. Except for packed GQA described below,
metadata is per query head. The metadata tensors must reside on the same CUDA
device as `q`, `k`, and `v`. The value ranges below are a hard caller contract.

#### Value ranges (caller contract)

Let `K_max = q2k_block_index.shape[-1]`. Only the active prefix of each
`q2k_block_index` row — the first `q2k_block_nums[b, h, m]` entries with
variable counts, or the first `block_sparse_num` entries with a fixed count —
is consumed. Values in the inactive suffix are ignored.

- Within each row, active `q2k_block_index` values must be unique integers in
  `[0, N_kv)`.
- For forward with variable counts, every `q2k_block_nums` value must be in
  `[0, K_max]` when `allow_empty_block_nums=True`, and in `[1, K_max]`
  otherwise. Backward variable counts may be in `[0, K_max]`.
- With fixed counts, `block_sparse_num` must be in `[1, K_max]`. The
  SM100/SM110 blk128 path additionally requires an even value, i.e. an even
  `block_sparse_num` in `[2, K_max]`.
- The `block_sizes` entry for every physical KV block referenced by an active
  `q2k_block_index` value must be in `[1, sparse_block_size]`. Entries for
  unreferenced physical KV blocks are ignored. A zero-sized referenced block is
  not supported; use `q2k_block_nums` (or the sparse index prefix) to drop the
  block instead.

Tensor value ranges and per-row uniqueness are not validated at runtime.
Violating the contract is unsupported and may produce invalid results or
invalid device memory accesses.

On the SM100/SM110 blk128 path, `pack_gqa=None` automatically packs GQA when
the GQA ratio `r = H_q / H_kv` divides 128. Packed metadata has shape
`(B, H_kv, ceil(S_q * r / 128), K_max)`. Pass `pack_gqa=False` to use the
unpacked `(B, H_q, ceil(S_q / 128), K_max)` contract on every architecture.
Explicit `pack_gqa=True` requires `r` to divide 128.

When `block_sizes=None`, each referenced physical KV block is treated as full.
Provide `block_sizes` whenever a referenced final block is only partially
valid.

`sparse_block_size=None` chooses blk64 on SM90/SM120 and blk128 on
SM100/SM110. Passing `sparse_block_size=64` explicitly selects the SM100/SM110
blk64 CuTe DSL path, whose shape support is narrower. `kv_splits` is available
on SM90 and the explicit Blackwell blk64 path; `use_clc` applies only to the
explicit Blackwell blk64 path.

`kv_splits=2..256` computes FP32 partial outputs and combines them, with
workspace growing linearly in the split count. SM90 accepts an explicit integer
split count. The SM100/SM110 blk64 path also accepts `kv_splits="auto"`; CLC is
disabled for split execution, and an explicit `use_clc=True` is incompatible
with `kv_splits>1`. Every sparse row must contain at least the selected number
of valid KV blocks so that every split is non-empty; this caller contract is not
validated at runtime. Automatic split selection uses metadata capacity rather
than per-row count values, so variable-count callers must also satisfy the
automatically selected split count. It falls back to a smaller split count when
the estimated live workspace does not fit the available CUDA allocator budget;
an explicit split count that exceeds that budget raises `RuntimeError`.

## Backward

Backward is an explicit API rather than a registered PyTorch autograd
operation. It recomputes probabilities from the forward output and LSE:

```python
dout = torch.randn_like(o)
grads = BSA.block_sparse_attention_backward(
    dout,
    q,
    k,
    v,
    o,
    lse,
    q2k_block_index,
    block_sparse_num=4,
    block_sizes=block_sizes,
    sparse_block_size=64,
)
dq, dk, dv = grads
```

The result keys are `dq_tensor`, `dk_tensor`, and `dv_tensor`. Optional
preallocated `dq_tensor`, `dk_tensor`, and `dv_tensor` arguments are supported.
The backward implementation builds a bucketed K-to-Q CSR task layout on the
GPU. `bucket_size_blocks` is an optional tuning override; leaving it unset uses
the backend default.

Backward defaults to blk64 on SM90 and blk128 on SM100/SM110. Pass the same
explicit `sparse_block_size` used by forward when selecting the Blackwell blk64
path. SM100/SM110 blk128 backward does not yet consume `block_sizes`; it
therefore requires full physical KV blocks and `block_sizes=None`.

## Current support

### Forward

| Architecture | Sparse block | Dtype | QK / V dimensions | Attention |
| --- | ---: | --- | --- | --- |
| SM90 | 64 | FP16, BF16 | each of 64, 96, 128 | MHA, GQA, MQA |
| SM100/SM110 | 128 | FP16, BF16 | QK=V=64, 96, or 128 | MHA, GQA, MQA |
| SM100/SM110 | 64 (explicit) | BF16 | QK=128, V=128 | MHA |
| SM120 | 64 | FP16, BF16 | QK=128, V=128 | MHA, GQA, MQA |

SM90 currently requires `S_q` to be a multiple of 64. Its fixed count may be
any positive value. The SM100/SM110 blk128 fixed count must be even and at
least two; SM120 and the explicit Blackwell blk64 path accept any positive
fixed count. Variable counts use `q2k_block_nums`. `allow_empty_block_nums`
defaults to `False`; when it is `True`, empty rows (`q2k_block_nums == 0`)
produce `O = 0` and `LSE = -inf`. SM90 selects the empty-row handling as a
compile-time specialization, so the default non-empty configuration keeps its
branch-free fast path. Split-KV execution therefore excludes empty rows.

### Backward

| Architecture | Sparse block | Dtype | Head dimension | Attention |
| --- | ---: | --- | ---: | --- |
| SM90 | 64 | BF16 | 128 | MHA |
| SM100/SM110 | 64 | BF16 | 128 | MHA |
| SM100/SM110 | 128 | BF16 | 64 or 128 | MHA |

Backward is not implemented for SM120. It requires equal QK/V dimensions and
does not currently support GQA/MQA.

## Limitations

The current sparse kernels do not implement causal or local masking, dropout,
`mask_mod`, `score_mod`, paged KV cache, softcap, or variable-length packed
sequences. Inputs must be rank four, use FP16/BF16 as allowed above, and have a
contiguous last (head-dimension) axis. Forward outputs are contiguous in the
requested BHSD or BSHD layout, including split-KV execution.

Compilation is lazy. The first call for a new static configuration JIT-compiles
the relevant kernel; subsequent calls reuse an in-process cache.

The current public surface consists of allocating function wrappers under
`cudnn.BSA`; there is no separate `APIBase` class or explicit `compile()`
lifecycle for BSA.

Correctness tests and FP32 references are under
`test/python/fe_api/bsa`.

## Source provenance

The kernel sources were adapted from the
[`Block-Sparse-Attention`](https://github.com/NVIDIA-JerryChen/Block-Sparse-Attention/tree/a9fa5f2966aa17fcf1ce2890c489d45a4a89acf1)
checkout at commit `a9fa5f2966aa17fcf1ce2890c489d45a4a89acf1`. See
`python/cudnn/block_sparse_attention/PROVENANCE.md` for the migrated scope and
integration changes.
