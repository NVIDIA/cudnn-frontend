# Flex Attention

**This is an experimental API and subject to change.**

## Overview

Flex Attention implements scaled dot-product attention with a reusable sparse
mask plan:

```text
O = softmax(scale * Q K^T + interval_mask) V
```

Rather than materializing a dense Boolean mask, every query row provides an
odd-length endpoint sequence whose visible keys are the interval union

```text
[0, F0) U [F1, F2) U [F3, F4) U ...
```

The endpoint representation supports a range of sparse attention patterns.
Rows in the following figure are queries, columns are keys, blue cells are
visible, and light-gray cells are masked:

![Supported static attention mask shapes](assets/static_mask_shapes.png)

`create_mask_plan` compiles these endpoints into architecture-native packed
forward and, when requested, backward metadata. The resulting `MaskPlan` can
be reused with new Q/K/V values that have the same geometry, dtype, and CUDA
device. A single `flex_attn_func` entry point handles both layouts: supplying
`cu_seqlens_q` and `cu_seqlens_k` when the plan is created selects the THD
variable-length path; omitting both selects fixed-length BSHD. Execution uses
the `FlexAttentionFwd` and `FlexAttentionBwd` `APIBase` adapters. Their
`check_support()`, `compile()`, and `execute()` lifecycle supports precompiled,
allocation-free launches with caller-provided outputs and workspace;
`flex_attn_func` composes the allocating wrappers behind PyTorch custom
autograd.

## Requirements

- SM90, SM100, or SM103 NVIDIA GPU
- CUDA-enabled PyTorch
- cuDNN Frontend (its CuTeDSL dependencies are required and install with it)

From a source checkout:

```bash
pip install -e .
pip install --group torch
```

For a published package, install `nvidia-cudnn-frontend` and a
compatible CUDA-enabled PyTorch separately.

## Fixed-length BSHD API

```python
import torch
from cudnn import create_mask_plan, flex_attn_func

B, S, H, D = 1, 4096, 8, 128
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Causal row q sees local key indices [0, q + 1).
mask_func = torch.arange(1, S + 1, device="cuda", dtype=torch.int32).view(1, 1, S)
plan = create_mask_plan(mask_func, q, k, v)

out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
out.backward(torch.randn_like(out))
```

Fixed-length tensor shapes are:

| Tensor | Shape | Dtype |
|---|---|---|
| `q` | `[B, Sq, Hq, Dqk]` | FP16 or BF16 |
| `k` | `[B, Sk, Hkv, Dqk]` | same as `q` |
| `v` | `[B, Sk, Hkv, Dv]` | same as `q` |
| `mask_func` | `[Hmask, nfunc, B * Sq]` | CUDA INT32 |
| `out` | `[B, Sq, Hq, Dv]` | same as `q` |
| `lse` | `[B, Hq, Sq]` | FP32 |

`Hq` must be divisible by `Hkv`, supporting MHA, GQA, and MQA. `Hmask` must
be `1` for a mask shared by all query heads or `Hq` for per-head masks.
`nfunc` must be positive and odd. The last mask dimension is flattened in
batch-major order, but every endpoint remains a sample-local key coordinate
in `[0, Sk]`. Endpoints must be nondecreasing within each query row.

With `return_lse=False` (the default), `flex_attn_func` returns only `out`.
With `return_lse=True`, it returns `(out, lse)`. LSE is still maintained
internally when gradients require it.

## Variable-length THD API

Variable-length Q/K/V are flattened across samples. Sequence geometry is
provided when the plan is built and is then owned by the plan:

```python
import torch
from cudnn import create_mask_plan, flex_attn_func

q_lengths = (192, 128)
k_lengths = (160, 96)
cu_q = torch.tensor((0, 192, 320), device="cuda", dtype=torch.int32)
cu_k = torch.tensor((0, 160, 256), device="cuda", dtype=torch.int32)
Hq, Hkv, D = 8, 2, 128

q = torch.randn(320, Hq, D, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(256, Hkv, D, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(256, Hkv, D, device="cuda", dtype=torch.float16, requires_grad=True)

# Each query sees its whole sample-local K sequence.
endpoints = torch.cat(
    [torch.full((q_length,), k_length, device="cuda", dtype=torch.int32)
     for q_length, k_length in zip(q_lengths, k_lengths)]
)
mask_func = endpoints.view(1, 1, -1)

plan = create_mask_plan(
    mask_func,
    q,
    k,
    v,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=max(q_lengths),
    max_seqlen_k=max(k_lengths),
)
out, lse = flex_attn_func(q, k, v, mask_plan=plan, return_lse=True)
```

In this mode Q has shape `[total_q, Hq, Dqk]`, K and V have leading extent
`total_k`, `out` has shape `[total_q, Hq, Dv]`, and LSE has shape
`[Hq, total_q]`. Both cumulative-length tensors are contiguous CUDA INT32
vectors of shape `[B + 1]`, start at zero, end at the corresponding total,
and describe lengths no greater than the supplied maxima. The plan clones
them, so later mutations of the caller's tensors do not change plan geometry.

## Plan and execution options

`create_mask_plan(..., pack_gqa=None, build_backward=None)` accepts:

- `pack_gqa`: select packed-GQA planning explicitly, or leave `None` for
  architecture/configuration-based selection.
- `build_backward`: build backward topology explicitly. With `None`, it is
  enabled when autograd is active and any Q/K/V sample tensor requires a
  gradient. A plan built without backward payloads cannot later be used for
  an autograd-enabled call requiring gradients.

The execution function accepts `softmax_scale` (default
`1 / sqrt(Dqk)`), `deterministic`, and `return_lse`. Plan reuse requires the
same fixed/variable mode, sequence and head geometry, dtype, and device used
at construction.

Keep `MaskPlan` outside the execution loop. Q/K/V values and tensor addresses
may change between calls, while the mask and plan geometry stay fixed:

```python
plan = create_mask_plan(mask_func, q_sample, k_sample, v_sample, build_backward=True)

for q, k, v in inputs_with_the_same_geometry:
    out = flex_attn_func(q, k, v, mask_plan=plan)
    out.backward(torch.randn_like(out))
```

The plan owns immutable forward metadata and, with `build_backward=True`, the
backward metadata. Mutable scheduler counters, semaphores, and accumulation
buffers are per-execution workspace rather than plan state, so one plan can be
used by independent calls.

## Explicit compile and execute APIs

Most PyTorch users should use `flex_attn_func`. Framework integrations can use
`FlexAttentionFwd` and `FlexAttentionBwd` directly. Constructors capture sample
tensor descriptors and a sample plan; `execute()` accepts new tensors and any
compatible plan. Outputs and the `workspace_size`-byte CUDA `uint8` workspace
must be allocated before `execute()`:

```python
import torch
from cudnn import FlexAttentionBwd, FlexAttentionFwd

plan = create_mask_plan(mask_func, q, k, v, build_backward=True)
out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device)
fwd = FlexAttentionFwd(q, k, v, out, plan, lse)
fwd.check_support()
fwd.compile()

fwd_workspace = torch.empty(fwd.workspace_size, device=q.device, dtype=torch.uint8)
fwd.execute(q, k, v, out, plan, lse, workspace=fwd_workspace)

do = torch.randn_like(out)
dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
bwd = FlexAttentionBwd(q, k, v, out, do, lse, dq, dk, dv, plan)
bwd.check_support()
bwd.compile()

bwd_workspace = torch.empty(bwd.workspace_size, device=q.device, dtype=torch.uint8)
bwd.execute(q, k, v, out, do, lse, dq, dk, dv, plan, workspace=bwd_workspace)
```

The same `plan`, compiled API objects, output buffers, and workspaces can be
reused for sequential calls with new tensor values that keep the compiled
shape, stride, dtype, and device. Concurrent calls may share the immutable
plan and compiled API objects, but each in-flight call needs separate outputs
and workspace.

`flex_attn_func` uses internal allocating helpers to manage outputs, workspaces,
compiled-object caches, and autograd state. Explicit backward requires the LSE
produced by forward and a plan built with backward metadata.

## Supported configurations and current limits

- FP16 and BF16 inputs; FP32 LSE
- fixed-length BSHD and true variable-length THD layouts
- `(Dqk, Dv)` with each dimension in `{8, 16, ..., 128}`, plus `(192, 128)`
  and the dedicated `(256, 256)` path
- forward and backward on SM90, SM100, and SM103

Paged KV cache, SplitKV, MLA, FP8, SM80, and SM120 are not implemented.
Q/K/V must be contiguous in their last dimension. The plan builder may impose
additional architecture-specific shared-memory constraints and reports them
as validation errors.

## Benchmark

The Flex-only static-mask benchmark and its protocol are documented in
[`benchmark/flex_attention/README.md`](../../../benchmark/flex_attention/README.md).
