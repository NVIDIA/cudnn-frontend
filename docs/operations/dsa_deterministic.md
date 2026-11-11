# Cake deterministic sparse attention and Lightning indexer

These explicit frontend APIs launch the pinned Miles Cake deterministic
kernels. They use BF16 queries/keys, FP32 score math, and a fixed four-way
key-gradient reduction. They do not use autograd or register a cuDNN graph
engine. Native stages compile on first use; warm the exact stages before capture.

## Sparse attention

`dsa_det_forward(q,kv,indices)` accepts contiguous BF16 `q[T,H,D]` and
`kv[S,D]`, with H in {8,16,32,64} and D in {512,576}. The first 512 key
channels also supply V. Int32 `indices[T,K]` contains global key rows;
K is positive and divisible by 64. Negative and out-of-range slots are holes.
Native forward and backward stages check each index against the logical key
count before loading KV, including each prefetched index tile.
Repeated indices contribute repeatedly. No causal mask or attention sink is
added. Scale defaults to `1/sqrt(D)`.

Forward returns BF16 `out[T,H,512]` and FP32 **base-2** `lse[T,H]`.
All-hole rows have zero output and the native large-negative finite LSE sentinel. Backward
uses both and returns BF16 `dq` and combined `dkv`.

```python
from cudnn.ops import dsa_prepare_segments, dsa_det_forward, dsa_det_backward

groups = dsa_prepare_segments(indices, kv.shape[0])
out, lse = dsa_det_forward(q, kv, indices)
dq, dkv = dsa_det_backward(q, kv, indices, out, lse, grad_out, **groups)
```

## Lightning indexer

`dsa_indexer_logits(q,k,weights,key_start,key_end)` accepts contiguous
BF16 `q[T,H,128]`, `k[S,128]`, FP32 `weights[T,H]`, and int32 bounds `[T]`.
H is in {8,16,32,64}. Bounds are caller-owned values in `[0,S]`, with
start <= end. It returns `logits[T,S] = sum_h weights * relu(q dot k)`;
there is **no score scale**. Keys outside each `[start,end)` range have
negative-infinity scores.

`dsa_indexer_backward(q,k,weights,indices,grad_scores,**groups)` differentiates
selected scores. It returns BF16 `dq`, FP32 `dk`, and FP32 `dweights`.
Indices are global int32 `[T,K]`; negative and out-of-range IDs are holes. K must be
a multiple of 64 for H<=16 and 32 otherwise. `grad_scores[T,K]` is FP32.
Use `dsa_prepare_segments(indices,S)` for this exact selected pattern.
The fused indexer/top-k API is a separate entry point with its own scale and
segment-local index contract.

## Scratch and segment metadata

`dsa_prepare_segments` allocates and stable-sorts outside execution. It returns
int32 `order`, `seg_start`, and `seg_count`. **Rebuild all three whenever index
values change.** Backward trusts these device values; it checks metadata without
copying values to the host. The grouping tensors must belong to the exact index
matrix and key count passed to backward. Preparation is rejected during capture.

All execution APIs accept caller output buffers and `stream=`. Attention forward
also accepts a 256-byte `workspace`; backward scratch is reported by
`dsa_det_workspace_size(T,H,D,S,K,indexer=False)` (use `indexer=True` for indexer
backward). Scratch is a contiguous CUDA uint8 vector aligned to 256 bytes.
Outputs and scratch must not overlap inputs, grouping tensors, or each other.

With all buffers supplied, warmed execution allocates no tensor storage and reads
no tensor values on the host. CUDA graph replay may change Q/K/weights/gradients.
Changing indices also requires matching updated grouping values at the captured
addresses. Keep all captured storage alive. Scratch holds every query-row partial;
this adapter does not chunk the query rows to cap memory.

The native manifests serve SM100/103/107. Toolkit floors follow the shared Cake
runtime. Validation in this change covers SM100 with CUDA 13.4; other architectures
were not run. The adapter uses the register-MMA stages and the 64-head cluster pair.
