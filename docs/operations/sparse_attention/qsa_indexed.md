# Indexed QSA training

These attention calls **consume existing indices**; they do not compute the
indexer score or top-k. See QSA entry points and the indexer ([docs](/operations/sparse_attention/qsa#finding-qsa-attention-and-its-indexer), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa.md#finding-qsa-attention-and-its-indexer))
for block selection, bitmap training and the internal two-pass backend.

`cudnn.ops.qsa_indexed_forward` and `qsa_indexed_backward` accept packed token
indices directly. The implementation lives in `cudnn.QSA.indexed` and shares
QSA's attention kernels. It supports BF16
12Q/1KV and head dimension 256 on SM100/SM103 (CuTeDSL >= 4.7) and
SM107 (CuTeDSL >= 4.8 with native SM107 compiler support). The architecture
coverage here applies to forward and general-selection backward; the stricter
`causal_topk` backward has the requirements listed below.

```python
from cudnn.ops import qsa_indexed_forward, qsa_indexed_backward

out, out_fp32, lse = qsa_indexed_forward(q, k, v, indices, scale=1 / 16)
dq, dk, dv = qsa_indexed_backward(grad_out, q, k, v, out_fp32, lse, indices, scale=1 / 16)
```

These explicit calls return `TupleDict` objects in the order shown. They do not
register an autograd function. Save the original FP32 output and natural-log
LSE; casting the BF16 output back to FP32 is insufficient.

| Tensor | Shape | Dtype |
|---|---|---|
| Q, output, grad_out, dQ | `[T, 12, 256]` | BF16 |
| K, V, dK, dV | `[T, 1, 256]` | BF16 |
| indices | `[T, W]` | int32 |
| out_fp32 | `[T, 12, 256]` | FP32 |
| lse | `[12, T]` | FP32 |

All tensors are contiguous on one CUDA device. Data tensors require 16-byte
alignment; indices and LSE require four-byte alignment. `T > 0` and
`T * 12 * 256 <= INT32_MAX`. Scale is a finite host scalar.

`W` is a positive multiple of four, at most 2052. Each four-slot group describes
one distinct complete block: `[start, start+1, start+2, start+3]`, or four `-1`
padding slots. A partial causal tail is a consecutive prefix followed by `-1`.
Valid tokens are absolute packed token IDs, belong to the query's document,
and never exceed the query's absolute token ID. A tail begins at the document's
own block origin; packed document starts need not be divisible by four. Groups
may appear in any order and padding groups may occur between selected groups.
Groups must not overlap or repeat within a row. These device-value relationships
are caller contracts; execution does not read indices back to the host.

The Relax layout with 512 complete four-token groups plus a four-slot causal
tail fits directly, including invalid tail slots. A 2051-wide carrier must be
padded explicitly by the caller; this API does not silently convert it. Empty
rows produce zero output/gradients and negative-infinity LSE. Selection is not
differentiated. Pass the same selection and scale to forward and backward.

A GPU hash table compacts the selected starts. Backward unions adjacent queries'
starts with owner bits, retaining each query's bounds and mask. Nonfinite K/V
uses the device-side per-query fallback described in QSA ([docs](/operations/sparse_attention/qsa), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa.md)). Every call
rebuilds metadata, including during CUDA Graph replay. Warm up before capture.

The public wrappers allocate outputs and one workspace. The internal APIBase
plan validates current storage and uses only caller-owned buffers in execute;
its cache retains immutable geometry and compiled kernels. Indexed metadata
storage is proportional to `T * W`, independent of the full key-block count.
At 128K with W=2052, compact IDs occupy 256.5 MiB in either direction, compared
with 16 GiB for bitmap-forward IDs or 8 GiB for paired bitmap-backward IDs. The
4 GiB bitmap is also unnecessary. These are metadata sizes, not total process
memory; saved state, outputs and FP32 gradient accumulators are additional.

## Complete blocks and causal tails

For top-512-block selection, backward can use the bounded-workspace
two-pass implementation through the same public entry point:

```python
# cu_seqlens is a CUDA int32 tensor, for example [0, 43691, 87382, 131072].
out, out_fp32, lse = qsa_indexed_forward(q, k, v, indices)
dq, dk, dv = qsa_indexed_backward(
    grad_out, q, k, v, out_fp32, lse, indices,
    selection_mode="causal_topk", cu_seqlens=cu_seqlens,
    max_seq_len=43691,
)
```

`selection_mode="general"` remains the default. The new mode declares the
following stricter input contract; it is never inferred from tensor shapes:

- Q/K/V, grad_out and saved FP32 output are finite. LSE is finite and comes
  from forward with the same indices and scale. The general path's nonfinite
  fallback does not apply to this mode.
- `W=2052` and `scale=1/16`. For query position `p` within its document, select
  exactly `min((p + 1) // 4, 512)` distinct, complete four-token blocks from
  that document's causal prefix, followed by its incomplete causal tail.
  Complete blocks fill consecutive groups starting at column zero, in any
  order. No padding holes or empty selections are allowed. Before 512 blocks
  are available, pad the unused groups with `-1`; the tail may follow the
  complete blocks or occupy the last four slots, as in Relax. After 512 blocks
  are available, the tail occupies the last four slots. Invalid tail slots
  and all unused slots are `-1`.
- Packed inputs require `cu_seqlens`: contiguous device int32 prefix sums
  `[B + 1]`, starting at zero, ending at `T`, and nondecreasing. Empty documents
  and document starts not divisible by four are supported. Indices use
  absolute packed token IDs; block origins restart at each document boundary.
  Without `cu_seqlens`, all `T` tokens are one document.
- Optional `max_seq_len` is a host integer in `[1, T]` bounding every document
  length. It defaults to `T`. Every replay must respect this bound. The kernel
  rereads device `cu_seqlens` on replay; boundaries are not cached.

These device-value promises are caller contracts, not host-side checks.
Use `general` for arbitrary selections. Passing `cu_seqlens` or `max_seq_len`
with `general` is rejected because that mode obtains its bounds from indices.
The new keyword-only arguments preserve existing positional calls.

Sparse 12Q/1KV two-pass execution requires the **CUDA 13 CuTeDSL compiler** in
addition to CuTeDSL >= 4.7 and SM100/SM103. A CUDA 13 PyTorch build alone does
not establish the DSL compiler version; unsupported compiler builds are
rejected before kernel compilation. See two-pass details ([docs](/operations/sparse_attention/qsa/qsa_two_pass), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa_two_pass.md))
for the bounded records and workspace sizing. At 128K, 12Q/1KV uses about
2.487 GiB of workspace, including 0.376 GiB of reusable sparse records.

The indexed entry is useful when a producer already has token IDs. It can be
slightly slower than the bitmap entry at some lengths. Use the unchanged Relax
comparison in `benchmark/qsa/benchmark_relax.py` for the intended workload;
Miles ratios do not establish a speedup over Relax, and attention microbenchmarks
do not measure selector, TE, tensor-parallel communication or model throughput.
