# Indexed QSA training

These attention calls **consume existing indices**; they do not compute the
indexer score or top-k. See [QSA entry points and the indexer](qsa.md#finding-qsa-attention-and-its-indexer)
for block selection, bitmap training and the SM100 backward engine.

`cudnn.ops.qsa_indexed_forward` and `qsa_indexed_backward` accept packed token
indices directly. The implementation lives in `cudnn.QSA.indexed` and shares
QSA's attention kernels. It supports BF16
12Q/1KV, head dimension 256, and SM100/SM103 with CuTeDSL >= 4.7 (the backward
needs the CUDA 13 build of the DSL).

```python
from cudnn.ops import qsa_indexed_forward, qsa_indexed_backward

out, out_fp32, lse = qsa_indexed_forward(q, k, v, indices, scale=1 / 16)
dq, dk, dv = qsa_indexed_backward(grad_out, q, k, v, out_fp32, lse, indices, scale=1 / 16, cu_seqlens=None)
```

These explicit calls return `TupleDict` objects in the order shown. They do not
register an autograd function. Save the original FP32 output and natural-log
LSE; casting the BF16 output back to FP32 is insufficient. `cu_seqlens` (int32
`[B + 1]` prefix sums) describes the documents of a packed buffer to the
backward; without it the rows are one sequence, which stays exact for packed
documents but serves only the first document on the dense prefix path.

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

A GPU hash table compacts the selected starts. Backward runs the engine
described in [QSA](qsa.md) over them; a nonfinite K/V value of a masked
token adds exact zeros, one of a selected token poisons its queries. Every call
rebuilds metadata, including during CUDA Graph replay. Warm up before capture.

The public wrappers allocate outputs and one workspace. The internal APIBase
plan validates current storage and uses only caller-owned buffers in execute;
its cache retains immutable geometry and compiled kernels. Indexed metadata
storage is proportional to `T * W`, independent of the full key-block count.
At 128K with W=2052, compact IDs occupy 256.5 MiB in either direction, compared
with 16 GiB for the bitmap entry's `[T, NB]` IDs. The 4 GiB bitmap is also
unnecessary. These are metadata sizes, not total process
memory; saved state, outputs and FP32 gradient accumulators are additional.

The indexed entry is useful when a producer already has token IDs. Use the unchanged Relax
comparison in `benchmark/qsa/benchmark_relax.py` for the intended workload;
Miles ratios do not establish a speedup over Relax, and attention microbenchmarks
do not measure selector, TE, tensor-parallel communication or model throughput.
