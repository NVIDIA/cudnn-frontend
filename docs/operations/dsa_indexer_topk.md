# Fused DSA indexer and exact top-k

This experimental API uses Cake's deterministic BF16 indexer on SM100a,
SM103a, and SM107a. It requires CUDA Toolkit 13.4 or newer. Kernels compile
on first use; warm the geometry before CUDA graph capture.

```python
from cudnn.ops import dsa_indexer_topk_forward

indices, scores = dsa_indexer_topk_forward(q, k, weights, cu_q, cu_k, top_k=2048)
```

The result is `TupleDict(indices=..., scores=...)`. This is a selection
operation with no autograd registration. It does not implement sparse
attention or the indexer's score-gradient operation.

| Input | Shape | Type |
| --- | --- | --- |
| `q` | `[Tq, 32, 128]` | BF16 |
| `k` | `[Tk, 128]` | BF16 |
| `weights` | `[Tq, 32]` | FP32 |
| `cu_seqlens_q`, `cu_seqlens_k` | `[B + 1]` | INT32 |
| Optional `q_causal_offsets` | `[B]` | INT64 |

Inputs reside on one CUDA device. Q and weights are contiguous and
16-byte aligned. K supports a row stride of at least 128 elements that is
a multiple of eight, with unit inner stride and a 16-byte-aligned base.
Prefixes and offsets are contiguous and aligned to their element size.
Packed query and key totals and the segment count must be positive.

`top_k` is in `[1, 4096]`. The score is
`sum_h(weights[t,h] * relu(softmax_scale * dot(q[t,h], k[j])))`.
The scale defaults to `1 / sqrt(128)` and must be finite and positive.
Any extra head weighting belongs in `weights`.

For query position `u` in one segment, key `j` is visible when
`j < floor((offset + u + 1) / ratio)`, within that key segment.
`ratio` is a positive integer. The offset is the supplied segment offset,
or `key_length - query_length` for ratio one, or zero for other ratios.

The kernel selects by descending score, breaking ties with the larger
segment-local key ID. Results are returned in ascending ID order. Unused
slots hold `-1` and negative infinity. Empty segments and rows with no
visible keys are supported within nonempty packed operands.

Prefix values are a device-data contract: each vector starts at zero,
never decreases, and ends at its packed total. Boundaries stay on the GPU.
No dense `[Tq, Tk]` score matrix is materialized.

## Allocation and streams

The wrapper allocates omitted outputs and workspace. Supply `indices`,
`scores`, and `workspace` to avoid device allocations after warmup.
Outputs are contiguous, 16-byte-aligned `[Tq, top_k]` INT32/FP32 tensors.
Use `dsa_indexer_topk_workspace_size(Tq, Tk, B, top_k, ratio=ratio, device=device)`
to size a contiguous, 256-byte-aligned UINT8 workspace. Its size includes
the launch ticket and storage for unread ABI operands. Outputs, scratch,
and inputs must not overlap. Concurrent calls need separate scratch.

`stream=None` uses the operand device's current PyTorch stream. A concrete
PyTorch stream or raw CUDA stream handle can be supplied. Supplied buffers
are recorded on that stream. A cached plan retains no tensor or workspace.

The package is copied unchanged from
[`radixark/miles-ext-kernels` at `f55fb996`](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce).
`python/cudnn/_cake/VENDORED_DSA_INDEXER.json` records revisions and file hashes.
