# Prepared BF16 Indexer Top-K

**This frontend-only API is experimental.** `IndexerTopKVarlen` selects row-relative
indices from variable-length BF16 score rows on GB300 (SM103) and Rubin (SM107).
The prepared class writes to caller-owned output and needs no scratch workspace.
The convenience wrapper allocates the output.

## Operation

For scores of shape `(T, N)`, lengths of shape `(T // next_n,)`, and row `r`,
the eligible prefix length is

```text
b = r // next_n
offset = r % next_n
L[r] = min(N, max(0, lengths[b] - (next_n - 1) + offset) // compress_ratio)
```

Length arithmetic widens to signed 64 bits before subtracting or dividing.
The result contains `min(K, L[r])` distinct indices into `scores[r, :L[r]]`
whose values form a largest-K multiset. Remaining slots contain `-1`.
Rows with `L[r] <= K` contain the eligible indices followed by padding.
Selected indices are not sorted by value. Selection among equal values is
unspecified; positive and negative zero compare as equal values.

The operator only selects indices. It does not generate indexer scores,
compress KV storage, map logical indices to physical cache pages, or compute
attention. Those operations remain the caller's responsibility.

## Supported inputs

| Argument | Contract |
| --- | --- |
| `scores` | Contiguous CUDA BF16 tensor, shape `(T, N)` |
| `lengths` | Contiguous CUDA Int32 tensor, shape `(T // next_n,)` |
| `indices` | Contiguous CUDA Int32 output, shape `(T, K)` |
| `T`, `N` | `1 <= T <= 512`, `1 <= N <= 262144` |
| `top_k` | `512`, `1024`, or `2048` |
| `next_n` | Positive integer dividing `T` |
| `compress_ratio` | Integer in `[1, 2147483647]` |

All tensors must be on the same device and have 16-byte-aligned base pointers.
The output must not overlap either input. Lazy negative or conjugate views are
rejected. Eligible scores may contain infinities, but not NaNs. Scores outside
the effective eligible prefix are ignored and may contain NaNs. An empty row
therefore returns only `-1` even when all of its score storage contains NaNs.
Device values are not read back to the CPU for validation.

SM103 requires CuTe DSL 4.7 or newer. SM107 requires 4.8 or newer and a compiler
with SM107 target support. Unsupported architectures or layouts are declined.
The supported input contract is not a guarantee of a speedup for every shape.

## Convenience wrapper

```python
import torch
import cudnn

scores = torch.randn((256, 32768), device="cuda", dtype=torch.bfloat16)
lengths = torch.full((256,), 32768, device="cuda", dtype=torch.int32)

result = cudnn.indexer_top_k_varlen_wrapper(
    scores, lengths, top_k=512, next_n=1, compress_ratio=1,
)
indices = result["indices"]
```

The wrapper returns `TupleDict(indices=...)`. It allocates the output on the
operand device and orders execution on the supplied stream, or the device's
current stream when no stream is supplied.

## Prepared execution

```python
plan = cudnn.IndexerTopKVarlen(scores, lengths, 512, next_n=1, compress_ratio=1)
plan.check_support()
plan.compile()
indices = torch.empty((256, 512), device=scores.device, dtype=torch.int32)
assert plan.scratch_workspace_bytes() == 0
plan.execute(scores, lengths, indices)
```

Preparation stores tensor metadata and compiled code. It does not retain the
sample tensors or allocate device memory. `execute` accepts fresh tensors with
the declared metadata and launches against their current pointers. It performs
no dtype conversion, output allocation, device-to-host read, or synchronization.

Compile before capturing execution in a CUDA graph. During replay, the graph
uses its captured buffer addresses and reads their current contents, including
current lengths. Keep those buffers alive for the graph's lifetime. Independent
streams or graphs may share a plan while using separate output buffers; callers
must order writes to shared input storage.

The existing `IndexerTopK` and `indexer_top_k_wrapper` APIs retain their own
dtype and optional-values contracts. This prepared API returns indices only.
