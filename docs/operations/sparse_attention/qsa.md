# Query Sparse Attention (QSA)

QSA applies attention to the four-token blocks selected for each query. It uses
the original K/V tensors; block selection is an input and receives no gradient.
The API follows the Miles packed training interface.
The source contract is Miles
[`block_sparse_attention.py` at `0cc6516`](https://github.com/radixark/miles/blob/0cc65167f991ab1f0d228f6c9b648d89bc60a203/miles/kernels/attention/qsa/block_sparse_attention.py).

```python
from torch import Tensor
from cudnn.ops import qsa_forward, qsa_backward

def qsa_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    block_first: Tensor,
    block_last: Tensor,
    scale: float,
    block_size: int = 4,
) -> TupleDict:
    ...  # out, out_fp32, lse

def qsa_backward(
    grad_out: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out_fp32: Tensor,
    lse: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    block_first: Tensor,
    block_last: Tensor,
    scale: float,
    block_size: int = 4,
) -> TupleDict:
    ...  # dq, dk, dv
```

Both calls return a `TupleDict`. Tuple unpacking follows the order above; string
keys have the same names. Forward and backward are explicit calls. The forward
call does not register a PyTorch autograd function.

## Finding QSA attention and its indexer

QSA's public functions use the `cudnn.ops.qsa_*` prefix. Their implementations
live under `cudnn.QSA`; the selector shares scorer and top-k kernels with DSA.
Choose an entry by its input and saved-state contract:

| Need | Entry point | Contract |
|---|---|---|
| Packed training with a block-selection bitmap | `cudnn.ops.qsa_forward`, `qsa_backward` | The bitmap API documented on this page; 24Q/2KV, 12Q/1KV or 6Q/1KV |
| Packed training with selected token IDs | `cudnn.ops.qsa_indexed_forward`, `qsa_indexed_backward` ([docs](/operations/sparse_attention/qsa/qsa_indexed), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa_indexed.md)) | 12Q/1KV or 6Q/1KV on SM100/SM103/SM107; shares the QSA attention kernels |
| Generate selected blocks from indexer Q and compressed K | `cudnn.ops.qsa_select_forward`, `qsa_select_decode_forward` | QSA selector over the shared DSA scorer and top-k; returns local block IDs, not token IDs |
| Decode selection over a paged compressed-key cache | `cudnn.ops.qsa_select_decode_paged_forward` | BF16 indexer cache; see the paged selector's architecture and layout requirements in its docstring |
| Integrated projection, compression, selection and attention | `GatedAttentionBlockFwd` with `QsaSpec(index_source="indexer")` ([docs](/operations/Attention/gated_attention_block), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/attention/gated_attention_block.md)) | Use the integrated block's own architecture, layout and inference support matrix |

**Indexed attention consumes indices; the indexer generates them.** The indexed
training calls do not run scoring or top-k. The selector's local block IDs need
expansion to the attention input format, including the incomplete causal tail
and the packed sequence's token origin. Its standalone wrapper can be imported
as follows:

```python
from cudnn.ops import qsa_select_forward, qsa_select_decode_forward, qsa_select_decode_paged_forward
```

QSA's selector scores each complete compressed block as
`scale * sum_h relu(qi[t, h] @ kbar[block])`. The existing
`cudnn.DSA.indexer_forward_top_k_wrapper` provides this ratio-causal scorer and
top-k operation with compression ratio four and constant head weights. Sharing
these kernels avoids a second implementation. QSA-specific compression, block
visibility and tail handling remain part of the QSA contract; this reuse does
not make QSA and DeepSeek attention interchangeable. Numerical rounding and
top-k tie rules must also be checked before replacing an external selector.

The internal two-pass backend ([docs](/operations/sparse_attention/qsa/qsa_two_pass), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa_two_pass.md)) has a different
saved-state and selection contract: 12Q/1KV or 24Q/2KV, canonical block lists,
optional packed `cu_seqlens`, and native FP32 saved output / `[Hq, T]` LSE
(or the earlier BF16 saved output / `[T, Hq]` LSE pair). For 12Q/1KV, the public
`qsa_indexed_backward(..., selection_mode="causal_topk")` selects this backend
when the caller declares the stricter complete-block and causal-tail contract ([docs](/operations/sparse_attention/qsa/qsa_indexed#complete-blocks-and-causal-tails), [GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/operations/sparse_attention/qsa_indexed.md#complete-blocks-and-causal-tails)).
The default indexed and bitmap calls retain their general selection contracts.
The implementation remains under `cudnn.QSA`; no public op class or
algorithm-specific entry is added.

The experimental QSA entries previously under `cudnn.DSA` and
`cudnn.gated_attention_block.qsa_select*` have moved without compatibility
aliases. These entries were absent from stable v1.31.0; users of development
builds should update their imports to the table above. DSA's shared indexer
APIs remain in DSA. The integrated gated-attention block remains its own API.

## Tensors

`T` is the total packed token count. `NB` is the total number of blocks across
all sequences. All tensors must be contiguous and on the same CUDA device.

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `q` | `[T, Hq, 256]` | BF16 | Query vectors |
| `k`, `v` | `[T, Hkv, 256]` | BF16 | Key and value vectors |
| `sel_blocks` | `[T, NB]` | uint8 | Nonzero selects a global block for that query |
| `lo`, `hi` | `[T]` | int32 | Inclusive allowed key-token bounds |
| `blk_base` | `[T]` | int32 | First global block of the query's sequence |
| `tok_base` | `[T]` | int32 | First global token of the query's sequence |
| `block_first`, `block_last` | `[NB]` | int32 | Inclusive global token span of each block |
| `out` | `[T, Hq, 256]` | BF16 | Attention output |
| `out_fp32` | `[T, Hq, 256]` | FP32 | Output before its BF16 cast; save for backward |
| `lse` | `[Hq, T]` | FP32 | Natural-log softmax normalizer; save for backward |
| `grad_out` | `[T, Hq, 256]` | BF16 | Upstream output gradient |
| `dq` | Same as `q` | BF16 | Query gradient |
| `dk`, `dv` | Same as `k`, `v` | BF16 | Key/value gradients, summed over query heads |

Q/K/V, `grad_out`, and `out_fp32` must be 16-byte aligned. FP32 LSE and int32
metadata must be 4-byte aligned. The implementation rejects unsupported layouts
before launch. It does not copy or repack inputs.

## Selection and components

For query `t` and key token `j`, the Miles block lookup is:

```python
b = blk_base[t] + (j - tok_base[t]) // block_size
allowed = lo[t] <= j <= hi[t] and 0 <= b < NB and sel_blocks[t, b] != 0
```

Metadata must describe a valid packed sequence partition. Each sequence starts
its own block numbering, including when its first token is not divisible by four.
Its block spans have at most four tokens; only the final block can be shorter.
The spans, bases, and token bounds must agree. Values remain on the device and
are a caller contract. No host read checks these relationships.

Forward uses dense tiles for short key spans and gathers selected blocks for
other tiles. GPU preparation checks the current K/V values and bounds, then
builds exact masks. Nonfinite K/V uses the gather path to exclude unused values
before arithmetic. Both paths compute scaled QK scores, apply the bounds, and
use online softmax followed by PV. They write FP32 output and natural-log LSE,
then cast output to BF16. Empty rows produce zero output and negative-infinity LSE.

Backward compacts the current selection, computes
`delta = sum(grad_out * out_fp32, dim=-1)`, reconstructs probabilities from QK
and LSE, and forms dQ/dK/dV. Accumulation uses FP32; output gradients use BF16.
K/V gradient contributions from all query heads are summed. Floating-point
atomic reductions can change low bits between runs.
On Blackwell and Rubin, the 12:1 and 6:1 paths batch consecutive queries (two
12-head or four 6-head queries fill one 24-column tile) and gather their selected
block union once. Owner bits retain each query's mask while sharing K/V loads.
The 24:2 path processes one query per CTA.
A device-side K/V scan selects a per-query fallback when nonfinite values are
present. The fallback reuses the compact union but applies each query's owner
bits and bounds before dQ arithmetic, preventing a masked NaN from crossing
between queries or packed sequences. Both launches test the current device flag;
only one computes gradients. The flag is rebuilt during CUDA graph replay.

For each query head and its shared KV head, the backward equations are:

```python
S = scale * Q @ K.T
P = exp(S - lse[:, None])  # zero outside the allowed keys
delta = sum(grad_out * out_fp32, dim=-1)
dP = grad_out @ V.T
dS = P * (dP - delta[:, None])
dQ = scale * dS @ K
dK = scale * dS.T @ Q
dV = P.T @ grad_out
```

The implementation evaluates these equations in tiles. It does not allocate
full score or probability matrices.

| Call | GPU component | Result |
|---|---|---|
| Forward | Tile bounds, K/V checks, exact mask preparation | Private tile modes, key ranges and token masks |
| Both | Selection compaction | Private block IDs and counts for gathered queries |
| Forward | Dense or gathered QK, online softmax, PV | `out`, `out_fp32`, `lse` |
| Backward | Preparation | FP32 `delta`; cleared gradient accumulators |
| Backward | Gather, probability reconstruction, gradient products | FP32 dQ/dK/dV accumulators |
| Backward | Final cast | BF16 `dq`, `dk`, `dv` |

The selection bitmap must include any desired causal tail. The implementation
does not add a tail automatically and does not truncate selection at 512 blocks.
Pass the same Q/K/V, selection, metadata, and scale to both calls. Casting `out`
back to FP32 does not reconstruct `out_fp32`.

## Support and execution

Supported geometry is BF16, head dimension 256, head counts 24Q/2KV, 12Q/1KV or
6Q/1KV (the full model and its two- and four-way tensor-parallel shards, the KV
head replicated at four-way), and `block_size=4`. `T > 0`, `NB > 0`, and `T * Hq * 256 <= INT32_MAX` are required.
`scale` is a finite host scalar, normally `1 / sqrt(256)`.

The public functions select one of two architecture families internally:

| Family | Forward kernel | Backward kernel |
|---|---|---|
| Blackwell SM100/SM103 | `cudnn_frost_qsa_fprop_sm100` | `cudnn_frost_qsa_bprop_sm100` (batched for 12:1 and 6:1, single-query for 24:2) |
| Rubin SM107 | `cudnn_frost_qsa_fprop_sm107` | Batched `cudnn_frost_qsa_bprop_sm100` (12:1, 6:1), `cudnn_frost_qsa_bprop_sm107` (24:2) |

Kernel symbols are implementation details; Rubin kernels compile for SM107.
Rubin compaction scans 16 selection bytes per lane and preserves block order.
CuTeDSL 4.7 or newer is
required; Rubin also requires a build with the `sm_107a` target.

Calls allocate outputs and scratch on the input device and enqueue work on its
current PyTorch CUDA stream. Warm up the chosen head geometry before CUDA graph
capture. Compiled kernels reuse symbolic token and block extents; metadata is
rebuilt from the current bitmap on each call. Forward scratch includes an int32
`[T, NB]` compact-index buffer and int32 `[T]` counts. The batched backward path on both families
uses `[ceil(T/NQ), NB]` indices and `[ceil(T/NQ)]` counts (`NQ` = 2 at 12:1, 4 at 6:1);
the 24:2 backward uses the forward shapes. Backward also uses FP32 gradient accumulators and `[T, Hq]`
delta. The batched backward safety scan also uses four int32 flags per 1024
K/V elements plus one reduced flag. The dense route also uses packed token
masks, tile ranges and K/V check flags. These buffers are rebuilt on each call.
Full-call benchmarks must include forward preparation, compaction, backward
preparation, and final gradient casts.
At `T=131072` and `NB=32768`, the forward compact-index buffer alone uses 16 GiB.
