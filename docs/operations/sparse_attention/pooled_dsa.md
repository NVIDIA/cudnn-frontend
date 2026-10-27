# Pooled DSA

Pooled DSA selects groups of index keys, then expands each selected group to
individual token IDs. `cudnn.ops.dsa` provides a pooled indexer and separate
selection, cache-update, and index-expansion APIs for Blackwell SM100/SM103.

## Selection from logits

`cudnn.ops.dsa.pooled_select_forward` accepts precomputed BF16 or FP32
indexer logits. It selects complete causal pools, expands them into token
IDs, appends incomplete-pool tail tokens, and fills padding with `-1` in
one CuTe DSL launch. It has separate prefill and decode kernels. It performs
no scoring, pooling, quantization, or attention softmax.

This is the API used by the vLLM pooled-selection adapter. DeepGEMM supplies
native BF16 logits; vLLM retains its existing pooled-key cache format.

```python
from cudnn.ops import dsa

size = dsa.pooled_select_workspace_size(logits, max_pools=32768)
workspace = torch.empty(size, dtype=torch.uint8, device=logits.device)
result = dsa.pooled_select_forward(
    logits, positions, selected_pools=512, pool_size=4,
    mode="prefill", max_pools=32768,
    output=output, workspace=workspace, current_stream=stream)
indices = result["indices"]
```

| Argument | Contract |
|---|---|
| `logits` | BF16/FP32 `(N,C)`, contiguous columns, positive non-overlapping row stride, 32-byte-aligned address |
| `positions` | INT32/INT64 `(N,)`, zero-based token positions; positive vector stride |
| `selected_pools` | Multiple of 32, from 32 to 2048; default 512 |
| `pool_size` | 4 or 16; default 4 |
| `mode` | `"decode"` or `"prefill"` |
| `max_pools` | Compile capacity, `C <= max_pools <= 32768`; defaults to `C` |
| `output` | Optional INT32 matrix, at least `N` rows and `selected_pools * pool_size + pool_size - 1` columns |
| `workspace` | Optional contiguous UINT8 vector, aligned to 256 bytes, sized with `pooled_select_workspace_size` |
| `current_stream` | Optional CUDA stream handle or torch stream; defaults to the current stream |

For position `p`, eligible columns are below
`min(C, floor((p + 1) / pool_size))`. Negative positions and positions at or
above `2**31 - 1` produce inactive rows. If the causal context fits within
`selected_pools * pool_size` tokens, the output contains all causal token IDs
in ascending order. Otherwise, selected pools occupy the fixed history
region and up to `pool_size - 1` tail tokens follow it. Extra rows and columns
are filled with `-1`. Tied IDs and selection order are unspecified. Eligible
logits must not contain NaNs. The caller supplies columns for the history it
wants ranked; the operation does not access a cache or page table.

All tensors must share a CUDA device and require no autograd. Output rows
may be padded, with contiguous columns and a positive non-overlapping row
stride. Output and workspace must not overlap each other or either input.
The int32 addressing limit requires `2*N*C < 2**31`.

The workspace query reads only tensor metadata. BF16 specializations whose
`max_pools` rounds up to 8192, 16384, or 32768 use 32 bytes. Other
specializations use `8*N*C` bytes. The workspace contains selector scratch;
it excludes input logits and output storage.

With output and workspace supplied, the API allocates no device storage,
copies or converts no tensors, reads no device values on the host, and does
not synchronize. It reuses compilation across changing row counts, columns,
and supported strides within `max_pools`. Warm up before CUDA graph capture.
The caller keeps buffers live until execution completes on the chosen
stream. When buffers are omitted, the function allocates them on that stream.
It returns `TupleDict(indices=output)`. Backward and autograd are unsupported.

`_compile_pooled_select` remains an internal compiler helper. Applications
should use the public selection and workspace-size functions above.

## Indexer forward

`pooled_indexer_forward` reads paged pooled keys, scores complete causal pools,
selects top-k pools, and expands them into token IDs. Optional cache updates
run first. Inputs begin after Q/K projection; Q rotation and quantization
remain upstream. Final sparse attention over token K/V is a separate operation.

The real-valued score for query `i` and pool `j` is
`sum_h weights[i,h] * max(dot(q[i,h], pooled_key[j]), 0)`.
Only `j < floor((positions[i] + 1) / pool_size)` is eligible. Pool selection
uses these index scores; it does not apply attention softmax.

BF16 weights select the SM100 DeepGEMM arithmetic contract: tensor-core
dot products accumulate in FP32, each ReLU result rounds to BF16, and four
BF16 fused multiply-add chains reduce heads `0,4,...`, `1,5,...`,
`2,6,...`, and `3,7,...`. BF16 adds combine chains 0+2, 1+3, then the
two results. The materialized logits are BF16. This is not equivalent to
rounding an FP32 weighted sum once at the end.

FP32 weights retain the FP32 reduction and logits path. The API never
casts weights; their dtype selects the arithmetic and scratch dtype.

| Argument | Dtype and shape | Meaning |
|---|---|---|
| `q` | BF16/E4M3 `(N,32,128)` or UINT8/INT8 `(N,32,64)` | Rotated index queries; FP4 packs two E2M1 values per byte |
| `weights` | BF16 or FP32 `(N,32)` | BF16 matches the native SM100 DeepGEMM head reduction and logits |
| `q_scale` | UINT8 `(N,32,4)` or INT32 `(N,32)` | Four UE8M0 scale bytes per head; required for FP4, optional for FP8 with `fp8_scale_format="ue8m0"` |
| `cache` | UINT8 `(physical_pages,page_bytes)` | Pooled-key cache; formats below |
| `block_table` | INT32 `(requests,pages_per_request)` | Logical-to-physical page mapping |
| `cu_query` | INT32 `(requests+1,)` | Offsets of packed query rows |
| `positions` | INT32/INT64 `(N,)` | Absolute token positions; negative rows are inactive |
| `output` | INT32 `(output_rows,width)` | Selected token IDs, padded with `-1` |
| `workspace` | Contiguous UINT8 vector | Caller-owned scratch, aligned to 256 bytes |

Pool sizes 4/16 and page sizes 64/128/256 are supported. `selected_pools`
must be a multiple of 32, at most 2048. The default is 512 pools of four
tokens, with output width at least 2051. Output rows may exceed `N`.
All padded rows and columns are filled with `-1`. Selection order and
the choice among tied scores are unspecified.

```python
from cudnn.ops import dsa

# Native SM100 FP8: BF16 weights, packed UE8M0 Q scales and cache scales.
result = dsa.pooled_indexer_forward(
    q, weights, cache, block_table, cu_query, positions, q_scale,
    cache_dtype="fp8", fp8_scale_format="ue8m0",
    mode="decode", max_query_tokens=1,
    current_stream=stream)
indices = result["indices"]
```

FP8 `q_scale=None` uses unit Q scales; callers may fold positive query
scales into weights upstream, as vLLM does. Explicit scales support four
different 32-channel scales per head. With the default
`fp8_scale_format="fp32"`, the cache uses FP32 scales and Q scales must
be folded into weights. Set the same format on the cache writer and reader.
FP4 uses packed UE8M0 scales and accepts signed or unsigned packed Q bytes.

For cache writes, pass all six keyword tensors `keys`, `gates`, `ape`,
`tail`, `pool_slots`, and `tail_slots` to the forward call.
They follow the cache-update contract below and have one row per packed
query.

`max_query_tokens` is a host upper bound on each request's query count,
not its context length. `mode="decode"` supports packed multi-token
requests and partial-pool continuation. `mode="prefill"` tiles two
queries per BF16 score program and four for FP32; cache writes require
pool-aligned chunk starts.
Empty requests are allowed. `cu_query` must be monotonic, start at zero,
and end at `N`. The bound must cover every offset difference. Each
request's complete causal pools must fit its valid page-table entries.
These device-resident metadata contracts are the caller's responsibility.

The prepared path uses two launches: paged scoring and fused radix top-k
with token expansion. The selector has separate prefill and decode kernels.
Each keeps selected pool IDs in shared memory and writes token IDs, tail
tokens, and padding directly. Cache writes add one fused launch. Scoring reads the
page table directly, including in prefill. There is no gathered key buffer.
The scorer and selector remain separate kernels behind one public API.
BF16 arithmetic matches DeepGEMM's reduction order; FP32 weights select
the FP32 path. The API needs Triton 3.7+ and CuTe DSL 4.8+.
Queries, keys, scales, and head weights must be finite.

Scratch contains logits and radix scratch. For `N` queries and `C`
pooled-key columns, FP32 uses `12*N*C` bytes. BF16 uses `10*N*C` bytes,
or `2*N*C + 32` when the `max_pools` power-of-two bucket is 8192, 16384,
or 32768 and selection keeps candidates in shared memory.
The function allocates scratch when omitted. Callers
may supply a 256-byte-aligned buffer with each region rounded up to
256 bytes. Shapes and positive row-major strides can change without recompilation within `max_pools`; columns must
be contiguous. `max_pools` defaults to the initial page-table capacity.
The top-k scratch index limit requires `2*N*C < 2**31`.

Warm up before capture. With caller-provided output and workspace the function
creates tensor views but does not allocate device storage, read device data on the host, copy tensors,
or synchronize. The caller owns all buffers and their lifetime, including
on an explicit CUDA stream. Writable output, workspace, cache, and tail
must not overlap any other bound tensor. No autograd contract is provided.

The return is `TupleDict(indices=output)`. Internal prepared plans are used
by the kernel benchmark to exclude repeated host-side plan construction.
Those plans are not public API exports.

## Pool cache update

`cudnn.ops.dsa.pooled_cache_update_forward` combines gated pooling, normalized
Hadamard-128, quantization, paged storage, and tail-cache updates in one
CuTe DSL launch. It needs Blackwell SM100 or SM103 and CuTe DSL 4.8 or later.

| Argument | Dtype and shape | Meaning |
|---|---|---|
| `keys`, `gates` | BF16 `(tokens, 128)` | Raw index keys and channel gates |
| `ape` | FP32 `(pool_size, 128)` | Learned position bias within a pool |
| `cache` | UINT8 `(pages, page_bytes)` | Caller-owned pooled-key cache |
| `tail` | BF16 `(requests, 2, pool_size, 128)` | Per-request K/gate ring; axis 1 selects K or gate |
| `pool_slots` | INT32/INT64 `(tokens,)` | Physical cache slot at a completed pool; `-1` otherwise |
| `tail_slots` | INT32/INT64 `(tokens,)` | Physical ring slot `request * pool_size + phase` |
| `positions` | INT32/INT64 `(tokens,)`, decode only | Absolute zero-based token position |
| `cu_seqlens` | INT32 `(requests + 1,)`, decode only | Offsets of packed query tokens |

For each channel, pooling applies softmax to `gates + ape` across the
pool's tokens. It forms the weighted sum of K, rounds to BF16, applies
Hadamard-128 divided by `sqrt(128)`, then rounds to BF16 again. The cache
stores this rotated key in the selected format:

| `cache_dtype` | Bytes per page of `P` pools | Page contents |
|---|---:|---|
| `bf16` | `P * 256` | `P` contiguous BF16 vectors |
| `fp8`, `fp8_scale_format="fp32"` | `P * 132` | `P * 128` E4M3 bytes, then `P` FP32 scales |
| `fp8`, `fp8_scale_format="ue8m0"` | `P * 132` | Same values, then four repeated UE8M0 bytes per pool, as in native vLLM |
| `fp4` | `P * 68` | `P * 64` packed E2M1 bytes, then four UE8M0 scales per pool |

FP8 uses one power-of-two scale per 128 channels. MXFP4 uses one per
32 channels; the earlier channel occupies the low nibble. FP4 rounds
to nearest even. All cache formats receive BF16 keys and gates.
There is no global FP4 scale.

`fp8_scale_format` defaults to `"fp32"` for existing callers. Select
`"ue8m0"` on both APIs to share a cache with SM100 DeepGEMM. Encoding
happens in the pooling kernel; no scale conversion kernel is launched.

`mode="prefill"` requires each request's new tokens to start on a pool
boundary. The kernel reads completed pools directly from packed inputs.
It copies only the last write to each ring slot. Partial-pool continuation
must use `mode="decode"`, which processes each request's packed tokens in
order and combines them with the existing ring. An empty request is allowed.

The caller must provide finite inputs, valid ring contents for prior tokens,
monotonic decode offsets starting at zero and ending at `tokens`, and unique
cache ownership per request. Pool slots must identify only complete groups.
Decode tail-slot phase must equal `position % pool_size`. Negative tail
slots or positions disable decode rows; negative pool slots disable cache
writes. The kernel bounds-checks physical cache and tail slots.

All tensors must share a CUDA device and have positive strides. Matrix
columns and tail token vectors are contiguous. Row/page padding and padding
between the tail's K/gate blocks are supported. Cache page stride and tensor
addresses must be four-byte aligned. Writable caches cannot overlap each
other or any input. Metadata values remain on the device; the caller is
responsible for the offset and ownership contracts above.

```python
from cudnn.ops import dsa

result = dsa.pooled_cache_update_forward(
    keys, gates, ape, cache, tail, pool_slots, tail_slots,
    positions, cu_seqlens, pool_size=4, page_size=64,
    cache_dtype="fp8", mode="decode", current_stream=stream)
```

Workspace size is zero. The function performs no device allocation, host read,
copy, or synchronization. Token/request/page counts and accepted strides
may change without recompilation. Pool size, page size, dtypes, mode, and
device form the compilation-cache key. Warm up before graph capture and keep the
buffers live until execution completes.

The function returns
`TupleDict(cache=cache, tail=tail)`. Both outputs are caller-owned state;
the function does not create or initialize them. No autograd contract is
provided.

## Index expansion

`cudnn.ops.dsa.pooled_expand_forward` writes a caller-owned INT32 matrix in one
CuTe DSL launch. It accepts INT32 or INT64 pool IDs and token positions.
It needs Blackwell SM100 or SM103 and CuTe DSL 4.8 or later.

| Argument | Shape | Meaning |
|---|---|---|
| `pool_ids` | `(source_rows, selected_pools)` | Local IDs from top-k; negative IDs are padding |
| `positions` | `(query_rows,)` | Zero-based token positions; negative values mark inactive rows |
| `output` | `(output_rows, width)` | Caller-owned INT32 token IDs |
| `row_map` | `(query_rows,)`, optional INT32 | Output row to source row; negative or out-of-range values mark inactive rows |

All tensors must be on the same CUDA device. Pool sizes 4 and 16 are supported.
Matrix columns must be contiguous; positive row and vector strides are
supported. Output must not overlap an input. Positions must be below
`2**31 - 1`. The result uses local token IDs and `-1` for unused entries.

`output_rows >= query_rows` and
`width >= selected_pools * pool_size + pool_size - 1` are required. Every
output element is written, including padded rows and columns.

For a token position `p`, completed pools cover
`floor((p + 1) / pool_size) * pool_size` tokens. Each selected completed pool
expands into consecutive token IDs. Up to `pool_size - 1` trailing causal
tokens follow the fixed history region. Future pool IDs are masked. If the
whole causal context fits in the history region, the result contains every
causal token in ascending order.

```python
from cudnn.ops import dsa

result = dsa.pooled_expand_forward(
    pool_ids, positions, pool_size=4, output=output, current_stream=stream)
```

With caller-provided output, the function performs no device allocation,
conversion, host read, or
synchronization. Its workspace size is zero. Row counts, selected-pool
counts, and accepted strides may change without recompilation. Dtypes,
pool size, device, and the presence of `row_map` form the compilation-cache key.
Warm up before capture; caller-owned inputs and output must remain live
until the launch completes.

The function allocates the output when omitted and returns a
`TupleDict` with the single key `indices`:

```python
result = dsa.pooled_expand_forward(pool_ids, positions, pool_size=4)
indices = result["indices"]
```

The pool selection and token expansion contract follows GLM's
[`SparseAttnIndexerKpool`](https://github.com/vllm-project/vllm/blob/bc162b3f9280c2dc05dfd5aa282d29551f8f6d33/vllm/models/glm5next/nvidia/sparse_indexer.py).
`ExpandIndices` receives already selected pools. `PoolCacheUpdate` only
updates caches. `IndexerForward` combines both with scoring and top-k;
none of these APIs performs final sparse attention.
