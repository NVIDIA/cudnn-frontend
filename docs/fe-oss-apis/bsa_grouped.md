# Exact grouped Block Sparse Attention

**Experimental standalone Torch API.** This opt-in transformation packs a
fully pairable blk64 sparse graph into native blk128 forward and backward.
It does not change the ordinary BSA API or register a cuDNN graph engine.

Two Q64 blocks can pair when their valid KV sets are identical. Two KV64
blocks can pair when their reverse Q neighborhoods are identical. The pairs
need not be adjacent and may differ by batch/head. Packing both axes forms
complete 128 × 128 attention blocks: every coarse edge expands to exactly
four original 64 × 64 edges. No masked attention is added and none is dropped.
Floating-point reduction order changes, so numerical equality is approximate.

```python
import torch
from cudnn import (
    block_sparse_attention_grouped_prepare,
    block_sparse_attention_grouped_forward,
    block_sparse_attention_grouped_backward,
)

q, k, v = [torch.randn(1, 512, 4, 128, device="cuda", dtype=torch.bfloat16)
           for _ in range(3)]
# This simple pattern is fully pairable: every Q64 block attends to KV64 0..3.
routes = torch.arange(4, device="cuda", dtype=torch.int32)
routes = routes.view(1, 1, 1, 4).expand(1, 4, 8, 4).contiguous()
grouping = block_sparse_attention_grouped_prepare(routes, num_kv_blocks=8)

o, lse = block_sparse_attention_grouped_forward(q, k, v, grouping)
do = torch.randn_like(o)
dq, dk, dv = block_sparse_attention_grouped_backward(do, q, k, v, o, lse, grouping)
```

The functions are also exported through `cudnn.BSA` and `cudnn.torch`.
Forward returns `TupleDict(o_tensor=..., lse_tensor=...)`; backward returns
`TupleDict(dq_tensor=..., dk_tensor=..., dv_tensor=...)`.

## Supported inputs

- SM100/SM103, CuTe DSL >= 4.7.0; measured here on SM100.
- Contiguous, 16-byte-aligned CUDA **BSHD**, BF16, D=128, equal Q/KV head
  counts. Q and KV lengths may differ; each must be a positive multiple of 128.
  Token lengths and per-head sparse edge counts must fit signed int32 indexing.
- Noncausal, full 64-token blocks, no dropout, GQA, variable sequence lengths,
  partial active pairing, or empty Q rows. Explicit backward; no autograd integration.
- Routes: int32 `(B, H, Nq/64, capacity)` on CUDA. `num_kv_blocks` is `Nkv/64`.
  `block_sparse_num` defaults to capacity. Optional int32 `q2k_block_nums`
  `(B, H, Nq/64)` overrides it with per-row valid prefix lengths.
- Valid indices must be unique and in range. Inactive capacity is ignored.
  Route order is immaterial: preparation compares sorted valid sets exactly.
  Every Q neighborhood and every KV reverse neighborhood must have an even
  number of members. Unused KV blocks may pair with other unused KV blocks.
  Invalid/unpairable input raises; there is no implicit dense or mixed fallback.
- O, dO, dQ/dK/dV use original BSHD order. LSE is FP32 `(B, H, Nq)` in
  natural logarithms and original Q order. Backward requires the same sparse
  graph and `softmax_scale` as forward (default `1/sqrt(128)`).

## Preparation and execution cost

`grouped_prepare` is an explicit, **blocking CPU preprocessing step**. It
copies metadata to the host, validates exact pairing, builds the permutations,
coarsened graph and backward CSR, and uploads these tensors. It rejects calls
inside CUDA Graph capture before the first host transfer. It returns
caller-owned metadata, not a FROST execution plan; do not mutate its tensors.
Changing the original routing tensors does not change this snapshot. Rebuild
it whenever a layer's sparse graph changes.

The forward/backward functions never analyze the graph on the CPU. They are
allocating Torch wrappers, execute on the operand device's current stream,
and can be captured **after warmup**. They allocate packed tensors and native
scratch; CUDA Graph replay reuses captured allocations. They do not promise
allocation-free eager execution or cold capture. Keep metadata alive through
capture/replay and use normal Torch stream dependencies when producing inputs.

Forward packs Q/K/V, calls native blk128 forward, then unpacks O/LSE.
Backward packs Q/K/V/O/dO/LSE, calls native blk128 preprocess/main/postprocess,
and unpacks dQ/dK/dV. Identity permutations skip copies. All accumulators reset
on every backward call and replay. This explicit representation change is
part of the requested grouped operation, not layout normalization in ordinary BSA.

CPU preparation can cost much more than one attention call. GPU-only speedups
do **not** establish a per-layer training speedup when routing changes every
step. Reuse across calls, already grouped routing, or a future GPU planner is
needed to amortize that cost. See the separately measured prepare times in
the [benchmark results](../../benchmark/bsa_grouped/README.md).

## Backward buckets

`bucket_size_blocks` partitions the **packed Q128** axis when preparing CSR.
It does not select a dQ group size. None preserves native blk128's bucket
policy (384 Q128 blocks for Nq=131072, H=40, hence three buckets).
Passing `Nq // 128` requests one bucket: each KV128 tile has a single dK/dV
owner and avoids cross-bucket dK/dV accumulation/conversion. dQ still accumulates
contributions from different KV tiles. One bucket can help short sparse lists
and hurt longer ones; it is an explicit knob, not an automatic performance claim.

## Validation and measurement

`test/python/block_sparse_attention/cutedsl/test_BSA_grouped.py` reconstructs
the exact graph and CSR, compares O/LSE/dQ/dK/dV against FP64 autograd, checks
nonadjacent pairs, different batch/head permutations, variable counts, ignored
capacity, unused NaN KV storage, invalid input rejection, side streams and
changed-input CUDA Graph replay.

The reproducible [benchmark](../../benchmark/bsa_grouped/README.md) includes
all GPU permutations and reports CPU preparation separately. Its synthetic
mask is **100% pairable on both axes**; unstructured sparse graphs generally
do not have this property. CAKE is an optional forward-only comparison using
the original CUDA implementation from FlashInfer PR #4593, not a CuTe DSL port.
