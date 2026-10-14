# Qwen Sparse Attention (QSA)

**This is an experimental API and subject to change.**

## Overview

Qwen Sparse Attention (QSA) is the sparse attention of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). An indexer selects up to 512 complete
4-token key blocks per query, and each query attends to those blocks plus the tokens of its own incomplete block.

`cudnn.QSA` provides the attention backward pass for Blackwell (SM100), implemented as CuTe DSL kernels. The indexer,
the attention forward pass and the indexer loss are not included.

## Semantics

For query `i`, query head `h` (KV head `h // 12`) and each selected token `j` of row `i`, with
`S_ij = q_i · k_j / 16` and `P_ij = exp(S_ij - lse_ih)`:

```text
Delta_i = rowsum(dO_i * O_i)
dS_ij   = P_ij * (dO_i · v_j - Delta_i)
dQ_i    = (1/16) * sum_j dS_ij k_j
dK_j    = (1/16) * sum_{i, h in the KV group} dS_ij q_i
dV_j    = sum_{i, h in the KV group} P_ij dO_i
```

## Index layout

`topk_idxs` is the `(T, 2051)` int32 matrix of selected token ids, shared by all heads. Row `i` holds, in order:

1. `nsel = min((i + 1) // 4, 512)` complete blocks in any order, each the four consecutive ids `4b, ..., 4b + 3`
   with `4b + 3 <= i`
2. the `(i + 1) % 4` tokens of the incomplete block, `4 * ((i + 1) // 4), ..., i`
3. `-1` in the remaining columns

This is the layout the Qwen3.8 indexer produces. The kernels do not validate it, and rows that deviate from it give
wrong gradients.

## Support surface

- SM100 (compute capability 10.0) with `nvidia-cutlass-dsl >= 4.7.0`.
- 24 query heads, 2 KV heads, head dim 256 and softmax scale 1/16.
- One sequence of `1 <= T <= 349525` tokens.

| Tensor | Shape | Dtype |
|---|---|---|
| `q`, `out`, `dout`, `dq` | `(T, 24, 256)` | bf16 |
| `k`, `v`, `dk`, `dv` | `(T, 2, 256)` | bf16 |
| `lse` | `(T, 24)` | fp32, natural-log LSE of the scaled scores |
| `topk_idxs` | `(T, 2051)` | int32 |

All tensors must be contiguous and on the same CUDA device. `check_support()` raises `NotImplementedError` or
`ValueError` for anything else.

## Numerics

- MMA operands are bf16 (`P` and `dS` are rounded to bf16). All accumulation is fp32, and the outputs are cast to
  bf16 at the end.
- dK, dV and part of dQ are reduced with fp32 atomics, so results are not bitwise reproducible.

## Usage

```bash
pip install nvidia-cudnn-frontend
pip install -U "nvidia-cutlass-dsl>=4.7.0"
```

The wrapper allocates the outputs and the workspace and caches one plan per tensor layout:

```python
from cudnn import QSA

dq, dk, dv = QSA.sparse_attention_backward_wrapper(q, k, v, out, dout, lse, topk_idxs)
```

The class API takes caller-owned outputs and workspace. `execute()` never allocates or synchronizes, so it can be
captured in a CUDA graph:

```python
import torch
from cudnn import QSA

plan = QSA.SparseAttentionBackward(q, k, v, out, dout, lse, topk_idxs)
plan.check_support()
plan.compile()

dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
workspace = torch.empty(plan.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
plan.execute(q, k, v, out, dout, lse, topk_idxs, dq, dk, dv, workspace=workspace)
```

The workspace holds the fp32 accumulators and the per-query index lists. Its size grows linearly with `T`, mostly
from the fp32 dQ accumulator (`T * 24 * 256 * 4` bytes).

## Implementation

Six kernels run on the launch stream:

| Kernel | Role |
|---|---|
| `qsa_bwd_prep_kernel` | computes `Delta` and zero-fills the accumulators |
| `qsa_bwd_partition_kernel` | groups each sparse query's blocks by KV pass |
| `qsa_bwd_bitmap_kernel` | builds the block bitmap of the dense queries |
| `qsa_bwd_sparse_kernel` | query-major path for the queries at or above `num_dense_queries` |
| `qsa_bwd_dense_pair_kernel` | key-tile-major path for the queries below `num_dense_queries` |
| `qsa_bwd_finalize_kernel` | sums the accumulators and writes bf16 `dq`, `dk` and `dv` |

A query below `num_dense_queries` selects a large share of its visible tokens (2048 of `i + 1`), so the dense kernel keeps dK and dV
of a 128-token key tile on chip across all the queries that see it. The sparse kernel walks the selected blocks of
the remaining queries in 16384-token KV passes, so that each pass's K/V and accumulators stay in L2. `num_dense_queries` is
`min(T, 24576)`: a query stays on the dense path while it selects at least 1/12 of its visible tokens. The
dense kernel is a programmatic dependent launch of the sparse kernel and fills SMs as the sparse grid drains.

## Limitations

- SM100 only, and only the geometry above.
- One sequence per call: no batch, packed (`cu_seqlens`) or context-parallel layouts.
- No deterministic mode.

## Testing

```bash
cd test/python
pytest qwen_sparse_attention/cutedsl/test_QSA_sparse_attention_backward.py          # L0
pytest -m L1 qwen_sparse_attention/cutedsl/test_QSA_sparse_attention_backward.py    # 64K and 256K sequences
```
