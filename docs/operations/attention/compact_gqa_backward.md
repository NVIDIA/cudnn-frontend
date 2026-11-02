# Compact GQA backward (SM107)

Experimental BF16 causal attention backward with runtime sequence packing.
Inputs use contiguous THD storage, eight query heads, one KV head, and head dimension 256.
LSE uses packed token-major FP32 storage. Scale is fixed at 1/16.

CuTe computes dS, dK, and dV with separate TMA loaders. Long sequences use
causal query bands and a two-CTA CuTe dQ kernel. Short sequences share packed
launches and use the installed cuBLAS API. No cuBLAS changes are required.

dS is computed in FP32 and stored in BF16. dQ reads BF16 dS and K, accumulates
in FP32, and writes BF16. dK/dV partials stay FP32 across bands before BF16 output.

```python
import torch
from cudnn import CompactGqaBackward, compact_gqa_backward

offsets = (0, 16384, 65536)
plan = CompactGqaBackward(q, k, v, o, do, lse, max_seqlen=49152, query_rows=32768)
plan.check_support()
plan.compile()
workspace = torch.empty(plan.scratch_workspace_bytes(), device=q.device, dtype=torch.uint8)
plan.initialize_workspace(workspace)
dq, dk, dv = compact_gqa_backward(
    q, k, v, o, do, lse, plan=plan, workspace=workspace,
    sequence_offsets=offsets,
)
plan.close()
```

Offsets are host integer prefixes covering the physical token buffer.
Optional `sequence_lengths` gives each sequence's valid length; omitted lengths
equal the physical spans. Padding gradients are zero. With neither argument,
the buffer contains one sequence. Unequal lengths, arbitrary counts, and tail
tiles are supported; lengths are runtime values and do not trigger recompilation.

Allocate gradients yourself and call `plan.execute(...)` to avoid output allocations.
Compile and initialize outside the timed region. Execute performs no host
synchronization or CUDA tensor allocation. Short packs allocate pinned host
metadata and pack K into caller-owned scratch. Use one plan per CUDA stream.

`max_seqlen` bounds scratch capacity and defaults to the initial token count.
The default dS budget is 8.00 GiB. With query_rows=32768 and groups=4,
total scratch is 8.91 GiB at 64K capacity. The budget bounds dS, not all scratch;
linear metadata and partial-gradient buffers still grow with capacity.
ds_budget_bytes and query_rows control the memory/launch-count tradeoff.
Query `scratch_workspace_bytes(new_capacity)` before resizing, then pass that
capacity to `initialize_workspace(..., max_seqlen=new_capacity)`.
Insufficient budgets are rejected before initialization or execution.
Reinitialize replacement workspaces and whenever another plan has used their contents.
Retain tensors and scratch until queued work completes. Outputs must not alias
inputs or workspace.

Requires an SM107-capable CuTe DSL build, CUDA development headers, installed
cuBLAS, and `g++`. The host shim builds during compile and is cached under
`$XDG_CACHE_HOME/cudnn_frontend`.
Bias, dropout, other scales, deterministic mode, and CUDA graphs are unsupported.
This API does not change cuDNN engine selection.

## Optional TE adapter

`cudnn.sdpa.bwd.compact_gqa.te` accepts immutable device prefixes registered
with their originating host offsets. Logical and physical prefixes may differ.
Object lifetime and tensor version guards reject stale registrations; raw device
writes remain the caller's responsibility.

`begin_step()` enables the candidate from the first training batch.
`report()` prints candidate and fallback counts. Verify coverage on every
attention rank; native-only runs do not validate the candidate.

## Validation status

Latest VR200 full-workload A/B: stock job 664477 and improved job 664478, using
the native TE/NeMo-RL integration. Mean policy-training time over all 11 steps
is 111.08 s → 109.46 s, a 1.46% reduction.
Excluding the first step gives 107.55 s → 106.00 s (1.45%).

Both jobs completed successfully. Step-10 validation accuracy is 72.91% for
stock and 74.10% for improved, above the 69.00% target. Both runs use the same
FE 1.29 image, configuration, and seed; only the GQA flag differs. Real rollouts
and node allocations differ, and valid tokens increase 1.11%; timings are not
normalized. Valid-token counts are not exact attention FLOP counts.

The review fixes passed 26 VR200 tests, and the native TE integration passed
three SM107 numerical cases before the full workload. The current develop merge
moves the test to the CI layout and preserves the compact GQA runtime files.
Earlier sanitizer results predate the review fixes.
