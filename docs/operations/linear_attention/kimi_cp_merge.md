# Kimi Context Parallel State Merge

Experimental explicit merge over **caller-gathered FP32** Kimi CP summaries.
No NCCL calls, local summary computation, or dtype-conversion kernels are hidden
inside this operation. The same operation merges rank summaries or local pieces.
It exposes the existing optimized IEEE FP32 state chain so downstream frameworks
can integrate Kimi CP without importing cuDNN Frontend private helpers.

For each piece `j`, in forward order:

```text
initial_states[j] = S
S = S @ M[j] + H[j]
```

`reverse=True` visits pieces in reverse order and uses `M[j].T`. Here M is the
**forward transition in row-state layout**; H is the corresponding backward
additive gradient G. Do not pass an already-transposed backward transition to
this mode. The optional composed transition follows the same order/transposes.

All multiplication, accumulation, running state/product, additive H and output
storage use IEEE FP32. The operation does not consult ambient Torch matmul
precision settings and has no lower-precision mode. This avoids error compounding
across long CP chains while reusing the state-chain implementation that is faster
than the evaluated BF16 and TF32 alternatives for the supported merge shapes.
This precision choice does not alter summary generation, other KDA arithmetic,
or the dtype and size of collective payloads.

## PyTorch wrapper

```python
from cudnn import kimi_cp_merge

# H: [CP, heads, V, K], M: [CP, heads, K, K], both compact CUDA FP32.
# Gather these tensors in sequence/rank order before calling the merge.
result = kimi_cp_merge(H, M)
incoming = result["initial_states"][rank:rank + 1]  # [1, heads, V, K]
tail = result["final_state"]                         # [1, heads, V, K]
```

Optional `initial_state` is compact CUDA FP32 `[1, heads, V, K]`; absence means
zero. `output_transition=True` returns the composed transition `[1, heads, K, K]`
under `result["transition"]`; otherwise that entry is None. Results form a
`TupleDict`, unpacking in `(initial_states, final_state, transition)` order.
K is 64 or 128; V is 64, 128, or 256. All buffers must be disjoint.

No autograd is registered. Use inside an explicit CP autograd integration and
merge the reverse state-gradient summaries separately during backward. Calling
with `requires_grad` operands while grad mode is enabled raises instead of
silently dropping gradients. The wrapper allocates outputs on the launch stream.
It does not insert waits for H/M/seed produced on another stream. Order those
writes before `current_stream`, and do not concurrently write or reuse any merge
input/output until that stream completes. Internal `record_stream` calls protect
allocator lifetime; they do not establish producer-to-consumer dependencies.

## Prepared, allocation-free API

```python
from cudnn import KimiCPMerge

plan = KimiCPMerge(H, M, incoming_states, final_state,
                   initial_state=seed).compile()
plan.execute(H, M, incoming_states, final_state, seed,
             current_stream=stream)
```

All execute-time buffers must match their compiled descriptors. Optional inputs
and outputs must match in both directions. Reverse direction is fixed at
compilation; use another plan to select the other direction. The plan owns no
CUDA storage and requires no caller workspace; only fixed shared memory is used
within each CTA. Warm execution supports CUDA Graph capture, new input
allocations and caller-selected streams on the device where the plan was
compiled. Compile a separate plan for each device. The same caller-side ordering
rule applies to `execute`: inputs must be ready on its launch stream, and the
bound buffers must not participate in another overlapping write or merge
execution until completion.

Currently supports FROST SM100/SM103/SM107 targets. The shared state-chain
primitives require CuTeDSL >= 4.7.0; SM107 additionally needs a DSL build with
the `sm_107a` target, first available in the public 4.8.0 wheel. The
cuDNN Frontend package requires `nvidia-cutlass-dsl[cu13]>=4.8.0`.
