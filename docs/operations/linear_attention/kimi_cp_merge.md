# Kimi Context Parallel State Merge

Experimental explicit merge over **caller-gathered FP32** Kimi CP summaries.
No NCCL calls, local summary computation, or dtype-conversion kernels are hidden
inside this operation. The same operation merges rank summaries or local pieces.
The explicit public merge defaults to TF32x3. Existing KDA graph calls retain
their default IEEE internal chain for compatibility.

For each piece `j`, in forward order:

```text
initial_states[j] = S
S = S @ M[j] + H[j]
```

`reverse=True` visits pieces in reverse order and uses `M[j].T`. Here M is the
**forward transition in row-state layout**; H is the corresponding backward
additive gradient G. Do not pass an already-transposed backward transition to
this mode. The optional composed transition follows the same order/transposes.

## Operand precision

| `precision` | Multiplication | Accumulator / running master / H addition |
| --- | --- | --- |
| `"ieee"` | Existing IEEE FP32 FMA chain | FP32 |
| `"bf16"` | BF16 Tensor Core operands, round-to-nearest-even | FP32 |
| `"tf32"` | TF32 Tensor Core operands, round-to-nearest-even | FP32 |
| `"tf32x3"` (default) | Three TF32 Tensor Core products | FP32 |

For TF32x3, `A_hi = round_tf32(A)`, `A_lo = round_tf32(A - A_hi)` (and likewise B).
The product is `A_hi B_hi + A_hi B_lo + A_lo B_hi`; the low-low term is omitted.
Residuals are formed from FP32 masters, never from previously BF16-rounded data.
Each step retains the resulting FP32 state before rounding its next MMA operands.
The additive H is not quantized. TF32x3 is approximate, not IEEE FP32 equivalence.
Assume finite inputs in the normal TF32 range; this is not a special-value or
subnormal-preserving IEEE emulation.

TF32x3 is the public default because merge error compounds with CP depth. It
retains FP32 master state while recovering the leading residual terms omitted by
a single TF32 product, providing substantially better long-chain accuracy than
BF16 or TF32 operands without paying for the scalar IEEE path. Callers can still
select IEEE when maximum accuracy is more important than Tensor Core throughput.

These are explicit numerical choices, **not autotuning knobs**. Precision is part
of the compiled-plan cache identity. This option does not alter summary-generation
MMA precision, other KDA arithmetic, or the dtype/size of collective payloads.

## PyTorch wrapper

```python
from cudnn import kimi_cp_merge

# H: [CP, heads, V, K], M: [CP, heads, K, K], both compact CUDA FP32.
# Gather these tensors in sequence/rank order before calling the merge.
result = kimi_cp_merge(H, M)  # TF32x3 by default
incoming = result["initial_states"][rank:rank + 1]  # [1, heads, V, K]
tail = result["final_state"]                     # [1, heads, V, K]
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

## Prepared, allocation-free API

```python
from cudnn import KimiCPMerge

plan = KimiCPMerge(H, M, incoming_states, final_state,
                   initial_state=seed, precision="tf32x3").compile()
plan.execute(H, M, incoming_states, final_state, seed,
             current_stream=stream)
```

All execute-time buffers must match their compiled descriptors. Optional inputs
and outputs must match in both directions. No global precision setting is changed.
Precision and reverse direction are fixed at compilation; use another plan to
select another mode, or pass the new mode to the allocating wrapper.
The plan owns no CUDA storage and requires no caller workspace; only fixed shared
memory is used within each CTA. Warm execution supports CUDA Graph capture,
new input allocations and caller-selected streams/devices.

Currently supports FROST SM100/SM103/SM107 targets; SM107 needs a DSL build with
the sm_107a target. Requires CUDA-enabled PyTorch and CuTeDSL >= 4.8.0. With
CUDA 13, install a DSL build with the cu13 extra
(`nvidia-cutlass-dsl[cu13]>=4.8.0`). Tensor Core fragment
layout follows the [NVIDIA PTX ISA mma.m16n8k8 specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-1688).

The shared internal `build_state_chain(..., merge_precision=...)` / compiled host
also accepts these modes, including ragged filled-piece tables, seed indices and
transition-only chains. Existing automatic KDA/GDN plans remain on IEEE unless a
caller of that internal builder/host explicitly selects another precision.
