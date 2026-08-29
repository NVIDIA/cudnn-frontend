# Bulk Causal Conv1d Forward (SM100-optimized)

`CausalConv1dBulkFwdSm100` and
`causal_conv1d_bulk_fwd_wrapper_sm100` expose the first native bulk
causal-convolution slice. Its full-width output state is designed to match the
companion decode-update contract, so the two APIs need no state
repacking when both are installed. The API is experimental.

## First native slice

The first optimized slice is the model-relevant BF16, width-four, no-bias,
fused-SiLU operation. Functional targets are the exact compute capabilities
SM80, SM86, SM87, SM89, SM90, SM100, SM103, SM110, SM120, and SM121. The
public class and wrapper retain their original `Sm100` suffix while this
experimental API evolves.

- dense input: contiguous `x[B, T, D]`
- packed input: contiguous `x[1, total_T, D]` plus contiguous CUDA int32
  `cu_seqlens[N + 1]`
- filter: contiguous `weight[D, 4]`
- optional initial state: contiguous `initial_state[N, D, 4]`
- output: contiguous `y`, with the same shape as `x`
- optional final state: contiguous `final_state[N, D, 4]`

Every tensor is CUDA-resident. Data tensors are BF16 and must have a 16-byte
aligned base pointer; `cu_seqlens` is int32 with a 4-byte aligned base pointer.
This API requires `nvidia-cutlass-dsl>=4.7.0`; the frontend package's broader
`cutedsl` extra deliberately retains its 4.5.0 floor for unrelated APIs.
The first slice is inference-only and rejects requires-grad inputs while grad
mode is enabled. There is no implicit dtype/layout conversion or fallback.
On SM100, SM103, SM110, SM120, and SM121, channel extents divisible by eight
use an aligned 128-bit channel-vector fast path with packed FP32 arithmetic.
SM80 through SM90, and every positive channel extent not divisible by eight,
use a predicated scalar path with the same public semantics. Only SM100/B200
has been performance-characterized; the other targets are functional support,
not a cross-architecture performance claim.

For dense input, `N == B`. For packed input, `B == 1`,
`cu_seqlens[0] == 0`, `cu_seqlens[-1] == total_T`, and every sequence length
is positive. Packed sequences must never read each other's tokens. `B` and `T`
must each fit int32, `B*T <= 2^31 - 16`, and
`N <= min(total_T, floor((2^31 - 1) / 2))`. The API also limits
`D <= 256 * 65535`, preserving a representable scalar-fallback CUDA grid for
every accepted channel extent. On that scalar path, each input/output or state
tensor it indexes may contain at most `2^31 - 1` elements.

The full-width state is intentional. For each token, the operation shifts the
four-lane state left, appends the token, then filters the updated state. The
returned final state follows the intended decode-update recurrence without
repacking:

```text
state = [state[1], state[2], state[3], x_t]
y_t = SiLU(sum_j state[j] * weight[j])
```

When `initial_state` is omitted, its observable lanes are zero. Returning a
final state must not mutate the input state.

The high-level wrapper always returns a `TupleDict` in this stable order:

```python
from cudnn import causal_conv1d_bulk_fwd_wrapper_sm100

result = causal_conv1d_bulk_fwd_wrapper_sm100(
    x,
    weight,
    cu_seqlens_tensor=cu_seqlens,
    initial_state_tensor=initial_state,
    output_final_state=True,
)
y = result["output_tensor"]
final_state = result["final_state_tensor"]
```

When `output_final_state=False`, `final_state_tensor` is a CUDA BF16 sentinel
with shape `(0,)`. An FLA adapter must translate that sentinel to `None`.

The lower-level class API follows the common FE-OSS lifecycle: construct it
from representative PyTorch tensors (metadata-only `TensorDesc` inputs are not
accepted), call `check_support()`, call `compile()` once,
then call `execute()` with preallocated output tensors. `B`, `D`, packed `N`,
and optional-tensor presence are compile-signature properties; `T` is symbolic
and may change between executions within the indexing limits above. Both APIs
accept an explicit CUDA stream and do not synchronize it on a valid launch.

## Backward contract

The matching backward is not implemented in this first slice. Its contract is
to return gradients for `x`, `weight`, and `initial_state` when that input
participates in autograd. If `final_state` contributes to the loss, backward
also consumes its upstream gradient. This makes the prefill-to-decode state
bridge differentiable instead of treating the state output as detached
metadata.

All four upstream final-state lanes must flow to the corresponding source in
`tail_4(initial_state || x_sequence)`, including sequences of length one or
two. An independent recurrence, rather than an external kernel implementation,
is the oracle for these tests. For every positive-length sequence,
`d_initial_state[..., 0]` is zero because the first state update discards that
lane.

The implementation may recompute the four-term preactivation in backward.
That policy is part of graph construction, not a requirement to save a large
intermediate. BF16 parameter gradients may accumulate in FP32 internally, but
the public autograd result follows the input parameter dtype, matching the
existing causal-convolution wrappers.

## Ecosystem adapters

The native primitive owns one explicit contract; adapters own ecosystem
differences.

- FLA uses `[B, T, D]`, `[D, W]`, `cu_seqlens`, and full-width `[N, D, W]`
  cache tensors. The exact contiguous, alignment-compatible, no-bias,
  no-residual, width-four BF16 SiLU/swish subset can be a zero-copy adapter.
  Nonstandard QKV views, scheduling hints that cannot be safely ignored,
  duplicate/empty packed segments, or any broader option must retain FLA's
  original route.
- Transformers Qwen3.5 calls a stateless bulk function with `[B, D, T]`; its
  cache manager owns and mutates the full-width `[B, D, W]` cache before that
  call. A function-level adapter can cover only stateless dense bulk. The
  initial/final-state bridge requires a module/cache-level adapter that
  preserves static cache addresses, and must fall back when `record_past=True`.
  Qwen3.5 always applies one convolution to fused mixed QKV, so its channel
  width is a separate measured signature rather than three split calls.
- Dao-AILab/causal-conv1d's bulk API uses `[B, D, T]`, a minimal
  `[B, D, W-1]` state, and optional `seq_idx[B, T]`. `seq_idx` is not
  equivalent to `cu_seqlens`: it can describe multiple segments per batch row,
  negative labels, and a contract incompatible with final-state output.
  Compatibility belongs in an adapter; the native API will not weaken the
  direct prefill-to-decode bridge to copy that ABI.

No external kernel source is to be copied into the implementation. External
projects are semantic, interface, and performance references only.

## Existing path and gap

`cudnn.ops.causal_conv1d_nwh` already provides dense `[B, T, D]` forward and
backward through fixed-size backend bindings, but its filter layout is
`[W, D]`. The ecosystem-facing native API deliberately uses `[D, W]`; an
adapter must not transpose and materialize the old layout on every call. The
old bindings also accept only `batch`, `dim`, `seq_len`, and `kernel_size`, not
packed boundaries or initial/final state pointers. A separate native path is
therefore required rather than presenting a Python loop or per-sequence
fallback as packed-kernel support.

If bias or residual support is added later, the fixed semantic order is
`residual + SiLU(conv + bias)`.

## Delivery order

1. Forward correctness for dense and packed inputs, including exact boundary
   and final-state tests. **Implemented.**
2. Backward correctness for `dx`, `dweight`, `d_initial_state`, and upstream
   `d_final_state`, including sequences shorter than the width.
3. FLA and Transformers adapters with explicit route proof and fail-closed
   support checks.
4. ComputeLab B200 optimization at representative sequence lengths 8192,
   16384, and 32768. Report direct-API CUDA-event elapsed, profiler kernel
   duration, and graph/model-proxy latency as distinct quantities, and compare
   arms in one interleaved process.
   **Prototype benchmarked; model routing remains a follow-up.**

The first performance gate is parity with the best available FLA route while
adding FE's packed-state forward; the matching backward remains next. FLA
Triton itself already implements those features. Broader dtype, width, bias,
and additional architecture support are follow-ups, not hidden fallback
behavior.
