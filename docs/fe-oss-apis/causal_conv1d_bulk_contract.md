# Bulk Causal Conv1d Contract (Prototype)

This document fixes the contract for the bulk causal-convolution work that
follows the SM100 decode-update primitive. It is a prototype plan, not a
shipped API guarantee.

## First native slice

The first optimized slice is the model-relevant BF16, width-four, no-bias,
fused-SiLU operation on SM100:

- dense input: contiguous `x[B, T, D]`
- packed input: contiguous `x[1, total_T, D]` plus contiguous CUDA int32
  `cu_seqlens[N + 1]`
- filter: contiguous `weight[D, 4]`
- optional initial state: contiguous `initial_state[N, D, 4]`
- output: contiguous `y`, with the same shape as `x`
- optional final state: contiguous `final_state[N, D, 4]`

For dense input, `N == B`. For packed input, `B == 1`,
`cu_seqlens[0] == 0`, `cu_seqlens[-1] == total_T`, and every sequence length
is positive. Packed sequences must never read each other's tokens.

The full-width state is intentional. For each token, the operation shifts the
four-lane state left, appends the token, then filters the updated state. The
returned final state can therefore be passed directly to the decode-update
primitive without repacking:

```text
state = [state[1], state[2], state[3], x_t]
y_t = SiLU(sum_j state[j] * weight[j])
```

When `initial_state` is omitted, its observable lanes are zero. Returning a
final state must not mutate the input state.

## Backward contract

Backward returns gradients for `x`, `weight`, and `initial_state` when that
input participates in autograd. If `final_state` contributes to the loss,
backward also consumes its upstream gradient. This makes the prefill-to-decode
state bridge differentiable instead of treating the state output as detached
metadata.

The implementation may recompute the four-term preactivation in backward.
That policy is part of graph construction, not a requirement to save a large
intermediate. BF16 parameter gradients may accumulate in FP32 internally, but
the public autograd result follows the input parameter dtype, matching the
existing causal-convolution wrappers.

## Ecosystem adapters

The native primitive owns one explicit contract; adapters own ecosystem
differences.

- FLA uses `[B, T, D]`, `[D, W]`, `cu_seqlens`, and full-width `[N, D, W]`
  cache tensors. Its supported no-bias width-four subset can be a zero-copy
  adapter. Residual, other activations, widths, dtypes, or architectures must
  retain FLA's original route.
- Transformers Qwen3.5 uses a full-width convolution cache as well. Its fused
  mixed-QKV channel width is a separate measured signature, not an assumption
  that split Q/K/V measurements transfer to it.
- Dao-AILab/causal-conv1d's bulk API uses `[B, D, T]` and a minimal
  `[B, D, W-1]` state. Compatibility belongs in an adapter; the native API
  will not weaken the direct prefill-to-decode bridge to copy that ABI.

No external kernel source is to be copied into the implementation. External
projects are semantic, interface, and performance references only.

## Existing path and gap

`cudnn.ops.causal_conv1d_nwh` already provides dense `[B, T, D]` forward and
backward through fixed-size backend bindings. Those bindings accept only
`batch`, `dim`, `seq_len`, and `kernel_size`; they do not accept packed
boundaries or initial/final state pointers. The new work therefore needs a
separate native path rather than presenting a Python loop or per-sequence
fallback as packed-kernel support.

## Delivery order

1. Forward correctness for dense and packed inputs, including exact boundary
   and final-state tests.
2. Backward correctness for `dx`, `dweight`, `d_initial_state`, and upstream
   `d_final_state`, including sequences shorter than the width.
3. FLA and Transformers adapters with explicit route proof and fail-closed
   support checks.
4. ComputeLab B200 optimization at representative sequence lengths 8192,
   16384, and 32768. Report direct kernel-active time separately from graph or
   model-proxy latency, and compare arms in one interleaved process.

The first performance gate is parity with the best available FLA route while
filling the packed-state/backward feature gap. Broader dtype, width, bias, and
architecture support are follow-ups, not hidden fallback behavior.
