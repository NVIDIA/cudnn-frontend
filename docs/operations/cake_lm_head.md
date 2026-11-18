# Cake LM-head log-probabilities

`cudnn.ops.LmHeadLogprobPlan` prepares deterministic LM-head token
log-probabilities and their gradients. It uses the pinned Cake CUDA chain on
SM100 with CUDA Toolkit 13.4 or later and TVM-FFI.

The initial contract uses contiguous BF16 `X[tokens, hidden]` and
`W[vocab, hidden]`, and int64 `labels[tokens]`. Tokens range from 1 to 4096;
hidden is a positive multiple of 512 and vocab a positive multiple of 256.
Labels must be valid vocabulary indices or -100 (ignored). The caller owns
that label-value contract; execution reads no tensor values on the host.

Forward writes FP32 `logp[tokens]` and `lse[tokens]`. Ignored rows have zero
logp. Backward reads FP32 `dlogp[tokens]` and the saved LSE, recomputes logits,
and writes FP32 `dX` and `dW` in the input shapes. Keep X, W, and labels unchanged
between forward and backward. Gradient outputs retain the GEMMs' FP32
accumulators; this API does not register a Torch autograd function.

```python
from cudnn.ops import LmHeadLogprobPlan

plan = LmHeadLogprobPlan(tokens, hidden, vocab, device=0)
workspace = torch.empty(plan.get_workspace_size(), device="cuda", dtype=torch.uint8)
logp = torch.empty(tokens, device="cuda", dtype=torch.float32)
lse = torch.empty_like(logp)
dX = torch.empty_like(X, dtype=torch.float32)
dW = torch.empty_like(W, dtype=torch.float32)
stream = torch.cuda.current_stream().cuda_stream
plan.forward(X, W, labels, logp, lse, workspace, stream=stream)
plan.backward(X, W, labels, lse, dlogp, dX, dW, workspace, stream=stream)
```

Construction compiles six low-level stages and stores host argument templates.
Execution uses only caller-owned storage, with no device allocation, label
compaction/readback, host synchronization, or compilation. Use distinct outputs
and scratch; overlapping writes fail before launch. Operand pointers need
16-byte alignment and workspace needs 256-byte alignment. The explicit stream
defaults to CUDA's default stream (0). Use the caller's stream for ordered input
production and output consumption.

This initial path uses one token chunk and materializes BF16 logits in scratch.
It does not serve temperature, weighted/policy loss, row compaction, multi-chunk
scheduling, or the nondeterministic upstream engine. No throughput claim is made.
The log-probability statistics use FP32 logits while the gradient recompute uses
BF16 logits and BF16 logit gradients, as defined by the pinned kernels.

Sources/build records come from
[miles-ext-kernels f55fb996](https://github.com/radixark/miles-ext-kernels/tree/f55fb99674be5edb0e1f6a0e4d6a911335da86ce).
The copied-file record is `cudnn/_cake/_vendor/VENDORED_LM_HEAD.json`.

## JAX

`cudnn.ops.lm_head_logprob_jax(x, weight, labels)` returns FP32 log-probabilities
and supports `jax.jit` and first-order reverse-mode differentiation. Inputs
retain the plan's BF16 shapes. Enable `jax_enable_x64` before creating int64
labels; label values must be vocabulary indices or `-100` (ignored).

```python
import jax
from cudnn.ops import lm_head_logprob_jax

jax.config.update("jax_enable_x64", True)
labels = jax.numpy.asarray(label_ids, dtype=jax.numpy.int64)
logp = jax.jit(lm_head_logprob_jax)(x, weight, labels)
dx, dw = jax.grad(lambda x, w: -lm_head_logprob_jax(x, w, labels).sum(),
                  argnums=(0, 1))(x, weight)
```

AD returns BF16 gradients matching the inputs. For FP32 gradients, call
`lm_head_logprob_forward_jax(x, weight, labels)` to obtain `(logp, lse)`, then
`lm_head_logprob_backward_jax(x, weight, labels, lse, dlogp)` to obtain `(dx, dw)`.

XLA owns outputs and scratch. A native FFI handler launches Cake on XLA's CUDA
stream; no Python callback or framework tensor conversion runs during execution.
Prepared native plans live for the process so compiled executables remain valid.
The initial contract requires one SM100 GPU in one process. Forward-mode AD,
higher derivatives, `vmap`, and cross-process export of compiled executables
are unsupported. Preparation currently requires PyTorch for the shared Cake
planner. A C++17 compiler and the CUDA toolkit are needed for the bridge.
