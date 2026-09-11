# NVFP4 Attention QAT Backward

**This is an experimental API and subject to change.**

## Overview

`nvfp4_attention_qat_backward` computes explicit Q, K, and V gradients for
scaled dot-product attention trained with NVFP4 fake quantization. It is a
Triton port of FastVideo's attention QAT backward at commit
`e9bbaca07d511b2ee7e16474dae6f923426223dc`:

<https://github.com/hao-ai-lab/FastVideo/blob/e9bbaca07d511b2ee7e16474dae6f923426223dc/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/attn_qat_train.py>

The operation fake-quantizes Q, K, and V to the NVFP4 E2M1 data format with an
E4M3 scale for every 16 values, then immediately dequantizes them for the
attention computation. The probability matrix follows two paths:

```text
Q_hat, K_hat, V_hat = fake_nvfp4(Q), fake_nvfp4(K), fake_nvfp4(V)
P = softmax(softmax_scale * Q_hat @ K_hat^T)
O_high_precision = P @ V_hat
dS = P * (dO @ V_hat^T - rowsum(O_high_precision * dO))
dQ = softmax_scale * dS @ K_hat
dK = softmax_scale * dS^T @ Q_hat
dV = fake_nvfp4(P)^T @ dO
```

- dQ and dK use the unquantized softmax probability, implementing the
  straight-through estimator (STE).
- dV uses the NVFP4 fake-quantized probability.

The default `backend="triton"` launches four kernels: fused Q fake-quantization/delta
preprocessing, K/V fake-quantization, dQ, and dK/dV. Causal backward skips
fully masked tiles while retaining the elementwise mask on the diagonal.
The local-scale conversion uses precise division so exact E2M1 midpoints
preserve round-to-nearest-even; no tolerance relaxation is required.
The production
non-causal SM100 configuration uses 64 by 64 tiles; other supported Blackwell
configurations use 32 by 32 tiles.

## Installation

From a source checkout, install the CuTe DSL base dependencies, Triton, and the
torch dependency group:

```bash
python -m pip install --upgrade "pip>=25.1"
python -m pip install -e ".[cutedsl,triton]" --group torch
```

For a published wheel, install a CUDA-enabled torch build separately:

```bash
pip install "nvidia-cudnn-frontend[cutedsl,triton]" torch
```

Triton 3.7 or newer is supported on Linux with Python 3.10 or newer for this
API.

## High-level wrapper

```python
import torch
from cudnn import nvfp4_attention_qat_backward

# BHSD tensors from the matching QAT forward pass.
q = torch.empty((1, 16, 4096, 128), device="cuda", dtype=torch.bfloat16)
k = torch.empty_like(q)
v = torch.empty_like(q)
high_precision_o = torch.empty_like(q)
do = torch.empty_like(q)
lse = torch.empty((1, 16, 4096), device="cuda", dtype=torch.float32)

result = nvfp4_attention_qat_backward(
    do,
    q,
    k,
    v,
    high_precision_o,
    lse,
    is_causal=False,
)
dq, dk, dv = result
```

The result keys are `dq_tensor`, `dk_tensor`, and `dv_tensor`. Optional
preallocated tensors with those names can be passed to the wrapper. Pass a
`cuda.CUstream` as `current_stream` to order wrapper allocations and all
kernel launches on an explicit stream.

`high_precision_o` is not the probability-quantized user-visible QAT output.
It must be the matching `softmax(Q_fake K_fake^T) @ V_fake` value saved before
probability fake quantization. `lse` is the corresponding natural-log
log-sum-exp statistic. Supplying forward auxiliaries from a different
quantization recipe produces incorrect gradients.

## Class API

`Nvfp4AttentionQatBackward` exposes explicit validation, compilation, and
execution. `execute` performs no allocations; the caller supplies contiguous
gradient buffers and a one-dimensional CUDA `torch.uint8` workspace.

```python
from cudnn import Nvfp4AttentionQatBackward

op = Nvfp4AttentionQatBackward(q, k, v, high_precision_o, do, lse)
op.check_support()
op.compile()

dq = torch.empty_like(q)
dk = torch.empty_like(k)
dv = torch.empty_like(v)
workspace = torch.empty(op.scratch_workspace_bytes(), device=q.device, dtype=torch.uint8)
op.execute(q, k, v, high_precision_o, do, lse, dq, dk, dv, workspace)
```

`compile()` materializes the selected backend's shape- and architecture-specialized
kernels without launching them. `execute()` reuses those artifacts.

## Opt-in FROST backend (SM100)

Both the class constructor and wrapper accept `backend="frost"`.
FROST is the backend name; CuTe DSL is its kernel implementation technology.
The default remains `"triton"`; this is not an automatic dispatch policy.
The FROST backend requires **CuTe DSL >= 4.7.0**, SM100, BF16, D128,
B=1, equal query/KV head counts, noncausal attention, and equal positive
sequence lengths divisible by 256. Unsupported declarations raise during
`check_support()`; there is no padding, layout conversion, or silent fallback.
Use the Triton backend for its wider support, including tails.

```python
op = Nvfp4AttentionQatBackward(
    q, k, v, high_precision_o, do, lse,
    backend="frost", head_chunk=0,
)
op.check_support()
op.compile()
workspace = torch.empty(op.scratch_workspace_bytes(), device=q.device, dtype=torch.uint8)
op.execute(q, k, v, high_precision_o, do, lse, dq, dk, dv, workspace)
```

This backend reuses the Triton Q/delta and KV quantizers, then launches a
two-CTA FROST dV/dS kernel implemented in CuTe DSL and two output-buffer BF16
batched GEMMs for dQ/dK. The quantizers write their BSHD intermediates directly; the main kernel
addresses caller-owned BHSD dO/dV natively. All stages honor `current_stream`.
Compilation is plan-time-only; no mutable global configuration is switched
between plans. The wrapper's bounded cache separates backend/head-chunk plans.

The tradeoff is a materialized BF16 dS workspace. `head_chunk=0` computes all
heads at once. A positive divisor of H reuses dS across head chunks, reducing
its allocation at the cost of more launches. Total workspace, in bytes, is:

```text
3 * H * S * 128 * 2     # fake Q/K/V
+ H * S * 4            # raw FP32 delta
+ H_chunk * S * S * 2  # BF16 dS, reused across chunks
```

For H=3/S=32768, dS alone is 6 GiB with all heads or 2 GiB with
`head_chunk=1`. This is not the total process memory footprint. Outputs,
workspace and inputs must not overlap; concurrent executions need separate
output/workspace storage. The wrapper allocates workspace each call; use the
class API for explicit reuse and CUDA Graph capture, and warm up execution
before capture to initialize the framework's GEMM runtime.

Both backends implement the same local-scale-floor NVFP4/STE contract.
The FROST path folds `softmax_scale` into dS before its BF16 store;
the Triton path applies it after gradient accumulation. Expect small BF16
rounding differences in dQ/dK, not bitwise identity.

`benchmark/nvfp4_attention_qat/benchmark_backends.py` checks correctness and
changed-input graph replay before timing complete backward with alternating
backend order. These are backward-component measurements, not model E2E.

## Tensor contract

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `q_tensor` | `(B, H, S_q, 128)` | BF16 | Forward query before fake quantization |
| `k_tensor`, `v_tensor` | `(B, H, S_kv, 128)` | BF16 | Forward key and value before fake quantization |
| `high_precision_o_tensor` | `(B, H, S_q, 128)` | BF16 | STE forward auxiliary described above |
| `do_tensor` | `(B, H, S_q, 128)` | BF16 | Gradient of the attention output |
| `lse_tensor` | `(B, H, S_q)` | FP32 | Natural-log softmax statistic |
| `dq_tensor` | same as Q | BF16 | Query gradient |
| `dk_tensor`, `dv_tensor` | same as K/V | BF16 | Key and value gradients |

All tensors must use contiguous, 16-byte-aligned BHSD storage and reside on one
CUDA device. `softmax_scale` defaults to `1 / sqrt(128)` and must match the
forward pass.

## Current support and limitations

The following describes the default Triton backend; the opt-in FROST
backend's narrower initial coverage is listed above.

- GPU: SM100, SM103, SM120, and SM121 Blackwell.
- Attention: MHA with equal query and KV head counts; head dimension 128.
- Sequence lengths: self-attention and non-causal cross-attention, including
  non-aligned tails.
- Causal mode: self-attention only.
- Dtype: BF16 activations and FP32 LSE.
- Not implemented: GQA/MQA, dropout, padding or packed variable-length
  sequences, bias, local masks, and deterministic-mode selection.
