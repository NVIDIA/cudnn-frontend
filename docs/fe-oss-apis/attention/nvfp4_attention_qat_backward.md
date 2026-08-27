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

The implementation uses split Triton dQ and dK/dV kernels. The production
non-causal SM100 configuration uses 64 by 64 tiles; other supported Blackwell
configurations use 32 by 32 tiles.

## Installation

From a source checkout, install the CuTe DSL base dependencies, Triton, and the
torch dependency group:

```bash
python -m pip install --upgrade "pip>=25.1"
pip install -e ".[cutedsl,triton]" --group torch
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

`compile()` materializes every shape- and architecture-specialized Triton
kernel without launching it. `execute()` then reuses those cached artifacts.

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

- GPU: SM100, SM103, SM120, and SM121 Blackwell.
- Attention: MHA with equal query and KV head counts; head dimension 128.
- Sequence lengths: self-attention and non-causal cross-attention, including
  non-aligned tails.
- Causal mode: self-attention only.
- Dtype: BF16 activations and FP32 LSE.
- Not implemented: GQA/MQA, dropout, padding or packed variable-length
  sequences, bias, local masks, and deterministic-mode selection.
