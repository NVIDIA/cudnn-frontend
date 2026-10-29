# Engram saved-state gate

Experimental SM100 forward and backward APIs for the four-stream Engram gate.
The backend name is `frost`. CUDA-enabled PyTorch, CuTe DSL >=4.7.0 and, for
backward, the existing `triton` extra (>=3.7.0) are required.

```bash
pip install 'nvidia-cudnn-frontend[triton]'
```

For each token and stream, the gate normalizes X and its key separately,
reduces `X * weight * key` in FP32, applies a signed square root with a
`1e-6` clamp, then sigmoid. The output is `X + gate * shared_value`, rounded
once to BF16. A false token mask makes the output equal X. The default
normalization epsilon is `1e-20`. The weight argument is the FP32 product of
the source q/k normalization weights; the caller computes their product-rule
gradients from the returned `grad_weight`.

| Tensor | Shape | Dtype |
|---|---|---|
| X, output, upstream, grad_X | `[N,4,5120]` | BF16 |
| KV, grad_KV | `[N,25600]` | BF16 |
| weight, grad_weight | `[4,5120]` | FP32 |
| token mask | `[N]` | bool |
| saved | `[N,4,4]` | FP32 |

All tensors are contiguous; floating tensors and workspace must be 16-byte
aligned. Packed KV contains four keys in its first 20480 columns and one
shared value in its last 5120 columns. Backward writes that same packed
layout directly, including the sum of the four shared-value contributions
before its BF16 cast. N is positive and at most 4194240; backward requires
N divisible by 64 and at most 8192. Only SM100 is currently supported.

The forward `saved` output stores `(gate,dot,rstd_X,rstd_key)` per stream.
Retain it and the matching X/KV/weight until backward finishes. Multiple
outstanding forwards need separate saved tensors. Passing state from a
different forward is invalid; the API does not read device data to verify
its identity. Outputs and scratch must not overlap inputs or each other.

```python
import cudnn

forward = cudnn.engram_gate_saved_forward(x, kv, weight, mask, backend="frost")
backward = cudnn.engram_gate_saved_backward(
    x, kv, weight, forward["saved"], grad_out, backend="frost"
)
# TupleDict order: (out,saved), then (grad_x,grad_kv,grad_weight).
```

For repeated execution or CUDA Graph capture, allocate outputs beforehand:

```python
import torch

stream_handle = torch.cuda.current_stream(x.device).cuda_stream
f = cudnn.EngramGateSavedForward(x, kv, weight, mask, backend="frost")
f.compile()
saved = torch.empty((x.shape[0],4,4), device=x.device, dtype=torch.float32)
out = torch.empty_like(x)
b = cudnn.EngramGateSavedBackward(x, kv, weight, saved, grad_out, backend="frost")
b.compile()
dx, dkv, dw = torch.empty_like(x), torch.empty_like(kv), torch.empty_like(weight)
workspace = b.allocate_workspace()
f.execute(x, kv, weight, mask, out, saved, current_stream=stream_handle)
b.execute(x, kv, weight, saved, grad_out, dx, dkv, dw, workspace,
          current_stream=stream_handle)
```

Forward needs no scratch. Backward scratch size is
`4 * (16*N + 4*5120*(N//16))` bytes: 20.25 MiB at N=4096 and 40.5 MiB at N=8192. The 16-token
backward tile limits register pressure; its partial weight gradients use a
fixed reduction order without atomics. Query `scratch_workspace_bytes()` when
allocating scratch rather than retaining a size from an older plan. Saved state
is a separate forward output.
Compile initializes all CuTe and Triton artifacts and launchers; execute
performs no allocation, conversion, compilation, or synchronization. The
convenience wrappers allocate and are intended for eager use.

The class APIs also accept `TensorDesc` declarations without device storage.
A declaration with `device="cuda"` binds to the current CUDA device when the
plan is constructed. Later changes to the current device do not redirect
the plan; execution tensors must reside on its original device.

This operator covers the floating gate, not embedding-table lookup,
projection GEMMs, FP8 quantization, table-gradient coalescing, optimizer or
collectives. Backward is explicit; the wrappers do not register autograd.
The normalization and signed-square-root semantics follow
[DeepSeek-V4.1 Engram](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py).
Performance and target-hardware validation must be established for the
integrated API before making release claims.
