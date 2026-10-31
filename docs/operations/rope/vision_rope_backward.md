# DSv4.1 Vision RoPE Backward

**Experimental.** `VisionRoPEBackward` fuses split-half Q/K rotary-embedding
backward and packed QKV-gradient output for the DSv4.1 vision encoder on
Blackwell SM100. It accepts the K-gradient layout produced by attention
backward directly, using one kernel without a separate layout copy.

The mathematical contract follows
[DeepSeek-V4.1-Flash `apply_rotary`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/vision.py).
The GPU implementation, shared-memory transpose, and fused output packing
are original to this implementation.

For each Q or K gradient, split the last dimension into two halves `(a, b)`:

```
dx_first  = a * cosine + b * sine
dx_second = b * cosine - a * sine
```

Each product and add/subtract rounds separately in FP32, followed by BF16
storage. The V gradient is copied bit for bit into its packed output slice.
Cosine and sine are caller-prepared constants; this API does not compute
their gradients or register a PyTorch autograd function.

## Tensor contract

All arguments are PyTorch CUDA tensors on the same device, with 16-byte
aligned addresses. Strides are in elements; `T` is the token count.

| Argument | Shape | Dtype | Stride |
| --- | --- | --- | --- |
| `grad_q`, `grad_v` | `(T, 16, 64)` | BF16 | `(64, 64*T, 1)` |
| `grad_k` | `(T, 16, 64)` | BF16 | `(1, 64*T, T)` |
| `cosine`, `sine` | `(T, 1, 32)` | FP32 | `(32, 32, 1)` |
| `grad_qkv` | `(T, 3072)` | BF16 | `(3072, 1)` |

The output is Q, then K, then V, with each group flattened over 16 heads
and 64 channels. Output storage must not overlap any input. Other layouts,
head counts, head dimensions, dtypes, and GPU architectures are rejected.
`T` must be positive and `ceil(T/32)` must fit the CUDA x-grid limit.
CuTe DSL >= 4.7.0 and the usual FE PyTorch dependencies are required.

## Usage

The wrapper allocates and returns output:

```python
result = cudnn.vision_rope_backward_wrapper(dq, dk, dv, cosine, sine, backend="frost")
dqkv = result["grad_qkv"]
```

Use the class API for caller-owned output, stream control, and CUDA Graph
capture. Construct and compile the plan before capture:

```python
plan = cudnn.VisionRoPEBackward(dq, dk, dv, cosine, sine, backend="frost")
plan.check_support()
plan.compile()
dqkv = torch.empty((dq.shape[0], 3072), device=dq.device, dtype=torch.bfloat16)
plan.execute(dq, dk, dv, cosine, sine, dqkv)
```

Construction also accepts `cudnn.api_base.TensorDesc` metadata. One compiled
plan handles different positive token counts with the same layout family
and device. `execute()` checks current buffers and launches one kernel; it
does not allocate, copy inputs, read device data on the host, or synchronize.
`scratch_workspace_bytes()` is zero. `current_stream` accepts a CUDA stream
handle; when omitted, execution uses the current PyTorch stream on the input
device. The caller must order input producers and output consumers on that
stream. Wrapper allocation also uses the supplied stream.

This is a backward operator API. Its scope excludes attention computation,
QKV projection, and a complete vision-layer or model training step.
