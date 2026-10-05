# Prepared BF16 Tail RoPE

`cudnn.TailRoPEForward` copies a contiguous BF16 tensor and rotates only its
last 64 channels. It accepts `[T,D]` or `[T,H,D]`, with `D=128` or `512`, and
separate, caller-prepared FP32 cosine and sine tables of shape `[T,32]`.
The implementation is experimental and supports SM100 with CuTe DSL 4.7.0 or newer.

For each adjacent pair `(a,b)` in the tail, it computes:

```text
real = BF16_RNE(FP32_FMA(a, cosine, -FP32_RNE(b * sine)))
imag = BF16_RNE(FP32_FMA(b, cosine,  FP32_RNE(a * sine)))
```

The unrotated prefix is copied unchanged. Negating the prepared sine table
applies the opposite angle. The FP32 multiply/FMA order and signed-zero behavior
are part of the numerical contract; reassociating the expressions can change BF16
rounding boundaries. Tables must contain finite FP32 values; execute does not
scan them or read device values back to the host.

The arithmetic follows the [DeepSeek-V4.1 inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py).
The vectorized kernel is a cuDNN Frontend implementation. This operation performs
no quantization, table gathering, or KV-cache packing. See [RoPE QDQ](rope_qdq.md)
for the separate quantization operations.

## Convenience wrapper

```python
import cudnn
import torch

x = torch.randn(4096, 64, 512, dtype=torch.bfloat16, device="cuda")
angle = torch.randn(4096, 32, dtype=torch.float32, device="cuda")
cosine, sine = angle.cos(), angle.sin()
result = cudnn.tail_rope(x, cosine, sine, backend="frost")
out = result["out"]
```

The wrapper allocates a disjoint output and returns `TupleDict(out=out)`.
Its compiled-kernel cache excludes `T`; prewarm the relevant head count, channel
count and device before capturing a CUDA Graph. An optional `stream` is a raw CUDA
stream handle; wrapper allocations and the launch are ordered on that stream.

## Prepared API

```python
out = torch.empty_like(x)
plan = cudnn.TailRoPEForward(x, cosine, sine, out, backend="frost")
plan.check_support()
plan.compile()
plan.execute(x, cosine, sine, out, current_stream=None)
```

The class also accepts `cudnn.api_base.TensorDesc` declarations. A declaration
with `device="cuda"` binds to the current GPU at construction; subsequent device
selection does not change the plan. Compilation specializes on head count,
channel count and device. Token count may change between executions, including
zero, provided the other contracts remain valid.

All tensors must be contiguous on the plan's GPU and their nonempty addresses
must be 16-byte aligned. The output must not overlap any input. There is no
workspace; `execute()` validates and launches without allocation, conversion,
table gathering or synchronization. The caller owns tensor lifetimes and
inter-stream dependencies. Unsupported layouts are rejected rather than copied.

`H` must be positive, `H*D < 2**31`, and the flattened element count must fit
the CUDA grid limit for the selected vector width. Device offsets and the dynamic
element-count argument use signed 64-bit arithmetic.

This is an explicit forward API with no autograd registration. Tensors with
`requires_grad=True` are rejected; detach only when a separate differentiation
contract is already provided by the caller. Opposite-angle forward execution
does not establish a backward API.

## Performance scope

The intended performance use is prefill. Small-token decode shapes have not
shown a consistent advantage over existing implementations. Prepared tables,
full tensor copies, GPU stage time and Graph replay wall time must be accounted
for consistently when comparing providers; stage speedups do not establish
complete attention or model throughput improvements.
