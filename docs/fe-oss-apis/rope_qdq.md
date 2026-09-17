# Tail RoPE + Microscaled QDQ

**Experimental frontend-only API for SM100.** `RopeQDQInplace` and
`rope_qdq_inplace` apply adjacent-pair RoPE to the last 64 channels, round that
tail to BF16, and quantize/dequantize all channels in contiguous groups of 32.
The result is written directly to the input BF16 storage.

| `quantization` | Input | Quantized values | Group scale | Minimum amax |
| --- | --- | --- | --- | --- |
| `"fp4"` | `[N,H,128]`, H=1,4,32 | E2M1, limit 6 | UE8M0, group 32 | `6 * 2**-126` |
| `"fp8"` | `[...,512]`, rank at least 2 | E4M3, limit 448 | UE8M0, group 32 | `1e-4` |

These are MX-style group scales. This API returns BF16 QDQ values, with no
packed low-precision tensor or scale output. It does not implement the
NVFP4 group-16/E4M3-scale format or a backward/straight-through estimator.

For each group, the scale is the power-of-two ceiling of the rounded FP32
product `max(amax, floor) * (1 / limit)`. Values are divided by that scale,
clamped to the finite range, rounded to the quantized format using
round-to-nearest-even, multiplied by the scale, and stored as BF16. Signed
zero and the BF16 rounding boundary before amax are preserved. RoPE uses
explicit FP32 FMA for each real/imaginary result.

The math follows the indexer and window/DSpark consumers in
[DeepSeek-V4.1-Flash model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
and its accompanying `kernel.py`. The fused kernels are authored here; CUDA
FP4/FP8 headers were consulted for PTX conversion operand order.

## Requirements and tensor contract

Install `nvidia-cudnn-frontend[triton]` and a CUDA-enabled PyTorch build.
Triton >=3.7.0 and a CUDA toolkit supporting SM100 are required. The only
backend is `backend="frost"`.

- `x`: contiguous CUDA BF16, with the shape above. Its storage is mutated.
- `cache`: contiguous CUDA FP32 `[P,64]`, holding 32 cosines followed by 32
  sines. Supply the desired base or scaled RoPE cache; the API does not
  generate frequencies.
- `positions`: contiguous CUDA int32 or int64. FP4 takes one ID per token,
  shared across H heads; FP8 takes one ID per flattened 512-channel row.
- All tensors must be on the same device. `x` must not overlap either input.
  Contiguous tensors at offsets below 16-byte alignment are supported natively.
- Position IDs must lie in `[0,P)`. Input values and post-RoPE BF16 values
  must be finite. These device-value preconditions are caller responsibilities;
  execution does not read device data back to the host to validate them.
- Tensor arguments with `requires_grad=True` are rejected. Callers must
  retain any original values needed by another consumer before this mutation.

## Convenience wrapper

```python
import cudnn

result = cudnn.rope_qdq_inplace(x, cache, positions, quantization="fp4")
assert result["out"] is x
```

The wrapper returns `TupleDict(out=x)` and reuses the compiled artifact cache.
Its first call can compile; use the class API to prepare before CUDA Graph
capture.

## Prepared class API

```python
op = cudnn.RopeQDQInplace(x, cache, positions, quantization="fp8", backend="frost")
op.check_support()
op.compile()
assert op.scratch_workspace_bytes() == 0
op.execute(x, cache, positions, current_stream=stream.cuda_stream)
```

`current_stream=None` selects PyTorch's current stream on the planned device.
The explicit argument accepts a raw CUDA stream handle. Callers order producer
work before this stream and retain tensor storage until it finishes.

The same plan accepts new token counts, FP8 leading dimensions, cache lengths,
positions, and pointer alignments. Quantization, FP4 head count, position dtype,
and device remain fixed. Each execution launches one prepared kernel, with no
tensor allocation, repacking, synchronization, or compilation. Token counts
must be positive and fit the CUDA grid limit.

The API is a component of the floating inference consumer. Full-model speed,
packed cache storage, and training gradients require separate integration.
