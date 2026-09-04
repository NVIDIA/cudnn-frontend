# NVFP4 block-scale conversion

`cudnn.ops.nvfp4_block_scale_quantize` and
`cudnn.ops.nvfp4_block_scale_dequantize` are explicit conversion operations for
the NVFP4 tensor representation consumed by block-scaled GEMMs. They are
experimental frontend-only APIs and require the `cutedsl` optional dependency.

The conversion is intentionally separate from GEMM. Quantized activations can
be reused by multiple projections, and callers can see and manage the packed
data and scale-factor lifetime directly. No graph-topology matcher or hidden
two-kernel executor is involved.

## Tensor contract

The initial implementation accepts contiguous PyTorch CUDA tensors on SM100 or newer:

| Tensor | Shape | Dtype / layout |
|---|---|---|
| input | `[1, M, K]` | BF16, contiguous |
| encode scale | `[1, 1, 1]` | FP32 |
| packed output | physical `[1, M, K/2]` | packed E2M1 (`torch.float4_e2m1fn_x2`; unsigned byte carrier when that torch dtype is unavailable) |
| block scales | `[1, M, K/16]` | E4M3 in `F8_128x4` physical order |
| dequantized output | `[1, M, K]` | BF16, contiguous |
| decode scale | `[1, 1, 1]` | FP32, multiplied after the per-block scale |

`M` must be positive and divisible by 128. `K` must be positive and divisible
by 64, with at most 1,024 sixteen-element groups per row. The packed bytes are
an implementation-compatible NVFP4 representation, not a promise that a
different quantizer will make identical midpoint choices.

Direct FROST composition requires a PyTorch build that exposes
`torch.float4_e2m1fn_x2`, because FROST's TVM-FFI signature is typed as FP4.
On older PyTorch builds the standalone converter instead returns the same bytes
in a `torch.uint8` carrier; that carrier remains valid for explicit storage and
the standalone dequantizer, but is not accepted directly by the current FROST
GEMM launch boundary.

## High-level use

```python
import torch
import cudnn

x = torch.randn((1, 16384, 2048), device="cuda", dtype=torch.bfloat16)
encode = torch.tensor([[[309.4100647]]], device="cuda", dtype=torch.float32)

packed, block_scales = cudnn.ops.nvfp4_block_scale_quantize(x, encode)

# With native PyTorch FP4 support, `packed` and `block_scales` can now be passed
# as materialized inputs to an existing FROST block-scale GEMM graph. Mark its
# scale descriptor as `cudnn.tensor_reordering.F8_128x4`.

decode = encode.reciprocal()
restored = cudnn.ops.nvfp4_block_scale_dequantize(packed, block_scales, decode)
```

The high-level quantizer allocates its two outputs and returns a `TupleDict`, so
both tuple unpacking and named access (`result["packed_tensor"]`,
`result["scale_tensor"]`) are supported. The dequantizer returns its BF16
tensor directly.

## Explicit plan lifecycle

`Nvfp4BlockScaleQuantizer` and `Nvfp4BlockScaleDequantizer` accept sample input
and preallocated output tensors. Call `check_support()`, `compile()`, then
`execute(...)`. The compiled `K` is fixed, while `M` is symbolic and may change
between executions as long as it remains divisible by 128. `execute()` never
allocates, converts, or copies its tensor arguments.

Both kernels in this package were authored for cuDNN Frontend from the public
NVFP4 arithmetic and `F8_128x4` layout contract; they do not embed code copied
from another framework's quantizer.
