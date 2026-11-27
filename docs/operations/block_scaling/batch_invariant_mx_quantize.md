# Batch-invariant MXFP8, MXFP4 and NVFP4 quantization

**Experimental.** `cudnn.ops.mxfp8_quantize_forward`, `cudnn.ops.mxfp4_quantize_forward` and
`cudnn.ops.nvfp4_quantize_forward` quantize the rows of a bf16, fp16 or fp32 tensor to MXFP8, MXFP4 and NVFP4 with one
declared recipe each. The codes and the UE8M0 scale bytes of a row
depend on that row alone: they are the same for any batch size, row position, tile configuration and SM count. Use them
where rollout and training must produce the same quantized operands, for example ahead of a
[batch-invariant block-scaled matmul](/operations/Matmul).

The output contract is FlashInfer's `mxfp8_quantize(..., batch_invariant=True)`, `mxfp4_quantize(..., batch_invariant=True)`
and `nvfp4_quantize(..., batch_invariant=True)`, and the bytes are the same. The two functions need CuTe DSL >= 4.8.0 and run on compute capability 10.0, 10.3 and 10.7.

## MX recipe

```text
per block of 32 consecutive elements of a row (fp32 arithmetic on the widened values):
amax        = max |x_j|
scale byte  = ue8m0_ceil( amax * fp32(1 / elem_max) )     elem_max = 448 (E4M3) or 6 (E2M1)
code_j      = elem( x_j * 2^(127 - scale byte) )           nearest-even, saturating
```

The recipe is Transformer Engine's MX recipe.

## NVFP4 recipe

```text
per block of 16 consecutive elements of a row (fp32 arithmetic, every operation correctly rounded):
amax        = max |x_j|
sf          = e4m3( amax * (gs * fp32(1/6)) )              nearest-even, saturating at 448; the stored scale byte
encode      = min( 1 / (sf * (1 / gs)), FLT_MAX )
code_j      = e2m1( x_j * encode )                         nearest-even, saturating at +-6, sign kept
```

`gs` is the global scale. The per-tensor class takes it as an input. The per-token class takes the inverse base scale
`g` (Transformer Engine: `1 / (448 * 6)`), derives each row's global scale from the row's amax, and returns it:

```text
gs0 = 1 / g
gs  = min( gs0 / row_amax, FLT_MAX ), or 1 when that is 0
per_token_scale = 1 / gs
```

## Use

```python
import torch
import cudnn

x = torch.randn(4096, 8192, device="cuda", dtype=torch.bfloat16)

codes, sf = cudnn.ops.mxfp8_quantize_forward(x, "128x4")   # float8_e4m3fn [4096, 8192], uint8 [M_pad * K / 32]
fp4, sf4 = cudnn.ops.mxfp4_quantize_forward(x, "128x4")    # uint8 [4096, 4096], uint8 [M_pad, K_pad / 32]
nv, nsf = cudnn.ops.nvfp4_quantize_forward(x, 0.37, "128x4")                       # uint8 [4096, 4096], uint8 [M_pad, K_pad / 16]
nv, nsf, pts = cudnn.ops.nvfp4_quantize_forward(x, 1 / (448 * 6), "128x4", per_token=True)   # pts: fp32 [4096]
```

| Function | Returns |
| --- | --- |
| `mxfp8_quantize_forward(input, sf_layout="128x4", enable_pdl=None, alignment=32)` | `x_q`: `float8_e4m3fn` of shape `(*input.shape[:-1], padded_K)`, with `padded_K = ceil(K / alignment) * alignment` (padding columns are zero bytes under a zero scale). `sf`: flat `uint8` UE8M0 scales. |
| `mxfp4_quantize_forward(input, sf_layout="128x4", enable_pdl=None)` | `fp4`: `uint8` of shape `(*input.shape[:-1], K / 2)`, two E2M1 codes per byte with element `2j` in the low nibble. `sf`: `uint8` UE8M0 scales as a `(rows, columns)` view. |
| `nvfp4_quantize_forward(input, global_scale, sf_layout="128x4", enable_pdl=None, per_token=False)` | `fp4`: as for MXFP4. `sf`: `uint8` E4M3 scales, one per 16 elements, as a `(rows, columns)` view. With `per_token=True` also `per_token_scale`, an fp32 vector with one element per row. |

- `input` has shape `(..., K)` with `K` a positive multiple of 32 (16 for NVFP4) and at most 131072 bytes per row. The leading
  dimensions are flattened to rows. A row stride or a base address that is not a multiple of 16 bytes is copied first;
  the copy changes placement, not values.
- `sf_layout` is `"128x4"` (rows padded to 128 and scale columns to 4, the layout block-scaled matmuls read), `"8x4"`
  (rows padded to 8, columns to 4) or `"linear"` (`M x K / 32`, or `M x K / 16` for NVFP4). Padding bytes are zero. FlashInfer's `SfLayout` members
  are accepted as well.
- `global_scale` is a Python float or a one-element tensor. A tensor on another device is copied to the input's device.
- `enable_pdl=None` follows the device: programmatic dependent launch on compute capability 9.0 and newer.
- The kernel runs on the current torch stream of `input.device` and is built for that device.
- Inputs outside the contract raise: a dtype other than bf16, fp16 or fp32, a `K` that is not a multiple of 32, a
  CPU tensor, an unknown layout, a global scale that is not a scalar or an unsupported architecture. Nothing falls back to another implementation.
- The functions take two keyword-only arguments for tests, `config` and `sm_count`, that force a member of the
  enumerated configuration space or another part's dispatcher pick. The bytes never depend on them.

## Guarantees

On the same GPU architecture and software stack, a row with identical values gives identical bytes alone, packed with
other rows, reordered, duplicated or padded to a tile, and across repeated launches. The tests compare raw bytes, never
a tolerance. They also compare against an independent torch implementation of each recipe and against FlashInfer's
batch-invariant quantizers. See the [test guide](../../../test/python/quantize/batch_invariant/README.md) and the
[determinism support matrix](/operations/determinism).

The kernels in `python/cudnn/quantize/batch_invariant/` are adapted from FlashInfer's `flashinfer/batch_invariant/`
and keep its Apache-2.0 notice; see [LICENSING.md](../../../LICENSING.md).
