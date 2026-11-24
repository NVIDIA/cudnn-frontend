
# Block Scaling

## Block Scale Quantize

The block scale quantize operation computes the quantized output
and scaling factor tensors from a higher precision tensor.

The MXFP8 recipe quantizes across 32 FP32 elements along the rows
(and optionally columns) to produce 32 FP8 output values (E4M3 or E5M2)
and 1 FP8 scaling factor (E8M0). The NVFP4 recipe quantizes across
16 FP32 elements along the rows to produce 16 FP4 output values (E2M1)
and 1 FP8 scaling factor (E4M3).

The FP8 block-128 recipe quantizes across 128 FP32 elements along the rows
to produce 128 FP8 output values (E4M3) and 1 FP32 scaling factor. The scale
is not rounded to a power of two or to a narrower format: it is
`amax(vals) / vmax_otype` evaluated in FP32 as `amax * RN(1/448)`, raised to
the smallest normal FP32 when `amax > 0`, and each value is multiplied by the
scale's approximate reciprocal before its conversion to E4M3. It is available
as a fused epilogue of the SM100 FROST `frost_gemm`
general block-scale kernels, selected by `block_size=128` with a `FLOAT` scale
output; see "FROST linear-scale operands" in MoE Grouped Matmul
([docs](/operations/MoeGroupedMatmul#frost-linear-scale-operands-gated-fc1-and-fp8-block-128-output),
[GitHub](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/MoeGroupedMatmul.md#frost-linear-scale-operands-gated-fc1-and-fp8-block-128-output))
for the exact formula and its handling of NaN and zero blocks.

The computation can be mathematically represented by the following equation:

$$ scale = quantize\_round\_up(amax(vals) / vmax\_otype) $$
$$ output = quantize\_round\_to\_even(vals / scale) $$

Where:
- vals is a block of elements.
- vmax_otype is the maximum value representable by the output data type.

### C++ API

```
std::array<std::shared_ptr<Tensor_attributes>, 2> block_scale_quantize(std::shared_ptr<Tensor_attributes> x,
                                                                       Block_scale_quantize_attributes);
```
where the output array is in the order of `[y, scale]`

Block_scale_quantize_attributes is a lightweight structure with setters:
```
Block_scale_quantize_attributes&
set_block_size(int32_t const value)

Block_scale_quantize_attributes&
set_axis(int64_t const value)

Block_scale_quantize_attributes&
set_transpose(bool const value)
```

## Block Scale Dequantize

The block scale dequantize operation computes the dequantized output
tensor from quantized input and scale tensors.

The computation can be mathematically represented by the following equation:

$$ output = dequantize(vals * scale) $$

Where:
- vals is a block of elements.
- scale is broadcast to the block size.

### C++ API

```
std::shared_ptr<Tensor_attributes> block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                                                          std::shared_ptr<Tensor_attributes> scale,
                                                          Block_scale_dequantize_attributes);
```

Block_scale_dequantize_attributes is a lightweight structure with setters:
```
Block_scale_dequantize_attributes&
set_block_size(int32_t const value, int32_t idx = 0)

Block_scale_dequantize_attributes&
set_block_size(const int32_t* values, int32_t len = 1)

Block_scale_dequantize_attributes&
set_block_size(const std::vector<int32_t>& values)
```
