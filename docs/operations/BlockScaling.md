
# Block Scaling

(block-scale-quantize)=
# Block Scale Quantize

The block scale quantize operation computes the quantized output
and scaling factor tensors from a higher precision tensor.

The MXFP8 recipe quantizes across 32 FP32 elements along the rows
(and optionally columns) to produce 32 FP8 output values (E4M3 or E5M2)
and 1 FP8 scaling factor (E8M0). The NVFP4 recipe quantizes across
16 FP32 elements along the rows to produce 16 FP4 output values (E2M1)
and 1 FP8 scaling factor (E4M3).

The computation can be mathematically represented as the following:

$$ scale = quantize\_round\_up(amax(vals) / vmax\_otype) $$
$$ output = quantize\_round\_to\_even(vals / scale) $$

where vals is a block of elements and vmax_otype is the maximum value
representable by the output data type.

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

(block-scale-dequantize)=
# Block Scale Dequantize

The block scale dequantize operation computes the dequantized output
tensor from quantized input and scale tensors.

The computation can be mathematically represented as the following:

$$ output = dequantize(vals * scale) $$

where vals is a block of elements and scale is broadcasted to the block
size.

### C++ API

```
std::shared_ptr<Tensor_attributes> block_scale_dequantize(std::shared_ptr<Tensor_attributes> x,
                                                          std::shared_ptr<Tensor_attributes> scale,
                                                          Block_scale_dequantize_attributes);
```

Block_scale_dequantize_attributes is a lightweight structure with setters:
```
Block_scale_dequantize_attributes&
set_block_size(int32_t const value)
```
