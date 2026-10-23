# block_scale_quantize

Append a block-scale quantization operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto [Y, scale] = graph->block_scale_quantize(
    X,
    fe::graph::Block_scale_quantize_attributes()
        .set_block_size(32)
        .set_axis(2)
        .set_transpose(false)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("block_scale_quantize"));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Tensor to quantize.

- `attributes` (`fe::graph::Block_scale_quantize_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_block_size(int32_t)`: Sets the number of elements per quantization block.

- `attributes.set_axis(int64_t)`: Selects the axis along which blocks are formed. If not called, the axis remains unset.

- `attributes.set_transpose(bool)`: Requests transposed quantized output layout when supported.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type for scale generation and conversion.

- `attributes.set_name(std::string)`: Operation name used for generated tensor names.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Quantized output tensor attributes.

- `scale` (`std::shared_ptr<Tensor_attributes>`): Generated block-scale tensor attributes.

## Errors

The method appends the node and returns tensor attributes immediately. Invalid tensor metadata, unsupported block sizes, output metadata, dtypes, reordering, architectures, and cuDNN-version limits are reported by later graph lifecycle `error_t` statuses from `validate`, `build_operation_graph`, `check_support`, or `build_plans`.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: FP8, MXFP8, NVFP4, and block scaling**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable or experimental by operation
- **Family cuDNN:** Operation-dependent
- **Family CUDA:** CUDA 12 or later; CUDA 13 used for this checkout
- **Family GPU:** Hopper for FP8; Blackwell for MXFP8/NVFP4 and SM100 kernels
- **Family platforms:** Linux (partially executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** C++ FP8 and MXFP8 attention samples executed on SM100; unsupported block-scaling paths are reported locally
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

The returned tensor attributes do not execute work or allocate memory. Set output dtype, dimensions, strides, and reordering on `Y` and `scale` when the backend cannot infer the intended storage.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2568) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4361).
{/* END GENERATED API SOURCES */}

## Examples

See the complete [MXFP8 block-scaled matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/block_scaled_matmul.cc), which dequantizes two inputs, performs a matmul, and block-quantizes the result.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.block_scale_quantize` reference](/api/python/block-scale-quantize) for Python exceptions, wrapper behavior, and current availability constraints.
