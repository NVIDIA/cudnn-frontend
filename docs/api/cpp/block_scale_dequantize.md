# block_scale_dequantize

Append a block-scale dequantization operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->block_scale_dequantize(
    X,
    scale,
    fe::graph::Block_scale_dequantize_attributes()
        .set_block_size({1, 32})
        .set_is_negative_scale(false)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("block_scale_dequantize"));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Quantized tensor to dequantize.

- `scale` (`std::shared_ptr<Tensor_attributes>`, required): Block-scale tensor applied to `X`.

- `attributes` (`fe::graph::Block_scale_dequantize_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_block_size(...)`: Sets per-axis block sizes. The class provides overloads for a scalar entry, pointer plus length, and vector-style initializer usage.

- `attributes.set_is_negative_scale(bool)`: Indicates whether scale values can be negative.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type for scale application and conversion.

- `attributes.set_name(std::string)`: Operation name used for generated tensor names.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Dequantized output tensor attributes.

## Errors

The method appends the node and returns tensor attributes immediately. Invalid tensor metadata, malformed block sizes, incompatible scale tensor metadata, unsupported output metadata, dtypes, reordering, architectures, and cuDNN-version limits are reported by later graph lifecycle `error_t` statuses from `validate`, `build_operation_graph`, `check_support`, or `build_plans`.

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

The output has its own tensor metadata. It does not mutate `X` in place and does not need to share dtype or stride with `X` or `scale`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2571) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4377).
{/* END GENERATED API SOURCES */}

## Examples

See the complete [MXFP8 block-scaled matmul sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/block_scaled_matmul.cc), which uses `F8_128x4`-reordered E8M0 scales to dequantize both matmul inputs.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.block_scale_dequantize` reference](/api/python/block-scale-dequantize) for Python exceptions, wrapper behavior, and current availability constraints.
