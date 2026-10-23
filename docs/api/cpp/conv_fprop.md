# conv_fprop

Append a forward convolution operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->conv_fprop(
    X,
    W,
    fe::graph::Conv_fprop_attributes()
        .set_padding({1, 1})
        .set_stride({1, 1})
        .set_dilation({1, 1})
        .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("conv"));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input activation tensor attributes.

- `W` (`std::shared_ptr<Tensor_attributes>`, required): Filter tensor attributes.

- `attributes` (`fe::graph::Conv_fprop_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_padding(std::vector<int64_t>)`: Sets symmetric pre-padding and post-padding.

- `attributes.set_pre_padding(...)` and `attributes.set_post_padding(...)`: Set asymmetric spatial padding.

- `attributes.set_stride(std::vector<int64_t>)`: Sets spatial stride.

- `attributes.set_dilation(std::vector<int64_t>)`: Sets spatial dilation.

- `attributes.set_convolution_mode(fe::ConvolutionMode_t)`: Sets the cuDNN convolution mode.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type.

- `attributes.set_name(std::string)`: Operation name used for generated tensor names.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Forward convolution output tensor attributes.

## Errors

The method appends the node and returns tensor attributes immediately. Invalid dimensions, padding lengths, tensor metadata, output metadata, and unsupported dtype/layout/architecture combinations are reported by lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Convolution forward and gradients**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** See selected engine support
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** Engine-dependent
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Forward, data-gradient, and weight-gradient samples executed in Python and C++ on SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

`X`, `W`, and `Y` do not need identical strides. Each tensor should carry metadata for its own runtime storage.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2432) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3937).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/conv_fprop_basic.cc)

<Code
  src="../../samples/cpp/conv_fprop_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.conv_fprop` reference](/api/python/conv-fprop) for Python exceptions, wrapper behavior, and current availability constraints.
