# conv_dgrad

Append a data-gradient convolution operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto dX = graph->conv_dgrad(
    dY,
    W,
    fe::graph::Conv_dgrad_attributes()
        .set_padding({1, 1})
        .set_stride({1, 1})
        .set_dilation({1, 1})
        .set_convolution_mode(fe::ConvolutionMode_t::CROSS_CORRELATION)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("dgrad"));
```

## Parameters

- `dY` (`std::shared_ptr<Tensor_attributes>`, required): Upstream output-gradient tensor attributes.

- `W` (`std::shared_ptr<Tensor_attributes>`, required): Forward filter tensor attributes.

- `attributes` (`fe::graph::Conv_dgrad_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_padding(std::vector<int64_t>)`: Sets symmetric pre-padding and post-padding.

- `attributes.set_pre_padding(...)` and `attributes.set_post_padding(...)`: Set asymmetric spatial padding.

- `attributes.set_stride(std::vector<int64_t>)`: Sets spatial stride from the forward convolution.

- `attributes.set_dilation(std::vector<int64_t>)`: Sets spatial dilation from the forward convolution.

- `attributes.set_convolution_mode(fe::ConvolutionMode_t)`: Sets the cuDNN convolution mode.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type.

- `attributes.set_name(std::string)`: Operation name used for generated tensor names.

## Returns

- `dX` (`std::shared_ptr<Tensor_attributes>`): Input-gradient tensor attributes. Set dimensions and strides explicitly when shape inference is insufficient.

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

`dY`, `W`, and `dX` do not need identical strides. Each tensor should carry metadata for its own runtime storage.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2436) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3991).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/conv_dgrad_basic.cc)

<Code
  src="../../samples/cpp/conv_dgrad_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.conv_dgrad` reference](/api/python/conv-dgrad) for Python exceptions, wrapper behavior, and current availability constraints.
