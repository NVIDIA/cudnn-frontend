# set_io_data_type

Set the default data type for non-virtual graph input and output tensors.

## C++ synopsis

```cpp
graph.set_io_data_type(fe::DataType_t::BFLOAT16);
```

## Parameters

- `type` (`fe::DataType_t`, required): Default element type used when a non-virtual tensor has `DataType_t::NOT_SET`.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_io_data_type()` itself returns `Graph&` and does not return `error_t`. Unsupported data types or incompatible operation/tensor combinations are reported by later lifecycle calls.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** 8.5+ for the C++ frontend; Python wrapper paths require 9.12+
- **Family CUDA:** A CUDA toolkit supported by the installed cuDNN release
- **Family GPU:** Architecture supported by the selected cuDNN engine
- **Family platforms:** Linux (executed); Windows (source-supported, not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Python and C++ representative samples executed with CUDA 13, cuDNN 9.15.1, and SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

The value is stored in the graph context. During node property inference, tensor attributes that did not set `data_type` explicitly inherit this IO data type when they are not virtual tensors.

Explicit tensor data types take precedence. Use `Tensor_attributes::set_data_type(...)` when one tensor in a graph needs a different type, such as an FP32 scale, bias, mean, or variance tensor in a mixed-precision normalization graph.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2328) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3515).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_io_data_type()`. Use the `cudnn.pygraph(io_data_type=...)` constructor argument.
