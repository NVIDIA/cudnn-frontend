# set_compute_data_type

Set the default compute precision for graph operation attributes.

## C++ synopsis

```cpp
graph.set_compute_data_type(fe::DataType_t::FLOAT);
```

## Parameters

- `type` (`fe::DataType_t`, required): Default compute type used when an operation attribute object has `compute_data_type == DataType_t::NOT_SET`.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_compute_data_type()` itself returns `Graph&` and does not return `error_t`. Unsupported compute types or unsupported operation/type combinations are reported by later lifecycle calls.

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

Operation attributes inherit this value during graph property inference unless they explicitly call their own `set_compute_data_type(...)`.

Compute type is separate from tensor storage type. For example, input and output tensors can use FP16 or BF16 while the operation compute type is FP32.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2330) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3521).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_compute_data_type()`. Use the `cudnn.pygraph(compute_data_type=...)` constructor argument.
