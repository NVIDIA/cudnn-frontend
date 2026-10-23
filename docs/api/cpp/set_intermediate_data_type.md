# set_intermediate_data_type

Set the default data type for virtual intermediate tensors.

## C++ synopsis

```cpp
graph.set_intermediate_data_type(fe::DataType_t::FLOAT);
```

## Parameters

- `type` (`fe::DataType_t`, required): Default element type used when a virtual tensor has `DataType_t::NOT_SET`.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

`set_intermediate_data_type()` itself returns `Graph&` and does not return `error_t`. Unsupported intermediate types, unsupported fusions, or incompatible tensor metadata are reported later by graph lifecycle calls.

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

The value is stored in the graph context. During node property inference, virtual tensors that did not set `data_type` explicitly inherit this intermediate data type.

This setting is commonly FP32 for mixed-precision graphs that keep inputs and outputs in FP16 or BF16 while accumulating or applying fused intermediate operations in FP32.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2326) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3509).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/graph_configuration.cc)

<Markdown src="/snippets/generated/cpp/graph_configuration/configuration.mdx" />

## Appendix: Python equivalent

There is no post-construction Python method named `set_intermediate_data_type()`. Use the `cudnn.pygraph(intermediate_data_type=...)` constructor argument.
