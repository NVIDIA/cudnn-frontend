# slice

Append a graph slice operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->slice(
    X,
    fe::graph::Slice_attributes()
        .set_name("slice")
        .set_slices({{1, 3}, {4, 12}, {0, 8}})
        .set_strides({1, 2, 1})
        .set_compute_data_type(fe::DataType_t::FLOAT));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input tensor attributes.

- `attributes` (`fe::graph::Slice_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_slices(std::vector<std::pair<int64_t, int64_t>>)`: Sets per-axis half-open start and end indices.

- `attributes.set_strides(std::vector<int64_t>)`: Sets per-axis slice steps.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Sliced output tensor attributes.

## Errors

The method appends the node and returns tensor attributes immediately. Invalid slice metadata, output metadata, dtype/layout support, and backend support are reported later by graph lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; compiled but not executed with cuDNN 9.15.1
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

This is a graph operation, not a host-side view. Runtime buffers are sliced only when the graph executes.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2564) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4343).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/slice_basic.cc)

<Code
  src="../../samples/cpp/slice_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.slice` reference](/api/python/slice) for Python exceptions, wrapper behavior, and current availability constraints.
