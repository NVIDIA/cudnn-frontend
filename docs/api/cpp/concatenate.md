# concatenate

Append a graph concatenate operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->concatenate(
    {X0, X1},
    fe::graph::Concatenate_attributes()
        .set_name("concat")
        .set_axis(1));
```

## Parameters

- `inputs` (`std::vector<std::shared_ptr<Tensor_attributes>>`, required): Tensor attributes to concatenate. Non-concatenated dimensions must be compatible.

- `attributes` (`fe::graph::Concatenate_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_axis(int64_t)`: Sets the concatenation axis.

- `attributes.set_in_place_index(int64_t)`: Optional backend in-place concat selection.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Concatenated output tensor attributes.

## Errors

The method appends the node and returns tensor attributes immediately. Invalid input ranks, dimensions, dtypes, in-place metadata, output metadata, and backend support are reported later by graph lifecycle `error_t` statuses.

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

This is a graph operation. Runtime buffers are concatenated only when the graph executes.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2575) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4394).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/concatenate_basic.cc)

<Code
  src="../../samples/cpp/concatenate_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.concatenate` reference](/api/python/concatenate) for Python exceptions, wrapper behavior, and current availability constraints.
