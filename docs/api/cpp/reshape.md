# reshape

Append a graph reshape operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->reshape(
    X,
    fe::graph::Reshape_attributes()
        .set_name("reshape")
        .set_reshape_mode(fe::ReshapeMode_t::VIEW_ONLY));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input tensor attributes.

- `attributes` (`fe::graph::Reshape_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_reshape_mode(fe::ReshapeMode_t)`: Sets `VIEW_ONLY` or `LOGICAL` reshape behavior.

- `attributes.set_dim(std::vector<int64_t>)` and `attributes.set_stride(std::vector<int64_t>)`: Optional attribute-level target metadata. Examples commonly set output dimensions and strides on the returned tensor instead.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type inherited from the base attributes API.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Reshaped output tensor attributes. Set target dimensions and strides before validation when they are not already encoded in attributes.

## Errors

The method appends the node and returns tensor attributes immediately. Invalid shape deduction, output metadata, reshape mode support, and backend support are reported later by graph lifecycle `error_t` statuses.

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

This is a graph operation, not a host-side view. Runtime buffers are reshaped only when the graph executes.

On cuDNN 9.15.1 and SM100, a terminal `VIEW_ONLY` reshape produced no valid
engine configuration. Keep it inside a supported fusion on that backend. The
repository's `LOGICAL` reshape example requires cuDNN 9.22 or later.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L482) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/reshape.h#L162).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API on the verified SM100/cuDNN 9.15.1 environment.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.reshape` reference](/api/python/reshape) for Python exceptions, wrapper behavior, and current availability constraints.
