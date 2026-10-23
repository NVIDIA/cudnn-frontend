# tensor

Create C++ graph tensor attributes from explicit `Tensor_attributes` metadata.

## C++ synopsis

```cpp
auto X = graph->tensor(fe::graph::Tensor_attributes()
                           .set_name("X")
                           .set_dim({B, S, D})
                           .set_stride({S * D, D, 1})
                           .set_data_type(fe::DataType_t::BFLOAT16));
```

## Parameters

- `Tensor_attributes` (`fe::graph::Tensor_attributes`, required): Attribute object describing logical dimensions, strides, dtype, virtual/output state, pass-by-value state, ragged offsets, reordering, name, and optional UID. `Tensor_attributes` is defined in `include/cudnn_frontend/graph_properties.h`.

- `set_dim(std::vector<int64_t>)`: Sets logical tensor dimensions. The number of values is the tensor rank.

- `set_stride(std::vector<int64_t>)`: Sets logical strides in elements. The rank must match `dim`.

- `set_data_type(fe::DataType_t)`: Sets the tensor element type. User inputs and outputs should usually set it explicitly.

- `set_is_virtual(bool)`: Marks a tensor that does not require a user-provided runtime buffer.

- `set_is_pass_by_value(bool)`: Marks scalar or small runtime parameters that are passed by value through the variant pack.

- `set_ragged_offset(std::shared_ptr<Tensor_attributes>)`: Sets optional packed or ragged layout offset metadata used by APIs that support ragged storage.

- `set_reordering_type(fe::TensorReordering_t)`: Sets optional reordered-layout metadata. MXFP8 scale-factor tensors use `F8_128x4`.

- `set_name(std::string)`: Sets the user-visible tensor name.

- `set_uid(int64_t)`: Sets an explicit frontend UID. Prefer generated UIDs unless serialized or externally described graphs require fixed IDs.

## Returns

- `X` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Tensor attributes registered as a graph input. The object can be passed to graph operations and configured further before validation.

## Errors

`Graph::tensor(Tensor_attributes const&)` returns tensor attributes immediately and does not return `error_t`. Metadata consistency is checked later by `Graph::validate()`, `Graph::build_operation_graph()`, support checking, and plan building.

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

Tensor attributes describe memory but do not allocate it. Runtime buffers are provided to `Graph::execute()` through the variant pack.

Related tensors do not need identical strides unless an operation-specific page states that requirement. Each tensor's stride must match its own runtime memory layout and the selected backend engine's constraints.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L604) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3565).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/tensor_basic.cc)

<Code
  src="../../samples/cpp/tensor_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.tensor` reference](/api/python/tensor) for Python exceptions, wrapper behavior, and current availability constraints.
