# logical_not

C++ pointwise equivalent for the Python `logical_not` binding.

## C++ synopsis

```cpp
auto Y = graph->pointwise(
    input,
    fe::graph::Pointwise_attributes()
        .set_name("logical_not")
        .set_mode(fe::PointwiseMode_t::LOGICAL_NOT)
        .set_compute_data_type(fe::DataType_t::BOOLEAN));
```

## Parameters

- `input` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Tensor attributes passed to the `Graph::pointwise` overload for this operation. The tensors must be shape-compatible under backend pointwise broadcasting rules.

- `attributes` (`fe::graph::Pointwise_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`. Set `mode` to `fe::PointwiseMode_t::LOGICAL_NOT`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type. The example uses `fe::DataType_t::BOOLEAN`.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Output tensor attributes for the pointwise operation. Mark it as an output when it should be written by execution.

## Errors

`Graph::pointwise` appends the node and returns tensor attributes. Invalid broadcasting, tensor metadata, unsupported dtypes, unsupported mode/layout combinations, or unsupported fusions are reported by later graph lifecycle `error_t` statuses.

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

The C++ frontend exposes a general `Graph::pointwise` API rather than one method per Python wrapper. This page maps the Python wrapper `logical_not` to `PointwiseMode_t::LOGICAL_NOT`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L468) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/pointwise.h#L304).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/logical_not_basic.cc)

<Code
  src="../../samples/cpp/logical_not_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.logical_not` reference](/api/python/logical-not) for Python exceptions, wrapper behavior, and current availability constraints.
