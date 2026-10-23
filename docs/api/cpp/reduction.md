# reduction

Append a reduction operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->reduction(
    X,
    fe::graph::Reduction_attributes()
        .set_name("sum")
        .set_mode(fe::ReductionMode_t::ADD)
        .set_compute_data_type(fe::DataType_t::FLOAT));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input tensor attributes.

- `attributes` (`fe::graph::Reduction_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_mode(fe::ReductionMode_t)`: Selects the reduction operation.

- `attributes.set_compute_data_type(fe::DataType_t)`: Compute type for accumulation.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Reduced output tensor attributes. Set output dimensions and strides before validation when the intended reduced axes are not otherwise inferable.

## Errors

Reduction node creation returns tensor attributes immediately. Invalid output shape metadata, unsupported modes, unsupported layouts, and unsupported dtypes are reported later by graph lifecycle `error_t` statuses.

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

`reduction` infers reduced axes from output tensor metadata. If the intended shape is ambiguous, set dimensions and strides on the returned output before validation.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L477) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/reduction.h#L198).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/reduction_basic.cc)

<Code
  src="../../samples/cpp/reduction_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.reduction` reference](/api/python/reduction) for Python exceptions, wrapper behavior, and current availability constraints.
