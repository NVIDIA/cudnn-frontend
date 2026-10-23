# genstats

Append a batch-normalization statistics generation operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto [sum, square_sum] = graph->genstats(
    X,
    fe::graph::Genstats_attributes()
        .set_name("stats")
        .set_compute_data_type(fe::DataType_t::FLOAT));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input tensor attributes.

- `attributes` (`fe::graph::Genstats_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Compute type for sum and square-sum accumulation.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `sum` (`std::shared_ptr<Tensor_attributes>`): Sum output tensor attributes.

- `square_sum` (`std::shared_ptr<Tensor_attributes>`): Sum-of-squares output tensor attributes.

## Errors

Genstats node creation returns tensor attributes immediately. Invalid tensor metadata, output metadata, unsupported layouts, unsupported dtypes, and unsupported architectures are reported later by graph lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Normalization families**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** Operation-dependent
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** Engine-dependent
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** LayerNorm and RMSNorm samples executed in Python and C++; additional C++ normalization samples executed on SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

The generated tensors are intended for normalization-style statistics paths. Use `reduction` for general reduction modes.

On CUDA 13.1, cuDNN 9.15.1, and SM100, standalone `genstats` produced no valid
engine configuration, while normalization samples that own their statistics
passed.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2470) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4007).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API on the verified SM100/cuDNN 9.15.1 environment.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.genstats` reference](/api/python/genstats) for Python exceptions, wrapper behavior, and current availability constraints.
