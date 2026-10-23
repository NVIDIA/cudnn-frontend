# rmsnorm_backward

C++ graph API counterpart for Python `rmsnorm_backward`.

## C++ synopsis

```cpp
auto [dX, dScale, dBias] = graph->rmsnorm_backward(
    dY, X, scale, inv_variance,
    fe::graph::Rmsnorm_backward_attributes()
        .has_dbias(has_dbias)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("rmsnorm_backward"));
```

## Parameters

- `dY`, `X`, `scale`, and `inv_variance` (`std::shared_ptr<Tensor_attributes>`, required): Gradient, forward input, scale, and saved inverse-variance tensor attributes.

- `attributes.has_dbias(bool)`: Selects whether the operation creates and returns the optional `dBias` tensor. When this is `false`, the third return value is `nullptr`.

- `attributes` (`fe::graph::Rmsnorm_backward_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type, commonly `FLOAT` for mixed precision.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `dX` (`std::shared_ptr<Tensor_attributes>`): Input gradient.

- `dScale` (`std::shared_ptr<Tensor_attributes>`): Scale gradient.

- `dBias` (`std::shared_ptr<Tensor_attributes> or nullptr`): Bias gradient when `attributes.has_dbias(true)` is used; otherwise `nullptr`.

## Errors

The C++ method appends the node and returns tensor attributes immediately. Invalid tensor metadata, statistics shapes, broadcast metadata, phase-dependent outputs, dtype/layout support, and architecture support are reported by later graph lifecycle `error_t` statuses from `validate`, `build_operation_graph`, `check_support`, or `build_plans`.

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

The C++ method returns tensor attributes only. Runtime buffers are provided later through the graph execution variant pack. Saved statistics needed by a backward graph must be materialized or carried to the backward graph explicitly.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2484) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4098).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/rmsnorm_backward_basic.cc)

<Code
  src="../../samples/cpp/rmsnorm_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.rmsnorm_backward` reference](/api/python/rmsnorm-backward) for Python exceptions, wrapper behavior, and current availability constraints.
