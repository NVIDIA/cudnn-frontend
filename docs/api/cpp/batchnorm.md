# batchnorm

C++ graph API counterpart for Python `batchnorm`.

## C++ synopsis

```cpp
auto [Y, mean, inv_var, out_mean, out_var] = graph->batchnorm(
    X, scale, bias,
    fe::graph::Batchnorm_attributes()
        .set_epsilon(epsilon)
        .set_previous_running_stats(running_mean, running_var, momentum)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("batchnorm"));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input tensor attributes.

- `scale` and `bias` (`std::shared_ptr<Tensor_attributes>`, required): Broadcast scale and bias tensors.

- `epsilon` (`std::shared_ptr<Tensor_attributes>`, required): Scalar epsilon tensor passed to `attributes.set_epsilon(epsilon)`.

- `running_mean`, `running_var`, and `momentum` (`std::shared_ptr<Tensor_attributes>`, optional as a group): Running-stat tensors passed to `attributes.set_previous_running_stats(running_mean, running_var, momentum)`. If this setter is not called, `out_mean` and `out_var` are returned as `nullptr`.

- `attributes` (`fe::graph::Batchnorm_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type, commonly `FLOAT` for mixed precision.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Normalized output.

- `mean` (`std::shared_ptr<Tensor_attributes>`): Saved mean.

- `inv_var` (`std::shared_ptr<Tensor_attributes>`): Saved inverse variance.

- `out_mean` (`std::shared_ptr<Tensor_attributes> or nullptr`): Updated running mean.

- `out_var` (`std::shared_ptr<Tensor_attributes> or nullptr`): Updated running variance.

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
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2412) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3784).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/batchnorm_basic.cc)

<Code
  src="../../samples/cpp/batchnorm_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.batchnorm` reference](/api/python/batchnorm) for Python exceptions, wrapper behavior, and current availability constraints.
