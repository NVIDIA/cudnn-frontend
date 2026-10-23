# batchnorm_inference

C++ graph API counterpart for Python `batchnorm_inference`.

## C++ synopsis

```cpp
auto Y = graph->batchnorm_inference(
    X, mean, inv_var, scale, bias,
    fe::graph::Batchnorm_inference_attributes()
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("batchnorm_inference"));
```

## Parameters

- `X` (`std::shared_ptr<Tensor_attributes>`, required): Input tensor attributes.

- `mean` and `inv_var` (`std::shared_ptr<Tensor_attributes>`, required): Inference statistics.

- `scale` and `bias` (`std::shared_ptr<Tensor_attributes>`, required): Broadcast scale and bias tensors.

- `attributes` (`fe::graph::Batchnorm_inference_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Optional compute type, commonly `FLOAT` for mixed precision.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Inference-normalized output.

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

On CUDA 13.1, cuDNN 9.15.1, and SM100, a standalone inference graph produced
no valid engine configuration, while the BatchNorm training forward and
backward samples passed.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2417) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3816).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API on the verified SM100/cuDNN 9.15.1 environment.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.batchnorm_inference` reference](/api/python/batchnorm-inference) for Python exceptions, wrapper behavior, and current availability constraints.
