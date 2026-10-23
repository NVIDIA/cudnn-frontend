# dbn_weight

Append batch-normalization weight-gradient helper outputs to the graph.

## C++ synopsis

```cpp
auto [dscale, dbias, eq_scale_dy, eq_scale_x, eq_bias] =
    graph.dbn_weight(dy, x, mean, inv_variance, scale,
                     fe::graph::DBN_weight_attributes());
```

## Parameters

- `dy` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Upstream gradient tensor.

- `x` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Forward input tensor.

- `mean` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Per-channel mean tensor.

- `inv_variance` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Per-channel inverse variance tensor.

- `scale` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Batch-normalization scale tensor.

- `attributes` (`fe::graph::DBN_weight_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

## Returns

- `dscale` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Gradient with respect to the scale tensor.

- `dbias` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Gradient with respect to the bias tensor.

- `eq_scale_dy` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Equivalent scale factor applied to `dy` in fused backward graphs.

- `eq_scale_x` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Equivalent scale factor applied to `x` in fused backward graphs.

- `eq_bias` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Equivalent bias term for fused backward graphs.

## Errors

`Graph::dbn_weight()` appends a node and returns tensor attributes. Missing required tensors, shape inference failures, unsupported fusion patterns, unsupported data types, unsupported cuDNN versions, and backend failures are reported by later graph lifecycle `error_t` statuses.

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

The implementation infers per-channel output dimensions from `dy` when output dimensions are not set. The source currently notes that inference from `dy` is the supported path.

This API is available in the C++ graph surface but is not exposed as a dedicated Python graph method in this repository.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2444) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3957).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/advanced_operations.cc)

<Markdown src="/snippets/generated/cpp/advanced_operations/dbn-weight.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `dbn_weight()` in `docs/api/python`.
