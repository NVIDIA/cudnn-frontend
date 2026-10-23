# bn_finalize

Append batch-normalization finalize statistics to the graph.

## C++ synopsis

```cpp
auto [eq_scale, eq_bias, saved_mean, saved_inv_variance,
      next_running_mean, next_running_var] =
    graph.bn_finalize(sum, sq_sum, scale, bias, epsilon, accum_count,
                      fe::graph::BN_finalize_attributes()
                          .set_previous_running_stats(prev_mean,
                                                      prev_var,
                                                      momentum));
```

## Parameters

- `sum` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Per-channel sum tensor, usually produced by `genstats()`.

- `sq_sum` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Per-channel sum-of-squares tensor, usually produced by `genstats()`.

- `scale` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Batch-normalization scale tensor.

- `bias` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Batch-normalization bias tensor.

- `epsilon` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Scalar epsilon tensor.

- `accum_count` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Scalar tensor containing the number of accumulated elements.

- `attributes` (`fe::graph::BN_finalize_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_previous_running_stats(mean, variance, momentum)`: Optional helper that supplies previous running mean, previous running variance, and momentum tensors. When all three are provided, the operation also returns updated running statistics.

## Returns

- `eq_scale` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Equivalent scale tensor for the normalized output.

- `eq_bias` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Equivalent bias tensor for the normalized output.

- `saved_mean` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Saved mean tensor for backward computation.

- `saved_inv_variance` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Saved inverse variance tensor for backward computation.

- `next_running_mean` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`): Updated running mean when previous running stats and momentum are supplied.

- `next_running_var` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`): Updated running variance when previous running stats and momentum are supplied.

## Errors

`Graph::bn_finalize()` appends a node and returns tensor attributes. Missing required tensors, invalid per-channel tensor metadata, unsupported data types, unsupported cuDNN versions, and backend failures are reported by later graph lifecycle `error_t` statuses.

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

Output per-channel tensor dimensions and strides are inferred from `sum` when not specified explicitly. The backend operation uses training finalize mode.

This API is available in the C++ graph surface but is not exposed as a dedicated Python graph method in this repository.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2424) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3661).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/advanced_operations.cc)

<Markdown src="/snippets/generated/cpp/advanced_operations/bn-finalize.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `bn_finalize()` in `docs/api/python`.
