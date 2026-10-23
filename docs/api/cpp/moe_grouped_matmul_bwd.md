# moe_grouped_matmul_bwd

Append a mixture-of-experts grouped matmul weight-gradient operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto dweight = graph->moe_grouped_matmul_bwd(
    doutput,
    token,
    first_token_offset,
    fe::graph::Moe_grouped_matmul_bwd_attributes()
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("moe_grouped_matmul_bwd"));
```

## Parameters

- `doutput` (`std::shared_ptr<Tensor_attributes>`, required): Upstream grouped matmul output gradient tensor attributes.

- `token` (`std::shared_ptr<Tensor_attributes>`, required): Forward token tensor attributes grouped by expert.

- `first_token_offset` (`std::shared_ptr<Tensor_attributes>`, required): `INT32` tensor giving each expert's first token offset.

- `attributes` (`fe::graph::Moe_grouped_matmul_bwd_attributes`, required): Attribute class defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_compute_data_type(fe::DataType_t)`: Sets compute and accumulation type.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `dweight` (`std::shared_ptr<Tensor_attributes>`): Expert weight-gradient tensor attributes, usually shape `[E, H, N]` in repository examples.

## Errors

Node creation returns tensor attributes immediately. Expert offsets, tensor metadata, unsupported dtypes, layouts, cuDNN version, cublasLt version, and architecture support are reported by lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: FP8, MXFP8, NVFP4, and block scaling**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable or experimental by operation
- **Family cuDNN:** Operation-dependent
- **Family CUDA:** CUDA 12 or later; CUDA 13 used for this checkout
- **Family GPU:** Hopper for FP8; Blackwell for MXFP8/NVFP4 and SM100 kernels
- **Family platforms:** Linux (partially executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** C++ FP8 and MXFP8 attention samples executed on SM100; unsupported block-scaling paths are reported locally
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

Repository tests gate `moe_grouped_matmul_bwd` on cuDNN 9.22.0 or later and cublasLt 13.5 or later. Treat those as requirements unless support probing proves otherwise.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2585) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4438).
{/* END GENERATED API SOURCES */}

## Examples

See the complete [MoE grouped weight-gradient sample](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/moe_grouped_matmul_backward.cc). It executes the repository test layout and checks every expert against a dense reference when the required cuDNN and cublasLt versions are available.

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.moe_grouped_matmul_bwd` reference](/api/python/moe-grouped-matmul-bwd) for Python exceptions, wrapper behavior, and current availability constraints.
