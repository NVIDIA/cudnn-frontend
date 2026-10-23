# moe_grouped_matmul

Append a mixture-of-experts grouped matmul forward operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto output = graph->moe_grouped_matmul(
    token,
    weight,
    first_token_offset,
    token_index,
    token_ks,
    fe::graph::Moe_grouped_matmul_attributes()
        .set_mode(fe::MoeGroupedMatmulMode_t::NONE)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_top_k(0)
        .set_name("moe_grouped_matmul"));
```

## Parameters

- `token` (`std::shared_ptr<Tensor_attributes>`, required): Token tensor grouped by expert. Repository tests use shape `[1, T, H]` with stride `[T * H, H, 1]`.

- `weight` (`std::shared_ptr<Tensor_attributes>`, required): Expert weight tensor. Repository tests use shape `[E, H, N]` with stride `[H * N, 1, H]`.

- `first_token_offset` (`std::shared_ptr<Tensor_attributes>`, required): `INT32` tensor giving each expert's first token offset. Repository tests use shape `[E, 1, 1]`.

- `token_index` (`std::shared_ptr<Tensor_attributes>` or `nullptr`): Optional token index tensor for routing modes that need it.

- `token_ks` (`std::shared_ptr<Tensor_attributes>` or `nullptr`): Optional per-token routing metadata.

- `attributes` (`fe::graph::Moe_grouped_matmul_attributes`, required): Attribute class defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_mode(fe::MoeGroupedMatmulMode_t)`: Selects the routing mode.

- `attributes.set_top_k(int64_t)`: Sets the top-k routing value.

- `attributes.set_compute_data_type(fe::DataType_t)`: Sets compute and accumulation type.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `output` (`std::shared_ptr<Tensor_attributes>`): Forward grouped matmul output tensor attributes, usually shape `[1, T, N]` in repository examples.

## Errors

Node creation returns tensor attributes immediately. Expert offsets, routing metadata, unsupported modes, dtypes, layouts, and architecture support are reported by lifecycle `error_t` statuses.

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

Token order and `first_token_offset` define expert grouping. Keep those metadata synchronized with runtime token buffers.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2578) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4413).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/moe_grouped_matmul_basic.cc)

<Code
  src="../../samples/cpp/moe_grouped_matmul_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.moe_grouped_matmul` reference](/api/python/moe-grouped-matmul) for Python exceptions, wrapper behavior, and current availability constraints.
