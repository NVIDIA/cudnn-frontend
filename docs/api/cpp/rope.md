# rope

Append a rotary positional embedding (RoPE) forward operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto Y = graph->rope(
    input,
    freqs,
    fe::graph::RoPE_attributes()
        .set_output_scale(1.0f)
        .set_rope_dim(0)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("rope"));
```

## Parameters

- `input` (`std::shared_ptr<Tensor_attributes>`, required): Tensor to rotate.

- `freqs` (`std::shared_ptr<Tensor_attributes>`, required): Frequency tensor, commonly `FLOAT` and broadcastable to `input`.

- `attributes` (`fe::graph::RoPE_attributes`, required): Attribute class defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_output_scale(float)`: Output scale multiplier.

- `attributes.set_rope_dim(int64_t)`: Rotary dimension. `0` uses the full input head dimension.

- `attributes.set_compute_data_type(fe::DataType_t)`: Compute type.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `Y` (`std::shared_ptr<Tensor_attributes>`): Rotated output tensor attributes.

## Errors

Node creation returns tensor attributes. Shape, frequency metadata, dtype/layout support, and fusion support are checked by later lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: SDPA and RoPE**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** Standalone sample requires cuDNN 9.24+; compiled but not executed with cuDNN 9.15.1
- **Family release status:** Stable with option-specific constraints
- **Family cuDNN:** Option and implementation dependent
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** SM80+ with narrower limits by implementation
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** SDPA, masking, cache, dropout, and selected backward samples executed on SM100; standalone RoPE samples were compiled or syntax-checked because they require cuDNN 9.24+
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

RoPE is commonly fused into attention graphs by feeding rotated `Q` and `K` to `sdpa`. Those rotated tensors must still satisfy the attention API's dtype, stride, grouping, and head-dimension constraints.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2476) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4062).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/rope_basic.cc)

<Code
  src="../../samples/cpp/rope_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.rope` reference](/api/python/rope) for Python exceptions, wrapper behavior, and current availability constraints.
