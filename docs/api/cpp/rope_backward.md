# rope_backward

Append a rotary positional embedding (RoPE) backward operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto dX = graph->rope_backward(
    dY,
    freqs,
    fe::graph::RoPE_backward_attributes()
        .set_output_scale(1.0f)
        .set_rope_dim(0)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_name("rope_backward"));
```

## Parameters

- `dY` (`std::shared_ptr<Tensor_attributes>`, required): Upstream gradient tensor.

- `freqs` (`std::shared_ptr<Tensor_attributes>`, required): Frequency tensor used by the matching forward operation.

- `attributes` (`fe::graph::RoPE_backward_attributes`, required): Attribute class defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_output_scale(float)`: Scale multiplier. Use the same value as the matching forward `rope` call.

- `attributes.set_rope_dim(int64_t)`: Rotary dimension. Use the same value as the matching forward `rope` call.

- `attributes.set_compute_data_type(fe::DataType_t)`: Compute type.

- `attributes.set_name(std::string)`: Operation name.

## Returns

- `dX` (`std::shared_ptr<Tensor_attributes>`): Input-gradient tensor attributes. The C++ graph marks this output non-virtual by default.

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

When RoPE feeds SDPA, the common backward pattern is `sdpa_backward` followed by `rope_backward` for the rotated query and key gradients.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2480) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4082).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/rope_backward_basic.cc)

<Code
  src="../../samples/cpp/rope_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.rope_backward` reference](/api/python/rope-backward) for Python exceptions, wrapper behavior, and current availability constraints.
