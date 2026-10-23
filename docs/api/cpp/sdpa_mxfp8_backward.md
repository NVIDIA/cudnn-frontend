# sdpa_mxfp8_backward

Append an MXFP8 scaled dot-product attention backward operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV] =
    graph->sdpa_fp8_backward(
        Q, Q_T, K, K_T, V,
        O_f16, dO_f16, dO, dO_T, Stats,
        SF_Q, SF_Q_T, SF_K, SF_K_T, SF_V, SF_dO, SF_dO_T,
        fe::graph::SDPA_fp8_backward_attributes()
            .set_name("sdpa_mxfp8_backward")
            .set_attn_scale(attn_scale));
```

## Parameters

- `Q`, `Q_T`, `K`, `K_T`, `V`, `O_f16`, `dO_f16`, `dO`, `dO_T`, and `Stats` (`std::shared_ptr<Tensor_attributes>`, required): MXFP8 data tensors, transposed forms, FP16/BF16 forward output and gradient forms, and softmax statistics.

- `SF_Q`, `SF_Q_T`, `SF_K`, `SF_K_T`, `SF_V`, `SF_dO`, and `SF_dO_T` (`std::shared_ptr<Tensor_attributes>`, required): MXFP8 block scale-factor tensors. They should use `FP8_E8M0` dtype and `F8_128x4` reordering with dimensions and strides matching the corresponding data tensor orientation.

- `attributes` (`fe::graph::SDPA_fp8_backward_attributes`, required): Attribute class defined in `include/cudnn_frontend/graph_properties.h`. The same attribute class is used for regular FP8 and MXFP8 backward.

- `attributes.set_attn_scale`, `set_padding_mask`, `set_seq_len_q`, `set_seq_len_kv`, `set_diagonal_alignment`, `set_diagonal_band_left_bound`, `set_diagonal_band_right_bound`, `set_deterministic_algorithm`, `set_dropout`, `set_sink_token`, `set_dsink_token`, `set_compute_data_type`, and `set_name`: Main setters corresponding to Python keyword arguments.

## Returns

- `dQ`, `dK`, `dV` (`std::shared_ptr<Tensor_attributes>`): Query, key, and value gradient tensor attributes.

- `Amax_dQ`, `Amax_dK`, `Amax_dV` (`std::shared_ptr<Tensor_attributes>`): Absolute-maximum tensor attributes for gradient quantization paths.

## Errors

The method returns tensor attributes immediately. Invalid MXFP8 scale-factor metadata, incompatible transposed tensors, unsupported masks, dropout, deterministic mode, dtypes, layouts, architectures, or cuDNN versions are reported by lifecycle `error_t` statuses.

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

C++ exposes MXFP8 backward as an overload of `Graph::sdpa_fp8_backward`; Python exposes the same overload as `graph.sdpa_mxfp8_backward(...)`.

MXFP8 samples require cuDNN 9.21.0 or later and Blackwell GPU architecture or newer.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2516) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4196).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_mxfp8_backward_basic.cc)

<Code
  src="../../samples/cpp/sdpa_mxfp8_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.sdpa_mxfp8_backward` reference](/api/python/sdpa-mxfp8-backward) for Python exceptions, wrapper behavior, and current availability constraints.
