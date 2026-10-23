# sdpa_fp8_backward

Append a regular FP8 scaled dot-product attention backward operation to a C++ frontend graph.

## C++ synopsis

```cpp
auto [dQ, dK, dV, Amax_dQ, Amax_dK, Amax_dV, Amax_dP] =
    graph->sdpa_fp8_backward(
        Q, K, V, O, dO, Stats,
        descale_q, descale_k, descale_v, descale_o, descale_dO,
        descale_s, descale_dP,
        scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
        fe::graph::SDPA_fp8_backward_attributes()
            .set_name("sdpa_fp8_backward")
            .set_attn_scale(attn_scale));
```

## Parameters

- `Q`, `K`, `V`, `O`, `dO`, and `Stats` (`std::shared_ptr<Tensor_attributes>`, required): Forward FP8 tensors, upstream output gradient, and softmax statistics.

- `descale_q`, `descale_k`, `descale_v`, `descale_o`, `descale_dO`, `descale_s`, and `descale_dP` (`std::shared_ptr<Tensor_attributes>`, required): Scalar-like descale tensors for the FP8 backward paths.

- `scale_s`, `scale_dQ`, `scale_dK`, `scale_dV`, and `scale_dP` (`std::shared_ptr<Tensor_attributes>`, required): Scalar-like scale tensors for score and gradient quantization paths.

- `attributes` (`fe::graph::SDPA_fp8_backward_attributes`, required): Attribute class defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_attn_scale`, `set_padding_mask`, `set_seq_len_q`, `set_seq_len_kv`, `set_diagonal_alignment`, `set_diagonal_band_left_bound`, `set_diagonal_band_right_bound`, `set_deterministic_algorithm`, `set_dropout`, `set_sink_token`, `set_dsink_token`, `set_compute_data_type`, and `set_name`: Main setters corresponding to Python keyword arguments.

## Returns

- `dQ`, `dK`, `dV` (`std::shared_ptr<Tensor_attributes>`): Query, key, and value gradient tensor attributes.

- `Amax_dQ`, `Amax_dK`, `Amax_dV`, `Amax_dP` (`std::shared_ptr<Tensor_attributes>`): Absolute-maximum tensor attributes for gradient quantization paths.

## Errors

The method returns tensor attributes immediately. Invalid metadata, incompatible forward/backward options, unsupported masks, dropout, deterministic mode, dtypes, layouts, architectures, or cuDNN versions are reported by lifecycle `error_t` statuses.

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

`SDPA_fp8_backward_attributes` also supports an optional caller-provided `DSINK_TOKEN` output through `set_dsink_token`; it is not part of the returned fixed-size array.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2516) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4196).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_fp8_backward_basic.cc)

<Code
  src="../../samples/cpp/sdpa_fp8_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.sdpa_fp8_backward` reference](/api/python/sdpa-fp8-backward) for Python exceptions, wrapper behavior, and current availability constraints.
