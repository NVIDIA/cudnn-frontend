# sdpa_fp8

Build a regular FP8 scaled dot-product attention operation in a C++ frontend
graph.

## C++ synopsis

```cpp
std::array<std::shared_ptr<fe::graph::Tensor_attributes>, 4>
fe::graph::Graph::sdpa_fp8(
    std::shared_ptr<fe::graph::Tensor_attributes> q,
    std::shared_ptr<fe::graph::Tensor_attributes> k,
    std::shared_ptr<fe::graph::Tensor_attributes> v,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_q,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_k,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_v,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_s,
    std::shared_ptr<fe::graph::Tensor_attributes> scale_s,
    std::shared_ptr<fe::graph::Tensor_attributes> scale_o,
    fe::graph::SDPA_fp8_attributes attributes);
```

Typical use:

```cpp
auto sdpa_fp8_options = fe::graph::SDPA_fp8_attributes()
                            .set_name("sdpa_fp8")
                            .set_generate_stats(generate_stats)
                            .set_attn_scale(attn_scale);

auto [O, Stats, Amax_S, Amax_O] =
    graph->sdpa_fp8(Q, K, V,
                    descale_q, descale_k, descale_v,
                    descale_s, scale_s, scale_o,
                    sdpa_fp8_options);
```

## Parameters

- `q`, `k`, `v` (`std::shared_ptr<Tensor_attributes>`, required): FP8 query,
  key, and value tensors. Regular FP8 samples use `FP8_E4M3` IO dtype and
  4-D shapes such as `{B, H, S, D}`.

- `descale_q`, `descale_k`, `descale_v` (`std::shared_ptr<Tensor_attributes>`,
  required): Scalar-like `FLOAT` descale tensors for Q, K, and V. Samples use
  shape `{1, 1, 1, 1}` and stride `{1, 1, 1, 1}`.

- `descale_s` (`std::shared_ptr<Tensor_attributes>`, required): Scalar-like
  `FLOAT` descale tensor for the FP8 score intermediate.

- `scale_s` (`std::shared_ptr<Tensor_attributes>`, required): Scalar-like
  `FLOAT` scale tensor for the FP8 score intermediate.

- `scale_o` (`std::shared_ptr<Tensor_attributes>`, required): Scalar-like
  `FLOAT` scale tensor for output quantization.

- `attributes` (`fe::graph::SDPA_fp8_attributes`, required): Forward attribute
  object. `SDPA_fp8_attributes` is an alias of `SDPA_attributes`, defined in
  `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_generate_stats(bool)`: Controls whether the returned `Stats`
  tensor is real. Use `true` for training/backward.

- `attributes.set_attn_scale(float or Tensor_attributes)`: Sets the score scale.

- `attributes.set_bias`, mask setters, sequence-length setters, dropout
  setters, page-table setters, sink-token setters, score-statistic setters, and
  score modifier function setters: Same logical role as the Python `sdpa_fp8`
  parameters and the standard `sdpa` attributes, subject to FP8 support limits.

- `attributes.set_compute_data_type(fe::DataType_t)` and
  `attributes.set_name(std::string)`: Set compute type and operation name.

## Returns

- `O` (`std::shared_ptr<Tensor_attributes>`): Attention output tensor.

- `Stats` (`std::shared_ptr<Tensor_attributes>` or `nullptr`): Softmax
  statistics when `set_generate_stats(true)` is used; otherwise `nullptr`.

- `Amax_S` (`std::shared_ptr<Tensor_attributes>`): Absolute maximum of the score
  tensor.

- `Amax_O` (`std::shared_ptr<Tensor_attributes>`): Absolute maximum of the
  output tensor.

## Errors

`Graph::sdpa_fp8` returns tensor attributes immediately. Invalid scale tensors,
mask combinations, page tables, output metadata, dtypes, strides, architecture,
or cuDNN version support are reported later through graph lifecycle `error_t`
statuses.

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

Regular FP8 is distinct from MXFP8. This overload uses scalar `FLOAT` scale and
descale tensors. The MXFP8 overload takes only Q/K/V block scale-factor tensors
and is documented separately in `sdpa_mxfp8.md`.

The repository C++ forward sample validates support only when cuDNN is 9.1.0 or
later and the device is Hopper or newer.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2496) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4146).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_fp8_basic.cc)

<Code
  src="../../samples/cpp/sdpa_fp8_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.sdpa_fp8` reference](/api/python/sdpa-fp8) for Python exceptions, wrapper behavior, and current availability constraints.
