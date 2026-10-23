# resample

Append a resampling or pooling forward operation to the graph.

## C++ synopsis

```cpp
auto [Y, Index] = graph.resample(
    X,
    fe::graph::Resample_attributes()
        .set_generate_index(false)
        .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)
        .set_padding_mode(fe::PaddingMode_t::NEG_INF_PAD)
        .set_window({2, 2})
        .set_stride({2, 2})
        .set_pre_padding({0, 0})
        .set_post_padding({0, 0}));
```

## Parameters

- `X` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Input tensor. The first two dimensions are treated as batch and channel dimensions; remaining dimensions are spatial dimensions.

- `attributes` (`fe::graph::Resample_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_generate_index(bool)`: Required. When `true` and the resampling mode is `MAXPOOL`, the operation returns an index tensor in addition to `Y`. When `false`, the returned `Index` pointer is `nullptr`.

- `attributes.set_resampling_mode(fe::ResampleMode_t)`: Required. Supported enum values include `AVGPOOL_EXCLUDE_PADDING`, `AVGPOOL_INCLUDE_PADDING`, `BILINEAR`, `NEAREST`, and `MAXPOOL`, subject to cuDNN backend support.

- `attributes.set_padding_mode(fe::PaddingMode_t)`: Required padding behavior, such as `NEG_INF_PAD`, `ZERO_PAD`, or `EDGE_VAL_PAD`, subject to the selected resampling mode.

- `attributes.set_window(...)`, `attributes.set_stride(...)`, `attributes.set_pre_padding(...)`, and `attributes.set_post_padding(...)`: Spatial vectors. Integer overloads are converted to `cudnnFraction_t` values with denominator `1`; fractional overloads can be used for modes that need fractional parameters.

## Returns

- `Y` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Output tensor attributes. If dimensions are not set explicitly, the frontend infers spatial output dimensions as `1 + (input + pre_padding + post_padding - window) / stride`.

- `Index` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`): Max-pooling index tensor attributes when `generate_index == true` and the mode is `MAXPOOL`; otherwise `nullptr`.

## Errors

`Graph::resample()` appends a node and returns tensor attributes. Missing `generate_index`, invalid resampling mode, unsupported padding/mode combinations, malformed tensor metadata, or backend failures are reported by later graph lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** 8.5+ for the C++ frontend; Python wrapper paths require 9.12+
- **Family CUDA:** A CUDA toolkit supported by the installed cuDNN release
- **Family GPU:** Architecture supported by the selected cuDNN engine
- **Family platforms:** Linux (executed); Windows (source-supported, not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Python and C++ representative samples executed with CUDA 13, cuDNN 9.15.1, and SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

When output strides are not set, the frontend generates NHWC-style strides. If an index tensor is produced, it inherits the output dimensions and receives generated strides when not set explicitly.

The deprecated `set_is_inference(bool)` attribute maps to `set_generate_index(!value)`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L481) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/resample.h#L279).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/resample_basic.cc)

<Code
  src="../../samples/cpp/resample_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

There is no direct Python graph binding documented for `resample()` in `docs/api/python`.
