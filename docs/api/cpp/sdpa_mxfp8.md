# sdpa_mxfp8

Build an MXFP8 scaled dot-product attention operation with the C++
`Graph::sdpa_fp8` MXFP8 overload.

## C++ synopsis

There is no separately named `Graph::sdpa_mxfp8` method in the C++ API. Use the
six-scale-factor overload of `Graph::sdpa_fp8`:

```cpp
std::array<std::shared_ptr<fe::graph::Tensor_attributes>, 3>
fe::graph::Graph::sdpa_fp8(
    std::shared_ptr<fe::graph::Tensor_attributes> q,
    std::shared_ptr<fe::graph::Tensor_attributes> k,
    std::shared_ptr<fe::graph::Tensor_attributes> v,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_q,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_k,
    std::shared_ptr<fe::graph::Tensor_attributes> descale_v,
    fe::graph::SDPA_fp8_attributes attributes);
```

Typical use:

```cpp
auto sdpa_fp8_options = fe::graph::SDPA_fp8_attributes()
                            .set_name("sdpa_fp8_mxfp8")
                            .set_generate_stats(true)
                            .set_attn_scale(attn_scale);

auto [O, Stats, Amax_O] =
    graph->sdpa_fp8(Q, K, V, SF_Q, SF_K, SF_V, sdpa_fp8_options);
```

The overload detects MXFP8 mode from scale-factor tensors with dtype
`FP8_E8M0` and reordering `F8_128x4`.

## Parameters

- `q`, `k`, `v` (`std::shared_ptr<Tensor_attributes>`, required): FP8 data
  tensors, usually `FP8_E4M3` with shape `{B, H, S, D}` and head-dimension
  stride `1`.

- `descale_q` / `SF_Q` (`std::shared_ptr<Tensor_attributes>`, required): MXFP8
  block scale-factor tensor for Q. It must use dtype `FP8_E8M0`, reordering
  `F8_128x4`, shape `{B, H, S_q_padded, D_scale_padded}`, and last stride `1`.

- `descale_k` / `SF_K` (`std::shared_ptr<Tensor_attributes>`, required): MXFP8
  block scale-factor tensor for K. It follows the same layout pattern as
  `SF_Q`, with `S_kv_padded` padded to a multiple of `128`.

- `descale_v` / `SF_V` (`std::shared_ptr<Tensor_attributes>`, required): MXFP8
  block scale-factor tensor for V. Shape is `{B, H, S_scale_padded, D_padded}`;
  the sequence-scale dimension is the contracting dimension and must have
  `stride[2] == 1`.

- `attributes` (`fe::graph::SDPA_fp8_attributes`, required): Attribute object.
  It is an alias of `SDPA_attributes`, defined in
  `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_generate_stats(bool)`: Controls whether `Stats` is returned
  as a real tensor.

- `attributes.set_attn_scale(float or Tensor_attributes)`: Sets the score scale.

- `attributes.set_diagonal_alignment`, `set_diagonal_band_left_bound`, and
  `set_diagonal_band_right_bound`: Preferred diagonal-band controls.

- Deprecated causal-mask setters: Available for compatibility. Prefer diagonal
  band bounds for new code.

- `attributes.set_sink_token(...)` and `attributes.set_unfuse_fma(bool)`: Set
  optional sink token and SM100 softmax FMA behavior where supported.

- `attributes.set_compute_data_type(fe::DataType_t)` and
  `attributes.set_name(std::string)`: Set compute type and operation name.

## Returns

- `O` (`std::shared_ptr<Tensor_attributes>`): Attention output tensor. C++
  samples set it to `BFLOAT16`.

- `Stats` (`std::shared_ptr<Tensor_attributes>` or `nullptr`): Softmax
  statistics when `set_generate_stats(true)` is used.

- `Amax_O` (`std::shared_ptr<Tensor_attributes>`): Absolute maximum of the
  output tensor.

## Errors

MXFP8 node creation returns tensor attributes immediately. Wrong scale-factor
dtype or reordering, unsupported scale shapes, invalid diagonal masks, output
metadata errors, and unsupported version or architecture are reported later by
graph lifecycle `error_t` statuses.

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

The Python docstring states that MXFP8 SDPA requires cuDNN 9.21.0 or later and
Blackwell GPU architecture or newer. Repository C++ samples enforce
`cudnnGetVersion() >= 92100` and a Blackwell-or-newer device check.

Q/K scale factors scale the head dimension in blocks of `32`; V scale factors
scale the sequence dimension in blocks of `32`. This is the main layout
difference from regular FP8 scalar scaling.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2496) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4146).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_mxfp8_basic.cc)

<Code
  src="../../samples/cpp/sdpa_mxfp8_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.sdpa_mxfp8` reference](/api/python/sdpa-mxfp8) for Python exceptions, wrapper behavior, and current availability constraints.
