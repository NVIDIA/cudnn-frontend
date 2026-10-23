# sdpa_backward

Build the backward pass for scaled dot-product attention in a C++ frontend
graph.

## C++ synopsis

```cpp
std::array<std::shared_ptr<fe::graph::Tensor_attributes>, 3>
fe::graph::Graph::sdpa_backward(
    std::shared_ptr<fe::graph::Tensor_attributes> q,
    std::shared_ptr<fe::graph::Tensor_attributes> k,
    std::shared_ptr<fe::graph::Tensor_attributes> v,
    std::shared_ptr<fe::graph::Tensor_attributes> o,
    std::shared_ptr<fe::graph::Tensor_attributes> dO,
    std::shared_ptr<fe::graph::Tensor_attributes> stats,
    fe::graph::SDPA_backward_attributes attributes);
```

Typical use:

```cpp
auto sdpa_options = fe::graph::SDPA_backward_attributes()
                        .set_name("sdpa_backward")
                        .set_attn_scale(1.0f / std::sqrt(static_cast<float>(d_qk)))
                        .set_compute_data_type(fe::DataType_t::FLOAT);

auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, sdpa_options);
```

## Parameters

- `q`, `k`, `v` (`std::shared_ptr<Tensor_attributes>`, required): Forward query,
  key, and value tensors. Shape, dtype, head grouping, head-dimension
  divisibility, and stride constraints must match the forward `sdpa` graph.

- `o` (`std::shared_ptr<Tensor_attributes>`, required): Forward SDPA output.

- `dO` (`std::shared_ptr<Tensor_attributes>`, required): Upstream output
  gradient. Shape must match `o`.

- `stats` (`std::shared_ptr<Tensor_attributes>`, required): Softmax statistics
  from the forward `sdpa` call with `set_generate_stats(true)`. It is commonly a
  `FLOAT` tensor with shape `{B, H_q, S_q, 1}`.

- `attributes` (`fe::graph::SDPA_backward_attributes`, required): Attribute
  object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_attn_scale(float or Tensor_attributes)`: Sets the same
  attention scale used by forward.

- `attributes.set_bias(...)` and `attributes.set_dbias(...)`: Set the forward
  bias and optional bias-gradient output tensor.

- `attributes.set_padding_mask(bool)`, `set_seq_len_q(...)`, and
  `set_seq_len_kv(...)`: Configure variable-length masking. When padding mask is
  enabled, both sequence length tensors are required.

- `attributes.set_dropout(...)`: Configures the same dropout semantics as the
  forward graph. Dropout is applied after softmax and before multiplication by
  `V`.

- `attributes.set_score_mod(...)` and score-modifier backward support: Configure
  custom score modifier function behavior when the graph uses flexible score
  modification.

- `attributes.set_sink_token(...)` and `set_dSink_token(...)`: Set optional
  sink-token input and sink-token gradient output.

- `attributes.set_diagonal_alignment(...)`,
  `set_diagonal_band_left_bound(...)`, and
  `set_diagonal_band_right_bound(...)`: Preferred diagonal-band mask controls.
  Do not combine the left-bound setter with deprecated sliding-window setters.

- Deprecated mask setters: causal-mask and sliding-window convenience setters
  remain available for compatibility, but new code should use diagonal-band
  bounds.

- `attributes.set_deterministic_algorithm(bool)`: Requests deterministic
  backward where supported.

- `attributes.set_compute_data_type(fe::DataType_t)` and
  `attributes.set_name(std::string)`: Set compute type and operation name.

## Returns

- `dQ` (`std::shared_ptr<Tensor_attributes>`): Query gradient tensor.

- `dK` (`std::shared_ptr<Tensor_attributes>`): Key gradient tensor.

- `dV` (`std::shared_ptr<Tensor_attributes>`): Value gradient tensor.

Optional `dbias` and `dSink_token` outputs are tensors passed through the
attributes object, not members of the returned array.

## Errors

`Graph::sdpa_backward` returns tensor attributes immediately. Mismatched forward
metadata, invalid saved stats, unsupported masks, custom score modifiers,
dropout, determinism requests, dtype/layout limits, and architecture limits are
reported by lifecycle `error_t` statuses.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: SDPA and RoPE**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
- **API-specific availability:** See local constraints
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

Backward support is often narrower than forward support. Verify every graph that
uses custom score modifier functions, dropout, sink tokens, padding masks, or
deterministic execution with `check_support()` before building plans.

`Scaled_dot_product_flash_attention_backward_attributes` is a deprecated alias
of `SDPA_backward_attributes`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2556) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4317).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_backward_basic.cc)

<Code
  src="../../samples/cpp/sdpa_backward_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.sdpa_backward` reference](/api/python/sdpa-backward) for Python exceptions, wrapper behavior, and current availability constraints.
