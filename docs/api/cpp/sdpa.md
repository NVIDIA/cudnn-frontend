# sdpa

Build a scaled dot-product attention (SDPA) operation in a cuDNN frontend C++
graph.

## C++ synopsis

```cpp
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

std::array<std::shared_ptr<fe::graph::Tensor_attributes>, 2>
fe::graph::Graph::sdpa(
    std::shared_ptr<fe::graph::Tensor_attributes> q,
    std::shared_ptr<fe::graph::Tensor_attributes> k,
    std::shared_ptr<fe::graph::Tensor_attributes> v,
    fe::graph::SDPA_attributes attributes);
```

Typical use:

```cpp
auto sdpa_options = fe::graph::SDPA_attributes()
                        .set_name("flash_attention")
                        .set_generate_stats(true)
                        .set_attn_scale(1.0f / std::sqrt(static_cast<float>(d_qk)))
                        .set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
                        .set_diagonal_band_right_bound(0)
                        .set_compute_data_type(fe::DataType_t::FLOAT);

auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);
```

`sdpa` computes:

```text
scores = matmul(q, transpose(k))
scores = scores * attn_scale              # only when attn_scale is provided
scores = score_mod(scores)                # optional score modifier function
scores = scores + bias                    # optional
scores = scores + alibi / diagonal mask   # optional additive term
scores = scores + padding mask            # optional additive term
Stats = logsumexp(scores)                 # only when generate_stats=true
probs = softmax(scores)
probs = dropout(probs)                    # optional; applied after softmax
O = matmul(probs, v)
```

Equivalently, with optional score modifier function $f$, additive bias $B$,
optional additive ALiBi/diagonal mask term $M_d$, and optional additive padding
mask term $M_p$:

$$
\begin{aligned}
S &= QK^\top, \\
\hat{S} &= f\left(\alpha S\right) + B + M_d + M_p, \\
\operatorname{Stats} &= \operatorname{logsumexp}\left(\hat{S}\right), \\
P &= \operatorname{softmax}\left(\hat{S}\right), \\
\tilde{P} &= \operatorname{dropout}\left(P\right), \\
O &= \tilde{P}V.
\end{aligned}
$$

Here $\alpha$ is `attn_scale` when provided; otherwise the multiplication by
$\alpha$ is omitted. $\operatorname{Stats}$ is generated only when
`attributes.set_generate_stats(true)` is used; its `logsumexp` reduction is
taken over the last score dimension, corresponding to the key/value sequence
positions. Terms for disabled options are omitted. Additive mask terms use `0`
for retained scores and a large negative value or negative infinity for masked
scores. When dropout is disabled, $\tilde{P} = P$; when enabled, dropout is
applied to the softmax probabilities before the multiplication by $V$.

The operation appends nodes to `graph` and returns graph tensor attributes. It
does not execute immediately; validate, build, and execute the graph in the
usual C++ frontend workflow.

## Parameters

Core parameters:

- `q` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Query tensor
  attributes. Must be 4-D with logical shape `(B, H_q, S_q, D_qk)`. The last
  stride, corresponding to the head dimension, must be `1`. For the standard
  FP16/BF16 `sdpa` path, `D_qk` must be a multiple of `8`. See Notes for
  architecture- and cuDNN-version-specific maximums.

- `k` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Key tensor
  attributes. Usually shape `(B, H_k, S_kv, D_qk)`. With paged attention, this is
  the key-cache container rather than the logical dense K tensor. Its head
  dimension must match `q`'s `D_qk`, which must be a multiple of `8` for the
  standard FP16/BF16 path.

- `v` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Value tensor
  attributes. Usually shape `(B, H_v, S_kv, D_v)`. With paged attention, this is
  the value-cache container. For the standard FP16/BF16 `sdpa` path, `D_v` must
  be a multiple of `8`.

- `attributes` (`fe::graph::SDPA_attributes`, required): Attribute object that
  carries all optional SDPA inputs, outputs, masks, score transformations, and
  implementation choices. `SDPA_attributes` is defined in
  `include/cudnn_frontend/graph_properties.h`. The public C++ `Graph::sdpa`
  overload takes this object by value and returns only `{O, Stats}` for backward
  compatibility, even though `SDPA_attributes` also defines an internal
  `SDPA_outputs` convenience struct.

`SDPA_attributes` setters:

- `attributes.set_name(std::string const& name)`: Sets the operation name used
  to name generated tensors, for example `"<name>::O"` and
  `"<name>::Stats"`.

- `attributes.set_generate_stats(bool value)`: Controls whether `Stats` is
  created. Use `true` for training when `sdpa_backward` will need softmax
  statistics. Use `false` for inference. In C++, omitting this setter behaves
  like `false` for the public return value, but new code should set it
  explicitly for readability and to match the Python binding.

- `attributes.set_attn_scale(float value)` or
  `attributes.set_attn_scale(std::shared_ptr<Tensor_attributes> value)`: Sets an
  optional scale multiplied into the attention scores after `Q @ K.T`. cuDNN
  does not infer `1 / sqrt(D_qk)`; pass it explicitly when desired. If a tensor
  is used, it is commonly scalar-shaped and pass-by-value. Do not set both the
  float and tensor forms; validation reports `ATTRIBUTE_NOT_SET` for that
  conflict.

- `attributes.set_bias(std::shared_ptr<Tensor_attributes> value)`: Adds an
  optional score bias broadcast-compatible with `(B, H_q, S_q, S_kv)`, commonly
  `(1, H_q, S_q, S_kv)` or `(B, 1, S_q, S_kv)`. Boolean bias tensors are not
  supported.

- `attributes.set_block_mask(std::shared_ptr<Tensor_attributes> value)`: Adds an
  optional packed block mask input. The common tile form is `UINT8` with shape
  `(B, H_q, ceil(S_q / 128), ceil(ceil(S_kv / 128) / 8))`. Requires an
  implementation and cuDNN version that support block masks. Do not force
  `fe::AttentionImplementation_t::COMPOSITE` when `block_mask` is provided; the
  composite implementation does not support `block_mask`. On SM10x, unified
  SDPA block masks require cuDNN 9.26 or newer because older backends have a
  masked-tile initialization bug.

- `attributes.set_alibi_mask(bool value)`: Enables ALiBi score offsets. In the
  FP16/BF16 support surface, ALiBi requires
  `attributes.set_diagonal_band_right_bound(0)`.

- `attributes.set_padding_mask(bool value)`: Enables variable sequence length
  masking. If this is `true`, both `set_seq_len_q` and `set_seq_len_kv` are
  required. If this is `false`, do not set sequence length tensors unless a
  custom score modifier function consumes them.

- `attributes.set_seq_len_q(std::shared_ptr<Tensor_attributes> value)` and
  `attributes.set_seq_len_kv(std::shared_ptr<Tensor_attributes> value)`: Set
  actual query and key/value sequence length tensors. They are normally `INT32`
  tensors with shape `{B, 1, 1, 1}` and stride `{1, 1, 1, 1}`. They are used
  with padding masks, paged attention, ragged layouts, or custom score modifier
  functions that need sequence lengths.

- `attributes.set_dropout(float probability, seed, offset)`: Enables internally
  generated Philox dropout. `probability` is the drop probability and cannot be
  `1.0`. `seed` and `offset` are scalar graph tensors, commonly shape
  `{1, 1, 1, 1}` and stride `{1, 1, 1, 1}`; repository C++ samples use `INT32`
  tensors. Dropout is applied after softmax and before multiplication by `V`.

- `attributes.set_dropout(mask, scale)`: Enables externally supplied dropout.
  The current C++ header order is `(mask, scale)`. `mask` should have the
  attention score shape `(B, H_q, S_q, S_kv)`, or the paged maximum KV length in
  the last dimension when paged attention uses an explicit maximum. `scale` is
  commonly a scalar pass-by-value `FLOAT` tensor containing
  `1 / (1 - probability)`. Do not combine this overload with probability-based
  dropout.

- `attributes.set_rng_dump(std::shared_ptr<Tensor_attributes> value)`: Sets an
  optional output tensor for dumping the generated dropout RNG mask. This is for
  debugging and is used only with probability-based dropout. Its shape should
  match the attention score matrix `(B, H_q, S_q, S_kv)`, or
  `(B, H_q, S_q, max_seq_len_kv)` when paged attention uses
  `set_paged_attention_max_seq_len_kv`. The usual contiguous stride is
  `{H_q * S_q * S_kv, S_q * S_kv, S_kv, 1}`, with `S_kv` replaced by
  `max_seq_len_kv` for paged attention. The data type should be `FLOAT`. The
  public `Graph::sdpa` return array does not include this tensor, so keep the
  pointer and UID you pass to `set_rng_dump`.

- `attributes.set_paged_attention_k_table(value)` and
  `attributes.set_paged_attention_v_table(value)`: Set page tables for looking
  up blocks in the key and value cache containers. Page table tensors are
  normally `INT32` with shape `{B, 1, table_size, 1}` and stride
  `{table_size, table_size, 1, 1}`. Values in both K and V tables are `INT32`
  block indices into their corresponding cache containers.

- `attributes.set_paged_attention_max_seq_len_kv(int value)`: Sets the logical
  maximum KV sequence length when paged attention is active. Do not set this
  unless `set_paged_attention_k_table` or `set_paged_attention_v_table` is used.
  If set, the last dimension of bias and `rng_dump` tensors must match this
  value.

- `attributes.set_compute_data_type(fe::DataType_t value)`: Sets the compute
  type for this operation. When left as `NOT_SET`, the graph-level compute type
  is used. For the unified implementation, compute type must be `FLOAT`.

- `attributes.set_implementation(fe::AttentionImplementation_t value)`: Selects
  the attention implementation. `AUTO` chooses a supported `COMPOSITE` or
  `UNIFIED` path. The explicit choices have different feature, version, and
  architecture support surfaces.

- `attributes.set_score_mod(fn)`: Installs a custom score modifier function.
  `fn` must be callable as
  `std::shared_ptr<fe::graph::Tensor_attributes>(std::shared_ptr<fe::graph::Graph>, std::shared_ptr<fe::graph::Tensor_attributes>)`.
  The function receives a temporary score-modifier graph and the raw score
  tensor, then returns the modified score tensor. In the FP16/BF16 support
  surface, do not combine `score_mod` with the built-in ALiBi, padding, causal,
  or sliding-window masking setters.

- `attributes.set_logit_max(std::shared_ptr<Tensor_attributes> value)` and
  `attributes.set_score_sum_exp(std::shared_ptr<Tensor_attributes> value)`: Set
  optional user-provided output tensors for the softmax logit maximum and the
  numerically stable sum of exponentials. Shapes are normally `(B, H_q, S_q, 1)`
  and data type `FLOAT`. They are filled through graph execution but are not
  returned by `Graph::sdpa`.

- `attributes.set_sink_token(std::shared_ptr<Tensor_attributes> value)`: Sets an
  optional sink-attention token tensor. Shape is `(1, H_q, 1, 1)`, stride is
  commonly `{H_q, 1, 1, 1}`, and data type is `FLOAT`.

- `attributes.set_unfuse_fma(bool value)`: SM100 softmax option. When `true`,
  uses unfused `__fmul_rn` plus `__fadd_rn` instead of fused FMA in selected
  softmax paths. Backend support is version-specific.

Preferred masking setters:

- `attributes.set_diagonal_alignment(fe::DiagonalAlignment_t alignment)`: Sets
  the alignment used by diagonal band masks. `TOP_LEFT` is the standard prefill
  alignment. `BOTTOM_RIGHT` is used for bottom-right aligned decode masks and
  variable-length diagonal alignment.

- `attributes.set_diagonal_band_left_bound(int value)`: Sets an optional left
  sliding-window bound. The value is an offset to the left of the main diagonal
  that remains visible. Do not use this together with deprecated
  `set_sliding_window_length`; they configure the same left-bound masking
  behavior.

- `attributes.set_diagonal_band_right_bound(int value)`: Sets an optional right
  diagonal bound. `0` with `TOP_LEFT` alignment is the preferred causal-mask
  spelling. Do not use this together with deprecated `set_causal_mask` or
  `set_causal_mask_bottom_right`; those setters also set the right bound to `0`.

Deprecated setters:

- `attributes.set_sliding_window_length(int value)`: Deprecated. Equivalent to
  `set_diagonal_band_left_bound(value)`. Do not combine both spellings.

- `attributes.set_causal_mask(bool value)`: Deprecated. When `value` is `true`,
  equivalent to `set_diagonal_alignment(TOP_LEFT)` and
  `set_diagonal_band_right_bound(0)`. Do not combine it with
  `set_causal_mask_bottom_right` or an explicit right bound.

- `attributes.set_causal_mask_bottom_right(bool value)`: Deprecated. When
  `value` is `true`, equivalent to `set_diagonal_alignment(BOTTOM_RIGHT)` and
  `set_diagonal_band_right_bound(0)`. Do not combine it with
  `set_causal_mask` or an explicit right bound.

- `attributes.set_is_inference(bool value)`: Deprecated compatibility option.
  `set_is_inference(false)` maps to `set_generate_stats(true)`;
  `set_is_inference(true)` maps to `set_generate_stats(false)`. Prefer
  `set_generate_stats`.

## Returns

- `O` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Output tensor
  attributes for the attention result. The logical output shape is
  `(B, H_q, S_q, D_v)`, where `D_v` must be a multiple of `8` for the standard
  FP16/BF16 path. The returned tensor is virtual until it is marked as a graph
  output with `O->set_output(true)`. Set its dimensions, strides, data type when
  needed, and UID before graph validation/build.

- `Stats` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`):
  Softmax statistics tensor. Present only when
  `attributes.set_generate_stats(true)` is used; otherwise `Stats == nullptr`.
  The usual shape is `(B, H_q, S_q, 1)` and data type is `FLOAT`. These
  statistics are required by `sdpa_backward`.

## Errors

- `std::runtime_error`: `Graph::sdpa` can throw immediately when
  `attributes.set_implementation(AUTO)` cannot find a suitable implementation
  for the current `SDPA_attributes`. The thrown message is
  `"No suitable implementation for given SDPA_attributes"`.

- `cudnn_frontend::error_t`: Most C++ frontend failures are returned later by
  graph methods such as `validate`, `build_operation_graph`,
  `create_execution_plans`, `check_support`, `build_plans`, `build`,
  `get_workspace_size`, and `execute`. Check `status.is_good()` or
  `status.is_bad()`, then inspect `status.get_code()` and
  `status.get_message()`.

- `error_code_t::ATTRIBUTE_NOT_SET`: Returned for missing required tensor
  attributes or invalid option combinations such as setting both tensor and
  float attention scale, enabling padding mask without both sequence length
  tensors, setting sequence length tensors without padding mask or score
  modifier usage, combining custom and probability-based dropout, or using
  dropout probability `1.0`.

- `error_code_t::INVALID_VALUE`: Returned for value-level validation failures,
  such as a negative right diagonal bound or an invalid left bound on cuDNN
  versions that require positive left bounds.

- `error_code_t::GRAPH_NOT_SUPPORTED`: Returned when the graph is well-formed
  but unsupported by the selected implementation, cuDNN version, GPU
  architecture, data type, tensor dimensions, mask combination, paged-attention
  configuration, or feature support surface.

- Other `error_code_t` values: The graph lifecycle can also return
  `SHAPE_DEDUCTION_FAILED`, `INVALID_TENSOR_NAME`, `INVALID_VARIANT_PACK`,
  `GRAPH_EXECUTION_PLAN_CREATION_FAILED`, `HEURISTIC_QUERY_FAILED`,
  `GRAPH_EXECUTION_FAILED`, `CUDA_API_FAILED`, `CUDNN_BACKEND_API_FAILED`,
  `INVALID_CUDA_DEVICE`, `HANDLE_ERROR`, or `NVRTC_COMPILATION_FAILED`,
  depending on which frontend, backend, CUDA, or optional OSS engine stage
  fails.

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

- `sdpa` is a graph-construction API. It returns tensor attributes, not device
  pointers, device tensors, or computed data.

- `Graph::sdpa` is the standard FP16/BF16 forward path. If
  `SDPA_attributes` does not set an internal MMA core mode, `Graph::sdpa` sets it
  to `DataType_t::HALF`. This path does not support FP32 Q, K, V, or O tensors
  as a public support surface. Use `FLOAT` for compute and auxiliary tensors
  such as `Stats`, `rng_dump`, `score_max`, `score_sum_exp`, and `sink_token`.
  FP8 attention uses separate C++ entry points such as `Graph::sdpa_fp8`.

- Q, K, V, and O are normally all FP16 or all BF16 in the standard path. Their
  strides do not need to be identical, and repository examples use different
  valid layouts for inputs and outputs. The last stride of Q, K, V, and O must
  be `1`.

- GQA/MQA requires `H_q % H_k == 0` and `H_q % H_v == 0`.

- `D_qk` and `D_v` must be multiples of `8` for the standard FP16/BF16 path. The
  separate FP8 SDPA entry points have stricter dimension rules, generally
  requiring multiples of `16` plus architecture/version-specific maximums.

- For the standard FP16/BF16 `sdpa` path, maximum head dimension limits are
  architecture- and cuDNN-version-specific. On Ampere (SM80), `D_qk` and `D_v`
  must be `<= 128` for cuDNN `<= 9.9.0`. On Hopper (SM90), they must be
  `<= 256` for cuDNN `<= 9.9.0`. On Blackwell (SM100), they must be `<= 128`
  for cuDNN `< 9.9.0`. For newer cuDNN versions, this frontend check does not
  impose an additional hard maximum beyond the multiple-of-8 requirement, but
  backend plan selection may still reject unsupported shapes.

- For the separate FP8 SDPA entry points, Hopper requires `D_qk` and `D_v` to be
  `<= 256` and multiples of `16`. Blackwell generally requires `D_qk` and `D_v`
  to be `<= 128` and multiples of `16`, with a cuDNN `>= 9.19.0` exception for
  `D_qk = 192` and `D_v = 128`.

- `attn_scale` is optional. If it is not set, scores are not scaled. This is not
  equivalent to automatically using `1 / sqrt(D_qk)`.

- If `padding_mask` is enabled, both `seq_len_q` and `seq_len_kv` are required.
  Paged attention also requires both sequence length tensors.

- Ragged or THD tensors are represented by setting ragged offsets on
  `Tensor_attributes` through `set_ragged_offset`. Ragged SDPA is supported only
  with padding-mask semantics or a custom score modifier that consumes sequence
  lengths. The C++ paged decode/prefill sample stores Q ragged offsets as
  element offsets of length `B + 1`.

- `block_mask` is a coarse, tile-level score mask for block-sparse attention
  patterns. The tensor is `UINT8`; each byte packs 8 adjacent KV tiles for a
  given batch, query head, and query tile. With standard `128 x 128` SDPA tiles,
  the shape is `(B, H_q, ceil(S_q / 128), ceil(ceil(S_kv / 128) / 8))`. A set
  bit keeps the corresponding score tile, while an unset bit masks that tile by
  adding negative infinity before softmax. Padded tile entries outside the
  logical `(S_q, S_kv)` score matrix are ignored. Repository samples gate this
  path to cuDNN `>= 9.14` and Blackwell-class hardware; the composite
  implementation does not support `block_mask`.

- `score_mod` is intended for custom score transformations such as softcap or
  custom masks. In the FP16/BF16 support surface, combining `score_mod` with
  built-in ALiBi, padding, causal, or sliding-window masks is not supported.

- `implementation=AUTO` is recommended. The `UNIFIED` implementation requires
  recent cuDNN versions and does not support every feature. In the current
  support surface, unified SDPA requires cuDNN `>= 9.13.1`, does not support
  dynamic shapes, supports `block_mask` starting in cuDNN `9.14`, supports
  paged-attention inputs and sequence lengths starting in cuDNN `9.15`, and
  supports bias, seed/offset dropout inputs, sink token, `RNG_DUMP`, `Max`, and
  `Sum_exp` on newer versions. Custom dropout mask inputs are not supported by
  the unified implementation. The `COMPOSITE` implementation does not support
  `block_mask`.

- cuDNN `9.10.0` and `9.10.1` are explicitly rejected for FP16/BF16 SDPA forward
  due to known backend issues. cuDNN `9.14.0` has a known limitation for
  non-causal sliding-window attention with `S_kv > 1024`.

- Sink token requires backend support. The C++ sample requires cuDNN `9.13` or
  newer. Decode-only mode, where `S_q == 1`, is not supported with `sink_token`
  in the current support surface.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2490) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L4132).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_bf16_causal.cc)

<Code
  src="../../samples/cpp/sdpa_bf16_causal.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `sdpa` reference](/api/python/sdpa). Python keyword arguments
map to `SDPA_attributes` setters, while C++ groups those settings in the
attributes value passed to `Graph::sdpa`.
