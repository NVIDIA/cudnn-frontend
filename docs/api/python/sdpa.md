# sdpa

Build a scaled dot-product attention (SDPA) operation in a cuDNN frontend graph.

## Python synopsis

```python
o, stats = graph.sdpa(
    q,
    k,
    v,
    *,
    attn_scale=None,
    bias=None,
    block_mask=None,
    use_alibi_mask=False,
    use_padding_mask=False,
    seq_len_q=None,
    seq_len_kv=None,
    dropout=None,
    rng_dump=None,
    paged_attention_k_table=None,
    paged_attention_v_table=None,
    paged_attention_max_seq_len_kv=None,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
    generate_stats=None,
    implementation=cudnn.attention_implementation.AUTO,
    score_mod=None,
    score_max=None,
    score_sum_exp=None,
    sink_token=None,
    unfuse_fma=False,
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    diagonal_band_left_bound=None,
    diagonal_band_right_bound=None,
    sliding_window_length=None,
    use_causal_mask=False,
    use_causal_mask_bottom_right=False,
    is_inference=None,
)
```

`sdpa` computes:

```text
scores = matmul(q, transpose(k))
scores = scores * attn_scale              # only when attn_scale is provided
scores = score_mod(scores)                # optional
scores = scores + bias                    # optional
scores = scores + alibi / diagonal mask   # optional additive term
scores = scores + padding mask            # optional additive term
stats = logsumexp(scores)                 # only when generate_stats=True
probs = softmax(scores)
probs = dropout(probs)                    # optional; applied after softmax
o = matmul(probs, v)
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
`generate_stats=True`; its `logsumexp` reduction is taken over the last score
dimension, corresponding to the key/value sequence positions. Terms for disabled
options are omitted. Additive mask terms use `0` for retained scores and a large
negative value or negative infinity for masked scores. When dropout is disabled,
$\tilde{P} = P$; when enabled, dropout is applied to the softmax probabilities
before the multiplication by $V$.

The operation appends nodes to `graph` and returns graph tensor attributes. It does
not execute immediately; validate, build, and execute the graph in the usual
PyGraph workflow.

## Parameters

Core parameters:

- `q` (`cudnn.tensor`, required): Query tensor. Must be 4-D with logical shape
  `(B, H_q, S_q, D_qk)`. The last stride, corresponding to head dimension, must
  be `1`. For the FP16/BF16 `sdpa` path, `D_qk` must be a multiple of `8`.
  See Notes for architecture- and cuDNN-version-specific maximums.

- `k` (`cudnn.tensor`, required): Key tensor. Usually shape
  `(B, H_k, S_kv, D_qk)`. With paged attention, this is the key-cache container
  rather than the logical dense K tensor. Its head dimension must match `q`'s
  `D_qk`, which must be a multiple of `8` for the FP16/BF16 path. See Notes for
  architecture- and cuDNN-version-specific maximums.

- `v` (`cudnn.tensor`, required): Value tensor. Usually shape
  `(B, H_v, S_kv, D_v)`. With paged attention, this is the value-cache
  container. For the FP16/BF16 `sdpa` path, `D_v` must be a multiple of `8`.
  See Notes for architecture- and cuDNN-version-specific maximums.

- `attn_scale` (`float`, `cudnn.tensor`, or `None`, default `None`): Optional
  scale multiplied into the attention scores after `Q @ K.T`. cuDNN does not
  infer `1 / sqrt(D_qk)`; pass it explicitly when desired. If a tensor is used,
  it must be a graph tensor, commonly scalar-shaped and pass-by-value.

- `bias` (`cudnn.tensor` or `None`, default `None`): Optional additive score
  bias broadcast-compatible with `(B, H_q, S_q, S_kv)`, commonly
  `(1, H_q, S_q, S_kv)` or `(B, 1, S_q, S_kv)`. Boolean bias tensors are not
  supported. Use `bias`, rather than `block_mask`, for an arbitrary per-score
  floating-point mask: use `0` for retained scores and a large negative value
  or negative infinity for masked scores.

- `block_mask` (`cudnn.tensor` or `None`, default `None`): Optional packed block
  mask input. The common tile form is `UINT8` with shape
  `(B, H_q, ceil(S_q / 128), ceil(ceil(S_kv / 128) / 8))`. Requires an
  implementation/version that supports block masks. Do not force
  `implementation=cudnn.attention_implementation.COMPOSITE` when `block_mask` is
  provided; the composite implementation does not support `block_mask`. This is
  a binary tile-visibility mask, not an arbitrary per-score floating-point mask.
  On SM10x, unified SDPA block masks require cuDNN 9.26 or newer because older
  backends have a masked-tile initialization bug.

- `use_alibi_mask` (`bool`, default `False`): Adds ALiBi score offsets. In the
  FP16/BF16 support surface, ALiBi requires `diagonal_band_right_bound=0`.

- `use_padding_mask` (`bool`, default `False`): Enables variable sequence length
  masking. Requires both `seq_len_q` and `seq_len_kv`.

- `seq_len_q` (`cudnn.tensor` or `None`, default `None`): Actual query sequence
  lengths, normally `INT32` shape `(B, 1, 1, 1)`. Used with
  `use_padding_mask=True`, paged attention, ragged layouts, or custom score
  modifiers that need sequence lengths.

- `seq_len_kv` (`cudnn.tensor` or `None`, default `None`): Actual key/value
  sequence lengths, normally `INT32` shape `(B, 1, 1, 1)`. Used with
  `use_padding_mask=True`, paged attention, ragged layouts, or custom score
  modifiers that need sequence lengths.

- `dropout` (`tuple` or `None`, default `None`): Optional dropout. Must be
  either a 3-element tuple
  `(probability: float, seed: cudnn.tensor, offset: cudnn.tensor)` for
  internally generated Philox dropout, or a 2-element tuple
  `(mask: cudnn.tensor, scale: cudnn.tensor)` for a custom dropout mask.

- `rng_dump` (`cudnn.tensor` or `None`, default `None`): Optional output tensor
  for dumping the generated dropout RNG mask. Used only with probability-based
  dropout. Its shape should match the attention score matrix,
  `(B, H_q, S_q, S_kv)`, or `(B, H_q, S_q, paged_attention_max_seq_len_kv)` when
  paged attention uses an explicit maximum KV length. The usual contiguous
  stride is `(H_q * S_q * S_kv, S_q * S_kv, S_kv, 1)`, with `S_kv` replaced by
  `paged_attention_max_seq_len_kv` for paged attention. The data type should be
  `cudnn.data_type.FLOAT`.

- `paged_attention_k_table` (`cudnn.tensor` or `None`, default `None`): Optional
  page table for looking up blocks in the key-cache container. Page table values
  are `INT32` block indices.

- `paged_attention_v_table` (`cudnn.tensor` or `None`, default `None`): Optional
  page table for looking up blocks in the value-cache container. Page table
  values are `INT32` block indices.

- `paged_attention_max_seq_len_kv` (`int` or `None`, default `None`): Logical
  maximum KV sequence length when paged attention is active. Optional in some
  cases, but recommended because it removes inference ambiguity. Do not specify
  this parameter unless `paged_attention_k_table` or `paged_attention_v_table` is
  provided.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for
  this operation. When `NOT_SET`, the graph-level compute type is used. For the
  unified implementation, compute type must be `FLOAT`.

- `name` (`str`, default `""`): Operation name used to name generated tensors,
  for example `"<name>::O"` and `"<name>::Stats"`.

- `generate_stats` (`bool` or `None`, default `None`): Whether to create the
  softmax statistics tensor required by SDPA backward. Exactly one of
  `generate_stats` and deprecated `is_inference` must be provided. Use `True`
  for training and `False` for inference.

- `implementation` (`cudnn.attention_implementation`, default `AUTO`): Selects
  the attention implementation. `AUTO` chooses a supported `COMPOSITE` or
  `UNIFIED` path. The explicit choices have different feature, version, and
  architecture support surfaces.

- `score_mod` (callable or `None`, default `None`): Optional callback
  `score_mod(sdpa_graph, score_tensor) -> cudnn.tensor` that inserts a custom
  score-modification subgraph before softmax. In the FP16/BF16 support surface,
  do not combine `score_mod` with the built-in ALiBi, padding, causal, or
  sliding-window masking parameters.

- `score_max` (`cudnn.tensor` or `None`, default `None`): Optional user-provided
  output tensor for the softmax logit maximum. Shape is normally
  `(B, H_q, S_q, 1)`, type `FLOAT`. It is filled through the graph execution
  variant pack and is not returned by `sdpa`.

- `score_sum_exp` (`cudnn.tensor` or `None`, default `None`): Optional
  user-provided output tensor for the numerically stable sum of exponentials.
  Shape is normally `(B, H_q, S_q, 1)`, type `FLOAT`. It is filled through the
  graph execution variant pack and is not returned by `sdpa`.

- `sink_token` (`cudnn.tensor` or `None`, default `None`): Optional
  sink-attention token tensor. Shape is `(1, H_q, 1, 1)`, type `FLOAT`.

- `unfuse_fma` (`bool`, default `False`): SM100 softmax option. When `True`,
  uses unfused `__fmul_rn` plus `__fadd_rn` instead of fused FMA in selected
  softmax paths.

Preferred masking parameters:

- `diagonal_alignment` (`cudnn.diagonal_alignment`, default `TOP_LEFT`):
  Alignment used by diagonal band masks. `TOP_LEFT` is the standard prefill
  alignment. `BOTTOM_RIGHT` is used for bottom-right aligned decode masks.

- `diagonal_band_left_bound` (`int` or `None`, default `None`): Optional left
  sliding-window bound. Values are offsets to the left of the main diagonal that
  remain visible. `None` means no left bound. Do not specify this together with
  the deprecated `sliding_window_length`; they configure the same left-bound
  masking behavior.

- `diagonal_band_right_bound` (`int` or `None`, default `None`): Optional right
  diagonal bound. `0` with `TOP_LEFT` alignment is the preferred causal-mask
  spelling. `None` means no right bound. Do not specify this together with the
  deprecated `use_causal_mask` or `use_causal_mask_bottom_right`; those
  parameters also set the right bound to `0`.

Deprecated parameters:

- `sliding_window_length` (`int` or `None`, default `None`): Deprecated.
  Equivalent to `diagonal_band_left_bound`. Cannot be combined with an explicit
  left bound.

- `use_causal_mask` (`bool`, default `False`): Deprecated. Equivalent to
  `diagonal_alignment=TOP_LEFT` and `diagonal_band_right_bound=0`. Cannot be
  combined with `use_causal_mask_bottom_right` or an explicit right bound.

- `use_causal_mask_bottom_right` (`bool`, default `False`): Deprecated.
  Equivalent to `diagonal_alignment=BOTTOM_RIGHT` and
  `diagonal_band_right_bound=0`. Cannot be combined with `use_causal_mask` or an
  explicit right bound.

- `is_inference` (`bool` or `None`, default `None`): Deprecated compatibility
  option. `is_inference=False` maps to `generate_stats=True`;
  `is_inference=True` maps to `generate_stats=False`. Prefer `generate_stats`.
  Exactly one of `is_inference` and `generate_stats` must be specified.

## Returns

- `o` (`cudnn.tensor`): Output tensor attributes for the attention result. The
  logical output shape is `(B, H_q, S_q, D_v)`, where `D_v` must be a multiple
  of `8` for the FP16/BF16 `sdpa` path. See Notes for architecture- and
  cuDNN-version-specific maximums. The returned tensor is virtual until it is
  marked as a graph output with `o.set_output(True)`. Set its dimensions,
  strides, data type, and UID as needed before graph validation/build.

- `stats` (`cudnn.tensor` or `None`): Softmax statistics tensor. Present only
  when `generate_stats=True`; otherwise `None`. The usual shape is
  `(B, H_q, S_q, 1)`, type `FLOAT`. These statistics are required by
  `sdpa_backward`.

## Raises

- `TypeError`: Raised by the Python binding when an argument cannot be converted
  to the required bound C++ type, for example when `q`, `k`, or `v` is not a
  `cudnn.tensor`, or an enum argument is not a valid cuDNN enum value.

- `RuntimeError`: Raised immediately by `sdpa()` for invalid Python keyword
  combinations or keyword value types checked in the binding. Common cases
  include omitting `generate_stats`, specifying both `generate_stats` and the
  deprecated `is_inference`, passing a non-`bool` value for either of those
  fields, passing an invalid `attn_scale`, passing non-integer diagonal bounds
  or `paged_attention_max_seq_len_kv`, combining deprecated causal-mask
  arguments with explicit `diagonal_band_right_bound`, combining
  `use_causal_mask` with `use_causal_mask_bottom_right`, combining
  `sliding_window_length` with `diagonal_band_left_bound`, or passing a malformed
  `dropout` tuple.

- `ValueError`: Raised later by graph methods such as `graph.validate()`,
  `graph.build()`, or `graph.get_workspace_size()` when the frontend reports an
  attribute or shape error. Examples include missing required tensor attributes,
  invalid tensor names, shape-deduction failures, or invalid variant-pack
  metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised later by validation, heuristic
  query, support checking, or plan creation when the graph is well-formed but not
  supported by the selected cuDNN backend, implementation, architecture, or
  version.

- `RuntimeError`: Raised later by graph execution or backend/CUDA calls for
  execution failures, backend API failures, CUDA API failures, invalid CUDA
  device errors, handle errors, or other runtime failures.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: SDPA and RoPE**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
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
  tensors.
- Exactly one of `generate_stats` and `is_inference` must be supplied. New code
  should use `generate_stats`.
- The Python `sdpa` binding described here supports FP16/BF16 Q, K, V, and O
  tensors. It uses FP32 for compute when `compute_data_type` or the graph compute
  type is `cudnn.data_type.FLOAT`, and auxiliary tensors such as `stats`,
  `rng_dump`, `score_max`, `score_sum_exp`, and `sink_token` are typically
  `FLOAT`. It does not support FP32 Q, K, V, or O tensors in this PyGraph SDPA
  path. FP8 has separate PyGraph entry points (`sdpa_fp8`, `sdpa_mxfp8`) even
  though the C++ internals share `SDPA_attributes`.
- GQA/MQA requires `H_q % H_k == 0` and `H_q % H_v == 0`.
- `D_qk` and `D_v` must be multiples of 8 for the FP16/BF16 path. The separate
  FP8 SDPA entry points have stricter dimension rules, generally requiring
  multiples of 16 plus architecture/version-specific maximums.
- For the FP16/BF16 `sdpa` path, maximum head dimension limits are
  architecture- and cuDNN-version-specific. On Ampere (SM80), `D_qk` and `D_v`
  must be `<= 128` for cuDNN `<= 9.9.0`. On Hopper (SM90), they must be
  `<= 256` for cuDNN `<= 9.9.0`. On Blackwell (SM100), they must be `<= 128`
  for cuDNN `< 9.9.0`. For newer cuDNN versions, this frontend check does not
  impose an additional hard maximum beyond the multiple-of-8 requirement, but
  backend plan selection may still reject unsupported shapes.
- For the separate FP8 SDPA entry points, Hopper requires `D_qk` and `D_v` to be
  `<= 256` and multiples of `16`. Blackwell generally requires `D_qk` and `D_v`
  to be `<= 128` and multiples of `16`, with a cuDNN `>= 9.19.0` exception for
  `D_qk=192` and `D_v=128`.
- The last stride of `q`, `k`, `v`, and `o` must be `1`.
- `attn_scale=None` means no score scaling. It does not mean
  `1 / sqrt(D_qk)`.
- If `use_padding_mask=True`, both `seq_len_q` and `seq_len_kv` are required.
  Paged attention also requires these sequence length tensors.
- Ragged/THD tensors are represented by setting ragged offsets on the tensor
  attributes. Ragged SDPA is supported only with padding-mask semantics.
- `block_mask` is a coarse, tile-level score mask for block-sparse attention
  patterns. The tensor is `UINT8`; each byte packs 8 adjacent KV tiles for a
  given batch, query head, and query tile. With the standard `128 x 128` SDPA
  tiles, the shape is `(B, H_q, ceil(S_q / 128), ceil(ceil(S_kv / 128) / 8))`.
  A set bit keeps the corresponding score tile, while an unset bit masks that
  tile by adding negative infinity before softmax. Padded tile entries outside
  the logical `(S_q, S_kv)` score matrix are ignored. `block_mask` requires the
  unified SDPA implementation support surface; the composite implementation does
  not support it. Use `bias` for an arbitrary per-score floating-point mask.
  The public cuDNN backend documentation for
  [`CUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC`](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html)
  describes an `INT8` value per KV tile, but the current frontend and its tests
  use the packed `UINT8` representation above. The per-tile `INT8` form did not
  produce a valid frontend engine configuration in repository testing. On
  SM10x, execute the packed form only with cuDNN 9.26 or newer.
- `score_mod` is intended for custom score transformations such as softcap or
  custom masks. In the FP16/BF16 support surface, combining `score_mod` with
  built-in ALiBi, padding, causal, or sliding-window masks is not supported.
  A named pass-by-value tensor created through the high-level `cudnn.Graph`
  wrapper with `graph.tensor(...)` cannot currently be resolved by
  `set_io_tuples()`. Use direct `pygraph` execution when a custom score modifier
  requires such a tensor.
- `implementation=AUTO` is recommended. The `UNIFIED` implementation requires
  recent cuDNN versions and does not support every feature. The `COMPOSITE`
  implementation does not support `block_mask`.
- cuDNN 9.10.0 and 9.10.1 are explicitly rejected for FP16/BF16 SDPA forward due
  to known issues in the backend.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1188) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L245).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/sdpa_bf16_causal.py)

<Code
  src="../../samples/python/sdpa_bf16_causal.py"
  language="python"
  maxLines={0}
/>

Packed block-mask sample: [C++ source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/sdpa_block_mask.cc)

## Appendix: C++ equivalent

See the [C++ `Graph::sdpa` reference](/api/cpp/sdpa) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
