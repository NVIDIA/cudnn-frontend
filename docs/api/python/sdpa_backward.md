# sdpa_backward

Build the backward pass for scaled dot-product attention in a cuDNN frontend
PyGraph.

## Python synopsis

```python
dQ, dK, dV = graph.sdpa_backward(
    q,
    k,
    v,
    o,
    dO,
    stats,
    *,
    attn_scale=None,
    bias=None,
    dbias=None,
    use_padding_mask=False,
    seq_len_q=None,
    seq_len_kv=None,
    dropout=None,
    rng_dump=None,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
    implementation=cudnn.attention_implementation.AUTO,
    score_mod=None,
    score_mod_bprop=None,
    sink_token=None,
    dSink_token=None,
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    diagonal_band_left_bound=None,
    diagonal_band_right_bound=None,
    use_causal_mask=False,
    use_causal_mask_bottom_right=False,
    sliding_window_length=None,
    use_deterministic_algorithm=False,
)
```

`sdpa_backward` differentiates the same attention computation built by
`sdpa`. The forward graph must have generated `stats` and the backward graph
must use matching score modifiers, masks, dropout configuration, attention
scale, and sequence lengths.

```text
scores = matmul(q, transpose(k))
scores = scores * attn_scale              # only when provided
scores = score_mod(scores)                # optional forward score modifier function
scores = scores + bias + masks            # optional additive terms
probs = softmax(scores, stats=stats)
probs = dropout(probs)                    # optional; same position as forward
o = matmul(probs, v)

dQ, dK, dV = backward(o, dO, stats, q, k, v)
dbias = optional score-bias gradient
dSink_token = optional sink-token gradient
```

In equation form, the backward call differentiates $O = \tilde{P}V$, where
$\tilde{P} = \operatorname{dropout}(\operatorname{softmax}(\hat{S}))$ and
$\hat{S}$ is the masked, biased, optionally modified score tensor documented by
`sdpa`.

$$
(dQ, dK, dV) =
\nabla_{Q,K,V}\; O\left(Q,K,V,\operatorname{Stats},dO\right)
$$

The optional `score_mod_bprop` callback supplies the derivative of the forward
score modifier function. Optional `dbias` and `dSink_token` tensors receive
additional gradients but are not part of the returned tuple.

## Parameters

- `q`, `k`, `v` (`cudnn.tensor`, required): Forward query, key, and value
  tensors. Shapes, dtypes, head grouping, head-dimension divisibility, stride
  constraints, and architecture limits must match the forward `sdpa` graph.

- `o` (`cudnn.tensor`, required): Forward SDPA output tensor. It must have the
  same logical shape and dtype as the forward output used to compute `dO`.

- `dO` (`cudnn.tensor`, required): Upstream gradient for `o`. Its shape must
  match `o`. It may use its own supported stride.

- `stats` (`cudnn.tensor`, required): Forward softmax statistics returned by
  `sdpa` when `generate_stats=True`. Usually shape `(B, H_q, S_q, 1)` and dtype
  `FLOAT`. The values must correspond to the same forward options used by this
  backward graph.

- `attn_scale` (`float`, `cudnn.tensor`, or `None`, default `None`): Same
  attention scale used by the forward graph.

- `bias` (`cudnn.tensor` or `None`, default `None`): Same forward additive bias
  tensor when the forward graph used bias.

- `dbias` (`cudnn.tensor` or `None`, default `None`): Optional output tensor for
  additive bias gradients. Keep and map this tensor in the variant pack because
  it is not returned by `sdpa_backward`.

- `use_padding_mask` (`bool`, default `False`): Enables the same variable-length
  padding mask used by forward. Requires both `seq_len_q` and `seq_len_kv`.

- `seq_len_q`, `seq_len_kv` (`cudnn.tensor` or `None`, default `None`): Actual
  query and key/value lengths, normally `INT32` tensors of shape `(B, 1, 1, 1)`.

- `dropout` (`tuple` or `None`, default `None`): Dropout configuration matching
  the forward graph. It must be either `(probability, seed, offset)` or
  `(mask, scale)`. Dropout is applied to softmax probabilities before the
  multiplication by `V`; backward must use the same mask or Philox seed/offset
  stream as forward.

- `rng_dump` (`cudnn.tensor` or `None`, default `None`): Optional RNG-mask dump
  tensor used with probability-based dropout for debugging. Shape, stride, and
  dtype follow the forward `sdpa` `rng_dump` contract.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type.
  Use the same type as the forward graph unless deliberately targeting a
  supported mixed path.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

- `implementation` (`cudnn.attention_implementation`, default `AUTO`): Selects
  the attention implementation. `AUTO` chooses between the supported
  `COMPOSITE` and `UNIFIED` paths. Use the same effective implementation as
  forward and verify that it supports every requested feature.

- `score_mod` (callable or `None`, default `None`): Forward score modifier
  function used to rebuild the score path in the backward graph.

- `score_mod_bprop` (callable or `None`, default `None`): Backward callback for
  the score modifier function. Provide it when `score_mod` is not a simple
  operation that the backend can differentiate through automatically.

- `sink_token` (`cudnn.tensor` or `None`, default `None`): Forward sink token
  tensor when the forward graph used sink attention.

- `dSink_token` (`cudnn.tensor` or `None`, default `None`): Optional output
  tensor for sink-token gradients. It is not returned by `sdpa_backward`.

- `diagonal_alignment` (`cudnn.diagonal_alignment`, default `TOP_LEFT`): Same
  diagonal-mask alignment used by forward.

- `diagonal_band_left_bound`, `diagonal_band_right_bound` (`int` or `None`,
  default `None`): Preferred diagonal-band mask bounds. Do not specify
  `diagonal_band_left_bound` with deprecated `sliding_window_length`; both
  configure the left sliding-window behavior. Do not combine
  `use_causal_mask_bottom_right=True` with an inconsistent
  `diagonal_alignment`.

- `use_deterministic_algorithm` (`bool`, default `False`): Requests a
  deterministic backward algorithm where the backend supports one.

Deprecated masking parameters:

- `use_causal_mask` (`bool`, default `False`): Deprecated causal-mask shortcut.
  Prefer `diagonal_band_right_bound=0` with the appropriate
  `diagonal_alignment`.

- `use_causal_mask_bottom_right` (`bool`, default `False`): Deprecated
  bottom-right causal-mask shortcut. Prefer
  `diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT` with
  `diagonal_band_right_bound=0`.

- `sliding_window_length` (`int` or `None`, default `None`): Deprecated sliding
  window shortcut. Prefer `diagonal_band_left_bound`.

## Returns

- `dQ` (`cudnn.tensor`): Gradient tensor for `q`. Shape follows `q`.

- `dK` (`cudnn.tensor`): Gradient tensor for `k`. Shape follows `k`.

- `dV` (`cudnn.tensor`): Gradient tensor for `v`. Shape follows `v`.

Set output dtype, stride, and `set_output(True)` on any gradient tensor that
must be written to a user buffer.

## Raises

- `TypeError`: Raised immediately for invalid tensor arguments, malformed
  dropout tuples, invalid callbacks, or enum conversion failures.

- `RuntimeError`: Raised immediately for mutually exclusive or inconsistent
  mask/dropout keyword combinations detected by the binding.

- `ValueError`: Raised by `graph.validate()` when forward/backward tensor
  metadata, saved statistics, masks, or optional gradient outputs are
  inconsistent.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan build
  when the selected backend cannot support the requested backward feature set.

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

Backward feature support is usually narrower than forward support. Treat any
custom score modifier, dropout, sink token, ragged layout, paged layout, or
determinism request as a feature that must be verified by `check_support()`.

The deprecated `scaled_dot_product_flash_attention_backward` Python API maps to
`sdpa_backward`. New documentation and examples should use `sdpa_backward`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1276) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L383).
{/* END GENERATED API SOURCES */}

## Examples

The causal GQA training sample executes `sdpa` and `sdpa_backward`, then checks
the output and all three input gradients against PyTorch:

<Code
  src="../../samples/python/sdpa_bf16_causal.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::sdpa_backward` reference](/api/cpp/sdpa-backward) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
