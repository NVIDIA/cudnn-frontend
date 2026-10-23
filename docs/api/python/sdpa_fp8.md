# sdpa_fp8

Build a regular FP8 scaled dot-product attention operation in a cuDNN frontend
PyGraph.

## Python synopsis

```python
o, stats, amax_s, amax_o = graph.sdpa_fp8(
    q,
    k,
    v,
    descale_q,
    descale_k,
    descale_v,
    descale_s,
    scale_s,
    scale_o,
    *,
    attn_scale=None,
    bias=None,
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
    fn=None,
    generate_stats=None,
    score_max=None,
    score_sum_exp=None,
    sink_token=None,
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=None,
    right_bound=None,
    sliding_window=None,
    use_causal_mask=False,
    use_causal_mask_bottom_right=False,
    is_inference=None,
)
```

`sdpa_fp8` computes the SDPA forward pass with FP8 Q, K, V, and output scaling:

```text
q_d = descale_q * q
k_d = descale_k * k
v_d = descale_v * v
scores = matmul(q_d, transpose(k_d))
scores = scores * attn_scale              # only when attn_scale is provided
scores = fn(scores)                       # optional score modifier function
scores = scores + bias                    # optional
scores = scores + alibi / diagonal mask   # optional additive term
scores = scores + padding mask            # optional additive term
stats = logsumexp(scores)                 # only when generate_stats=True
probs = softmax(scores)
probs = dropout(probs)                    # optional; applied after softmax
o_unscaled = matmul(probs, v_d)
amax_s = max(abs(scores))
amax_o = max(abs(o_unscaled))
o = quantize_fp8(o_unscaled, scale_o)
```

Equivalently, with dequantized inputs $\bar{Q}$, $\bar{K}$, and $\bar{V}$,
optional score modifier function $f$, additive bias $B$, optional additive
mask terms $M_d$ and $M_p$, and quantization function $Q_{\mathrm{fp8}}$:

$$
\begin{aligned}
\bar{Q} &= d_q Q,\quad \bar{K} = d_k K,\quad \bar{V} = d_v V, \\
S &= \bar{Q}\bar{K}^{\top}, \\
\hat{S} &= f\left(\alpha S\right) + B + M_d + M_p, \\
\operatorname{Stats} &= \operatorname{logsumexp}\left(\hat{S}\right), \\
P &= \operatorname{softmax}\left(\hat{S}\right), \\
\tilde{P} &= \operatorname{dropout}\left(P\right), \\
\bar{O} &= \tilde{P}\bar{V}, \\
\operatorname{Amax}_S &= \max\left|\hat{S}\right|,\quad
\operatorname{Amax}_O = \max\left|\bar{O}\right|, \\
O &= Q_{\mathrm{fp8}}\left(s_o \bar{O}\right).
\end{aligned}
$$

Here $d_q$, $d_k$, $d_v$, and $s_o$ correspond to `descale_q`,
`descale_k`, `descale_v`, and `scale_o`. The `scale_s` and `descale_s`
tensors control the backend's FP8 score-intermediate scaling; the equation shows
the logical score tensor $\hat{S}$ and its reported $\operatorname{Amax}_S$.
Terms for disabled options are omitted. When dropout is disabled,
$\tilde{P} = P$.

## Parameters

- `q` (`cudnn.tensor`, required): FP8 query tensor. Usually shape
  `(B, H_q, S_q, D_qk)` with head-dimension stride `1`. Repository FP8 samples
  use `cudnn.data_type.FP8_E4M3` through graph IO dtype or explicit tensor dtype.

- `k` (`cudnn.tensor`, required): FP8 key tensor. Usually shape
  `(B, H_k, S_kv, D_qk)`. With paged attention, this is the key-cache container
  and `paged_attention_k_table` supplies logical page lookup.

- `v` (`cudnn.tensor`, required): FP8 value tensor. Usually shape
  `(B, H_v, S_kv, D_v)`. With paged attention, this is the value-cache
  container and `paged_attention_v_table` supplies logical page lookup.

- `descale_q`, `descale_k`, `descale_v` (`cudnn.tensor`, required): Scalar-like
  `FLOAT` descale tensors for Q, K, and V. Repository examples use shape
  `(1, 1, 1, 1)` and stride `(1, 1, 1, 1)`.

- `descale_s` (`cudnn.tensor`, required): Scalar-like `FLOAT` descale tensor for
  the FP8 score intermediate.

- `scale_s` (`cudnn.tensor`, required): Scalar-like `FLOAT` scale tensor for the
  FP8 score intermediate.

- `scale_o` (`cudnn.tensor`, required): Scalar-like `FLOAT` scale tensor for
  output quantization.

- `attn_scale` (`float`, `cudnn.tensor`, or `None`, default `None`): Optional
  scale multiplied into the attention scores after `Q @ K.T`. Pass
  `1 / sqrt(D_qk)` explicitly when that is desired.

- `bias` (`cudnn.tensor` or `None`, default `None`): Optional additive score
  bias broadcast-compatible with `(B, H_q, S_q, S_kv)`.

- `use_alibi_mask` (`bool`, default `False`): Enables ALiBi score offsets where
  supported.

- `use_padding_mask` (`bool`, default `False`): Enables variable-length masking.
  Requires both `seq_len_q` and `seq_len_kv`.

- `seq_len_q`, `seq_len_kv` (`cudnn.tensor` or `None`, default `None`): Actual
  query and key/value lengths, normally `INT32` tensors with shape
  `(B, 1, 1, 1)`.

- `dropout` (`tuple` or `None`, default `None`): Optional dropout. Must be
  either `(probability: float, seed: cudnn.tensor, offset: cudnn.tensor)` or
  `(mask: cudnn.tensor, scale: cudnn.tensor)`. Dropout is applied after softmax
  and before multiplication by V.

- `rng_dump` (`cudnn.tensor` or `None`, default `None`): Optional RNG-mask dump
  for probability-based dropout. It should have attention-score shape
  `(B, H_q, S_q, S_kv)` or `(B, H_q, S_q, paged_attention_max_seq_len_kv)` for
  paged attention, contiguous score stride ending in `1`, and `FLOAT` dtype.

- `paged_attention_k_table`, `paged_attention_v_table` (`cudnn.tensor` or
  `None`, default `None`): Optional `INT32` page tables. Entries are block
  indices into the key or value cache container.

- `paged_attention_max_seq_len_kv` (`int` or `None`, default `None`): Logical
  maximum KV sequence length for paged attention. Do not specify it unless at
  least one page table is provided.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type. FP8
  samples use `FLOAT`.

- `name` (`str`, default `""`): Operation name.

- `fn` (callable or `None`, default `None`): Optional score modifier function
  `fn(sdpa_graph, score_tensor) -> cudnn.tensor`.

- `generate_stats` (`bool` or `None`, default `None`): Whether to return softmax
  statistics for backward. Exactly one of `generate_stats` and deprecated
  `is_inference` must be specified.

- `score_max`, `score_sum_exp` (`cudnn.tensor` or `None`, default `None`):
  Optional user-provided score-statistic output tensors. They are filled through
  the variant pack and are not part of the returned tuple.

- `sink_token` (`cudnn.tensor` or `None`, default `None`): Optional sink token
  bias. Shape is `(1, H_q, 1, 1)` and dtype is `FLOAT`.

- `diagonal_alignment` (`cudnn.diagonal_alignment`, default `TOP_LEFT`):
  Alignment for diagonal band masks.

- `left_bound`, `right_bound` (`int` or `None`, default `None`): Preferred
  diagonal-band bounds. Do not combine `left_bound` with deprecated
  `sliding_window`; do not combine `right_bound` with deprecated causal mask
  shortcuts.

Deprecated parameters:

- `sliding_window` (`int` or `None`, default `None`): Deprecated left-window
  shortcut. Prefer `left_bound`.

- `use_causal_mask` (`bool`, default `False`): Deprecated causal shortcut.
  Prefer `right_bound=0` with the desired `diagonal_alignment`.

- `use_causal_mask_bottom_right` (`bool`, default `False`): Deprecated
  bottom-right causal shortcut. Prefer `diagonal_alignment=BOTTOM_RIGHT` with
  `right_bound=0`.

- `is_inference` (`bool` or `None`, default `None`): Deprecated compatibility
  spelling. `generate_stats` is the negation of `is_inference`; prefer
  `generate_stats`.

## Returns

- `o` (`cudnn.tensor`): Attention output tensor. Set output shape, stride, and
  dtype explicitly before validation when inference cannot derive them. Regular
  FP8 examples write FP8 output.

- `stats` (`cudnn.tensor` or `None`): Softmax statistics tensor when
  `generate_stats=True`; otherwise `None`. Required by `sdpa_fp8_backward`.

- `amax_s` (`cudnn.tensor`): Absolute maximum of the score tensor. Examples use
  shape `(1, 1, 1, 1)`, stride `(1, 1, 1, 1)`, and `FLOAT` dtype.

- `amax_o` (`cudnn.tensor`): Absolute maximum of the unscaled output tensor.
  Examples use shape `(1, 1, 1, 1)`, stride `(1, 1, 1, 1)`, and `FLOAT` dtype.

## Raises

- `TypeError`: Raised immediately for invalid tensor arguments, enum values,
  score modifier function, scalar values, or malformed Python types.

- `RuntimeError`: Raised immediately for conflicting deprecated and preferred
  parameters, malformed dropout tuples, or inconsistent page-table options
  detected by the binding.

- `ValueError`: Raised by `graph.validate()` for malformed tensor metadata,
  missing `generate_stats`/`is_inference` choice, invalid output metadata, or
  inconsistent mask and sequence-length metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or
  `graph.build_plans()` when the feature combination, cuDNN version, GPU
  architecture, dtype, stride, or implementation is unsupported.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: FP8, MXFP8, NVFP4, and block scaling**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
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

Regular FP8 is distinct from MXFP8. `sdpa_fp8` uses scalar `FLOAT` scale and
descale tensors. `sdpa_mxfp8` uses block-wise `FP8_E8M0` scale-factor tensors
with `F8_128x4` reordering.

The C++ FP8 forward sample validates support only for cuDNN 9.1.0 or later on
Hopper or newer GPUs. Treat that as the documented support gate unless local
support checks on your installed cuDNN build prove a narrower or broader set.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1350) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L539).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API in the current checkout.

## Appendix: C++ equivalent

See the [C++ `Graph::sdpa_fp8` reference](/api/cpp/sdpa-fp8) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
