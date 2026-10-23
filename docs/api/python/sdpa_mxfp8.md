# sdpa_mxfp8

Build an MXFP8 scaled dot-product attention operation in a cuDNN frontend
PyGraph.

## Python synopsis

```python
o, stats, amax_o = graph.sdpa_mxfp8(
    q,
    k,
    v,
    descale_q,
    descale_k,
    descale_v,
    *,
    attn_scale=None,
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    diagonal_band_left_bound=None,
    diagonal_band_right_bound=None,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
    generate_stats,
    sink_token=None,
    unfuse_fma=False,
    use_causal_mask=False,
    use_causal_mask_bottom_right=False,
)
```

`sdpa_mxfp8` computes SDPA with FP8 data tensors and block-wise MXFP8 scale
factors:

```text
q_d = block_dequantize(q, descale_q)
k_d = block_dequantize(k, descale_k)
v_d = block_dequantize(v, descale_v)
scores = matmul(q_d, transpose(k_d))
scores = scores * attn_scale              # only when attn_scale is provided
scores = scores + diagonal / causal mask  # optional additive term
stats = logsumexp(scores)                 # only when generate_stats=True
probs = softmax(scores)
o_unscaled = matmul(probs, v_d)
amax_o = max(abs(o_unscaled))
o = output_cast(o_unscaled)
```

Equivalently, with block-dequantized inputs $\bar{Q}$, $\bar{K}$, and
$\bar{V}$, optional additive diagonal mask term $M_d$, and output cast function
$C$:

$$
\begin{aligned}
\bar{Q} &= \operatorname{block\_dequant}(Q, S_Q), \\
\bar{K} &= \operatorname{block\_dequant}(K, S_K), \\
\bar{V} &= \operatorname{block\_dequant}(V, S_V), \\
S &= \bar{Q}\bar{K}^{\top}, \\
\hat{S} &= \alpha S + M_d, \\
\operatorname{Stats} &= \operatorname{logsumexp}\left(\hat{S}\right), \\
P &= \operatorname{softmax}\left(\hat{S}\right), \\
\bar{O} &= P\bar{V}, \\
\operatorname{Amax}_O &= \max\left|\bar{O}\right|, \\
O &= C\left(\bar{O}\right).
\end{aligned}
$$

Here $S_Q$, $S_K$, and $S_V$ correspond to `descale_q`, `descale_k`, and
`descale_v`. The factor $\alpha$ is `attn_scale` when provided; otherwise the
multiplication by $\alpha$ is omitted. Terms for disabled masks are omitted.
`sdpa_mxfp8` does not expose the regular FP8 score modifier function, dropout,
bias, padding mask, or paged-attention parameters.

## Parameters

- `q` (`cudnn.tensor`, required): Query tensor with dtype
  `cudnn.data_type.FP8_E4M3`. Shape is `(B, H, S_q, D)`. The head-dimension
  stride should be `1` for the supported layouts.

- `k` (`cudnn.tensor`, required): Key tensor with dtype
  `cudnn.data_type.FP8_E4M3`. Shape is `(B, H, S_kv, D)`.

- `v` (`cudnn.tensor`, required): Value tensor with dtype
  `cudnn.data_type.FP8_E4M3`. Shape is `(B, H, S_kv, D)` in the documented
  MXFP8 forward signature.

- `descale_q` (`cudnn.tensor`, required): MXFP8 block scale-factor tensor for
  Q. Must use dtype `cudnn.data_type.FP8_E8M0` and
  `reordering_type=cudnn.tensor_reordering.F8_128x4`. Shape is
  `(B, H, S_q_padded, D_scale_padded)`, where `D_scale = ceil(D / 32)` and the
  scale dimension is padded to a multiple of `4`. The last stride must be `1`.

- `descale_k` (`cudnn.tensor`, required): MXFP8 block scale-factor tensor for
  K. Same dtype, reordering, shape pattern, and last-stride requirement as
  `descale_q`; `S_kv_padded` is padded to a multiple of `128`.

- `descale_v` (`cudnn.tensor`, required): MXFP8 block scale-factor tensor for
  V. Must use dtype `FP8_E8M0` and reordering `F8_128x4`. Shape is
  `(B, H, S_scale_padded, D_padded)`, where `S_scale = ceil(S_kv / 32)` and the
  scale dimension is padded to a multiple of `4`. The sequence-scale stride
  `stride[2]` must be `1`.

- `attn_scale` (`float`, `cudnn.tensor`, or `None`, default `None`): Optional
  score scale applied after `Q @ K.T`.

- `diagonal_alignment` (`cudnn.diagonal_alignment`, default `TOP_LEFT`):
  Alignment used by diagonal band masks.

- `diagonal_band_left_bound` (`int` or `None`, default `None`): Optional left
  diagonal-band bound. Do not combine it with deprecated sliding-window
  spellings; `sdpa_mxfp8` does not expose a non-deprecated `sliding_window`
  parameter.

- `diagonal_band_right_bound` (`int` or `None`, default `None`): Optional right
  diagonal-band bound. Prefer `diagonal_band_right_bound=0` over deprecated
  causal-mask shortcuts.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type.
  Examples construct the graph with `compute_data_type=cudnn.data_type.FLOAT`.

- `name` (`str`, default `""`): Operation name.

- `generate_stats` (`bool`, required): Whether to create the softmax
  statistics tensor. Use `True` when a backward graph will consume `stats`.

- `sink_token` (`cudnn.tensor` or `None`, default `None`): Optional sink token
  bias. Shape is `(1, H, 1, 1)` and dtype is `FLOAT`.

- `unfuse_fma` (`bool`, default `False`): SM100 softmax option that requests
  unfused multiply and add instructions instead of fused FMA where supported.

Deprecated masking parameters:

- `use_causal_mask` (`bool`, default `False`): Deprecated causal-mask shortcut.
  Prefer `diagonal_band_right_bound=0`.

- `use_causal_mask_bottom_right` (`bool`, default `False`): Deprecated
  bottom-right causal shortcut. Prefer
  `diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT` with
  `diagonal_band_right_bound=0`.

## Returns

- `o` (`cudnn.tensor`): Attention output tensor. The output dtype is configured
  on the returned tensor; repository examples set it to `BFLOAT16`.

- `stats` (`cudnn.tensor` or `None`): Softmax statistics when
  `generate_stats=True`; otherwise `None`.

- `amax_o` (`cudnn.tensor`): Absolute maximum of the output tensor, normally
  shape `(1, 1, 1, 1)`, stride `(1, 1, 1, 1)`, and dtype `FLOAT`.

## Raises

- `TypeError`: Raised immediately for invalid tensor arguments, missing required
  `generate_stats`, invalid enum values, or malformed scalar values.

- `RuntimeError`: Raised immediately for conflicting deprecated and preferred
  mask options detected by the binding.

- `ValueError`: Raised by `graph.validate()` for malformed MXFP8 scale-factor
  metadata, wrong reordering, wrong dtype, invalid output metadata, or
  inconsistent diagonal-mask options.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or
  `graph.build_plans()` when MXFP8 SDPA is unsupported by the installed cuDNN
  version, GPU architecture, dtype, stride, or feature combination.

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

The live docstring states that MXFP8 SDPA requires cuDNN 9.21.0 or later and a
Blackwell GPU architecture or newer. Repository C++ samples enforce the same
support gate with `cudnnGetVersion() >= 92100` and a Blackwell-or-newer device
check.

MXFP8 is not regular FP8 with different scalar values. Regular FP8 uses scalar
`FLOAT` scale/descale tensors. MXFP8 uses block-wise `FP8_E8M0` scale factors
with `F8_128x4` reordering and different scale dimensions for Q/K versus V.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1450) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L678).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API in the current checkout.

## Appendix: C++ equivalent

See the [C++ `Graph::sdpa_mxfp8` reference](/api/cpp/sdpa-mxfp8) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
