# sdpa_fp8_backward

Append a regular FP8 scaled dot-product attention backward operation to a PyGraph.

## Python synopsis

```python
dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP = graph.sdpa_fp8_backward(
    q, k, v, o, dO, stats,
    descale_q, descale_k, descale_v, descale_o, descale_dO,
    descale_s, descale_dP,
    scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
    attn_scale=None,
    use_padding_mask=False,
    seq_len_q=None,
    seq_len_kv=None,
    use_causal_mask=False,
    use_causal_mask_bottom_right=False,
    diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=None,
    right_bound=None,
    use_deterministic_algorithm=False,
    dropout=None,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
    sink_token=None,
    dSink_token=None,
)
```

## Parameters

- `q`, `k`, `v` (`cudnn.tensor`, required): Forward FP8 query, key, and value tensors.

- `o` (`cudnn.tensor`, required): Forward FP8 attention output.

- `dO` (`cudnn.tensor`, required): Upstream FP8 output gradient.

- `stats` (`cudnn.tensor`, required): Softmax statistics returned by `sdpa_fp8(..., generate_stats=True)`.

- `descale_q`, `descale_k`, `descale_v`, `descale_o`, `descale_dO`, `descale_s`, `descale_dP` (`cudnn.tensor`, required): Scalar-like descale tensors for Q, K, V, O, dO, score, and dP paths. Repository examples use `FLOAT` tensors with shape `(1, 1, 1, 1)`.

- `scale_s`, `scale_dQ`, `scale_dK`, `scale_dV`, `scale_dP` (`cudnn.tensor`, required): Scalar-like scale tensors for score and gradient quantization paths.

- `attn_scale` (`float`, `cudnn.tensor`, or `None`, default `None`): Optional score scale. Use the same value as the matching forward call.

- `use_padding_mask` (`bool`, default `False`): Enables variable-length masking. Requires `seq_len_q` and `seq_len_kv` when used.

- `seq_len_q`, `seq_len_kv` (`cudnn.tensor` or `None`, default `None`): Actual query and key/value lengths, normally `INT32` tensors.

- `use_causal_mask` (`bool`, default `False`): Deprecated causal shortcut. Prefer `right_bound=0` with the desired `diagonal_alignment`.

- `use_causal_mask_bottom_right` (`bool`, default `False`): Deprecated bottom-right causal shortcut. Prefer `diagonal_alignment=BOTTOM_RIGHT` with `right_bound=0`.

- `diagonal_alignment` (`cudnn.diagonal_alignment`, default `TOP_LEFT`): Alignment for diagonal-band masking.

- `left_bound`, `right_bound` (`int` or `None`, default `None`): Diagonal-band bounds. `right_bound` conflicts with deprecated causal-mask shortcuts.

- `use_deterministic_algorithm` (`bool`, default `False`): Requests deterministic algorithm selection where supported.

- `dropout` (`tuple` or `None`, default `None`): Optional dropout metadata. The backward binding accepts a 3-element tuple: `(probability: float, seed: cudnn.tensor, offset: cudnn.tensor)` or `(mask: cudnn.tensor, scale: cudnn.tensor, scale_inv: cudnn.tensor)`.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type. FP8 examples use `FLOAT`.

- `name` (`str`, default `""`): Operation name.

- `sink_token` (`cudnn.tensor` or `None`, default `None`): Optional sink token bias from the matching forward operation.

- `dSink_token` (`cudnn.tensor` or `None`, default `None`): Optional caller-provided output tensor for sink token gradient. It is filled through graph execution but is not part of the returned 7-tensor tuple.

## Returns

- `dQ`, `dK`, `dV` (`cudnn.tensor`): FP8 query, key, and value gradient graph tensors.

- `amax_dQ`, `amax_dK`, `amax_dV`, `amax_dP` (`cudnn.tensor`): Absolute-maximum tensors for gradient quantization paths. Examples use shape `(1, 1, 1, 1)` and dtype `FLOAT`.

## Raises

- `TypeError`: Raised immediately for invalid tensor arguments, enum values, scalar values, or malformed Python types.

- `RuntimeError`: Raised immediately for conflicting causal-mask options, invalid `left_bound` or `right_bound` values, invalid `attn_scale`, or malformed dropout tuples.

- `ValueError`: Raised by `graph.validate()` for malformed tensor metadata, incompatible forward/backward options, or invalid output metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan build when the GPU architecture, cuDNN version, dtype, layout, mask, dropout, or deterministic option is unsupported.

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

`sdpa_fp8_backward` must match the forward `sdpa_fp8` configuration: Q/K/V/O tensor metadata, `stats`, scale/descale tensors, attention scale, masks, dropout, and sink-token use must be consistent.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1534) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L836).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API in the current checkout.

## Appendix: C++ equivalent

See the [C++ `Graph::sdpa_fp8_backward` reference](/api/cpp/sdpa-fp8-backward) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
