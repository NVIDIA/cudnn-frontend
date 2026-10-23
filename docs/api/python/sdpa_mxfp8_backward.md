# sdpa_mxfp8_backward

Append an MXFP8 scaled dot-product attention backward operation to a PyGraph.

## Python synopsis

```python
dQ, dK, dV, amax_dQ, amax_dK, amax_dV = graph.sdpa_mxfp8_backward(
    q, q_T, k, k_T, v,
    o_f16, dO_f16, dO, dO_T, stats,
    descale_q, descale_q_T, descale_k, descale_k_T,
    descale_v, descale_dO, descale_dO_T,
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

- `q`, `k`, `v` (`cudnn.tensor`, required): Forward MXFP8 FP8 data tensors.

- `q_T`, `k_T`, `dO_T` (`cudnn.tensor`, required): Transposed tensor forms required by the MXFP8 backward implementation.

- `o_f16`, `dO_f16` (`cudnn.tensor`, required): FP16 or BF16 forward output and upstream gradient forms consumed by the implementation.

- `dO` (`cudnn.tensor`, required): FP8 upstream output gradient.

- `stats` (`cudnn.tensor`, required): Softmax statistics returned by `sdpa_mxfp8(..., generate_stats=True)`.

- `descale_q`, `descale_q_T`, `descale_k`, `descale_k_T`, `descale_v`, `descale_dO`, `descale_dO_T` (`cudnn.tensor`, required): MXFP8 block scale-factor tensors. They should use dtype `FP8_E8M0` and `reordering_type=cudnn.tensor_reordering.F8_128x4`, with dimensions and strides matching the corresponding data tensor orientation.

- `attn_scale` (`float`, `cudnn.tensor`, or `None`, default `None`): Optional score scale. Use the same value as the matching forward call.

- `use_padding_mask` (`bool`, default `False`): Enables variable-length masking. Requires `seq_len_q` and `seq_len_kv` when used.

- `seq_len_q`, `seq_len_kv` (`cudnn.tensor` or `None`, default `None`): Actual query and key/value lengths, normally `INT32` tensors.

- `use_causal_mask` (`bool`, default `False`): Deprecated causal shortcut. Prefer `right_bound=0` with the desired `diagonal_alignment`.

- `use_causal_mask_bottom_right` (`bool`, default `False`): Deprecated bottom-right causal shortcut. Prefer `diagonal_alignment=BOTTOM_RIGHT` with `right_bound=0`.

- `diagonal_alignment` (`cudnn.diagonal_alignment`, default `TOP_LEFT`): Alignment for diagonal-band masking.

- `left_bound`, `right_bound` (`int` or `None`, default `None`): Diagonal-band bounds. `right_bound` conflicts with deprecated causal-mask shortcuts.

- `use_deterministic_algorithm` (`bool`, default `False`): Requests deterministic algorithm selection where supported.

- `dropout` (`tuple` or `None`, default `None`): Optional dropout metadata. The backward binding accepts a 3-element tuple: `(probability: float, seed: cudnn.tensor, offset: cudnn.tensor)` or `(mask: cudnn.tensor, scale: cudnn.tensor, scale_inv: cudnn.tensor)`.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type. MXFP8 examples use `FLOAT`.

- `name` (`str`, default `""`): Operation name.

- `sink_token` (`cudnn.tensor` or `None`, default `None`): Optional sink token bias from the matching forward operation.

- `dSink_token` (`cudnn.tensor` or `None`, default `None`): Optional caller-provided output tensor for sink token gradient. It is filled through graph execution but is not part of the returned 6-tensor tuple.

## Returns

- `dQ`, `dK`, `dV` (`cudnn.tensor`): Query, key, and value gradient graph tensors.

- `amax_dQ`, `amax_dK`, `amax_dV` (`cudnn.tensor`): Absolute-maximum tensors for gradient quantization paths. Examples use shape `(1, 1, 1, 1)` and dtype `FLOAT`.

## Raises

- `TypeError`: Raised immediately for invalid tensor arguments, enum values, scalar values, or malformed Python types.

- `RuntimeError`: Raised immediately for conflicting causal-mask options, invalid `left_bound` or `right_bound` values, invalid `attn_scale`, or malformed dropout tuples.

- `ValueError`: Raised by `graph.validate()` for malformed MXFP8 scale-factor metadata, incompatible transposed tensors, inconsistent forward/backward options, or invalid output metadata.

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

MXFP8 backward is exposed in C++ as an overload of `Graph::sdpa_fp8_backward`; the Python binding gives it the explicit name `sdpa_mxfp8_backward`.

The live MXFP8 forward docstring and samples require cuDNN 9.21.0 or later and Blackwell GPU architecture or newer. Treat the same support surface as applying to MXFP8 backward unless local probing proves otherwise.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1621) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/sdpa.cpp#L1010).
{/* END GENERATED API SOURCES */}

## Examples

No executable sample is published for this API in the current checkout.

## Appendix: C++ equivalent

See the [C++ `Graph::sdpa_mxfp8_backward` reference](/api/cpp/sdpa-mxfp8-backward) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
