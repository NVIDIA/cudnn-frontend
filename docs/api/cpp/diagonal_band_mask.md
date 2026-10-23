# diagonal_band_mask

Append a diagonal-band masking operation to an attention score tensor.

## C++ synopsis

```cpp
auto masked_scores = graph.diagonal_band_mask(
    scores,
    negative_inf,
    seq_len_q,
    seq_len_kv,
    cu_seq_len_q,
    cu_seq_len_kv,
    left_bound,
    shift_right_bound,
    fe::graph::DiagonalBandMask_attributes());
```

## Parameters

- `x` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Input score tensor to mask. The output shape and stride default to this tensor's shape and stride.

- `b` (`std::shared_ptr<fe::graph::Tensor_attributes>`, required): Fill value tensor used when a score is masked. For attention masks this is normally a scalar or tensor containing negative infinity.

- `seq_len_q` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`, optional): Query sequence lengths used to adjust the diagonal relation for variable-length attention.

- `seq_len_kv` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`, optional): Key/value sequence lengths used to adjust the diagonal relation for variable-length attention.

- `cu_seq_len_q` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`, optional): Cumulative query sequence offsets for packed variable-length attention.

- `cu_seq_len_kv` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`, optional): Cumulative key/value sequence offsets for packed variable-length attention.

- `left_bound` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`, optional): Left sliding-window bound. When provided, the mask compares the shifted column index against the row index.

- `shift_right_bound` (`std::shared_ptr<fe::graph::Tensor_attributes>` or `nullptr`, optional): Nonzero right-bound shift used for causal or chunked-prefill masking. A causal right bound of zero is represented by leaving this as `nullptr`. This must not be specified together with `left_bound`.

- `attributes` (`fe::graph::DiagonalBandMask_attributes`, required): Attribute object defined in `include/cudnn_frontend/graph_properties.h`.

- `attributes.set_comparison_mode(fe::PointwiseMode_t)`: Optional comparison mode for the generated mask. The default is `CMP_GT`.

## Returns

- `Y` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Masked output tensor attributes. If shape or stride are unset, they are inferred from `x`.

## Errors

`Graph::diagonal_band_mask()` appends a node and returns tensor attributes. `validate()` reports `INVALID_VALUE` if both `left_bound` and `shift_right_bound` are specified. Tensor metadata errors and backend failures are reported by later lifecycle calls.

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

This helper is a low-level building block used by attention score modifiers. cuDNN 9.21 and newer can represent it as a unified diagonal-band-mask backend operation; older supported builds expand it into pointwise operations.

To express a right-bound mask, leave `left_bound` as `nullptr`; `shift_right_bound` optionally shifts the right bound when the shift is nonzero. To express a left sliding-window mask, provide `left_bound` and leave `shift_right_bound` as `nullptr`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node_interface.h#L491) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/node/diagonal_band_mask.h#L441).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/advanced_operations.cc)

<Markdown src="/snippets/generated/cpp/advanced_operations/diagonal-mask.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `diagonal_band_mask()` in `docs/api/python`. Python attention APIs expose higher-level diagonal-band options such as `diagonal_band_left_bound` and `diagonal_band_right_bound`.
