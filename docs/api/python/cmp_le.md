# cmp_le

Compare input less than or equal to comparison elementwise.

## Python synopsis

```python
Y = graph.cmp_le(input, comparison, compute_data_type=cudnn.data_type.NOT_SET, name="")
```

## Parameters

- `input` (`cudnn.tensor`, required): Input tensor to compare.

- `comparison` (`cudnn.tensor`, required): Comparison tensor. It must be shape-compatible with `input` under pointwise broadcasting rules.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for the pointwise operation. If `NOT_SET`, graph or backend defaults are used.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Graph tensor for the pointwise output. Its shape is inferred from the input tensors and pointwise broadcasting rules. The output is a tensor attribute object; it is not materialized until graph execution.

## Raises

- `TypeError`: Raised immediately when tensor arguments, scalar options, enum values, or keyword names cannot be converted to the bound C++ types.

- `ValueError`: Raised later by `graph.validate()` for invalid broadcasting, malformed tensor metadata, or shape-deduction failures.

- `cudnn.cudnnGraphNotSupportedError`: Raised by `graph.check_support()` or `graph.build_plans()` when the pointwise mode, dtype, layout, fusion, architecture, or selected backend plan is unsupported.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** 8.5+ for the C++ frontend; Python wrapper paths require 9.12+
- **Family CUDA:** A CUDA toolkit supported by the installed cuDNN release
- **Family GPU:** Architecture supported by the selected cuDNN engine
- **Family platforms:** Linux (executed); Windows (source-supported, not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** Python and C++ representative samples executed with CUDA 13, cuDNN 9.15.1, and SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

This operation appends a pointwise node to the graph and returns tensor attributes. It does not execute immediately. Inputs do not need identical strides; each tensor only needs strides that describe its runtime storage and are accepted by the selected backend engine.

Broadcasting follows the backend pointwise broadcasting rules. When the output must be materialized, call `set_output(True)` and set the desired output dtype before graph validation.

This Python binding maps to C++ `PointwiseMode_t::CMP_LE`.

On CUDA 13.1, cuDNN 9.15.1, and SM100, neither a terminal comparison nor the
same comparison fused as a `binary_select` mask had a valid engine
configuration.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pointwise.cpp#L1104) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pointwise.cpp#L1104).
{/* END GENERATED API SOURCES */}

## Examples

No executable SM100 sample is published for this comparison mode.

## Appendix: C++ equivalent

See the [C++ `Graph::cmp_le` reference](/api/cpp/cmp-le) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
