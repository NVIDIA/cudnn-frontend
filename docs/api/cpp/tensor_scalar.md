# tensor_scalar

Create scalar tensor attributes through scalar overloads of `Graph::tensor`.

## C++ synopsis

```cpp
auto alpha = graph->tensor(0.5f, fe::graph::ScalarType::COMPILE_TIME_CONST);
auto beta = graph->tensor(int64_t{0}, fe::graph::ScalarType::RUNTIME_PARAM);
```

## Parameters

- `scalar` (`float`, `half`, `nv_bfloat16`, `int32_t`, `int64_t`, or `double`, required): Scalar value used to construct the graph tensor.

- `scalar_type` (`fe::graph::ScalarType`, required): `COMPILE_TIME_CONST` embeds the scalar in the graph. `RUNTIME_PARAM` expects a runtime scalar value when the graph executes.

## Returns

- `alpha` (`std::shared_ptr<fe::graph::Tensor_attributes>`): Rank-1 scalar tensor attributes registered as a graph input.

## Errors

Scalar `Graph::tensor(...)` overloads return tensor attributes immediately and do not return `error_t`. Invalid scalar usage is reported later by graph lifecycle calls.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
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

Scalar tensor overloads are useful for constants and runtime parameters in pointwise, normalization, attention, and other graph operations. Use explicit `Tensor_attributes` when an operation expects a shaped pass-by-value tensor rather than a scalar constant.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L604) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L3565).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/tensor_scalar_basic.cc)

<Code
  src="../../samples/cpp/tensor_scalar_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.tensor_scalar` reference](/api/python/tensor-scalar) for Python exceptions, wrapper behavior, and current availability constraints.
