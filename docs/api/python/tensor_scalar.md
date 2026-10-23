# tensor_scalar

Create a rank-1 scalar graph tensor from a Python scalar value.

## Python synopsis

```python
alpha = graph.tensor_scalar(value, scalar_type)
```

`tensor_scalar` is the Python binding for scalar `Graph::tensor(...)` overloads. It is commonly used for compile-time constants or runtime scalar parameters consumed by pointwise, normalization, SDPA, and other graph operations.

## Parameters

- `value` (`float` or `int`, required): Scalar value used to construct the graph tensor. Python overload resolution accepts floating-point and integer scalar values.

- `scalar_type` (`cudnn.scalar_type`, required): `cudnn.scalar_type.COMPILE_TIME_CONST` embeds the value in the graph. `cudnn.scalar_type.RUNTIME_PARAM` marks the scalar as a runtime parameter that must be supplied through the execution variant pack when required by the graph.

## Returns

- `alpha` (`cudnn.tensor`): A rank-1 scalar graph tensor. The returned tensor can be passed to graph operations or further configured with tensor attribute setters when a consuming operation requires explicit metadata.

## Raises

- `TypeError`: Raised immediately when `value` or `scalar_type` cannot be converted to one of the bound scalar overloads.

- `ValueError`: Raised later by `graph.validate()` when the scalar is used in a malformed graph or required runtime scalar metadata is missing.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the scalar usage is unsupported by the selected backend path.

- `RuntimeError`: Raised by later lifecycle methods for CUDA, cuDNN backend, handle, invalid value, or execution failures.

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

`tensor_scalar` is distinct from `graph.tensor(..., is_pass_by_value=True)`. Use `tensor_scalar` for scalar constants or scalar runtime parameters when a graph operation accepts scalar graph tensors directly. Use explicit `graph.tensor(...)` metadata when the operation expects a shaped pass-by-value tensor.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L522) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L522).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/tensor_scalar_basic.py)

<Code
  src="../../samples/python/tensor_scalar_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::tensor_scalar` reference](/api/cpp/tensor-scalar) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
