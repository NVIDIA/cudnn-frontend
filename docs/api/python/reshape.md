# reshape

Append a graph reshape operation to a PyGraph.

## Python synopsis

```python
Y = graph.reshape(
    input,
    name="",
    reshape_mode=cudnn.reshape_mode.VIEW_ONLY,
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Input tensor to reshape.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

- `reshape_mode` (`cudnn.reshape_mode`, default `VIEW_ONLY`): Reshape behavior. `VIEW_ONLY` changes the tensor view without changing physical memory. `LOGICAL` requests a lexicographic logical reshape where supported.

## Returns

- `Y` (`cudnn.tensor`): Reshaped output graph tensor. The live docstring states that the caller should set the output dimensions; call `Y.set_dim(...)` and usually `Y.set_stride(...)` before validation when the target shape is not otherwise known.

## Raises

- `TypeError`: Raised immediately for invalid tensor, enum, or name argument types.

- `ValueError`: Raised by `graph.validate()` for invalid shape deduction, incompatible reshape metadata, or malformed tensor metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan build when the reshape mode or tensor layout is unsupported by the selected backend path.

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

`reshape` is a graph operation, not a host-side view over runtime storage. In `VIEW_ONLY` mode, the logical output metadata must still describe the runtime buffer correctly.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1205) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L373).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/reshape_basic.py)

<Code
  src="../../samples/python/reshape_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::reshape` reference](/api/cpp/reshape) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
