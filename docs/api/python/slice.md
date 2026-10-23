# slice

Append a graph slice operation to a PyGraph.

## Python synopsis

```python
Y = graph.slice(
    input,
    slices=[],
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Input tensor to slice. Its dimensions are used to normalize each Python `slice` object.

- `slices` (`list[slice]`, default `[]`): Python slice objects, one per input dimension. The binding normalizes each slice using Python sequence semantics for that dimension and forwards start, stop, and step values to `Slice_attributes`. Per-axis step comes from `slice.step`; omitted steps default to `1`.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Optional compute type for the operation.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Sliced output graph tensor. Mark it as an output and set dtype, shape, or stride when inference is ambiguous.

## Raises

- `RuntimeError`: Raised immediately when a Python slice cannot be normalized for the input dimension.

- `TypeError`: Raised immediately for invalid tensor, slice-list, enum, or name argument types.

- `ValueError`: Raised by `graph.validate()` for invalid shape deduction or malformed tensor metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan build when the slice operation is unsupported by the selected backend path.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** Standalone sample requires CUDA 13.1+, cuDNN 9.22+, and SM100+; syntax-checked with cuDNN 9.15.1
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

This API appends a graph operation node. It is not Python-side slicing over a runtime buffer. Do not use Python slicing syntax on `cudnn.tensor` objects; use `graph.slice(...)`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L951) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L195).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/slice_basic.py)

<Code
  src="../../samples/python/slice_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::slice` reference](/api/cpp/slice) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
