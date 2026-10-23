# concatenate

Append a graph concatenate operation to a PyGraph.

## Python synopsis

```python
Y = graph.concatenate(
    inputs,
    axis,
    in_place_index=None,
    name="",
)
```

## Parameters

- `inputs` (`list[cudnn.tensor]`, required): Tensors to concatenate. They must have compatible ranks, dtypes, and non-concatenated dimensions under backend rules.

- `axis` (`int`, required): Concatenation axis.

- `in_place_index` (`int or None`, default `None`): Optional backend in-place concat selection. Leave as `None` unless using a documented backend in-place pattern.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Concatenated output graph tensor. Mark it as an output and set dtype, shape, or stride when inference is ambiguous.

## Raises

- `TypeError`: Raised immediately for invalid input list, tensor, axis, in-place index, or name argument types.

- `ValueError`: Raised by `graph.validate()` for incompatible ranks, dimensions, dtypes, in-place metadata, or output tensor metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan build when concatenation is unsupported by the selected backend path.

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

`concatenate` appends a graph operation node. Runtime buffers are combined only when the graph executes.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1240) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L396).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/concatenate_basic.py)

<Code
  src="../../samples/python/concatenate_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::concatenate` reference](/api/cpp/concatenate) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
