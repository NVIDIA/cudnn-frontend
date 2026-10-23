# reduction

Append a reduction operation to a PyGraph.

## Python synopsis

```python
Y = graph.reduction(
    input,
    mode,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Tensor to reduce.

- `mode` (`cudnn.reduction_mode`, required): Reduction operation, such as `ADD`, `MUL`, `MIN`, `MAX`, or `AMAX`, depending on enum support in the installed cuDNN frontend.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for the reduction. Use `FLOAT` for common mixed-precision statistics and accumulation paths.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `Y` (`cudnn.tensor`): Reduced output graph tensor. Its dimensions identify which axes are reduced. If inference cannot determine the intended shape, set dimensions and strides on the returned tensor before validation.

## Raises

- `TypeError`: Raised immediately for invalid tensor, enum, or name argument types.

- `ValueError`: Raised by `graph.validate()` when output shape metadata does not identify a valid reduction or tensor metadata is malformed.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the reduction mode, dtype, layout, architecture, or output shape is unsupported.

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

`reduction` uses the returned output tensor shape to decide what is reduced. A dimension retained at the same size is usually preserved, while a dimension set to `1` or removed by backend rules can be reduced. Keep output shape metadata explicit in graphs where inference is ambiguous.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L1141) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L355).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/reduction_basic.py)

<Code
  src="../../samples/python/reduction_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::reduction` reference](/api/cpp/reduction) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
