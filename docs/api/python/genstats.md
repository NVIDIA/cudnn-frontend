# genstats

Append a batch-normalization statistics generation operation to a PyGraph.

## Python synopsis

```python
sum_tensor, square_sum_tensor = graph.genstats(
    input,
    compute_data_type=cudnn.data_type.NOT_SET,
    name="",
)
```

## Parameters

- `input` (`cudnn.tensor`, required): Tensor to summarize. In repository examples, this is often the output of a convolution or activation chain that feeds normalization-style statistics.

- `compute_data_type` (`cudnn.data_type`, default `NOT_SET`): Compute type for sum and square-sum accumulation. Use `FLOAT` for common mixed-precision statistics.

- `name` (`str`, default `""`): Operation name used for generated tensor names.

## Returns

- `sum_tensor` (`cudnn.tensor`): Sum output graph tensor over the backend-defined statistics axes. Mark it as an output and set dtype when it must be materialized.

- `square_sum_tensor` (`cudnn.tensor`): Sum-of-squares output graph tensor over the same statistics axes. Mark it as an output and set dtype when it must be materialized.

## Raises

- `TypeError`: Raised immediately for invalid tensor, enum, or name argument types.

- `ValueError`: Raised by `graph.validate()` when tensor metadata or output metadata is invalid.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the dtype, layout, architecture, or statistics shape is unsupported.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Normalization families**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** python
- **API-specific availability:** See local constraints
- **Family release status:** Stable
- **Family cuDNN:** Operation-dependent
- **Family CUDA:** See installed cuDNN release
- **Family GPU:** Engine-dependent
- **Family platforms:** Linux (executed); Windows (not executed)
- **Family backends:** cuDNN backend
- **Family languages:** Python; C++
- **Family optional and runtime dependencies:** cuDNN; CUDA
- **Recorded verification:** LayerNorm and RMSNorm samples executed in Python and C++; additional C++ normalization samples executed on SM100
- [Compare feature availability](/getting-started/compatibility)
{/* END GENERATED API AVAILABILITY */}

`genstats` returns graph tensor attributes only. Runtime buffers for the returned sum and square-sum tensors are supplied later through graph execution.

The generated statistics are specialized for normalization-style graphs. Use `reduction` when you need a general reduction mode such as `ADD` or `AMAX`.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L946) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/pygraph/pygraph.cpp#L344).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/genstats_basic.py)

<Code
  src="../../samples/python/genstats_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::genstats` reference](/api/cpp/genstats) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
