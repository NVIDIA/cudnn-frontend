# tensor

Create a graph tensor attribute from explicit shape, stride, dtype, and tensor metadata.

## Python synopsis

```python
X = graph.tensor(
    dim,
    stride,
    data_type=cudnn.data_type.NOT_SET,
    is_virtual=False,
    is_pass_by_value=False,
    ragged_offset=None,
    reordering_type=cudnn.tensor_reordering.NONE,
    name="",
    uid=-1,
)
```

`graph.tensor(...)` creates a `cudnn.tensor` graph attribute only. It does not allocate device memory. Non-virtual inputs and outputs receive their runtime buffers later through the variant pack passed to `graph.execute(...)` or `graph.execute_plan_at_index(...)`.

## Parameters

- `dim` (`Sequence[int]`, required): Logical tensor dimensions. The number of values is the tensor rank. Use positive dimension values for static shapes. Graphs created with dynamic-shape or override-shape support may accept compatible runtime shapes through lifecycle APIs that take `override_shapes`.

- `stride` (`Sequence[int]`, required): Logical strides in elements, not bytes. `stride` must have the same length as `dim`. Related tensors do not need to have the same stride unless an operation-specific page says so.

- `data_type` (`cudnn.data_type`, default `cudnn.data_type.NOT_SET`): Tensor element type. If left as `NOT_SET`, operation or graph defaults may fill it later, but user-visible inputs and outputs are clearer when the type is set explicitly.

- `is_virtual` (`bool`, default `False`): Marks the tensor as an internal graph intermediate that does not require a user-provided runtime buffer. A materialized graph output should normally be non-virtual and marked with `set_output(True)`.

- `is_pass_by_value` (`bool`, default `False`): Marks a scalar or small parameter as passed by value through the variant pack. This is used for runtime scalar inputs such as normalization epsilon, attention scale, dropout seed, or dropout offset.

- `ragged_offset` (`cudnn.tensor` or `None`, default `None`): Optional tensor describing packed ragged storage offsets. It is consumed by APIs that support ragged or THD layouts. The offset tensor must satisfy the dtype, shape, and stride requirements of the consuming operation.

- `reordering_type` (`cudnn.tensor_reordering`, default `cudnn.tensor_reordering.NONE`): Optional special memory-reordering metadata. MXFP8 scale-factor tensors commonly use `cudnn.tensor_reordering.F8_128x4`.

- `name` (`str`, default `""`): User-visible tensor name. Names are useful for debugging, generated output names, serialization, and readable error messages.

- `uid` (`int`, default `-1`): Optional explicit frontend UID. Leave this as `-1` for normal graph construction. Explicit UIDs are mainly useful for serialized graphs, external descriptors, tests, or variant packs keyed by UID.

## Returns

- `X` (`cudnn.tensor`): A Python binding of C++ `Tensor_attributes`. The returned object can be passed to graph operations and further configured with methods such as `set_output(True)`, `set_data_type(...)`, `set_dim(...)`, and `set_stride(...)`.

## Raises

- `TypeError`: Raised immediately when `dim`, `stride`, enums, booleans, `ragged_offset`, `name`, or `uid` cannot be converted to the bound C++ types.

- `ValueError`: Raised later by `graph.validate()` for malformed tensor metadata, missing required attributes, invalid names, duplicate explicit UIDs, or shape-deduction failures in consuming operations.

- `cudnn.cudnnGraphNotSupportedError`: Raised by support checking or plan building when the tensor dtype, layout, dimensions, or consuming operation are unsupported by the selected backend path.

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

A `cudnn.tensor` describes memory but does not own it. The same tensor object, or its UID, must be mapped to a real buffer in the execution variant pack when it is a non-virtual input or output.

Tensor dtype and stride compatibility are operation-specific. For example, a matrix multiplication can use different input strides, while SDPA has stricter head-dimension stride requirements. Check the page for the operation that consumes the tensor.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L411) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L411).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/tensor_basic.py)

<Code
  src="../../samples/python/tensor_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::tensor` reference](/api/cpp/tensor) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
