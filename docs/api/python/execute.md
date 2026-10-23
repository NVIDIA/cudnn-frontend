# execute

Execute the selected built plan with a variant pack and workspace buffer.

## Python synopsis

```python
graph.execute(
    tensor_to_device_buffer,
    workspace,
    handle=None,
    override_uids=None,
    override_shapes=None,
    override_strides=None,
)
```

## Parameters

- `tensor_to_device_buffer` (`dict[cudnn.tensor | int, torch.Tensor | int | object with __dlpack__]`, required): Variant pack mapping graph tensors or tensor UIDs to runtime buffers. All non-virtual inputs and outputs required by the selected plan must be present.

- `workspace` (`torch.Tensor`, integer device pointer, or object with `__dlpack__`, required): Workspace buffer with at least `graph.get_workspace_size()` bytes for the selected plan and runtime overrides.

- `handle` (`int` or `None`, default `None`): cuDNN handle pointer. If omitted, the handle associated with the graph is used when available.

- `override_uids`, `override_shapes`, `override_strides` (objects or `None`, default `None`): Optional runtime shape/stride override metadata. The sequences must have matching lengths when used.

## Returns

- `None`: Returned on success. Output buffers in `tensor_to_device_buffer` are written by cuDNN.

## Raises

- `TypeError`: Raised immediately when Python arguments cannot be converted to the bound C++ types, for example an invalid enum, a non-sequence list argument, or a non-integer plan index.

- `ValueError`: Raised for frontend validation errors that map to `std::invalid_argument`, such as missing tensor attributes, shape-deduction failures, invalid tensor names, or invalid variant-pack metadata.

- `cudnn.cudnnGraphNotSupportedError`: Raised when a well-formed graph or plan candidate is not supported by the selected heuristics, backend implementation, architecture, cuDNN version, or graph format.

- `RuntimeError`: Raised for cuDNN backend API failures, CUDA API failures, handle errors, graph execution failures, invalid values reported by the frontend, NVRTC compilation failures, or invalid device state.

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

Python converts `torch.Tensor` values with `.data_ptr()`, integer values as raw pointers, and other objects through DLPack. Pass-by-value tensors still need valid CPU-side pointers when they are user-provided runtime parameters.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2085) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2085).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/execute_basic.py)

<Code
  src="../../samples/python/execute_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::execute` reference](/api/cpp/execute) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
