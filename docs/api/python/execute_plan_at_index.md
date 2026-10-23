# execute_plan_at_index

Execute a specific built plan index with a variant pack and workspace buffer.

## Python synopsis

```python
graph.execute_plan_at_index(
    tensor_to_device_buffer,
    workspace,
    index,
    handle=None,
    override_uids=None,
    override_shapes=None,
    override_strides=None,
)
```

## Parameters

- `tensor_to_device_buffer` (`dict[cudnn.tensor | int, torch.Tensor | int | object with __dlpack__]`, required): Variant pack mapping graph tensors or UIDs to runtime buffers.

- `workspace` (`torch.Tensor`, integer device pointer, or object with `__dlpack__`, required): Workspace buffer sized for the selected `index`, usually from `get_workspace_size_plan_at_index(index)`.

- `index` (`int`, required): Plan index to execute.

- `handle` (`int` or `None`, default `None`): cuDNN handle pointer.

- `override_uids`, `override_shapes`, `override_strides` (objects or `None`, default `None`): Runtime shape/stride override metadata.

## Returns

- `None`: Returned on success. Output buffers in the variant pack are written by the selected plan.

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

Use this method for autotuning or validating individual candidates after building plans with `build_plans(ALL)` or `build_plan_at_index(index)`. The workspace size can differ by index.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2049) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L2049).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/execute_plan_at_index_basic.py)

<Code
  src="../../samples/python/execute_plan_at_index_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::execute_plan_at_index` reference](/api/cpp/execute-plan-at-index) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
