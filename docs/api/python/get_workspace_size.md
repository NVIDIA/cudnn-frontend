# get_workspace_size

Return the workspace size in bytes required by the selected built plan.

## Python synopsis

```python
workspace_size = graph.get_workspace_size()
workspace_size = graph.get_workspace_size(
    handle=None,
    override_uids=None,
    override_shapes=None,
    override_strides=None,
)
```

## Parameters

- `handle` (`int` or `None`, default `None`): Optional cuDNN handle pointer used for runtime workspace queries with override shapes.

- `override_uids` (`Sequence[int]` or `None`, default `None`): Tensor UIDs whose runtime shapes and strides are overridden.

- `override_shapes` (`Sequence[Sequence[int]]` or `None`, default `None`): Runtime shapes corresponding one-to-one with `override_uids`.

- `override_strides` (`Sequence[Sequence[int]]` or `None`, default `None`): Runtime strides corresponding one-to-one with `override_uids`.

## Returns

- `workspace_size` (`int`): Number of bytes required for the selected plan, including frontend-owned workspace and backend execution-plan workspace.

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

Call this after building a plan. For runtime override-shape queries, `override_uids`, `override_shapes`, and `override_strides` must have the same length. The override workspace query path requires cuDNN 9.23.0 in the C++ implementation and is bypassed for registered open-source engines.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L1856) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/python/cudnn/_pygraph.py#L1856).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/get_workspace_size_basic.py)

<Code
  src="../../samples/python/get_workspace_size_basic.py"
  language="python"
  maxLines={0}
/>

## Appendix: C++ equivalent

See the [C++ `Graph::get_workspace_size` reference](/api/cpp/get-workspace-size) for its overloads, `error_t` behavior, resource ownership, and current availability constraints.
