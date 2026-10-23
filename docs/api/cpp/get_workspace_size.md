# get_workspace_size

Return the workspace size in bytes required by the selected built plan.

## C++ synopsis

```cpp
int64_t workspace_size = 0;
auto status = graph.get_workspace_size(workspace_size);
// Convenience overload:
auto bytes = graph.get_workspace_size();
```

## Parameters

- `workspace_size` (`int64_t&`, output): Receives workspace bytes for the selected candidate.

- `handle`, `override_uids`, `override_shapes`, `override_strides` (optional overload): Query runtime workspace for dynamic or override-shape execution.

## Returns

- `status` (`fe::error_t`): `OK` when workspace size is queried successfully. The convenience overload returns `int64_t` and logs errors instead of returning `error_t`.

## Errors

Most C++ lifecycle calls return `fe::error_t`. Check `status.is_good()` or `status.is_bad()` before continuing. Errors can report invalid graph attributes, unsupported backend graphs, failed heuristic queries, failed execution-plan creation, CUDA/cuDNN backend failures, invalid variant packs, invalid plan indices, unsupported runtime overrides, or execution failures.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Stable
- **API language:** cpp
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

The returned size includes frontend workspace such as auxiliary tensor initialization space plus backend plan workspace.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1383) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1383).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/get_workspace_size_basic.cc)

<Code
  src="../../samples/cpp/get_workspace_size_basic.cc"
  language="cpp"
  maxLines={0}
/>

## Appendix: Python equivalent

See the [Python `cudnn.pygraph.get_workspace_size` reference](/api/python/get-workspace-size) for Python exceptions, wrapper behavior, and current availability constraints.
