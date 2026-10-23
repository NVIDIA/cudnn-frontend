# get_autotune_workspace_size

Return the workspace size needed to autotune all built candidate plans.

## C++ synopsis

```cpp
int64_t workspace_size = graph.get_autotune_workspace_size();
```

## Parameters

This method takes no parameters.

## Returns

- `workspace_size` (`int64_t`): Maximum cuDNN workspace size across the graph's execution plans, plus frontend-owned workspace such as auxiliary tensor initialization space.

## Errors

`get_autotune_workspace_size()` returns an integer and does not return `error_t`. If called before candidate plans exist, the plan-list maximum is empty and the method can only account for frontend workspace already known to the graph.

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

Use this helper before `autotune()` when allocating a workspace large enough for every candidate plan that autotune may execute. For executing a selected plan, prefer `get_workspace_size()`, which returns the workspace required by the selected candidate.

On CUDA 13.1, cuDNN 9.15.1, and SM100, this helper dereferenced null plan slots
after `BuildPlanPolicy_t::ALL` retained entries for plans that failed to build.
Until the implementation skips those slots, query
`get_workspace_size_plan_at_index()` for each successfully built plan and take
the maximum.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1534) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1534).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/autotune.mdx" />

## Appendix: Python equivalent

There is no direct Python equivalent documented in `docs/api/python`.
