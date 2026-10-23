# deselect_shared_mem_greater_than

Reject candidate plans whose shared-memory requirement is greater than a byte threshold.

## C++ synopsis

```cpp
graph.deselect_shared_mem_greater_than(64 * 1024);
```

## Parameters

- `workspace` (`int64_t`, required): Maximum allowed shared-memory size in bytes for a candidate plan.

## Returns

- `graph` (`fe::graph::Graph&`): The same graph object, enabling fluent chaining.

## Errors

This filter helper returns `Graph&` and does not return `error_t`. If filtering leaves no viable candidate, a later `check_support()` or `build_plans()` call reports the failure.

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

The method sets the plan-list maximum shared-memory allowance. Use it after `create_execution_plans()` and before `check_support()` or `build_plans()` when you need to exclude high shared-memory plans.

This is separate from `deselect_workspace_greater_than()`, which filters by external workspace bytes.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2899) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2899).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/planning.mdx" />

## Appendix: Python equivalent

There is no direct Python equivalent documented in `docs/api/python`.
