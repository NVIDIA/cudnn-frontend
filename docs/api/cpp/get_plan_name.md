# get_plan_name

Return the name of the currently selected execution-plan candidate.

## C++ synopsis

```cpp
std::string plan_name;
auto status = graph.get_plan_name(plan_name);
```

## Parameters

- `name` (`std::string&`, output): Receives the name of the current plan candidate.

## Returns

- `status` (`fe::error_t`): `OK` when the selected candidate has a plan name that can be queried.

## Errors

The returned `error_t` can report that no candidate has been selected, the candidate index is invalid, or the underlying plan-list query failed. Check `status.is_good()` before using the output string.

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

`get_plan_name()` is a convenience wrapper around `get_plan_name_at_index(plans.candidate, name)`. Call it after the graph has created and selected a candidate plan, typically after `build_plans()`.

Use `get_plan_name_at_index(index, name)` when inspecting a specific plan instead of the currently selected candidate.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1362) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1362).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/inspection.mdx" />

## Appendix: Python equivalent

Python exposes `get_plan_name_at_index(index)`, but there is no direct Python method named `get_plan_name()`.
