# autotune

Execute built candidate plans, measure runtime, and select the fastest plan.

## C++ synopsis

```cpp
auto status = graph.autotune(handle, variant_pack, workspace);
```

## Parameters

- `handle` (`cudnnHandle_t`, required): cuDNN handle used for candidate execution and CUDA timing events.

- `tensor_uid_to_pointer_map` (`std::unordered_map<int64_t, void*>&`, required overload): Runtime variant pack keyed by tensor UID.

- `tensor_to_pointer_map` (`std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&`, required overload): Runtime variant pack keyed by tensor attributes. The overload converts tensor keys to UIDs before autotuning.

- `workspace` (`void*`, required): Device workspace large enough for any candidate plan that autotune may execute. The canonical sample uses checked per-plan workspace queries because the current `get_autotune_workspace_size()` implementation can dereference failed-plan slots.

- `user_impl` (`void*`, optional, default `nullptr`): Reserved for future use by the implementation.

## Returns

- `status` (`fe::error_t`): `OK` when autotune completes. Individual candidate failures during warmup or timed iterations are skipped; the method still succeeds if it can process the plan list.

## Errors

The returned `error_t` can report failures from `execute_plan_at_index()`, invalid variant packs, CUDA event or stream failures, missing or invalid built plans, or other CUDA/cuDNN backend failures. Check `status.is_good()` before executing the selected plan.

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

Call `autotune()` after `create_execution_plans()`, `check_support()`, and `build_plans(fe::BuildPlanPolicy_t::ALL)` when you want measurements rather than heuristic selection alone.

The method performs a warmup execution for each built plan, then up to 100 timed iterations. It records the best observed time, sorts plans by measured runtime, and sets the selected candidate to the fastest plan.

When an all-plan build contains failed-plan slots, compute the autotune
workspace with checked `get_workspace_size_plan_at_index()` calls. The aggregate
`get_autotune_workspace_size()` helper dereferenced null slots in the recorded
CUDA 13.1, cuDNN 9.15.1, and SM100 environment.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1542) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1542).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/autotune.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `autotune()` in `docs/api/python`.
