# apply_oss_slot_indices_to_plans

Re-apply prepared variant-pack slot indices to OSS execution plans.

## C++ synopsis

```cpp
graph.apply_oss_slot_indices_to_plans();
```

## Parameters

This method takes no parameters.

## Returns

This method returns `void`.

## Errors

`apply_oss_slot_indices_to_plans()` does not return `error_t`. It updates OSS plan metadata from the current variant-pack template.

## Notes

{/* BEGIN GENERATED API AVAILABILITY */}
**Availability: Core Graph operations and lifecycle**

- **API disposition:** Public
- **API stability:** Experimental
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

This is an advanced maintenance helper. The graph calls it automatically from `prepare_variant_pack_template()` and after `autotune()` reorders candidate plans. Most users should not call it manually.

The method depends on the current plan list and the prepared variant-pack template. Calling it before those structures are ready has no useful effect for normal graph execution.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1636) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L1636).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/inspection.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `apply_oss_slot_indices_to_plans()` in `docs/api/python`.
