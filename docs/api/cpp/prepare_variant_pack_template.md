# prepare_variant_pack_template

Prepare the cached variant-pack template used by fast execution overloads.

## C++ synopsis

```cpp
auto status = graph.prepare_variant_pack_template();
```

## Parameters

This method takes no parameters.

## Returns

- `status` (`fe::error_t`): `OK` when the template is prepared or was already prepared.

## Errors

The returned `error_t` can report failures while collecting pass-by-value tensors, workspace-backed tensors, replacement slots, or OSS engine slot metadata.

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

`build_plans()` prepares the variant-pack template automatically. The `execute()` and `execute_plan_at_index()` overloads also lazily prepare it if needed. Manual calls are rarely required, but can be useful before inspecting `get_variant_pack_uids_sorted()` or before a latency-sensitive first execution.

The template stores sorted UIDs, pre-filled pass-by-value entries, user-fillable slots, workspace-relative slots, and replacement slots used by frontend fallbacks.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2207) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2207).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/inspection.mdx" />

## Appendix: Python equivalent

There is no direct Python graph binding documented for `prepare_variant_pack_template()` in `docs/api/python`.
