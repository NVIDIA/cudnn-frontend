# get_variant_pack_uids_sorted

Return the sorted tensor UIDs that the caller must provide in a variant pack.

## C++ synopsis

```cpp
std::vector<int64_t> uids = graph.get_variant_pack_uids_sorted();
```

## Parameters

This method takes no parameters.

## Returns

- `uids` (`std::vector<int64_t>`): Sorted user-provided tensor UIDs from the prepared variant-pack template.

## Errors

`get_variant_pack_uids_sorted()` returns a vector and does not return `error_t`. If called before the variant-pack template has been prepared, the returned vector can be empty.

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

The variant-pack template is prepared at the end of `build_plans()` and during deserialization warmup. The returned UIDs are user-fillable slots only. Internal workspace slots, replacement targets, and pass-by-value entries already handled by the frontend are not returned.

This method is useful when building a UID-keyed variant pack manually. The higher-level `execute()` overloads that take tensor-attribute keys usually do not require it.


{/* BEGIN GENERATED API SOURCES */}
Current source: [declaration](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2857) and [implementation](https://github.com/NVIDIA/cudnn-frontend/blob/0a07d4ee28f8ae2f2106bb766b1878cc732ee80c/include/cudnn_frontend/graph_interface.h#L2857).
{/* END GENERATED API SOURCES */}

## Examples

Complete sample: [source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/plan_inspection.cc)

<Markdown src="/snippets/generated/cpp/plan_inspection/inspection.mdx" />

## Appendix: Python equivalent

The Python wrapper has an internal `_get_variant_pack_uids_sorted()` method used by the wrapper implementation. It is not documented as a normal Python API page.
